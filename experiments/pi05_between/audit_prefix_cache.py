"""同真实权重比较原生联合前向与 RHSCI 分离 KV 前向，不更新模型。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.nn import functional as F

from openpi.rhsci.model import Pi05RHSCI, time_embedding
from openpi.rhsci.processing import Processor


def joint_velocity(model, image, tokens, token_mask, noisy, time):
    """调用已有骨干联合入口；前缀不能看到后缀，后缀内部双向。"""
    native = model.paligemma_with_expert
    vision = native.embed_image(image)
    language = native.embed_language_tokens(tokens) * math.sqrt(
        native.paligemma.config.text_config.hidden_size
    )
    prefix = torch.cat((vision, language), dim=1)
    valid = torch.cat(
        (torch.ones(vision.shape[:2], dtype=torch.bool, device=image.device), token_mask), dim=1
    )
    prefix_length = valid.shape[1]
    valid = F.pad(valid, (0, noisy.shape[1]), value=True)
    allowed = valid[:, None, None, :].expand(-1, 1, valid.shape[1], -1).clone()
    allowed[:, :, :prefix_length, prefix_length:] = False
    attention = torch.zeros(allowed.shape, device=image.device).masked_fill(
        ~allowed, torch.finfo(torch.float32).min
    )
    conditioning = F.silu(model.time_mlp_out(F.silu(model.time_mlp_in(time_embedding(time, 1024)))))
    (_, hidden), _ = native(
        attention_mask=attention,
        position_ids=valid.long().cumsum(-1) - 1,
        inputs_embeds=[prefix, model.action_in_proj(noisy)],
        adarms_cond=[None, conditioning],
        use_cache=False,
    )
    return model.action_out_proj(hidden.float()).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "tokenizer", "recorded-inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sample-steps", type=int, default=0)
    args = parser.parse_args()
    if args.sample_steps < 0:
        parser.error("--sample-steps must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("Need 12 GiB free for isolated inference")
    saved = json.loads((args.checkpoint / "rsci_config.json").read_text())
    processor = Processor(
        args.dataset, args.tokenizer, saved["variant"], normalization=saved["normalization"]
    )
    model = Pi05RHSCI(saved["variant"], compile_inference=False)
    model.load_state_dict(
        load_file(str(args.checkpoint / "model.safetensors")), strict=True, assign=True
    )
    model.to("cuda").eval()
    native = model.paligemma_with_expert
    modules = (native.paligemma.language_model, native.gemma_expert.model)
    content = args.recorded_inputs.read_bytes()
    rows = []
    samples = []
    low, high = (np.asarray(processor.stats["action"][key], np.float32) for key in ("q01", "q99"))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        for record in map(json.loads, content.splitlines()):
            image_path = args.recorded_inputs.parent / (
                f"episode-{record['episode']:03d}-frame-{record['frame']:04d}.png"
            )
            batch = processor(
                dict(
                    images=[np.asarray(Image.open(image_path))],
                    text=record["instruction"],
                    state=np.asarray(record["state"], np.float32),
                    actions=np.asarray(record["expert_actions"], np.float32),
                    rsci_history_valid=np.ones(4, bool),
                    rsci_ardy_target=np.zeros(128),
                    rsci_ardy_valid=False,
                )
            )
            image, tokens, token_mask, clean = [
                batch[key][None].cuda() for key in ("images", "tokens", "token_mask", "actions")
            ]
            noise = torch.randn(
                clean.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(17)
            )
            for time_value in (1.0, 0.1):
                time = torch.full((1,), time_value, device="cuda")
                noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * clean
                outputs = {}
                for mode in ("sdpa", "eager"):
                    for module in modules:
                        module.config._attn_implementation = mode
                    _, mask, cache = model.encode(image[:, -1], tokens, token_mask)
                    original = [(key.clone(), value.clone()) for key, value in cache]
                    outputs[mode] = model.action_out_proj(
                        model.expert(model.action_in_proj(noisy), time, mask, cache).float()
                    ).float()
                    if any(
                        not torch.equal(key, old_key) or not torch.equal(value, old_value)
                        for (key, value), (old_key, old_value) in zip(cache, original)
                    ):
                        raise ValueError("Expert mutated the prefix cache")
                outputs["joint"] = joint_velocity(
                    model, image[:, -1], tokens, token_mask, noisy, time
                )
                row = dict(frame=record["frame"], time=time_value, cache_unchanged=True)
                for left, right in (("eager", "joint"), ("sdpa", "eager")):
                    delta = outputs[left] - outputs[right]
                    row[f"{left}_vs_{right}"] = dict(
                        max_abs=float(delta.abs().max()),
                        relative_l2=float(delta.norm() / outputs[right].norm().clamp_min(1e-12)),
                        arm_rmse_normalized=float(delta[..., 14:28].square().mean().sqrt()),
                    )
                rows.append(row)
                print(json.dumps(row), flush=True)
            if args.sample_steps:
                for module in modules:
                    module.config._attn_implementation = "sdpa"
                cached = model.sample_actions(
                    image, tokens, token_mask, num_steps=args.sample_steps, noise=noise.clone()
                )
                for module in modules:
                    module.config._attn_implementation = "eager"
                joint = noise.clone()
                for index in range(args.sample_steps):
                    time = torch.full((1,), 1 - index / args.sample_steps, device="cuda")
                    velocity = joint_velocity(model, image[:, -1], tokens, token_mask, joint, time)
                    joint = joint - velocity / args.sample_steps
                entry = dict(
                    frame=record["frame"],
                    input_image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                )
                for name, actions in (("cached_sdpa", cached), ("native_joint_eager", joint)):
                    raw = (actions[0].float().cpu().numpy() + 1) / 2 * (high - low + 1e-6) + low
                    entry[name] = raw[:, 14:28].tolist()
                samples.append(entry)
                print(
                    json.dumps(dict(sample_frame=record["frame"], sample_steps=args.sample_steps)),
                    flush=True,
                )
    report = dict(
        checkpoint=str(args.checkpoint),
        input_sha256=hashlib.sha256(content).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,
        sample_steps=args.sample_steps,
        samples=samples,
        limitation="Fixed recorded inputs and teacher noisy actions for velocity comparisons; optional complete sampling uses identical seed17 noise with the existing Euler formula. No parameter update, execution or success-rate claim. Joint native path always uses eager attention; cached eager velocity control separates attention backend effects.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
