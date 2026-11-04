"""比较真实模型在专家加噪路径与自主采样路径上的动作估计，不改采样器。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from openpi.rhsci.model import Pi05RHSCI
from openpi.rhsci.processing import Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "tokenizer", "recorded-inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[17, 29, 43])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("Need 12 GiB free for an isolated model")
    saved = json.loads((args.checkpoint / "rsci_config.json").read_text())
    if saved["variant"] != "VGQ-AE":
        raise ValueError("This image-only diagnosis is for the selected VGQ-AE checkpoint")
    processor = Processor(
        args.dataset, args.tokenizer, saved["variant"], normalization=saved["normalization"]
    )
    model = Pi05RHSCI(saved["variant"], compile_inference=False)
    model.load_state_dict(
        load_file(str(args.checkpoint / "model.safetensors")), strict=True, assign=True
    )
    model.to("cuda").eval()
    content = args.recorded_inputs.read_bytes()
    records = [json.loads(line) for line in content.splitlines()]
    if [(row["episode"], row["frame"]) for row in records] != [
        (0, frame) for frame in (0, 30, 60, 90)
    ]:
        raise ValueError("Need the four previously verified episode-0 training inputs")
    low, high = (np.asarray(processor.stats["action"][key], np.float32) for key in ("q01", "q99"))
    arm_scale = torch.tensor((high - low + 1e-6)[14:28] / 2, device="cuda")

    def arm_rmse(difference):
        return float((difference[..., 14:28].float() * arm_scale).square().mean().sqrt())

    results, initial_prefix = [], None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        for row in records:
            image_path = args.recorded_inputs.parent / f"episode-000-frame-{row['frame']:04d}.png"
            sample = dict(
                images=[np.asarray(Image.open(image_path))],
                text=row["instruction"],
                state=np.asarray(row["state"], np.float32),
                actions=np.asarray(row["expert_actions"], np.float32),
                rsci_history_valid=np.ones(4, bool),
                rsci_ardy_target=np.zeros(128),
                rsci_ardy_valid=False,
            )
            batch = processor(sample)
            images, tokens, token_mask = [
                batch[key][None].to("cuda") for key in ("images", "tokens", "token_mask")
            ]
            clean = batch["actions"][None].to("cuda")
            _, mask, cache = model.encode(images[:, -1], tokens, token_mask)
            if initial_prefix is None:
                initial_prefix = (mask, cache)
            for seed in args.noise_seeds:
                noise = torch.randn(
                    clean.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(seed)
                )
                baseline = model.sample_actions(images, tokens, token_mask, noise=noise.clone())
                inputs, velocities = [], []
                input_hook = model.action_in_proj.register_forward_pre_hook(
                    lambda module, values: inputs.append(values[0].clone())
                )
                output_hook = model.action_out_proj.register_forward_hook(
                    lambda module, values, output: velocities.append(output.float().clone())
                )
                try:
                    predicted = model.sample_actions(
                        images, tokens, token_mask, noise=noise.clone()
                    )
                finally:
                    input_hook.remove()
                    output_hook.remove()
                if (
                    not torch.equal(baseline, predicted)
                    or len(inputs) != 10
                    or len(velocities) != 10
                ):
                    raise ValueError("Instrumentation changed sampling or missed native steps")
                steps = []
                for index, (self_noisy, self_velocity) in enumerate(zip(inputs, velocities)):
                    time = torch.full((1,), 1 - index / 10, device="cuda")
                    teacher_noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * clean
                    target_velocity = noise - clean
                    entry = dict(
                        time=float(time[0]),
                        self_path_clean_estimate_arm_rmse_rad=arm_rmse(
                            self_noisy - time[:, None, None] * self_velocity - clean
                        ),
                    )
                    for name, prefix in (
                        ("current_image", (mask, cache)),
                        ("initial_image", initial_prefix),
                    ):
                        velocity = model.action_out_proj(
                            model.expert(model.action_in_proj(teacher_noisy), time, *prefix).float()
                        ).float()
                        error = velocity - target_velocity
                        entry[name] = dict(
                            teacher_path_velocity_mse=float(error.square().mean()),
                            teacher_path_clean_estimate_arm_rmse_rad=arm_rmse(
                                teacher_noisy - time[:, None, None] * velocity - clean
                            ),
                        )
                    steps.append(entry)
                result = dict(
                    episode=row["episode"],
                    frame=row["frame"],
                    seed=seed,
                    image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    instrumented_sample_bitwise_equal=True,
                    final_arm_rmse_to_v2_training_label_rad=arm_rmse(predicted - clean),
                    steps=steps,
                )
                results.append(result)
                print(json.dumps(result), flush=True)
    report = dict(
        checkpoint=str(args.checkpoint),
        rsci_config=saved,
        recorded_inputs_sha256=hashlib.sha256(content).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=results,
        limitation="One recorded training episode, no augmentation or parameter updates. Teacher-path velocity target is valid on the constructed straight noisy-expert path; no such target is asserted for the autonomous off-path samples. Initial-image prefix replacement isolates visual conditioning on identical noisy actions, not a success-rate or training-recipe counterfactual.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
