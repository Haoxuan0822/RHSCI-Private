"""捕获真实训练forward动作分支，以相同增强图/噪声/时间核对eval速度场。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from openpi.rhsci.data import SimpleRHSCIDataset
from openpi.rhsci.model import Pi05RHSCI
from openpi.rhsci.processing import Processor, collate


def difference(left, right):
    delta = left.float() - right.float()
    return dict(
        bitwise_equal=torch.equal(left, right),
        max_abs_error=float(delta.abs().max()),
        relative_l2=float(delta.norm() / right.float().norm().clamp_min(1e-12)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[30, 60, 90])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        raise RuntimeError("Need an isolated GPU with 16 GiB free")
    saved = json.loads((args.checkpoint / "rsci_config.json").read_text())
    if saved["variant"] != "VGQ-AE":
        raise ValueError("Expected selected VGQ-AE checkpoint")
    processor = Processor(
        args.dataset, args.tokenizer, "VGQ-AE", normalization=saved["normalization"]
    )
    dataset = SimpleRHSCIDataset(args.dataset, "STATE")
    model = Pi05RHSCI("VGQ-AE", compile_inference=False, execution=saved.get("execution", "both"))
    model.load_state_dict(
        load_file(str(args.checkpoint / "model.safetensors")), strict=True, assign=True
    )
    model.cuda()
    low, high = [np.asarray(processor.stats["action"][key]) for key in ("q01", "q99")]
    arm_scale = torch.as_tensor((high - low + 1e-6)[14:28] / 2, device="cuda")
    rows = []
    for frame in args.frames:
        if frame < 6 or frame + 30 > dataset.episodes[0]["length"]:
            raise ValueError("Need full history/action window in episode zero")
        history = [dataset.raw_sample(frame + offset) for offset in (-6, -4, -2, 0)]
        sample = dict(history[-1], images=np.concatenate([item["images"] for item in history]))
        sample.update(
            rsci_history_valid=np.ones(4, bool),
            rsci_ardy_target=np.zeros(128, np.float32),
            rsci_ardy_valid=False,
        )
        batch = {key: value.cuda() for key, value in collate([processor(sample)]).items()}
        for seed in args.seeds:
            captured = {}
            encode, expert = model.encode, model.expert

            def capture_encode(image, tokens, mask):
                if "image" not in captured:
                    captured["image"] = image.clone()
                return encode(image, tokens, mask)

            def capture_expert(hidden, time, mask, cache, **kwargs):
                if "expert" not in captured:
                    captured["expert"] = (hidden.clone(), time.clone(), mask, cache)
                return expert(hidden, time, mask, cache, **kwargs)

            def capture_velocity(module, values, output):
                captured.setdefault("velocity", output.clone())

            model.train()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model.encode, model.expert = capture_encode, capture_expert
            hook = model.action_out_proj.register_forward_hook(capture_velocity)
            try:
                with (
                    torch.no_grad(),
                    torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
                ):
                    losses = model(**batch)
            finally:
                hook.remove()
                model.encode, model.expert = encode, expert
            hidden, time, mask, cache = captured["expert"]
            expected = captured["velocity"]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                train_repeat = model.action_out_proj(expert(hidden, time, mask, cache).float())
                model.eval()
                eval_same_cache = model.action_out_proj(expert(hidden, time, mask, cache).float())
                _, eval_mask, eval_cache = encode(
                    captured["image"], batch["tokens"], batch["token_mask"]
                )
                eval_reencoded = model.action_out_proj(
                    expert(hidden, time, eval_mask, eval_cache).float()
                )
            clean_delta = (expected.float() - eval_reencoded.float()) * time[:, None, None]
            row = dict(
                frame=frame,
                seed=seed,
                time=float(time.item()),
                action_loss=float(losses["action_loss"]),
                train_repeat=difference(train_repeat, expected),
                eval_same_cache=difference(eval_same_cache, expected),
                eval_reencoded=difference(eval_reencoded, expected),
                implied_clean_arm_rmse_rad=float(
                    (clean_delta[..., 14:28] * arm_scale).square().mean().sqrt()
                ),
                augmented_image_sha256=hashlib.sha256(
                    captured["image"].float().cpu().numpy().tobytes()
                ).hexdigest(),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = dict(
        checkpoint=str(args.checkpoint),
        config=saved,
        rows=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation="No optimizer/backward/rollout. Auxiliary targets are masked placeholders; "
        "action branch executes actual forward with native augmentation and real four-frame "
        "history. Same captured noisy-action embedding, time and augmented image are reused "
        "to isolate train/eval mode; this does not test the unaugmented deployment distribution "
        "or entire training dynamics.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
