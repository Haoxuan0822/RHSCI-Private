"""同一权重/噪声跨轨迹比较输入尺度，不更改部署处理器或执行模型动作。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from openpi.rhsci.data import SimpleRHSCIDataset
from openpi.rhsci.model import Pi05RHSCI
from openpi.rhsci.processing import Processor


def crop_image(image, fraction):
    if fraction == 1:
        return image
    height, width = image.shape[-2:]
    h, w = int(height * fraction), int(width * fraction)
    top, left = (height - h) // 2, (width - w) // 2
    selected = (image / 2 + 0.5)[..., top : top + h, left : left + w]
    return (
        F.interpolate(
            selected.flatten(0, 1), size=(height, width), mode="bilinear", align_corners=False
        ).reshape_as(image)
        * 2
        - 1
    )


def metrics(predictions, experts):
    p, e = predictions[..., 14:28], experts[..., 14:28]
    jump = np.linalg.norm(p[1:, 0] - p[:-1, -1], axis=-1)
    inside = np.linalg.norm(np.diff(p, axis=1), axis=-1)
    return dict(
        arm_rmse_rad=float(np.sqrt(np.mean((p - e) ** 2))),
        boundary_jump_l2_rad=jump.tolist(),
        median_boundary_jump_l2_rad=float(np.median(jump)),
        median_inside_l2_rad=float(np.median(inside)),
        boundary_inside_ratio=float(np.median(jump) / max(np.median(inside), 1e-12)),
        first_action_error_l2_rad=np.linalg.norm(p[:, 0] - e[:, 0], axis=-1).tolist(),
        last_action_error_l2_rad=np.linalg.norm(p[:, -1] - e[:, -1], axis=-1).tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(0, 91, 10)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.975, 0.95])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(not 0 < fraction <= 1 for fraction in args.fractions) or 1.0 not in args.fractions:
        raise ValueError("Need valid crop fractions including raw=1")
    if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        raise RuntimeError("Need an isolated GPU with 16 GiB free")
    saved = json.loads((args.checkpoint / "rsci_config.json").read_text())
    if saved["variant"] not in ("VGQ-AE", "NO-STATE", "STATE"):
        raise ValueError("Unsupported policy variant")
    dataset = SimpleRHSCIDataset(args.dataset, "STATE")
    processor = Processor(
        args.dataset, args.tokenizer, saved["variant"], normalization=saved["normalization"]
    )
    model = Pi05RHSCI(saved["variant"], compile_inference=False)
    model.load_state_dict(
        load_file(str(args.checkpoint / "model.safetensors")), strict=True, assign=True
    )
    model.cuda().eval()
    low, high = [np.asarray(processor.stats["action"][key], np.float32) for key in ("q01", "q99")]
    scale = (high - low + 1e-6) / 2
    offset = low + scale
    frames = [0, 30, 60, 90]
    rows, episode_results = [], []
    for episode in args.episodes:
        slot = next(
            i for i, item in enumerate(dataset.episodes) if item["episode_index"] == episode
        )
        if dataset.episodes[slot]["length"] < 120:
            raise ValueError(f"Episode {episode} is too short")
        start = 0 if slot == 0 else int(dataset.ends[slot - 1])
        experts, raw_experts = [], []
        predictions = {(fraction, seed): [] for fraction in args.fractions for seed in args.seeds}
        for frame in frames:
            sample = dataset.raw_sample(start + frame)
            sample.update(
                rsci_history_valid=np.ones(4, bool),
                rsci_ardy_target=np.zeros(128),
                rsci_ardy_valid=False,
            )
            batch = processor(sample)
            raw_experts.append(sample["actions"])
            experts.append(batch["actions"].numpy() * scale + offset)
            tokens, mask = [batch[key][None].cuda() for key in ("tokens", "token_mask")]
            image = batch["images"][None].cuda()
            for fraction in args.fractions:
                altered = crop_image(image, fraction)
                for seed in args.seeds:
                    noise = torch.randn(
                        (1, 30, 36),
                        device="cuda",
                        generator=torch.Generator("cuda").manual_seed(seed),
                    )
                    with (
                        torch.inference_mode(),
                        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
                    ):
                        normalized = model.sample_actions(
                            altered, tokens, mask, num_steps=10, noise=noise
                        )
                    physical = normalized[0].float().cpu().numpy() * scale + offset
                    predictions[(fraction, seed)].append(physical)
                    rows.append(
                        dict(
                            episode=episode,
                            frame=frame,
                            fraction=fraction,
                            seed=seed,
                            rgb_sha256=hashlib.sha256(sample["images"].tobytes()).hexdigest(),
                            predictions=physical.tolist(),
                        )
                    )
        experts = np.stack(experts)
        result = dict(
            episode=episode,
            frames=frames,
            expert_actions_v2=experts.tolist(),
            expert_raw_to_v2_arm_rmse_rad=float(
                np.sqrt(np.mean((np.stack(raw_experts)[..., 14:28] - experts[..., 14:28]) ** 2))
            ),
            conditions=[
                dict(fraction=fraction, seed=seed, **metrics(np.stack(values), experts))
                for (fraction, seed), values in predictions.items()
            ],
        )
        episode_results.append(result)
        print(json.dumps({"episode": episode, "conditions": result["conditions"]}), flush=True)
    report = dict(
        checkpoint=str(args.checkpoint),
        config=saved,
        dataset=str(args.dataset),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        episode_results=episode_results,
        rows=rows,
        limitation="Ten selected training episodes and their first 120 actions only, not held-out "
        "success rates or closed-loop behavior. Same seed reused across frames to pair treatment; "
        "three seeds are not independent episodes. STATE receives its real measured state. "
        "Comparisons across checkpoints are not a controlled training-method ablation. "
        "Center crop isolates scale without training "
        "rotation/color. Expert metrics use actual v2-normalized/denormalized labels, preserving "
        "the saved normalization; no production processing change or parameter update.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
