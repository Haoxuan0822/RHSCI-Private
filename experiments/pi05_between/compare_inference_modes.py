"""Compare inference backends, solver steps or archived adapters on real recorded inputs."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.nn import functional as F

from openpi.rhsci.augmentation import augment_training_images
from openpi.rhsci.model import Pi05RHSCI
from openpi.rhsci.processing import Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--recorded-inputs",
        type=Path,
        help="Pair trace images with aligned recorded training inputs",
    )
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[17])
    parser.add_argument(
        "--augmentation-seeds", type=int, nargs="+", help="Compare native training augmentation"
    )
    parser.add_argument(
        "--center-crop-fractions",
        type=float,
        nargs="+",
        help="Isolate crop/resize without color or rotation",
    )
    parser.add_argument(
        "--archived-source", type=Path, help="Compare archived rsci_pi05 model and processor"
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        nargs="+",
        help="Compare eager solver step counts instead of compiling",
    )
    args = parser.parse_args()
    if args.archived_source is not None and args.sampling_steps is not None:
        parser.error("Choose archived source or solver-step comparison")
    if args.recorded_inputs is not None and args.sampling_steps is None:
        parser.error("Recorded image pairing requires --sampling-steps")
    if args.augmentation_seeds is not None and args.recorded_inputs is None:
        parser.error("Training augmentation diagnosis requires paired recorded inputs")
    if args.center_crop_fractions is not None:
        if args.recorded_inputs is None or args.augmentation_seeds is not None:
            parser.error("Center crop diagnosis requires paired inputs and no random augmentation")
        if any(not 0 < fraction < 1 for fraction in args.center_crop_fractions):
            parser.error("Center crop fractions must be between zero and one")
        if len(set(args.center_crop_fractions)) != len(args.center_crop_fractions):
            parser.error("Repeated center crop fractions")
    if args.augmentation_seeds and len(set(args.augmentation_seeds)) != len(
        args.augmentation_seeds
    ):
        parser.error("Repeated augmentation seeds")
    if args.sampling_steps is None and args.noise_seeds != [17]:
        parser.error("Multiple seeds require --sampling-steps")
    if args.sampling_steps is not None and any(step < 1 for step in args.sampling_steps):
        parser.error("Sampling steps must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("Need 12 GiB free for isolated inference comparison")
    saved = json.loads((args.checkpoint / "rsci_config.json").read_text())
    if args.recorded_inputs is not None and saved["variant"] not in ("VGQ-AE", "NO-STATE"):
        raise ValueError("Image-only pairing requires a variant that ignores state")
    # rotary 的非持久 buffer 不在 checkpoint；沿用正常构造，不能只从 meta 赋权重。
    model = Pi05RHSCI(saved["variant"], compile_inference=False)
    model.load_state_dict(
        load_file(str(args.checkpoint / "model.safetensors")), strict=True, assign=True
    )
    model.to("cuda").eval()
    processor = Processor(
        args.dataset, args.tokenizer, saved["variant"], normalization=saved["normalization"]
    )
    archived_model = archived_processor = None
    source_info = None
    if args.archived_source is not None:
        sys.path.insert(0, str(args.archived_source.parent))
        reference_model = importlib.import_module("rsci_pi05.model")
        reference_processing = importlib.import_module("rsci_pi05.processing")
        # 复用已加载的同一骨干和完整参数，避免额外复制数GB权重；比较两版方法调用。
        archived_model = reference_model.Pi05RSCI(
            saved["variant"],
            backbone=model.paligemma_with_expert,
            compile_inference=False,
        )
        archived_model.load_state_dict(model.state_dict(), strict=True, assign=True)
        archived_model.to("cuda").eval()
        if saved["normalization"] != "simple-quantile-zero-span-clipped-v2":
            raise ValueError("This archived processor supports only its fixed v2 normalization")
        archived_processor = reference_processing.Processor(
            args.dataset, args.tokenizer, saved["variant"]
        )
        contract = json.loads((args.checkpoint.parent / "launch_contract.json").read_text())
        source_info = {}
        for name in ("model.py", "processing.py"):
            digest = hashlib.sha256((args.archived_source / name).read_bytes()).hexdigest()
            expected = contract["code"].get("src/rsci_pi05/" + name)
            source_info[name] = {
                "sha256": digest,
                "checkpoint_contract_sha256": expected,
                "matches_checkpoint_contract": digest == expected,
            }
    compiled = (
        torch.compile(model.sample_actions, mode="max-autotune")
        if args.sampling_steps is None and archived_model is None
        else None
    )
    chunks = [
        json.loads(line)
        for line in args.trace.read_text().splitlines()
        if json.loads(line)["kind"] == "model_chunk"
    ]
    if args.recorded_inputs is not None:
        recorded = {
            row["frame"]: row
            for line in args.recorded_inputs.read_text().splitlines()
            if (row := json.loads(line))["episode"] == 0
        }
        paired = []
        for chunk in chunks:
            reference = recorded[chunk["step"]]
            if reference["instruction"] != args.instruction:
                raise ValueError("Recorded instruction differs")
            for source in ("rendered_replay", "training_recorded"):
                item = dict(chunk, input_source=source, expert_actions=reference["expert_actions"])
                if source == "training_recorded":
                    item["image_path"] = str(
                        args.recorded_inputs.parent / f"episode-000-frame-{chunk['step']:04d}.png"
                    )
                    item["observed_state"] = {"states": [reference["state"]]}
                paired.append(item)
        chunks = paired
    if args.augmentation_seeds is not None:
        chunks = [
            dict(chunk, augmentation_seed=seed)
            for chunk in chunks
            for seed in (None, *args.augmentation_seeds)
        ]
    if args.center_crop_fractions is not None:
        chunks = [
            dict(chunk, center_crop_fraction=fraction)
            for chunk in chunks
            for fraction in (None, *args.center_crop_fractions)
        ]
    span = np.array(processor.stats["action"]["q99"]) - np.array(processor.stats["action"]["q01"])
    results = []
    for chunk in chunks:
        sample = dict(
            text=args.instruction,
            images=[np.asarray(Image.open(chunk["image_path"]))],
            actions=np.zeros((30, 36), np.float32),
            state=np.asarray(chunk["observed_state"]["states"][0], np.float32),
            rsci_history_valid=np.ones(4, bool),
            rsci_ardy_target=np.zeros(128),
            rsci_ardy_valid=False,
        )
        batch = processor(sample)
        inputs = [batch[key][None].to("cuda") for key in ("images", "tokens", "token_mask")]
        augmentation_seed = chunk.get("augmentation_seed")
        if augmentation_seed is not None:
            # 固定增强参数且保留调用前RNG；采样噪声使用下方独立generator。
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(augmentation_seed)
                inputs[0] = augment_training_images(inputs[0])
        crop_fraction = chunk.get("center_crop_fraction")
        if crop_fraction is not None:
            # 单独复用原生增强首步的裁剪/resize公式；不旋转、不改颜色。
            image = inputs[0]
            height, width = image.shape[-2:]
            crop_height, crop_width = int(height * crop_fraction), int(width * crop_fraction)
            top, left = (height - crop_height) // 2, (width - crop_width) // 2
            cropped = (image / 2 + 0.5)[..., top : top + crop_height, left : left + crop_width]
            inputs[0] = (
                F.interpolate(
                    cropped.flatten(0, 1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).reshape_as(image)
                * 2
                - 1
            )
        # 固定同一噪声，只比较执行后端，不影响在线服务的随机数或控制流程。
        noise = torch.randn(
            1, 30, 36, device="cuda", generator=torch.Generator("cuda").manual_seed(17)
        )
        if archived_model is not None:
            old_batch = archived_processor(sample)
            same_processing = {
                key: torch.equal(value, old_batch[key]) for key, value in batch.items()
            }
            if not all(same_processing.values()):
                raise ValueError(f"Archived processor differs: {same_processing}")
            old_inputs = [
                old_batch[key][None].to("cuda") for key in ("images", "tokens", "token_mask")
            ]
            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
            ):
                current = model.sample_actions(*inputs, num_steps=10, noise=noise.clone())
                previous = archived_model.sample_actions(
                    *old_inputs, num_steps=10, noise=noise.clone()
                )
            difference = (current - previous).abs()[0].cpu().numpy()
            row = {
                "step": chunk["step"],
                "processing_equal": same_processing,
                "normalized_max_abs": float(difference.max()),
                "physical_max_abs": float((difference * (span + 1e-6) / 2).max()),
                "archived_first_arm_action": (
                    (previous[0, 0, 14:28].float().cpu().numpy() + 1) / 2 * (span[14:28] + 1e-6)
                    + np.array(processor.stats["action"]["q01"])[14:28]
                ).tolist(),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            continue
        if args.sampling_steps is not None:
            for count, seed in (
                (count, seed) for count in args.sampling_steps for seed in args.noise_seeds
            ):
                noise = torch.randn(
                    1, 30, 36, device="cuda", generator=torch.Generator("cuda").manual_seed(seed)
                )
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
                ):
                    predicted = model.sample_actions(*inputs, num_steps=count, noise=noise.clone())
                arms = (predicted[0, :, 14:28].float().cpu().numpy() + 1) / 2 * (
                    span[14:28] + 1e-6
                ) + np.array(processor.stats["action"]["q01"])[14:28]
                row = {
                    "step": chunk["step"],
                    "sampling_steps": count,
                    "seed": seed,
                    "input_source": chunk.get("input_source", "trace"),
                    "augmentation_seed": augmentation_seed,
                    "center_crop_fraction": crop_fraction,
                    "processed_image_sha256": hashlib.sha256(
                        inputs[0].contiguous().cpu().numpy().tobytes()
                    ).hexdigest(),
                    "first_arm_action": arms[0].tolist(),
                    "last_arm_action": arms[-1].tolist(),
                    "first_vs_observed_arm_l2_rad": float(
                        np.linalg.norm(arms[0] - sample["state"][14:28])
                    ),
                }
                if "expert_actions" in chunk:
                    expert_arms = np.asarray(chunk["expert_actions"])[:, 14:28]
                    row["arm_expert_rmse_rad"] = float(np.sqrt(np.mean((arms - expert_arms) ** 2)))
                    row["arm_actions"] = arms.tolist()
                results.append(row)
                print(json.dumps(row), flush=True)
            continue
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
        ):
            eager = model.sample_actions(*inputs, num_steps=10, noise=noise.clone())
            optimized = compiled(*inputs, num_steps=10, noise=noise.clone())
        difference = (eager - optimized).abs()[0].cpu().numpy()
        arms = difference[:, 14:28] * (span[14:28] + 1e-6) / 2
        row = {
            "step": chunk["step"],
            "normalized_max_abs": float(difference.max()),
            "arms_max_abs_rad": float(arms.max()),
            "arms_max_l2_rad": float(np.linalg.norm(arms, axis=-1).max()),
            "eager_first_arm_action": (
                (eager[0, 0, 14:28].float().cpu().numpy() + 1) / 2 * (span[14:28] + 1e-6)
                + np.array(processor.stats["action"]["q01"])[14:28]
            ).tolist(),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "variant": saved["variant"],
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "seed": 17,
                "noise_seeds": args.noise_seeds,
                "augmentation_seeds": args.augmentation_seeds,
                "center_crop_fractions": args.center_crop_fractions,
                "recorded_inputs": str(args.recorded_inputs) if args.recorded_inputs else None,
                "num_steps": args.sampling_steps or 10,
                "archived_source": str(args.archived_source) if source_info else None,
                "source_info": source_info,
                "shared_current_native_backbone": archived_model is not None,
                "chunks": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
