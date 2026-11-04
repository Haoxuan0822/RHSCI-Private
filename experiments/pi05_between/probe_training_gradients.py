"""生产尺寸模型执行方式梯度对照；区分正式与合成ARDY标签，不更新参数。"""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from openpi.rhsci.data import SimpleRHSCIDataset
from openpi.rhsci.model import Pi05RHSCI
from openpi.rhsci.processing import Processor, collate


def compare_gradients(actual, expected):
    """保留所有参数的比较摘要，不用均值掩盖个别参数漏梯度。"""
    if actual.keys() != expected.keys():
        raise ValueError("Trainable parameter names differ")
    squared_error = squared_reference = 0.0
    worst, mismatched, outside_tolerance = (0.0, None), [], []
    groups = {}
    for name, value in actual.items():
        reference = expected[name]
        if value is None or reference is None:
            raise ValueError(f"Missing trainable gradient: {name}")
        difference = value - reference
        error = float(difference.abs().max())
        if error > worst[0]:
            worst = error, name
        squared_error += float(difference.square().sum(dtype=torch.float64))
        squared_reference += float(reference.square().sum(dtype=torch.float64))
        group = (
            "shared_action_expert"
            if name.startswith("paligemma_with_expert.gemma_expert")
            else "action_projection"
            if name.startswith("action_")
            else "shared_time"
            if name.startswith("time_mlp")
            else "auxiliary_only"
        )
        totals = groups.setdefault(group, [0.0, 0.0, 0.0])
        totals[0] += float(difference.square().sum(dtype=torch.float64))
        totals[1] += float(reference.square().sum(dtype=torch.float64))
        totals[2] = max(totals[2], error)
        if not torch.equal(value, reference):
            mismatched.append(name)
        # 沿用现有validate_pi05_speed的BF16容差，同时报告原始误差与逐位差异。
        if not torch.allclose(value, reference, rtol=0.03, atol=5e-4):
            outside_tolerance.append(name)
    return dict(
        parameters=len(actual),
        bitwise_equal=not mismatched,
        mismatched_parameters=mismatched,
        outside_existing_bf16_tolerance=outside_tolerance,
        max_abs_error=worst[0],
        worst_parameter=worst[1],
        global_relative_l2=(squared_error / max(squared_reference, 1e-30)) ** 0.5,
        groups={
            name: dict(
                relative_l2=(error / max(norm, 1e-30)) ** 0.5,
                reference_l2=norm**0.5,
                max_abs_error=maximum,
            )
            for name, (error, norm, maximum) in groups.items()
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "tokenizer", "archived-source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=60)
    parser.add_argument("--repeat-control", action="store_true")
    parser.add_argument("--sidecar", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    free, total = torch.cuda.mem_get_info()
    if free < 16 * 1024**3:
        raise RuntimeError("Need 16 GiB free for isolated forward/backward diagnostics")
    torch.cuda.set_per_process_memory_fraction((free - 2 * 1024**3) / total)
    saved = json.loads((args.checkpoint / "rsci_config.json").read_text())
    if saved["variant"] != "VGQ-AE":
        raise ValueError("Expected the selected VGQ-AE checkpoint")
    contract = json.loads((args.checkpoint.parent / "launch_contract.json").read_text())
    source = args.archived_source / "model.py"
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != contract["code"]["src/rsci_pi05/model.py"]
    ):
        raise ValueError("Archived model does not match the actual training contract")
    sys.path.insert(0, str(args.archived_source.parent))
    archived_module = importlib.import_module("rsci_pi05.model")
    if Path(archived_module.__file__).resolve() != source.resolve():
        raise ValueError("Imported another archived implementation")
    if args.sidecar is not None:
        if (
            hashlib.sha256((args.sidecar / "manifest.json").read_bytes()).hexdigest()
            != contract["sidecar"]
        ):
            raise ValueError("ARDY sidecar manifest differs from the training contract")
        dataset = SimpleRHSCIDataset(args.dataset, "VGQ-AE", sidecar=args.sidecar)
    else:
        dataset = SimpleRHSCIDataset(args.dataset, "STATE")
    if args.frame < 6 or args.frame + 30 > dataset.episodes[0]["length"]:
        raise ValueError("Need four valid history frames and a full episode-0 action window")
    if args.sidecar is not None:
        sample = dataset.raw_sample(args.frame)
        if not sample["rsci_ardy_valid"]:
            raise ValueError("Selected formal ARDY label is invalid")
    else:
        samples = [dataset.raw_sample(args.frame + offset) for offset in (-6, -4, -2, 0)]
        sample = dict(samples[-1], images=np.concatenate([row["images"] for row in samples]))
        sample.update(
            rsci_history_valid=np.ones(4, bool),
            rsci_ardy_target=np.zeros(128, np.float32),
            rsci_ardy_valid=True,
        )
    processor = Processor(
        args.dataset, args.tokenizer, "VGQ-AE", normalization=saved["normalization"]
    )
    batch = {key: value.cuda() for key, value in collate([processor(sample)]).items()}
    model = Pi05RHSCI("VGQ-AE", compile_inference=False)
    model.load_state_dict(
        load_file(str(args.checkpoint / "model.safetensors")), strict=True, assign=True
    )
    model.cuda()
    original = archived_module.Pi05RSCI(
        "VGQ-AE", backbone=model.paligemma_with_expert, compile_inference=False, execution="both"
    )
    original.load_state_dict(model.state_dict(), strict=True, assign=True)
    original.cuda()
    reports = []
    formal_target = batch["ardy_target"].clone() if args.sidecar is not None else None
    formal_reference = None
    targets = (
        ["formal_ardy"] if args.sidecar is not None else ["synthetic_zero", "synthetic_linear"]
    )
    profiles = ["reference", "query", "history", "both", "authenticated_original_both"]
    if args.repeat_control:
        targets.append("formal_masked" if args.sidecar is not None else "synthetic_masked")
        profiles = [
            "reference",
            "reference_repeat",
            "query",
            "history",
            "both",
            "both_repeat",
            "authenticated_original_both",
        ]
    for target_name in targets:
        target_index = int(target_name == "synthetic_linear")
        batch["ardy_target"] = (
            formal_target.clone()
            if formal_target is not None
            else torch.zeros(1, 128, device="cuda")
            if target_index == 0
            else torch.linspace(-1, 1, 128, device="cuda")[None]
        )
        batch["ardy_valid"].fill_(not target_name.endswith("masked"))
        reference = reference_rng = both_gradients = None
        for name in profiles:
            model.zero_grad(set_to_none=True)
            original.zero_grad(set_to_none=True)
            active = original if name == "authenticated_original_both" else model
            if active is model:
                active.execution = name.removesuffix("_repeat")
            active.train()
            torch.manual_seed(292285 + target_index)
            torch.cuda.manual_seed_all(292285 + target_index)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                result = active(**batch)
            result["loss"].backward()
            gradients = {
                key: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                for key, parameter in active.named_parameters()
                if parameter.requires_grad
            }
            if any(
                value is not None and not torch.isfinite(value).all()
                for value in gradients.values()
            ):
                raise FloatingPointError("Nonfinite gradients")
            if any(
                parameter.grad is not None
                for parameter in active.paligemma_with_expert.paligemma.parameters()
            ):
                raise ValueError("Unexpected frozen VLM gradients")
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
            if reference is None:
                reference, reference_rng = gradients, rng
            row = dict(
                target=target_name,
                profile=name,
                losses={key: float(value.detach()) for key, value in result.items()},
                comparison_to_current_reference=compare_gradients(gradients, reference),
                rng_matches_reference=all(torch.equal(a, b) for a, b in zip(rng, reference_rng)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            if name == "both":
                both_gradients = gradients
            if name == "reference" and target_name == "formal_ardy":
                formal_reference = gradients
            if name == "reference" and target_name == "formal_masked":
                row["comparison_to_formal_combined"] = compare_gradients(
                    gradients, formal_reference
                )
            if name in ("both_repeat", "authenticated_original_both"):
                row["comparison_to_current_both"] = compare_gradients(gradients, both_gradients)
            reports.append(row)
            print(json.dumps(row), flush=True)
            del gradients, result
        del reference, reference_rng, both_gradients
    report = dict(
        checkpoint=str(args.checkpoint),
        frame=args.frame,
        sidecar=str(args.sidecar) if args.sidecar is not None else None,
        history_frames=[args.frame + offset for offset in (-6, -4, -2, 0)],
        archived_model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=reports,
        scope="Production-size trained weights, actual episode-0 RGB/action history, batch 1 BF16, native augmentation/noise and backward. Formal targets are used only when --sidecar matches the training contract and passes the native dataset/sidecar validation; otherwise explicitly synthetic targets exercise computational equivalence. Masked versus combined gradient subtraction includes measured backward variability. No optimizer, parameter update, DDP or claim of historical training reproduction.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
