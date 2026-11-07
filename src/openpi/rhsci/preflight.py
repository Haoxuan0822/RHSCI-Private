"""pi05: 数据、资产与恢复前置检查。"""

import json

from rhsci.data.multitask import (
    build_training_dataset,
    dataset_children,
    preflight_indices,
    shared_statistics,
)
from rhsci.data.visual_history import visual_history_offsets

from .checkpoints import checkpoint_ok, latest_checkpoint, sha256


def run_training_preflight(args, paths):
    """检查本地资产和真实数据处理结果，不启动训练。返回 (处理器, 数据集)；π0.5 仅查数据时处理器为 None。"""
    import numpy as np

    from .data import SimpleRHSCIDataset

    statistics = shared_statistics(paths, "pi05")

    if not args.check_data:
        from safetensors import safe_open

        from .processing import Processor, collate

        asset_manifest = paths["checkpoint"] / "RSCI_ASSETS.json"
        # 清单是可选的显式资产审计；正常加载由模型/分词器自身检查结构。
        if asset_manifest.is_file() and args.check:
            manifest = json.loads(asset_manifest.read_text())
            for name, record in manifest["files"].items():
                path = paths["checkpoint"] / name
                if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
                    raise ValueError(f"Initialization checksum/size mismatch: {path}")
            if sha256(paths["tokenizer"]) != manifest["tokenizer_sha256"]:
                raise ValueError("Tokenizer SHA256 mismatch")
        with safe_open(paths["checkpoint"] / "model.safetensors", framework="pt") as stream:
            if "action_in_proj.weight" not in stream.keys():
                raise ValueError("Not an official π0.5 checkpoint")
        processor = Processor(
            paths["dataset"],
            paths["tokenizer"],
            args.model,
            statistics=statistics,
        )
    else:
        processor = None
    dataset = build_training_dataset(
        paths,
        SimpleRHSCIDataset,
        args.model,
        processor,
        ardy_delta_frames=getattr(args, "ardy_delta_frames", 6),
        pad_actions="datasets" in paths,
    )

    offsets = np.array(visual_history_offsets(args.model))
    for child in dataset_children(dataset):
        indices = preflight_indices(child, args.model)
        for index in indices:
            sample = child.raw_sample(index)
            if args.model in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480"):
                np.testing.assert_array_equal(
                    sample["rsci_history_valid"], offsets + sample["frame_index"] >= 0
                )
            if (sample["state"] is not None) != (args.model == "STATE"):
                raise ValueError("Unexpected state dependency")
        if processor:
            batch = collate([child[indices[0]], child[indices[-1]]])
            if (
                batch["actions"].shape != (2, 30, 36)
                or not np.isfinite(batch["actions"].numpy()).all()
            ):
                raise ValueError("Invalid normalized action shape/value")
    print(
        f"PI05_DATA_CHECK_OK task={args.task} model={args.model} samples={len(dataset)}; no training started",
        flush=True,
    )
    return processor, dataset


def validate_run(args, paths, identity, digest, *, allow_started=False):
    """Reject incompatible runs and unsafe resumes before starting a worker."""
    from rhsci.runtime.numerics import reject_numerical_failure

    reject_numerical_failure(paths["output"])
    record = paths["output"] / "launch_contract.json"
    if record.exists() and json.loads(record.read_text()) != identity:
        raise ValueError(
            f"{args.task}: existing run has different configuration; use a new run-tag"
        )
    resume = args.resume or latest_checkpoint(paths["output"], len(args.gpus.split(",")), digest)
    if resume is not None:
        if resume.resolve().parent != paths["output"].resolve() or not checkpoint_ok(
            resume, len(args.gpus.split(",")), digest, verify_hash=True
        ):
            raise ValueError(f"{args.task}: incomplete, corrupt, or mismatched resume checkpoint")
    elif not allow_started and (paths["output"] / "training_started.json").exists():
        raise RuntimeError(
            f"{args.task}: no complete checkpoint; refusing restart from zero; use a new run-tag for an intentional fresh run"
        )
    return resume
