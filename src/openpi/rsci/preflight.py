"""pi05: 数据、资产与恢复前置检查。"""

import json

from .checkpoints import checkpoint_ok, latest_checkpoint, sha256

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def run_training_preflight(args, paths):
    """检查本地资产和真实数据处理结果，不启动训练。返回 (处理器, 数据集)；π0.5 仅查数据时处理器为 None。"""
    import numpy as np

    from .data import SimpleRSCIDataset
    from .processing import Processor, collate

    if not args.check_data:
        from safetensors import safe_open

        asset_manifest = paths["checkpoint"] / "RSCI_ASSETS.json"
        if not asset_manifest.is_file():
            raise ValueError(
                "Missing RSCI_ASSETS.json; register local weights and tokenizer with "
                "python -m openpi.rsci.prepare_assets --checkpoint PATH --tokenizer PATH; "
                "see baselines/pi05/README.md"
            )
        manifest = json.loads(asset_manifest.read_text())
        for name, record in manifest["files"].items():
            path = paths["checkpoint"] / name
            if path.stat().st_size != record["size"]:
                raise ValueError(f"Initialization file size mismatch: {path}")
            if not args.worker and args.guardian_fd is None and sha256(path) != record["sha256"]:
                raise ValueError(f"Initialization SHA256 mismatch: {path}")
        if sha256(paths["tokenizer"]) != manifest["tokenizer_sha256"]:
            raise ValueError("Tokenizer SHA256 mismatch")
        with safe_open(paths["checkpoint"] / "model.safetensors", framework="pt") as stream:
            if "action_in_proj.weight" not in stream.keys():
                raise ValueError("Not an official π0.5 checkpoint")
        processor = Processor(
            paths["dataset"],
            paths["tokenizer"],
            args.model,
            state_measurement_noise_std_rad=args.state_measurement_noise_std_rad,
        )
    else:
        processor = None
    dataset = SimpleRSCIDataset(
        paths["dataset"],
        args.model,
        processor,
        paths["sidecar"],
        cache=paths["input_cache"],
    )

    for index in (0, 6, int(dataset.ends[0]) - 1, int(dataset.ends[0]), 6):
        sample = dataset.raw_sample(index)
        if args.model == "VGQ-AE":
            np.testing.assert_array_equal(
                sample["rsci_history_valid"], np.array([-6, -4, -2, 0]) + sample["frame_index"] >= 0
            )
        if (sample["state"] is not None) != (args.model == "STATE"):
            raise ValueError("Unexpected state dependency")
    if processor:
        batch = collate([dataset[0], dataset[6]])
        if batch["actions"].shape != (2, 30, 36) or not np.isfinite(batch["actions"].numpy()).all():
            raise ValueError("Invalid normalized action shape/value")
    print(
        f"PI05_DATA_CHECK_OK task={args.task} model={args.model} samples={len(dataset)}; no training started",
        flush=True,
    )
    return processor, dataset


def validate_run(args, paths, identity, digest, *, allow_started=False):
    """Reject incompatible runs and unsafe resumes before starting a worker."""
    from rsci.runtime.numerics import reject_numerical_failure

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
