"""egovla: 数据、资产与恢复前置检查。"""

import json
from pathlib import Path

from .checkpoints import atomic_json, sha256

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def asset_files(root):
    """冻结基础目录的完整文件集合；不能漏过加载优先级更高的新增文件。"""
    result = {}
    for path in root.rglob("*"):
        if path == root / "RSCI_ASSETS.json":
            continue
        if path.is_symlink() and (path.is_dir() or not path.is_file()):
            raise ValueError(f"Unsupported asset directory/broken symlink: {path}")
        if path.is_file():
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Asset escapes frozen base directory: {path}")
            result[path.relative_to(root).as_posix()] = path
    return result


def assets(root, register=False, verify=False):
    root = Path(root).resolve()
    manifest_path = root / "RSCI_ASSETS.json"
    if register:
        if manifest_path.exists():
            raise FileExistsError("Existing asset manifest is immutable; verify it instead")
        for name in (
            "config.json",
            "llm/config.json",
            "vision_tower/config.json",
            "mm_projector/config.json",
        ):
            if not (root / name).is_file():
                raise FileNotFoundError(f"Missing official component: {root / name}")
        names = asset_files(root)
        if not any(path.suffix in (".safetensors", ".bin") for path in names.values()):
            raise ValueError("No pretrained tensor files")
        atomic_json(
            manifest_path,
            dict(
                schema="egovla-base-local-v1",
                files={
                    name: dict(size=path.stat().st_size, sha256=sha256(path))
                    for name, path in names.items()
                },
            ),
        )
    if not manifest_path.is_file():
        # 原生模型目录无需先写 RHSCI 收据；只读取配置并记录现有权重位置。
        configuration = {}
        for name in (
            "config.json",
            "llm/config.json",
            "vision_tower/config.json",
            "mm_projector/config.json",
        ):
            configuration[name] = json.loads((root / name).read_text())
        weights = sorted(
            path
            for path in root.rglob("*")
            if path.suffix in (".safetensors", ".bin") and path.is_file()
        )
        if not weights:
            raise FileNotFoundError(f"No pretrained tensor files: {root}")
        return dict(
            schema="egovla-native-local-v1",
            config=configuration,
            weights={path.relative_to(root).as_posix(): path.stat().st_size for path in weights},
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "egovla-base-local-v1":
        raise ValueError("Unsupported EgoVLA asset manifest")
    actual = asset_files(root)
    recorded = manifest.get("files")
    if not isinstance(recorded, dict) or not recorded or set(actual) != set(recorded):
        added = (
            sorted(set(actual) - set(recorded or {}))
            if isinstance(recorded, dict)
            else sorted(actual)
        )
        removed = sorted(set(recorded or {}) - set(actual)) if isinstance(recorded, dict) else []
        raise ValueError(f"Asset file inventory changed: added={added}, removed={removed}")
    for name, record in manifest["files"].items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root)
            or path.stat().st_size != record["size"]
            or (verify and sha256(path) != record["sha256"])
        ):
            raise ValueError(f"Asset size/SHA mismatch: {name}")
    return manifest


def check_data(paths, variant="VGQ-AE"):
    import numpy as np

    from .data import OFFSETS, SimpleDataset

    dataset = SimpleDataset(
        paths["dataset"],
        paths["sidecar"],
        variant=variant,
        cache=paths["input_cache"],
    )

    for index in (0, 6, int(dataset.ends[0]) - 1, int(dataset.ends[0])):
        sample = dataset.raw_sample(index)
        if variant == "VGQ-AE":
            np.testing.assert_array_equal(
                sample["history_valid"], OFFSETS + sample["frame_index"] >= 0
            )
        if sample["actions"].shape != (30, 36) or any(
            "depth" in key or (variant == "VGQ-AE" and "state" in key) for key in sample
        ):
            raise ValueError("Invalid action window or unexpected state/depth")
    print(
        f"EGOVLA_DATA_CHECK_OK task={paths['task']} samples={len(dataset)}; no training started",
        flush=True,
    )
    return dataset


def validate_model_and_worker_preprocessing(args, paths):
    """加载本地模型，比较父进程与真实 spawn worker 的处理结果，不进行训练更新。"""
    from .native import load_native
    from .processing import Collator
    from .trainer import configure_vlm_execution

    model, processor = load_native(
        paths["checkpoint"], vlm_training=args.vlm_training, variant=args.model
    )
    configure_vlm_execution(model, args)
    dataset = check_data(paths, args.model)
    dataset.processor = processor
    batch = Collator(processor.tokenizer.pad_token_id)([dataset[0], dataset[6]])
    if batch["actions"].shape != (2, 30, 36):
        raise ValueError("Invalid model batch")
    # 主进程预处理通过不代表worker能用；真实spawn读取同样两帧，
    # 核对模板、动作标签、RGB/ARDY全字段，且不运行任何训练更新。
    import torch
    from torch.utils.data import DataLoader, Subset

    worker_loader = DataLoader(
        Subset(dataset, [0, 6]),
        batch_size=2,
        num_workers=1,
        multiprocessing_context="spawn",
        timeout=120,
        collate_fn=Collator(processor.tokenizer.pad_token_id),
    )
    (worker_batch,) = list(worker_loader)
    if worker_batch.keys() != batch.keys():
        raise ValueError("Worker preprocessing keys differ from parent")
    for key in batch:
        torch.testing.assert_close(
            worker_batch[key],
            batch[key],
            rtol=0,
            atol=0,
            msg=f"Worker preprocessing mismatch: {key}",
        )
    print("EGOVLA_SPAWN_PROCESSOR_CHECK_OK; no training started", flush=True)
    print(
        "EGOVLA_PARAMETERS "
        + json.dumps(
            dict(
                vlm_training=args.vlm_training,
                total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                vlm_trainable=sum(
                    p.numel()
                    for module in (
                        model.backbone.get_llm(),
                        model.backbone.get_vision_tower(),
                        model.backbone.get_mm_projector(),
                    )
                    for p in module.parameters()
                    if p.requires_grad
                ),
            )
        ),
        flush=True,
    )
    print("EGOVLA_MODEL_PROCESSOR_CHECK_OK; no training started")
