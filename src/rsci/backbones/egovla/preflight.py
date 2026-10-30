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


def assets(root, register=False, verify=True):
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
        raise FileNotFoundError(
            f"EgoVLA assets not prepared: {manifest_path}; no automatic download"
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
