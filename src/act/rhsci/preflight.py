"""act: 数据、资产与恢复前置检查。"""

import json
from pathlib import Path

from rhsci.runtime.environment import get_env

from .checkpoints import sha256

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def check_data(args, paths):
    import numpy as np
    import pyarrow.parquet as pq

    from .data import ACTDataset

    dataset = ACTDataset(
        paths["dataset"],
        paths["sidecar"],
        args.model,
        args.seed,
        cache=paths["input_cache"],
    )
    info = json.loads((paths["dataset"] / "meta/info.json").read_text())
    episodes = [
        json.loads(line)
        for line in (paths["dataset"] / "meta/episodes.jsonl").read_text().splitlines()
        if line
    ]
    if info["total_episodes"] != len(episodes) or len(dataset) != sum(
        episode["length"] for episode in episodes
    ):
        raise ValueError("Dataset metadata and frame inventory disagree")
    scanned = 0
    for path in sorted((paths["dataset"] / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["action", "states"]).to_pydict()
        action = np.asarray(table["action"], np.float32)
        state = np.asarray(table["states"], np.float32)
        if (
            action.shape != (len(state), 36)
            or state.shape[1:] != (32,)
            or not (np.isfinite(action).all() and np.isfinite(state).all())
        ):
            raise ValueError(f"Invalid raw action/state: {path}")
        normalized = dataset.processor.field(
            dict(actions=action, states=np.pad(state, ((0, 0), (0, 4))))
        )
        if not (
            np.isfinite(normalized["actions"]).all() and np.isfinite(normalized["states"]).all()
        ):
            raise ValueError(f"Nonfinite native normalization: {path}")
        scanned += len(action)
    if scanned != len(dataset):
        raise ValueError("Parquet inventory does not match dataset length")
    first_length = episodes[0]["length"]
    for index in sorted(
        {
            0,
            min(6, len(dataset) - 1),
            first_length - 1,
            min(first_length, len(dataset) - 1),
            len(dataset) - 1,
        }
    ):
        sample = dataset[index]
        if sample["action"].shape != (16, 36) or sample["observation.images"].shape != (
            1,
            3,
            224,
            224,
        ):
            raise ValueError("Native ACT action/image shape mismatch")
        if index in (first_length - 1, len(dataset) - 1) and sample["action_is_pad"].sum() != 15:
            raise ValueError("Episode tail padding changed")
    print(
        f"ACT_DATA_CHECK_OK task={args.task} model={args.model} samples={len(dataset)} scanned={scanned}; no training started",
        flush=True,
    )
    return dataset


def initialization():
    """禁止torchvision隐式联网；由已有本地ImageNet文件初始化，不改变其它模型环境。"""
    import torch

    path = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    if not path.is_file():
        raise FileNotFoundError(
            f"Official ImageNet ResNet18 initialization missing: {path}; no automatic download"
        )
    digest = sha256(path)
    if not digest.startswith("f37072fd"):
        raise ValueError("ImageNet ResNet18 SHA prefix mismatch")
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def nccl_runtime():
    """Bind the explicitly preloaded runtime; never replace a shared installation."""
    name = get_env("RHSCI_NCCL_LIBRARY")
    if name is None:
        return None
    import ctypes

    path = Path(name).resolve(strict=True)
    version = ctypes.c_int()
    library = ctypes.CDLL(None)
    expected = tuple(map(int, get_env("RHSCI_NCCL_VERSION", "2.27.3").split(".")))
    expected = expected[0] * 10000 + expected[1] * 100 + expected[2]
    if library.ncclGetVersion(ctypes.byref(version)) != 0 or version.value != expected:
        raise RuntimeError(f"ACT requires the selected preloaded NCCL runtime {expected}")
    mapped = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libnccl.so" in line
    }
    if mapped != {path}:
        raise RuntimeError(f"Unexpected loaded NCCL libraries: {mapped}")
    return dict(path=str(path), sha256=sha256(path), version=version.value)
