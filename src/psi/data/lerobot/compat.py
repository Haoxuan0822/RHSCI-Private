from __future__ import annotations

import functools
import importlib
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

_simple_rgb_loading = ContextVar("psi_simple_rgb_loading", default=False)


@contextmanager
def simple_rgb_parquet_loading():
    """只在SIMPLE RGB构造期间启用数值列投影，原生视频/时间查询不变。"""
    token = _simple_rgb_loading.set(True)
    try:
        yield
    finally:
        _simple_rgb_loading.reset(token)

try:
    from lerobot.common.datasets.lerobot_dataset import (  # type: ignore
        LeRobotDataset,
        LeRobotDatasetMetadata,
        MultiLeRobotDataset,
    )

    LEROBOT_LAYOUT = "common"
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import (  # type: ignore
        LeRobotDataset,
        LeRobotDatasetMetadata,
        MultiLeRobotDataset,
    )

    LEROBOT_LAYOUT = "datasets"


def complete_numeric_episode_counts(statistics, episodes):
    """仅在内存中补齐旧SIMPLE逐帧数值统计的样本数，不修改磁盘统计。

    states/action的每个样本对应一个轨迹帧；图像等可能经过子采样的统计
    不能用轨迹长度猜测。已有count原样保留，聚合仍交给LeRobot实现。
    """
    for episode_index, features in statistics.items():
        for feature, values in features.items():
            if "count" in values:
                continue
            if feature not in ("states", "action"):
                raise ValueError(f"Cannot infer missing statistics count: {feature}")
            length = episodes[episode_index]["length"]
            if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
                raise ValueError(f"Invalid episode length for statistics: {episode_index}")
            values["count"] = np.asarray([length], dtype=np.int64)
    return statistics


def _install_episode_count_compatibility():
    """在原生元数据读取边界适配，覆盖其内部单库/多库构造，不复制加载器。"""
    native = importlib.import_module(LeRobotDatasetMetadata.__module__)
    load = getattr(native, "load_episodes_stats", None)
    if load is None or getattr(load, "_rhsci_episode_counts", False):
        return

    @functools.wraps(load)
    def load_with_counts(root):
        statistics = load(root)
        if any(
            "count" not in value
            for features in statistics.values()
            for value in features.values()
        ):
            complete_numeric_episode_counts(statistics, native.load_episodes(root))
        return statistics

    load_with_counts._rhsci_episode_counts = True
    native.load_episodes_stats = load_with_counts


_install_episode_count_compatibility()


def _install_simple_rgb_projection():
    native = importlib.import_module(LeRobotDatasetMetadata.__module__)
    load = LeRobotDataset.load_hf_dataset
    if getattr(load, "_rhsci_rgb_projection", False):
        return

    @functools.wraps(load)
    def load_projected(self):
        if not _simple_rgb_loading.get():
            return load(self)
        # 使用元数据声明的轨迹顺序；不读取未使用的depth或来源相关附加列。
        episodes = self.episodes if self.episodes is not None else list(self.meta.episodes)
        files = [str(self.root / self.meta.get_data_file_path(index)) for index in episodes]
        dataset = native.load_dataset(
            "parquet", data_files=files, split="train",
            columns=[
                "states", "action", "timestamp", "frame_index",
                "episode_index", "index", "task_index",
            ],
        )
        dataset.set_transform(native.hf_transform_to_torch)
        return dataset

    load_projected._rhsci_rgb_projection = True
    LeRobotDataset.load_hf_dataset = load_projected


_install_simple_rgb_projection()

__all__ = [
    "LEROBOT_LAYOUT",
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "MultiLeRobotDataset",
]
