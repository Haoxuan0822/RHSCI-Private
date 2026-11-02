"""π0.5 的有效归一化范围；分位数退化时用实际极值区分恒定与稀有变化。"""

import json
from pathlib import Path

import numpy as np

NORMALIZATION_VERSION = "simple-quantile-rare-range-clipped-v3"
LEGACY_NORMALIZATION = "simple-quantile-zero-span-clipped-v2"


def build_normalization_ranges(stats):
    """返回可写入契约/断点的范围；仅退化分位数维度回退至数据的 min/max。"""
    ranges = {}
    for key in ("action", "states"):
        entry = stats[key]
        low, high = (np.asarray(entry[name], np.float32) for name in ("q01", "q99"))
        if (
            low.ndim != 1
            or low.shape != high.shape
            or not np.isfinite(low).all()
            or not np.isfinite(high).all()
            or np.any(high < low)
        ):
            raise ValueError(f"Invalid π0.5 quantile statistics: {key}")
        degenerate = high - low <= 1e-4 * (np.abs(high) + np.abs(low) + 1e-8)
        if degenerate.any():
            if "min" not in entry or "max" not in entry:
                raise ValueError(
                    f"π0.5 {key}: degenerate quantiles require true min/max statistics"
                )
            minimum, maximum = (np.asarray(entry[name], np.float32) for name in ("min", "max"))
            if (
                minimum.shape != low.shape
                or maximum.shape != high.shape
                or not np.isfinite(minimum).all()
                or not np.isfinite(maximum).all()
                or np.any(minimum > low)
                or np.any(maximum < high)
            ):
                raise ValueError(f"Invalid π0.5 min/max statistics: {key}")
            low = np.where(degenerate, minimum, low)
            high = np.where(degenerate, maximum, high)
        ranges[key] = {"low": low.tolist(), "high": high.tolist()}
    validate_normalization_ranges(ranges)
    return ranges


def validate_normalization_ranges(ranges):
    """校验保存的逐维范围；high == low 才表示真正恒定。"""
    for key in ("action", "states"):
        low, high = (np.asarray(ranges[key][name], np.float32) for name in ("low", "high"))
        if (
            low.ndim != 1
            or low.shape != high.shape
            or not np.isfinite(low).all()
            or not np.isfinite(high).all()
            or np.any(high < low)
        ):
            raise ValueError(f"Invalid π0.5 normalization ranges: {key}")


def load_normalization_ranges(root):
    """读取小型数据统计，不读取 RGB、训练样本或权重。"""
    return build_normalization_ranges(json.loads((Path(root) / "meta/stats.json").read_text()))
