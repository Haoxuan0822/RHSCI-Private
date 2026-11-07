"""四帧视觉采样协议；ARDY 教师窗口由原 sidecar 独立定义。"""


def visual_history_offsets(variant: str | None) -> tuple[int, int, int, int]:
    """返回 50Hz 下从旧到新的帧偏移；后缀指定四帧覆盖的毫秒数。"""
    if variant in ("VGQ-AE-480", "VGQ-AE-Mo-480"):
        return (-24, -16, -8, 0)
    if variant in ("VGQ-AE-240", "VGQ-AE-Mo-240"):
        return (-12, -8, -4, 0)
    return (-6, -4, -2, 0)
