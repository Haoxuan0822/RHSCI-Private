"""四帧视觉采样协议；ARDY 教师窗口由原 sidecar 独立定义。"""


def visual_history_offsets(variant: str | None) -> tuple[int, int, int, int]:
    """返回 50Hz 下从旧到新的帧偏移；240 变体仅扩大视觉采样跨度。"""
    if variant == "VGQ-AE-240":
        return (-12, -8, -4, 0)
    return (-6, -4, -2, 0)
