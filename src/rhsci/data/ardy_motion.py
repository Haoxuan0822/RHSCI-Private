"""Mo 变体的同 episode 差分标签与配置，不改变当前 ARDY 监督。"""

import math

import numpy as np

MOTION_VARIANTS = ("VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480")
DEFAULT_DELTA_FRAMES = 6
DEFAULT_MOTION_WEIGHT = 0.1


def resolve_motion_delta(variant, delta_frames=None):
    """显式间隔优先；Mo-480 默认回看240ms，其余版本沿用120ms。"""
    if delta_frames is not None:
        return delta_frames
    return 12 if variant == "VGQ-AE-Mo-480" else DEFAULT_DELTA_FRAMES


def validate_motion_config(delta_frames, weight):
    if type(delta_frames) is not int or delta_frames < 1:
        raise ValueError("ARDY motion delta must be a positive integer number of frames")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("ARDY motion loss weight must be finite and nonnegative")


def add_motion_arguments(parser):
    parser.add_argument(
        "--ardy-delta-frames",
        type=int,
        default=None,
        help="Backward ARDY interval at 50 Hz (Mo-480: 12/240 ms; other Mo: 6/120 ms)",
    )
    parser.add_argument(
        "--motion-weight",
        type=float,
        default=DEFAULT_MOTION_WEIGHT,
        help="Mo variants: difference loss weight (default 0.1)",
    )


def motion_config(args):
    """仅新方法写入差分配置，保留旧方法的契约与随机初始化路径。"""
    variant = getattr(args, "model", getattr(args, "variant", None))
    delta = resolve_motion_delta(variant, getattr(args, "ardy_delta_frames", None))
    weight = getattr(args, "motion_weight", DEFAULT_MOTION_WEIGHT)
    validate_motion_config(delta, weight)
    # 预检与数据构造也从同一 Namespace 读取已经解析的默认间隔。
    args.ardy_delta_frames = delta
    if variant not in MOTION_VARIANTS:
        if delta != DEFAULT_DELTA_FRAMES or weight != DEFAULT_MOTION_WEIGHT:
            raise ValueError("Motion options require a VGQ-AE-Mo variant")
        return {}
    return dict(ardy_delta_frames=delta, motion_weight=weight)


def ardy_motion_target(reader, episode, frame, delta_frames=DEFAULT_DELTA_FRAMES):
    """返回 z(t)-z(t-Δ) 和两端联合有效性；不跨 episode、不把填充当历史。"""
    validate_motion_config(delta_frames, DEFAULT_MOTION_WEIGHT)
    current = reader.get(episode, frame)
    if frame < delta_frames:
        return np.zeros(128, dtype=np.float32), False
    past = reader.get(episode, frame - delta_frames)
    if not (current.ardy_loss_mask and past.ardy_loss_mask):
        return np.zeros(128, dtype=np.float32), False
    current_latent = np.asarray(current.ardy_body_latent, dtype=np.float32)
    past_latent = np.asarray(past.ardy_body_latent, dtype=np.float32)
    if current_latent.shape != (128,) or past_latent.shape != (128,):
        raise ValueError("ARDY motion requires 128D labels in the same latent coordinates")
    if not np.isfinite(current_latent).all() or not np.isfinite(past_latent).all():
        raise ValueError("Valid ARDY motion endpoints must be finite")
    difference = current_latent - past_latent
    if not np.isfinite(difference).all():
        raise ValueError("ARDY motion difference must be finite")
    return difference, True
