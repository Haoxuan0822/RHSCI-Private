"""Mo 变体的同 episode 差分标签与配置，不改变当前 ARDY 监督。"""

import math

import numpy as np

MOTION_VARIANTS = ("VGQ-AE-Mo", "VGQ-AE-Mo-240")
DEFAULT_DELTA_FRAMES = 6
DEFAULT_MOTION_WEIGHT = 0.1


def validate_motion_config(delta_frames, weight):
    if type(delta_frames) is not int or delta_frames < 1:
        raise ValueError("ARDY motion delta must be a positive integer number of frames")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("ARDY motion loss weight must be finite and nonnegative")


def add_motion_arguments(parser):
    parser.add_argument(
        "--ardy-delta-frames",
        type=int,
        default=DEFAULT_DELTA_FRAMES,
        help="Mo variants: backward ARDY difference interval at 50 Hz (default 6; 120 ms)",
    )
    parser.add_argument(
        "--motion-weight",
        type=float,
        default=DEFAULT_MOTION_WEIGHT,
        help="Mo variants: difference loss weight (default 0.1)",
    )


def motion_config(args):
    """仅新方法写入差分配置，保留旧方法的契约与随机初始化路径。"""
    delta = getattr(args, "ardy_delta_frames", DEFAULT_DELTA_FRAMES)
    weight = getattr(args, "motion_weight", DEFAULT_MOTION_WEIGHT)
    validate_motion_config(delta, weight)
    variant = getattr(args, "model", getattr(args, "variant", None))
    if variant not in MOTION_VARIANTS:
        if delta != DEFAULT_DELTA_FRAMES or weight != DEFAULT_MOTION_WEIGHT:
            raise ValueError("Motion options require VGQ-AE-Mo or VGQ-AE-Mo-240")
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
