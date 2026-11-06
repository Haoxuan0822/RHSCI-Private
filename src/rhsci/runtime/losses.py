"""跨骨干辅助监督：有效标签平均及 DDP 梯度归一化。"""

import torch
from torch.nn import functional as F


def masked_auxiliary_loss(
    prediction,
    target,
    valid,
    *,
    smooth_l1=False,
    global_reduction=True,
    normalization_denominator=None,
):
    """输入 [B,D] 预测/目标和 [B] 有效性，返回标量；空标签保留零梯度计算图。

    先清除无效目标，再求逐样本误差。DDP 会平均梯度，因此以全局有效数
    为分母并乘 world_size；验证或独立计算可关闭全局归一化。
    """
    if (
        prediction.ndim != 2
        or target.shape != prediction.shape
        or valid.shape != prediction.shape[:1]
    ):
        raise ValueError("auxiliary target/mask shapes do not match prediction")
    valid = valid.bool()
    if not torch.isfinite(target[valid]).all():
        raise ValueError("valid auxiliary labels must be finite")
    target = torch.where(valid[:, None], target, torch.zeros_like(target))
    loss = F.smooth_l1_loss if smooth_l1 else F.mse_loss
    errors = loss(prediction.float(), target.float(), reduction="none").mean(-1)
    world = 1
    if global_reduction and torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
    if normalization_denominator is None:
        count = valid.sum().to(errors.dtype)
        if world > 1:
            torch.distributed.all_reduce(count)
        denominator = count.clamp_min(1)
    else:
        # 训练器已汇总整个累积组的全局有效数，并补偿其 backward 除数。
        denominator = normalization_denominator.to(device=errors.device, dtype=errors.dtype)
    return (errors * valid).sum() * world / denominator


def masked_ardy_loss(
    prediction,
    target,
    valid,
    *,
    global_reduction=True,
    normalization_denominator=None,
):
    """ARDY 标签固定为 128 维；误差与分布式归一化复用同一个辅助损失。"""
    if prediction.ndim != 2 or prediction.shape[-1] != 128:
        raise ValueError("ARDY prediction must have shape [B,128]")
    return masked_auxiliary_loss(
        prediction,
        target,
        valid,
        global_reduction=global_reduction,
        normalization_denominator=normalization_denominator,
    )
