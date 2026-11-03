"""Ψ0 四图 VLM Query 与状态重建；保持原生动作损失和 RTC 尺度。"""

from dataclasses import dataclass

import torch

from rhsci.runtime.losses import masked_auxiliary_loss

CONTROL_VARIANTS = ("VGQ-VLM", "STATE-RECONSTRUCTION")


@dataclass(frozen=True)
class ControlSpec:
    method: str
    auxiliary_weight: float = 0.1


def control_spec(cfg):
    return ControlSpec(cfg.rsci_variant, cfg.rsci_ardy_weight)


def apply_execution(vlm, mode):
    """省略未参与损失的词表 logits，可关闭 VLM 重算；不改变参数集合。"""
    if mode not in ("reference", "logits", "both"):
        raise ValueError("Unknown VGQ-VLM execution mode")
    if mode in ("logits", "both"):
        original = vlm.forward

        def limited_forward(*args, **kwargs):
            kwargs.setdefault("logits_to_keep", 1)
            return original(*args, **kwargs)

        vlm.forward = limited_forward
    if mode == "both":
        vlm.gradient_checkpointing_disable()


def action_mse(prediction, target, mask, weights):
    """保持 Ψ0 原尺度：时间求和、batch 平均，最后按动作维权重求和。"""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Action target/mask shape mismatch")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite action prediction/target")
    errors = (prediction.float() - target.float()).square() * mask
    return errors.sum(1).mean(0).mul(weights).sum()


def reconstruction_loss(prediction, target, valid):
    """按全局有效关节元素归一化；无效标签可为 NaN，空标签保留零梯度图。"""
    if prediction.shape != target.shape or valid.shape != target.shape or target.shape[-1] != 31:
        raise ValueError("State supervision requires [B,31] target and element validity")
    valid = valid.bool()
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all():
        raise FloatingPointError("Nonfinite valid reconstruction value")
    safe_target = torch.where(valid, target, prediction.detach())
    error = (prediction.float() - safe_target.float()).square()
    count = valid.float().sum().detach()
    world = 1
    if torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        torch.distributed.all_reduce(count)
    return (error * valid).sum() * world / count.clamp_min(1)


def forward_loss(
    model, batch, spec, *, loss_weights=None, diffusion_steps=1000,
    rtc=True, max_delay=8, eval_features=None, sigma=None, noise=None, delay=None,
):
    """动作与辅助监督共享 flow 样本；真实 state 仅作重建标签。

    验证也重新计算四帧辅助前向，不用 eval_features 的单帧采样缓存。
    sigma/noise/delay 可由数值对照测试显式传入。
    """
    actions = batch["actions"]
    if actions.ndim != 3 or not torch.isfinite(actions).all():
        raise ValueError("Expected finite [B,T,D] action labels")
    batch_size, horizon, dim = actions.shape
    sigma = torch.rand(batch_size, device=actions.device) if sigma is None else sigma
    noise = torch.randn_like(actions) if noise is None else noise
    if sigma.shape != (batch_size,) or noise.shape != actions.shape:
        raise ValueError("Invalid shared flow sample")
    sigmas = sigma[:, None].expand(-1, horizon)
    if rtc:
        if not 0 < max_delay <= horizon:
            raise ValueError("RTC delay must be positive and no greater than horizon")
        if delay is None:
            delay = torch.randint(0, max_delay, (batch_size,), device=actions.device)
        if delay.shape != (batch_size,) or (delay < 0).any() or (delay >= max_delay).any():
            raise ValueError("Invalid RTC delay")
        prefix = torch.arange(horizon, device=actions.device)[None] < delay[:, None]
        sigmas = torch.where(prefix, 0.0, sigmas)
    else:
        prefix = torch.zeros((batch_size, horizon), device=actions.device, dtype=torch.bool)
    noisy = (1 - sigmas[..., None]) * actions + sigmas[..., None] * noise
    timestep = sigmas * diffusion_steps if rtc else sigma * diffusion_steps
    # 即使外部 batch 意外带入真实 state，也不让其成为学生条件。
    kwargs = {key: batch[key] for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
    kwargs.update(action_samples=noisy, states=torch.zeros_like(batch["states"]), timestep=timestep)
    kwargs.update(history_vlm_inputs=batch["history_vlm_inputs"], history_valid=batch["history_valid"])
    output = model(**kwargs, compute_rsci_aux=True, return_dict=True)
    mask = batch.get("actions_mask", torch.ones_like(actions)).float() * (~prefix)[..., None]
    if mask.shape != actions.shape or not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError("Invalid action padding/RTC mask")
    weights = torch.full((dim,), 1 / dim, device=actions.device) if loss_weights is None else loss_weights
    imitation = action_mse(output.action, noise - actions, mask, weights)
    losses = {"loss": imitation, "loss_action": imitation}
    if spec.method == "VGQ-VLM":
        valid = batch["ardy_valid"].bool() & batch["history_valid"].bool().all(-1)
        auxiliary = masked_auxiliary_loss(output.ardy_prediction, batch["ardy_target"], valid)
        losses.update(loss_ardy=auxiliary, loss_ardy_weighted=spec.auxiliary_weight * auxiliary)
    elif spec.method == "STATE-RECONSTRUCTION":
        # 沿用 VGQ-AE 的有效窗口；ARDY 数值不参与重建监督。
        valid = batch["state_valid"].bool() & batch["ardy_valid"].bool()[:, None]
        auxiliary = reconstruction_loss(output.ardy_prediction, batch["state_target"], valid)
        losses.update(loss_state=auxiliary, loss_state_weighted=spec.auxiliary_weight * auxiliary)
    else:
        raise ValueError(f"Unknown control method: {spec.method}")
    losses["loss"] = imitation + spec.auxiliary_weight * auxiliary
    if not all(torch.isfinite(value).all() for value in losses.values()):
        raise FloatingPointError("Nonfinite control loss")
    return losses
