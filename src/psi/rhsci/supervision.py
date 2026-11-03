"""当前身体真值只作监督，不进入学生条件；支持普通/spawn DataLoader。"""
import numpy as np
import torch

def state_labels(raw, field):
    """沿用原生训练集min/max归一化，31维关节作目标，维32命令高度与padding不回归。"""
    raw = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
    if raw.shape != (1, 32):
        raise ValueError(f'Expected one current raw32 state, got {raw.shape}')
    if field.state_min is None or field.state_max is None or not field.normalize_state:
        raise ValueError('Controls require explicit loaded native state normalization statistics')
    low, high = np.asarray(field.state_min), np.asarray(field.state_max)
    if low.shape != (36,) or high.shape != (36,) or not np.isfinite([low, high]).all():
        raise ValueError('Expected finite padded36 native state statistics')
    valid = np.isfinite(raw[0, :31])
    # 原生零跨度分支输出常数0，不是可学习关节量，明确排除其重建损失。
    ill = np.abs(high-low) < 1e-4*(np.abs(high)+np.abs(low)+1e-8)
    valid &= ~ill[:31]
    padded = np.pad(raw.astype(np.float32), ((0, 0), (0, 4)))
    target = field.normalize_state_func(padded)[0, :31].copy()
    target[~valid] = 0
    return target, valid

def supervision_fields(variant, raw, field):
    if variant != "STATE-RECONSTRUCTION":
        raise ValueError("State labels are only used by STATE-RECONSTRUCTION")
    target, valid = state_labels(raw, field)
    return dict(state_target=target, state_valid=valid)
