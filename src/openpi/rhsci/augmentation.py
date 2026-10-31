"""调用原生π0.5 PyTorch增强；四帧共享一组几何和颜色随机参数。"""

from types import SimpleNamespace

import torch

from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch

AUGMENTATION_VERSION = "openpi-pytorch-shared-history-v1"


@torch.no_grad()
def augment_training_images(images):
    """仅在训练前向调用；展平时间维使历史与当前复用官方单相机增强。"""
    if images.ndim != 5 or images.shape[2] != 3 or min(images.shape[-2:]) < 2:
        raise ValueError("Expected images [B,T,3,H,W] with H,W >= 2")
    batch, frames, channels, height, width = images.shape
    observation = SimpleNamespace(
        images={"base_0_rgb": images.reshape(-1, channels, height, width).float()},
        image_masks={},
        state=images.new_zeros(batch * frames, 1),
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    # 生产输入已由Processor处理为224；小模型回归允许较小分辨率，不额外resize。
    processed = preprocess_observation_pytorch(
        observation, train=True, image_keys=("base_0_rgb",), image_resolution=(height, width)
    )
    return processed.images["base_0_rgb"].reshape(batch, frames, channels, height, width)
