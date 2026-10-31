"""RHSCI encoder-guided queries use the official DINOv2-S/14 RGB backbone."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Dinov2Config, Dinov2Model


def validate_rgb(rgb):
    """Validate original RGB images as uint8 (B,3,H,W)."""
    if rgb is None or rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.dtype != torch.uint8:
        raise ValueError("independent RGB requires original uint8 rgb_images (B,3,H,W)")


class DINOv2RGBAdapter(nn.Module):
    def __init__(self, pretrained_path, target_size=(196, 350), *, restore=False):
        super().__init__()
        self.target_size = (
            (target_size, target_size) if isinstance(target_size, int) else tuple(target_size)
        )
        if len(self.target_size) != 2 or any(s <= 0 or s % 14 for s in self.target_size):
            raise ValueError("DINOv2 input size must be positive and divisible by 14")
        # 完整 checkpoint 已包含 RGB 骨干；恢复时不依赖原始模型目录，不重复读取初始权重。
        config = Dinov2Config(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            image_size=518,
            patch_size=14,
        )
        config._attn_implementation = "sdpa"
        self.register_buffer("weight_sha256", torch.zeros(32, dtype=torch.uint8))
        self.register_buffer("config_sha256", torch.zeros(32, dtype=torch.uint8))
        if restore:
            self.encoder = Dinov2Model(config)
        else:
            if pretrained_path is None or not Path(pretrained_path).is_dir():
                raise ValueError(
                    "independent RGB training requires a local rsci_rgb_encoder_path directory"
                )
            self.encoder, loading = Dinov2Model.from_pretrained(
                pretrained_path,
                local_files_only=True,
                output_loading_info=True,
                attn_implementation="sdpa",
            )
            if any(
                loading.get(key)
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            ):
                raise ValueError("RGB bootstrap checkpoint must contain complete DINOv2 weights")
            # 严格限定首版官方 S/14 架构，保证完整恢复时参数形状及计算契约一致。
            for key in (
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "image_size",
                "patch_size",
                "mlp_ratio",
                "use_swiglu_ffn",
                "qkv_bias",
                "num_channels",
                "layer_norm_eps",
                "hidden_act",
                "layerscale_value",
                "hidden_dropout_prob",
                "attention_probs_dropout_prob",
                "drop_path_rate",
                "use_mask_token",
            ):
                if getattr(self.encoder.config, key) != getattr(config, key):
                    raise ValueError(
                        f"RGB encoder must use official DINOv2-S/14 configuration: {key}"
                    )
            # SHA256 随完整 checkpoint 保存；不会把数据盘路径当作资产身份或发布权重。
            directory = Path(pretrained_path)
            weight_file = directory / "model.safetensors"
            if not weight_file.is_file():
                weight_file = directory / "pytorch_model.bin"
            for source, target in (
                (weight_file, self.weight_sha256),
                (directory / "config.json", self.config_sha256),
            ):
                digest = hashlib.sha256()
                with source.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                target.copy_(torch.tensor(list(digest.digest()), dtype=torch.uint8))
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # 外层 model.train() 不得重新启用冻结骨干的 dropout/stochastic depth。
        return self

    @torch.no_grad()
    def forward(self, rgb_images):
        validate_rgb(rgb_images)
        parameter = next(self.encoder.parameters())
        pixels = rgb_images.to(device=parameter.device, dtype=torch.float32) / 255
        # 保留完整视野，无 center crop；官方 RGB mean/std，使用配置指定的编码尺寸。
        pixels = F.interpolate(
            pixels, size=self.target_size, mode="bicubic", align_corners=False, antialias=True
        ).clamp(0, 1)
        mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        pixels = ((pixels - mean) / std).to(parameter.dtype)
        tokens = self.encoder(pixels, return_dict=True).last_hidden_state[:, 1:]
        expected = (self.target_size[0] // 14) * (self.target_size[1] // 14)
        if tokens.shape[1:] != (expected, 384):
            raise ValueError("DINOv2 patch output does not match the configured row-major grid")
        return tokens
