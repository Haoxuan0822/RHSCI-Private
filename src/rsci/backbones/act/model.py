"""原生 ACT + 训练期 Encoder-Guided Query；没有 Reader/VLM。"""

from dataclasses import asdict

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from act.models.act import ACTConfig, ACTPolicy

VARIANTS = ("STATE", "NO-STATE", "EGQ-AE")


def official_config(variant="EGQ-AE", *, pretrained=True):
    if variant not in VARIANTS:
        raise ValueError(variant)
    # 参数取实际 SIMPLE 脚本；未覆盖的结构直接沿用原生 ACTConfig。
    return ACTConfig(
        chunk_size=16,
        n_action_steps=16,
        state_dim=36,
        action_dim=36,
        use_robot_state=variant == "STATE",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" if pretrained else None,
    )


def masked_ardy_loss(prediction, target, valid):
    """按全rank有效样本求mean；空rank仍保留完整图，DDP平均后梯度一致。"""
    valid = valid.bool()
    # 无效标签允许NaN，但不能用0*NaN屏蔽，也不能让无效预测逃过数值门禁。
    target = torch.where(valid[:, None], target.float(), torch.zeros_like(target.float()))
    errors = (prediction.float() - target).square().mean(-1)
    denominator = valid.sum().to(errors.dtype)
    world = 1
    if dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(denominator)
    return (errors * valid).sum() * world / denominator.clamp_min(1)


class RSCIACT(nn.Module):
    def __init__(self, variant="EGQ-AE", config=None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        config = config or official_config(variant)
        if config.use_robot_state != (variant == "STATE") or config.use_env_state:
            raise ValueError("State ablation contract mismatch")
        self.variant = variant
        self.policy = ACTPolicy(config)
        self.rsci_config = dict(
            schema="act-rsci-v1",
            variant=variant,
            act=asdict(config),
            history_offsets=[-6, -4, -2, 0],
            ardy_dim=128,
            ardy_weight=0.1,
            reader=False,
            inference_query=False,
        )
        if variant == "EGQ-AE":
            self.time_embedding = nn.Embedding(4, config.dim_model)
            self.state_query = nn.Embedding(1, config.dim_model)
            self.ardy_head = nn.Linear(config.dim_model, 128)

    def _native_batch(self, batch):
        native = {
            key: value
            for key, value in batch.items()
            if key.startswith("observation.") or key in ("action", "action_is_pad")
        }
        if self.variant != "STATE":
            # 上游无state分支仅在分配mask/z时仍读取此键的device。
            # 零元素载体没有数值、投影或token；真实state即使被意外传来亦立即丢弃。
            native["observation.state"] = batch["observation.images"].new_empty(0)
        elif native["observation.state"].ndim == 3:
            native["observation.state"] = native["observation.state"].squeeze(1)
        return native

    def auxiliary_prediction(self, history_images, current_map):
        model = self.policy.model
        if history_images.ndim != 5 or history_images.shape[1] != 4:
            raise ValueError("Expected B×4×3×H×W history")
        batch = history_images.shape[0]
        # 当前帧ResNet已经在原生动作前向中计算，历史仅额外计算前三帧。
        maps = model.backbone(history_images[:, :3].reshape(-1, *history_images.shape[2:]))[
            "feature_map"
        ]
        maps = torch.cat((maps.reshape(batch, 3, *maps.shape[1:]), current_map[:, None]), 1)
        maps = maps.reshape(-1, *maps.shape[2:])
        spatial = model.encoder_cam_feat_pos_embed(maps).to(maps.dtype).flatten(2).permute(2, 0, 1)
        projected = model.encoder_img_feat_input_proj(maps)
        tokens = projected.flatten(2).transpose(1, 2).reshape(batch, 4, -1, projected.shape[1])
        positions = (
            spatial[:, :1].transpose(0, 1)[:, None] + self.time_embedding.weight[None, :, None]
        )
        positions = (
            positions.expand(batch, -1, -1, -1)
            .reshape(batch, -1, projected.shape[1])
            .transpose(0, 1)
        )
        tokens = tokens.reshape(batch, -1, projected.shape[1]).transpose(0, 1)
        # 辅助memory只有视觉：不能复用混有动作posterior z的动作encoder输出。
        memory = model.encoder(tokens, pos_embed=positions)
        query = self.state_query.weight[:, None].expand(-1, batch, -1)
        latent = model.decoder(
            torch.zeros_like(query), memory, decoder_pos_embed=query, encoder_pos_embed=positions
        )[0]
        return self.ardy_head(latent)

    def forward(self, **batch):
        native = self._native_batch(batch)
        maps = []
        needs_aux = self.variant == "EGQ-AE" and "ardy_target" in batch
        hook = (
            self.policy.model.backbone.register_forward_hook(
                lambda module, args, output: maps.append(output["feature_map"])
            )
            if needs_aux
            else None
        )
        try:
            actions, (mu, logvar) = self.policy.model(native)
        finally:
            if hook is not None:
                hook.remove()
        # 逐项保留原生ACT损失和padding分母；仅取消原生.item()同步。
        l1 = (
            F.l1_loss(batch["action"], actions, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
        ardy = l1.new_zeros(())
        if needs_aux:
            if len(maps) != 1:
                raise ValueError("RHSCI supports exactly one current camera")
            prediction = self.auxiliary_prediction(batch["history_images"], maps[0])
            ardy = masked_ardy_loss(prediction, batch["ardy_target"], batch["ardy_valid"])
        return dict(
            loss=l1 + self.policy.config.kl_weight * kl + 0.1 * ardy,
            l1_loss=l1.detach(),
            kl_loss=kl.detach(),
            ardy_loss=ardy.detach(),
        )

    @torch.no_grad()
    def predict_action(self, batch):
        # 原生predict_action会设置eval；不传action，保证推理z=0且不读取history/query。
        self.eval()
        native = self._native_batch(batch)
        native.pop("action", None)
        native.pop("action_is_pad", None)
        return self.policy.predict_action(native)

    @torch.no_grad()
    def select_action(self, batch):
        self.eval()
        native = self._native_batch(batch)
        native.pop("action", None)
        return self.policy.select_action(native)

    def reset(self):
        self.policy.reset()
