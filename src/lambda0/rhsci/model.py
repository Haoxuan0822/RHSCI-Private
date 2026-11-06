"""保留官方 flow/RTC/动作头；Query 仅走共享 AE 辅助支路。"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from lambda0.models.modeling_lambda0 import Lambda0
from rhsci.runtime.losses import masked_ardy_loss as masked_ardy_loss


class HistoryReader(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(16, width) * 0.02)
        self.time = nn.Parameter(torch.randn(4, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, 8, batch_first=True)

    def forward(self, frames, masks, valid):
        result = []
        for index, (features, mask) in enumerate(zip(frames, masks, strict=True)):
            if not mask.bool().any(-1).all():
                raise ValueError("Every frame needs a valid VLM context token")
            queries = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
            tokens = self.attention(
                queries, features, features, key_padding_mask=~mask.bool(), need_weights=False
            )[0]
            result.append(tokens + self.time[index])
        return torch.cat(result, dim=1), valid.repeat_interleave(16, dim=1)


class Lambda0RHSCI(Lambda0):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        wants_info = kwargs.pop("output_loading_info", False)
        model, info = super().from_pretrained(*args, output_loading_info=True, **kwargs)
        if any(
            info.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise RuntimeError(f"RHSCI checkpoint failed strict loading: {info}")
        return (model, info) if wants_info else model

    def __init__(self, config, vlm=None):
        super().__init__(config, vlm=vlm)
        if getattr(config, "rsci_variant", "STATE") in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-Mo"):
            self.install_auxiliary()

    def install_auxiliary(self):
        if hasattr(self, "rsci_reader"):
            return
        width, ae_width = self.config.vlm_hidden_size, self.config.expert_hidden_size
        self.rsci_reader = HistoryReader(width)
        self.rsci_query = nn.Parameter(torch.randn(1, 1, ae_width) * 0.02)
        self.rsci_ardy = nn.Sequential(nn.LayerNorm(ae_width), nn.Linear(ae_width, 128))
        if self.config.rsci_variant == "VGQ-AE-Mo":
            from rhsci.data.ardy_motion import validate_motion_config

            validate_motion_config(
                self.config.rsci_ardy_delta_frames, self.config.rsci_motion_weight
            )
            self.rsci_motion = nn.Sequential(nn.LayerNorm(ae_width), nn.Linear(ae_width, 128))
        # 新模块继承官方权重的设备/精度，不改变已有 VLM/AE 参数。
        reference = next(self.action_expert.parameters())
        self.rsci_reader.to(device=reference.device, dtype=reference.dtype)
        self.rsci_ardy.to(device=reference.device, dtype=reference.dtype)
        if hasattr(self, "rsci_motion"):
            self.rsci_motion.to(device=reference.device, dtype=reference.dtype)
        self.rsci_query.data = self.rsci_query.data.to(
            device=reference.device, dtype=reference.dtype
        )

    def forward_vlm(self, **kwargs):
        encoded = super().forward_vlm(**kwargs)
        if getattr(self, "_rsci_capture", False):
            self._rsci_current = encoded
        return encoded

    def auxiliary(self, current, history, history_valid, embodiment_id):
        if history_valid.shape != (embodiment_id.shape[0], 4) or not history_valid[:, -1].all():
            raise ValueError("Expected four chronological masks and a valid current frame")
        expert = self.action_expert
        encoded = [super(Lambda0RHSCI, self).forward_vlm(**frame) for frame in history] + [current]
        query = self.rsci_query.expand(embodiment_id.shape[0], -1, -1)
        query = query + expert.embodiment_bias(embodiment_id).unsqueeze(1)
        query = query + expert._type_embedding(3, query)
        for layer, block in enumerate(expert.blocks):
            memory, valid = self.rsci_reader(
                [frame["vlm_hidden_states"][layer] for frame in encoded],
                [frame["vlm_attention_mask"] for frame in encoded],
                history_valid,
            )
            memory = expert.context_projection(expert.context_norms[layer](memory))
            memory = memory * expert.context_scales[layer] + expert._type_embedding(0, memory)
            tokens = torch.cat((memory.to(query.dtype), query), dim=1)
            # context 不读 Query；Query 可读四帧 Reader 和自身。动作分支不读这条 memory。
            allowed = torch.ones(tokens.shape[0], 65, 65, dtype=torch.bool, device=tokens.device)
            allowed[:, :64, 64:] = False
            allowed[:, :, :64] &= valid[:, None, :]
            if expert.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, allowed, use_reentrant=False)
            else:
                tokens = block(tokens, allowed)
            query = tokens[:, -1:]
        latent = expert.output_norm(query)[:, 0]
        prediction = self.rsci_ardy(latent)
        if self.config.rsci_variant == "VGQ-AE-Mo":
            return prediction, self.rsci_motion(latent)
        return prediction

    def forward(
        self,
        *args,
        history=None,
        history_valid=None,
        ardy_target=None,
        ardy_valid=None,
        motion_target=None,
        motion_valid=None,
        auxiliary_normalization=None,
        **kwargs,
    ):
        variant = getattr(self.config, "rsci_variant", "STATE")
        self._rsci_capture = variant in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-Mo")
        try:
            losses = super().forward(*args, **kwargs)
            if variant in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-Mo"):
                if history is None or len(history) != 3:
                    raise ValueError("VGQ-AE requires three historical frames plus current")
                prediction = self.auxiliary(
                    self._rsci_current, history, history_valid, kwargs["embodiment_id"]
                )
                if variant == "VGQ-AE-Mo":
                    prediction, motion_prediction = prediction
                normalization = auxiliary_normalization or {}
                auxiliary_loss = masked_ardy_loss(
                    prediction, ardy_target, ardy_valid, global_reduction=self.training,
                    normalization_denominator=normalization.get("ardy"),
                )
                # 官方训练器只接受其原生 metric key；总 loss 包含辅助项，独立值留给验收/调试。
                self.last_ardy_loss = auxiliary_loss.detach()
                losses["loss"] = losses["loss"] + 0.1 * auxiliary_loss
                if variant == "VGQ-AE-Mo":
                    motion_loss = masked_ardy_loss(
                        motion_prediction,
                        motion_target,
                        motion_valid,
                        global_reduction=self.training,
                        normalization_denominator=normalization.get("motion"),
                    )
                    self.last_motion_loss = motion_loss.detach()
                    losses["loss"] = losses["loss"] + self.config.rsci_motion_weight * motion_loss
            if self.training:
                from .runtime import check_finite

                check_finite(losses.values(), "loss")
            return losses
        finally:
            self._rsci_capture = False
            self._rsci_current = None
