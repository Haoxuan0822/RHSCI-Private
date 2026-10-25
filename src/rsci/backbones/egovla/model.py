"""四帧只通过辅助 Query 更新原生 EgoVLA 动作专家，不注入动作分支。"""

import torch
from torch import nn


class Reader(nn.Module):
    """沿用 π0.5 对照：每帧16个 Reader、时间 embedding，再投影回 VLM 宽度。"""

    def __init__(self, width):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, 16, width) / width**0.5)
        self.attn = nn.MultiheadAttention(width, 8, batch_first=True)
        self.time = nn.Parameter(torch.randn(4, width) / width**0.5)
        self.to_vlm = nn.Linear(width, width)

    def forward(self, frames, valid):
        memories, masks = [], []
        for index, (features, mask, _) in enumerate(frames):
            if not mask.any(-1).all():
                raise ValueError("Every frame needs at least one unmasked VLM context token")
            read = self.attn(
                self.queries.expand(features.shape[0], -1, -1),
                features,
                features,
                key_padding_mask=~mask,
                need_weights=False,
            )[0]
            memories.append(self.to_vlm(read + self.time[index]))
            masks.append(valid[:, index : index + 1].expand(-1, 16))
        mask = torch.cat(masks, 1)
        return torch.where(mask[..., None], torch.cat(memories, 1), 0), mask


def decode_latents(decoder, latents, memory, mask):
    """原生四层 forward 的无 state 路径；Query 也复用同一 first_norm/layers。

    原函数即使 use_proprio=False 仍读取 proprio.shape 来 reshape。这里直接采用
    显式 B×Q×D，彻底消除 state/伪 state 输入；每层仍重新拼入同一 memory。
    """
    if decoder.use_proprio or decoder.sep_proprio:
        raise ValueError("VGQ-AE forbids proprioception in the decoder")
    if latents.ndim != 3 or memory.ndim != 3 or mask.shape != memory.shape[:2]:
        raise ValueError("Expected batched latents and boolean context mask")
    length = latents.shape[1]
    padding = torch.cat(
        (~mask.bool(), torch.zeros(latents.shape[:2], dtype=torch.bool, device=latents.device)),
        dim=1,
    )
    hidden, context = decoder.first_norm(latents), decoder.first_norm(memory)
    for layer in decoder.layers:
        hidden = layer(torch.cat((context, hidden), dim=1), src_key_padding_mask=padding)[
            :, -length:
        ]
    return hidden


def masked_ardy_loss(prediction, target, valid):
    """全局有效帧归一化；无效目标不能用 NaN×0 污染损失。"""
    if target.shape != prediction.shape or valid.shape != prediction.shape[:1]:
        raise ValueError("ARDY target/mask shape mismatch")
    valid = valid.bool()
    safe_target = torch.where(valid[:, None], target, torch.zeros_like(target))
    error = (prediction.float() - safe_target.float()).square().mean(-1)
    count = valid.float().sum().detach()
    world = 1
    if torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        torch.distributed.all_reduce(count)
    return torch.where(valid, error, 0).sum() * world / count.clamp_min(1)


class EgoVLARSCI(nn.Module):
    def __init__(self, backbone, vlm_training="full", variant="VGQ-AE"):
        super().__init__()
        if vlm_training not in ("full", "frozen"):
            raise ValueError("VLM training must be full or frozen")
        self.vlm_training = vlm_training
        if variant not in ("VGQ-AE", "STATE"):
            raise ValueError("Unknown model variant")
        self.variant = variant
        self.backbone = backbone
        config = backbone.config
        if (
            config.traj_decoder_type != "transformer_action_vector"
            or config.action_output_dim != 36
        ):
            raise ValueError("Requires the official SIMPLE 36D TransformerActionVector")
        config.use_proprio = variant == "STATE"
        decoder = backbone.get_traj_decoder().decoder
        decoder.use_proprio = config.use_proprio
        decoder.sep_proprio = False
        # Train the selected VLM components and keep the remaining base weights frozen.
        backbone.requires_grad_(False)
        if vlm_training == "full":
            for module in (
                backbone.get_llm(),
                backbone.get_vision_tower(),
                backbone.get_mm_projector(),
            ):
                module.requires_grad_(True)
        decoder.requires_grad_(True)
        decoder.proprio_projection.requires_grad_(variant == "STATE")
        for key in ("tune_language_model", "tune_vision_tower", "tune_mm_projector"):
            setattr(config, key, vlm_training == "full")
        width = config.hidden_size
        if variant == "VGQ-AE":
            self.reader = Reader(width)
            self.query = nn.Parameter(torch.randn(1, 1, width) / width**0.5)
            self.ardy_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128))
        self.rsci_config = dict(
            schema="egovla-vgq-ae-v1",
            variant="VGQ-AE",
            vlm=vlm_training,
            history=[-6, -4, -2, 0],
            horizon=30,
            action_dim=36,
            ardy_dim=128,
            ardy_offset=0,
            ardy_weight=0.1,
            reader="16-per-frame",
            query_to_action=False,
            inference_query=False,
            normalization="raw-action-vector",
            decoder="official-transformer-action-vector-4-layer",
        )
        if variant == "STATE":
            self.rsci_config.update(
                variant="STATE", history=[0], ardy_weight=0.0, reader=None, state_dim=32
            )
        # VLM保留官方加载的BF16（全量训练也不强制扩成FP32）；AE/辅助参数沿用FP32。
        for parameter in self.decoder.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.train()

    @property
    def decoder(self):
        return self.backbone.get_traj_decoder().decoder

    def train(self, mode=True):
        super().train(mode)
        if self.vlm_training == "frozen":
            for module in (
                self.backbone.get_llm(),
                self.backbone.get_vision_tower(),
                self.backbone.get_mm_projector(),
            ):
                module.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """使用原生逐层非重入重算，保留四帧及动作/ARDY的完整梯度。"""
        if self.vlm_training != "full":
            return
        options = {"use_reentrant": False, **(gradient_checkpointing_kwargs or {})}
        if options["use_reentrant"]:
            raise ValueError("Shared four-frame VLM requires non-reentrant checkpointing")
        self.backbone.get_llm().model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=options
        )
        vision = self.backbone.get_vision_tower()
        vision = getattr(vision, "vision_tower", vision)
        if hasattr(vision, "gradient_checkpointing_enable"):
            vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs=options)

    def encode(self, image, input_ids, labels, attention_mask):
        # 全量模式下动作与ARDY两条loss都能回传；冻结消融及推理仍不记录VLM计算图。
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.vlm_training == "full"):
            return self._encode(image, input_ids, labels, attention_mask)

    def _encode(self, image, input_ids, labels, attention_mask):
        # 直接调用官方图像展开、padding/位置与图文拼接，不换成 π0.5 的双向注意力。
        _, positions, mask, _, embeddings, expanded_labels = (
            self.backbone.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, labels, image
            )
        )
        if mask is None or expanded_labels is None:
            raise ValueError("Explicit labels and attention mask are required")
        # next_token_loss_coeff=0：跳过无用的词表 head/CE，只取同一语言主干的末层表征。
        output = self.backbone.get_llm().model(
            inputs_embeds=embeddings,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        config = self.backbone.config
        action_mask = (
            (expanded_labels >= config.input_placeholder_end_token_idx)
            & (expanded_labels <= config.input_placeholder_start_token_idx)
            & mask.bool()
        )
        if not (action_mask.sum(-1) == 30).all():
            raise ValueError(
                "Official placeholder tokens must select exactly 30 actions per sample"
            )
        return hidden, mask.bool() & ~action_mask, action_mask

    def action_prediction(self, current, states=None):
        features, mask, action_mask = current
        latents = features[action_mask].reshape(features.shape[0], 30, features.shape[-1])
        if self.variant == "STATE":
            if (
                states is None
                or states.shape != (features.shape[0], 32)
                or not torch.isfinite(states).all()
            ):
                raise ValueError("Official STATE requires finite current 32D state")
            return (
                self.decoder(latents.flatten(0, 1), dict(proprio=states), features, mask)["pred"]
                .reshape(-1, 30, 36)
                .float()
            )
        hidden = decode_latents(self.decoder, latents, features, mask)
        return self.decoder.output_projection(hidden).float()

    def forward(
        self,
        images,
        input_ids,
        labels,
        attention_mask,
        actions,
        action_mask,
        history_valid=None,
        ardy_target=None,
        ardy_valid=None,
        states=None,
    ):
        if (
            images.shape[1] != (1 if self.variant == "STATE" else 4)
            or actions.shape[1:] != (30, 36)
            or action_mask.shape != actions.shape
        ):
            raise ValueError("Expected chronological four frames and 30×36 actions/mask")
        if self.variant == "VGQ-AE" and (
            history_valid.shape != images.shape[:2] or not history_valid[:, -1].all()
        ):
            raise ValueError("Current image must be valid")
        if self.variant == "VGQ-AE" and states is not None:
            raise ValueError("VGQ-AE must not receive state")
        current = self.encode(images[:, -1], input_ids, labels, attention_mask)
        prediction = self.action_prediction(current, states)
        # 与官方 raw_action_vector 完全相同：先每个 timestep 按有效维归一，再求平均。
        difference = (prediction - actions.float()) * action_mask
        action_loss = (difference.square().sum(-1) / action_mask.sum(-1).clamp_min(1)).mean()
        if self.variant == "STATE":
            return dict(loss=action_loss, action_loss=action_loss.detach())
        # 历史分批编码控制显存；它们不进入动作路径，当前帧只编码一次。
        frames = [
            self.encode(images[:, index], input_ids, labels, attention_mask) for index in range(3)
        ]
        frames.append(current)
        memory, mask = self.reader(frames, history_valid.bool())
        latent = decode_latents(
            self.decoder, self.query.expand(images.shape[0], -1, -1), memory, mask
        )[:, 0]
        ardy_loss = masked_ardy_loss(self.ardy_head(latent.float()), ardy_target, ardy_valid)
        return dict(
            loss=action_loss + 0.1 * ardy_loss,
            action_loss=action_loss.detach(),
            ardy_loss=ardy_loss.detach(),
        )

    @torch.no_grad()
    def sample_actions(self, images, input_ids, labels, attention_mask, states=None):
        """普通 VGQ-AE：单当前帧、无 Query/Reader/历史，无扩散/RTC 等额外采样。"""
        if self.training or images.shape[1] != 1:
            raise ValueError("Inference requires eval mode and exactly one current frame")
        return self.action_prediction(
            self.encode(images[:, 0], input_ids, labels, attention_mask), states
        )
