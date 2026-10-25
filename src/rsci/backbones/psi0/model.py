"""RHSCI：方案选择、完整 VLM LoRA，以及进入原 VLM/AE 的四帧身体 Query。"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("STATE", "NO-STATE", "VGQ-AE", "EGQ-AE", "VGQ-VLM", "EGQ-VLM", "VGQ-VLM*", "EGQ-VLM*")


@dataclass(frozen=True)
class RSCISpec:
    """从一个方案名推导结构；不允许用户逐项拼出互相矛盾的布尔开关。"""

    name: str

    def __post_init__(self):
        if self.name not in VARIANTS:
            raise ValueError(f"Unknown RHSCI variant {self.name!r}; choose {VARIANTS}")

    @property
    def query(self):
        return self.name not in ("STATE", "NO-STATE")

    @property
    def encoder_guided(self):
        return self.name.startswith("EGQ-")

    @property
    def at_vlm(self):
        return "-VLM" in self.name

    @property
    def forward_latent(self):
        return self.name.endswith("*")


class LoRALinear(nn.Module):
    """保留原线性层，只学习 BA 增量；完整 checkpoint 保存原权重和适配权重。"""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        # 基座通常为 bf16；增量保留 fp32，避免小学习率更新被 bf16 参数精度吞掉。
        self.lora_A = nn.Parameter(
            base.weight.new_empty(rank, base.in_features, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            base.weight.new_zeros(base.out_features, rank, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        delta = F.linear(F.linear(inputs.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return self.base(inputs) + delta.to(inputs.dtype) * self.scale


def configure_vlm(vlm, mode, rank=8, alpha=16):
    """LoRA 覆盖视觉、语言和跨模态线性投影；不适配从未用于动作的 lm_head。"""
    if mode not in ("frozen", "lora"):
        raise ValueError("RHSCI VLM mode must be frozen or lora")
    vlm.requires_grad_(False)
    paths = []
    if mode == "lora":
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        for name, module in list(vlm.named_modules()):
            if isinstance(module, nn.Linear) and name != "lm_head":
                parent_path, _, leaf = name.rpartition(".")
                parent = vlm.get_submodule(parent_path) if parent_path else vlm
                setattr(parent, leaf, LoRALinear(module, rank, alpha))
                paths.append(name)
        if not paths:
            raise ValueError("VLM contains no supported linear layers for LoRA")
    vlm.eval()  # LoRA 不依赖 dropout；冻结模式也保留 Query 输入梯度。
    return paths


class RSCIQuery(nn.Module):
    """四帧可经 Reader 压缩，再由原 VLM 或 AE 处理一枚 learnable Query。"""

    def __init__(self, spec, vlm_dim, ae_dim, heads, use_reader=True):
        super().__init__()
        self.spec = spec
        self.use_reader = use_reader
        width = vlm_dim if spec.at_vlm else ae_dim
        self.state_query = nn.Parameter(torch.randn(1, 1, width) / width**0.5)
        if use_reader:
            self.reader_queries = nn.Parameter(torch.randn(1, 16, width) / width**0.5)
            self.reader = nn.MultiheadAttention(width, math.gcd(width, heads), batch_first=True)
        self.time_embeddings = nn.Parameter(torch.randn(4, width) / width**0.5)
        self.encoder_projection = nn.Linear(384, width) if spec.encoder_guided else None
        self.ardy_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128))
        self.register_buffer("schema_version", torch.tensor(1 if use_reader else 2))

    def memory(self, frames, masks, valid):
        if len(frames) != 4 or len(masks) != 4 or valid.shape != (frames[0].shape[0], 4):
            raise ValueError(
                "RHSCI requires chronological [t-6,t-4,t-2,t] visual frames and [B,4] validity"
            )
        if not valid[:, -1].all():
            raise ValueError("RHSCI current frame must be valid")
        tokens, token_masks = [], []
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            mask = mask.bool() & valid[:, i : i + 1].bool()
            frame = torch.where(mask[..., None], frame, 0)
            if self.encoder_projection is not None:
                frame = self.encoder_projection(frame)
            if not self.use_reader:
                tokens.append(frame + self.time_embeddings[i])
                token_masks.append(mask)
                continue
            # episode 开头历史可全部无效；开放一个零 key 避免 MHA 的全 mask NaN，
            # 输出随后按原帧有效性屏蔽，不能将填充帧当作真实历史。
            safe_mask = mask.clone()
            safe_mask[~safe_mask.any(-1), 0] = True
            read = self.reader(
                self.reader_queries.expand(frame.shape[0], -1, -1),
                frame,
                frame,
                key_padding_mask=~safe_mask,
                need_weights=False,
            )[0]
            tokens.append(read + self.time_embeddings[i])
            token_masks.append(mask.any(-1, keepdim=True).expand(-1, read.shape[1]))
        memory, mask = torch.cat(tokens, 1), torch.cat(token_masks, 1)
        return torch.where(mask[..., None], memory, 0), mask


class RSCIPolicyMixin:
    """仅 RHSCI 配置进入此路径；原 PSI0 的前向保持不变。"""

    def rsci_history_from_request(self, condition, instruction, model_transform, current_rgb=None):
        """显式采样帧协议，不把不规则的服务请求缓冲假装成 t-6/-4/-2 历史。"""
        if getattr(self, "rsci_spec", None) is None or not self.rsci_spec.forward_latent:
            return {}
        import numpy as np
        from PIL import Image

        frames = np.asarray(condition.get("rsci_rgb_frames"))
        valid = np.asarray(condition.get("rsci_history_valid"))
        offsets = np.asarray(condition.get("rsci_frame_offsets"))
        if (
            frames.dtype != np.uint8
            or frames.ndim != 4
            or frames.shape[0] != 4
            or frames.shape[-1] != 3
        ):
            raise ValueError("starred RHSCI requires condition.rsci_rgb_frames uint8 [4,H,W,3]")
        if valid.dtype != np.bool_ or valid.shape != (4,) or not valid[-1]:
            raise ValueError(
                "starred RHSCI requires condition.rsci_history_valid bool [4] with valid current frame"
            )
        if not np.array_equal(offsets, [-6, -4, -2, 0]):
            raise ValueError(
                "RHSCI frame offsets must be [-6,-4,-2,0] on the training sampling clock"
            )
        if current_rgb is None or not np.array_equal(frames[-1], current_rgb):
            raise ValueError("RHSCI history current frame must equal the request egocentric RGB")
        history = {"history_valid": torch.from_numpy(valid.copy())[None].to(self.device)}
        if self.rsci_spec.encoder_guided:
            history["history_rgb_images"] = (
                torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)[None].to(self.device)
            )
        else:
            resize, crop = model_transform.resize(), model_transform.center_crop()
            history["history_vlm_inputs"] = [
                model_transform.build_qwenvl_inputs(
                    self.vlm_processor, [crop(resize(Image.fromarray(frame)))], instruction
                ).to(self.device)
                for frame in frames[:3]
            ]
        return history

    @classmethod
    def _restore_rsci(cls, state, cfg, device):
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

        # 只读取本地结构/processor，模型及 LoRA/DINO 权重全部来自完整 checkpoint。
        config = AutoConfig.from_pretrained(cfg.model_name_or_path, local_files_only=True)
        for component in (config, config.text_config, config.vision_config):
            component._attn_implementation = "sdpa"
        vlm = Qwen3VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
        embedding_key = "vlm_model.model.language_model.embed_tokens.weight"
        head_key = "vlm_model.lm_head.weight"
        if embedding_key not in state and config.text_config.tie_word_embeddings:
            state[embedding_key] = state[head_key]
        embedding = state[embedding_key]
        if vlm.get_input_embeddings().weight.shape != embedding.shape:
            vlm.resize_token_embeddings(embedding.shape[0], mean_resizing=False)
        model = cls(cfg, vlm, restore_from_state_dict=True)
        # safetensors 会省略共享参数中的一份，恢复时补齐同一个 embedding。
        if config.text_config.tie_word_embeddings:
            state.setdefault(head_key, embedding)
        model.load_state_dict(state, strict=True)
        model.vlm_processor = AutoProcessor.from_pretrained(
            cfg.model_name_or_path, local_files_only=True
        )
        if cfg.noise_scheduler != "flow":
            raise ValueError("RHSCI launcher/checkpoint requires flow matching")
        model.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=cfg.train_diffusion_steps
        )
        model.action_horizon, model.action_dim, model.device = (
            cfg.action_chunk_size,
            cfg.action_dim,
            device,
        )
        return model.to(device).eval()

    def _init_rsci(self, cfg, restore):
        self.rsci_spec = RSCISpec(cfg.rsci_variant) if cfg.rsci_variant else None
        if self.rsci_spec is None:
            return
        self.rsci_lora_paths = configure_vlm(
            self.vlm_model, cfg.rsci_vlm_mode, cfg.rsci_lora_rank, cfg.rsci_lora_alpha
        )
        if cfg.gradient_checkpointing and cfg.rsci_vlm_mode == "lora":
            self.vlm_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if self.rsci_spec.name != "STATE":
            self.action_header.obs_proj._obs_strat = None
            self.action_header.obs_proj._obs_proc.requires_grad_(False)
        if self.rsci_spec.query:
            self.action_header.rsci = RSCIQuery(
                self.rsci_spec,
                cfg.view_feature_dim,
                cfg.hidden_dim,
                cfg.nhead,
                use_reader=cfg.rsci_use_reader,
            )
        if self.rsci_spec.encoder_guided:
            from rsci.backbones.psi0.rgb_encoder import DINOv2RGBAdapter

            # DINOv2 预处理和冻结方式，独立于 VLM 内视觉 LoRA。
            self.rsci_rgb_encoder = DINOv2RGBAdapter(
                cfg.rsci_rgb_encoder_path, cfg.rsci_encoder_input_size, restore=restore
            )

    def _rsci_visual_memory(
        self, current, attention_mask, history_vlm_inputs, history_rgb_images, history_valid
    ):
        spec, module = self.rsci_spec, self.action_header.rsci
        if history_valid is None:
            raise ValueError(
                "RHSCI Query requires history_valid, including during starred inference"
            )
        if spec.encoder_guided:
            if (
                history_rgb_images is None
                or history_rgb_images.ndim != 5
                or history_rgb_images.shape[1:3] != (4, 3)
            ):
                raise ValueError("EGQ requires uint8 history_rgb_images [B,4,3,H,W]")
            batch, _, _, height, width = history_rgb_images.shape
            encoded = self.rsci_rgb_encoder(history_rgb_images.reshape(batch * 4, 3, height, width))
            frames = list(encoded.reshape(batch, 4, encoded.shape[1], -1).unbind(1))
            masks = [
                torch.ones(frame.shape[:2], dtype=torch.bool, device=frame.device)
                for frame in frames
            ]
        else:
            if history_vlm_inputs is None or len(history_vlm_inputs) != 3:
                raise ValueError("VGQ Query requires three past VLM inputs")
            frames = [
                self.vlm_model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[
                    -1
                ]
                for inputs in history_vlm_inputs
            ] + [current]
            masks = [inputs["attention_mask"] for inputs in history_vlm_inputs] + [attention_mask]
            if not spec.at_vlm:
                frames = [
                    self.action_header.obs_proj.project_vl_tokens(frame.unsqueeze(1))
                    for frame in frames
                ]
        return module.memory(frames, masks, history_valid.bool())

    def _rsci_query_latent(self, current, mask, **history):
        module, spec = self.action_header.rsci, self.rsci_spec
        memory, memory_mask = self._rsci_visual_memory(current, mask, **history)
        query = module.state_query.expand(current.shape[0], -1, -1)
        if spec.at_vlm:
            # 真正调用同一个 VLM 的语言 Transformer，而非另挂一个辅助 Transformer。
            # VGQ 的前缀来自四帧 VLM 输出；EGQ 的前缀仅来自四帧 DINOv2。
            embeds = torch.cat((memory, query), 1)
            valid = torch.cat((memory_mask, torch.ones_like(memory_mask[:, :1])), 1)
            positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
            language_model = self.vlm_model.model.language_model
            output = language_model(
                inputs_embeds=embeds,
                attention_mask=valid,
                position_ids=positions[None].expand(3, -1, -1),
                use_cache=False,
                return_dict=True,
            )
            latent = output.last_hidden_state[:, -1:]
        else:
            # Query 作为 AE 的 action stream 输入读取四帧 memory，复用所有原 AE blocks。
            # 不输入真实/带噪动作，防止 ARDY 回归偷看动作；固定零时间条件。
            temb = self.action_header.time_ins_embed(
                torch.zeros(current.shape[0], device=current.device)
            )
            for block in self.action_header.transformer_blocks:
                query, updated_memory = block(
                    action_hidden_states=query,
                    obs_hidden_states=memory,
                    temb=temb,
                    obs_token_mask=memory_mask.float(),
                )
                if updated_memory is not None:
                    memory = updated_memory
            latent = query
        return latent

    def _rsci_inference_features(self, hidden, mask, history):
        """去噪循环外调用一次；无星号时不创建/执行 Query 或历史编码。"""
        if getattr(self, "rsci_spec", None) is None or not self.rsci_spec.forward_latent:
            return hidden, mask
        was_4d = hidden.ndim == 4
        current = hidden[:, 0] if was_4d else hidden
        latent = self._rsci_query_latent(
            current,
            mask,
            history_vlm_inputs=history.get("history_vlm_inputs"),
            history_rgb_images=history.get("history_rgb_images"),
            history_valid=history.get("history_valid"),
        )
        current = torch.cat((current, latent), 1)
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), 1)
        return (current.unsqueeze(1) if was_4d else current), mask

    def _rsci_forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        action_samples,
        states,
        timestep,
        *,
        compute_rsci_aux=False,
        history_vlm_inputs=None,
        history_rgb_images=None,
        history_valid=None,
        vlm_hidden_states=None,
        return_dict=True,
    ):
        from psi.models.psi0 import HumanFoundationModelOutput

        if vlm_hidden_states is not None and self.training:
            raise ValueError("precomputed VLM features are restricted to evaluation")
        current = vlm_hidden_states
        if current is None:
            current = self.vlm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-1]
        prediction = None
        if self.rsci_spec.query and (compute_rsci_aux or self.rsci_spec.forward_latent):
            latent = self._rsci_query_latent(
                current,
                attention_mask,
                history_vlm_inputs=history_vlm_inputs,
                history_rgb_images=history_rgb_images,
                history_valid=history_valid,
            )
            if compute_rsci_aux:
                prediction = self.action_header.rsci.ardy_head(latent[:, 0])
            if self.rsci_spec.forward_latent:
                current = torch.cat((current, latent), 1)
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(attention_mask[:, :1])), 1
                )
        output = self.action_header(
            hidden_states=None,
            timestep=timestep,
            joint_attention_kwargs=dict(
                action_hidden_embeds=action_samples,
                views=current.unsqueeze(1),
                obs=states,
                traj2ds=None,
            ),
            vlm_attn_mask=attention_mask,
            return_dict=return_dict,
        )
        return HumanFoundationModelOutput(action=output.action, ardy_prediction=prediction)


def masked_auxiliary_loss(prediction, target, valid, *, smooth_l1=False):
    """无标签样本不参与归一化；全无效仍保留到预测头的零梯度路径。"""
    if target.shape != prediction.shape or valid.shape != prediction.shape[:1]:
        raise ValueError("auxiliary target/mask shapes do not match prediction")
    valid = valid.bool()
    if not torch.isfinite(target[valid]).all():
        raise ValueError("valid auxiliary labels must be finite")
    target = torch.where(valid[:, None], target, torch.zeros_like(target))
    fn = F.smooth_l1_loss if smooth_l1 else F.mse_loss
    error = fn(prediction.float(), target.float(), reduction="none").mean(-1)
    count = valid.sum().to(error.dtype)
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # DDP 将梯度平均，先按全局有效标签数归一化，再抵消梯度平均的 world_size。
        torch.distributed.all_reduce(count)
        world_size = torch.distributed.get_world_size()
    return (error * valid).sum() * world_size / count.clamp_min(1)
