"""复用 SIMPLE 原生 Gemma/adaRMSNorm；冻结 VLM KV 与 AE 反向图分离。"""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma import modeling_gemma

from .augmentation import AUGMENTATION_VERSION, augment_training_images


def time_embedding(time, width):
    fraction = torch.linspace(0, 1, width // 2, device=time.device, dtype=torch.float64)
    angle = time.double()[:, None] * (2 * math.pi / (4e-3 * (4 / 4e-3) ** fraction))
    return torch.cat((angle.sin(), angle.cos()), -1).float()


class Reader(nn.Module):
    """与现有对照保持每帧16个可学习 Reader；四帧按时间顺序拼接。"""

    def __init__(self, vlm_width, ae_width):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, 16, ae_width) / ae_width**0.5)
        self.attn = nn.MultiheadAttention(
            ae_width, 8, kdim=vlm_width, vdim=vlm_width, batch_first=True
        )
        self.time = nn.Parameter(torch.randn(4, ae_width) / ae_width**0.5)
        self.to_vlm = nn.Linear(ae_width, vlm_width)

    def forward(self, frames, valid):
        values, masks = [], []
        for index, (features, mask) in enumerate(frames):
            # 起始填充帧不计入 memory，安全占位避免全屏蔽 softmax 产生 NaN。
            read = self.attn(
                self.queries.expand(features.shape[0], -1, -1),
                features,
                features,
                key_padding_mask=~mask,
                need_weights=False,
            )[0]
            values.append(self.to_vlm(read + self.time[index]))
            masks.append(valid[:, index : index + 1].expand(-1, 16))
        mask = torch.cat(masks, 1)
        memory = torch.cat(values, 1)
        return torch.where(mask[..., None], memory, 0), mask


class Pi05RSCI(nn.Module):
    def __init__(
        self,
        variant="VGQ-AE",
        backbone=None,
        vlm_width=2048,
        ae_width=1024,
        compile_inference=True,
        execution="reference",
        state_measurement_noise_std_rad=0.0,
    ):
        super().__init__()
        from rsci.data.state_noise import validate_scale

        self.state_measurement_noise_std_rad = validate_scale(
            state_measurement_noise_std_rad, variant
        )
        if execution not in ("reference", "history", "query", "both"):
            raise ValueError(f"Unknown execution profile: {execution}")
        self.execution = execution
        if variant not in ("STATE", "NO-STATE", "VGQ-AE"):
            raise ValueError(variant)
        if backbone is None:
            from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

            def specs(width, mlp):
                return SimpleNamespace(
                    width=width, mlp_dim=mlp, depth=18, num_heads=8, num_kv_heads=1, head_dim=256
                )

            backbone = PaliGemmaWithExpertModel(
                specs(2048, 16384),
                specs(1024, 4096),
                use_adarms=[False, True],
                precision="bfloat16",
            )
        self.paligemma_with_expert = backbone
        self.action_in_proj = nn.Linear(36, ae_width)
        self.action_out_proj = nn.Linear(ae_width, 36)
        self.time_mlp_in = nn.Linear(ae_width, ae_width)
        self.time_mlp_out = nn.Linear(ae_width, ae_width)
        self.variant = variant
        self.rsci_config = dict(
            variant=variant,
            horizon=30,
            action_dim=36,
            vlm="frozen",
            history=[-6, -4, -2, 0],
            ardy_weight=0.1,
            reader="16-per-frame-KV-projection-v1",
            normalization="simple-quantile-zero-span-clipped-v2",
            training_augmentation=AUGMENTATION_VERSION,
            inference_compile="max-autotune" if compile_inference else "disabled",
            execution=execution,
            state_measurement_noise_std_rad=self.state_measurement_noise_std_rad,
        )
        if variant == "VGQ-AE":
            self.reader = Reader(vlm_width, ae_width)
            self.query = nn.Parameter(torch.randn(1, 1, ae_width) / ae_width**0.5)
            self.ardy_head = nn.Sequential(nn.LayerNorm(ae_width), nn.Linear(ae_width, 128))
        backbone.paligemma.requires_grad_(False).eval()
        # 两个 lm_head 都不参与动作，移除可避免无用词表权重及共享张量保存问题。
        backbone.paligemma.lm_head = nn.Identity()
        backbone.gemma_expert.lm_head = nn.Identity()
        for module in (backbone.paligemma.language_model, backbone.gemma_expert.model):
            module.config._attn_implementation = "sdpa"
        for param in self.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        # 恢复原 SIMPLE 的默认采样编译；训练 forward 不编译，首次采样才生成内核。
        if compile_inference:
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

    def train(self, mode=True):
        super().train(mode)
        self.paligemma_with_expert.paligemma.eval()
        return self

    @torch.no_grad()
    def encode(self, image, tokens, token_mask):
        native = self.paligemma_with_expert
        vision = native.embed_image(image)
        language = native.embed_language_tokens(tokens) * math.sqrt(
            native.paligemma.config.text_config.hidden_size
        )
        hidden = torch.cat((vision, language), 1)
        mask = torch.cat(
            (torch.ones(vision.shape[:2], dtype=torch.bool, device=image.device), token_mask), 1
        )
        positions = mask.long().cumsum(-1) - 1
        # 原生 prefix 是双向注意力；显式4D遮罩，不能误用默认因果语言注意力。
        attention = torch.zeros(mask.shape[0], 1, mask.shape[1], mask.shape[1], device=image.device)
        attention.masked_fill_(~mask[:, None, None, :], torch.finfo(torch.float32).min)
        out = native.paligemma.language_model(
            inputs_embeds=hidden, attention_mask=attention, position_ids=positions, use_cache=True
        )
        return out.last_hidden_state, mask, out.past_key_values

    def memory_cache(self, memory, mask):
        """将 VLM 引导的 Reader 特征转为原生 AE 可读 KV；不运行第二遍完整 VLM。"""
        language = self.paligemma_with_expert.paligemma.language_model
        positions = mask.long().cumsum(-1).clamp_min(1) - 1
        cos, sin = language.rotary_emb(memory, positions)
        cache = []
        for layer in language.layers:
            hidden, _ = layer.input_layernorm(memory)
            size = (*hidden.shape[:-1], -1, layer.self_attn.head_dim)
            key = layer.self_attn.k_proj(hidden).view(size).transpose(1, 2)
            value = layer.self_attn.v_proj(hidden).view(size).transpose(1, 2)
            key, _ = modeling_gemma.apply_rotary_pos_emb(key, key, cos, sin)
            cache.append((key, value))
        return cache

    def expert(self, hidden, time, mask, cache, *, recompute=True):
        """真正逐层 checkpoint；已有 KV 只读，反向重算不会把动作追加到前缀。"""
        model = self.paligemma_with_expert.gemma_expert.model
        cond = F.silu(
            self.time_mlp_out(F.silu(self.time_mlp_in(time_embedding(time, hidden.shape[-1]))))
        )
        positions = (
            mask.long().sum(-1, keepdim=True)
            + torch.arange(hidden.shape[1], device=hidden.device)[None]
        )
        cos, sin = model.rotary_emb(hidden, positions)
        full_mask = torch.cat(
            (mask, torch.ones(hidden.shape[:2], device=mask.device, dtype=torch.bool)), 1
        )
        attention = torch.zeros(
            hidden.shape[0], 1, hidden.shape[1], full_mask.shape[1], device=hidden.device
        )
        attention.masked_fill_(~full_mask[:, None, None, :], torch.finfo(torch.float32).min)
        for index, layer in enumerate(model.layers):
            key, value = cache[index]

            def run(h, conditioning, k, v, layer=layer, index=index):
                # 每层只需其自身前缀KV，显式传入保证 Reader 的梯度参与重算。
                keys = [None] * len(model.layers)
                keys[index] = (k, v)
                return layer(
                    h,
                    attention_mask=attention,
                    position_ids=positions,
                    position_embeddings=(cos, sin),
                    past_key_value=keys,
                    use_cache=False,
                    adarms_cond=conditioning,
                )[0]

            hidden = (
                checkpoint(run, hidden, cond, key, value, use_reentrant=False)
                if self.training and recompute
                else run(hidden, cond, key, value)
            )
        return model.norm(hidden, cond)[0]

    def encode_history(self, images, tokens, token_mask):
        """只合并batch维；每个样本的三张历史图仍有独立文本、位置和注意力。"""
        if self.execution not in ("history", "both"):
            return [self.encode(images[:, index], tokens, token_mask)[:2] for index in range(3)]
        batch, frames = images.shape[:2]
        if frames != 3:
            raise ValueError("History encoding requires exactly three past frames")
        # 排列为b0t0,b0t1,b0t2,b1t0...，文本必须按同一顺序重复。
        features, mask, _ = self.encode(
            images.flatten(0, 1),
            tokens.repeat_interleave(frames, dim=0),
            token_mask.repeat_interleave(frames, dim=0),
        )
        features = features.reshape(batch, frames, *features.shape[1:])
        mask = mask.reshape(batch, frames, -1)
        return [(features[:, index], mask[:, index]) for index in range(frames)]

    def forward(
        self,
        images,
        tokens,
        token_mask,
        actions,
        history_valid=None,
        ardy_target=None,
        ardy_valid=None,
        state_noise_std_rad=None,
    ):
        if actions.shape[1:] != (30, 36) or images.shape[1] != (
            4 if self.variant == "VGQ-AE" else 1
        ):
            raise ValueError("Expected 30×36 actions and recipe-specific image count")
        if self.training:
            # 当前动作分支和历史辅助分支必须使用同一次增强后的当前图像。
            images = augment_training_images(images)
        if self.training and self.state_measurement_noise_std_rad > 0:
            if state_noise_std_rad is None or not torch.all(
                state_noise_std_rad == self.state_measurement_noise_std_rad
            ):
                raise ValueError("STATE training requires matching raw-state noise from Processor")
        current = self.encode(images[:, -1], tokens, token_mask)
        time = (
            torch.distributions.Beta(1.5, 1.0).sample((actions.shape[0],)).to(actions.device)
            * 0.999
            + 0.001
        )
        noise = torch.randn_like(actions)
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        predicted = self.action_out_proj(
            self.expert(self.action_in_proj(noisy), time, current[1], current[2]).float()
        )
        action_loss = (predicted - (noise - actions)).square().mean()
        loss = action_loss
        output = dict(action_loss=action_loss.detach())
        if self.variant == "VGQ-AE":
            if images.shape[1] != 4 or history_valid is None or not history_valid[:, -1].all():
                raise ValueError("VGQ-AE requires chronological four-frame input")
            frames = self.encode_history(images[:, :3], tokens, token_mask)
            frames.append(current[:2])
            memory, mask = self.reader(frames, history_valid)
            latent = self.expert(
                self.query.expand(actions.shape[0], -1, -1),
                torch.zeros_like(time),
                mask,
                self.memory_cache(memory, mask),
                recompute=self.execution not in ("query", "both"),
            )[:, 0]
            prediction = self.ardy_head(latent.float())
            errors = (prediction - ardy_target).square().mean(-1)
            # DDP平均梯度：按全局有效标签数量归一化，而非各卡有效数量分别平均。
            count = ardy_valid.float().sum().detach()
            world = 1
            if torch.distributed.is_initialized():
                world = torch.distributed.get_world_size()
                torch.distributed.all_reduce(count)
            auxiliary = (errors * ardy_valid).sum() * world / count.clamp_min(1)
            loss = loss + 0.1 * auxiliary
            output["ardy_loss"] = auxiliary.detach()
        return dict(loss=loss, **output)

    @torch.no_grad()
    def sample_actions(self, images, tokens, token_mask, num_steps=10, noise=None):
        """普通 VGQ-AE 推理只有当前单帧；不运行 Reader、Query 或 ARDY。"""
        if images.shape[1] != 1 or num_steps < 1:
            raise ValueError("Inference requires exactly one current frame and positive num_steps")
        _, mask, cache = self.encode(images[:, -1], tokens, token_mask)
        actions = (
            noise
            if noise is not None
            else torch.randn(images.shape[0], 30, 36, device=images.device)
        )
        for step in range(num_steps):
            time = torch.full((images.shape[0],), 1 - step / num_steps, device=images.device)
            velocity = self.action_out_proj(
                self.expert(self.action_in_proj(actions), time, mask, cache).float()
            )
            actions = actions - velocity / num_steps
        return actions

    def load_initial(self, root):
        from pathlib import Path

        from safetensors.torch import load_file

        state = load_file(str(Path(root) / "model.safetensors"))
        expected = self.state_dict()
        # 官方 safetensors 对 tied embedding 只保存 lm_head 那份，先恢复别名再移除无用头。
        for name, value in expected.items():
            if (
                name.startswith("paligemma_with_expert.paligemma.")
                and name.endswith("embed_tokens.weight")
                and name not in state
            ):
                tied = state.get("paligemma_with_expert.paligemma.lm_head.weight")
                if tied is None or tied.shape != value.shape:
                    raise ValueError("Missing official tied VLM embedding")
                state[name] = tied
        # 只允许新增RHSCI模块、无用词表头及官方32→SIMPLE36动作接口扩维。
        for name in list(state):
            if ".lm_head." in name:
                del state[name]
            elif name in expected and state[name].shape != expected[name].shape:
                old, new = state[name], expected[name].clone()
                if name == "action_in_proj.weight" and old.shape == (new.shape[0], 32):
                    # 与 SIMPLE 原 train_pytorch.py 一致：复制最后4列扩维，不重置AE。
                    new = torch.cat((old, old[:, -4:]), dim=1)
                elif (
                    name in ("action_out_proj.weight", "action_out_proj.bias")
                    and old.shape[0] == 32
                ):
                    new = torch.cat((old, old[-4:]), dim=0)
                else:
                    raise ValueError(
                        f"Unexpected pretrained shape mismatch {name}: {old.shape} vs {new.shape}"
                    )
                state[name] = new
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(
            not name.startswith(("reader.", "query", "ardy_head.")) for name in missing
        ):
            raise ValueError(
                f"Unexpected initialization keys: missing={missing}, unexpected={unexpected}"
            )
