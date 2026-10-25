"""RHSCI：四帧分别压缩为16个token，独立Query调用共享动作DiT。"""
import math

import torch
from torch import nn
import torch.distributed as dist


class VisualStateQuery(nn.Module):
    def __init__(self, memory_dim, query_dim, output_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, query_dim) / math.sqrt(query_dim))
        # 图像/语言各8个Reader，保留GR00T交替cross-attention语义，总数仍为每帧16。
        self.readers = nn.Parameter(torch.randn(2, 8, memory_dim) / math.sqrt(memory_dim))
        self.reader = nn.MultiheadAttention(memory_dim, math.gcd(memory_dim, 8), batch_first=True)
        self.time_embedding = nn.Parameter(torch.randn(4, memory_dim) / math.sqrt(memory_dim))
        self.ardy_head = nn.Sequential(nn.LayerNorm(output_dim), nn.Linear(output_dim, 128))

    @torch.no_grad()
    def reset_parameters(self):
        # HF低内存加载可能跳过构造期初始化；裸Parameter与MHA必须显式初始化。
        nn.init.normal_(self.query, std=1 / math.sqrt(self.query.shape[-1]))
        nn.init.normal_(self.readers, std=1 / math.sqrt(self.readers.shape[-1]))
        nn.init.normal_(self.time_embedding, std=1 / math.sqrt(self.time_embedding.shape[-1]))
        self.reader.out_proj.reset_parameters()
        self.reader._reset_parameters()
        for layer in self.ardy_head:
            layer.reset_parameters()

    def memory(self, frames, valid):
        if len(frames) != 4 or valid.shape != (frames[0]["backbone_features"].shape[0], 4):
            raise ValueError("Expected four chronological frames and [B,4] history validity")
        if not valid[:, -1].all():
            raise ValueError("Current frame must be valid")
        tokens, masks, image_masks = [], [], []
        for index, frame in enumerate(frames):
            features = frame["backbone_features"]
            if not torch.isfinite(features).all():
                raise ValueError("Nonfinite VLM features")
            image = frame["image_mask"].bool()
            available = frame["backbone_attention_mask"].bool() & valid[:, index, None].bool()
            for kind in range(2):
                selected = available & (image if kind == 0 else ~image)
                present = selected.any(-1)
                # 无效历史不能造成全mask softmax NaN；输出再严格清零并屏蔽。
                safe = selected.clone()
                safe[~present, 0] = True
                source = features.masked_fill(~selected[..., None], 0)
                read = self.reader(self.readers[kind][None].expand(features.shape[0], -1, -1),
                                   source, source, key_padding_mask=~safe, need_weights=False)[0]
                read = (read + self.time_embedding[index]) * present[:, None, None]
                tokens.append(read)
                masks.append(present[:, None].expand(-1, 8))
                image_masks.append(torch.full_like(masks[-1], kind == 0))
        memory = torch.cat(tokens, 1)
        mask, image_mask = torch.cat(masks, 1), torch.cat(image_masks, 1)
        if not (mask & image_mask).any(-1).all() or not (mask & ~image_mask).any(-1).all():
            raise ValueError("Query needs valid current image and text memory")
        return memory, mask, image_mask

    def forward(self, frames, valid, dit):
        memory, mask, image_mask = self.memory(frames, valid)
        query = self.query.expand(memory.shape[0], -1, -1)
        # 不传动作/state/动作噪声；复用同一个DiT，时间条件固定0。
        result = dit(hidden_states=query, encoder_hidden_states=memory,
                     encoder_attention_mask=mask, image_mask=image_mask,
                     backbone_attention_mask=mask,
                     timestep=torch.zeros(memory.shape[0], device=memory.device, dtype=torch.long))
        return self.ardy_head(result[:, 0]).float()


def masked_ardy_loss(prediction, target, valid):
    """按有效样本全局平均；DDP平均梯度后仍等于全局有效样本MSE。"""
    valid = valid.bool().reshape(-1)
    if target.shape != prediction.shape or prediction.shape[-1] != 128:
        raise ValueError("ARDY target/prediction must have shape [B,128]")
    target = target.float().masked_fill(~valid[:, None], 0)
    errors = (prediction.float() - target).square().mean(-1)
    denominator = valid.sum().to(errors.dtype)
    world = 1
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(denominator)
    return (errors * valid).sum() * world / denominator.clamp_min(1)
