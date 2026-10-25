"""RHSCI显存优化：保持HF参数分组，分片FP32 Adam状态并安全保存/恢复。"""

from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer


def unwrap_zero(optimizer):
    # Accelerate只包一层；禁止误把普通Adam当成已分片优化器。
    optimizer = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(optimizer, ZeroRedundancyOptimizer):
        raise TypeError("Expected a ZeroRedundancyOptimizer")
    return optimizer


class ShardedOptimizerTrainerMixin:
    """复用生产Trainer；CPU回归也必须走相同构造及断点入口。"""

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        prepare = self.accelerator.prepare_optimizer

        def prepare_optimizer(optimizer, device_placement=None):
            # Accelerate默认读取全量state_dict再搬设备；ZeRO尚未收集时该操作会报错。
            # 模型参数已经在本rank设备上，ZeRO负责其本地状态，不需要再次搬迁。
            if isinstance(optimizer, ZeroRedundancyOptimizer):
                device_placement = False
            return prepare(optimizer, device_placement=device_placement)

        self.accelerator.prepare_optimizer = prepare_optimizer

    def create_optimizer(self):
        if self.optimizer is None:
            if not dist.is_initialized():
                raise RuntimeError("Sharded Adam requires torchrun (including world-size one)")
            # 原生HF负责decay/no-decay分组和AdamW默认值；此时尚无动量，不复制状态。
            native = super().create_optimizer()
            if not isinstance(native, torch.optim.AdamW):
                raise TypeError("Memory recipe requires the original AdamW optimizer")
            # torch2.7 defaults含内部decoupled_weight_decay，不能直接回传AdamW构造器。
            optimizer_class, kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
            self.optimizer = ZeroRedundancyOptimizer(
                native.param_groups,
                optimizer_class=optimizer_class,
                parameters_as_bucket_view=False,
                overlap_with_ddp=False,
                **kwargs,
            )
        return self.optimizer

    def _save_optimizer_and_scheduler(self, output_dir):
        zero = unwrap_zero(self.optimizer)
        # 所有rank共同收集，rank0得到CPU全量状态，保留现有完整HF断点/数值验收契约。
        zero.consolidate_state_dict(to=0)
        try:
            super()._save_optimizer_and_scheduler(output_dir)
        finally:
            # PyTorch 2.7保留收集缓存；保存后释放，避免rank0常驻整份CPU动量。
            zero._all_state_dicts.clear()
        dist.barrier()

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return
        path = Path(checkpoint)
        if not (path / "optimizer.pt").is_file() or not (path / "scheduler.pt").is_file():
            raise ValueError("Missing complete optimizer/scheduler checkpoint")
        # HF多卡默认先把整份动量加载到每张GPU，会抵消分片。CPU mmap后仅迁移本rank份额。
        state = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True, mmap=True)
        zero = unwrap_zero(self.optimizer)
        zero.load_state_dict(state)
        del state
        # torch2.7 ZeRO把标量step强制留CPU；fused AdamW原生策略要求step随参数设备。
        # 只规范本地分片，已在GPU的动量无需复制，避免CUDA恢复后首个fused更新失败。
        zero.optim.load_state_dict(zero.optim.state_dict())
        self.lr_scheduler.load_state_dict(
            torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
        )
        dist.barrier()


def enable_dit_recomputation(model):
    # 动作路径与ARDY Query共用同一DiT实例，两次调用均生效；不改变推理行为。
    model.action_head.model.gradient_checkpointing = True
