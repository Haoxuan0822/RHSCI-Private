"""只汇总标签计数；各骨干仍按原生循环计算动作损失和反向传播。"""

import torch
import torch.distributed as dist


def attach_auxiliary_normalization(batches, device, backward_divisor):
    """为一个参数更新组附上 ARDY/Mo 各自的分母，不持有前向图。

    DDP 平均梯度、训练器除以 backward_divisor；公共 loss 补偿这两项，
    最终得到整个更新组内所有 rank 的有效标签平均。先 clamp 再除，
    保证仅一个有效标签、全空以及不足整组的尾批次均保持正确尺度。
    """
    if not batches:
        return batches
    counts = {}
    for batch in batches:
        inputs = batch.get("inputs", batch)
        prefix = "rsci_" if "rsci_ardy_valid" in inputs else ""
        for name in ("ardy", "motion"):
            key = f"{prefix}{name}_valid"
            if key not in inputs:
                continue
            valid = inputs[key].bool()
            # GR 的模型还联合历史 mask；其它骨干的 sidecar mask 已包含此条件。
            if prefix:
                valid = valid & inputs["rsci_history_valid"].bool().all(-1)
            count = valid.sum().to(device=device, dtype=torch.float32)
            counts[name] = counts.get(name, 0) + count
    if not counts:
        return batches
    names = sorted(counts)
    totals = torch.stack([counts[name] for name in names])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    denominators = totals.clamp_min(1) / backward_divisor
    normalization = dict(zip(names, denominators.unbind()))
    for batch in batches:
        batch.get("inputs", batch)["auxiliary_normalization"] = normalization
    return batches


class AuxiliaryAccumulationTrainerMixin:
    """复用 HF 已预取的累积组，避免额外读取数据或缓存模型前向。"""

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches, num_items = super().get_batch_samples(epoch_iterator, num_batches, device)
        if self.args.gradient_accumulation_steps == 1:
            return batches, num_items
        # 新版 HF 按实际尾组长度缩放；当前受支持旧版按配置的累积次数缩放。
        # 派生 training_step 可能只包装父方法，沿 MRO 检查真正的 HF 实现。
        divisor = self.args.gradient_accumulation_steps
        for cls in type(self).__mro__:
            if cls.__module__ == "transformers.trainer":
                if "current_gradient_accumulation_steps" in cls.training_step.__code__.co_names:
                    divisor = len(batches)
                break
        attach_auxiliary_normalization(batches, device, divisor)
        return batches, num_items


def normalized_auxiliary_batches(dataloader, accumulation_steps, device):
    """Ψ0 原生逐微批循环的标签预取；保留 Accelerate 的最后一批标记。"""
    iterator = iter(dataloader)
    for start in range(0, len(dataloader), accumulation_steps):
        batches, end_flags = [], []
        for _ in range(min(accumulation_steps, len(dataloader) - start)):
            batches.append(next(iterator))
            end_flags.append(getattr(dataloader, "end_of_dataloader", False))
        attach_auxiliary_normalization(batches, device, accumulation_steps)
        for batch, end_flag in zip(batches, end_flags):
            if hasattr(dataloader, "end_of_dataloader"):
                dataloader.end_of_dataloader = end_flag
            yield batch
    # 让 Accelerate 完成迭代器退出，注销 active_dataloader；不多消费样本。
    next(iterator, None)
