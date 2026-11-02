"""恢复 HF 训练的数据 epoch、累计尾更新，并隔离 worker 与模型随机流。"""

import inspect
import math
import textwrap
from functools import update_wrapper
from types import FunctionType


def isolate_worker_rng(dataloader, seed):
    """将 worker 启动/重建的种子生成与模型噪声、dropout 的主 RNG 分离。"""
    import torch

    base = getattr(dataloader, "base_dataloader", dataloader)
    base.generator = torch.Generator().manual_seed(seed)
    return dataloader


def skip_preserving_epoch(dataloader, num_batches=0):
    """跳过已消费批次，返回在首次预取前恢复原 epoch 的 Accelerate loader。"""
    from accelerate import skip_first_batches

    epoch = getattr(dataloader, "iteration", None)
    if epoch is None:
        raise RuntimeError("Cannot safely resume: prepared dataloader has no explicit epoch")
    skipped = skip_first_batches(dataloader, num_batches)
    if not hasattr(skipped, "set_epoch"):
        raise RuntimeError("Cannot safely resume: skipped dataloader cannot preserve epoch")
    # Accelerate新建DataLoaderShard的iteration默认0；第一次__iter__会用它重设sampler。
    # 必须在迭代（包括prefetch）发生前恢复原loader已经设置的真实epoch。
    skipped.set_epoch(epoch)
    return skipped


def epoch_safe_training_loop(trainer_class):
    """保留上游循环的同一份bytecode，只替换本派生类方法的skip函数绑定。

    不覆盖transformers.trainer或accelerate全局函数，不影响其他Trainer/活动任务。
    上游接口改变时拒绝启动，不能静默失去此恢复保证。
    """
    from accelerate import skip_first_batches

    original = trainer_class._inner_training_loop
    if (
        "skip_first_batches" not in original.__code__.co_names
        or original.__globals__.get("skip_first_batches") is not skip_first_batches
    ):
        raise RuntimeError(
            "Unsupported HF resume loop: inspect its batch-skip implementation before training"
        )
    namespace = dict(original.__globals__, skip_first_batches=skip_preserving_epoch)
    method = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    method.__kwdefaults__ = original.__kwdefaults__
    return update_wrapper(method, original)


def tail_safe_training_loop(trainer_class):
    """返回同时修复 epoch 和累计尾部的 HF 循环，不修改依赖的全局实现。

    HF 的 step 从恢复后的首批重新计数，steps_in_epoch 仍是整轮批数。
    比如整轮10批、累计4、已跳过4批，尾批step+1是6，真实轮内位置是10；
    同步判断必须加steps_skipped，否则最后2批只有反向而没有optimizer更新。
    """
    original = epoch_safe_training_loop(trainer_class)
    source = textwrap.dedent(inspect.getsource(trainer_class._inner_training_loop))
    old = "do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch"
    new = (
        "do_sync_step = (step + 1 + steps_skipped) % args.gradient_accumulation_steps == 0 "
        "or (step + 1 + steps_skipped) == steps_in_epoch"
    )
    if source.count(old) != 1:
        raise RuntimeError(
            "Unsupported HF epoch-tail loop; inspect native implementation before training"
        )
    namespace = dict(original.__globals__)
    exec(compile(source.replace(old, new), "<rhsci-local-epoch-tail>", "exec"), namespace)
    return update_wrapper(namespace[original.__name__], original)


def tail_safe_training_values(trainer_class):
    """让每轮更新预算包含尾部不足累计组，避免HF4.51按floor定位恢复epoch。

    复用上游样本统计与配置处理；只有旧版floor和实际ceil更新数不同时重算步数。
    HF4.53已用ceil，此时直接返回原结果。与tail_safe_training_loop配套使用。
    """
    original = trainer_class.set_initial_training_values

    def training_values(self, args, dataloader, total_train_batch_size):
        values = original(self, args, dataloader, total_train_batch_size)
        (
            epochs,
            updates_per_epoch,
            examples,
            samples,
            epoch_based,
            batches_per_epoch,
            max_steps,
        ) = values
        if batches_per_epoch is None:
            return values  # 无长度loader仍由显式max_steps控制。
        actual_updates = max(math.ceil(batches_per_epoch / args.gradient_accumulation_steps), 1)
        if actual_updates == updates_per_epoch:
            return values
        if epoch_based:
            max_steps = math.ceil(args.num_train_epochs * actual_updates)
        elif batches_per_epoch:
            epochs = math.ceil(max_steps / actual_updates)
        return epochs, actual_updates, examples, samples, epoch_based, batches_per_epoch, max_steps

    return update_wrapper(training_values, original)
