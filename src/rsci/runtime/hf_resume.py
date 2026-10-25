"""Preserve sampler epochs when an RHSCI Hugging Face trainer skips resumed batches."""

from functools import update_wrapper
from types import FunctionType


def skip_preserving_epoch(dataloader, num_batches=0):
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
