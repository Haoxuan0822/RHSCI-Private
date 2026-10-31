"""EgoVLA局部修复HF4.53.2恢复后epoch尾部更新；不修改依赖全局或GR00T。"""

import inspect
import textwrap
from functools import update_wrapper

from rsci.runtime.hf_resume import epoch_safe_training_loop


def tail_safe_training_loop(trainer_class):
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
    exec(compile(source.replace(old, new), "<egovla-local-epoch-tail>", "exec"), namespace)
    return update_wrapper(namespace[original.__name__], original)
