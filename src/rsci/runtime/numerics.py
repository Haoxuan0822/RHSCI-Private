"""跨 rank 数值失败门禁：保留现场，禁止把 NaN 当作可自动重试的瞬时故障。"""

from pathlib import Path

import torch
import torch.distributed as dist

from rsci.runtime.artifacts import atomic_json

FAILURE = "NUMERICAL_FAILURE.json"


def require_finite(values, output, step, stage):
    # 每阶段只同步一个标量；任一 rank 异常时所有 rank 都在反向/更新前退出。
    valid = torch.stack([torch.isfinite(value.detach()).all() for value in values]).all().int()
    if dist.is_initialized():
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        if not dist.is_initialized() or dist.get_rank() == 0:
            atomic_json(
                Path(output) / FAILURE,
                dict(
                    step=step,
                    stage=stage,
                    reason="Nonfinite value on at least one rank; inspect worker traceback before a new run",
                ),
            )
        raise FloatingPointError(f"Nonfinite {stage} at step {step}; automatic recovery disabled")


def reject_numerical_failure(output):
    if (Path(output) / FAILURE).exists():
        raise RuntimeError(
            "Numerical failure recorded; refusing automatic resume; inspect the run and use a new run-tag"
        )
