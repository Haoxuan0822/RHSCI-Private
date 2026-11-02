"""共享完整检查点重试循环；模型适配器提供命令与检查点校验。"""

import os
import time
from pathlib import Path

from .artifacts import atomic_json
from .numerics import reject_numerical_failure
from .processes import run_worker_process


def run_checkpoint_guardian(
    output,
    *,
    max_steps,
    resume,
    latest_checkpoint,
    checkpoint_ok,
    validate_run,
    build_command,
    label,
    pass_fds=(),
):
    """最多执行四次 worker；仅从本 run 的完整点恢复，不重复回放无进展的断点。

    latest_checkpoint/validate_run 不接收参数；checkpoint_ok 接收路径及 verify_hash；
    build_command 接收恢复路径。锁由调用方持有，模型/配方与断点格式由适配器定义。
    """
    output = Path(output)
    initial_resume, previous_step = resume, -1
    environment = {
        key: value for key, value in os.environ.items() if key not in ("NCCL_PROTO", "NCCL_ALGO")
    }
    for attempt in range(4):
        reject_numerical_failure(output)
        validate_run()
        resume = initial_resume if attempt == 0 and initial_resume else latest_checkpoint()
        if resume:
            resume = Path(resume)
            if resume.resolve().parent != output.resolve() or not checkpoint_ok(
                resume, verify_hash=True
            ):
                raise ValueError("Resume must be a verified complete checkpoint from this run")
            step = int(resume.name.split("-")[-1])
            if step >= max_steps:
                print(f"{label} already complete: {resume}", flush=True)
                return
            if attempt and step <= previous_step:
                raise RuntimeError("No new complete checkpoint; refusing repeated replay")
            previous_step = step
        elif attempt or (output / "training_started.json").exists():
            raise RuntimeError("No complete checkpoint; refusing automatic restart from zero")
        command = build_command(resume)
        atomic_json(
            output / "training_started.json",
            dict(attempt=attempt + 1, command=command, time=time.time()),
        )
        print(f"{label} attempt={attempt + 1} resume={resume}", flush=True)
        result = run_worker_process(command, pass_fds=pass_fds, env=environment)
        if result == 0 and checkpoint_ok(output / f"checkpoint-{max_steps}"):
            return
    raise RuntimeError(f"{label} recovery attempts exhausted")
