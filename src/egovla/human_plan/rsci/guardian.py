"""egovla: 训练进程守护与完整断点重试。"""

import json
import os
import sys

from rsci.data.cache_config import cache_arguments
from rsci.paths import with_nccl

from .checkpoints import atomic_json, checkpoint_ok, latest

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def guardian(args, paths, value, digest):
    import fcntl
    import subprocess

    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "guardian.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = output / "launch_contract.json"
        if previous.exists() and json.loads(previous.read_text()) != value:
            raise ValueError("Output belongs to a different training contract; use a new run-tag")
        atomic_json(previous, value)
        atomic_json(output / "guardian.json", dict(pid=os.getpid(), contract=digest))
        world = len(args.gpus.split(","))
        attempts_at_step = {}
        for attempt in range(1, 4):
            if (output / "NUMERICAL_FAILURE.json").exists():
                raise RuntimeError("Numerical failure; no automatic restart")
            resume = latest(output, world, digest)
            if resume is not None and not checkpoint_ok(resume, world, digest, verify=True):
                raise RuntimeError("Resume checkpoint failed complete SHA/numerical validation")
            if checkpoint_ok(output / f"checkpoint-{args.max_steps}", world, digest, verify=True):
                print("EGOVLA_TRAINING_COMPLETE", flush=True)
                return
            if resume is None and (attempt > 1 or (output / "STARTED.json").exists()):
                raise RuntimeError("No complete checkpoint; refusing automatic restart from zero")
            step = int(resume.name.split("-")[-1]) if resume else 0
            attempts_at_step[step] = attempts_at_step.get(step, 0) + 1
            if attempts_at_step[step] > 2:
                raise RuntimeError("Repeated failure without checkpoint progress")
            atomic_json(output / "STARTED.json", dict(attempt=attempt, resume=step))
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={world}",
                "-m",
                "human_plan.rsci.train",
                *[arg for arg in sys.argv[1:] if arg != "--guardian"],
                "--worker",
                *cache_arguments(paths["input_cache"]),
            ]
            print(f"EGOVLA attempt={attempt} resume={resume}", flush=True)
            process = subprocess.Popen(
                with_nccl(command, expected_version="2.27.3"), start_new_session=True
            )

            from rsci.runtime.processes import wait_worker

            result = wait_worker(process)
            if result == 0 and checkpoint_ok(
                output / f"checkpoint-{args.max_steps}", world, digest, verify=True
            ):
                return
        raise RuntimeError("Training retries exhausted; inspect guardian.log")
