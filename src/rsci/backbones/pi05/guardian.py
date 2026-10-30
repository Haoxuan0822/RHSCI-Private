"""pi05: 训练进程守护与完整断点重试。"""

import os
import sys

from rsci.data.cache_config import cache_arguments
from rsci.paths import with_nccl

from .checkpoints import atomic_json, checkpoint_ok, latest_checkpoint
from .contract import contract

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def guardian(args, paths, digest):
    import signal
    import subprocess
    import time

    world = len(args.gpus.split(","))
    child = None
    previous_step = -1

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for attempt in range(4):
        from rsci.runtime.numerics import reject_numerical_failure

        reject_numerical_failure(paths["output"])
        if contract(args, paths)[1] != digest:
            raise ValueError("Source/data/dependencies changed; refusing automatic resume")
        resume = (
            args.resume
            if attempt == 0 and args.resume
            else latest_checkpoint(paths["output"], world, digest)
        )
        if resume:
            if resume.resolve().parent != paths["output"].resolve() or not checkpoint_ok(
                resume, world, digest, verify_hash=True
            ):
                raise ValueError("Resume must be complete and from this experiment")
            step = int(resume.name.split("-")[-1])
            if step >= args.max_steps:
                return
            if attempt and step <= previous_step:
                raise RuntimeError("No new complete checkpoint; refusing repeated replay")
            previous_step = step
        elif attempt or (paths["output"] / "training_started.json").exists():
            raise RuntimeError(
                "No complete checkpoint; refusing automatic restart from zero; inspect the earlier worker error"
            )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world}",
            f"--master_port={32000 + int(digest[:4], 16) % 1000}",
            "--module",
            "rsci.backbones.pi05.train",
            *[value for value in sys.argv[1:] if not value.startswith("--guardian-fd")],
            "--worker",
            *cache_arguments(paths["input_cache"]),
        ]
        # 不把父命令的旧 resume 带入下一次恢复。
        if "--resume" in command:
            index = command.index("--resume")
            del command[index : index + 2]
        if resume:
            command += ["--resume", str(resume)]
        command = with_nccl(command, args.nccl_library)
        atomic_json(
            paths["output"] / "training_started.json",
            dict(attempt=attempt + 1, command=command, time=time.time()),
        )
        print(f"PI05 attempt={attempt + 1} resume={resume}", flush=True)
        child = subprocess.Popen(
            command,
            start_new_session=True,
            pass_fds=(args.guardian_fd,),
            env={
                key: value
                for key, value in os.environ.items()
                if key not in ("NCCL_PROTO", "NCCL_ALGO")
            },
        )
        code_result = child.wait()
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if code_result == 0 and checkpoint_ok(
            paths["output"] / f"checkpoint-{args.max_steps}", world, digest
        ):
            return
    raise RuntimeError("Recovery attempts exhausted")
