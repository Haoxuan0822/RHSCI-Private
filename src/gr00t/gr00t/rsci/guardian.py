"""gr00t: 训练进程守护与完整断点重试。"""

import os

from rsci.data.cache_config import cache_arguments
from rsci.paths import with_nccl, worker_arguments

from .config import build_argument_parser
from .contract import build_training_contract

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def run_training_guardian(args, paths, digest):
    """运行持锁守护并等待 worker；仅从通过校验的完整断点恢复，失败后保留日志。"""
    import signal
    import subprocess
    import sys
    import time

    from .checkpoints import checkpoint_ok, latest_checkpoint

    output = paths["output"]
    world = len(args.gpus.split(","))
    previous_step = -1
    child = None

    def stop_child(signum, frame):
        # 只清理由本守护创建的torchrun进程组，不扫描/终止其他训练和评测。
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_child)
    signal.signal(signal.SIGINT, stop_child)
    for attempt in range(4):
        from rsci.runtime.numerics import reject_numerical_failure

        reject_numerical_failure(output)
        _, current_digest = build_training_contract(args, paths)
        if current_digest != digest:
            raise ValueError("Training configuration/data changed; refusing recovery")
        resume = (
            args.resume
            if attempt == 0 and args.resume
            else latest_checkpoint(output, world, digest)
        )
        started = (output / "training_started.json").exists()
        if resume:
            if resume.resolve().parent != output.resolve() or not checkpoint_ok(
                resume, world, digest, verify_hash=True
            ):
                raise ValueError("Resume must be a verified complete checkpoint from this run")
            step = int(resume.name.split("-")[-1])
            if step >= args.max_steps:
                print(f"Already complete: {resume}", flush=True)
                return
            if attempt and step <= previous_step:
                raise RuntimeError("No new complete checkpoint; refusing repeated replay")
            previous_step = step
        elif started or attempt:
            raise RuntimeError("No complete checkpoint; refusing automatic restart from zero")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world}",
            f"--master_port={31000 + int(digest[:4], 16) % 1000}",
            "--module",
            "gr00t.rsci.train",
            "--worker",
            *worker_arguments(args, build_argument_parser()),
            *cache_arguments(paths["input_cache"]),
        ]
        if resume:
            command += ["--resume", str(resume)]
        command = with_nccl(command, args.nccl_library)
        from .checkpoints import atomic_json

        atomic_json(
            output / "training_started.json",
            {"attempt": attempt + 1, "command": command, "time": time.time()},
        )
        print(f"GR00T attempt {attempt + 1}: {command}", flush=True)
        child = subprocess.Popen(
            command,
            start_new_session=True,
            env={
                key: value
                for key, value in os.environ.items()
                if key not in ("NCCL_PROTO", "NCCL_ALGO")
            },
            pass_fds=(args.guardian_fd,),
        )
        result = child.wait()
        # torchrun正常清理各rank；若异常退出，补清该次进程组，仍保留其他任务。
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if result == 0 and checkpoint_ok(output / f"checkpoint-{args.max_steps}", world, digest):
            return
    raise RuntimeError("GR00T recovery attempts exhausted")
