"""Ψ0 原生检查点重试与子进程清理；调用方持有运行锁。"""

import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from rhsci.paths import CODE_ROOT
from rhsci.runtime.processes import cleanup, process_table, remember_tree

from .checkpoints import checkpoint_ok
from .contract import build_training_contract

CODE = CODE_ROOT


def latest_checkpoint(output, ranks):
    if any(Path(output).rglob("NUMERICAL_FAILURE.json")):
        raise RuntimeError("NUMERICAL_FAILURE: automatic recovery is disabled")
    candidates = [
        path
        for path in output.glob("finetune/*/checkpoints/ckpt_*")
        if path.name.removeprefix("ckpt_").isdigit() and checkpoint_ok(path, ranks)
    ]
    return max(
        candidates,
        key=lambda path: (int(path.name.removeprefix("ckpt_")), path.stat().st_mtime),
        default=None,
    )


def run_training_guardian(args, cfg, command, output, ranks):
    """运行持锁守护并等待 worker；仅从通过校验的完整断点恢复，失败后保留日志。"""
    if any(Path(output).rglob("NUMERICAL_FAILURE.json")):
        raise RuntimeError("NUMERICAL_FAILURE: automatic recovery is disabled")
    tracked = {}

    def stop(signum, frame):
        cleanup(tracked)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous_step = -1
    for attempt in range(4):
        identity_path = output / "launch_contract.json"
        if identity_path.exists() and json.loads(
            identity_path.read_text()
        ) != build_training_contract(cfg, ranks):
            raise ValueError(
                "RHSCI training configuration changed; refusing recovery with a different recipe"
            )
        resume = args.resume if attempt == 0 and args.resume else latest_checkpoint(output, ranks)
        runs_exist = (
            any(output.glob("finetune/*/run_config.json"))
            or (output / "training_started.json").exists()
        )
        if resume:
            if not checkpoint_ok(resume, ranks) or output.resolve() not in resume.resolve().parents:
                raise ValueError(
                    "Resume must be a complete checkpoint in this experiment's output directory"
                )
            saved = json.loads((resume.parent.parent / "run_config.json").read_text())
            saved_model = saved["model"]
            if (
                saved_model != cfg.model_dump(mode="json")["model"]
                or saved["data"] != cfg.model_dump(mode="json")["data"]
            ):
                raise ValueError(
                    "Checkpoint model/data do not match the resolved RHSCI configuration"
                )
            step = int(resume.name.removeprefix("ckpt_"))
            if step >= cfg.train.max_training_steps:
                print(f"Complete: {resume}", flush=True)
                return
            if attempt > 0 and step <= previous_step:
                raise RuntimeError(
                    "No new complete checkpoint since previous recovery; refusing repeated replay"
                )
            previous_step = step
        elif runs_exist or attempt > 0:
            raise RuntimeError(
                "No complete checkpoint; refusing to restart this experiment from step zero"
            )
        launch = list(command)
        if resume:
            launch += [f"--train.resume-from-checkpoint={resume}", "--train.skip-resumed-steps"]
        print(f"Attempt {attempt + 1}: {shlex.join(launch)}", flush=True)
        with (output / "training_started.json").open("w") as marker:
            json.dump(dict(attempt=attempt + 1, command=launch, started_at=time.time()), marker)
        child = subprocess.Popen(
            launch,
            cwd=CODE,
            start_new_session=True,
            env={k: v for k, v in os.environ.items() if k not in ("NCCL_PROTO", "NCCL_ALGO")},
            pass_fds=(args.guardian_fd,) if args.guardian_fd is not None else (),
        )
        tracked = {}
        try:
            remember_tree(child.pid, tracked, process_table(), own_group=child.pid)
            while child.poll() is None:
                remember_tree(child.pid, tracked, process_table(), own_group=child.pid)
                time.sleep(1)
        finally:
            remember_tree(child.pid, tracked, process_table(), own_group=child.pid)
            cleanup(tracked)
        if child.returncode == 0:
            checkpoint = latest_checkpoint(output, ranks)
            if (
                checkpoint
                and int(checkpoint.name.removeprefix("ckpt_")) >= cfg.train.max_training_steps
            ):
                print(f"Complete: {checkpoint}", flush=True)
                return
            raise RuntimeError("Training exited without a complete final checkpoint")
    raise RuntimeError("RHSCI recovery attempts exhausted")
