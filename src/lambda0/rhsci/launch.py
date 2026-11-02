"""独立守护：只从已验收完整断点恢复，数值错误不重试。"""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rhsci.runtime.artifacts import atomic_json
from rhsci.runtime.processes import cleanup, process_table, remember_tree, run_worker_process

from .config import build_argument_parser, build_training_config


def run_training_guardian(args, argv, output):
    """运行持锁守护并等待 worker；仅从通过校验的完整断点恢复，失败后保留日志。"""
    with (output / "guardian.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for attempt in range(4):
            log = output / f"attempt-{time.time_ns()}.log"
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={len(args.gpus.split(','))}",
                "--module",
                "lambda0.rhsci.train",
                *argv,
                "--worker",
            ]
            if attempt:
                command.extend(["--resume", "auto"])
            with log.open("wb", buffering=0) as stream, log.open(errors="replace") as reader:
                worker_pid = None
                tracked = {}

                def observe_worker():
                    remember_tree(worker_pid, tracked, process_table(), own_group=worker_pid)
                    print(reader.read(), end="", flush=True)

                def record_worker(process):
                    nonlocal worker_pid
                    worker_pid = process.pid
                    observe_worker()
                    atomic_json(
                        output / "worker.json", dict(pid=worker_pid, command=command, log=str(log))
                    )

                def cleanup_worker(process):
                    try:
                        remember_tree(process.pid, tracked, process_table(), own_group=process.pid)
                        cleanup(tracked)
                    finally:
                        print(reader.read(), end="", flush=True)

                returncode = run_worker_process(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus},
                    # 守护遭 SIGKILL 时不能转发信号，worker 仍持锁，拒绝叠加启动。
                    pass_fds=(lock.fileno(),),
                    on_start=record_worker,
                    on_wait=observe_worker,
                    on_exit=cleanup_worker,
                )
            if returncode == 0:
                complete = output / "training_complete.json"
                if (
                    not complete.is_file()
                    or json.loads(complete.read_text()).get("global_step") != args.max_steps
                ):
                    raise RuntimeError(f"Worker exited without {args.max_steps}-step completion")
                return
            text = log.read_text(errors="replace")
            if any(
                key in text
                for key in (
                    "Nonfinite",
                    "non-finite",
                    "FloatingPointError",
                    "out of memory",
                    "digest mismatch",
                    "contract/config/source mismatch",
                    "Existing run differs",
                    "source differs",
                    "runtime differs",
                )
            ):
                raise RuntimeError(f"Non-retryable failure; inspect {log}")
            # 从零失败不能自动再训；只接收本 run 已验证断点。
            last = output / "last"
            receipt = last / "rsci_complete.json"
            if not receipt.is_file():
                raise RuntimeError(f"No verified complete checkpoint; inspect {log}")
            from .runtime import verify_checkpoint

            verify_checkpoint(last, json.loads(receipt.read_text())["contract"])
            print(f"Attempt {attempt} failed; verified checkpoint available, retrying.", flush=True)
        raise RuntimeError("Retry limit reached")


def main():
    argv = [value for value in sys.argv[1:] if value != "--guardian"]
    args = build_argument_parser().parse_args(argv)
    config = build_training_config(args)
    output = Path(config["output"])
    if "--guardian" in sys.argv:
        run_training_guardian(args, argv, output)
        return
    checked = subprocess.run([sys.executable, "-m", "lambda0.rhsci.train", *argv, "--dry-run"])
    if checked.returncode:
        raise SystemExit(checked.returncode)
    checked = subprocess.run([sys.executable, "-m", "lambda0.rhsci.train", *argv, "--check-data"])
    if checked.returncode:
        raise SystemExit(checked.returncode)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "launch.lock").open("a+") as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "guardian.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
                active = False
            except BlockingIOError:
                active = True
        log = output / "guardian.log"
        if not active:
            with log.open("ab", buffering=0) as stream:
                process = subprocess.Popen(
                    [sys.executable, "-m", "lambda0.rhsci.launch", *argv, "--guardian"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            print(f"λ₀ guardian PID={process.pid}; log={log}", flush=True)
        else:
            print(f"λ₀ guardian already active; log={log}", flush=True)
    try:
        subprocess.run(["tail", "-n", "50", "-F", str(log)])
    except KeyboardInterrupt:
        print("退出日志查看，后台守护继续。")


if __name__ == "__main__":
    main()
