"""独立守护：只从已验收完整断点恢复，数值错误不重试。"""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .train import configuration, parser


def guardian(args, argv, output):
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
                "rsci.backbones.lambda0.train",
                *argv,
                "--worker",
            ]
            if attempt:
                command.extend(["--resume", "auto"])
            with log.open("wb", buffering=0) as stream:
                process = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus},
                    start_new_session=True,
                )
                (output / "worker.json").write_text(
                    json.dumps(dict(pid=process.pid, command=command, log=str(log)))
                )
                # 守护日志包含训练实时输出；前台 Ctrl+C 只停止 tail。
                with log.open() as reader:
                    while process.poll() is None:
                        text = reader.read()
                        if text:
                            print(text, end="", flush=True)
                        time.sleep(0.5)
                    print(reader.read(), end="", flush=True)
                returncode = process.wait()
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
    args = parser().parse_args(argv)
    config = configuration(args)
    output = Path(config["output"])
    if "--guardian" in sys.argv:
        guardian(args, argv, output)
        return
    checked = subprocess.run(
        [sys.executable, "-m", "rsci.backbones.lambda0.train", *argv, "--dry-run"]
    )
    if checked.returncode:
        raise SystemExit(checked.returncode)
    checked = subprocess.run(
        [sys.executable, "-m", "rsci.backbones.lambda0.train", *argv, "--check-data"]
    )
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
                    [sys.executable, "-m", "rsci.backbones.lambda0.launch", *argv, "--guardian"],
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
