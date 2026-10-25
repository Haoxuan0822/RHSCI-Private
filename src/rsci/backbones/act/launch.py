"""后台守护独立持锁；前台只跟随日志。中断无完整断点不得从零重跑。"""

import fcntl
import json
import os
import subprocess
import sys

from rsci.runtime.processes import wait_worker

from .checkpoints import atomic_json, checkpoint_ok, latest


def guardian(args, paths, value=None, digest=None):
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "guardian.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if value is None:
            from .train import check_data, contract, digest_contract

            # 先持锁再做耗时真实数据验收，避免重复启动或把预检时间当守护未启动。
            value = contract(args, paths)
            digest = digest_contract(value)
            check_data(args, paths)
        previous = output / "launch_contract.json"
        if previous.exists() and json.loads(previous.read_text()) != value:
            raise ValueError("Output training contract changed; use a new run-tag")
        atomic_json(previous, value)
        atomic_json(output / "guardian.json", dict(pid=os.getpid(), contract=digest))
        world = len(args.gpus.split(","))
        for attempt in range(1, 4):
            if (output / "NUMERICAL_FAILURE.json").exists():
                raise RuntimeError("Numerical failure prohibits automatic restart")
            resume = latest(output, world, digest)
            if resume and not checkpoint_ok(resume, world, digest, verify=True):
                raise RuntimeError("Checkpoint failed SHA/numerical validation")
            if checkpoint_ok(output / f"checkpoint-{args.max_steps}", world, digest, verify=True):
                print("ACT_TRAINING_COMPLETE", flush=True)
                return
            if resume is None and ((output / "STARTED.json").exists() or attempt > 1):
                raise RuntimeError("No complete checkpoint; refusing automatic restart from zero")
            atomic_json(output / "STARTED.json", dict(attempt=attempt, resume=str(resume)))
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={world}",
                "-m",
                "rsci.backbones.act.train",
                *[arg for arg in sys.argv[1:] if arg != "--guardian"],
                "--worker",
            ]
            process = subprocess.Popen(
                command,
                start_new_session=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus},
            )
            result = wait_worker(process)
            if result == 0 and checkpoint_ok(
                output / f"checkpoint-{args.max_steps}", world, digest, verify=True
            ):
                return
        raise RuntimeError("ACT retry budget exhausted")


def main():
    from .train import initialization, parser, resolve

    args = parser().parse_args()
    from rsci.paths import with_nccl

    if args.nccl_library:
        os.environ["RSCI_NCCL_LIBRARY"] = str(args.nccl_library.resolve(strict=True))
    if not os.environ.get("RSCI_NCCL_PRELOADED"):
        command = [sys.executable, "-m", "rsci.backbones.act.launch", *sys.argv[1:]]
        wrapped = with_nccl(command, args.nccl_library, expected_version="2.27.3")
        if wrapped != command:
            os.execvpe(wrapped[0], wrapped, {**os.environ, "RSCI_NCCL_PRELOADED": "1"})
    output = resolve(args)["output"]
    initialization()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "guardian.lock").open("a+") as guard_lock:
            try:
                fcntl.flock(guard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                active = False
            except BlockingIOError:
                active = True
        log_path = output / "guardian.log"
        if not active:
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "rsci.backbones.act.train", *sys.argv[1:], "--guardian"],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            print(f"ACT guardian PID={process.pid}", flush=True)
            import time

            for _ in range(100):
                with (output / "guardian.lock").open("a+") as probe:
                    try:
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        break
                if process.poll() is not None:
                    raise RuntimeError(f"ACT guardian startup failed; see {log_path}")
                time.sleep(0.1)
            else:
                raise RuntimeError(
                    f"ACT guardian startup lock timed out; inspect PID={process.pid}"
                )
        print(f"ACT log={log_path}", flush=True)
    try:
        subprocess.run(["tail", "-n", "50", "-F", str(log_path)])
    except KeyboardInterrupt:
        print("退出日志查看；后台训练保留。")


if __name__ == "__main__":
    main()
