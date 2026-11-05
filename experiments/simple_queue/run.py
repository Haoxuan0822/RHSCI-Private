"""Run the selected GR00T models on the three-level official old evaluation package."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from plan import ACTIVE_RUNS, CURRENT, LEVELS, OUTPUT, PREPARATION, REPO, run_key, worker_command


def rerun_on_gpus(gpus, output, *, resume=False, auto_gpus=False, backbone="gr00t"):
    """按任务优先级派发；接管在途worker，空闲卡立即接下一等级任务。"""
    import fcntl

    from plan import PI05_THREE_RUNS, ROOT, RUNS

    from rhsci.runtime.processes import process_table

    output.mkdir(parents=True, exist_ok=resume)
    scheduler_lock = (output / "scheduler.lock").open("a")
    fcntl.flock(scheduler_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(
        os.environ, PYTHONPATH=str(REPO / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
    )
    processes = {}
    handles = []

    def alive(job):
        row = process_table().get(job["pid"])
        return row is not None and row[2] == job["birth"] and row[3] != "Z"

    pending = []
    for task in ("oven", "handover", "faucet"):
        source = PI05_THREE_RUNS if backbone == "pi05" else RUNS
        runs = [run for run in source if run["backbone"] == backbone and run["task"] == task]
        for level in LEVELS:
            for run in runs:
                base = output / run_key(run) / run["variant"] / level
                marker = base / "COMPLETE.json"
                metadata = base / "worker.json"
                if resume and metadata.exists():
                    info = json.loads(metadata.read_text())
                    pid, gpu = info["worker_pid"], info["gpu"]
                    row = process_table().get(pid)
                    if row is not None and row[3] != "Z":
                        command = Path(f"/proc/{pid}/cmdline").read_bytes()
                        expected_output = str(output / run_key(run)).encode()
                        if b"run_evaluation.py" not in command or expected_output not in command:
                            raise RuntimeError(f"Worker PID identity changed: {metadata}")
                        if gpu not in gpus or gpu in processes:
                            raise RuntimeError(f"Unexpected occupied GPU{gpu}: {metadata}")
                        processes[gpu] = dict(pid=pid, birth=row[2], process=None, marker=marker)
                        print("GR00T_WORKER_ADOPTED", gpu, run_key(run), level, pid, flush=True)
                        continue
                    if marker.exists() and json.loads(marker.read_text())["episodes"] == 50:
                        continue
                    raise RuntimeError(f"Stopped incomplete worker; logs retained: {metadata}")
                pending.append((run, level))
    try:
        while pending or processes:
            maintain_gpu_reservation()
            # 先让原GR00T派发器派完18项，随后仅接用它释放的卡，避免两队列抢卡。
            if backbone == "pi05":
                previous = ROOT / "evals/gr00t-zero-arms-pre1102-5cycles-20261104"
                if len(list(previous.glob("gr00t-*/*/dr-level-*/worker.json"))) < 18:
                    print("WAIT_GR00T_DISPATCH", flush=True)
                    time.sleep(10)
                    continue
            for gpu, job in list(processes.items()):
                process, marker = job["process"], job["marker"]
                code = process.poll() if process is not None else None
                if alive(job):
                    continue
                if (
                    code not in (None, 0)
                    or not marker.exists()
                    or json.loads(marker.read_text())["episodes"] != 50
                ):
                    raise RuntimeError(f"GR00T worker failed: GPU{gpu}, exit={code}, {marker}")
                del processes[gpu]
            available = set(gpus)
            if auto_gpus:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=20,
                )
                compute = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=gpu_uuid",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=20,
                )
                identities = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=20,
                )
                computing = set(compute.stdout.splitlines())
                occupied = {
                    int(index)
                    for line in identities.stdout.splitlines()
                    for index, uuid in [line.split(",")]
                    if uuid.strip() in computing
                }
                # Isaac其他卡的渲染进程会在GPU0附带少量图形上下文；无计算进程且
                # 仅有少量显存时允许接用，不能把这些上下文误判为一项活动评测。
                available = {
                    int(index)
                    for line in result.stdout.splitlines()
                    for index, memory in [line.split(",")]
                    if int(memory) < 4096 and int(index) in gpus and int(index) not in occupied
                }
            for gpu in gpus:
                if gpu in processes or gpu not in available or not pending:
                    continue
                ready = next(
                    (
                        (run, level)
                        for run, level in pending
                        if (PREPARATION / f"{run_key(run)}.json").exists()
                    ),
                    None,
                )
                if ready is None:
                    continue
                run, level = ready
                key = run_key(run)
                receipt = PREPARATION / f"{key}.json"
                if not receipt.exists():
                    print("WAIT_CHECKPOINT", key, flush=True)
                    continue
                info = json.loads(receipt.read_text())
                run = dict(run, horizon=info["horizon"])
                command = worker_command(run, gpu, level, info["run_dir"], output_root=output)
                directory = output / key
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"command-{level}.sh").write_text(shlex.join(command) + "\n")
                log = (directory / f"worker-{level}.log").open("x")
                handles.append(log)
                process = subprocess.Popen(
                    command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
                marker = directory / run["variant"] / level / "COMPLETE.json"
                processes[gpu] = dict(
                    pid=process.pid,
                    birth=process_table()[process.pid][2],
                    process=process,
                    marker=marker,
                )
                pending.remove(ready)
                print("GR00T_RERUN_START", gpu, key, level, process.pid, flush=True)
            if pending or processes:
                time.sleep(10)
        (output / "COMPLETE.json").write_text(json.dumps({"models": 6, "episodes": 900}) + "\n")
    finally:
        for job in processes.values():
            if alive(job):
                try:
                    os.kill(job["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for job in processes.values():
            if job["process"] is not None:
                job["process"].wait()
        for handle in handles:
            handle.close()
        scheduler_lock.close()


next_pause_renewal = 0.0


def maintain_gpu_reservation():
    """使用本机现有接口续暂停空闲占卡程序，避免长评测中自动恢复占卡。"""
    global next_pause_renewal
    if time.monotonic() >= next_pause_renewal:
        result = subprocess.run(
            ["/root/bin/xiancun", "pause", "360"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        print("KEEPALIVE_PAUSED_FOR_EVALUATION", result.stdout.strip(), flush=True)
        next_pause_renewal = time.monotonic() + 3600


def current_complete():
    workers = list(CURRENT.glob("*/*/worker.json"))
    if len(workers) != 8:
        return False
    for worker in workers:
        marker = worker.parent / "COMPLETE.json"
        if not marker.exists():
            return False
        if json.loads(marker.read_text())["episodes"] != 50:
            raise RuntimeError(f"Wrong episode count: {marker}")
        for repeat in range(1, 6):
            report = json.loads(
                (worker.parent / f"repeat-{repeat:02d}/verified_results.json").read_text()
            )
            if report["episodes"] != 10 or set(report["stats"]) != {str(i) for i in range(10)}:
                raise RuntimeError(f"Unverified evaluation: {worker.parent}")
        pid = json.loads(worker.read_text())["worker_pid"]
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes()
        except FileNotFoundError:
            continue
        if b"experiments/pi05_between/run_evaluation.py" in args:
            return False
    return True


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preparation = PREPARATION
    while not current_complete() or not (preparation / "READY.json").exists():
        maintain_gpu_reservation()
        if (preparation / "FAILED.txt").exists():
            raise RuntimeError(f"Follow-up preparation failed: {preparation / 'FAILED.txt'}")
        print("WAIT_CURRENT_400_AND_ASSETS", flush=True)
        time.sleep(30)
    ready = json.loads((preparation / "READY.json").read_text())
    # 用户最新范围：不再评测π0.5，只用官方旧包的三个等级。
    waves = [
        [run for run in ACTIVE_RUNS if run["task"] == task]
        for task in ("oven", "handover", "faucet")
    ]
    environment = dict(
        os.environ, PYTHONPATH=str(REPO / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
    )
    for wave in waves:
        processes = []
        handles = []
        try:
            for slot, run in enumerate(wave):
                key = run_key(run)
                directory = OUTPUT / key
                directory.mkdir(parents=True, exist_ok=True)
                for offset, level in enumerate(LEVELS):
                    gpu = slot * len(LEVELS) + offset
                    command = worker_command(run, gpu, level, ready[key]["run_dir"])
                    lines = [shlex.join(command[:3])]
                    lines.extend(
                        shlex.join(command[index : index + 2])
                        for index in range(3, len(command), 2)
                    )
                    (directory / f"command-gpu{gpu}.sh").write_text(" \\\n  ".join(lines) + "\n")
                    log = (directory / f"worker-gpu{gpu}.log").open("w")
                    handles.append(log)
                    print("FOLLOWUP_START", key, gpu, level, shlex.join(command), flush=True)
                    processes.append(
                        (
                            run,
                            level,
                            subprocess.Popen(
                                command,
                                cwd=REPO,
                                env=environment,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            ),
                        )
                    )
            failures = []
            while any(process.poll() is None for _, _, process in processes):
                maintain_gpu_reservation()
                time.sleep(30)
            for run, level, process in processes:
                code = process.wait()
                marker = OUTPUT / run_key(run) / run["variant"] / level / "COMPLETE.json"
                if code or not marker.exists() or json.loads(marker.read_text())["episodes"] != 50:
                    failures.append((run_key(run), level, code))
            if failures:
                raise RuntimeError(f"Follow-up wave failed; logs retained: {failures}")
        finally:
            for _, _, process in processes:
                if process.poll() is None:
                    process.terminate()
            for _, _, process in processes:
                process.wait()
            for handle in handles:
                handle.close()
    (OUTPUT / "COMPLETE.json").write_text(
        json.dumps({"models": len(ACTIVE_RUNS), "episodes": 50 * len(LEVELS) * len(ACTIVE_RUNS)})
        + "\n"
    )


if __name__ == "__main__":

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", choices=range(8))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-gpus", action="store_true")
    parser.add_argument("--backbone", choices=["gr00t", "pi05"], default="gr00t")
    args = parser.parse_args()
    if args.gpus:
        if len(set(args.gpus)) != len(args.gpus) or args.output_root is None:
            parser.error("Use distinct --gpus and a fresh --output-root")
        rerun_on_gpus(
            args.gpus,
            args.output_root,
            resume=args.resume,
            auto_gpus=args.auto_gpus,
            backbone=args.backbone,
        )
    else:
        main()
