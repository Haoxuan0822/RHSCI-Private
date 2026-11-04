"""Run the selected GR00T models on the three-level official old evaluation package."""

import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from plan import ACTIVE_RUNS, CURRENT, LEVELS, OUTPUT, PREPARATION, REPO, run_key, worker_command

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
    main()
