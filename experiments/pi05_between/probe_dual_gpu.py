"""Measure a second independent evaluation; preserve the existing production worker."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
OUT = ROOT / "evals/dual-gpu2-capacity-probe-20261104"


def main():
    OUT.mkdir(exist_ok=False)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("run_evaluation.py")),
        "--gpu",
        "2",
        "--slot",
        "1",
        "--variant",
        "STATE",
        "--level",
        "dr-level-0",
        "--num-repeats",
        "1",
        "--num-episodes",
        "1",
        "--output-root",
        str(OUT),
    ]
    import shlex

    (OUT / "command.sh").write_text(shlex.join(command) + "\n")
    samples = []
    result = {"formal_evaluation": False, "minimum_free_mib": 2048, "command": command}
    with (OUT / "worker.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

        def stop(signum, frame):
            raise SystemExit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        try:
            while child.poll() is None:
                query = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        "2",
                        "--query-gpu=memory.used,memory.free,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                used, free, utilization = map(int, query.strip().split(","))
                sample = dict(
                    time=time.time(), used_mib=used, free_mib=free, utilization=utilization
                )
                samples.append(sample)
                with (OUT / "memory.jsonl").open("a") as stream:
                    stream.write(json.dumps(sample) + "\n")
                if free < 2048:
                    result["status"] = "stopped_at_memory_reserve"
                    break
                time.sleep(2)
            else:
                result["status"] = "completed" if child.returncode == 0 else "worker_failed"
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=120)
            result.update(
                exit_code=child.returncode, peak_used_mib=max(s["used_mib"] for s in samples)
            )
            (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
