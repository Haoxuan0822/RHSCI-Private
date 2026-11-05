"""暂停已核验的四路后续STATE，保留烤箱三级及递物L0，为对应VGQ立即腾卡。"""

import json
import os
import signal
from pathlib import Path

from rhsci.runtime.processes import cleanup, process_table, remember_tree, survivors

ROOT = Path(
    "/root/workspace/Haoxuan/RHSCI-runtime/evals/pi05-three-tasks-zero-arms-pre1102-5cycles-20261105"
)


def main():
    scheduler = 4072540
    command = Path(f"/proc/{scheduler}/cmdline").read_bytes()
    if b"simple_queue/run.py" not in command or str(ROOT).encode() not in command:
        raise RuntimeError("Scheduler identity mismatch")
    selected = []
    for metadata in ROOT.rglob("worker.json"):
        info = json.loads(metadata.read_text())
        if info["variant"] != "STATE":
            continue
        task = info["task"]
        if ("Handover" in task and info["level"] != "dr-level-0") or "OpenFaucet" in task:
            pid = info["worker_pid"]
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"run_evaluation.py" not in raw or str(ROOT).encode() not in raw:
                raise RuntimeError(f"Worker identity mismatch: {pid}")
            selected.append((metadata, info))
    if len(selected) != 4:
        raise RuntimeError("Expected exactly four STATE workers to pause")
    os.kill(scheduler, signal.SIGSTOP)
    os.kill(scheduler, signal.SIGKILL)
    tracked = {}
    table = process_table()
    for metadata, info in selected:
        pid = info["worker_pid"]
        tracked[pid] = table[pid][2]
        remember_tree(pid, tracked, table)
        (metadata.parent / "PAUSED.json").write_text(
            json.dumps(
                {
                    "status": "paused_by_user_for_paired_evaluation",
                    "completed_episodes_retained": True,
                    "interrupted_episode_is_failure": False,
                    "resume_pending": True,
                    "previous_worker": info,
                },
                indent=2,
            )
            + "\n"
        )
    cleanup(tracked)
    print(
        json.dumps(
            {
                "paused": [(info["gpu"], info["task"], info["level"]) for _, info in selected],
                "survivors": survivors(tracked),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
