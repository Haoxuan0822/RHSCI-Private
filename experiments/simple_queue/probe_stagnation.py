"""复现烤箱停滞并记录三层关节目标；不修改正式评测或模型。"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments/pi05_between")]
from run_evaluation import owns_listener
from rhsci.runtime.processes import (
    cleanup,
    process_table,
    remember_tree,
    run_worker_process,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evals/gr00t-oven-stagnation-20261106"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lease = (ROOT / f"evals/.gpu-worker-leases/gpu{args.gpu}-slot1.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    port = 8166
    tracked = {}
    server = None
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONPATH=str(REPO / "src"),
        OMP_NUM_THREADS="8",
        NO_PROXY="localhost,127.0.0.1",
        no_proxy="localhost,127.0.0.1",
    )
    command = [
        str(ROOT / ".venv-gr00t-rsci/bin/python"),
        "-m",
        "rhsci.inference.http_server",
        "--backbone",
        "gr00t",
        "--run-dir",
        str(ROOT / "checkpoints/followup/gr00t-VGQ-AE-oven/inference"),
        "--checkpoint-step",
        "20000",
        "--device",
        "cuda:0",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--no-deterministic-eval",
    ]

    def monitor():
        free = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(args.gpu),
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        if free < 2048:
            raise RuntimeError(f"Diagnostic-only memory guard: {free} MiB free")
        if server is not None:
            remember_tree(server.pid, tracked, process_table(), own_group=server.pid)
            if server.poll() is not None:
                raise RuntimeError(f"Diagnostic model exited: {server.returncode}")

    try:
        with (args.output / "server.log").open("w") as log:
            server = subprocess.Popen(
                command,
                env=env,
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            remember_tree(server.pid, tracked, process_table(), own_group=server.pid)
            (args.output / "probe.json").write_text(
                json.dumps(
                    dict(
                        gpu=args.gpu,
                        server_pid=server.pid,
                        command=command,
                        scope="diagnostic only; independent stochastic replay, not original telemetry",
                    ),
                    indent=2,
                )
            )
            deadline = time.monotonic() + 900
            while not owns_listener(server.pid, port):
                monitor()
                if time.monotonic() > deadline:
                    raise TimeoutError("Diagnostic server readiness timeout")
                time.sleep(2)
            sim_env = dict(
                env,
                VIRTUAL_ENV=str(ROOT / ".venv-simple"),
                MUJOCO_GL="egl",
                OMNI_KIT_ACCEPT_EULA="Y",
                OMNI_KIT_ALLOW_ROOT="1",
                HF_HUB_OFFLINE="0",
                TRANSFORMERS_OFFLINE="0",
                SIMPLE_HOME=str(ROOT / "external/SIMPLE-official"),
            )
            sim_env.pop("PYTHONPATH", None)
            sim_env["PATH"] = str(ROOT / ".venv-simple/bin") + ":" + env["PATH"]
            sim = [
                str(ROOT / ".venv-simple/bin/python"),
                "-u",
                str(REPO / "experiments/pi05_between/eval_decoupled_wbc.py"),
                "simple/G1WholebodyOpenOvenTeleop-v0",
                "psi0_decoupled_wbc",
                "train",
                "--data-dir",
                str(
                    ROOT
                    / "eval-official-pre1102/G1WholebodyOpenOvenTeleop-v0/dr-level-1"
                ),
                "--port",
                str(port),
                "--headless",
                "--num-episodes",
                "1",
                "--episode-start",
                "7",
                "--max-episode-steps",
                "1000",
                "--success-criteria",
                "0.5",
                "--eval-dir",
                str(args.output / "stats"),
                "--save-video",
                "--renderer-gpu",
                str(args.gpu),
                "--dds-domain-id",
                "226",
                "--repair-scene-references",
                "--trace-control",
                str(args.output / "trace.jsonl"),
            ]
            (args.output / "client-command.json").write_text(json.dumps(sim, indent=2))
            with (args.output / "console.log").open("w") as sim_log:
                result = run_worker_process(
                    sim,
                    env=sim_env,
                    cwd=args.output,
                    stdout=sim_log,
                    stderr=subprocess.STDOUT,
                    on_wait=monitor,
                )
            print("DIAGNOSTIC_EXIT", result, flush=True)
            if result:
                raise RuntimeError(f"Diagnostic simulation exited: {result}")
    finally:
        if server is not None:
            remember_tree(server.pid, tracked, process_table(), own_group=server.pid)
        cleanup(tracked)
        lease.close()


if __name__ == "__main__":
    main()
