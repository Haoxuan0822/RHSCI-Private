"""Run one GPU's five cycles through the native RHSCI and official SIMPLE interfaces."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from summarize_results import read_stats

ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
REPO = Path("/root/workspace/RHSCI-Private")
TASK = "G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
VGQ = (
    ROOT
    / "hf/RHSCI/model-59a1ea14/checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1"
)
STATE = ROOT / "checkpoints/pi05-STATE-between-40000"
OUT = ROOT / "evals/pi05-between-30x30-5cycles-20261104"


def worker(args):
    import numpy as np
    import requests
    from PIL import Image

    from rhsci.inference.protocol import RequestMessage, ResponseMessage
    from rhsci.runtime.processes import cleanup, process_table, remember_tree, run_worker_process

    variant = args.variant
    level = args.level
    task = args.task
    dataset = ROOT / "data/processed" / task
    base = args.output_root / variant / level
    base.mkdir(parents=True, exist_ok=True)
    if (base / "worker.json").exists():
        raise RuntimeError(f"Refusing to overwrite an existing worker: {base}")
    port = 8120 + args.gpu + 8 * args.slot
    domain = 100 + args.gpu + 8 * args.slot
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONUNBUFFERED="1",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / f"cache/torchinductor-pi05-gpu{args.gpu}"),
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        OMP_NUM_THREADS="8",
        TORCHINDUCTOR_COMPILE_THREADS="4",
    )
    command = [
        str(ROOT / f".venv-{args.backbone}-rsci/bin/python"),
        "-m",
        "rhsci.inference.http_server",
        "--backbone",
        args.backbone,
        "--run-dir",
        str(args.run_dir or (STATE if variant == "STATE" else VGQ)),
        "--checkpoint-step",
        str(args.checkpoint_step),
        "--device",
        "cuda:0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-deterministic-eval",
    ]
    if args.backbone == "pi05":
        command.extend(
            [
                "--dataset",
                str(dataset),
                "--tokenizer",
                str(ROOT / "models/paligemma_tokenizer.model"),
            ]
        )
    tracked = {}
    server = None

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        with (base / "server.log").open("w") as log:
            server = subprocess.Popen(
                command,
                env=env,
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            table = process_table()
            tracked[server.pid] = table[server.pid][2]
            (base / "worker.json").write_text(
                json.dumps(
                    {
                        "worker_pid": os.getpid(),
                        "server_pid": server.pid,
                        "gpu": args.gpu,
                        "variant": variant,
                        "level": level,
                        "port": port,
                        "dds_domain": domain,
                        "slot": args.slot,
                        "command": command,
                        "backbone": args.backbone,
                        "task": task,
                        "prediction_horizon": args.expected_horizon,
                        "execution_horizon": args.expected_horizon,
                        "rtc": False,
                        "cycles": args.num_repeats,
                        "episodes_per_cycle": args.num_episodes,
                        "video_rounds": args.video_rounds,
                        "evaluation_root": str(args.evaluation_root),
                    },
                    indent=2,
                )
                + "\n"
            )
            client = requests.Session()
            client.trust_env = False
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; see {log.name}")
                try:
                    response = client.get(f"http://127.0.0.1:{port}/health", timeout=2)
                    response.raise_for_status()
                    break
                except requests.RequestException:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Model load readiness timeout")
                    time.sleep(2)
            # Compile and verify the actual model response before starting Isaac Sim.
            image = np.asarray(
                Image.open(
                    ROOT / "logs/simple-smoke/mujoco_isaac" / TASK / "head_stereo_left.png"
                ).convert("RGB")
            )
            if args.backbone == "pi05":
                stats = json.loads((dataset / "meta/stats.json").read_text())
                state = np.asarray(stats["states"]["mean"], dtype=np.float32)[None]
            else:
                state = np.zeros((1, 32), dtype=np.float32)
            request = RequestMessage(
                image={"rgb_head_stereo_left": image},
                instruction="Pick up the object from one table and place it into the container on the other table.",
                history={"reset": True},
                state={"states": state},
                condition={},
                gt_action=[],
                dataset_name="simple",
                timestamp="warmup",
            )
            response = client.post(
                f"http://127.0.0.1:{port}/act", json=request.serialize(), timeout=1800
            )
            response.raise_for_status()
            action = ResponseMessage.deserialize(response.json()).action
            assert action.shape == (args.expected_horizon, 36) and np.isfinite(action).all()
            (base / "warmup.json").write_text(
                json.dumps(
                    {"shape": list(action.shape), "finite": True, "evaluation_episode": False}
                )
                + "\n"
            )
            print("MODEL_READY", args.gpu, variant, level, action.shape, flush=True)
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

            def monitor():
                remember_tree(server.pid, tracked, process_table(), own_group=server.pid)
                if server.poll() is not None:
                    raise RuntimeError("Policy server exited during evaluation")

            for repeat in range(1, args.num_repeats + 1):
                destination = base / f"repeat-{repeat:02d}"
                destination.mkdir(exist_ok=False)
                sim_command = [
                    str(ROOT / ".venv-simple/bin/python"),
                    "-u",
                    str(Path(__file__).with_name("eval_decoupled_wbc.py")),
                    "simple/" + task,
                    "psi0_decoupled_wbc",
                    "train",
                    "--data-format",
                    "lerobot",
                    "--sim-mode",
                    "mujoco_isaac",
                    "--data-dir",
                    str(args.evaluation_root / task / level),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--headless",
                    "--num-episodes",
                    str(args.num_episodes),
                    "--episode-start",
                    "0",
                    "--max-episode-steps",
                    str(args.max_episode_steps),
                    "--success-criteria",
                    str(args.success_criteria),
                    "--eval-dir",
                    str(destination / "stats"),
                    "--save-video",
                    "--evaluation-round",
                    str(repeat),
                    "--video-rounds",
                    str(args.video_rounds),
                    "--renderer-gpu",
                    str(args.gpu),
                    "--dds-domain-id",
                    str(domain),
                    "--cpu-threads",
                    "8",
                ]
                import shlex

                if args.repair_scene_references:
                    sim_command.append("--repair-scene-references")

                (destination / "command.sh").write_text(
                    "# Working directory: "
                    + str(destination)
                    + "\n"
                    + " \\\n  ".join(shlex.quote(part) for part in sim_command)
                    + "\n"
                )
                print("CYCLE_START", args.gpu, variant, level, repeat, flush=True)
                with (destination / "console.log").open("w") as sim_log:
                    code = run_worker_process(
                        sim_command,
                        cwd=destination,
                        env=sim_env,
                        stdout=sim_log,
                        stderr=subprocess.STDOUT,
                        on_wait=monitor,
                    )
                if code:
                    raise RuntimeError(f"Official evaluator exited {code}; see {sim_log.name}")
                # Kit close 可能提前结束 Python，退出码 0 不能代替实际 episode 记录。
                stats_path = destination / "stats/eval_stats.txt"
                stats = read_stats(stats_path, args.num_episodes)
                if set(stats) != set(range(args.num_episodes)):
                    raise RuntimeError(
                        f"Incomplete official results: {sorted(stats)}; {stats_path}"
                    )
                (destination / "verified_results.json").write_text(
                    json.dumps(
                        {"episodes": args.num_episodes, "stats": stats, "source": str(stats_path)},
                        indent=2,
                    )
                    + "\n"
                )
            (base / "COMPLETE.json").write_text(
                json.dumps(
                    {"episodes": args.num_episodes * args.num_repeats, "cycles": args.num_repeats}
                )
                + "\n"
            )
            print("WORKER_COMPLETE", args.gpu, variant, level, flush=True)
    finally:
        if server is not None:
            remember_tree(server.pid, tracked, process_table(), own_group=server.pid)
            cleanup(tracked)
            server.wait(timeout=40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--slot", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num-episodes", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--variant", choices=["STATE", "VGQ-AE", "NO-STATE"], required=True)
    parser.add_argument("--backbone", choices=["pi05", "gr00t"], default="pi05")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint-step", type=int, default=40000)
    parser.add_argument("--expected-horizon", type=int, default=30)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--success-criteria", type=float, default=0.4)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=ROOT / "SIMPLE-eval-20261103/snapshot-fb67cd9a",
        help="Explicit official evaluation package root; recorded in worker metadata",
    )
    parser.add_argument(
        "--level",
        choices=["dr-level-0", "dr-level-1", "dr-level-2", "dr-level-hard"],
        required=True,
    )
    parser.add_argument("--num-repeats", type=int, default=5)
    parser.add_argument("--video-rounds", type=int, default=2)
    parser.add_argument("--repair-scene-references", action="store_true")
    arguments = parser.parse_args()
    if arguments.video_rounds < 2:
        parser.error("Retain at least two complete video rounds")
    if (
        arguments.backbone != "pi05" or arguments.variant == "NO-STATE"
    ) and arguments.run_dir is None:
        parser.error("This backbone/variant requires an explicit --run-dir")
    worker(arguments)
