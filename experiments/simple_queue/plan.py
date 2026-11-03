"""User-requested follow-up runs, reusing the existing SIMPLE evaluation worker."""

from pathlib import Path

ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
REPO = Path("/root/workspace/RHSCI-Private")
CURRENT = ROOT / "evals/pi05-between-30x30-5cycles-20261104"
OUTPUT = ROOT / "evals/followup-pi05-gr00t-5cycles-20261104"
REVISION = "59a1ea14d4346496d4e55c9fa58ea2f3330fd298"
LEVELS = ("dr-level-0", "dr-level-1", "dr-level-2", "dr-level-hard")
TASKS = {
    "between": ("G1WholebodyLocomotionPickBetweenTablesTeleop-v0", 1200, 0.4),
    "oven": ("G1WholebodyOpenOvenTeleop-v0", 1000, 0.5),
    "handover": ("G1WholebodyHandoverTeleop-v0", 800, 0.9),
    "faucet": ("G1WholebodyOpenFaucetTeleop-v0", 1000, 0.2),
}
RUNS = [
    dict(
        backbone="pi05",
        variant="NO-STATE",
        task="between",
        step=40000,
        horizon=30,
        archive="checkpoint/PI05/NO-STATE/between/pi05-nostate-between-restart-20261028-v1/checkpoint-40000.tar.zst",
    ),
    *[
        dict(
            backbone="gr00t",
            variant=variant,
            task=task,
            step=20000,
            horizon=16,
            archive=f"checkpoint/Gr00t/{variant}/{task}/priority-rgb-reader-20261028-v1/checkpoint-20000.tar.zst",
        )
        for task in ("oven", "handover", "faucet")
        for variant in ("STATE", "VGQ-AE")
    ],
]

# 只续传烤箱两份；其他指定模型仅使用已经下载完整的归档。
DOWNLOAD_RUNS = [run for run in RUNS if run["task"] == "oven"]
ACTIVE_RUNS = [
    run
    for run in RUNS
    if run in DOWNLOAD_RUNS
    or (ROOT / "hf/RHSCI/model-59a1ea14" / run["archive"]).is_file()
]


def run_key(run):
    return f"{run['backbone']}-{run['variant']}-{run['task']}"


def worker_command(run, gpu, level, run_dir):
    """声明实际启动参数；模型、仿真与进程清理仍由已有 worker 完成。"""
    task, maximum, success = TASKS[run["task"]]
    return [
        str(ROOT / ".venv-pi05-rsci/bin/python"),
        "-u",
        str(REPO / "experiments/pi05_between/run_evaluation.py"),
        "--gpu",
        str(gpu),
        "--backbone",
        run["backbone"],
        "--variant",
        run["variant"],
        "--task",
        task,
        "--run-dir",
        str(run_dir),
        "--checkpoint-step",
        str(run["step"]),
        "--expected-horizon",
        str(run["horizon"]),
        "--max-episode-steps",
        str(maximum),
        "--success-criteria",
        str(success),
        "--output-root",
        str(OUTPUT / run_key(run)),
        "--level",
        level,
        "--num-repeats",
        "5",
    ]
