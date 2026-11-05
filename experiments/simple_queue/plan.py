"""User-requested follow-up runs, reusing the existing SIMPLE evaluation worker."""

from pathlib import Path

ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
REPO = Path("/root/workspace/RHSCI-Private")
CURRENT = ROOT / "evals/pi05-between-30x30-5cycles-20261104"
PREPARATION = ROOT / "evals/followup-pi05-gr00t-5cycles-20261104/preparation"
OUTPUT = ROOT / "evals/gr00t-official-pre1102-5cycles-20261104"
EVALUATION_ROOT = ROOT / "eval-official-pre1102"
REVISION = "59a1ea14d4346496d4e55c9fa58ea2f3330fd298"
LEVELS = ("dr-level-0", "dr-level-1", "dr-level-2")
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

PI05_THREE_RUNS = [
    dict(
        backbone="pi05",
        variant=variant,
        task=task,
        step=40000,
        horizon=None,
        archive=f"checkpoint/PI05/{variant}/{task}/{recipe}/checkpoint-40000.tar.zst",
    )
    for task in ("oven", "handover", "faucet")
    for variant in ("STATE", "VGQ-AE")
    for recipe in [
        "pi05-three-tasks-20261029-v1"
        if (task, variant) == ("oven", "STATE")
        else "pi05-three-tasks-fast-20261029-v1"
    ]
]

# 用户已停止补下载；仅使用本地已经完整下载的指定GR00T归档。
DOWNLOAD_RUNS = []
ACTIVE_RUNS = [
    run
    for run in RUNS
    if run["backbone"] == "gr00t" and (ROOT / "hf/RHSCI/model-59a1ea14" / run["archive"]).is_file()
]


def run_key(run):
    return f"{run['backbone']}-{run['variant']}-{run['task']}"


def worker_command(run, gpu, level, run_dir, *, output_root=OUTPUT):
    """声明实际启动参数；模型、仿真与进程清理仍由已有 worker 完成。"""
    task, maximum, success = TASKS[run["task"]]
    return [
        str(ROOT / f".venv-{run['backbone']}-rsci/bin/python"),
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
        str(output_root / run_key(run)),
        "--evaluation-root",
        str(EVALUATION_ROOT),
        "--level",
        level,
        "--num-repeats",
        "5",
        "--video-rounds",
        "1",
        "--repair-scene-references",
    ]
