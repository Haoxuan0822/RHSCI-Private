"""PSI0 RHSCI 统一训练、只读预检和独立断点守护；不下载资产。"""

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from rhsci.data.state_noise import validate_scale
from rhsci.paths import CODE_ROOT
from rhsci.runtime.environment import get_env

from .controls import CONTROL_VARIANTS
from .defaults import build_training_arguments
from .model import VARIANTS

if TYPE_CHECKING:
    from psi.config.train.finetune_simple_psi0_config import DynamicLaunchConfig

CODE = CODE_ROOT


class TrainingPlan(NamedTuple):
    """已解析的原生配置与启动计划；保留旧四元组的索引和解包方式。"""

    config: "DynamicLaunchConfig"
    command: list[str]
    output: Path
    ranks: int


def build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行解析器并读取环境默认值；返回解析器，不加载模型或启动训练。"""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--model", choices=(*VARIANTS, *CONTROL_VARIANTS), required=True, help="带星号的方案请用引号，例如 'VGQ-VLM*'"
    )
    argument_parser.add_argument("--execution", choices=("reference", "logits", "both"), default="reference", help="VGQ-VLM execution only; both also disables VLM checkpointing")
    argument_parser.add_argument("--task", required=True, help="本地 SIMPLE repo/task 名")
    argument_parser.add_argument("--vlm", choices=("frozen", "lora"), default="frozen")
    argument_parser.add_argument("--gpus", default=get_env("CUDA_VISIBLE_DEVICES") or "0")
    argument_parser.add_argument("--port", type=int)
    argument_parser.add_argument("--data-root", type=Path)
    argument_parser.add_argument("--sidecar-root", type=Path)
    argument_parser.add_argument("--vlm-path", type=Path)
    argument_parser.add_argument("--ae-path", type=Path)
    argument_parser.add_argument("--dino-path", type=Path)
    argument_parser.add_argument("--batch-size", type=int, default=16, help="每卡 batch size")
    argument_parser.add_argument(
        "--no-reader", action="store_true", help="Use full visual tokens for query memory"
    )
    argument_parser.add_argument(
        "--ardy-offset", type=int, default=0, help="Future ARDY target offset in 50 Hz frames"
    )
    argument_parser.add_argument(
        "--state-measurement-noise-std-rad",
        type=float,
        nargs="?",
        const=0.02,
        default=0.0,
        help="Training noise before normalization; flag alone selects 0.02 rad; STATE only",
    )
    argument_parser.add_argument("--num-workers", type=int, default=8)
    argument_parser.add_argument("--max-steps", type=int, default=40000)
    argument_parser.add_argument("--output-dir", type=Path)
    argument_parser.add_argument(
        "--nccl-library",
        type=Path,
        help="Optional validated NCCL shared library; otherwise use the environment runtime",
    )
    argument_parser.add_argument("--seed", type=int, default=292285)
    argument_parser.add_argument(
        "--run-tag", default="main", help="源码/超参数改变时选择新的独立运行名"
    )
    argument_parser.add_argument(
        "--resume", type=Path, help="完整 RHSCI checkpoint；默认自动查找同组最新完整断点"
    )
    modes = argument_parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run", action="store_true", help="解析并打印配置，不加载样本/模型或启动训练"
    )
    modes.add_argument(
        "--check", action="store_true", help="检查本地模型、数据与正式 ARDY 标签，不启动训练"
    )
    argument_parser.add_argument("--guardian-fd", type=int, help=argparse.SUPPRESS)
    return argument_parser


def resolve_training_plan(args: argparse.Namespace) -> TrainingPlan:
    """校验 CLI 参数并应用 Ψ0 配方；返回配置、启动命令、输出目录和进程数。"""
    noise = validate_scale(args.state_measurement_noise_std_rad, args.model)
    if args.model in CONTROL_VARIANTS and (args.no_reader or args.ardy_offset):
        raise ValueError("Controls use their fixed Reader/target-time protocol")
    if args.execution != "reference" and args.model != "VGQ-VLM":
        raise ValueError("Execution profiles require VGQ-VLM")
    if args.ardy_offset < 0 or (
        (args.ardy_offset or args.no_reader) and args.model in ("STATE", "NO-STATE")
    ):
        raise ValueError("Future ARDY and --no-reader require an RHSCI Query variant")
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", args.task)
        or not re.fullmatch(r"\d+(,\d+)*", args.gpus)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag)
    ):
        raise ValueError(
            "task must be a simple repo name; gpus must be comma-separated device indices"
        )
    if (
        len(set(args.gpus.split(","))) != len(args.gpus.split(","))
        or args.batch_size < 1
        or args.num_workers < 1
        or args.max_steps < 1
    ):
        raise ValueError("GPU indices must be unique, batch-size and num-workers positive")
    runtime = Path(get_env("PSI_RUNTIME_ROOT", "/opt/rsci-data")).resolve()
    model_slug = args.model.replace("*", "-star")
    if args.execution != "reference":
        model_slug += "-" + args.execution
    if args.no_reader:
        model_slug += "-no-reader"
    if args.ardy_offset:
        model_slug += f"-F{args.ardy_offset}"
    if noise:
        model_slug += f"-noise-{noise:g}rad"
    group = f"{args.task}/{model_slug}/{args.vlm}/seed-{args.seed}/{args.run_tag}"
    output = args.output_dir or runtime / "runs/psi0" / group
    local = runtime / "cache/checkpoints/psi0"
    updates = {
        "exp": f"RHSCI-{args.task}-{args.model.replace('*', '-star')}-{args.vlm}-{args.run_tag}",
        "seed": args.seed,
        "train.output-dir": output,
        "train.train_batch_size": args.batch_size,
        "train.max_training_steps": args.max_steps,
        "train.num-workers": args.num_workers,
        "log.report_to": "None",
        # Reuse the validated optimizer and transfer settings.
        "train.val-num-workers": args.num_workers,
        "train.prefetch-factor": 2,
        "data.root_dir": args.data_root or runtime / "data/simple",
        "data.train-repo-ids": args.task,
        "model.model_name_or_path": args.vlm_path
        or local / "pre.fast.1by1.2601091803.ckpt.ego200k.he30k",
        "model.pretrained-action-header-path": args.ae_path
        or local / "postpre.1by1.pad36.2601131206.ckpt.he30k",
        "model.rsci-variant": args.model,
        "model.rsci-vlm-mode": args.vlm,
        "model.state-measurement-noise-std-rad": noise,
        "data.transform.field.state-measurement-noise-std-rad": noise,
        "data.rsci-ardy-offset-frames": args.ardy_offset,
    }
    if args.model in CONTROL_VARIANTS:
        updates["model.rsci-execution"] = args.execution
    query = args.model not in ("STATE", "NO-STATE")
    if query:
        updates["data.rsci-ardy-sidecar-root"] = (
            args.sidecar_root or runtime / "data/simple-ardy-sidecars" / args.task
        )
    if args.model.startswith("EGQ-"):
        updates["model.rsci-rgb-encoder-path"] = (
            args.dino_path or runtime / "cache/checkpoints/dinov2-small"
        )
    # Apply the model, dataset and output choices to the native training defaults.
    old = build_training_arguments(runtime)
    values = [old[0]]
    excluded = {"--model.no-tune-vlm", "--train.skip-resumed-steps"}
    for item in old[1:]:
        key = item.split("=", 1)[0].removeprefix("--")
        if (
            item in excluded
            or key in updates
            or key.startswith("data.rsci-ardy-sidecar-root")
            or key.startswith("train.resume-from-checkpoint")
        ):
            continue
        values.append(item)
    updates["data.transform.repack.rsci-variant"] = args.model
    values += [f"--{key}={value}" for key, value in updates.items()]
    if args.no_reader:
        values.append("--model.no-rsci-use-reader")
    values += ["--train.optimizer-fused", "--train.pin-memory", "--train.non-blocking-transfer"]
    # 严禁环境中旧实验的 resume 隐式接入当前实验。
    import tyro

    from psi.config.train.finetune_simple_psi0_config import DynamicLaunchConfig

    cfg = tyro.cli(
        DynamicLaunchConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=values[1:]
    )
    cfg.data.transform.repack.rsci_variant = args.model
    if cfg.auto_tag_run:
        raise ValueError("RHSCI must not auto-commit unrelated workspace edits")
    ranks = len(args.gpus.split(","))
    port = args.port or 29600 + int(hashlib.sha256(group.encode()).hexdigest()[:4], 16) % 1000
    if not 1024 <= port <= 65535:
        raise ValueError("Invalid distributed port")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={ranks}",
        f"--master_port={port}",
        str(CODE / "scripts/train.py"),
        *values,
    ]
    if args.nccl_library:
        command = [
            sys.executable,
            str(CODE / "scripts/with_nccl_runtime.py"),
            "--library",
            str(args.nccl_library),
            "--",
            *command,
        ]
    return TrainingPlan(cfg, command, output, ranks)
