"""官方 λ₀ SIMPLE 后训练入口；额外提供只读检查及 VGQ-AE。"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

from rhsci.data.ardy_motion import add_motion_arguments, motion_config
from rhsci.paths import add_data_arguments, dataset_paths
from rhsci.runtime.environment import get_env

from .contracts import RUNTIME


class LambdaTrainingSettings(TypedDict):
    """交给原生 resolver 的训练覆盖项；其他配方继续由原生默认值提供。"""

    iters: int
    num_domains: int
    trainable_action_space_ids: list[int]
    save_interval: int
    num_workers: int


class LambdaTrainingConfig(TypedDict):
    """可直接序列化的原生配置；字段名和 JSON 形状保持兼容。"""

    stage: str
    initialization: str
    midtrain_weight_source: str
    prediction_type: str
    model: str
    data: str
    data_root: str
    output: str
    seed: int
    batch_size: int
    global_batch_size: int
    gradient_accumulation_steps: int
    resume: str
    training: LambdaTrainingSettings


def build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行解析器并读取环境默认值；返回解析器，不加载模型或启动训练。"""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(argument_parser)
    add_motion_arguments(argument_parser)
    argument_parser.add_argument("--task", required=True)
    argument_parser.add_argument("--bundle", type=Path, help="Prepared RHSCI lambda0 data bundle")
    argument_parser.add_argument(
        "--data-root", type=Path, help="Root used to resolve bundle metadata paths"
    )
    argument_parser.add_argument("--max-steps", type=int, default=40000)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--variant", choices=("STATE", "VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480"), default="VGQ-AE")
    argument_parser.add_argument("--runtime-root", type=Path, default=RUNTIME)
    argument_parser.add_argument(
        "--checkpoint", type=Path, help="本地官方 Stage-II raw checkpoint，不自动下载"
    )
    argument_parser.add_argument("--gpus", default=get_env("CUDA_VISIBLE_DEVICES") or "0")
    argument_parser.add_argument("--batch-size", type=int, default=4)
    argument_parser.add_argument("--global-batch-size", type=int, default=16)
    argument_parser.add_argument("--num-workers", type=int, default=4)
    argument_parser.add_argument("--run-tag", default="main")
    argument_parser.add_argument("--resume", choices=("none", "auto"), default="auto")
    argument_parser.add_argument(
        "--check", action="store_true", help="只读配置与路径检查，不加载模型"
    )
    argument_parser.add_argument(
        "--check-data", action="store_true", help="真实数据/官方归一化/四帧检查，不训练"
    )
    argument_parser.add_argument(
        "--check-model", action="store_true", help="CPU严格加载真实Stage-II与处理器，不训练"
    )
    argument_parser.add_argument(
        "--dry-run", action="store_true", help="官方完整解析并验证资产/数据，不训练"
    )
    argument_parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return argument_parser


def build_training_config(args: argparse.Namespace) -> LambdaTrainingConfig:
    """校验 CLI 参数；返回供原生 λ₀ resolver 使用的训练配置。"""
    motion_config(args)
    if Path(args.run_tag).name != args.run_tag or args.run_tag in ("", ".", ".."):
        raise ValueError("run-tag must be a plain directory name")
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus) or any(not gpu.isdecimal() for gpu in gpus):
        raise ValueError("gpus must contain distinct integer IDs")
    from lambda0.training.runner import batch_layout

    accumulation = batch_layout(args.global_batch_size, args.batch_size, len(gpus))
    if args.num_workers < 0 or args.max_steps < 1:
        raise ValueError("num-workers must be nonnegative and max-steps positive")
    dataset_paths(
        SimpleNamespace(
            task=args.task, dataset=args.dataset, sidecar=args.sidecar, model=args.variant
        )
    )
    output = (
        args.output_dir
        or args.runtime_root / "runs/lambda0" / args.task / args.variant / args.run_tag
    )
    bundle = args.bundle or args.runtime_root / "bundles/lambda0" / f"{args.task}-safe"
    model = args.checkpoint or args.runtime_root / "cache/checkpoints/lambda0-midtrain-100k"
    domains = 30
    if (model / "config.json").is_file():
        domains = max(8, int(json.loads((model / "config.json").read_text())["num_domains"]))
    config = LambdaTrainingConfig(
        stage="posttrain",
        initialization="midtrain",
        midtrain_weight_source="raw",
        prediction_type="velocity",
        model=str(model),
        data=str(bundle),
        data_root=str(args.data_root or args.runtime_root / "data"),
        output=str(output),
        seed=args.seed,
        batch_size=args.batch_size,
        global_batch_size=args.global_batch_size,
        gradient_accumulation_steps=accumulation,
        resume=args.resume,
        training=LambdaTrainingSettings(
            iters=args.max_steps,
            num_domains=domains,
            trainable_action_space_ids=[3],
            save_interval=1000,
            num_workers=args.num_workers,
        ),
    )
    return config
