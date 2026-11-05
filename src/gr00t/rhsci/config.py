"""GR00T RHSCI训练worker：官方模型/处理器，map数据集，完整HF断点。"""

import argparse
from pathlib import Path

from rhsci.data.ardy_motion import add_motion_arguments, motion_config
from rhsci.data.cache_config import add_input_cache_arguments, resolve_input_cache
from rhsci.paths import CheckpointTrainingPaths, add_data_arguments, dataset_paths
from rhsci.runtime.environment import get_env


def build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行解析器并读取环境默认值；返回解析器，不加载模型或启动训练。"""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(argument_parser)
    add_motion_arguments(argument_parser)
    add_input_cache_arguments(argument_parser, env_var="RHSCI_GR00T_INPUT_CACHE")
    argument_parser.add_argument(
        "--nccl-library",
        type=Path,
        help="Optional alternate NCCL library; otherwise use the environment runtime",
    )
    argument_parser.add_argument(
        "--model", choices=("STATE", "NO-STATE", "VGQ-AE", "VGQ-AE-Mo"), default="VGQ-AE"
    )
    argument_parser.add_argument("--task", required=True)
    argument_parser.add_argument(
        "--vlm",
        choices=("top4", "frozen"),
        default="top4",
        help="SIMPLE默认训练顶部4层；frozen为显式消融",
    )
    argument_parser.add_argument("--gpus", default=get_env("CUDA_VISIBLE_DEVICES") or "0")
    argument_parser.add_argument(
        "--batch-size", type=int, default=None, help="默认按卡数和累计自动得到全局batch24"
    )
    argument_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="每次优化更新累计的微批次数；改变后使用新run-tag",
    )
    argument_parser.add_argument("--num-workers", type=int, default=4)
    argument_parser.add_argument("--run-tag", default="main")
    argument_parser.add_argument("--max-steps", type=int, default=20000)
    argument_parser.add_argument("--seed", type=int, default=292285)
    argument_parser.add_argument(
        "--checkpoint", type=Path, help="官方初始化目录；默认数据盘GR00T-N1.6-3B"
    )
    argument_parser.add_argument("--resume", type=Path, help="本实验完整checkpoint；默认自动寻找")
    argument_parser.add_argument(
        "--check", action="store_true", help="真实数据/CPU处理器预检，不加载完整模型"
    )
    argument_parser.add_argument(
        "--check-data",
        action="store_true",
        help="仅真实RGB/动作/正式ARDY及episode身份检查，无需Eagle依赖",
    )
    argument_parser.add_argument("--dry-run", action="store_true", help="仅打印参数和路径")
    argument_parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    argument_parser.add_argument("--guardian-fd", type=int, help=argparse.SUPPRESS)
    return argument_parser


def resolve_training_paths(args: argparse.Namespace) -> CheckpointTrainingPaths:
    """校验 CLI 参数并补全默认值；返回数据、缓存及输出路径，保留各骨干的环境设置。"""
    motion_config(args)
    import re

    if not re.fullmatch(r"\d+(,\d+)*", args.gpus) or len(set(args.gpus.split(","))) != len(
        args.gpus.split(",")
    ):
        raise ValueError("--gpus must list unique comma-separated device indices")
    if args.batch_size is None:
        divisor = len(args.gpus.split(",")) * args.gradient_accumulation_steps
        if divisor < 1 or 24 % divisor:
            raise ValueError("Global batch 24 must be divisible by world_size × accumulation")
        args.batch_size = 24 // divisor
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag)
        or args.batch_size < 1
        or args.num_workers < 0
        or args.gradient_accumulation_steps < 1
    ):
        raise ValueError("Invalid run-tag/batch/workers")
    runtime = Path(get_env("PSI_RUNTIME_ROOT", "/opt/rsci-data"))
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    paths = CheckpointTrainingPaths(
        **dataset_paths(args),
        input_cache=resolve_input_cache(args),
        checkpoint=args.checkpoint or runtime / "cache/checkpoints/GR00T-N1.6-3B",
        output=args.output_dir
        or runtime
        / "runs/gr00t"
        / args.task
        / args.model
        / args.vlm
        / f"seed-{args.seed}"
        / args.run_tag,
    )
    return paths


def optimization_kwargs():
    """从原生训练配置和当前锁定HF版本取值，不复用Psi0的优化参数。"""
    from transformers import TrainingArguments

    from gr00t.configs.training.training_config import TrainingConfig

    native = TrainingConfig()
    values = {
        key: getattr(native, key)
        for key in (
            "learning_rate",
            "lr_scheduler_type",
            "weight_decay",
            "warmup_ratio",
            "warmup_steps",
            "max_grad_norm",
            "optim",
        )
    }
    # 官方experiment没有覆盖这些值，因此沿用同版本HF TrainingArguments默认值。
    values.update(
        {
            key: TrainingArguments.__dataclass_fields__[key].default
            for key in ("adam_beta1", "adam_beta2", "adam_epsilon")
        }
    )
    return values


def build_training_arguments(
    args: argparse.Namespace, paths: CheckpointTrainingPaths, *, validation_steps=0
):
    """从已解析参数和输出路径构造本骨干原生 TrainingArguments，返回参数对象。

    生产和CPU回归共用同一TrainingArguments构造入口。"""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(paths["output"]),
        max_steps=validation_steps or args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        **optimization_kwargs(),
        bf16=True,
        tf32=True,
        logging_steps=1 if validation_steps else 10,
        save_steps=1000,
        save_strategy="no" if validation_steps else "steps",
        save_total_limit=None,
        save_safetensors=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers else None,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
        ignore_data_skip=False,
        report_to="none",
        accelerator_config={"non_blocking": True},
    )
