"""SIMPLE EgoVLA VGQ-AE 训练入口：官方模型/配方，RHSCI 数据及安全恢复。"""

import argparse
import json
import os
import re
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
    add_input_cache_arguments(argument_parser, env_var="RHSCI_INPUT_CACHE")
    argument_parser.add_argument("--task", required=True)
    argument_parser.add_argument("--model", choices=("VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480", "STATE"), default="VGQ-AE")
    argument_parser.add_argument(
        "--vlm-training",
        choices=("full", "frozen"),
        default="full",
        help="默认按SIMPLE全量微调视觉/投影/语言；frozen仅作显式消融",
    )
    argument_parser.add_argument("--gpus", default=get_env("CUDA_VISIBLE_DEVICES") or "0")
    argument_parser.add_argument("--batch-size", type=int, default=16)
    argument_parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    argument_parser.add_argument("--num-workers", type=int, default=8)
    argument_parser.add_argument("--preset", choices=("simple-115epoch",))
    argument_parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="视觉/语言非重入重算；默认启用，关闭时使用独立输出目录",
    )
    length = argument_parser.add_mutually_exclusive_group()
    length.add_argument("--max-steps", type=int)
    length.add_argument("--num-train-epochs", type=int)
    argument_parser.add_argument("--seed", type=int, default=292285)
    argument_parser.add_argument("--checkpoint", type=Path)
    argument_parser.add_argument("--run-tag", default="main")
    argument_parser.add_argument(
        "--check-data", action="store_true", help="只核对正式数据，不需要 EgoVLA 权重/环境"
    )
    argument_parser.add_argument(
        "--check", action="store_true", help="完整CPU初始化和处理器预检，不训练"
    )
    argument_parser.add_argument("--dry-run", action="store_true")
    argument_parser.add_argument(
        "--register-assets", action="store_true", help="为已下载的本地基础权重生成SHA清单，不下载"
    )
    argument_parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    argument_parser.add_argument("--guardian", action="store_true", help=argparse.SUPPRESS)
    return argument_parser


def resolve_training_paths(args: argparse.Namespace) -> CheckpointTrainingPaths:
    """校验 CLI 参数并补全默认值；返回数据、缓存及输出路径，保留各骨干的环境设置。"""
    motion_config(args)
    if args.preset == "simple-115epoch":
        if args.max_steps is not None or args.num_train_epochs not in (None, 115):
            raise ValueError("simple-115epoch requires 115 epochs, without --max-steps")
        if (
            args.batch_size != 16
            or args.gradient_accumulation_steps != 4
            or len(args.gpus.split(",")) != 8
            or args.vlm_training != "full"
        ):
            raise ValueError("simple-115epoch requires full VLM, 8 GPUs, batch 16, accumulation 4")
        args.num_train_epochs = 115
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    if args.max_steps is None:
        args.max_steps = 40000
    if (
        not re.fullmatch(r"\d+(,\d+)*", args.gpus)
        or len(set(args.gpus.split(","))) != len(args.gpus.split(","))
        or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag)
        or min(args.batch_size, args.gradient_accumulation_steps, args.max_steps) < 1
        or args.num_workers < 0
    ):
        raise ValueError("Invalid GPU list, tag, batch, accumulation, steps or workers")
    runtime = Path(get_env("PSI_RUNTIME_ROOT", "/opt/rsci-data")).resolve()
    paths = CheckpointTrainingPaths(
        **dataset_paths(args),
        input_cache=resolve_input_cache(args),
        checkpoint=(
            args.checkpoint or runtime / "cache/checkpoints/EgoVLA/SIMPLE-pretrain"
        ).resolve(),
        output=args.output_dir
        or runtime
        / "runs/egovla"
        / args.task
        / args.model
        / args.vlm_training
        / f"seed-{args.seed}"
        / args.run_tag,
    )
    if args.num_train_epochs is not None:
        if args.num_train_epochs < 1:
            raise ValueError("Epochs must be positive")
        import math

        episodes = [
            json.loads(line)
            for line in (paths["dataset"] / "meta/episodes.jsonl").read_text().splitlines()
            if line
        ]
        samples = sum(max(0, e["length"] - 29) for e in episodes)
        batches = math.ceil(samples / (args.batch_size * len(args.gpus.split(","))))
        # HF4.53.2保留每个epoch末尾不足累计组的更新；不得丢弃尾部样本。
        args.max_steps = args.num_train_epochs * math.ceil(
            batches / args.gradient_accumulation_steps
        )
    return paths


def build_training_arguments(args: argparse.Namespace, paths: CheckpointTrainingPaths, cpu=False):
    """从已解析参数和输出路径构造本骨干原生 TrainingArguments，返回参数对象。"""
    from transformers import TrainingArguments

    # constant 不是 constant_with_warmup；原 SIMPLE 脚本虽然写了3%，实际不执行预热。
    return TrainingArguments(
        output_dir=str(paths["output"]),
        max_steps=-1 if args.num_train_epochs else (args.max_steps or 40000),
        num_train_epochs=args.num_train_epochs or 3,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        max_grad_norm=1.0,
        # 迁移Psi0执行加速，保留EgoVLA自身AdamW超参数；CPU验收使用非fused实现。
        optim="adamw_torch" if cpu else "adamw_torch_fused",
        bf16=not cpu,
        tf32=not cpu,
        use_cpu=cpu,
        gradient_checkpointing=args.gradient_checkpointing and args.vlm_training == "full",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=not cpu,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        remove_unused_columns=False,
        # 全量VLM时词表head不参与next_token_loss=0，与官方一样需要unused参数检测。
        ddp_find_unused_parameters=args.vlm_training == "full",
        save_steps=1000,
        save_total_limit=None,
        accelerator_config={"use_seedable_sampler": True, "non_blocking": not cpu},
        save_safetensors=True,
        logging_steps=1,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )
