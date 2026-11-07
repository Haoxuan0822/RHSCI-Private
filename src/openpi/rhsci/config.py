"""π0.5 RHSCI：统一配方、真实预检、完整断点和后台故障恢复。"""

import argparse
from pathlib import Path

from rhsci.data.ardy_motion import add_motion_arguments, motion_config
from rhsci.data.cache_config import add_input_cache_arguments, resolve_input_cache
from rhsci.data.multitask import add_multitask_arguments, multitask_paths
from rhsci.paths import TrainingPaths, add_data_arguments
from rhsci.runtime.environment import get_env


class Pi05TrainingPaths(TrainingPaths):
    """π0.5 在公共训练路径之外需要初始化权重与 SentencePiece 词表。"""

    checkpoint: Path
    tokenizer: Path


def build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行解析器并读取环境默认值；返回解析器，不加载模型或启动训练。"""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(argument_parser)
    add_multitask_arguments(argument_parser)
    argument_parser.add_argument(
        "--statistics-batch-size",
        type=int,
        default=256,
        help="Joint RunningStats batch size (official SIMPLE-mix-33: 256); final partial batch is dropped",
    )
    add_motion_arguments(argument_parser)
    add_input_cache_arguments(argument_parser, env_var="RHSCI_INPUT_CACHE")
    argument_parser.add_argument(
        "--nccl-library",
        type=Path,
        help="Optional alternate NCCL library; otherwise use the environment runtime",
    )
    argument_parser.add_argument(
        "--model",
        choices=("STATE", "NO-STATE", "VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480"),
        default="VGQ-AE",
    )
    argument_parser.add_argument("--task", required=True)
    argument_parser.add_argument("--vlm", choices=("frozen",), default="frozen")
    argument_parser.add_argument(
        "--execution",
        choices=("reference", "history", "query", "both"),
        default="reference",
        help="执行方式验证；默认保留原路径，history合并历史batch，query关闭Query重算",
    )
    argument_parser.add_argument("--gpus", default=get_env("CUDA_VISIBLE_DEVICES") or "0")
    argument_parser.add_argument("--batch-size", type=int, default=16, help="每卡微批量")
    argument_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    argument_parser.add_argument("--num-workers", type=int, default=8)
    argument_parser.add_argument("--run-tag", default="main")
    argument_parser.add_argument("--max-steps", type=int, default=40000)
    argument_parser.add_argument("--seed", type=int, default=292285)
    argument_parser.add_argument("--tokenizer", type=Path)
    argument_parser.add_argument("--checkpoint", type=Path)
    argument_parser.add_argument("--resume", type=Path)
    argument_parser.add_argument(
        "--check", action="store_true", help="真实数据/处理器/权重文件检查，不启动训练"
    )
    argument_parser.add_argument(
        "--check-data", action="store_true", help="只检查RGB/动作/正式ARDY，不要求模型权重"
    )
    argument_parser.add_argument("--dry-run", action="store_true")
    argument_parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    argument_parser.add_argument("--guardian-fd", type=int, help=argparse.SUPPRESS)
    return argument_parser


def resolve_training_paths(args: argparse.Namespace) -> Pi05TrainingPaths:
    """校验 CLI 参数并补全默认值；返回数据、缓存及输出路径，保留各骨干的环境设置。"""
    motion_config(args)
    import re

    if not re.fullmatch(r"\d+(,\d+)*", args.gpus) or len(set(args.gpus.split(","))) != len(
        args.gpus.split(",")
    ):
        raise ValueError("GPU list must contain unique device indices")
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag)
        or min(args.batch_size, args.gradient_accumulation_steps) < 1
        or args.num_workers < 0
    ):
        raise ValueError("Invalid run-tag/batch/accumulation/workers")
    runtime = Path(get_env("PSI_RUNTIME_ROOT", "/opt/rsci-data"))
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.statistics_batch_size < 1:
        raise ValueError("statistics-batch-size must be positive")
    paths = Pi05TrainingPaths(
        **multitask_paths(args),
        input_cache=resolve_input_cache(args),
        checkpoint=args.checkpoint or runtime / "cache/checkpoints/openpi/pi05_droid",
        tokenizer=args.tokenizer or runtime / "assets/paligemma_tokenizer.model",
        output=args.output_dir
        or runtime
        / "runs/pi05"
        / args.task
        / args.model
        / args.vlm
        / f"seed-{args.seed}"
        / args.execution
        / args.run_tag,
    )
    if "datasets" in paths:
        paths["statistics_batch_size"] = args.statistics_batch_size
    return paths


def simple_lr_multiplier(step, max_steps=40000):
    """与 SIMPLE train_pytorch.py 一致，返回学习率相对1e-4的倍率。"""
    import math

    if step < 1000:
        return 1 / 1001 + (1 - 1 / 1001) * step / 1000
    progress = min(1.0, (step - 1000) / max(1, max_steps - 1000))
    return 1e-4 + (1 - 1e-4) * 0.5 * (1 + math.cos(math.pi * progress))


def build_training_arguments(args: argparse.Namespace, paths: Pi05TrainingPaths):
    """从已解析参数和输出路径构造本骨干原生 TrainingArguments，返回参数对象。

    生产与CPU配方核验共用，优化配置不再套用Psi0参数。"""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(paths["output"]),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=1e-4,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        weight_decay=1e-10,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1,
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        save_steps=1000,
        save_total_limit=None,
        save_safetensors=True,
        logging_steps=10,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers else None,
        accelerator_config={"non_blocking": True},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )
