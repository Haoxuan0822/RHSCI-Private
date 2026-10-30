"""π0.5 RHSCI：统一配方、真实预检、完整断点和后台故障恢复。"""

import argparse
import os
from pathlib import Path

from rsci.data.cache_config import add_input_cache_arguments, resolve_input_cache
from rsci.data.state_noise import validate_scale
from rsci.paths import add_data_arguments, dataset_paths

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(p)
    add_input_cache_arguments(p, env_var="RSCI_INPUT_CACHE")
    p.add_argument(
        "--nccl-library",
        type=Path,
        help="Optional alternate NCCL library; otherwise use the environment runtime",
    )
    p.add_argument("--model", choices=("STATE", "NO-STATE", "VGQ-AE"), default="VGQ-AE")
    p.add_argument(
        "--state-measurement-noise-std-rad",
        type=float,
        nargs="?",
        const=0.02,
        default=0.0,
        help="Training noise before state normalization/tokenization; STATE only",
    )
    p.add_argument("--task", required=True)
    p.add_argument("--vlm", choices=("frozen",), default="frozen")
    p.add_argument(
        "--execution",
        choices=("reference", "history", "query", "both"),
        default="reference",
        help="执行方式验证；默认保留原路径，history合并历史batch，query关闭Query重算",
    )
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    p.add_argument("--batch-size", type=int, default=16, help="每卡微批量")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--run-tag", default="main")
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--seed", type=int, default=292285)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--check", action="store_true", help="真实数据/处理器/权重文件检查，不启动训练")
    p.add_argument(
        "--check-data", action="store_true", help="只检查RGB/动作/正式ARDY，不要求模型权重"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--guardian-fd", type=int, help=argparse.SUPPRESS)
    return p


def resolve(args):
    import re

    validate_scale(args.state_measurement_noise_std_rad, args.model)

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
    runtime = Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data"))
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    paths = dataset_paths(args)
    paths["input_cache"] = resolve_input_cache(args)
    paths.update(
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
    if args.output_dir is None and args.state_measurement_noise_std_rad:
        paths["output"] = paths["output"].with_name(
            paths["output"].name + f"-noise-{args.state_measurement_noise_std_rad:g}rad"
        )
    return paths


def simple_lr_multiplier(step, max_steps=40000):
    """与 SIMPLE train_pytorch.py 一致，返回学习率相对1e-4的倍率。"""
    import math

    if step < 1000:
        return 1 / 1001 + (1 - 1 / 1001) * step / 1000
    progress = min(1.0, (step - 1000) / max(1, max_steps - 1000))
    return 1e-4 + (1 - 1e-4) * 0.5 * (1 + math.cos(math.pi * progress))


def training_arguments(args, paths):
    """生产与CPU配方核验共用，优化配置不再套用Psi0参数。"""
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
