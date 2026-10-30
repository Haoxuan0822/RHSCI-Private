"""SIMPLE ACT配方；统一STATE/NO-STATE/EGQ-AE，独立完整恢复与数值门禁。"""

import argparse
import os
import re
from pathlib import Path

from rsci.data.cache_config import add_input_cache_arguments, resolve_input_cache
from rsci.paths import add_data_arguments, dataset_paths

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(p)
    add_input_cache_arguments(p, env_var="RSCI_ACT_SPACE_CACHE")
    p.add_argument("--task", required=True)
    p.add_argument(
        "--torch-home",
        type=Path,
        help="External Torch cache containing the ImageNet ResNet18 weights",
    )
    p.add_argument("--nccl-library", type=Path, help="Optional explicitly validated NCCL runtime")
    p.add_argument("--model", choices=("STATE", "NO-STATE", "EGQ-AE"), default="EGQ-AE")
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--run-tag", default="main")
    p.add_argument("--check-data", action="store_true")
    p.add_argument(
        "--check", action="store_true", help="随机初始化全尺寸CPU检查，不下载ImageNet权重、不训练"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--guardian", action="store_true", help=argparse.SUPPRESS)
    return p


def resolve(args):
    if (
        not re.fullmatch(r"\d+(,\d+)*", args.gpus)
        or len(set(args.gpus.split(","))) != len(args.gpus.split(","))
        or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag)
        or min(args.batch_size, args.gradient_accumulation_steps, args.max_steps) < 1
        or args.num_workers < 0
    ):
        raise ValueError("Invalid GPU list, tag, batch, accumulation, steps or workers")
    runtime = Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data")).resolve()
    paths = dataset_paths(args)
    paths["input_cache"] = resolve_input_cache(args)
    paths["output"] = (
        args.output_dir
        or runtime / "runs/act" / args.task / args.model / f"seed-{args.seed}" / args.run_tag
    )
    args.torch_home = (
        args.torch_home or Path(os.environ.get("TORCH_HOME", str(runtime / "cache/act/torch")))
    ).resolve()
    os.environ["TORCH_HOME"] = str(args.torch_home)
    return paths


def training_arguments(args, paths, *, cpu=False):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(paths["output"]),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=1e-4,
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=1e-6,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs={},
        max_grad_norm=0.0,
        bf16=not cpu,
        tf32=False,
        use_cpu=cpu,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=not cpu,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers else None,
        dataloader_drop_last=True,
        accelerator_config={"use_seedable_sampler": False, "non_blocking": not cpu},
        ddp_find_unused_parameters=False,
        save_steps=1000,
        save_total_limit=None,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=1,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )
