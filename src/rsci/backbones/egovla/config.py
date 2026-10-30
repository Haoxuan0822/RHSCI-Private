"""SIMPLE EgoVLA VGQ-AE 训练入口：官方模型/配方，RHSCI 数据及安全恢复。"""

import argparse
import json
import os
import re
from pathlib import Path

from rsci.data.cache_config import add_input_cache_arguments, resolve_input_cache
from rsci.paths import add_data_arguments, dataset_paths

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(p)
    add_input_cache_arguments(p, env_var="RSCI_INPUT_CACHE")
    p.add_argument("--task", required=True)
    p.add_argument("--model", choices=("VGQ-AE", "STATE"), default="VGQ-AE")
    p.add_argument(
        "--vlm-training",
        choices=("full", "frozen"),
        default="full",
        help="默认按SIMPLE全量微调视觉/投影/语言；frozen仅作显式消融",
    )
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--preset", choices=("simple-115epoch",))
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="视觉/语言非重入重算；默认启用，关闭时使用独立输出目录",
    )
    length = p.add_mutually_exclusive_group()
    length.add_argument("--max-steps", type=int)
    length.add_argument("--num-train-epochs", type=int)
    p.add_argument("--seed", type=int, default=292285)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--run-tag", default="main")
    p.add_argument(
        "--check-data", action="store_true", help="只核对正式数据，不需要 EgoVLA 权重/环境"
    )
    p.add_argument("--check", action="store_true", help="完整CPU初始化和处理器预检，不训练")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--register-assets", action="store_true", help="为已下载的本地基础权重生成SHA清单，不下载"
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--guardian", action="store_true", help=argparse.SUPPRESS)
    return p


def resolve(args):
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
    runtime = Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data")).resolve()
    paths = dataset_paths(args)
    paths["input_cache"] = resolve_input_cache(args)
    paths.update(
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


def training_arguments(args, paths, cpu=False):
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
