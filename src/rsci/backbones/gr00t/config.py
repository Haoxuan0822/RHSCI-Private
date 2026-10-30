"""GR00T RHSCI训练worker：官方模型/处理器，map数据集，完整HF断点。"""

import argparse
import os
from pathlib import Path

from rsci.data.cache_config import add_input_cache_arguments, resolve_input_cache
from rsci.paths import add_data_arguments, dataset_paths

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(p)
    add_input_cache_arguments(p, env_var="RSCI_GR00T_INPUT_CACHE")
    p.add_argument(
        "--nccl-library",
        type=Path,
        help="Optional alternate NCCL library; otherwise use the environment runtime",
    )
    p.add_argument("--model", choices=("STATE", "NO-STATE", "VGQ-AE"), default="VGQ-AE")
    p.add_argument("--task", required=True)
    p.add_argument(
        "--vlm",
        choices=("top4", "frozen"),
        default="top4",
        help="SIMPLE默认训练顶部4层；frozen为显式消融",
    )
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    p.add_argument(
        "--batch-size", type=int, default=None, help="默认按卡数和累计自动得到全局batch24"
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="每次优化更新累计的微批次数；改变后使用新run-tag",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--run-tag", default="main")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=292285)
    p.add_argument("--checkpoint", type=Path, help="官方初始化目录；默认数据盘GR00T-N1.6-3B")
    p.add_argument("--resume", type=Path, help="本实验完整checkpoint；默认自动寻找")
    p.add_argument("--check", action="store_true", help="真实数据/CPU处理器预检，不加载完整模型")
    p.add_argument(
        "--check-data",
        action="store_true",
        help="仅真实RGB/动作/正式ARDY及episode身份检查，无需Eagle依赖",
    )
    p.add_argument("--dry-run", action="store_true", help="仅打印参数和路径")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--guardian-fd", type=int, help=argparse.SUPPRESS)
    return p


def resolve(args):
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
    runtime = Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data"))
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    paths = dataset_paths(args)
    paths["input_cache"] = resolve_input_cache(args)
    paths.update(
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
    from gr00t.configs.training.training_config import TrainingConfig
    from transformers import TrainingArguments

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


def training_arguments(args, paths, *, validation_steps=0):
    """生产和CPU回归共用同一TrainingArguments构造入口。"""
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
