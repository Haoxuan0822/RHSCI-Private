"""egovla: 启动身份与源码契约。"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from rsci.paths import CODE_ROOT

from .checkpoints import sha256

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。
CODE = CODE_ROOT


def build_training_contract(args, paths, manifest):
    """根据已解析参数、数据与源码生成恢复身份；调用方据此拒绝不兼容续训。返回契约字典。"""
    from rsci.data.ardy_sidecar import dataset_content_fingerprint
    from rsci.data.input_cache import cache_identity

    sources = sorted((CODE / "src/egovla/human_plan/rsci").glob("*.py"))
    sources += sorted((CODE / "src/egovla").rglob("*.py"))
    sources += [
        CODE / "scripts/train/egovla.sh",
        CODE / "src/rsci/paths.py",
        CODE / "src/egovla/pyproject.toml",
        CODE / "src/egovla/uv.lock",
    ]
    sources += sorted((CODE / "src/rsci/runtime").glob("*.py"))
    sources += sorted((CODE / "src/rsci/data").glob("*.py"))
    sources = sorted(set(sources))
    return dict(
        schema="egovla-vgq-ae-v1",
        task=paths["task"],
        model=args.model,
        vlm=args.vlm_training,
        dataset=str(paths["dataset"]),
        dataset_fingerprint=dataset_content_fingerprint(paths["dataset"]),
        input_cache=cache_identity(paths["dataset"], config=paths["input_cache"]),
        sidecar_manifest=sha256(paths["sidecar"] / "manifest.json")
        if args.model == "VGQ-AE"
        else None,
        initialization=str(paths["checkpoint"]),
        assets=manifest,
        batch=args.batch_size,
        accumulation=args.gradient_accumulation_steps,
        world=len(args.gpus.split(",")),
        workers=args.num_workers,
        seed=args.seed,
        steps=args.max_steps,
        epochs=args.num_train_epochs,
        attention=dict(language="sdpa", vision="sdpa"),
        gradient_checkpointing=dict(
            enabled=args.gradient_checkpointing and args.vlm_training == "full",
            use_reentrant=False,
            scope="language-and-vision",
        ),
        execution=dict(
            optimizer="adamw_torch_fused",
            pin_memory=True,
            non_blocking=True,
            gradient_as_bucket_view=True,
            cuda_allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers else None,
            nccl_shm_disable=os.environ.get("NCCL_SHM_DISABLE", "0"),
            nccl_library=os.environ.get("RSCI_NCCL_LIBRARY"),
            nccl_library_sha256=sha256(Path(os.environ["RSCI_NCCL_LIBRARY"]))
            if os.environ.get("RSCI_NCCL_LIBRARY")
            else None,
        ),
        sampler="accelerate-seedable-epoch-shuffle-v2-preserve-skip-epoch",
        optimizer=dict(
            lr=2e-5,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0.0,
            clip=1.0,
            scheduler="constant",
            warmup_ratio=0.03,
            warmup_effective=False,
        ),
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "accelerate", "torchcodec")
        },
        code={str(path.relative_to(CODE)): sha256(path) for path in sources},
    )


def compute_contract_digest(value):
    """按稳定 JSON 顺序计算训练契约 SHA256，返回用于完整断点校验的摘要。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
