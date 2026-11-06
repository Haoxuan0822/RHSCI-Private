"""egovla: 启动身份与源码契约。"""

import os
from pathlib import Path

from rhsci.data.ardy_motion import motion_config
from rhsci.paths import CODE_ROOT as CODE_ROOT
from rhsci.runtime.contracts import compute_contract_digest as compute_contract_digest
from rhsci.runtime.contracts import reuse_run_contract
from rhsci.runtime.environment import get_env

from .checkpoints import sha256

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。
CODE = CODE_ROOT


def build_training_contract(args, paths, manifest):
    """根据模型、数据与配方生成恢复身份；源码编辑和依赖版本不作为恢复门槛。返回契约字典。"""
    from rhsci.data.ardy_sidecar import dataset_content_fingerprint
    from rhsci.data.input_cache import cache_identity

    return reuse_run_contract(
        dict(
            schema="egovla-vgq-ae-v1",
            **motion_config(args),
            task=paths["task"],
            model=args.model,
            vlm=args.vlm_training,
            dataset=str(paths["dataset"]),
            dataset_fingerprint=dataset_content_fingerprint(paths["dataset"]),
            input_cache=cache_identity(paths["dataset"], config=paths["input_cache"]),
            sidecar_manifest=sha256(paths["sidecar"] / "manifest.json")
            if args.model in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-Mo")
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
                cuda_allocator=get_env("PYTORCH_CUDA_ALLOC_CONF"),
                persistent_workers=args.num_workers > 0,
                prefetch_factor=2 if args.num_workers else None,
                nccl_shm_disable=get_env("NCCL_SHM_DISABLE", "0"),
                nccl_library=get_env("RHSCI_NCCL_LIBRARY"),
                nccl_library_sha256=sha256(Path(os.environ["RHSCI_NCCL_LIBRARY"]))
                if get_env("RHSCI_NCCL_LIBRARY")
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
        ),
        paths["output"],
    )
