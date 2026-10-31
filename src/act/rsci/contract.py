"""act: 启动身份与源码契约。"""

import hashlib
import json

from rsci.paths import CODE_ROOT as CODE_ROOT
from rsci.runtime.contracts import reuse_run_contract

from .checkpoints import sha256
from .preflight import initialization, nccl_runtime

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。

CODE = CODE_ROOT


def build_training_contract(args, paths):
    """根据模型、数据与配方生成恢复身份；源码编辑和依赖版本不作为恢复门槛。返回契约字典。"""
    from rsci.data.ardy_sidecar import dataset_content_fingerprint
    from rsci.data.raw_input_cache import cache_identity

    return reuse_run_contract(
        dict(
            schema="act-rsci-v1",
            input_cache=cache_identity(
                paths["dataset"],
                dataset_content_fingerprint(paths["dataset"]),
                config=paths["input_cache"],
            ),
            variant=args.model,
            task=paths["task"],
            dataset=str(paths["dataset"]),
            fingerprint=dataset_content_fingerprint(paths["dataset"]),
            sidecar_manifest=sha256(paths["sidecar"] / "manifest.json")
            if args.model == "EGQ-AE"
            else None,
            initialization=initialization(),
            stats=sha256(paths["dataset"] / "meta/stats_psi0.json"),
            normalization=json.loads((paths["dataset"] / "meta/stats_psi0.json").read_text()),
            batch=args.batch_size,
            accumulation=args.gradient_accumulation_steps,
            world=len(args.gpus.split(",")),
            workers=args.num_workers,
            seed=args.seed,
            steps=args.max_steps,
            optimizer=dict(
                lr=1e-4,
                betas=[0.95, 0.999],
                eps=1e-8,
                weight_decay=1e-6,
                groups="all-parameters-one-group",
                fused=True,
                clip=None,
                warmup=1000,
                scheduler="cosine",
                scheduler_kwargs={},
            ),
            processing="native-act-bounds-nearest-center-color-v1-epoch-index-rng",
            execution=dict(
                nccl_runtime=nccl_runtime(),
                bf16=True,
                tf32=False,
                pin_memory=True,
                non_blocking=True,
                persistent_workers=args.num_workers > 0,
                prefetch_factor=2,
                ddp_gradient_as_bucket_view=True,
                flash_attention_extension=False,
            ),
        ),
        paths["output"],
    )


def compute_contract_digest(value):
    """按稳定 JSON 顺序计算训练契约 SHA256，返回用于完整断点校验的摘要。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
