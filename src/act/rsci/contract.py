"""act: 启动身份与源码契约。"""

import hashlib
import importlib.metadata
import json

from rsci.paths import CODE_ROOT

from .checkpoints import sha256
from .preflight import initialization, nccl_runtime

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。

CODE = CODE_ROOT


def contract(args, paths):
    from rsci.data.ardy_sidecar import dataset_content_fingerprint

    files = list((CODE / "src/act/rsci").glob("*.py")) + [
        CODE / name
        for name in (
            "src/act/models/act.py",
            "src/psi/config/transform.py",
            "src/psi/config/augmentation.py",
            "src/psi/data/lerobot/compat.py",
            "src/rsci/data/ardy_sidecar.py",
            "src/psi/utils/utils.py",
            "src/rsci/runtime/hf_resume.py",
            "src/rsci/runtime/artifacts.py",
            "src/rsci/runtime/processes.py",
            "src/rsci/paths.py",
            "scripts/train/act_rsci.sh",
        )
    ]
    from rsci.data.raw_input_cache import cache_identity

    files += sorted((CODE / "src/rsci/data").glob("*.py"))
    files += sorted((CODE / "src/rsci/runtime").glob("*.py"))
    return dict(
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
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "accelerate",
                "lerobot",
                "torchcodec",
                "datasets",
                "av",
            )
        },
        code={str(path.relative_to(CODE)): sha256(path) for path in files},
    )


def digest_contract(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
