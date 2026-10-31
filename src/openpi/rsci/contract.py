"""pi05: 启动身份与源码契约。"""

import hashlib
import importlib.metadata
import json

from rsci.data.state_noise import STATE_NOISE_LOCATION
from rsci.paths import CODE_ROOT

from .checkpoints import sha256

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def build_training_contract(args, paths):
    """根据已解析参数、数据与源码生成恢复身份；调用方据此拒绝不兼容续训。返回 (契约字典, SHA256)。"""
    from rsci.data.ardy_sidecar import dataset_content_fingerprint
    from rsci.data.input_cache import cache_identity

    from .augmentation import AUGMENTATION_VERSION

    code = CODE_ROOT
    sources = list((code / "src/openpi/rsci").glob("*.py"))
    sources += list((code / "src/openpi/models_pytorch/transformers_replace").rglob("*.py"))
    sources += [
        code / "src/openpi/models_pytorch/gemma_pytorch.py",
        code / "src/rsci/data/ardy_sidecar.py",
        code / "src/openpi/models_pytorch/preprocessing_pytorch.py",
    ]
    sources += [
        code / "src/rsci/data/loader_cleanup.py",
        code / "scripts/with_nccl_runtime.py",
        code / "scripts/train/pi05.sh",
        code / "src/rsci/paths.py",
    ]
    sources += sorted((code / "src/rsci/runtime").glob("*.py"))
    sources += sorted((code / "src/rsci/data").glob("*.py"))
    sources = sorted(set(sources))
    identity = dict(
        schema=2,
        model=args.model,
        vlm=args.vlm,
        task=args.task,
        execution=args.execution,
        recipe="SIMPLE-pi05-default-v4",
        optimizer=dict(betas=[0.9, 0.95], eps=1e-8, weight_decay=1e-10, clip_norm=1),
        scheduler="SIMPLE-cosine-start-peak/1001-end-1e-8",
        inference_compile="max-autotune",
        batch=args.batch_size,
        accumulation=args.gradient_accumulation_steps,
        world_size=len(args.gpus.split(",")),
        seed=args.seed,
        workers=args.num_workers,
        steps=args.max_steps,
        lr=1e-4,
        warmup=1000,
        horizon=30,
        ardy_weight=0.1,
        preprocessing="RGB-bilinear-pad224-quantile-zero-span-clipped-v2",
        state_measurement_noise_std_rad=args.state_measurement_noise_std_rad,
        state_noise_location=STATE_NOISE_LOCATION,
        training_augmentation=AUGMENTATION_VERSION,
        dataset=dataset_content_fingerprint(paths["dataset"]),
        input_cache=cache_identity(paths["dataset"], config=paths["input_cache"]),
        initialization=str(paths["checkpoint"].resolve()),
        initialization_stat={
            key: getattr((paths["checkpoint"] / "model.safetensors").stat(), key)
            for key in ("st_size", "st_mtime_ns", "st_ctime_ns")
        },
        asset_manifest=sha256(paths["checkpoint"] / "RSCI_ASSETS.json"),
        tokenizer=sha256(paths["tokenizer"]),
        code={str(p.relative_to(code)): sha256(p) for p in sources},
        dependencies={
            key: importlib.metadata.version(key)
            for key in ("torch", "transformers", "accelerate", "torchcodec", "sentencepiece")
        },
    )
    if args.model == "VGQ-AE":
        identity["sidecar"] = sha256(paths["sidecar"] / "manifest.json")
    return identity, hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
