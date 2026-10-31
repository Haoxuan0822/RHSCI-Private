"""gr00t: 启动身份与源码契约。"""

import hashlib
import importlib.metadata
import json

from rsci.paths import CODE_ROOT

from .config import optimization_kwargs

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def build_training_contract(args, paths):
    """根据已解析参数、数据与源码生成恢复身份；调用方据此拒绝不兼容续训。返回 (契约字典, SHA256)。"""
    from rsci.data.ardy_sidecar import dataset_content_fingerprint
    from rsci.data.input_cache import cache_identity

    from .checkpoints import sha256

    code = CODE_ROOT
    sources = list((code / "src/gr00t/gr00t/rsci").glob("*.py"))
    sources += list((code / "src/gr00t/gr00t/model/gr00t_n1d6").glob("*.py"))
    sources += list(
        (code / "src/gr00t/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2").glob("*.py")
    )
    sources += [
        code / "src/gr00t/gr00t/model/modules/eagle_backbone.py",
        code / "src/gr00t/gr00t/model/modules/dit.py",
        code / "src/gr00t/gr00t/configs/model/gr00t_n1d6.py",
        code / "src/rsci/data/ardy_sidecar.py",
        code / "baselines/gr00t/requirements.txt",
        code / "src/gr00t/gr00t/configs/training/training_config.py",
        code / "scripts/train/gr00t.sh",
        code / "src/rsci/paths.py",
    ]
    sources += sorted((code / "src/rsci/runtime").glob("*.py"))
    sources += sorted((code / "src/rsci/data").glob("*.py"))
    sources = sorted(set(sources))
    identity = dict(
        schema=2,
        recipe="SIMPLE-GR00T-zero1-recompute-v1",
        sampler="epoch-index-augmentation-preserve-skip-v1",
        model=args.model,
        vlm=args.vlm,
        task=args.task,
        batch=args.batch_size,
        memory_optimization="zero1-fp32-adam-and-nonreentrant-dit-v1",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_size=len(args.gpus.split(",")),
        seed=args.seed,
        workers=args.num_workers,
        max_steps=args.max_steps,
        optimization=optimization_kwargs(),
        ardy_weight=0.1,
        tune_top_llm_layers=4 if args.vlm == "top4" else 0,
        color_jitter=dict(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        valid_action_horizon=16,
        history_offsets=[-6, -4, -2, 0],
        reader_layout="8image+8text-per-frame",
        dataset_fingerprint=dataset_content_fingerprint(paths["dataset"]),
        input_cache=cache_identity(paths["dataset"], config=paths["input_cache"]),
        initialization=str(paths["checkpoint"].resolve()),
        initialization_config=sha256(paths["checkpoint"] / "config.json"),
        initialization_index=sha256(paths["checkpoint"] / "model.safetensors.index.json"),
        sources={str(path.relative_to(code)): sha256(path) for path in sources},
        dependencies={
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "diffusers",
                "torchcodec",
                "albumentations",
                "accelerate",
            )
        },
    )
    if args.model == "VGQ-AE":
        identity["sidecar_manifest"] = sha256(paths["sidecar"] / "manifest.json")
    return identity, hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
