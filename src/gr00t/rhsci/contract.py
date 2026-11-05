"""gr00t: 启动身份与源码契约。"""

from rhsci.data.ardy_motion import motion_config
from rhsci.paths import CODE_ROOT as CODE_ROOT
from rhsci.runtime.contracts import compute_contract_digest, reuse_run_contract

from .config import optimization_kwargs


def build_training_contract(args, paths):
    """根据模型、数据与配方生成恢复身份；源码编辑和依赖版本不作为恢复门槛。返回 (契约字典, SHA256)。"""
    from rhsci.data.ardy_sidecar import dataset_content_fingerprint
    from rhsci.data.input_cache import cache_identity

    from .checkpoints import sha256

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
    )
    if args.model in ("VGQ-AE", "VGQ-AE-Mo"):
        identity["sidecar_manifest"] = sha256(paths["sidecar"] / "manifest.json")
    identity.update(motion_config(args))
    identity = reuse_run_contract(identity, paths["output"])
    return identity, compute_contract_digest(identity)
