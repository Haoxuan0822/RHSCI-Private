"""pi05: 启动身份与源码契约。"""

from rhsci.data.state_noise import STATE_NOISE_LOCATION
from rhsci.paths import CODE_ROOT as CODE_ROOT
from rhsci.runtime.contracts import compute_contract_digest, reuse_run_contract

from .checkpoints import sha256


def build_training_contract(args, paths):
    """根据模型、数据与配方生成恢复身份；源码编辑和依赖版本不作为恢复门槛。返回 (契约字典, SHA256)。"""
    from rhsci.data.ardy_sidecar import dataset_content_fingerprint
    from rhsci.data.input_cache import cache_identity

    from .augmentation import AUGMENTATION_VERSION

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
        asset_manifest=sha256(paths["checkpoint"] / "RSCI_ASSETS.json")
        if (paths["checkpoint"] / "RSCI_ASSETS.json").is_file()
        else None,
        tokenizer=sha256(paths["tokenizer"]),
    )
    if args.model == "VGQ-AE":
        identity["sidecar"] = sha256(paths["sidecar"] / "manifest.json")
    identity = reuse_run_contract(identity, paths["output"])
    return identity, compute_contract_digest(identity)
