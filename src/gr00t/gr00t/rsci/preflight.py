"""gr00t: 数据、资产与恢复前置检查。"""

import json

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def build_processor(args, paths):
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

    from .data import modality, statistics

    config = json.loads((paths["checkpoint"] / "config.json").read_text())
    settings = json.loads((paths["checkpoint"] / "processor_config.json").read_text())[
        "processor_kwargs"
    ]
    settings.update(
        modality_configs=modality(args.model),
        statistics=statistics(paths["dataset"], args.model),
        rsci_variant=args.model,
        use_relative_action=False,
        max_state_dim=config["max_state_dim"],
        max_action_dim=config["max_action_dim"],
        max_action_horizon=config["action_horizon"],
        use_albumentations=True,
        color_jitter_params=dict(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
    )
    return Gr00tN1d6Processor(
        **settings,
        transformers_loading_kwargs={"trust_remote_code": True, "local_files_only": True},
    )


def preflight(args, paths):
    from .data import SimpleRSCIDataset

    checkpoint = paths["checkpoint"]
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if not (checkpoint / shard).is_file():
            raise FileNotFoundError(checkpoint / shard)
    processor = build_processor(args, paths)
    dataset = SimpleRSCIDataset(
        paths["dataset"],
        args.model,
        processor,
        paths["sidecar"],
        seed=args.seed,
        cache=paths["input_cache"],
    )

    # 起始无效历史及完整历史各一个真实样本，包含官方Eagle分词/图像处理。
    batch = processor.collator([dataset[0], dataset[6]])["inputs"]
    if batch["action"].shape != (
        2,
        config_horizon := processor.max_action_horizon,
        processor.max_action_dim,
    ):
        raise ValueError("Invalid padded action shape")
    if args.model == "VGQ-AE":
        if (
            len(batch["rsci_history"]) != 3
            or batch["rsci_history_valid"][0].all()
            or not batch["rsci_history_valid"][1].all()
        ):
            raise ValueError("Invalid chronological history mask")
    print(
        f"GR00T_RSCI_CHECK_OK task={args.task} model={args.model} samples={len(dataset)} padded_horizon={config_horizon}; no training started",
        flush=True,
    )
    return processor, dataset


def check_training_environment():
    import importlib.util

    expected = {
        "torch": "2.7.0",
        "transformers": "4.51.3",
        "diffusers": "0.35.1",
        "torchcodec": "0.4.0",
    }
    failures = []
    for name, version in expected.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = "missing"
        if installed.split("+")[0] != version:
            failures.append(f"{name}=={version} (found {installed})")
    if importlib.util.find_spec("flash_attn") is None:
        failures.append("flash-attn==2.7.4.post1")
    if failures:
        raise RuntimeError(
            "GR00T training environment is not ready: "
            + ", ".join(failures)
            + "; see baselines/gr00t/README.md"
        )
