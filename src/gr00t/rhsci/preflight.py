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


def run_training_preflight(args, paths):
    """检查本地资产和真实数据处理结果，不启动训练。返回 (处理器, 数据集)。"""
    from .data import SimpleRHSCIDataset

    checkpoint = paths["checkpoint"]
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if not (checkpoint / shard).is_file():
            raise FileNotFoundError(checkpoint / shard)
    processor = build_processor(args, paths)
    dataset = SimpleRHSCIDataset(
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


def validate_training_data(args, paths):
    """检查训练数据、归一化与历史帧边界，不进行模型训练更新。

    只读取真实数据，验证集边界、历史帧及 STATE/ARDY 字段，不加载模型。"""
    import numpy as np

    from .data import SimpleRHSCIDataset

    dataset = SimpleRHSCIDataset(
        paths["dataset"],
        args.model,
        sidecar=paths["sidecar"],
        seed=args.seed,
        cache=paths["input_cache"],
    )

    # 混乱次序、第一集边界与下一集开头，验证不会跨episode读取历史/标签。
    indices = [6, 0, int(dataset.ends[0]), int(dataset.ends[0]) - 1, 6]
    for index in indices:
        sample = dataset.raw_sample(index)
        if args.model == "VGQ-AE":
            frame = sample.metadata["frame_index"]
            if not np.array_equal(
                sample.metadata["rsci_history_valid"], np.array([-6, -4, -2, 0]) + frame >= 0
            ):
                raise ValueError("History crossed episode boundary")
            target = dataset.reader.get(sample.metadata["episode_index"], frame)
            np.testing.assert_array_equal(
                sample.metadata["rsci_ardy_target"], target.ardy_body_latent
            )
        if bool(sample.states) != (args.model == "STATE"):
            raise ValueError("Unexpected state dependency")
    print(
        f"GR00T_RSCI_DATA_OK task={args.task} model={args.model} samples={len(dataset)}; no model loaded",
        flush=True,
    )
