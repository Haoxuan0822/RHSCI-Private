"""gr00t: 数据、资产与恢复前置检查。"""

import json

from rhsci.data.multitask import (
    build_training_dataset,
    dataset_children,
    preflight_indices,
    shared_statistics,
)
from rhsci.data.visual_history import visual_history_offsets


def build_processor(args, paths):
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

    from .data import modality, statistics

    config = json.loads((paths["checkpoint"] / "config.json").read_text())
    settings = json.loads((paths["checkpoint"] / "processor_config.json").read_text())[
        "processor_kwargs"
    ]
    settings.update(
        modality_configs=modality(args.model),
        statistics=statistics(paths["dataset"], args.model, raw=shared_statistics(paths, "gr00t")),
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
    dataset = build_training_dataset(
        paths,
        SimpleRHSCIDataset,
        args.model,
        processor,
        seed=args.seed,
        ardy_delta_frames=getattr(args, "ardy_delta_frames", 6),
    )

    # 起始无效历史及完整历史各一个真实样本，包含官方Eagle分词/图像处理。
    config_horizon = processor.max_action_horizon
    for child in dataset_children(dataset):
        indices = preflight_indices(child, args.model)
        selected = [indices[0], indices[1] if len(indices) > 1 else indices[0]]
        batch = processor.collator([child[index] for index in selected])["inputs"]
        if batch["action"].shape != (2, config_horizon, processor.max_action_dim):
            raise ValueError("Invalid padded action shape")
        if args.model in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480"):
            import numpy as np

            expected = np.stack(
                [child.raw_sample(index).metadata["rsci_history_valid"] for index in selected]
            )
            if len(batch["rsci_history"]) != 3 or not np.array_equal(
                batch["rsci_history_valid"], expected
            ):
                raise ValueError("Invalid chronological history mask")
    print(
        f"GR00T_RSCI_CHECK_OK task={args.task} model={args.model} samples={len(dataset)} padded_horizon={config_horizon}; no training started",
        flush=True,
    )
    if "datasets" in paths:
        from .multitask import build_sharded_mixture

        dataset = build_sharded_mixture(paths, dataset, processor)
    return processor, dataset


def validate_training_data(args, paths):
    """检查训练数据、归一化与历史帧边界，不进行模型训练更新。

    只读取真实数据，验证集边界、历史帧及 STATE/ARDY 字段，不加载模型。"""
    import numpy as np

    from .data import SimpleRHSCIDataset

    shared_statistics(paths, "gr00t")
    dataset = build_training_dataset(
        paths,
        SimpleRHSCIDataset,
        args.model,
        seed=args.seed,
        ardy_delta_frames=getattr(args, "ardy_delta_frames", 6),
    )

    # 混乱次序、第一集边界与下一集开头，验证不会跨episode读取历史/标签。
    offsets = np.array(visual_history_offsets(args.model))
    for child in dataset_children(dataset):
        for index in preflight_indices(child, args.model):
            sample = child.raw_sample(index)
            if args.model in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480"):
                frame = sample.metadata["frame_index"]
                if not np.array_equal(sample.metadata["rsci_history_valid"], offsets + frame >= 0):
                    raise ValueError("History crossed episode boundary")
                target = child.reader.get(sample.metadata["episode_index"], frame)
                np.testing.assert_array_equal(
                    sample.metadata["rsci_ardy_target"], target.ardy_body_latent
                )
            if bool(sample.states) != (args.model == "STATE"):
                raise ValueError("Unexpected state dependency")
    print(
        f"GR00T_RSCI_DATA_OK task={args.task} model={args.model} samples={len(dataset)}; no model loaded",
        flush=True,
    )
