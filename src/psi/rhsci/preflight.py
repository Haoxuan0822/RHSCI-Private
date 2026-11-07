"""Ψ0 本地资产与真实数据预处理检查；不启动训练。"""

import json
from pathlib import Path

from rhsci.data.visual_history import visual_history_offsets


def run_training_preflight(cfg):
    """检查本地资产和真实数据处理结果，不启动训练。验证失败抛出异常，通过时返回 None。"""
    from transformers import AutoProcessor

    from psi.data.lerobot import LeRobotDatasetWrapper
    vlm = Path(cfg.model.model_name_or_path)
    required = [
        vlm / "config.json",
        Path(cfg.model.pretrained_action_header_path) / "action_header.safetensors",
        Path(cfg.data.root_dir) / cfg.data.train_repo_ids[0] / cfg.data.transform.field.stat_path,
    ]
    if not any(
        (vlm / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")
    ):
        raise ValueError(f"Missing local VLM weights: {vlm}")
    if cfg.model.rsci_variant.startswith("EGQ-"):
        dino = Path(cfg.model.rsci_rgb_encoder_path)
        required += [dino / "config.json"]
        if not any((dino / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            raise ValueError(f"Missing local DINOv2 weights: {dino}")
    for path in required:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing local asset: {path}")
    # wrapper 执行正式教师、数据指纹及逐 episode sidecar 长度门禁。
    wrappers = [LeRobotDatasetWrapper(cfg.data, split=split) for split in ("train", "val")]
    if cfg.data.rsci_ardy_offset_frames:
        from rhsci.data.ardy_future import supervision_counts

        for wrapper in wrappers:
            for reader in wrapper.ardy_readers:
                counts = supervision_counts(reader, cfg.data.rsci_ardy_offset_frames)
                if not counts["valid_ardy_frames"]:
                    raise ValueError("No valid future ARDY targets in one of the datasets")
                print("RSCI_ARDY_SUPERVISION " + json.dumps(counts), flush=True)
    processor = AutoProcessor.from_pretrained(str(vlm), local_files_only=True)
    # 每个split的每个数据集都预处理一个样本；不只检查混合后的第一个任务。
    for wrapper in wrappers:
        start = 0
        packs = getattr(wrapper.base_dataset, "_datasets", None) or [wrapper.base_dataset]
        for pack in packs:
            if cfg.model.rsci_variant and "observation.depth.egocentric" in pack.hf_dataset.column_names:
                raise ValueError("RHSCI RGB-only reader must exclude unused depth column")
            index = start + min(-visual_history_offsets(cfg.model.rsci_variant)[0], len(pack) - 1)
            _check_processed_sample(cfg, wrapper[index], processor)
            start += len(pack)
    print(
        "RHSCI local assets, formal dataset gates and CPU preprocessing/collation samples passed; no training started.",
        flush=True,
    )


def _check_processed_sample(cfg, raw_sample, processor):
    """核对一个原生样本的当前输入、四帧辅助标签和批处理结构。"""
    from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether

    sample = cfg.data.transform(
        raw_sample,
        vlm_processor=processor,
        no_aug=True,
    )
    batch = PaddedCollatorForTogether(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id
    )([sample])
    if "depths" in batch:
        raise ValueError("RHSCI must not request depth")
    if cfg.model.rsci_variant not in ("STATE", "NO-STATE"):
        if "ardy_target" not in batch or batch["history_valid"].shape != (1, 4):
            raise ValueError("Invalid four-frame ARDY sample")
        if cfg.model.rsci_variant.startswith("VGQ-") and len(batch["history_vlm_inputs"]) != 3:
            raise ValueError("Invalid VGQ historical inputs")
        if cfg.model.rsci_variant.startswith("EGQ-") and batch["history_rgb_images"].shape[1:3] != (
            4,
            3,
        ):
            raise ValueError("Invalid EGQ historical inputs")
    if cfg.model.rsci_variant == "STATE-RECONSTRUCTION":
        if batch["state_target"].shape != (1, 31) or batch["state_valid"].shape != (1, 31) or batch["states"].count_nonzero():
            raise ValueError("Invalid state31 supervision or leaking student state")
    if cfg.model.rsci_variant == "VGQ-VLM":
        from transformers import AutoConfig
        from .vlm_query import four_frame_inputs
        current = {key: batch[key] for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
        packed = four_frame_inputs([*batch["history_vlm_inputs"], current], AutoConfig.from_pretrained(cfg.model.model_name_or_path, local_files_only=True))
        if packed["input_ids"].shape[1] + 1 > cfg.model.model_max_length:
            raise ValueError("Direct four-image Query exceeds context budget")
