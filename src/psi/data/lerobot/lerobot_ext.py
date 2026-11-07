from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING
from contextlib import nullcontext
if TYPE_CHECKING:
    from psi.config.data_lerobot import LerobotDataConfig
    # from psi.config.data_simple import SimpleDataConfig

import torch
import logging
from psi.data.lerobot.compat import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
    simple_rgb_parquet_loading,
)
from psi.utils import resolve_path
from psi.config.transform import LerobotRepackTransform, SimpleRepackTransform


class CachedRGBLeRobotDataset(LeRobotDataset):
    def _query_videos(self, query_timestamps, ep_idx):
        return {
            key: self.rgb_frame_cache.read(
                self.root / self.meta.get_video_file_path(ep_idx, key), timestamps, self.tolerance_s
            ).squeeze(0)
            for key, timestamps in query_timestamps.items()
        }


class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    """ A wrapper around LeRobotDataset to support multiple datasets.
    """

    def __init__(
        self,
        data_cfg: LerobotDataConfig,
        split: str = "train"
    ):
        repo_ids = data_cfg.train_repo_ids if split == "train" else data_cfg.val_repo_ids
        first_repo = repo_ids[0] if isinstance(repo_ids, list) else repo_ids
        dataset_meta = LeRobotDatasetMetadata(first_repo, resolve_path(f"{data_cfg.root_dir}/{first_repo}"))
        assert isinstance(data_cfg.transform.repack, LerobotRepackTransform)
        delta_timestamps = data_cfg.transform.repack.delta_timestamps(dataset_meta.fps)

        self.ardy_reader = None
        self.ardy_readers = []
        repack = data_cfg.transform.repack
        self.ardy_offset = getattr(data_cfg, "rsci_ardy_offset_frames", 0)
        if repack.rsci_variant in ("VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480") and self.ardy_offset:
            raise ValueError(f"{repack.rsci_variant} requires the unchanged current ARDY target")
        self.motion_delta = (
            data_cfg.rsci_ardy_delta_frames if repack.rsci_variant in ("VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480") else None
        )
        if self.motion_delta is not None and self.ardy_offset:
            raise ValueError("VGQ-AE-Mo requires current ARDY supervision")
        if self.ardy_offset and getattr(repack, "rsci_variant", None) in (None, "STATE", "NO-STATE"):
            raise ValueError("Future ARDY requires an RHSCI Query variant")
        if len(repo_ids) > 1:
            root_dir = data_cfg.root_dir
            lerobot_dataset_class = MultiLeRobotDataset
        else:
            repo_ids = first_repo
            root_dir = resolve_path(f"{data_cfg.root_dir}/{first_repo}")
            lerobot_dataset_class = LeRobotDataset

        cache_root = getattr(data_cfg, "rgb_frame_cache_root", None)
        if cache_root:
            if len(data_cfg.train_repo_ids if split == "train" else data_cfg.val_repo_ids) != 1:
                raise ValueError("RGB frame cache currently requires one dataset")
            lerobot_dataset_class = CachedRGBLeRobotDataset

        rgb_only = isinstance(repack, SimpleRepackTransform) and bool(repack.rsci_variant)
        with simple_rgb_parquet_loading() if rgb_only else nullcontext():
            self.base_dataset = lerobot_dataset_class(
                repo_ids,# type: ignore
                root=root_dir,
                delta_timestamps=delta_timestamps, # type: ignore
                image_transforms=None,
            )

        packs = getattr(self.base_dataset, "_datasets", None) or [self.base_dataset]
        for pack in packs:
            if pack.meta.fps != dataset_meta.fps:
                raise ValueError("Joint datasets must use the same sampling frequency")
            # RHSCI的RGB实验复用RGBD封装时，不将未使用的整幅深度转换成Tensor。
            # 仅改变内存列视图，磁盘/metadata/数据指纹及RGB历史采样保持原样。
            depth_key = 'observation.depth.egocentric'
            if getattr(repack, 'rsci_variant', None):
                if depth_key in pack.hf_dataset.column_names:
                    pack.hf_dataset = pack.hf_dataset.remove_columns([depth_key])
        if getattr(repack, "rsci_variant", None) not in (None, "STATE", "NO-STATE"):
            self._load_ardy_readers(data_cfg, packs)
        if cache_root:
            from rhsci.data.rgb_frame_cache import RGBFrameCache
            if self.base_dataset.video_backend != "torchcodec":
                raise ValueError("RGB cache requires the verified torchcodec backend")
            self.base_dataset.rgb_frame_cache = RGBFrameCache(resolve_path(cache_root), root_dir)
        self._cache = {}

    def _load_ardy_readers(self, data_cfg, packs):
        """逐数据集校验正式标签，保留各自局部 episode 编号，避免混用同号标签。"""
        import json
        from pathlib import Path
        from rhsci.data.ardy_sidecar import ArdySidecarReader, validate_processed_dataset_context

        if not data_cfg.rsci_ardy_sidecar_root:
            raise ValueError("RHSCI Query requires formal ARDY sidecars")
        if len(packs) > 1 and not data_cfg.rsci_ardy_sidecars_by_repo:
            raise ValueError("Multiple datasets require rsci_ardy_sidecars_by_repo")
        root = Path(resolve_path(data_cfg.rsci_ardy_sidecar_root))
        for pack in packs:
            sidecar = root / pack.repo_id if data_cfg.rsci_ardy_sidecars_by_repo else root
            context = json.loads((sidecar / "manifest.json").read_text())["context"]
            validate_processed_dataset_context(pack.root, context)
            reader = ArdySidecarReader(sidecar, context, require_formal_teacher=True)
            if set(reader._episodes) != set(pack.meta.episodes):
                raise ValueError(f"ARDY episode identities do not match {pack.repo_id}")
            for episode in pack.meta.episodes.values():
                index = int(episode["episode_index"])
                if len(reader._episodes[index]["ardy_body_latent"]) != int(episode["length"]):
                    raise ValueError(f"ARDY frame count mismatch: {pack.repo_id}, episode {index}")
            self.ardy_readers.append(reader)
        if len(self.ardy_readers) == 1:
            self.ardy_reader = self.ardy_readers[0]

    def frame_task_weights(self, rules: dict[str, float]) -> torch.Tensor:
        """复用官方逐帧权重语义：匹配多条时取最大值，未匹配为1，不按任务数均分。"""
        import numpy as np

        packs = getattr(self.base_dataset, "_datasets", None) or [self.base_dataset]
        chunks = []
        for pack in packs:
            task_index = np.asarray(pack.hf_dataset.with_format("numpy")["task_index"]).reshape(-1)
            per_task = np.ones(int(task_index.max()) + 1, dtype=np.float64)
            for index, task in pack.meta.tasks.items():
                weights = [weight for text, weight in rules.items() if text in str(task).lower()]
                if weights:
                    per_task[int(index)] = max(weights)
            chunks.append(per_task[task_index])
        weights = np.concatenate(chunks)
        if len(weights) != len(self):
            raise ValueError("Sampling weights do not match dataset frame count")
        if np.all(weights == 1):
            logging.getLogger(__name__).warning("task_sample_weights %s matched no non-unit weights", rules)
        return torch.as_tensor(weights, dtype=torch.double)

    def __getitem__(self, idx) -> dict:
        sample = self.base_dataset[idx]
        if self.ardy_readers:
            reader = self.ardy_readers[int(sample.get("dataset_index", 0))]
            from rhsci.data.ardy_future import ardy_target_at_offset
            sample["ardy_target"], sample["ardy_valid"] = ardy_target_at_offset(
                reader, int(sample["episode_index"]), int(sample["frame_index"]),
                self.ardy_offset)
            if self.motion_delta is not None:
                from rhsci.data.ardy_motion import ardy_motion_target

                sample["motion_target"], sample["motion_valid"] = ardy_motion_target(
                    reader, int(sample["episode_index"]), int(sample["frame_index"]),
                    self.motion_delta,
                )
        return sample

    def __len__(self):
        return len(self.base_dataset)

    @property
    def episode_data_index(self):
        return self.base_dataset.episode_data_index # type: ignore

    @property
    def num_episodes(self):
        return self.base_dataset.num_episodes

    @property
    def num_frames(self):
        return self.base_dataset.num_frames

    @property
    def meta(self):
        return self.base_dataset.meta # type: ignore

    @property
    def stats(self):
        return self.base_dataset.stats if type(self.base_dataset) == MultiLeRobotDataset else self.base_dataset.meta.stats # type: ignore
