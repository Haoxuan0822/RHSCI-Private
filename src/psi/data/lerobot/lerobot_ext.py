from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from psi.config.data_lerobot import LerobotDataConfig
    # from psi.config.data_simple import SimpleDataConfig

import torch
from psi.data.lerobot.compat import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from psi.utils import resolve_path
from psi.config.transform import LerobotRepackTransform


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
        repack = data_cfg.transform.repack
        self.ardy_offset = getattr(data_cfg, "rsci_ardy_offset_frames", 0)
        if self.ardy_offset and getattr(repack, "rsci_variant", None) in (None, "STATE", "NO-STATE"):
            raise ValueError("Future ARDY requires an RHSCI Query variant")
        if getattr(repack, "rsci_variant", None) not in (None, "STATE", "NO-STATE"):
            if len(repo_ids) != 1 or not data_cfg.rsci_ardy_sidecar_root:
                raise ValueError("RHSCI ARDY requires one task and data.rsci_ardy_sidecar_root")
            import json
            from pathlib import Path
            from rsci.data.ardy_sidecar import ArdySidecarReader, validate_processed_dataset_context
            sidecar = Path(resolve_path(data_cfg.rsci_ardy_sidecar_root))
            context = json.loads((sidecar / "manifest.json").read_text())["context"]
            validate_processed_dataset_context(resolve_path(f"{data_cfg.root_dir}/{first_repo}"), context)
            self.ardy_reader = ArdySidecarReader(sidecar, context, require_formal_teacher=True)
            episodes = dataset_meta.episodes
            for episode in episodes.values():
                index = int(episode["episode_index"])
                arrays = self.ardy_reader._episodes.get(index)
                if arrays is None or len(arrays["ardy_body_latent"]) != int(episode["length"]):
                    raise ValueError(f"RHSCI ARDY sidecar episode {index} frame count mismatch")


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

        self.base_dataset = lerobot_dataset_class(
            repo_ids,# type: ignore
            root=root_dir,
            delta_timestamps=delta_timestamps, # type: ignore
            image_transforms=None,
        )

        if len(data_cfg.train_repo_ids if split == "train" else data_cfg.val_repo_ids) == 1:
            # RHSCI的RGB实验复用RGBD封装时，不将未使用的整幅深度转换成Tensor。
            # 仅改变内存列视图，磁盘/metadata/数据指纹及RGB历史采样保持原样。
            depth_key = 'observation.depth.egocentric'
            if getattr(repack, 'rsci_variant', None):
                if depth_key in self.base_dataset.hf_dataset.column_names:
                    self.base_dataset.hf_dataset = self.base_dataset.hf_dataset.remove_columns([depth_key])
        if cache_root:
            from rsci.data.rgb_frame_cache import RGBFrameCache
            if self.base_dataset.video_backend != "torchcodec":
                raise ValueError("RGB cache requires the verified torchcodec backend")
            self.base_dataset.rgb_frame_cache = RGBFrameCache(resolve_path(cache_root), root_dir)
        self._cache = {}

    def __getitem__(self, idx) -> dict:
        sample = self.base_dataset[idx]
        if self.ardy_reader is not None:
            from rsci.data.ardy_future import ardy_target_at_offset
            sample["ardy_target"], sample["ardy_valid"] = ardy_target_at_offset(
                self.ardy_reader, int(sample["episode_index"]), int(sample["frame_index"]),
                self.ardy_offset)
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
