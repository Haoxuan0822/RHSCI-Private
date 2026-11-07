from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Dict, List, TYPE_CHECKING
from psi.config.config import DataConfig
from pathlib import Path
from psi.utils import resolve_data_path
import os
import json
import math

from psi.config.transform import ActionStateTransform
class LerobotDataConfig(DataConfig):
    root_dir: str
    rsci_ardy_sidecar_root: str | None = None
    rsci_ardy_sidecars_by_repo: bool = False
    rsci_ardy_offset_frames: int = Field(default=0, ge=0, strict=True)
    rsci_ardy_delta_frames: int = Field(default=6, gt=0, strict=True)
    rgb_frame_cache_root: str | None = None
    train_repo_ids: List[str] = Field(default_factory=list)
    val_repo_ids: List[str] = Field(default_factory=list)
    task_sample_weights: List[str] = Field(default_factory=list)

    def parsed_task_sample_weights(self) -> Dict[str, float]:
        """沿用官方按任务文本子串匹配的采样规则；返回小写子串到正权重。"""
        rules = {}
        for entry in self.task_sample_weights:
            if "=" not in entry:
                raise ValueError(f"task_sample_weights entry {entry!r} is not '<substring>=<weight>'")
            key, value = entry.rsplit("=", 1)
            weight = float(value)
            if not key.strip() or not math.isfinite(weight) or weight <= 0:
                raise ValueError("task_sample_weights requires a non-empty substring and finite weight > 0")
            rules[key.strip().lower()] = weight
        return rules

    @model_validator(mode="after")
    def check_task_sample_weights(self):
        self.parsed_task_sample_weights()
        return self

    @model_validator(mode="after")
    def resolve_psi_home_paths(self):
        """与官方一致，仅在 PSI_HOME 下相对路径确实存在时解析数据根。"""
        root = Path(self.root_dir)
        candidate = Path(os.environ.get("PSI_HOME", "/psi")) / root
        if not root.is_absolute() and candidate.exists():
            self.root_dir = str(candidate)
        return self

    @model_validator(mode="after")
    def check_repo_ids(self):
        if len(self.train_repo_ids) == 0:
            raise ValueError("train_repo_ids must be provided")
        if len(self.val_repo_ids) == 0:
            self.val_repo_ids = [self.train_repo_ids[0]]
        return self

    @model_validator(mode="after")
    def load_stats(self):
        if not isinstance(self.transform.field, ActionStateTransform):
            return self
        if (
            not Path(self.transform.field.stat_path).is_absolute() and
            self.transform.field.action_max is None
        ):
            fpath = resolve_data_path(
                Path(self.root_dir) / self.train_repo_ids[0] / self.transform.field.stat_path
            )
            if not os.path.exists(fpath):
                return self
            with open(fpath, "r") as f:
                stats = json.load(f)
                self.transform.field.populate_stats(stats)
        return self

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        from psi.data.lerobot import LeRobotDatasetWrapper
        from psi.data.dataset import Dataset as MapStyleDataset

        transform_kwargs = {"no_aug": split != "train", **transform_kwargs}
        train_dataset = LeRobotDatasetWrapper(self, split=split)
        dataset = MapStyleDataset(self, train_dataset, transform_kwargs=transform_kwargs)
        rules = self.parsed_task_sample_weights()
        if split == "train" and rules:
            dataset.sample_weights = train_dataset.frame_task_weights(rules)
        return dataset

    def mock(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        dataset = self.__call__(split, transform_kwargs=transform_kwargs, **kwargs)
        return dataset[0]
