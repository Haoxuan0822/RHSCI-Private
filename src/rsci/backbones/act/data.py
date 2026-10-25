"""原生LeRobot窗口/处理链；只扩展训练期四帧视觉和ARDY旁路。"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from psi.config.augmentation import CenterCrop, ColorJitter, Normalize, ResizeImage
from psi.config.transform import ActionStateTransform, ACTModelTransform, SimpleRepackTransform
from psi.utils import pt_to_pil

OFFSETS = [-6, -4, -2, 0]
IMAGE = "observation.images.egocentric"


class Processor:
    def __init__(self, stats_path):
        self.repack = SimpleRepackTransform(
            action_chunk_size=16, pad_action_dim=36, pad_state_dim=36
        )
        self.field = ActionStateTransform(
            stat_path=str(stats_path)
            if not isinstance(stats_path, dict)
            else "__checkpoint_stats__",
            action_norm_type="bounds",
            stat_action_key="action",
            stat_state_key="states",
            normalize_state=True,
            pad_action_dim=36,
            pad_state_dim=36,
        )
        if isinstance(stats_path, dict):
            self.field.populate_stats(stats_path)
        self.image = ACTModelTransform(
            resize=ResizeImage(),
            center_crop=CenterCrop(),
            color_jitter=ColorJitter(),
            normalize=Normalize(),
            img_aug=True,
        )
        for name in ("action_min", "action_max", "state_min", "state_max"):
            value = np.asarray(getattr(self.field, name), dtype=np.float32)
            if value.shape != (36,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid official bounds: {name}")

    def __call__(self, raw, variant, training=True):
        current = dict(raw)
        frames = current[IMAGE]
        if frames.ndim == 3:
            frames = frames[None]
        if frames.ndim != 4 or frames.shape[0] != (4 if variant == "EGQ-AE" else 1):
            raise ValueError("Expected four history frames or one current frame")
        current[IMAGE] = frames[-1]
        current["states"] = current["states"][-1:]
        packed = self.field(self.repack(current))
        # 先严格执行原生当前帧处理，再复用同一随机状态处理历史；不额外推进调用方RNG。
        rng = torch.get_rng_state()
        batch = self.image(packed, no_aug=not training)
        batch = {key: torch.as_tensor(value) for key, value in batch.items()}
        batch["observation.state"] = batch["observation.state"].squeeze(0)
        if variant != "STATE":
            del batch["observation.state"]
        if variant == "EGQ-AE":
            history = []
            for frame in frames[:-1]:
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(rng)
                    historical = dict(packed, observations=[pt_to_pil(frame, normalized=False)])
                    history.append(
                        self.image(historical, no_aug=not training)["observation.images"][0]
                    )
            history.append(batch["observation.images"][0])
            batch["history_images"] = torch.stack(history)
        return batch

    def inference(self, image, states=None):
        # 推理使用同一个原生图像处理器，禁止训练增强和历史/标签输入。
        dummy = dict(
            observations=[image],
            states=np.empty(0, np.float32),
            actions=np.zeros((16, 36), np.float32),
        )
        result = {"observation.images": self.image(dummy, no_aug=True)["observation.images"][None]}
        if states is not None:
            states = np.asarray(states, np.float32).reshape(-1)
            if states.shape != (32,) or not np.isfinite(states).all():
                raise ValueError(
                    "STATE inference requires measured32D, including measured waist and previous commanded height"
                )
            result["observation.state"] = torch.from_numpy(
                self.field.normalize_state_func(np.pad(states, (0, 4)))
            )[None]
        return result


class ACTDataset(Dataset):
    def __init__(self, root, sidecar, variant, seed=2026, training=True):
        from psi.data.lerobot.compat import LeRobotDatasetMetadata

        from .video_cache import CachedLeRobotDataset

        self.root, self.variant, self.seed, self.training = Path(root), variant, seed, training
        self.processor = Processor(self.root / "meta/stats_psi0.json")
        meta = LeRobotDatasetMetadata(self.root.name, root=self.root)
        if meta.fps != 50:
            raise ValueError("RHSCI historical offsets require 50Hz")
        delta = self.processor.repack.delta_timestamps(meta.fps)
        if variant == "EGQ-AE":
            delta[IMAGE] = [value / meta.fps for value in OFFSETS]
        self.native = CachedLeRobotDataset(
            self.root.name,
            root=self.root,
            delta_timestamps=delta,
            image_transforms=None,
            video_backend=None,
        )
        if variant == "EGQ-AE":
            # 复用已核验的通用sidecar读取器，不加载EgoVLA模型或其训练配方。
            from rsci.data.ardy_sidecar import ArdySidecarReader, validate_processed_dataset_context

            context = json.loads((Path(sidecar) / "manifest.json").read_text())["context"]
            validate_processed_dataset_context(self.root, context)
            self.reader = ArdySidecarReader(sidecar, context, require_formal_teacher=True)
            episodes = [
                json.loads(line)
                for line in (self.root / "meta/episodes.jsonl").read_text().splitlines()
            ]
            if set(self.reader._episodes) != {e["episode_index"] for e in episodes} or any(
                len(self.reader._episodes[e["episode_index"]]["ardy_body_latent"]) != e["length"]
                for e in episodes
            ):
                raise ValueError("ARDY/RGB episode/frame inventory mismatch")

    def __len__(self):
        return len(self.native)

    def __getitem__(self, index):
        epoch, index = index if isinstance(index, tuple) else (0, index)
        raw = self.native[index]
        # 绑定样本随机性，persistent workers和跨epoch中途恢复不依赖worker消费/预取进度。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed + epoch * len(self) + index)
            sample = self.processor(raw, self.variant, self.training)
        if self.variant == "EGQ-AE":
            target = self.reader.get(int(raw["episode_index"]), int(raw["frame_index"]))
            sample.update(
                ardy_target=torch.from_numpy(target.ardy_body_latent.copy()),
                ardy_valid=torch.tensor(target.ardy_loss_mask),
            )
            if int(raw["frame_index"]) < 6 and target.ardy_loss_mask:
                raise ValueError(
                    "Insufficient history must mask ARDY only, not drop action samples"
                )
        if any(
            not torch.isfinite(value).all()
            for value in sample.values()
            if value.is_floating_point()
        ):
            raise FloatingPointError(f"Nonfinite ACT data at epoch={epoch}, index={index}")
        return sample


class EpochSampler(Sampler):
    """显式(epoch,index)，由Accelerate.set_epoch传递；不由worker猜测epoch。"""

    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(
            (self.epoch, index) for index in torch.randperm(len(self), generator=generator).tolist()
        )
