"""继承官方 SIMPLE handler/collator；只补历史、正式 ARDY 和可恢复采样。"""

import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from lambda0.data.domain_handler.psi_simple import PsiSimpleHandler, decode_video_segment
from lambda0.data.heterogeneous_collator import HeterogeneousActionCollator
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


@contextmanager
def local_rng(seed):
    # 数据随机流与 flow noise/Dropout 分离；重启 worker 和预取不改变实际样本。
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    random.seed(seed)
    np.random.seed(seed % (2**32))
    generator = torch.Generator().manual_seed(seed)
    torch.set_rng_state(generator.get_state())
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class SafePsiSimpleHandler(PsiSimpleHandler):
    from .contracts import SAFE_NORMALIZATION as normalization_contract


class RSCIPsiSimpleHandler(SafePsiSimpleHandler):
    def __init__(self, meta, num_views):
        super().__init__(meta, num_views)
        if self.fps != 50:
            raise ValueError("RHSCI history requires exactly 50Hz")
        from rsci.data.ardy_sidecar import ArdySidecarReader, validate_processed_dataset_context

        sidecar = Path(meta["rsci_sidecar"])
        if not sidecar.is_absolute():
            sidecar = self.root.parent.parent / sidecar
        context = json.loads((sidecar / "manifest.json").read_text())["context"]
        validate_processed_dataset_context(self.root, context)
        self.reader = ArdySidecarReader(sidecar, context, require_formal_teacher=True)
        for item in meta["datalist"]:
            rows = self.reader._episodes[int(item["episode_index"])]["ardy_body_latent"]
            if len(rows) != int(item["length"]):
                raise ValueError("ARDY episode/frame identity mismatch")

    def _build_sample(self, **kwargs):
        # 当前帧两条分支复用同一增强结果；历史重放当前帧的 torch 随机参数。
        rng = torch.get_rng_state()
        sample = super()._build_sample(**kwargs)
        after = torch.get_rng_state()
        item, index, image_aug = kwargs["item"], kwargs["index"], kwargs["image_aug"]
        offsets = np.array([-6, -4, -2, 0]) + index
        start = max(0, index - 6)
        if self.storage_layout == "episode_v2":
            _, video_path = self._episode_paths(int(item["episode_index"]))
            video_start = 0
        else:
            video_path = self.root / self.video_path_template
            video_start = int(item["video_start_frame"])
        frames = decode_video_segment(
            video_path, start_frame=video_start + start, length=index - start + 1, fps=self.fps
        )
        history = []
        try:
            for frame_index in offsets[:3]:
                torch.set_rng_state(rng)
                history.append(image_aug(Image.fromarray(frames[max(0, int(frame_index)) - start])))
        finally:
            torch.set_rng_state(after)
        target = self.reader.get(int(item["episode_index"]), index)
        sample.update(
            history=torch.stack(history),
            history_valid=torch.from_numpy(offsets >= 0),
            ardy_target=torch.tensor(np.array(target.ardy_body_latent, copy=True)),
            ardy_valid=torch.tensor(bool(target.ardy_loss_mask)),
        )
        # 不让真实 state 经 processor 文本或连续 AE 路径进入 VGQ-AE。
        sample["proprio"].zero_()
        sample["proprio_valid_mask"].zero_()
        return sample


class RSCICollator(HeterogeneousActionCollator):
    def __call__(self, samples):
        batch = super().__call__(samples)
        for key in ("history", "history_valid", "ardy_target", "ardy_valid"):
            if key in samples[0]:
                batch[key] = torch.stack([sample[key] for sample in samples])
        return batch


class IndexedOfficialDataset(Dataset):
    def __init__(self, reader):
        self.reader = reader
        from lambda0.data.domain_handler.registry import get_handler_cls

        self.handlers = {
            name: get_handler_cls(meta["handler"])(meta, reader.num_views)
            for name, meta in reader.metas.items()
        }

    def __getitem__(self, index):
        seed = int.from_bytes(
            hashlib.sha256(
                f"{self.reader.sampling_seed}/{self.reader.shard_index}/{index}".encode()
            ).digest()[:8],
            "little",
        )
        with local_rng(seed):
            request = self.reader.sampler.sample()
            meta = self.reader.metas[request.dataset_name]
            sample = self.handlers[request.dataset_name].sample_episode(
                request.episode_index,
                num_actions=self.reader.num_actions,
                image_aug=self.reader.image_aug,
                lang_aug_map=meta.get("lang_aug_map"),
                sample_index=request.timestep_index,
            )
            return self.reader._finalize_sample(request.dataset_name, sample, request)


class CursorSampler(Sampler):
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        yield from range(self.loader.consumed * self.loader.batch_size, 2**62)


class ResumableLoader:
    def __init__(self, reader, batch_size, num_workers):
        self.dataset, self.batch_size, self.consumed = reader, batch_size, 0
        self.loader = DataLoader(
            IndexedOfficialDataset(reader),
            batch_size=batch_size,
            sampler=CursorSampler(self),
            num_workers=num_workers,
            collate_fn=RSCICollator(),
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(reader.sampling_seed),
        )

    def __iter__(self):
        for batch in self.loader:
            self.consumed += 1
            yield batch

    def state_dict(self):
        return dict(
            consumed=self.consumed,
            batch_size=self.batch_size,
            seed=self.dataset.sampling_seed,
            world=self.dataset.num_shards,
        )

    def load_state_dict(self, state):
        current = self.state_dict()
        if any(state.get(key) != current[key] for key in current if key != "consumed"):
            raise ValueError("Exact data resume requires unchanged batch/seed/rank/world")
        if not isinstance(state["consumed"], int) or state["consumed"] < 0:
            raise ValueError("Invalid data cursor")
        self.consumed = state["consumed"]
