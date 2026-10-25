"""读取与 π0.5 对照相同的正式数据；STATE 读取当前状态，动作保留物理单位。"""

import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

OFFSETS = np.array([-6, -4, -2, 0], dtype=np.int64)


class SimpleDataset(Dataset):
    def __init__(self, root, sidecar, processor=None, variant="VGQ-AE"):
        from rsci.data.ardy_sidecar import ArdySidecarReader, validate_processed_dataset_context

        self.root, self.processor = Path(root), processor
        self.variant = variant
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.episodes = [
            json.loads(line)
            for line in (self.root / "meta/episodes.jsonl").read_text().splitlines()
            if line
        ]
        self.tasks = {
            int(value["task_index"]): value["task"]
            for value in map(json.loads, (self.root / "meta/tasks.jsonl").read_text().splitlines())
        }
        if (
            self.info["fps"] != 50
            or len(self.episodes) != self.info["total_episodes"]
            or sum(episode["length"] for episode in self.episodes) != self.info["total_frames"]
        ):
            raise ValueError("Invalid/incomplete 50Hz dataset")
        self.ends = np.cumsum([max(0, episode["length"] - 30 + 1) for episode in self.episodes])
        self._tables, self._decoders, self._pid = OrderedDict(), OrderedDict(), os.getpid()
        if variant == "STATE":
            return
        context = json.loads((Path(sidecar) / "manifest.json").read_text())["context"]
        validate_processed_dataset_context(self.root, context)
        self.reader = ArdySidecarReader(sidecar, context, require_formal_teacher=True)
        if set(self.reader._episodes) != {episode["episode_index"] for episode in self.episodes}:
            raise ValueError("ARDY episode identity mismatch")
        for episode in self.episodes:
            if (
                len(self.reader._episodes[episode["episode_index"]]["ardy_body_latent"])
                != episode["length"]
            ):
                raise ValueError("ARDY frame count mismatch")
        self._tables, self._decoders, self._pid = OrderedDict(), OrderedDict(), os.getpid()

    def __len__(self):
        return int(self.ends[-1])

    def path(self, episode, kind):
        index = int(episode["episode_index"])
        return self.root / self.info[kind + "_path"].format(
            episode_index=index,
            episode_chunk=index // self.info.get("chunks_size", 1000),
            video_key="observation.images.egocentric",
        )

    def _table(self, episode):
        import pyarrow.parquet as pq

        index = episode["episode_index"]
        if index not in self._tables:
            table = pq.read_table(
                self.path(episode, "data"),
                columns=["action", "episode_index", "frame_index", "task_index"]
                + (["states"] if self.variant == "STATE" else []),
            ).to_pydict()
            table["action"] = np.stack(table["action"]).astype(np.float32)
            if (
                table["action"].shape != (episode["length"], 36)
                or not np.array_equal(table["frame_index"], np.arange(episode["length"]))
                or set(table["episode_index"]) != {index}
            ):
                raise ValueError("Parquet frame/action identity mismatch")
            self._tables[index] = table
            if len(self._tables) > 4:
                self._tables.popitem(last=False)
        self._tables.move_to_end(index)
        return self._tables[index]

    def raw_sample(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        # fork/spawn 均不得共享 FFmpeg 文件描述符；沿用已有 π0.5 加速读取的缓存边界。
        if self._pid != os.getpid():
            self._tables, self._decoders, self._pid = OrderedDict(), OrderedDict(), os.getpid()
        slot = int(np.searchsorted(self.ends, index, side="right"))
        frame = int(index - (self.ends[slot - 1] if slot else 0))
        episode = self.episodes[slot]
        episode_id = episode["episode_index"]
        table = self._table(episode)
        requested = frame + (np.array([0]) if self.variant == "STATE" else OFFSETS)
        from torchcodec.decoders import VideoDecoder

        if episode_id not in self._decoders:
            decoder = VideoDecoder(
                str(self.path(episode, "video")), device="cpu", num_ffmpeg_threads=1
            )
            if len(decoder) != episode["length"]:
                raise ValueError("Video/parquet length mismatch")
            self._decoders[episode_id] = decoder
            if len(self._decoders) > 2:
                self._decoders.popitem(last=False)
        self._decoders.move_to_end(episode_id)
        images = (
            self._decoders[episode_id].get_frames_at(indices=np.maximum(requested, 0).tolist()).data
        )
        actions = table["action"][frame : frame + 30]
        if not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite action: episode={episode_id}, frame={frame}")
        if self.variant == "STATE":
            states = np.asarray(table["states"][frame], dtype=np.float32)
            if states.shape != (32,) or not np.isfinite(states).all():
                raise ValueError("Invalid current 32D states")
            return dict(
                images=images.permute(0, 2, 3, 1).numpy(),
                actions=actions,
                states=states,
                text=self.tasks[int(table["task_index"][frame])],
                episode_index=episode_id,
                frame_index=frame,
            )
        target = self.reader.get(episode_id, frame)
        if target.ardy_body_latent.shape != (128,) or (
            target.ardy_loss_mask and not np.isfinite(target.ardy_body_latent).all()
        ):
            raise ValueError("Invalid formal ARDY target")
        return dict(
            images=images.permute(0, 2, 3, 1).numpy(),
            actions=actions,
            text=self.tasks[int(table["task_index"][frame])],
            history_valid=requested >= 0,
            ardy_target=np.array(target.ardy_body_latent, copy=True),
            ardy_valid=bool(target.ardy_loss_mask),
            episode_index=episode_id,
            frame_index=frame,
        )

    def __getitem__(self, index):
        sample = self.raw_sample(index)
        return self.processor(sample) if self.processor else sample

    def __getstate__(self):
        return {**self.__dict__, "_tables": OrderedDict(), "_decoders": OrderedDict(), "_pid": None}
