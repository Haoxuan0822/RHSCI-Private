"""将现有SIMPLE新数据只读接入官方GR00T处理器，保留episode/frame身份。"""

import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
    VLAStepData,
)
from torch.utils.data import Dataset

TAG = EmbodimentTag.G1_LOCO_DOWNSTREAM
OFFSETS = np.array([-6, -4, -2, 0], dtype=np.int64)
GROUPS = {
    "left_hand": (0, 7),
    "right_hand": (7, 14),
    "left_arm": (14, 21),
    "right_arm": (21, 28),
    "rpy": (28, 31),
    "height": (31, 32),
    "torso_vx": (32, 33),
    "torso_vy": (33, 34),
    "torso_vyaw": (34, 35),
    "target_yaw": (35, 36),
}
STATE_KEYS = list(GROUPS)[:6]


def modality(variant, *, training=True):
    return {
        TAG.value: {
            "video": ModalityConfig(
                delta_indices=OFFSETS.tolist() if training and variant == "VGQ-AE" else [0],
                modality_keys=["rs_view"],
            ),
            "state": ModalityConfig(
                delta_indices=[0], modality_keys=STATE_KEYS if variant == "STATE" else []
            ),
            "action": ModalityConfig(
                delta_indices=list(range(16)),
                modality_keys=list(GROUPS),
                action_configs=[
                    ActionConfig(
                        ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT
                    )
                    for _ in GROUPS
                ],
            ),
            "language": ModalityConfig(
                delta_indices=[0], modality_keys=["annotation.human.task_description"]
            ),
        }
    }


def statistics(root, variant):
    raw = json.loads((Path(root) / "meta/stats.json").read_text())
    result = {"state": {}, "action": {}}
    for kind, source, keys in (
        ("action", "action", list(GROUPS)),
        ("state", "states", STATE_KEYS if variant == "STATE" else []),
    ):
        for key in keys:
            start, stop = GROUPS[key]
            result[kind][key] = {
                name: value[start:stop]
                for name, value in raw[source].items()
                if name in ("min", "max", "mean", "std", "q01", "q99")
            }
    return {TAG.value: result}


class SimpleRSCIDataset(Dataset):
    """map-style避免原生随机shard丢失身份，并允许Trainer按同一seed恢复采样进度。"""

    def __init__(self, root, variant, processor=None, sidecar=None):
        self.root, self.variant, self.processor = Path(root), variant, processor
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["fps"] != 50:
            raise ValueError("RHSCI history offsets require 50Hz data")
        self.episodes = [
            json.loads(line)
            for line in (self.root / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.tasks = {
            int(value["task_index"]): value["task"]
            for value in (
                json.loads(line)
                for line in (self.root / "meta/tasks.jsonl").read_text().splitlines()
                if line.strip()
            )
        }
        if (
            len(self.episodes) != self.info["total_episodes"]
            or sum(e["length"] for e in self.episodes) != self.info["total_frames"]
        ):
            raise ValueError("Incomplete dataset metadata")
        # 沿用GR00T SIMPLE有效action horizon16；所有组使用完全相同的采样集合。
        self.ends = np.cumsum([max(0, e["length"] - 15) for e in self.episodes])
        self.reader = None
        if variant == "VGQ-AE":
            from rsci.data.ardy_sidecar import ArdySidecarReader, validate_processed_dataset_context

            if sidecar is None:
                raise ValueError("VGQ-AE requires a formal ARDY sidecar")
            context = json.loads((Path(sidecar) / "manifest.json").read_text())["context"]
            validate_processed_dataset_context(self.root, context)
            self.reader = ArdySidecarReader(sidecar, context, require_formal_teacher=True)
            if set(self.reader._episodes) != {e["episode_index"] for e in self.episodes}:
                raise ValueError("ARDY episode identities do not match dataset")
            for episode in self.episodes:
                if (
                    len(self.reader._episodes[episode["episode_index"]]["ardy_body_latent"])
                    != episode["length"]
                ):
                    raise ValueError("ARDY frame count mismatch")
        self._tables, self._decoders = OrderedDict(), OrderedDict()
        self._cache_pid = os.getpid()

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
            # 显式列裁剪，不读取原包中未使用的整幅depth。
            columns = ["action", "episode_index", "frame_index", "task_index"]
            if self.variant == "STATE":
                columns.append("states")
            table = pq.read_table(self.path(episode, "data"), columns=columns).to_pydict()
            for key in ("action", "states"):
                if key in table:
                    table[key] = np.stack(table[key]).astype(np.float32)
            if len(table["action"]) != episode["length"] or not np.array_equal(
                table["frame_index"], np.arange(episode["length"])
            ):
                raise ValueError("Parquet frame identity mismatch")
            if set(table["episode_index"]) != {index} or table["action"].shape[1] != 36:
                raise ValueError("Parquet episode/action mismatch")
            self._tables[index] = table
            if len(self._tables) > 4:
                self._tables.popitem(last=False)
        self._tables.move_to_end(index)
        return self._tables[index]

    def raw_sample(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Linux DataLoader默认fork，不调用__getstate__。预检打开的FFmpeg句柄
        # 会被所有worker继承并共享文件偏移，必须在子进程首次读取前独立重建。
        if self._cache_pid != os.getpid():
            self._tables, self._decoders = OrderedDict(), OrderedDict()
            self._cache_pid = os.getpid()
        slot = int(np.searchsorted(self.ends, index, side="right"))
        frame = int(index - (self.ends[slot - 1] if slot else 0))
        episode = self.episodes[slot]
        episode_id = episode["episode_index"]
        table = self._table(episode)
        offsets = OFFSETS if self.variant == "VGQ-AE" else np.array([0])
        requested = frame + offsets
        clipped = np.maximum(requested, 0)
        from torchcodec.decoders import VideoDecoder

        video_path = self.path(episode, "video")
        try:
            if episode_id not in self._decoders:
                decoder = VideoDecoder(str(video_path), device="cpu", num_ffmpeg_threads=1)
                if len(decoder) != episode["length"]:
                    raise ValueError("RGB video/parquet frame counts do not match")
                self._decoders[episode_id] = decoder
                if len(self._decoders) > 2:
                    self._decoders.popitem(last=False)
            self._decoders.move_to_end(episode_id)
            images = (
                self._decoders[episode_id]
                .get_frames_at(indices=clipped.tolist())
                .data.permute(0, 2, 3, 1)
                .numpy()
            )
        except (RuntimeError, ValueError) as exc:
            # 不跳样本/换帧/静默重试；保留原始异常并给出可直接复现的样本身份。
            worker = torch.utils.data.get_worker_info()
            raise RuntimeError(
                f"RGB decode failed: video={video_path} episode={episode_id} "
                f"frame={frame} indices={clipped.tolist()} sample={index} pid={os.getpid()} "
                f"worker={worker.id if worker else 'main'}: {exc}"
            ) from exc
        states = {}
        if self.variant == "STATE":
            state = table["states"][frame : frame + 1]
            if state.shape != (1, 32) or not np.isfinite(state).all():
                raise ValueError("Expected finite 32D SIMPLE state")
            states = {
                key: state[:, start:stop]
                for key, (start, stop) in GROUPS.items()
                if key in STATE_KEYS
            }
        actions = table["action"][frame : frame + 16]
        if not np.isfinite(actions).all():
            raise ValueError("Nonfinite action")
        metadata = {"episode_index": episode_id, "frame_index": frame}
        if self.reader is not None:
            target = self.reader.get(episode_id, frame)
            metadata.update(
                rsci_history_valid=(requested >= 0),
                rsci_ardy_target=target.ardy_body_latent,
                rsci_ardy_valid=np.bool_(target.ardy_loss_mask),
            )
        return VLAStepData(
            images={"rs_view": list(images)},
            states=states,
            actions={key: actions[:, start:stop] for key, (start, stop) in GROUPS.items()},
            text=self.tasks[int(table["task_index"][frame])],
            embodiment=TAG,
            metadata=metadata,
        )

    def __getitem__(self, index):
        sample = self.raw_sample(index)
        return (
            self.processor([{"type": "episode_step", "content": sample}])
            if self.processor
            else sample
        )

    def __getstate__(self):
        # spawn/forkserver的pickle路径也不传递FFmpeg句柄；fork由PID检查处理。
        return {
            **self.__dict__,
            "_tables": OrderedDict(),
            "_decoders": OrderedDict(),
            "_cache_pid": None,
        }
