"""原始 RGB 的磁盘共享只读缓存，不缓存在线增强。"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RGBFrameCache:
    """每个进程仅 mmap 同一份 uint8 文件；采样后复制四帧，不复制整条轨迹。"""

    def __init__(self, root, dataset_root, *, require_verified=True):
        self.root = Path(root).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("version") != 1 or not self.manifest.get("complete"):
            raise ValueError("RGB cache must be complete and version 1")
        if require_verified and not self.manifest.get("pixel_exact_verified"):
            raise ValueError("RGB cache must pass pixel-exact verification first")
        codec_version = self.manifest.get("torchcodec_version")
        if codec_version and codec_version != importlib.metadata.version("torchcodec"):
            raise ValueError("RGB cache decoder version mismatch; rebuild and verify")
        if self.manifest["info_sha256"] != sha256_file(self.dataset_root / "meta/info.json"):
            raise ValueError("RGB cache dataset metadata mismatch")
        self._maps = {}
        self._pid = os.getpid()
        # 启动时核验全部视频身份；未缓存的视频不得静默混用旧缓存。
        for relative, entry in self.manifest["videos"].items():
            video = self.dataset_root / relative
            if sha256_file(video) != entry["video_sha256"]:
                raise ValueError(f"RGB cache video identity mismatch: {relative}")
            for key in ("frames", "timestamps"):
                path = self.root / entry[key]
                if path.stat().st_size != entry[key + "_bytes"]:
                    raise ValueError(f"RGB cache truncated: {path}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = {}  # spawn worker 不序列化已映射的整条视频。
        return state

    def read(self, video_path, timestamps, tolerance_s):
        if self._pid != os.getpid():
            self._maps = {}  # fork 后重新建立本进程只读映射。
            self._pid = os.getpid()
        relative = Path(video_path).resolve().relative_to(self.dataset_root).as_posix()
        entry = self.manifest["videos"].get(relative)
        if entry is None:
            raise ValueError(f"RGB cache missing video: {relative}")
        if relative not in self._maps:
            frames = np.load(self.root / entry["frames"], mmap_mode="r", allow_pickle=False)
            pts = np.load(self.root / entry["timestamps"], mmap_mode="r", allow_pickle=False)
            if frames.dtype != np.uint8 or list(frames.shape) != entry["shape"]:
                raise ValueError("RGB cache frame schema mismatch")
            if pts.shape != (len(frames),) or not np.isfinite(pts).all():
                raise ValueError("RGB cache timestamp schema mismatch")
            self._maps[relative] = frames, pts
        frames, pts = self._maps[relative]
        # 与当前 LeRobot torchcodec 路径完全相同：round(ts*fps)，再在请求帧中匹配 PTS。
        indices = [round(ts * entry["average_fps"]) for ts in timestamps]
        if not indices or min(indices) < 0 or max(indices) >= len(frames):
            raise ValueError("RGB cache query frame out of range")
        query = torch.tensor(timestamps)
        loaded = torch.tensor(pts[indices].tolist())
        distance = torch.cdist(query[:, None], loaded[:, None], p=1)
        minimum, nearest = distance.min(1)
        if not bool((minimum < tolerance_s).all()):
            raise ValueError("RGB cache query timestamp outside tolerance")
        selected = np.asarray(indices)[nearest.numpy()]
        return torch.from_numpy(frames[selected].copy()).float().div_(255)
