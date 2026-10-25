"""仅缓存增强前uint8像素和CPU解码器；不缓存增强、标签或随机状态。"""

import os
from collections import OrderedDict
from pathlib import Path

import torch

from psi.data.lerobot.compat import LeRobotDataset


class VideoCache:
    def __init__(self, max_decoders=16, max_frame_bytes=64 * 1024 * 1024):
        self.max_decoders = max_decoders
        self.max_frame_bytes = max_frame_bytes
        self._reset()

    def _reset(self):
        # fork/spawn后不能继承FFmpeg句柄；每个worker独立惰性创建。
        self.pid = os.getpid()
        self.decoders = OrderedDict()
        self.frames = OrderedDict()
        self.frame_bytes = 0

    def __getstate__(self):
        return dict(max_decoders=self.max_decoders, max_frame_bytes=self.max_frame_bytes)

    def __setstate__(self, state):
        self.__init__(**state)

    def decode(self, path, timestamps, tolerance_s):
        from torchcodec.decoders import VideoDecoder

        if self.pid != os.getpid():
            self._reset()
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        identity = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity in self.decoders:
            decoder = self.decoders.pop(identity)
        else:
            decoder = VideoDecoder(str(path), device="cpu", seek_mode="approximate")
        self.decoders[identity] = decoder
        while len(self.decoders) > self.max_decoders:
            self.decoders.popitem(last=False)
        # 沿用原生fps取整、PTS容差、最近邻选择与float32/255转换。
        indices = [round(ts * decoder.metadata.average_fps) for ts in timestamps]
        missing = list(dict.fromkeys(i for i in indices if (identity, i) not in self.frames))
        # 先固定本次窗口的引用，避免小缓存淘汰本窗口稍后还需读取的重复帧。
        fresh = {i: self.frames[(identity, i)] for i in indices if (identity, i) in self.frames}
        if missing:
            batch = decoder.get_frames_at(indices=missing)
            fresh.update(
                {
                    i: (frame.clone(), pts.item())
                    for i, frame, pts in zip(missing, batch.data, batch.pts_seconds, strict=True)
                }
            )
        selected = []
        for i in indices:
            key = (identity, i)
            if key in self.frames:
                value = self.frames.pop(key)
                self.frame_bytes -= value[0].numel() * value[0].element_size()
            else:
                value = fresh[i]
            selected.append(value)
            size = value[0].numel() * value[0].element_size()
            if size <= self.max_frame_bytes:
                self.frames[key] = value
                self.frame_bytes += size
            while self.frame_bytes > self.max_frame_bytes:
                old, _ = self.frames.popitem(last=False)[1]
                self.frame_bytes -= old.numel() * old.element_size()
        loaded_ts = torch.tensor([pts for _, pts in selected])
        distances = torch.cdist(torch.tensor(timestamps)[:, None], loaded_ts[:, None], p=1)
        minimum, nearest = distances.min(1)
        if not (minimum < tolerance_s).all():
            raise AssertionError(
                f"Video timestamp tolerance exceeded: {path}, {minimum}, {tolerance_s}"
            )
        # stack产生独立张量，后续处理不能写坏缓存内容。
        return torch.stack([selected[i][0] for i in nearest]).float() / 255


class CachedLeRobotDataset(LeRobotDataset):
    def _query_videos(self, query_timestamps, ep_idx):
        if self.video_backend != "torchcodec":
            return super()._query_videos(query_timestamps, ep_idx)
        if not hasattr(self, "_rsci_video_cache"):
            self._rsci_video_cache = VideoCache()
        return {
            key: self._rsci_video_cache.decode(
                self.root / self.meta.get_video_file_path(ep_idx, key), times, self.tolerance_s
            ).squeeze(0)
            for key, times in query_timestamps.items()
        }
