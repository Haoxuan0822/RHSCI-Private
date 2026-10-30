"""复用ACT原始uint8和小张量mmap；不缓存增强、归一化或可训练表征。"""

import numpy as np

from .cache_config import InputCacheConfig
from .raw_input_cache import RawInputCache, safe_path
from .raw_input_cache import cache_identity as raw_cache_identity


def cache_identity(root, *, config=InputCacheConfig()):
    if config.root is None:
        return None
    from .ardy_sidecar import dataset_content_fingerprint

    identity = raw_cache_identity(root, dataset_content_fingerprint(root), config=config)
    # 保留各骨干已有契约字段，验收级别不改变缓存身份。
    return dict(
        directory=identity["directory"],
        manifest_sha256=identity["manifest_sha256"],
        fingerprint=identity["fingerprint"],
        pixels="raw-uint8-no-augmentation",
    )


class InputCache(RawInputCache):
    def frames(self, episode, indices):
        record = self.manifest["episodes"][str(episode)]
        key = str(episode)
        if key not in self.arrays:
            pixels = np.load(
                safe_path(self.directory, record["pixels"]), mmap_mode="r", allow_pickle=False
            )
            if pixels.dtype != np.uint8 or list(pixels.shape) != record["shape"]:
                raise ValueError("Input cache RGB layout changed")
            self.arrays[key] = pixels
        return self.arrays[key][indices].transpose(0, 2, 3, 1).copy()

    def episode_table(self, episode, episodes, include_state=False):
        start = 0
        for record in episodes:
            if record["episode_index"] == episode["episode_index"]:
                break
            start += record["length"]
        stop = start + episode["length"]
        keys = ["action", "episode_index", "frame_index", "task_index"] + (
            ["states"] if include_state else []
        )
        return {key: self.table.columns[key][start:stop].numpy() for key in keys}
