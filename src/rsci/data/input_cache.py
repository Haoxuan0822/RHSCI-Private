"""复用ACT原始uint8和小张量mmap；不缓存增强、归一化或可训练表征。"""
import os
from pathlib import Path
import numpy as np

from .raw_input_cache import RawInputCache, read_manifest, safe_path
from rsci.runtime.artifacts import sha256


def cache_identity(root, *, env_var="RSCI_INPUT_CACHE"):
    # 按骨干选择缓存环境变量；共享格式但不共享训练配方。
    cache_root = os.environ.get(env_var)
    if not cache_root:
        return None
    from .ardy_sidecar import dataset_content_fingerprint
    directory, value, digest = read_manifest(cache_root, str(Path(root).resolve()))
    if value['fingerprint'] != dataset_content_fingerprint(root):
        raise ValueError('Input cache dataset fingerprint differs')
    # Guardian fully authenticates cache/source bytes. Workers retain cheap inventory/stat gates.
    if int(os.environ.get('WORLD_SIZE', '1')) == 1:
        for name, record in value['files'].items():
            if sha256(safe_path(directory, name)) != record['sha256']:
                raise ValueError(f'Corrupt input cache: {name}')
        for record in value['episodes'].values():
            if sha256(safe_path(root, record['source'])) != record['source_sha256']:
                raise ValueError('Input cache source content changed')
    return dict(directory=str(directory), manifest_sha256=digest,
                fingerprint=value['fingerprint'], pixels='raw-uint8-no-augmentation')


class InputCache(RawInputCache):
    def frames(self, episode, indices):
        record = self.manifest['episodes'][str(episode)]
        key = str(episode)
        if key not in self.arrays:
            pixels = np.load(safe_path(self.directory, record['pixels']), mmap_mode='r', allow_pickle=False)
            if pixels.dtype != np.uint8 or list(pixels.shape) != record['shape']:
                raise ValueError('Input cache RGB layout changed')
            self.arrays[key] = pixels
        return self.arrays[key][indices].transpose(0, 2, 3, 1).copy()

    def episode_table(self, episode, episodes, include_state=False):
        start = 0
        for record in episodes:
            if record['episode_index'] == episode['episode_index']:
                break
            start += record['length']
        stop = start + episode['length']
        keys = ['action', 'episode_index', 'frame_index', 'task_index'] + (['states'] if include_state else [])
        return {key: self.table.columns[key][start:stop].numpy() for key in keys}
