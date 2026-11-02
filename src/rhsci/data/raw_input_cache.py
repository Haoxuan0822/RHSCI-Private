"""Read-only input caches: original table values and pre-augmentation RGB."""

import json
from pathlib import Path

import numpy as np
import torch

from rhsci.runtime.artifacts import sha256

from .cache_config import InputCacheConfig

SCHEMA = "act-raw-input-cache-v1"


def safe_path(root, name):
    path = Path(root) / name
    if (
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or not path.resolve().is_relative_to(Path(root).resolve())
    ):
        raise ValueError("Invalid input-cache relative path")
    return path


def read_manifest(cache_root, dataset_root):
    dataset_root = Path(dataset_root).resolve()
    directory = Path(cache_root).resolve() / dataset_root.name
    manifest_path = directory / "manifest.json"
    value = json.loads(manifest_path.read_text())
    if value["schema"] != SCHEMA or value["dataset"] != dataset_root.name:
        raise ValueError("Input cache dataset/schema mismatch")
    episodes = [
        json.loads(line) for line in (dataset_root / "meta/episodes.jsonl").read_text().splitlines()
    ]
    expected = {str(e["episode_index"]): e["length"] for e in episodes}
    if set(value["episodes"]) != set(expected) or value["rows"] != sum(expected.values()):
        raise ValueError("Input cache episode/row inventory mismatch")
    for ep, length in expected.items():
        record = value["episodes"][ep]
        if (
            record["frames"] != length
            or record["shape"] != [length, 3, 360, 640]
            or record["fps"] != 50
        ):
            raise ValueError("Input cache video layout differs from native dataset")
        source = safe_path(dataset_root, record["source"])
        stat = source.stat()
        if [stat.st_size, stat.st_mtime_ns] != record["source_stat"]:
            raise ValueError("Input cache source video changed")
    for name, record in value["files"].items():
        if safe_path(directory, name).stat().st_size != record["size"]:
            raise ValueError("Input cache payload size mismatch")
    if value["table"] not in value["files"]:
        raise ValueError("Input cache table is not authenticated")
    for record in value["episodes"].values():
        if record["pixels"] not in value["files"] or record["pts"] not in value["files"]:
            raise ValueError("Input cache RGB/PTS is not authenticated")
    return directory, value, sha256(manifest_path)


def cache_identity(dataset_root, dataset_fingerprint, *, config=InputCacheConfig()):
    cache_root = config.root
    if not cache_root:
        return None
    directory, value, digest = read_manifest(cache_root, str(Path(dataset_root).resolve()))
    if value["fingerprint"] != dataset_fingerprint:
        raise ValueError("Input cache content fingerprint mismatch")
    if config.verification == "full":
        for name, record in value["files"].items():
            if sha256(safe_path(directory, name)) != record["sha256"]:
                raise ValueError(f"Corrupt input cache: {name}")
        for record in value["episodes"].values():
            if sha256(safe_path(dataset_root, record["source"])) != record["source_sha256"]:
                raise ValueError("Input cache video content changed")
    return dict(
        schema=SCHEMA,
        directory=str(directory),
        manifest_sha256=digest,
        fingerprint=value["fingerprint"],
        pixels="native-torchcodec-uint8",
        augmentation="uncached-native-epoch-index-rng",
    )


class TensorRows:
    """Narrow native LeRobot table interface backed by shared file pages."""

    def __init__(self, columns, features=None, indices=None):
        self.columns, self.features, self.indices = columns, features, indices
        self.column_names = list(columns)

    def __len__(self):
        return (
            len(self.indices)
            if self.indices is not None
            else len(next(iter(self.columns.values())))
        )

    def __getitem__(self, key):
        if isinstance(key, str):
            values = self.columns[key]
            if self.indices is not None:
                values = values[self.indices]
            # Native callers stack these rows, producing an independent tensor.
            return list(values.unbind())
        index = self.indices[key] if self.indices is not None else key
        return {k: v[index].clone() for k, v in self.columns.items()}

    def select(self, indices):
        if self.indices is not None:
            indices = [self.indices[i] for i in indices]
        return TensorRows(self.columns, self.features, indices)


class RawInputCache:
    """Open read-only mmap pixels lazily; spawn serializes paths, never arrays."""

    def __init__(self, cache_root, dataset_root, features=None):
        self.cache_root, self.dataset_root = str(cache_root), str(Path(dataset_root).resolve())
        self.features = features
        self.directory, self.manifest, self.manifest_sha = read_manifest(
            self.cache_root, self.dataset_root
        )
        self.arrays = {}
        columns = torch.load(
            safe_path(self.directory, self.manifest["table"]),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if set(columns) != set(self.manifest["columns"]) or any(
            len(v) != self.manifest["rows"] for v in columns.values()
        ):
            raise ValueError("Input cache table columns/row count mismatch")
        self.table = TensorRows(columns, features)
        self.by_source = {
            str(safe_path(self.dataset_root, record["source"]).resolve()): record
            for record in self.manifest["episodes"].values()
        }

    def __getstate__(self):
        return dict(
            cache_root=self.cache_root, dataset_root=self.dataset_root, features=self.features
        )

    def __setstate__(self, state):
        self.__init__(**state)

    def decode(self, path, timestamps, tolerance_s):
        path = str(Path(path).resolve())
        if path not in self.by_source:
            raise ValueError("Video absent from complete input cache")
        record = self.by_source[path]
        stat = Path(path).stat()
        if [stat.st_size, stat.st_mtime_ns] != record["source_stat"]:
            raise ValueError("Cached source video changed during training")
        if path not in self.arrays:
            pixels = np.load(
                safe_path(self.directory, record["pixels"]), mmap_mode="r", allow_pickle=False
            )
            pts = np.load(
                safe_path(self.directory, record["pts"]), mmap_mode="r", allow_pickle=False
            )
            if (
                pixels.dtype != np.uint8
                or list(pixels.shape) != record["shape"]
                or pts.dtype != np.float64
                or pts.shape != (record["frames"],)
            ):
                raise ValueError("Cached RGB/PTS tensor layout changed")
            self.arrays[path] = pixels, pts
        pixels, pts = self.arrays[path]
        indices = [round(ts * record["fps"]) for ts in timestamps]
        loaded_ts = torch.tensor([float(pts[i]) for i in indices])
        minimum, nearest = torch.cdist(
            torch.tensor(timestamps)[:, None], loaded_ts[:, None], p=1
        ).min(1)
        if not (minimum < tolerance_s).all():
            raise AssertionError("Cached video timestamp tolerance exceeded")
        # Stack copies uint8 pixels, preserving duplicate/reordered indices and the
        # native float32/255 conversion. Augmentations cannot write into the mmap.
        return (
            torch.from_numpy(np.stack([pixels[indices[i]] for i in nearest.tolist()])).float() / 255
        )
