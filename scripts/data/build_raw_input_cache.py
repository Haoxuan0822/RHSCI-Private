"""Build full immutable input cache, publish atomically, retain source videos."""

import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from rhsci.data.ardy_sidecar import dataset_content_fingerprint
from rhsci.data.cache_config import InputCacheConfig, validate_cache_dataset_names
from rhsci.data.lerobot import RGB_KEY, load_raw_lerobot_dataset
from rhsci.data.raw_input_cache import SCHEMA, cache_identity
from rhsci.runtime.artifacts import atomic_json, sha256

CODE = Path(__file__).resolve().parents[2]


def build_video(args):
    from torchcodec.decoders import VideoDecoder

    (root, pending, ep, length, relative) = args
    torch.set_num_threads(1)
    (root, pending) = (Path(root), Path(pending))
    source = root / relative
    stat = source.stat()
    source_hash = sha256(source)
    decoder = VideoDecoder(str(source), device="cpu", seek_mode="approximate", num_ffmpeg_threads=1)
    n = len(decoder)
    fps = decoder.metadata.average_fps
    if n != length or fps != 50:
        raise ValueError((ep, n, length, fps))
    pixel_name = f"episode-{ep:06d}.npy"
    pts_name = f"episode-{ep:06d}-pts.npy"
    pixels = np.lib.format.open_memmap(
        pending / pixel_name, mode="w+", dtype=np.uint8, shape=(n, 3, 360, 640)
    )
    pts = np.lib.format.open_memmap(pending / pts_name, mode="w+", dtype=np.float64, shape=(n,))
    for first in range(0, n, 64):
        batch = decoder.get_frames_at(indices=list(range(first, min(n, first + 64))))
        pixels[first : first + len(batch.data)] = batch.data.numpy()
        pts[first : first + len(batch.data)] = batch.pts_seconds.numpy()
    if not np.isfinite(pts).all() or not np.all(np.diff(pts) > 0):
        raise ValueError("Invalid full cached PTS")
    # 独立解码器覆盖尾帧、重复帧及随机乱序，像素与PTS均须完全一致。
    query = [n - 1, 0, min(6, n - 1), 0] + torch.randperm(
        n, generator=torch.Generator().manual_seed(2026 + ep)
    )[:16].tolist()
    other = VideoDecoder(str(source), device="cpu", seek_mode="approximate", num_ffmpeg_threads=1)
    check = other.get_frames_at(indices=query)
    if not np.array_equal(pixels[query], check.data.numpy()) or not np.array_equal(
        pts[query], check.pts_seconds.numpy()
    ):
        raise ValueError("Random seek pixels/PTS differ")
    pixels.flush()
    pts.flush()
    del pixels, pts
    if source.stat().st_size != stat.st_size or source.stat().st_mtime_ns != stat.st_mtime_ns:
        raise ValueError("Source changed while building")
    files = {
        name: dict(size=(pending / name).stat().st_size, sha256=sha256(pending / name))
        for name in [pixel_name, pts_name]
    }
    return (
        str(ep),
        dict(
            source=relative,
            source_stat=[stat.st_size, stat.st_mtime_ns],
            source_sha256=source_hash,
            frames=n,
            fps=fps,
            shape=[n, 3, 360, 640],
            pixels=pixel_name,
            pts=pts_name,
            random_pixel_pts_queries=len(query),
        ),
        files,
    )


def build_cache(root, cache_root, workers):
    """逐库原子发布，保留原始视频；已建缓存核验通过后才能复用。"""
    target = cache_root / root.name
    with (cache_root / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            cache_identity(root, dataset_content_fingerprint(root), config=InputCacheConfig(cache_root))
            print("ALREADY_BUILT", target, flush=True)
            return
        fingerprint = dataset_content_fingerprint(root)
        # 保留标准 <父目录>/<数据集名> 布局，在暂存区复用正式完整校验。
        staging = Path(tempfile.mkdtemp(prefix=root.name + ".building-", dir=cache_root))
        pending = staging / root.name
        pending.mkdir()
        ds = load_raw_lerobot_dataset(root)
        hf = ds.hf_dataset.select_columns(
            [k for k in ds.hf_dataset.column_names if not k.startswith("observation.depth.")]
        )
        columns = {k: torch.stack(v).contiguous() for (k, v) in hf[:].items()}
        if not all((torch.isfinite(v).all() for v in columns.values() if v.is_floating_point())):
            raise ValueError("Nonfinite cache table")
        torch.save(columns, pending / "table.pt")
        manifest = dict(
            schema=SCHEMA,
            dataset=root.name,
            fingerprint=fingerprint,
            rows=len(ds),
            columns=list(columns),
            table="table.pt",
            episodes={},
            files={
                "table.pt": dict(
                    size=(pending / "table.pt").stat().st_size, sha256=sha256(pending / "table.pt")
                )
            },
        )
        jobs = []
        episodes = [
            json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()
        ]
        for ep in episodes:
            relative = str(ds.meta.get_video_file_path(ep["episode_index"], RGB_KEY))
            jobs.append((str(root), str(pending), ep["episode_index"], ep["length"], relative))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for i, (ep, record, files) in enumerate(pool.map(build_video, jobs), 1):
                manifest["episodes"][ep] = record
                manifest["files"].update(files)
                print(
                    json.dumps(
                        dict(
                            task=root.name,
                            episodes_done=i,
                            episodes_total=len(jobs),
                            frames_done=sum((x["frames"] for x in manifest["episodes"].values())),
                        )
                    ),
                    flush=True,
                )
        atomic_json(pending / "manifest.json", manifest)
        identity = cache_identity(root, fingerprint, config=InputCacheConfig(staging))
        if dataset_content_fingerprint(root) != fingerprint:
            raise ValueError("Source dataset changed while building input cache")
        # 表、视频及源数据身份全部核验通过后才原子发布。
        os.rename(pending, target)
        staging.rmdir()
        identity["directory"] = str(target.resolve())
        atomic_json(
            cache_root / f"{root.name}-verified.json",
            dict(
                time=time.time(),
                identity=identity,
                bytes=sum((x["size"] for x in manifest["files"].values())),
                episodes=len(episodes),
                frames=len(ds),
            ),
        )
        print("CACHE_VERIFIED", root.name, json.dumps(identity), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    datasets = ap.add_mutually_exclusive_group(required=True)
    datasets.add_argument("--dataset", type=Path)
    datasets.add_argument("--datasets", type=Path, nargs="+", help="Build one cache per dataset")
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    cli = ap.parse_args()
    if cli.workers < 1:
        ap.error("--workers must be positive")
    cache_root = cli.cache_root.resolve()
    if cache_root.is_relative_to(CODE):
        ap.error("Cache must live outside the source checkout")
    roots = [root.resolve(strict=True) for root in (cli.datasets or [cli.dataset])]
    if len(set(roots)) != len(roots):
        ap.error("Dataset roots must not be duplicated")
    validate_cache_dataset_names(roots)
    # 生成时必须从原始视频读取，不能继承当前进程的缓存配置。
    torch.set_num_threads(1)
    cache_root.mkdir(parents=True, exist_ok=True)
    for root in roots:
        build_cache(root, cache_root, cli.workers)


if __name__ == "__main__":
    main()
