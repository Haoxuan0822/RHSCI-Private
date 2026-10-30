"""Build full immutable input cache, publish atomically, retain source videos."""

import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import torch

from rsci.backbones.act.data import IMAGE, ACTDataset
from rsci.backbones.act.train import parser, resolve
from rsci.data.ardy_sidecar import dataset_content_fingerprint
from rsci.data.cache_config import InputCacheConfig
from rsci.data.raw_input_cache import SCHEMA, cache_identity
from rsci.runtime.artifacts import atomic_json, sha256

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    cli = ap.parse_args()
    if cli.workers < 1:
        ap.error("--workers must be positive")
    CACHE = cli.cache_root.resolve()
    if CACHE.is_relative_to(CODE):
        ap.error("Cache must live outside the source checkout")
    # 生成时必须从原始视频读取，不能继承当前进程的缓存配置。
    torch.set_num_threads(1)
    paths = resolve(
        parser().parse_args(
            ["--task", cli.dataset.name, "--model", "STATE", "--dataset", str(cli.dataset)]
        )
    )
    root = paths["dataset"]
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / root.name
    with (CACHE / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            print("ALREADY_BUILT", target, flush=True)
            return
        pending = CACHE / (root.name + f".building-{os.getpid()}")
        pending.mkdir()
        ds = ACTDataset(root, None, "STATE", 2026)
        hf = ds.native.hf_dataset.select_columns(
            [k for k in ds.native.hf_dataset.column_names if not k.startswith("observation.depth.")]
        )
        columns = {k: torch.stack(v).contiguous() for (k, v) in hf[:].items()}
        if not all((torch.isfinite(v).all() for v in columns.values() if v.is_floating_point())):
            raise ValueError("Nonfinite cache table")
        torch.save(columns, pending / "table.pt")
        manifest = dict(
            schema=SCHEMA,
            dataset=root.name,
            fingerprint=dataset_content_fingerprint(root),
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
            relative = str(ds.native.meta.get_video_file_path(ep["episode_index"], IMAGE))
            jobs.append((str(root), str(pending), ep["episode_index"], ep["length"], relative))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=cli.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for i, (ep, record, files) in enumerate(pool.map(build_video, jobs), 1):
                manifest["episodes"][ep] = record
                manifest["files"].update(files)
                print(
                    json.dumps(
                        dict(
                            task=cli.dataset.name,
                            episodes_done=i,
                            episodes_total=len(jobs),
                            frames_done=sum((x["frames"] for x in manifest["episodes"].values())),
                        )
                    ),
                    flush=True,
                )
        atomic_json(pending / "manifest.json", manifest)
        # 全量写入和逐视频核验完成后原子发布；原始视频始终保留。
        os.rename(pending, target)
        identity = cache_identity(root, manifest["fingerprint"], config=InputCacheConfig(CACHE))
        atomic_json(
            CACHE / f"{root.name}-verified.json",
            dict(
                time=time.time(),
                identity=identity,
                bytes=sum((x["size"] for x in manifest["files"].values())),
                episodes=len(episodes),
                frames=len(ds),
            ),
        )
        print("CACHE_VERIFIED", cli.dataset.name, json.dumps(identity), flush=True)


if __name__ == "__main__":
    main()
