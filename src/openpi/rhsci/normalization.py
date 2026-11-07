"""Official RunningStats over LeRobot state and terminal-padded action queries.

Only numeric columns are loaded: RGB/text transforms in HfmInputs do not change
these values. Batch order and drop-tail follow openpi.compute_norm_stats, without
requiring its JAX/model/video dependency graph or retaining the corpus in RAM.
"""

import json

import numpy as np
import pyarrow.parquet as pq

from openpi.shared.normalize import RunningStats

from .data import HORIZON


def numeric_episodes(roots):
    """Read metadata order, never discover extra/unlisted parquet files by glob."""
    for root in roots:
        info = json.loads((root / "meta/info.json").read_text())
        episodes = [
            json.loads(line)
            for line in (root / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if (
            info["fps"] != 50
            or len(episodes) != info["total_episodes"]
            or sum(ep["length"] for ep in episodes) != info["total_frames"]
        ):
            raise ValueError(f"Invalid joint dataset metadata: {root}")
        seen = set()
        for ep in episodes:
            index, length = ep["episode_index"], ep["length"]
            if index in seen or length < 1:
                raise ValueError(f"Invalid episode identity/length: {root}/{index}")
            seen.add(index)
            path = root / info["data_path"].format(
                episode_index=index, episode_chunk=index // info.get("chunks_size", 1000)
            )
            table = pq.read_table(
                path, columns=["action", "states", "frame_index", "episode_index"]
            ).to_pydict()
            if not np.array_equal(table["frame_index"], np.arange(length)) or set(
                table["episode_index"]
            ) != {index}:
                raise ValueError(f"Parquet episode/frame mismatch: {path}")
            arrays = {key: np.asarray(table[key], dtype=np.float32) for key in ("action", "states")}
            for key, width in (("action", 36), ("states", 32)):
                if arrays[key].shape != (length, width) or not np.isfinite(arrays[key]).all():
                    raise ValueError(f"Invalid numeric column: {path}/{key}")
            yield arrays


def joint_statistics(roots, batch_size):
    """Use native 5000-bin streaming quantiles and float32 action chunks.

    Batching crosses episode/root boundaries, but each action query is clamped within
    its own episode. The last incomplete *global* batch is excluded, as upstream.
    """
    if batch_size < 1:
        raise ValueError("statistics-batch-size must be positive")
    running = {key: RunningStats() for key in ("action", "states")}
    bounds = {key: [None, None] for key in running}
    pending = {key: [] for key in running}
    count = batches = 0
    for arrays in numeric_episodes(roots):
        length = len(arrays["action"])
        for start in range(0, length, batch_size):
            frames = np.arange(start, min(start + batch_size, length))
            pending["states"].extend(arrays["states"][frames])
            pending["action"].extend(
                arrays["action"][np.minimum(frames[:, None] + np.arange(HORIZON), length - 1)]
            )
            count += len(frames)
            if count >= batch_size:
                for key in running:
                    batch = np.stack(pending[key][:batch_size])
                    del pending[key][:batch_size]
                    running[key].update(batch)
                    flat = batch.reshape(-1, batch.shape[-1])
                    lo, hi = bounds[key]
                    bounds[key] = [
                        flat.min(0) if lo is None else np.minimum(lo, flat.min(0)),
                        flat.max(0) if hi is None else np.maximum(hi, flat.max(0)),
                    ]
                batches += 1
                count -= batch_size
    if batches == 0 or batches * batch_size < 2:
        raise ValueError(
            "Joint statistics need at least one full batch and two frames; lower --statistics-batch-size"
        )
    result = {}
    for key, accumulator in running.items():
        stats = accumulator.get_statistics()
        result[key] = {
            name: np.asarray(getattr(stats, name)).tolist()
            for name in ("mean", "std", "q01", "q99")
        }
        result[key].update(min=bounds[key][0].tolist(), max=bounds[key][1].tolist())
    return result
