"""Prepare RHSCI normalization statistics from a processed SIMPLE G1 training dataset."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def prepare_stats(dataset, *, overwrite=False):
    root = Path(dataset).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    dimensions = {"action": 36, "states": 32}
    if info.get("fps") != 50:
        raise ValueError("Processed SIMPLE data must declare 50 FPS")
    for key, size in dimensions.items():
        if info.get("features", {}).get(key, {}).get("shape") != [size]:
            raise ValueError(f"Processed SIMPLE {key} must declare shape [{size}]")
    paths = sorted((root / "data").glob("*/*.parquet"))
    if not paths:
        raise ValueError(f"No LeRobot episode parquet files in {root / 'data'}")
    values = {key: [] for key in dimensions}
    for path in paths:
        table = pq.read_table(path, columns=list(dimensions)).to_pydict()
        lengths = set()
        for key, size in dimensions.items():
            array = np.asarray(table[key], dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != size or not len(array):
                raise ValueError(f"Invalid {key} shape in {path}: {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite {key} in {path}")
            values[key].append(array)
            lengths.add(len(array))
        if len(lengths) != 1:
            raise ValueError(f"Action/state frame counts differ in {path}")
    stats = {}
    for key, arrays in values.items():
        array = np.concatenate(arrays)
        stats[key] = {
            "min": array.min(axis=0).tolist(),
            "max": array.max(axis=0).tolist(),
            "mean": array.mean(axis=0).tolist(),
            "std": array.std(axis=0).tolist(),
            "q01": np.quantile(array, 0.01, axis=0).tolist(),
            "q99": np.quantile(array, 0.99, axis=0).tolist(),
        }
    pending = []
    for name in ("stats.json", "stats_psi0.json"):
        destination = root / "meta" / name
        existing = json.loads(destination.read_text()) if destination.exists() else {}
        if not isinstance(existing, dict):
            raise ValueError(f"Expected statistics object: {destination}")
        if destination.exists() and not overwrite:
            for key in dimensions:
                for metric, value in stats[key].items():
                    saved = np.asarray(existing.get(key, {}).get(metric, []))
                    if saved.shape != np.shape(value) or not np.allclose(
                        saved, value, rtol=1e-6, atol=1e-7
                    ):
                        raise FileExistsError(
                            f"Conflicting {key}.{metric} in {destination}; "
                            "keep checkpoint statistics or explicitly use --overwrite for a new run"
                        )
            continue
        pending.append((destination, dict(existing, **stats)))
    for destination, payload in pending:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as output:
            temporary = Path(output.name)
            try:
                json.dump(payload, output, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace stats for a new run")
    args = parser.parse_args()
    prepare_stats(args.dataset, overwrite=args.overwrite)
    print(f"RHSCI normalization ready: {args.dataset / 'meta'}")


if __name__ == "__main__":
    main()
