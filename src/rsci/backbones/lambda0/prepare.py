"""将现有 LeRobot-v2 元数据交给官方 prepare；不复制数据或重写归一化。"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from rsci.paths import dataset_paths

from .contracts import RUNTIME, sha256, verify_upstream


def safe_bundle(output, data_root=None):
    """只对分位数退化且真实变化的通道回退训练 min/max；其余官方参数逐值保留。"""
    import shutil

    import numpy as np
    from lambda0.data.prepare import raw_episode
    from lambda0.runtime.paths import metadata_paths

    from .contracts import SAFE_NORMALIZATION

    destination = output.with_name(output.name + "-safe")
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(output, destination)
    for path in sorted((destination / "train").glob("*.json")):
        raw = json.loads(path.read_text())
        meta = metadata_paths(raw, path, data_root or RUNTIME / "data")
        rows = [raw_episode(meta, item)[:2] for item in meta["datalist"]]
        stats_path = destination / "stats" / path.name
        stats = json.loads((output / "stats" / path.name).read_text())
        changes = {}
        for kind, slot in (("proprio", 0), ("action", 1)):
            values = np.concatenate([row[slot] for row in rows])
            block = stats[kind]
            low, high = np.array(block["q01"]), np.array(block["q99"])
            minimum, maximum = values.min(axis=0), values.max(axis=0)
            span = maximum - minimum
            changed = (high - low < np.maximum(1e-5, span * 0.01)) & (span > 1e-5)
            changes[kind] = np.flatnonzero(changed).tolist()
            low[changed], high[changed] = minimum[changed], maximum[changed]
            scale = (high - low + 1e-6) / 2
            block.update(
                q01=low.tolist(),
                q99=high.tolist(),
                scale=scale.tolist(),
                location=(low + scale).tolist(),
            )
        stats.update(normalization_contract=SAFE_NORMALIZATION, rsci_degenerate_fallback=changes)
        stats_path.write_text(json.dumps(stats, indent=2))
        for split in ("train", "validation"):
            file = destination / split / path.name
            if file.exists():
                metadata = json.loads(file.read_text())
                metadata["normalization_contract"] = SAFE_NORMALIZATION
                file.write_text(json.dumps(metadata, indent=2))
    print(f"Prepared {destination}; explicit safe-normalization derivative, no training started.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--sidecar", type=Path, help="Formal ARDY sidecar directory")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    verify_upstream()
    paths = dataset_paths(
        SimpleNamespace(task=args.task, dataset=args.dataset, sidecar=args.sidecar, model="VGQ-AE")
    )
    root = paths["dataset"]
    info = json.loads((root / "meta/info.json").read_text())
    items = [
        json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line
    ]
    episodes, frames = len(items), sum(item["length"] for item in items)
    if info["fps"] != 50 or (episodes, frames) != (info["total_episodes"], info["total_frames"]):
        raise ValueError("Incomplete dataset or incompatible frame rate")

    for item in items:
        tasks = item.get("tasks", [])
        if len(tasks) != 1:
            raise ValueError("SIMPLE metadata requires exactly one task per episode")
        item["task"] = tasks[0]
    output = args.output or args.runtime_root / "bundles/lambda0" / args.task
    if output.exists() or output.with_name(output.name + "-safe").exists():
        raise FileExistsError(output)
    source = output.parent / f"{args.task}-source"
    source.mkdir(parents=True, exist_ok=True)
    meta = dict(
        dataset_name=f"rsci_simple_{args.task}",
        handler="lambda0.data.domain_handler.psi_simple:PsiSimpleHandler",
        embodiment_id=7,
        action_space="robot_g1_simple_sim",
        coordinate_contract="psi_simple_absolute_v1",
        normalization_contract="quantile_q01_q99_v1",
        root_path=str(root),
        storage_layout="episode_v2",
        data_path=info["data_path"],
        video_path=info["video_path"],
        fps=50,
        chunks_size=info.get("chunks_size", 1000),
        episode_count=episodes,
        total_frames=frames,
        datalist=items,
        rsci_sidecar=str(paths["sidecar"]),
    )
    (source / "metadata.json").write_text(json.dumps(meta, indent=2))
    configuration = source.parent / f"{args.task}-source.json"
    configuration.write_text(
        json.dumps(dict(stage="posttrain", data=dict(metas_path=str(source))), indent=2)
    )
    from lambda0.data.prepare import main as official_prepare

    official_prepare(
        [
            "--source-config",
            str(configuration),
            "--data-root",
            str(args.runtime_root / "data"),
            "--output",
            str(output),
            "--seed",
            str(args.seed),
        ]
    )
    # bundle 同时服务 STATE/VGQ；只在 VGQ 运行时换 handler，统计/划分完全相同。
    fingerprints = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted((root / "meta").glob("*"))
        + sorted((root / "data").rglob("*.parquet"))
        + sorted((root / "videos").rglob("*.mp4"))
        if path.is_file()
    }
    (output / "rsci_provenance.json").write_text(
        json.dumps(
            dict(
                dataset_root=str(root),
                task=args.task,
                files=fingerprints,
                sidecar_context=sha256(paths["sidecar"] / "manifest.json"),
            ),
            indent=2,
        )
    )
    safe_bundle(output, args.runtime_root / "data")


if __name__ == "__main__":
    main()
