"""用权重原v2公式审计动作标签损失；不改统计、模型或数据。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.rhsci.processing import Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    saved = json.loads((args.checkpoint / "rsci_config.json").read_text())
    if saved["normalization"] != "simple-quantile-zero-span-clipped-v2":
        raise ValueError("This diagnosis is specific to the saved v2 checkpoint")
    processor = Processor(
        args.dataset, args.tokenizer, saved["variant"], normalization=saved["normalization"]
    )
    stats = processor.stats["action"]
    low, high = (np.asarray(stats[key], np.float32) for key in ("q01", "q99"))

    def roundtrip(actions):
        # 与正式serve.py的反变换逐项相同，不启用此前已撤回的v3原型。
        return (processor.normalize(actions, "action") + 1) / 2 * (high - low + 1e-6) + low

    info = json.loads((args.dataset / "meta/info.json").read_text())
    episodes = [
        json.loads(line) for line in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()
    ]
    errors = []
    for episode in episodes:
        index = episode["episode_index"]
        path = args.dataset / info["data_path"].format(
            episode_index=index, episode_chunk=index // info["chunks_size"]
        )
        actions = np.asarray(
            pq.read_table(path, columns=["action"])["action"].to_pylist(), np.float32
        )
        if actions.shape != (episode["length"], 36) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid action labels: {path}")
        errors.append(roundtrip(actions) - actions)
    errors = np.concatenate(errors)
    if len(errors) != info["total_frames"]:
        raise ValueError("Training frame count differs from metadata")
    content = args.predictions.read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    windows = []
    for episode in sorted({row["episode"] for row in rows}):
        selected = sorted(
            (row for row in rows if row["episode"] == episode), key=lambda row: row["frame"]
        )
        if any(b["frame"] - a["frame"] != 30 for a, b in zip(selected, selected[1:])):
            raise ValueError("Recorded windows must be consecutive 30-step chunks")
        expert = np.asarray([row["expert_actions"] for row in selected], np.float32)
        transformed = roundtrip(expert)
        item = dict(
            episode=episode,
            arm_label_roundtrip_rmse_rad=float(
                np.sqrt(np.mean((transformed[:, :, 14:28] - expert[:, :, 14:28]) ** 2))
            ),
            arm_label_roundtrip_max_abs_rad=float(
                np.abs(transformed[:, :, 14:28] - expert[:, :, 14:28]).max()
            ),
            original_boundary_l2_rad=np.linalg.norm(
                expert[1:, 0, 14:28] - expert[:-1, -1, 14:28], axis=-1
            ).tolist(),
            roundtrip_boundary_l2_rad=np.linalg.norm(
                transformed[1:, 0, 14:28] - transformed[:-1, -1, 14:28], axis=-1
            ).tolist(),
            models={},
        )
        for model in ("STATE", "VGQ-AE"):
            actions = np.asarray([row["models"][model]["actions"] for row in selected])
            item["models"][model] = dict(
                arm_rmse_to_raw_expert_rad=float(
                    np.sqrt(np.mean((actions[:, :, 14:28] - expert[:, :, 14:28]) ** 2))
                ),
                arm_rmse_to_actual_training_label_rad=float(
                    np.sqrt(np.mean((actions[:, :, 14:28] - transformed[:, :, 14:28]) ** 2))
                ),
            )
        windows.append(item)
    report = dict(
        normalization=saved["normalization"],
        stats_sha256=hashlib.sha256((args.dataset / "meta/stats.json").read_bytes()).hexdigest(),
        predictions_sha256=hashlib.sha256(content).hexdigest(),
        training_episodes=len(episodes),
        training_frames=len(errors),
        training_roundtrip_rmse_per_dimension=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        training_roundtrip_max_abs_per_dimension=np.abs(errors).max(axis=0).tolist(),
        recorded_windows=windows,
        limitation="Actual v2 label roundtrip only. This is not a learned-model counterfactual or a claim that training normalization has no effect on learning. No v3 migration or production change.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"training_frames": len(errors), "recorded_windows": windows}))


if __name__ == "__main__":
    main()
