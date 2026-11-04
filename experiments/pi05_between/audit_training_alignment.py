"""Audit recorded RGB timing and quantify phase errors in paired pi05 predictions."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def audit_episode(root, info, episode):
    index = int(episode["episode_index"])
    fields = dict(episode_index=index, episode_chunk=index // info["chunks_size"])
    table = pq.read_table(
        root / info["data_path"].format(**fields),
        columns=["frame_index", "episode_index", "timestamp", "action", "states"],
    ).to_pydict()
    length = int(episode["length"])
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,r_frame_rate,avg_frame_rate,start_time,duration",
            "-of",
            "json",
            str(root / info["video_path"].format(**fields)),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    video = json.loads(probe.stdout)["streams"][0]
    actions, states = np.asarray(table["action"]), np.asarray(table["states"])
    timestamps = np.asarray(table["timestamp"])
    checks = dict(
        frame_indices_exact=np.array_equal(table["frame_index"], np.arange(length)),
        episode_indices_exact=set(table["episode_index"]) == {index},
        frame_count_exact=int(video["nb_frames"]) == length,
        actions_valid=actions.shape == (length, 36) and bool(np.isfinite(actions).all()),
        states_valid=states.shape == (length, 32) and bool(np.isfinite(states).all()),
    )
    timing_error = float(np.max(np.abs(timestamps - np.arange(length) / info["fps"])))
    checks["timestamps_match_50hz"] = timing_error < 1e-4
    for name in ("r_frame_rate", "avg_frame_rate"):
        numerator, denominator = map(int, video[name].split("/"))
        checks[name + "_matches"] = numerator / denominator == info["fps"]
    return dict(
        episode=index,
        length=length,
        checks=checks,
        video=video,
        timestamp_max_abs_error_s=timing_error,
        # 观察腰部是否是上一条指令，不能仅凭字段名认定为实际关节。
        previous_command_waist_max_abs_error_rad=float(
            np.max(np.abs(states[1:, 28:31] - actions[:-1, 28:31]))
        ),
    )


def phase_rows(root, info, predictions):
    table = pq.read_table(
        root / info["data_path"].format(episode_index=0, episode_chunk=0), columns=["action"]
    ).to_pydict()
    # 首段只比较前240帧，避免后面再次出现相似姿态造成任意远的相位匹配。
    expert = np.asarray(table["action"])[:240, 14:28]
    results = []
    for row in json.loads(predictions.read_text())["chunks"]:
        arms = np.asarray(row["arm_actions"])
        error = np.sqrt(np.mean((arms[:, None] - expert[None]) ** 2, axis=-1))
        nearest = error.argmin(-1)
        results.append(
            dict(
                step=row["step"],
                seed=row["seed"],
                input_source=row["input_source"],
                nearest_expert_action_frames=nearest.tolist(),
                nearest_arm_rmse_rad=error[np.arange(len(arms)), nearest].tolist(),
                limitation="Nearest arm pose within first 240 expert frames; not a decoded model state or unique temporal alignment.",
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    info = json.loads((args.dataset / "meta/info.json").read_text())
    if info["fps"] != 50:
        raise ValueError("Expected the recorded 50 Hz pi05 training dataset")
    episodes = [
        json.loads(line) for line in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda episode: audit_episode(args.dataset, info, episode), episodes))
    report = dict(
        dataset=str(args.dataset),
        episodes=rows,
        all_structural_timing_checks_pass=all(all(row["checks"].values()) for row in rows),
        total_frames=sum(row["length"] for row in rows),
        prediction_phase=phase_rows(args.dataset, info, args.predictions),
        limitation="Container timing and parquet indices do not independently prove physical synchronization of recorded RGB and proprioception. No model, data, or running service changed.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("episodes", "prediction_phase")
            }
        )
    )


if __name__ == "__main__":
    main()
