"""核对公开原始轨迹到实际训练轨迹的身份、命名关节映射和视频帧。"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_videos(left, right):
    captures = [cv2.VideoCapture(str(path)) for path in (left, right)]
    count, max_difference, absolute_sum, pixels = 0, 0, 0, 0
    try:
        if not all(cap.isOpened() for cap in captures):
            raise ValueError("Cannot open both source and training videos")
        while True:
            frames = [cap.read() for cap in captures]
            if not any(ok for ok, _ in frames):
                break
            if not all(ok for ok, _ in frames):
                raise ValueError("Video frame counts differ")
            delta = np.abs(frames[0][1].astype(np.int16) - frames[1][1].astype(np.int16))
            max_difference = max(max_difference, int(delta.max()))
            absolute_sum += int(delta.sum())
            pixels += delta.size
            count += 1
    finally:
        for cap in captures:
            cap.release()
    return dict(
        frames=count,
        max_pixel_difference=max_difference,
        mean_absolute_difference=absolute_sum / pixels if pixels else None,
        source_sha256=digest(left),
        training_sha256=digest(right),
    )


def compare_motion(paths):
    """跨渲染器只比较运动时序；固定重叠窗口和留出帧，避免按lag改变样本数。"""
    reports = [json.loads(path.read_text()) for path in paths]
    xy = [np.asarray(value["trajectory_tip_wrist_reference_px"]) for value in reports]
    velocity = [np.diff((v[:, :2] - v[:, 2:3]).reshape(len(v), -1), axis=0) for v in xy]
    if len(velocity[0]) != len(velocity[1]) or len(velocity[0]) < 120:
        raise ValueError("Motion comparison needs matching videos with at least 121 frames")
    indices = np.arange(30, len(velocity[0]) - 30)
    train_mask = (indices // 5) % 2 == 0
    scores = []
    for lag in range(-30, 31):
        x = np.column_stack([velocity[0][indices], np.ones(len(indices))])
        y = velocity[1][indices + lag]
        fit = np.linalg.lstsq(x[train_mask], y[train_mask], rcond=None)[0]
        error = y[~train_mask] - x[~train_mask] @ fit
        scores.append(dict(lag_frames=lag, heldout_mse_px2=float(np.mean(error**2))))
    horizontal = [(value[:, :2] - value[:, 2:3])[:, :, 0] for value in xy]
    position_scores = []
    for lag in range(-30, 31):
        x = np.column_stack([horizontal[0][indices], np.ones(len(indices))])
        y = horizontal[1][indices + lag]
        fit = np.linalg.lstsq(x[train_mask], y[train_mask], rcond=None)[0]
        error = y[~train_mask] - x[~train_mask] @ fit
        position_scores.append(dict(lag_frames=lag, heldout_mse_px2=float(np.mean(error**2))))
    return dict(
        inputs=[dict(path=str(path), sha256=digest(path)) for path in paths],
        max_forward_backward_error_px=[r["max_forward_backward_error_px"] for r in reports],
        fixed_reference_frames=indices.tolist(),
        ranking=sorted(scores, key=lambda item: item["heldout_mse_px2"]),
        horizontal_position_ranking=sorted(
            position_scores, key=lambda item: item["heldout_mse_px2"]
        ),
        limitation="Selected 2D landmarks and first 121 frames only; a temporal consistency "
        "check across renderers, not proof of absolute image-state synchronization. "
        "Velocity and position fits may disagree; do not select only a favorable metric.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-reports", type=Path, nargs=2)
    args = parser.parse_args()
    original = jsonl(args.raw / "episodes.jsonl")
    processed = jsonl(args.dataset / "meta/episodes.jsonl")
    matches = []
    for episode in processed:
        config = json.loads(episode["environment_config"])
        candidates = []
        for source in original:
            source_config = json.loads(source["environment_config"])
            if (
                source["length"] == episode["length"]
                and source_config["dr_state_dict"] == config["dr_state_dict"]
                and source_config["layout"] == config["layout"]
            ):
                candidates.append(source["episode_index"])
        matches.append(dict(training=episode["episode_index"], source_candidates=candidates))
    selected = next(item for item in matches if item["training"] == args.episode)
    if len(selected["source_candidates"]) != 1:
        raise ValueError("Selected episode needs a unique original identity")
    source_id = selected["source_candidates"][0]
    raw_info = json.loads((args.raw / "info.json").read_text())
    info = json.loads((args.dataset / "meta/info.json").read_text())
    raw_path = args.raw / f"episode_{source_id:06d}.parquet"
    data_path = args.dataset / info["data_path"].format(
        episode_index=args.episode, episode_chunk=args.episode // info.get("chunks_size", 1000)
    )
    source = pq.read_table(raw_path).to_pydict()
    train = pq.read_table(data_path, columns=["states", "action", "timestamp"]).to_pydict()
    names = raw_info["features"]["observation.state"]["names"]
    action_names = raw_info["features"]["action"]["names"]
    order = (
        names[29:32]
        + names[34:36]
        + names[32:34]
        + names[36:43]
        + names[15:29]
        + [names[13], names[14], names[12]]
    )
    height = np.asarray(source["teleop.base_height_command"])
    state = np.column_stack(
        [
            np.asarray(source["observation.state"])[:, [names.index(name) for name in order]],
            np.r_[0.74, height[:-1]],
        ]
    )
    action = np.column_stack(
        [
            np.asarray(source["action"])[:, [action_names.index(name) for name in order]],
            height,
            np.asarray(source["teleop.navigate_command"]),
        ]
    )
    video_path = args.dataset / info["video_path"].format(
        episode_index=args.episode,
        episode_chunk=args.episode // info.get("chunks_size", 1000),
        video_key="observation.images.egocentric",
    )
    report = dict(
        raw=str(args.raw),
        dataset=str(args.dataset),
        episode_identity_matches=matches,
        selected=selected,
        source_parquet_sha256=digest(raw_path),
        training_parquet_sha256=digest(data_path),
        state_max_abs_error=float(np.max(np.abs(state - np.asarray(train["states"])))),
        action_max_abs_error=float(np.max(np.abs(action - np.asarray(train["action"])))),
        timestamp_max_abs_error=float(
            np.max(np.abs(np.asarray(source["timestamp"]) - np.asarray(train["timestamp"])))
        ),
        video=compare_videos(args.raw / f"episode_{source_id:06d}.mp4", video_path),
        limitations=[
            "Only the selected episode has source parquet/video content verified.",
            "Metadata matching alone does not verify the other episodes' content.",
            "Source img_state_delta is a hard-coded zero in current official recorder; "
            "it cannot prove physical image-state synchronization.",
            "Agreement with source does not establish source recordings are correctly synchronized.",
        ],
    )
    if args.motion_reports:
        report["motion_alignment"] = compare_motion(args.motion_reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "episode_identity_matches"}
        )
    )


if __name__ == "__main__":
    main()
