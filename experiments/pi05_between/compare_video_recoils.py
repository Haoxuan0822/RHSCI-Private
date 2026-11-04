"""只读已有VGQ录像，按等级比较前121帧的手部二维回撤；不调用模型。"""

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path

import cv2
import numpy as np
from track_historical_video import track_points


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows(xy, centers):
    result = []
    for center in centers:
        incoming = xy[center] - xy[center - 6]
        outgoing = xy[center + 6] - xy[center]
        before, after = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        cosine = float(np.dot(incoming, outgoing) / max(before * after, 1e-12))
        result.append(
            dict(frame=center, before_px=float(before), after_px=float(after), cosine=cosine)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--wrist", type=int, nargs=2, default=[224, 253])
    parser.add_argument("--relative-to", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    paired = (
        {row["identity"]: row for row in json.loads(args.relative_to.read_text())["rows"]}
        if args.relative_to
        else None
    )
    rows, tiles = [], []
    for label, root in (("old", args.old_root), ("new", args.new_root)):
        for level in range(3):
            for repeat in (1, 2):
                run = root / f"dr-level-{level}" / f"repeat-{repeat:02d}"
                config_path = run / "effective_eval_config.json"
                if config_path.exists():
                    config = json.loads(config_path.read_text())
                else:
                    config_path = run / "command.sh"
                    command = shlex.split(config_path.read_text().replace("\\\n", " "))
                    config = {
                        "diagnostic_execute_horizon": (
                            int(command[command.index("--diagnostic-execute-horizon") + 1])
                            if "--diagnostic-execute-horizon" in command
                            else None
                        )
                    }
                if config.get("diagnostic_execute_horizon") not in (None, 30):
                    raise ValueError("Only 30-step execution videos can be compared")
                log_path = run / ("eval.log" if label == "old" else "console.log")
                returned_lengths = set(
                    re.findall(r"Received (\d+) actions from server\.", log_path.read_text())
                )
                if returned_lengths != {"30"}:
                    raise ValueError(f"Expected actual 30-action chunks in {log_path}")
                for video in sorted(run.rglob("head_stereo_left_*.mp4")):
                    if video.stem not in ("head_stereo_left_success", "head_stereo_left_failed"):
                        continue
                    identity = f"{label}-l{level}-r{repeat}-{video.parent.name}"
                    row = dict(
                        identity=identity,
                        package=label,
                        level=level,
                        repeat=repeat,
                        video=str(video),
                        video_sha256=digest(video),
                        config_sha256=digest(config_path),
                        log_sha256=digest(log_path),
                        returned_action_lengths=sorted(returned_lengths),
                        success=video.stem.endswith("_success"),
                    )
                    capture = cv2.VideoCapture(str(video))
                    ok, frame = capture.read()
                    capture.release()
                    try:
                        if not ok:
                            raise ValueError("Cannot decode initial frame")
                        xy, errors = track_points(video, [args.wrist], 121)
                        track = xy[:, 0]
                        row.update(
                            wrist_initial_px=args.wrist,
                            trajectory_wrist_px=track.tolist(),
                            max_forward_backward_error_px=errors.max(0).tolist(),
                            windows={
                                "wrist_raw": dict(
                                    boundary=windows(track, [30, 60, 90]),
                                    control=windows(track, [15, 45, 75]),
                                )
                            },
                        )
                        if paired is not None:
                            other = paired[identity]
                            if "error" in other or other["video_sha256"] != row["video_sha256"]:
                                raise ValueError("Paired track is invalid or names another video")
                            relative = track - np.asarray(other["trajectory_wrist_px"])
                            row["windows"]["wrist_relative"] = dict(
                                boundary=windows(relative, [30, 60, 90]),
                                control=windows(relative, [15, 45, 75]),
                            )
                        cv2.circle(frame, tuple(args.wrist), 3, (0, 0, 255), 1)
                    except ValueError as error:
                        row["error"] = str(error)
                    rows.append(row)
                    if ok:
                        left = args.wrist[0] - 34
                        tile = frame[170:280, left : left + 120].copy()
                        cv2.putText(
                            tile,
                            identity.replace("episode_", "e"),
                            (1, 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.25,
                            (0, 0, 255),
                            1,
                        )
                        tiles.append(tile)
                    print(identity, "rejected" if "error" in row else "tracked", flush=True)
    summary = []
    for label in ("old", "new"):
        for level in range(3):
            selected = [r for r in rows if r["package"] == label and r["level"] == level]
            valid = [r for r in selected if "error" not in r]
            for method in ("wrist_raw", "wrist_relative") if paired else ("wrist_raw",):
                for phase in ("boundary", "control"):
                    samples = [w for r in valid for w in r["windows"][method][phase]]
                    summary.append(
                        dict(
                            package=label,
                            level=level,
                            method=method,
                            phase=phase,
                            videos=len(selected),
                            tracked=len(valid),
                            windows=len(samples),
                            reversal_counts={
                                str(threshold): sum(
                                    w["cosine"] < -0.5
                                    and min(w["before_px"], w["after_px"]) >= threshold
                                    for w in samples
                                )
                                for threshold in (0.5, 1, 2)
                            },
                        )
                    )
    report = dict(
        script_sha256=digest(Path(__file__)),
        tracking_script_sha256=digest(Path(__file__).with_name("track_historical_video.py")),
        relative_source=(
            dict(path=str(args.relative_to), sha256=digest(args.relative_to))
            if args.relative_to
            else None
        ),
        rows=rows,
        summary=summary,
        scope="First121 frames only, selected body-surface pixels and 6-frame motion windows; "
        "not full-episode smoothness, 3D joint recoil, contact-stage alignment, independent "
        "trials or a causal old/new package test. Rejections remain explicit. Raw image "
        "motion includes camera motion. Initial landmarks require montage inspection.",
    )
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if tiles:
        canvas = np.zeros((((len(tiles) + 9) // 10) * 110, 1200, 3), np.uint8)
        for i, tile in enumerate(tiles):
            canvas[(i // 10) * 110 : (i // 10 + 1) * 110, (i % 10) * 120 : (i % 10 + 1) * 120] = (
                tile
            )
        cv2.imwrite(str(args.output_dir / "initial-landmarks.jpg"), canvas)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
