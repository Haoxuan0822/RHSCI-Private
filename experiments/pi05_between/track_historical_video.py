"""核对历史录像中的二维手部回撤；不从像素运动推导三维关节或任务成功。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def track_points(video, points, frames):
    """跟踪调用方指定的像素点；任一点不可靠时拒绝整条短轨迹。"""
    points = np.asarray(points, np.float32)[:, None]
    capture = cv2.VideoCapture(str(video))
    trajectory = [points[:, 0].copy()]
    errors = []
    before = None
    try:
        for index in range(frames):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}")
            after = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if before is not None:
                options = dict(winSize=(21, 21), maxLevel=3)
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    before, after, points, None, **options
                )
                returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    after, before, next_points, None, **options
                )
                error = np.linalg.norm(returned - points, axis=2)[:, 0]
                if not status.all() or not backward_status.all() or error.max() > 0.5:
                    raise ValueError(f"Unreliable tracking at frame {index}: {error}")
                errors.append(error)
                trajectory.append(next_points[:, 0].copy())
                points = next_points
            before = after
    finally:
        capture.release()
    return np.stack(trajectory), np.stack(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("tip", "wrist", "reference"):
        parser.add_argument(f"--{name}", type=float, nargs=2, required=True)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--boundaries", type=int, nargs="+", default=[24, 48, 72, 96])
    parser.add_argument("--window", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.window < 1 or any(
        b - args.window < 0 or b + args.window >= args.frames for b in args.boundaries
    ):
        parser.error("Every boundary needs a complete before/after window")
    xy, errors = track_points(args.video, [args.tip, args.wrist, args.reference], args.frames)
    # 扣除同一录像中静态背景点的运动，避免把整体相机平移当手部回撤。
    relative = xy[:, :2] - xy[:, 2:3]
    windows = []
    for boundary in args.boundaries:
        incoming = relative[boundary] - relative[boundary - args.window]
        outgoing = relative[boundary + args.window] - relative[boundary]
        windows.append(
            dict(
                boundary_frame=boundary,
                incoming_tip_wrist_px=incoming.tolist(),
                outgoing_tip_wrist_px=outgoing.tolist(),
                direction_cosine=(
                    (incoming * outgoing).sum(-1)
                    / np.maximum(
                        np.linalg.norm(incoming, axis=-1) * np.linalg.norm(outgoing, axis=-1),
                        1e-12,
                    )
                ).tolist(),
            )
        )
    report = dict(
        video=str(args.video),
        points=dict(tip=args.tip, wrist=args.wrist, reference=args.reference),
        frames=args.frames,
        window=args.window,
        max_forward_backward_error_px=errors.max(0).tolist(),
        trajectory_tip_wrist_reference_px=xy.tolist(),
        windows=windows,
        limitation="2D tracking of selected landmarks; not 3D joints, all episodes, or a success-rate estimate. A negative cosine alone does not prove a bug.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if not key.startswith("trajectory")}
        )
    )


if __name__ == "__main__":
    main()
