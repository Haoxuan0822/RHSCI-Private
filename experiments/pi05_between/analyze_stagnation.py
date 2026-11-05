"""对齐物理单位预测、环境入口下发和双臂/腰部真值；导出停滞窗口。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

JOINT_NAMES = [
    f"{side}_{name}"
    for side in ("left", "right")
    for name in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
] + ["waist_roll", "waist_pitch", "waist_yaw"]

GOAL_INDICES = [*range(3, 10), *range(17, 24), 1, 2, 0]
WBC_INDICES = [*range(15, 22), *range(29, 36), 13, 14, 12]
BODY_INDICES = [*range(15, 29), 13, 14, 12]
QPOS_INDICES = [*range(22, 29), *range(36, 43), 20, 21, 19]


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value) ** 2)))


def motion(value):
    return dict(
        step_rms_rad=rms(np.diff(value, axis=0)),
        path_l2_rad=float(np.linalg.norm(np.diff(value, axis=0), axis=1).sum()),
        net_l2_rad=float(np.linalg.norm(value[-1] - value[0])),
        joint_range_median_rad=float(np.median(np.ptp(value, axis=0))),
        joint_range_max_rad=float(np.max(np.ptp(value, axis=0))),
        net_rms_rad=rms(value[-1] - value[0]),
    )


def analyze(trace, start, stop):
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    policy = [row for row in rows if row["phase"] == "policy"]
    kinds = {
        kind: {row["step"]: row for row in policy if row["kind"] == kind}
        for kind in ("env_step", "wbc_goal", "wbc_action", "model_chunk")
    }
    raw = {}
    complete_actions = {}
    for step, row in kinds["model_chunk"].items():
        for offset, action in enumerate(row["actions"]):
            if step + offset in raw:
                raise ValueError("Overlapping model chunks")
            raw[step + offset] = np.array(action[14:31])
            complete_actions[step + offset] = np.array(action)
    steps = sorted(
        set(raw)
        & kinds["env_step"].keys()
        & kinds["wbc_goal"].keys()
        & kinds["wbc_action"].keys()
        & set(range(start, stop))
    )
    if steps != list(range(start, stop)):
        raise ValueError(f"Incomplete window: {len(steps)}/{stop - start}")
    predicted = np.array([raw[s] for s in steps])
    goals = np.array(
        [kinds["wbc_goal"][s]["goal"]["target_upper_body_pose"] for s in steps]
    )
    wbc = np.array([kinds["wbc_action"][s]["q"] for s in steps])[:, WBC_INDICES]
    dispatched = np.array(
        [kinds["env_step"][s]["dispatched"]["target_q"] for s in steps]
    )[:, BODY_INDICES]
    before = np.array([kinds["env_step"][s]["qpos_before"] for s in steps])[
        :, QPOS_INDICES
    ]
    actual = np.array([kinds["env_step"][s]["qpos"] for s in steps])[:, QPOS_INDICES]
    observed = np.array([kinds["env_step"][s]["observed_joints"] for s in steps])[
        :, BODY_INDICES
    ]
    base = np.array([kinds["env_step"][s]["qpos"][:3] for s in steps])
    navigation = np.array([complete_actions[s][32:36] for s in steps])
    chunks = [row for s, row in kinds["model_chunk"].items() if start <= s < stop]
    reversals = []
    chunk_starts = sorted(kinds["model_chunk"])
    for previous, current in zip(chunk_starts, chunk_starts[1:]):
        if start <= current < stop:
            progress = raw[current - 1] - raw[previous]
            jump = raw[current] - raw[current - 1]
            scale = np.linalg.norm(progress) * np.linalg.norm(jump)
            reversals.append(
                dict(
                    step=current,
                    previous_progress_l2_rad=float(np.linalg.norm(progress)),
                    boundary_jump_l2_rad=float(np.linalg.norm(jump)),
                    cosine=float(np.dot(progress, jump) / scale)
                    if scale > 1e-12
                    else None,
                )
            )
    summary = dict(
        trace=str(trace),
        start_step=start,
        stop_step_exclusive=stop,
        simulation_window_seconds=[start / 50, stop / 50],
        model_requests_in_window=len(chunks),
        analyzed_joints=JOINT_NAMES,
        base_position_range_m=np.ptp(base, axis=0).tolist(),
        navigation_min=navigation.min(axis=0).tolist(),
        navigation_max=navigation.max(axis=0).tolist(),
        chunk_boundaries=reversals,
        model_inference_seconds_median=float(
            np.median([r["duration"] for r in chunks])
        ),
        model_to_goal_max_abs_rad=float(
            np.abs(predicted - goals[:, GOAL_INDICES]).max()
        ),
        wbc_to_dispatched_max_abs_rad=float(np.abs(wbc - dispatched).max()),
        isaac_vs_mujoco_max_abs_rad=float(np.abs(observed - actual).max()),
        predicted_vs_current_rms_rad=rms(predicted - before),
        dispatched_vs_current_rms_rad=rms(dispatched - before),
        prediction_vs_dispatch_rms_rad=rms(predicted - dispatched),
        previous_prediction_vs_dispatch_rms_rad=rms(predicted[:-1] - dispatched[1:]),
        motion={
            name: motion(value)
            for name, value in [
                ("predicted", predicted),
                ("dispatched", dispatched),
                ("actual", actual),
            ]
        },
        arm_motion={
            name: motion(value[:, :14])
            for name, value in [
                ("predicted", predicted),
                ("dispatched", dispatched),
                ("actual", actual),
            ]
        },
        arm_predicted_vs_current_rms_rad=rms(predicted[:, :14] - before[:, :14]),
        arm_prediction_vs_dispatch_rms_rad=rms(predicted[:, :14] - dispatched[:, :14]),
        joints=[
            dict(
                name=name,
                prediction_range_rad=float(np.ptp(predicted[:, i])),
                dispatched_range_rad=float(np.ptp(dispatched[:, i])),
                actual_range_rad=float(np.ptp(actual[:, i])),
                predicted_vs_current_rms_rad=rms(predicted[:, i] - before[:, i]),
                dispatched_vs_current_rms_rad=rms(dispatched[:, i] - before[:, i]),
            )
            for i, name in enumerate(JOINT_NAMES)
        ],
    )
    output = trace.parent / f"window-{start}-{stop}"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "joints.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "sim_time_s",
                "joint",
                "model_target_rad",
                "dispatched_rad",
                "actual_before_rad",
                "actual_after_rad",
            ]
        )
        for k, step in enumerate(steps):
            for j, name in enumerate(JOINT_NAMES):
                writer.writerow(
                    [
                        step,
                        step / 50,
                        name,
                        predicted[k, j],
                        dispatched[k, j],
                        before[k, j],
                        actual[k, j],
                    ]
                )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(9, 2, figsize=(15, 19), sharex=True)
    for j, name in enumerate(JOINT_NAMES):
        ax = axes[j % 7, j // 7] if j < 14 else axes[7 + (j - 14) // 2, (j - 14) % 2]
        for label, values in [
            ("Model physical target", predicted),
            ("Dispatched target", dispatched),
            ("Measured joint", before),
        ]:
            ax.plot(np.array(steps) / 50, values[:, j], label=label, linewidth=1)
        ax.set_title(name)
        ax.set_ylabel("rad")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("Simulation seconds")
    axes[-1, 1].set_xlabel("Simulation seconds")
    axes[-1, 1].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "joint-comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--start", type=int, default=600)
    parser.add_argument("--stop", type=int, default=900)
    args = parser.parse_args()
    analyze(args.trace, args.start, args.stop)
