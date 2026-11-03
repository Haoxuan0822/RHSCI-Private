"""Compare native model chunks, WBC targets and MuJoCo motion from a diagnostic trace."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(path):
    """按策略步关联各层真实记录；返回数值证据，不将截短诊断计入成功率。"""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    policy = [row for row in rows if row["phase"] == "policy"]
    chunks = [row for row in policy if row["kind"] == "model_chunk"]
    env = {row["step"]: row for row in policy if row["kind"] == "env_step"}
    goals = {row["step"]: row for row in policy if row["kind"] == "wbc_goal"}
    wbc = {row["step"]: row for row in policy if row["kind"] == "wbc_action"}
    steps = sorted(env.keys() & goals.keys() & wbc.keys())
    if len(steps) < 31 or any(right != left + 1 for left, right in zip(steps, steps[1:])):
        raise ValueError("Need at least 31 consecutive completed policy steps")
    # 已核对官方机器人模型：upper body 是 waist3 / left arm7 / left hand7 /
    # right arm7 / right hand7；MuJoCo qpos 按树遍历，手臂和手指交错排列，
    # 不能把 motor 的 body29 顺序套用到 qpos。双臂分别为 22:29 / 36:43。
    upper_arm_indices = [*range(3, 10), *range(17, 24)]
    wbc_arm_indices = [*range(15, 22), *range(29, 36)]
    target = np.array([goals[step]["goal"]["target_upper_body_pose"] for step in steps])
    output = np.array([wbc[step]["q"] for step in steps])
    actual = np.array([env[step]["qpos"][22:29] + env[step]["qpos"][36:43] for step in steps])
    target_delta = np.linalg.norm(np.diff(target[:, upper_arm_indices], axis=0), axis=1)
    output_delta = np.linalg.norm(np.diff(output[:, wbc_arm_indices], axis=0), axis=1)
    chunk_starts = {row["step"] for row in chunks[1:]}
    boundary = np.array([step in chunk_starts for step in steps[1:]])
    inside_median = float(np.median(target_delta[~boundary]))
    # 在控制器之前核对每一步，区分模型原始回跳和动作转换/队列错位。
    mapping_errors = []
    for chunk in chunks:
        for offset, action in enumerate(chunk["actions"]):
            step = chunk["step"] + offset
            if step in goals:
                mapped = np.array(goals[step]["goal"]["target_upper_body_pose"])
                mapping_errors.append(np.abs(mapped[upper_arm_indices] - action[14:28]).max())
    info = {
        "trace": str(path),
        "policy_steps": len(steps),
        "chunks": len(chunks),
        "model_seconds": [row["duration"] for row in chunks],
        "sim_step_seconds": sorted(
            {round(env[step]["sim_after"] - env[step]["sim_before"], 7) for step in steps}
        ),
        "wall_step_seconds_min_median_max": np.percentile(
            [env[step]["duration"] for step in steps], [0, 50, 100]
        ).tolist(),
        "boundary_steps": sorted(chunk_starts),
        "target_arm_boundary_l2_rad": target_delta[boundary].tolist(),
        "target_arm_inside_median_l2_rad": inside_median,
        "target_boundary_to_inside_ratio": float(np.median(target_delta[boundary]))
        / max(inside_median, 1e-12),
        "wbc_arm_boundary_l2_rad": output_delta[boundary].tolist(),
        "raw_action_to_wbc_arm_mapping_max_abs_rad": float(max(mapping_errors)),
    }
    if all("observed_state" in row for row in chunks):
        states = np.array([row["observed_state"]["states"][0] for row in chunks])
        info["chunk_first_action_vs_observed_arms_l2_rad"] = [
            float(np.linalg.norm(np.array(row["actions"])[0, 14:28] - state[14:28]))
            for row, state in zip(chunks, states)
        ]
    if all("observed_joints" in env[step] for step in steps):
        observed = np.array([env[step]["observed_joints"][15:29] for step in steps])
        info["isaac_vs_mujoco_arm_max_abs_rad"] = float(np.abs(observed - actual).max())
    return info, (steps, target, output, actual, target_delta, output_delta)


def plot(path, info, arrays):
    """保存标准科学图，显示块边界前后各控制层的变化。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, target, output, actual, target_delta, output_delta = arrays
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(steps, target[:, 5], label="Model target")
    axes[0].plot(steps, output[:, 17], label="WBC target")
    axes[0].plot(steps, actual[:, 2], label="MuJoCo joint")
    axes[0].set_ylabel("Left shoulder yaw (rad)")
    axes[1].plot(steps[1:], target_delta, label="Model arm target delta")
    axes[1].plot(steps[1:], output_delta, label="WBC arm target delta")
    axes[1].set_ylabel("14-joint L2 delta (rad)")
    axes[1].set_xlabel("Control step (50 Hz simulation)")
    for axis in axes:
        for boundary in info["boundary_steps"]:
            axis.axvline(boundary, color="gray", linestyle="--", alpha=0.5)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Diagnostic only: chunk boundaries and actual robot motion")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    info, arrays = analyze(args.trace)
    output = args.trace.with_name("control_analysis.json")
    output.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    if args.plot:
        plot(args.trace.with_name("control_analysis.png"), info, arrays)
