"""只读同一初态短诊断，区分动作未生成、控制未执行和分块往复。"""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_control_trace import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    rows = [row for row in rows if row["phase"] == "policy"]
    chunks = [row for row in rows if row["kind"] == "model_chunk"]
    env = [row for row in rows if row["kind"] == "env_step"]
    info, arrays = analyze(args.trace)
    _, _, _, actual, _, _ = arrays
    target = np.concatenate([row["actions"] for row in chunks])[:, 14:28]
    chunk_actions = [np.asarray(row["actions"]) for row in chunks]
    states = [np.asarray(row["observed_state"]["states"])[0] for row in chunks]
    base = np.asarray([row["qpos"][:3] for row in env])
    box = np.asarray([row["objects"]["target"][:3] for row in env])
    progress = [action[-1, 14:28] - action[0, 14:28] for action in chunk_actions]
    jumps = [
        right[0, 14:28] - left[-1, 14:28] for left, right in zip(chunk_actions, chunk_actions[1:])
    ]
    result = dict(
        scope="Short rerun with seed17; not the original recording's RNG or aggregate success.",
        control=info,
        finite_actions=bool(np.isfinite(target).all()),
        target_arm_path_l2_rad=float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum()),
        target_arm_net_l2_rad=float(np.linalg.norm(target[-1] - target[0])),
        actual_arm_path_l2_rad=float(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum()),
        actual_arm_net_l2_rad=float(np.linalg.norm(actual[-1] - actual[0])),
        actual_arm_peak_joint_excursion_rad=float(np.ptp(actual, axis=0).max()),
        base_xy_max_displacement_m=float(np.linalg.norm(base[:, :2] - base[0, :2], axis=1).max()),
        box_max_displacement_m=float(np.linalg.norm(box - box[0], axis=1).max()),
        first_to_last_chunk_start_arm_l2_rad=float(
            np.linalg.norm(chunk_actions[-1][0, 14:28] - chunk_actions[0][0, 14:28])
        ),
        lower_policy_calls={
            mode: sum(row["kind"] == "lower_policy" and row["mode"] == mode for row in rows)
            for mode in ("stand", "walk")
        },
        chunks=[
            dict(
                step=row["step"],
                within_chunk_arm_net_l2_rad=float(np.linalg.norm(delta)),
                first_target_vs_observed_arm_l2_rad=float(
                    np.linalg.norm(action[0, 14:28] - state[14:28])
                ),
                last_target_vs_observed_arm_l2_rad=float(
                    np.linalg.norm(action[-1, 14:28] - state[14:28])
                ),
                navigate_min=action[:, 32:36].min(axis=0).tolist(),
                navigate_max=action[:, 32:36].max(axis=0).tolist(),
            )
            for row, action, state, delta in zip(
                chunks, chunk_actions, states, progress, strict=True
            )
        ],
        boundary_jump_vs_previous_progress_cosine=[
            float(np.dot(jump, delta) / max(np.linalg.norm(jump) * np.linalg.norm(delta), 1e-12))
            for jump, delta in zip(jumps, progress[:-1], strict=True)
        ],
    )
    out = args.trace.with_name("stall_analysis.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
