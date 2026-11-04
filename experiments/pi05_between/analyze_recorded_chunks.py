"""量化已有原训练图预测的动作块接续；不启动模型或仿真。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(rows, model):
    """仅关联同轨迹相邻30步窗口，保留专家自然边界作参照。"""
    predictions, experts, states = [], [], []
    for row in rows:
        prediction = np.asarray(row["models"][model]["actions"], dtype=float)
        expert = np.asarray(row["expert_actions"], dtype=float)
        state = np.asarray(row["state"], dtype=float)
        if prediction.shape != (30, 36) or expert.shape != (30, 36) or state.shape != (32,):
            raise ValueError("Expected complete 30x36 actions and 32D state")
        if not all(np.isfinite(value).all() for value in (prediction, expert, state)):
            raise ValueError("Nonfinite recorded actions/state")
        predictions.append(prediction[:, 14:28])
        experts.append(expert[:, 14:28])
        states.append(state[14:28])
    predictions, experts = np.stack(predictions), np.stack(experts)
    inside = np.linalg.norm(np.diff(predictions, axis=1), axis=-1)
    boundaries = []
    for index in range(1, len(rows)):
        if rows[index]["frame"] - rows[index - 1]["frame"] != 30:
            raise ValueError("Need consecutive 30-step recorded windows")
        jump = predictions[index, 0] - predictions[index - 1, -1]
        # 与专家最近6帧运动比较方向；负余弦只描述反向，不单独判定bug。
        direction = experts[index - 1, -1] - experts[index - 1, -7]
        denominator = np.linalg.norm(jump) * np.linalg.norm(direction)
        boundaries.append(
            dict(
                frame=rows[index]["frame"],
                predicted_jump_l2_rad=float(np.linalg.norm(jump)),
                expert_jump_l2_rad=float(
                    np.linalg.norm(experts[index, 0] - experts[index - 1, -1])
                ),
                previous_last_to_current_state_l2_rad=float(
                    np.linalg.norm(predictions[index - 1, -1] - states[index])
                ),
                current_first_to_current_state_l2_rad=float(
                    np.linalg.norm(predictions[index, 0] - states[index])
                ),
                jump_vs_recent_expert_direction_cosine=(
                    float(np.dot(jump, direction) / denominator) if denominator > 1e-12 else None
                ),
            )
        )
    return dict(
        episode=rows[0]["episode"],
        model=model,
        chunks=len(rows),
        arm_expert_rmse_rad=float(np.sqrt(np.mean((predictions - experts) ** 2))),
        inside_step_median_l2_rad=float(np.median(inside)),
        boundary_to_inside_ratio=float(
            np.median([row["predicted_jump_l2_rad"] for row in boundaries])
            / max(float(np.median(inside)), 1e-12)
        ),
        boundaries=boundaries,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.predictions.read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    identities = [(row["episode"], row["frame"]) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("Repeated episode/frame identity")
    results = []
    for episode in sorted({row["episode"] for row in rows}):
        selected = sorted(
            (row for row in rows if row["episode"] == episode), key=lambda row: row["frame"]
        )
        if len(selected) < 2:
            raise ValueError("Need at least two chunks per episode")
        for model in ("STATE", "VGQ-AE"):
            results.append(analyze(selected, model))
    report = dict(
        source=str(args.predictions),
        source_sha256=hashlib.sha256(content).hexdigest(),
        results=results,
        limitation="Recorded expert RGB inputs, not closed-loop rollouts. Native random samples; STATE/VGQ-AE are different trained weights, not a controlled method ablation. Negative direction cosine alone does not prove a bug.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "boundaries"} for row in results]))


if __name__ == "__main__":
    main()
