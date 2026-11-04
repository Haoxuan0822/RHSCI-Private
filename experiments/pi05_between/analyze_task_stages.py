"""只读完整控制轨迹，量化目标抬升、接近篮子和落地的几何事件。"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def first_sustained(steps, condition, duration=5):
    """返回连续满足阈值的首步；阈值事件不替代抓取接触或官方成功判定。"""
    for index in range(len(condition) - duration + 1):
        if condition[index : index + duration].all():
            return int(steps[index])
    return None


def analyze(directory):
    trace = directory / "control_trace.jsonl"
    content = trace.read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    rows = [row for row in rows if row["kind"] == "env_step" and row["phase"] == "policy"]
    steps = np.asarray([row["step"] for row in rows])
    if not len(rows) or not np.array_equal(np.diff(steps), np.ones(len(rows) - 1)):
        raise ValueError(f"Missing or discontinuous policy steps: {trace}")
    target = np.asarray([row["objects"]["target"][:3] for row in rows])
    basket = np.asarray([row["objects"]["container"][:3] for row in rows])
    base = np.asarray([row["qpos"][:3] for row in rows])
    if not all(np.isfinite(value).all() for value in (target, basket, base)):
        raise ValueError(f"Nonfinite physical poses: {trace}")
    distance = np.linalg.norm(target[:, :2] - basket[:, :2], axis=1)
    travel = np.linalg.norm(base[:, :2] - base[0, :2], axis=1)
    stats = directory / "stats/eval_stats.txt"
    result = dict(
        directory=str(directory.resolve()),
        trace_sha256=hashlib.sha256(content).hexdigest(),
        official_stats=stats.read_text() if stats.exists() else None,
        policy_steps=len(rows),
        target_initial_xyz_m=target[0].tolist(),
        target_final_xyz_m=target[-1].tolist(),
        target_z_range_m=[float(target[:, 2].min()), float(target[:, 2].max())],
        minimum_target_basket_xy_distance_m=float(distance.min()),
        first_lift_above_initial_008m=first_sustained(steps, target[:, 2] > target[0, 2] + 0.08),
        first_basket_xy_within_010m=first_sustained(steps, distance < 0.1),
        first_target_z_below_010m=first_sustained(steps, target[:, 2] < 0.1),
        first_base_travel_above_030m=first_sustained(steps, travel > 0.3),
    )
    return result, (steps, target[:, 2], distance, travel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = []
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for directory in args.directories:
        result, series = analyze(directory)
        results.append(result)
        for axis, values in zip(axes, series[1:], strict=True):
            axis.plot(series[0], values, label=directory.name)
    for axis, label in zip(
        axes, ["Target origin z (m)", "Target to basket XY (m)", "Base XY travel (m)"], strict=True
    ):
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Policy step (50 Hz simulation)")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"))
    plt.close(figure)
    args.output.write_text(
        json.dumps(
            dict(
                scope="Geometric thresholds sustained for 5 policy steps; no contact/grasp classifier. "
                "Independent native RNG trials, not paired causal estimates or aggregate success rates.",
                trajectories=results,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
