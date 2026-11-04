"""分解已有固定种子预测的边界误差，区分三样本均值偏差与样本波动。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--recorded-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    content = args.predictions.read_bytes()
    recorded_content = args.recorded_inputs.read_bytes()
    chunks = json.loads(content)["chunks"]
    identities = [(row["input_source"], row["seed"], row["step"]) for row in chunks]
    if len(set(identities)) != len(chunks):
        raise ValueError("Repeated source/seed/step identity")
    seeds = sorted({row["seed"] for row in chunks})
    steps = sorted({row["step"] for row in chunks})
    if len(seeds) < 2 or len(steps) < 2 or any(b - a != 30 for a, b in zip(steps, steps[1:])):
        raise ValueError("Need multiple seeds and consecutive 30-step chunks")
    recorded = {
        row["frame"]: row
        for line in recorded_content.splitlines()
        if (row := json.loads(line))["episode"] == 0
    }
    expert = np.asarray([recorded[step]["expert_actions"] for step in steps])[:, :, 14:28]
    expert_jump = expert[1:, 0] - expert[:-1, -1]
    results = []
    for source in sorted({row["input_source"] for row in chunks}):
        index = {(row["seed"], row["step"]): row for row in chunks if row["input_source"] == source}
        actions = np.asarray(
            [[index[seed, step]["arm_actions"] for step in steps] for seed in seeds]
        )
        if actions.shape != (len(seeds), len(steps), 30, 14) or not np.isfinite(actions).all():
            raise ValueError("Expected complete finite seed/step/30x14 array")
        jump = actions[:, 1:, 0] - actions[:, :-1, -1]
        inside = np.linalg.norm(np.diff(actions, axis=2), axis=-1)
        inside_median = np.median(inside, axis=(1, 2))
        boundary_median = np.median(np.linalg.norm(jump, axis=-1), axis=1)
        error = jump - expert_jump
        mean_error = error.mean(0)
        mean_square = np.mean(np.sum(error**2, axis=-1), axis=0)
        squared_mean = np.sum(mean_error**2, axis=-1)
        variance = np.mean(np.sum((error - mean_error) ** 2, axis=-1), axis=0)
        if not np.allclose(mean_square, squared_mean + variance, rtol=1e-12, atol=1e-14):
            raise ValueError("Finite-sample error decomposition failed")
        results.append(
            dict(
                source=source,
                per_seed=[
                    dict(
                        seed=seed,
                        boundary_median_l2_rad=float(boundary_median[index]),
                        inside_median_l2_rad=float(inside_median[index]),
                        boundary_to_inside_ratio=float(
                            boundary_median[index] / max(inside_median[index], 1e-12)
                        ),
                        arm_expert_rmse_rad=float(np.sqrt(np.mean((actions[index] - expert) ** 2))),
                    )
                    for index, seed in enumerate(seeds)
                ],
                boundary_frames=steps[1:],
                expert_boundary_l2_rad=np.linalg.norm(expert_jump, axis=-1).tolist(),
                mean_prediction_boundary_l2_rad=np.linalg.norm(jump.mean(0), axis=-1).tolist(),
                boundary_error_mean_square_rad2=mean_square.tolist(),
                boundary_error_squared_mean_rad2=squared_mean.tolist(),
                boundary_error_variance_rad2=variance.tolist(),
                pooled_squared_mean_fraction=float(squared_mean.sum() / mean_square.sum()),
                first_left_shoulder_yaw_mean_rad=actions[:, :, 0, 2].mean(0).tolist(),
                first_left_shoulder_yaw_population_std_rad=actions[:, :, 0, 2].std(0).tolist(),
            )
        )
    report = dict(
        predictions_sha256=hashlib.sha256(content).hexdigest(),
        recorded_inputs_sha256=hashlib.sha256(recorded_content).hexdigest(),
        seeds=seeds,
        steps=steps,
        results=results,
        limitation="Exact decomposition of these finite seed samples, not an estimate of the true conditional expectation or a confidence interval. Averaged predictions were never executed; this is not a smoothing fix or a success-rate test.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
