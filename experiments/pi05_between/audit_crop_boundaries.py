"""只读已有预测，逐边界核查裁剪效果，防止三边界中位数掩盖反例。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--recorded-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.predictions.read_bytes()
    recorded_content = args.recorded_inputs.read_bytes()
    index = {}
    for row in json.loads(content)["chunks"]:
        key = (row["input_source"], row["seed"], row["center_crop_fraction"], row["step"])
        if key in index:
            raise ValueError(f"Repeated prediction: {key}")
        index[key] = row
    recorded = {
        row["frame"]: row
        for line in recorded_content.splitlines()
        if (row := json.loads(line))["episode"] == 0
    }
    steps = sorted({key[-1] for key in index})
    if steps != [0, 30, 60, 90]:
        raise ValueError("Expected four aligned 30-step windows")
    expert = np.asarray([recorded[step]["expert_actions"] for step in steps])[:, :, 14:28]
    if expert.shape != (4, 30, 14) or not np.isfinite(expert).all():
        raise ValueError("Incomplete expert reference")
    expert_jump = expert[1:, 0] - expert[:-1, -1]
    results = []
    identities = sorted({key[:2] for key in index})
    crops = sorted({key[2] for key in index if key[2] is not None})
    for source, seed in identities:
        cases = {}
        for crop in [None, *crops]:
            actions = np.asarray([index[source, seed, crop, step]["arm_actions"] for step in steps])
            if actions.shape != expert.shape or not np.isfinite(actions).all():
                raise ValueError(f"Incomplete finite predictions: {source}, {seed}, {crop}")
            jump = actions[1:, 0] - actions[:-1, -1]
            prior_progress = actions[:-1, -1] - actions[:-1, 0]
            start_error = actions[1:, 0] - expert[1:, 0]
            end_error = actions[:-1, -1] - expert[:-1, -1]
            if not np.allclose(jump - expert_jump, start_error - end_error, atol=1e-12):
                raise ValueError("Boundary error decomposition did not match")
            norms = np.linalg.norm(jump, axis=-1)
            denominator = norms * np.linalg.norm(prior_progress, axis=-1)
            cases[crop] = dict(
                crop_fraction=crop,
                inside_median_l2_rad=float(
                    np.median(np.linalg.norm(np.diff(actions, axis=1), axis=-1))
                ),
                boundary_l2_rad=norms.tolist(),
                boundary_excess_error_l2_rad=np.linalg.norm(jump - expert_jump, axis=-1).tolist(),
                start_error_l2_rad=np.linalg.norm(start_error, axis=-1).tolist(),
                end_error_l2_rad=np.linalg.norm(end_error, axis=-1).tolist(),
                boundary_vs_prior_progress_cosine=[
                    float(product / divisor) if divisor > 0 else None
                    for product, divisor in zip(np.sum(jump * prior_progress, axis=-1), denominator)
                ],
                expert_rmse_rad=float(np.sqrt(np.mean((actions - expert) ** 2))),
            )
        baseline = np.asarray(cases[None]["boundary_l2_rad"])
        comparisons = []
        for crop in crops:
            delta = np.asarray(cases[crop]["boundary_l2_rad"]) - baseline
            comparisons.append(dict(crop_fraction=crop, boundary_l2_change_rad=delta.tolist()))
        results.append(
            dict(source=source, seed=seed, cases=list(cases.values()), comparisons=comparisons)
        )
    report = dict(
        predictions_sha256=digest(content),
        recorded_inputs_sha256=digest(recorded_content),
        script_sha256=digest(Path(__file__).read_bytes()),
        prediction_count=len(index),
        boundary_frames=steps[1:],
        expert_boundary_l2_rad=np.linalg.norm(expert_jump, axis=-1).tolist(),
        results=results,
        limitation="Existing episode-0 snapshots and finite seeds only; no new model calls, "
        "physics, smoothing, configuration change or success-rate claim. Negative cosine "
        "means reversal relative to the preceding predicted chunk, not ground-truth motion.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
