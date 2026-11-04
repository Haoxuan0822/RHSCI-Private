"""比较静态原姿态和仅腰部替换的固定噪声预测，核对重复训练输入对照。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_predictions(path):
    """按图源、裁剪、种子及帧号索引，拒绝重复或不完整输出。"""
    content = path.read_bytes()
    rows = json.loads(content)["chunks"]
    index = {}
    for row in rows:
        key = (row["input_source"], row["center_crop_fraction"], row["seed"], row["step"])
        if key in index:
            raise ValueError(f"Repeated prediction: {path}: {key}")
        index[key] = row
    return content, index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-predictions", "counterfactual-predictions", "recorded-inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline_content, baseline = read_predictions(args.baseline_predictions)
    corrected_content, corrected = read_predictions(args.counterfactual_predictions)
    if baseline.keys() != corrected.keys():
        raise ValueError("The two runs do not have matching prediction identities")
    recorded_content = args.recorded_inputs.read_bytes()
    recorded = {
        row["frame"]: row
        for line in recorded_content.splitlines()
        if (row := json.loads(line))["episode"] == 0
    }
    steps = sorted({key[3] for key in baseline})
    if steps != [0, 30, 60, 90]:
        raise ValueError("Expected four aligned expert-replay frames")
    expert = np.asarray([recorded[step]["expert_actions"] for step in steps])[:, :, 14:28]
    identities = sorted({key[:3] for key in baseline}, key=str)
    results, differences, controls = [], [], []
    for source, crop, seed in identities:
        arrays = []
        for name, index in (("replay", baseline), ("expert_waist", corrected)):
            rows = [index[source, crop, seed, step] for step in steps]
            actions = np.asarray([row["arm_actions"] for row in rows])
            if actions.shape != (4, 30, 14) or not np.isfinite(actions).all():
                raise ValueError("Expected complete finite four-frame arm predictions")
            arrays.append(actions)
            jumps = np.linalg.norm(actions[1:, 0] - actions[:-1, -1], axis=-1)
            inside = np.linalg.norm(np.diff(actions, axis=1), axis=-1)
            results.append(
                dict(
                    mode=name,
                    source=source,
                    crop_fraction=crop,
                    seed=seed,
                    boundary_to_inside_ratio=float(
                        np.median(jumps) / max(np.median(inside), 1e-12)
                    ),
                    arm_expert_rmse_rad=float(np.sqrt(np.mean((actions - expert) ** 2))),
                    frame90_first_left_shoulder_yaw_rad=float(actions[-1, 0, 2]),
                    boundary_start_error_l2_rad=np.linalg.norm(
                        actions[1:, 0] - expert[1:, 0], axis=-1
                    ).tolist(),
                    boundary_end_error_l2_rad=np.linalg.norm(
                        actions[:-1, -1] - expert[:-1, -1], axis=-1
                    ).tolist(),
                )
            )
        difference = arrays[1] - arrays[0]
        differences.append(
            dict(
                source=source,
                crop_fraction=crop,
                seed=seed,
                arm_rmse_change_rad=float(np.sqrt(np.mean(difference**2))),
                max_abs_change_rad=float(np.abs(difference).max()),
                per_frame_first_action_change_l2_rad=np.linalg.norm(
                    difference[:, 0], axis=-1
                ).tolist(),
            )
        )
        if source == "training_recorded":
            controls.append(
                dict(
                    crop_fraction=crop,
                    seed=seed,
                    actions_exact=bool(np.array_equal(*arrays)),
                    processed_images_exact=all(
                        baseline[source, crop, seed, step]["processed_image_sha256"]
                        == corrected[source, crop, seed, step]["processed_image_sha256"]
                        for step in steps
                    ),
                )
            )
    report = dict(
        baseline_sha256=hashlib.sha256(baseline_content).hexdigest(),
        counterfactual_sha256=hashlib.sha256(corrected_content).hexdigest(),
        recorded_inputs_sha256=hashlib.sha256(recorded_content).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        steps=steps,
        results=results,
        differences=differences,
        repeated_training_controls=controls,
        limitation="Four static snapshots, not a dynamics or success-rate test. Replay floating base retained; correcting waist joints does not recover the complete training camera trajectory. Ratios alone do not measure task success.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
