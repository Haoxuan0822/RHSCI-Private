"""只读比较已有接近阶段的换块回撤与物体运动，不把局部顿挫等同任务失败。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_control_trace import analyze


def analyze_approach(path):
    content = path.read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    chunks = [row for row in rows if row["kind"] == "model_chunk"]
    physical = [row for row in rows if row["kind"] == "env_step" and row["phase"] == "policy"]
    if [row["step"] for row in physical] != list(range(len(physical))) or len(physical) < 180:
        raise ValueError("Need at least 180 consecutive completed steps from step0")
    if any(np.asarray(row["actions"]).shape != (30, 36) for row in chunks):
        raise ValueError("Only matched 30/30 traces are supported")
    if [row["step"] for row in chunks] != list(range(0, len(physical), 30)):
        raise ValueError("Model chunks do not cover the physical trace at 30-step cadence")
    control, _ = analyze(path)
    if control["raw_action_to_wbc_arm_mapping_max_abs_rad"] != 0:
        raise ValueError("Model/WBC arm mapping differs")
    box = np.asarray([row["objects"]["target"][:3] for row in physical])
    arms = np.asarray([row["qpos"][22:29] + row["qpos"][36:43] for row in physical])
    if not np.isfinite(box).all() or not np.isfinite(arms).all():
        raise ValueError("Nonfinite physical trajectory")
    above = box[:, 2] > box[0, 2] + 0.08
    first_lift = next((i for i in range(len(above) - 4) if above[i : i + 5].all()), None)
    boundaries = []
    for left, right in zip(chunks, chunks[1:]):
        a, b = (np.asarray(row["actions"])[:, 14:28] for row in (left, right))
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite model targets")
        progress, jump = a[-1] - a[0], b[0] - a[-1]
        norm_square = float(progress @ progress)
        if norm_square < 1e-12:
            raise ValueError("Near-zero progress makes the projected retention ill-defined")
        boundaries.append(
            dict(
                step=right["step"],
                prior_chunk_progress_l2_rad=float(np.linalg.norm(progress)),
                boundary_jump_l2_rad=float(np.linalg.norm(jump)),
                projected_retention=float((b[0] - a[0]) @ progress / norm_square),
                jump_vs_prior_progress_cosine=float(
                    jump @ progress / max(np.linalg.norm(jump) * np.linalg.norm(progress), 1e-12)
                ),
            )
        )
    windows = []
    for steps in (91, 180):
        selected = [row for row in boundaries if row["step"] < steps]
        actual = arms[:steps]
        path_length = float(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum())
        windows.append(
            dict(
                steps=steps,
                all_before_sustained_lift=first_lift is None or first_lift >= steps,
                boundaries=selected,
                retention_median=float(np.median([row["projected_retention"] for row in selected])),
                actual_arm_net_l2_rad=float(np.linalg.norm(actual[-1] - actual[0])),
                actual_arm_path_l2_rad=path_length,
                box_max_rise_m=float(np.max(box[:steps, 2]) - box[0, 2]),
                box_max_displacement_m=float(np.linalg.norm(box[:steps] - box[0], axis=1).max()),
            )
        )
    if not windows[0]["all_before_sustained_lift"]:
        raise ValueError("First91-step window crosses the declared lift stage")
    config_path = path.parent / "effective_eval_config.json"
    config_content = config_path.read_bytes()
    config = json.loads(config_content)
    return dict(
        trace=str(path.resolve()),
        trace_sha256=hashlib.sha256(content).hexdigest(),
        config_sha256=hashlib.sha256(config_content).hexdigest(),
        task=config["env_id"],
        data_dir=config["data_dir"],
        episode=config["episode_start"],
        noise_seed=config.get("diagnostic_noise_seed"),
        noise_stream=config.get("diagnostic_noise_stream"),
        episode_limit=config["max_episode_steps"],
        first_sustained_lift_step=first_lift,
        model_to_wbc_arm_max_abs_rad=control["raw_action_to_wbc_arm_mapping_max_abs_rad"],
        windows=windows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, help="label=path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = {}
    for value in args.trace:
        name, path = value.split("=", 1)
        if not name or name in records:
            raise ValueError("Need distinct nonempty trace labels")
        records[name] = analyze_approach(Path(path))
    if len({(row["task"], row["episode"]) for row in records.values()}) != 1:
        raise ValueError("Different task or initial-state indices")
    report = dict(
        records=records,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="All traces: same task/episode number, verified30/30 and exact arm mapping. "
        "First91 steps precede a sustained8cm lift in every trace; first180 is descriptive "
        "and may span different stages. Projected retention is signed joint-space geometry, "
        "not a probability, physical success measure or causal percentage. Old/new runs "
        "are not RNG paired. The new original/plus006 pair declares the same noise seed "
        "and changes spawn only, but contains a single episode. These are not aggregate "
        "old/new smoothness estimates; a short-run False is not a full-task model failure.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    for name, record in records.items():
        print(
            json.dumps(
                dict(
                    label=name,
                    first_lift=record["first_sustained_lift_step"],
                    windows=record["windows"],
                )
            )
        )


if __name__ == "__main__":
    main()
