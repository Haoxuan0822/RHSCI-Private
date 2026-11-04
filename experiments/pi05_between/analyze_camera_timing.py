"""核对实际相机参考帧、SDK时间和策略消费的观测图像。"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-policy-steps", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    content = args.trace.read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    envs = [row for row in rows if row["kind"] == "env_step"]
    policy = [row for row in envs if row["phase"] == "policy"]
    if [row["step"] for row in policy] != list(range(args.expected_policy_steps)):
        raise ValueError("Need the complete consecutive policy-step trace")
    if not envs or any(not row.get("cameras") for row in envs):
        raise ValueError("Missing camera timing records")
    queries, previous = [], None
    for row in rows:
        if row["kind"] == "env_step":
            previous = row
        elif row["kind"] == "model_chunk":
            if previous is None:
                raise ValueError("Model query has no preceding observed frame")
            camera = previous["cameras"]["head_stereo_left"]
            queries.append(
                dict(
                    step=row["step"],
                    preceding_observation_phase=previous["phase"],
                    preceding_observation_step=previous["step"],
                    reference_frame=camera["direct_frame"],
                    matches_preceding_observation_sha256=(
                        row["image_sha256"] == camera["image_sha256"]
                    ),
                )
            )
    if [query["step"] for query in queries] != list(range(0, args.expected_policy_steps, 30)):
        raise ValueError("Need one complete 30-step model chunk at each expected boundary")
    cameras = {}
    for name in envs[0]["cameras"]:
        values = [row["cameras"][name] for row in envs]
        frames = np.asarray([row["direct_frame"] for row in values], dtype=np.int64)
        times = np.asarray([row["direct_sim_time"] for row in values])
        age = np.asarray([row["isaac_world_time"] for row in values]) - times
        if not np.isfinite(times).all() or not np.isfinite(age).all():
            raise ValueError(f"Non-finite camera timestamps: {name}")
        cameras[name] = dict(
            observations=len(values),
            reference_frame_delta_counts=dict(Counter(map(int, np.diff(frames)))),
            reference_time_delta_min_median_max_s=np.percentile(
                np.diff(times), [0, 50, 100]
            ).tolist(),
            world_minus_reference_time_min_median_max_s=np.percentile(age, [0, 50, 100]).tolist(),
            callback_minus_direct_frame_counts=dict(
                Counter(row["callback_frame"] - row["direct_frame"] for row in values)
            ),
            repeated_image_pairs=sum(
                a["image_sha256"] == b["image_sha256"] for a, b in zip(values, values[1:])
            ),
        )
    report = dict(
        trace=str(args.trace),
        trace_sha256=hashlib.sha256(content).hexdigest(),
        observation_phases=dict(Counter(row["phase"] for row in envs)),
        cameras=cameras,
        model_queries=queries,
        limitation="ReferenceTime and RGB use the same render product; metadata alone does not establish exact per-pixel or bone-transform synchronization. Isaac and MuJoCo have separate clocks. Image hashes verify which returned observation the policy consumed, not physical image correctness.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
