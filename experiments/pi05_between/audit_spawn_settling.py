"""量化配置出生位移与真实WBC稳定阶段位移；不把两者相加自动判作生成器bug。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(metadata, trace, episode):
    metadata_bytes, trace_bytes = metadata.read_bytes(), trace.read_bytes()
    entries = [json.loads(line) for line in metadata_bytes.splitlines()]
    item = next(row for row in entries if row["episode_index"] == episode)
    config = item["environment_config"]
    if isinstance(config, str):
        config = json.loads(config)
    configured = np.asarray(config["dr_state_dict"]["spatial"]["g1_sonic"]["position"])
    rows = [json.loads(line) for line in trace_bytes.splitlines()]
    stable = [row for row in rows if row["kind"] == "env_step" and row["phase"] == "stabilize"]
    policy = [row for row in rows if row["kind"] == "env_step" and row["phase"] == "policy"]
    if not stable or not policy:
        raise ValueError("Need completed stabilization and at least one policy step")
    root = np.asarray(stable[-1]["qpos"][:3])
    target = np.asarray(stable[-1]["objects"]["target"][:3])
    return dict(
        metadata=str(metadata.resolve()),
        trace=str(trace.resolve()),
        episode=episode,
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
        configured_spawn_xyz_m=configured.tolist(),
        stabilization_steps=len(stable),
        first_recorded_stabilization_base_xyz_m=stable[0]["qpos"][:3],
        last_stabilization_base_xyz_m=root.tolist(),
        last_stabilization_target_xyz_m=target.tolist(),
        last_stabilization_base_minus_configured_xy_m=(root[:2] - configured[:2]).tolist(),
        target_minus_base_x_before_policy_m=float(target[0] - root[0]),
        stabilization_target_z_range_m=[
            float(min(row["objects"]["target"][2] for row in stable)),
            float(max(row["objects"]["target"][2] for row in stable)),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for version in ("old", "new"):
        parser.add_argument(f"--{version}-metadata", type=Path, required=True)
        parser.add_argument(f"--{version}-trace", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old = analyze(args.old_metadata, args.old_trace, args.episode)
    new = analyze(args.new_metadata, args.new_trace, args.episode)
    result = dict(
        scope="Measured cumulative spawn translation and WBC settling. Does not establish the "
        "unpublished dataset generator's intent, double-stabilization code bug, or success-rate causality.",
        old=old,
        new=new,
        new_minus_old_configured_xy_m=(
            np.asarray(new["configured_spawn_xyz_m"])[:2]
            - np.asarray(old["configured_spawn_xyz_m"])[:2]
        ).tolist(),
        new_minus_old_stabilized_xy_m=(
            np.asarray(new["last_stabilization_base_xyz_m"])[:2]
            - np.asarray(old["last_stabilization_base_xyz_m"])[:2]
        ).tolist(),
        new_stabilized_minus_old_configured_xy_m=(
            np.asarray(new["last_stabilization_base_xyz_m"])[:2]
            - np.asarray(old["configured_spawn_xyz_m"])[:2]
        ).tolist(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
