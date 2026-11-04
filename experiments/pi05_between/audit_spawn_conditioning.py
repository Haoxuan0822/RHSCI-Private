"""将训练/新旧初态距离与既有回撤统计对应；区分配置、实测和因果。"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr


def identity(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def displacement(config):
    state = config["dr_state_dict"]
    robot = np.asarray(state["spatial"]["g1_sonic"]["position"])
    target = np.asarray(state["spatial"][state["target"]["uid"]]["position"])
    return target[:2] - robot[:2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "old-root",
        "new-root",
        "recoil-report",
        "calibration",
        "raw",
        "old-trace",
        "new-trace",
        "xml",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bindings = json.loads(args.manifest.read_text())["context"]["source_binding"]
    training = np.array(
        [displacement(b["raw_source"]["raw_environment_config"]) for b in bindings.values()]
    )
    low, high = training[:, 0].min(), training[:, 0].max()
    recoil = json.loads(args.recoil_report.read_text())
    groups, configs, metadata = [], {}, []
    for label, root in (("old", args.old_root), ("new", args.new_root)):
        for level in range(3):
            path = root / f"dr-level-{level}/meta/episodes.jsonl"
            metadata.append(identity(path))
            episodes = [json.loads(line) for line in path.read_text().splitlines()]
            configs[label, level] = {}
            rows = []
            for episode in episodes:
                index = episode["episode_index"]
                c = episode["environment_config"]
                c = json.loads(c) if isinstance(c, str) else c
                configs[label, level][index] = c
                delta = displacement(c)
                matched = [
                    r
                    for r in recoil["rows"]
                    if r["package"] == label
                    and r["level"] == level
                    and r["identity"].endswith(f"episode_{index}")
                ]
                if len(matched) != 2 or any("error" in r for r in matched):
                    raise ValueError(
                        "Need exactly two reliable left-wrist recordings per initial state"
                    )
                windows = [w for r in matched for w in r["windows"]["wrist_raw"]["boundary"]]
                count = sum(
                    w["cosine"] < -0.5 and min(w["before_px"], w["after_px"]) >= 1 for w in windows
                )
                rows.append(
                    dict(
                        episode=index,
                        configured_target_minus_robot_xy_m=delta.tolist(),
                        outside_training_initial_x=not low <= delta[0] <= high,
                        reversals=count,
                        windows=len(windows),
                    )
                )
            groups.append(
                dict(
                    package=label,
                    level=level,
                    episodes=rows,
                    x_range_m=[
                        min(r["configured_target_minus_robot_xy_m"][0] for r in rows),
                        max(r["configured_target_minus_robot_xy_m"][0] for r in rows),
                    ],
                    outside_training_initial_x=sum(r["outside_training_initial_x"] for r in rows),
                    within_group_descriptive_spearman=float(
                        spearmanr(
                            [r["configured_target_minus_robot_xy_m"][0] for r in rows],
                            [r["reversals"] for r in rows],
                        ).statistic
                    ),
                )
            )
    level_pairs = []
    for label in ("old", "new"):
        for index in range(10):
            a, b = [configs[label, level][index] for level in (0, 1)]
            states = [c["dr_state_dict"] for c in (a, b)]
            role_poses = [
                dict(robot=s["spatial"]["g1_sonic"], target=s["spatial"][s["target"]["uid"]])
                for s in states
            ]
            level_pairs.append(
                dict(
                    package=label,
                    episode=index,
                    same_robot_target_pose=role_poses[0] == role_poses[1],
                    changed_dr_groups=[k for k in states[0] if states[0][k] != states[1][k]],
                )
            )
    source = bindings["0"]["raw_source"]
    if identity(args.raw)["sha256"] != source["raw_sha256"]:
        raise ValueError("RAW source differs from training manifest")
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    raw = pq.read_table(
        args.raw, columns=["observation.state", "observation.base_pose", "observation.object_poses"]
    )
    raw_q = model.qpos0.copy()
    raw_q[:7] = raw["observation.base_pose"][0].as_py()
    addresses = [model.jnt_qposadr[model.joint(n).id] for n in source["joint_names"]]
    raw_q[addresses] = raw["observation.state"][0].as_py()
    target_index = source["object_roles"].index("target")
    target = np.asarray(raw["observation.object_poses"][0].as_py()).reshape(-1, 7)[target_index, :3]
    samples = [("training_episode0_frame0", raw_q, target)]
    for label, path in (("old_episode4", args.old_trace), ("new_episode4", args.new_trace)):
        stable = [
            r
            for r in map(json.loads, path.read_text().splitlines())
            if r["kind"] == "env_step" and r["phase"] == "stabilize"
        ][-1]
        samples.append(
            (label, stable["qpos"][: model.nq], np.asarray(stable["objects"]["target"][:3]))
        )
    calibration = json.loads(args.calibration.read_text())
    geometry = []
    for label, qpos, target in samples:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        body = model.body("torso_link").id
        torso = np.eye(4)
        torso[:3, :3], torso[:3, 3] = data.xmat[body].reshape(3, 3), data.xpos[body]
        views = []
        for mount in calibration["camera_mounts_torso_to_ros"]:
            camera = np.linalg.inv(torso @ np.asarray(mount)) @ np.r_[target, 1]
            pixel = np.asarray(calibration["intrinsic"]) @ camera[:3]
            views.append(
                dict(
                    camera_depth_m=float(camera[2]),
                    projected_target_origin_px=(pixel[:2] / pixel[2]).tolist(),
                )
            )
        geometry.append(
            dict(label=label, target_minus_base_x_m=float(target[0] - data.qpos[0]), views=views)
        )
    result = dict(
        inputs=[
            identity(p)
            for k, p in vars(args).items()
            if k not in ("old_root", "new_root", "output")
        ],
        metadata=metadata,
        script_sha256=identity(Path(__file__))["sha256"],
        training_configured_initial_xy_m=training.tolist(),
        training_configured_initial_x_range_m=[float(low), float(high)],
        groups=groups,
        level0_level1_pairs=level_pairs,
        measured_geometry=geometry,
        scope="Configured initial displacement is not measured post-stabilization distance, "
        "all-frame training support, or a causal test. Geometry uses one RAW training frame "
        "and two existing episode4 traces only, with four imperfect mount calibrations. "
        "Within-group correlations are descriptive across10 initial states, not independent "
        "window-level tests. Matched robot/target poses do not hold distractors/scenes fixed.",
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k in ("groups", "measured_geometry")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
