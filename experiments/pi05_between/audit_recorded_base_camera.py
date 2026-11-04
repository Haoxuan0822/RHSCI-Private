"""用身份匹配的原始基座轨迹核验相机反事实；只做CPU正运动学。"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation_error(left, right):
    cosine = (np.trace(left.T @ right) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "raw-info", "manifest", "dataset", "trace", "render", "xml", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # 先核对来源，避免将官方包episode0误当训练包episode0。
    source = json.loads(args.manifest.read_text())["context"]["source_binding"]["0"]["raw_source"]
    if digest(args.raw) != source["raw_sha256"]:
        raise ValueError("Raw parquet differs from training manifest")
    if digest(args.raw_info) != source["raw_metadata_sha256"]:
        raise ValueError("Raw joint metadata differs from training manifest")
    import pyarrow.parquet as pq

    info = json.loads(args.raw_info.read_text())
    raw = pq.read_table(args.raw, columns=["observation.state", "observation.base_pose"])
    processed_path = args.dataset / "data/chunk-000/episode_000000.parquet"
    processed = pq.read_table(processed_path, columns=["states"])
    joints = np.asarray(raw["observation.state"].to_pylist())
    bases = np.asarray(raw["observation.base_pose"].to_pylist())
    states = np.asarray(processed["states"].to_pylist())
    names = info["features"]["observation.state"]["names"]
    arm_names = [
        name for name in names if any(key in name for key in ("shoulder", "elbow", "wrist"))
    ]
    waist_names = ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"]
    upper = joints[:, [names.index(name) for name in arm_names + waist_names]]
    if len(states) != source["num_frames"]:
        raise ValueError("Frame count differs from source binding")
    if upper.shape != states[:, 14:31].shape or not np.array_equal(upper, states[:, 14:31]):
        raise ValueError("Recorded upper joints do not exactly match processed training frames")
    if source["frame_offset"] != 0:
        raise ValueError("This audit requires the verified same-frame rerender binding")
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[model.joint(name).id] for name in names]
    waist_addresses = [model.jnt_qposadr[model.joint(name).id] for name in waist_names]
    torso = model.body("torso_link").id

    def torso_pose(qpos):
        data.qpos[:] = qpos[: model.nq]
        mujoco.mj_forward(model, data)
        return data.xmat[torso].reshape(3, 3).copy(), data.xpos[torso].copy()

    snapshots, previous = {}, None
    for row in map(json.loads, args.trace.read_text().splitlines()):
        if row["kind"] == "env_step":
            previous = row
        elif row["kind"] == "model_chunk":
            if previous is None:
                raise ValueError("No preceding physical snapshot")
            snapshots[row["step"]] = previous
    render = json.loads((args.render / "result.json").read_text())
    estimated = {row["frame"]: row for row in render["records"] if row["mode"] == "expert_camera"}
    frames = [0, 30, 60, 90]
    if any(frame not in snapshots or frame not in estimated for frame in frames):
        raise ValueError("Missing one of four paired snapshots")
    records = []
    for frame in frames:
        true_q = model.qpos0.copy()
        true_q[:7] = bases[frame]
        true_q[addresses] = joints[frame]
        true_rotation, true_position = torso_pose(true_q)
        replay_q = np.asarray(snapshots[frame]["qpos"])[: model.nq].copy()
        corrected_q = replay_q.copy()
        corrected_q[:7] = estimated[frame]["base_qpos"]
        corrected_q[waist_addresses] = states[frame, 28:31]
        true_base_q = replay_q.copy()
        true_base_q[:7] = bases[frame]
        true_waist_q = replay_q.copy()
        true_waist_q[waist_addresses] = states[frame, 28:31]
        comparisons = {}
        for name, qpos in (
            ("replay", replay_q),
            ("recorded_base_only", true_base_q),
            ("recorded_waist_only", true_waist_q),
            ("estimated_camera", corrected_q),
        ):
            rotation, position = torso_pose(qpos)
            comparisons[name] = dict(
                torso_rotation_error_deg=rotation_error(true_rotation, rotation),
                torso_position_error_m=float(np.linalg.norm(true_position - position)),
                base_position_error_m=float(np.linalg.norm(true_q[:3] - qpos[:3])),
                base_qpos=qpos[:7].tolist(),
            )
        records.append(
            dict(frame=frame, recorded_base_qpos=bases[frame].tolist(), comparisons=comparisons)
        )
    report = dict(
        source_files={
            key: dict(path=str(path.resolve()), sha256=digest(path))
            for key, path in vars(args).items()
            if key not in ("dataset", "render", "output")
        },
        processed_parquet_sha256=digest(processed_path),
        render_sha256=digest(args.render / "result.json"),
        script_sha256=digest(Path(__file__)),
        raw_member=source["raw_parquet"],
        frames=len(states),
        arm_and_waist_state_max_abs_rad=float(np.max(np.abs(upper - states[:, 14:31]))),
        records=records,
        scope="Exact raw parquet and joint metadata SHA match the training provenance. All "
        "processed arm/waist states must match same-frame raw states bitwise. Torso "
        "rotation uses native MuJoCo FK with recorded floating base and joints; for a "
        "rigidly attached camera this equals its angular pose error. Position is torso "
        "origin, not camera optical center. Single-factor base/waist replacements are "
        "kinematic comparisons, not physically validated control interventions. "
        "This does not verify recorded RGB capture "
        "timing, run a policy or compare old/new evaluation success rates.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(frames=len(states), records=records), indent=2))


if __name__ == "__main__":
    main()
