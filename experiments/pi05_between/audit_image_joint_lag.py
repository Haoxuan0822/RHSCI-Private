"""用原始机器人FK和已跟踪像素检查图像/关节的相对帧偏移；仅CPU。"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def pose(position, rotation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def fit_point(projections, pixels, train):
    # 每个候选lag只拟合一个固定在刚体上的3D点；不为每帧拟合相机或点。
    linear = pixels[:, :, None] * projections[:, 2:3, :3] - projections[:, :2, :3]
    rhs = projections[:, :2, 3] - pixels * projections[:, 2:3, 3]
    initial = np.linalg.lstsq(linear[train].reshape(-1, 3), rhs[train].reshape(-1), rcond=None)[0]

    def project(point):
        camera = np.einsum("nij,j->ni", projections[:, :, :3], point) + projections[:, :, 3]
        return camera[:, :2] / camera[:, 2:3], camera[:, 2]

    fit = least_squares(lambda point: (project(point)[0] - pixels)[train].ravel(), initial)
    predicted, depth = project(fit.x)
    error = np.linalg.norm(predicted - pixels, axis=1)
    return dict(
        local_point_m=fit.x.tolist(),
        point_norm_m=float(np.linalg.norm(fit.x)),
        minimum_depth_m=float(depth.min()),
        train_mse_px2=float(np.mean(error[train] ** 2)),
        heldout_mse_px2=float(np.mean(error[~train] ** 2)),
        heldout_p50_p95_px=np.percentile(error[~train], [50, 95]).tolist(),
        fit_success=bool(fit.success),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "raw-info", "manifest", "trace", "render", "motion", "xml", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--observed-trace", type=Path)
    parser.add_argument("--eval-config", type=Path)
    args = parser.parse_args()
    if bool(args.observed_trace) != bool(args.eval_config):
        parser.error("Online poses require both --observed-trace and --eval-config")
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.manifest.read_text())["context"]["source_binding"]["0"]["raw_source"]
    for path, expected in (
        (args.raw, source["raw_sha256"]),
        (args.raw_info, source["raw_metadata_sha256"]),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Raw identity differs from training provenance")
    info = json.loads(args.raw_info.read_text())
    raw = pq.read_table(args.raw, columns=["observation.state", "observation.base_pose"])
    joints = np.asarray(raw["observation.state"].to_pylist())
    bases = np.asarray(raw["observation.base_pose"].to_pylist())
    names = info["features"]["observation.state"]["names"]
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[model.joint(name).id] for name in names]
    torso = model.body("torso_link").id

    def body_pose(body):
        return pose(data.xpos[body], data.xmat[body].reshape(3, 3))

    snapshots, previous = {}, None
    for row in map(json.loads, args.trace.read_text().splitlines()):
        if row["kind"] == "env_step":
            previous = row
        elif row["kind"] == "model_chunk":
            snapshots[row["step"]] = previous
    render = json.loads((args.render / "result.json").read_text())
    # Isaac 4.5 Camera.get_world_pose()默认world相机轴；按其W_U @ U_R转ROS。
    world_to_ros_axes = np.eye(4)
    world_to_ros_axes[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    mounts = []
    for row in render["records"]:
        if row["mode"] != "replay":
            continue
        data.qpos[:] = np.asarray(snapshots[row["frame"]]["qpos"])[: model.nq]
        mujoco.mj_forward(model, data)
        camera = row["cameras"]["head_stereo_left"]
        world_camera = pose(
            camera["position"],
            Rotation.from_quat(camera["quaternion"], scalar_first=True).as_matrix(),
        )
        mounts.append(np.linalg.inv(body_pose(torso)) @ world_camera @ world_to_ros_axes)
    if len(mounts) != 4:
        raise ValueError("Need four independent camera-mount snapshots")
    mount_error = float(np.max(np.abs(np.asarray(mounts) - mounts[0])))
    sensor = source["raw_environment_config"]["sensor_cfgs"]["head_stereo"]
    online_poses = None
    online_sensor = None
    online_metadata = None
    if args.observed_trace:
        config = json.loads(args.eval_config.read_text())
        episodes_path = Path(config["data_dir"]) / "meta/episodes.jsonl"
        online_metadata = dict(
            path=str(episodes_path.resolve()),
            sha256=hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
        )
        episode = next(
            row
            for row in map(json.loads, episodes_path.read_text().splitlines())
            if row["episode_index"] == config["episode_start"]
        )
        environment = json.loads(episode["environment_config"])
        online_sensor = environment["sensor_cfgs"]["head_stereo"]
        # 旧采集配置省略后增字段；只接受关闭深度流/无覆盖内参的显式空值。
        optional_fields = {
            "capture_depth": False,
            "capture_pointcloud": False,
            "intrinsics": None,
            "depth_intrinsics": None,
            "depth_to_color_extrinsics": None,
        }
        normalized = []
        for config_sensor in (sensor, online_sensor):
            normalized.append({**optional_fields, **config_sensor})
        if normalized[0] != normalized[1]:
            raise ValueError("Online sensor config differs from calibration provenance")
        stable, steps = None, []
        for row in map(json.loads, args.observed_trace.read_text().splitlines()):
            if row["kind"] != "env_step":
                continue
            if row["phase"] == "stabilize":
                if steps:
                    raise ValueError("Multiple episodes are not supported")
                stable = row
            elif len(steps) < 120:
                if row["step"] != len(steps):
                    raise ValueError("Online policy steps must be contiguous from zero")
                steps.append(row)
        if stable is None or len(steps) != 120:
            raise ValueError("Need final stabilized state and first 120 policy afterstates")
        # 官方VideoRecorder在稳定后重置writer：frame0为稳定末态，frame1为step0后态。
        online_poses = np.asarray([row["qpos"][: model.nq] for row in [stable, *steps]])
    width, height = sensor["width"], sensor["height"]
    focal = width / (2 * np.tan(sensor["fov"] / 2))
    intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    motion = json.loads(args.motion.read_text())
    xy = np.asarray(motion["trajectory_tip_wrist_reference_px"])
    if len(xy) != 121 or max(motion["max_forward_backward_error_px"]) > 0.5:
        raise ValueError("Need reliable 121-frame tracked landmarks")
    candidates = {
        "tip": ("left_hand_index_1_link", "left_hand_middle_1_link"),
        "wrist": ("left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link"),
    }
    projections = {name: [[] for _ in mounts] for group in candidates.values() for name in group}
    for frame in range(121):
        data.qpos[:] = model.qpos0
        if online_poses is None:
            data.qpos[:7] = bases[frame]
            data.qpos[addresses] = joints[frame]
        else:
            data.qpos[:] = online_poses[frame]
        mujoco.mj_forward(model, data)
        for calibration, mount in enumerate(mounts):
            camera_world = np.linalg.inv(body_pose(torso) @ mount)
            for name in projections:
                projections[name][calibration].append(
                    intrinsic @ (camera_world @ body_pose(model.body(name).id))[:3]
                )
    indices = np.arange(30, 90)
    train = (indices // 5) % 2 == 0
    results = []
    for point_index, (label, bodies) in enumerate(candidates.items()):
        for body in bodies:
            for calibration in range(len(mounts)):
                matrices = np.asarray(projections[body][calibration])
                scores = []
                for lag in range(-30, 31):
                    score = fit_point(matrices[indices + lag], xy[indices, point_index], train)
                    scores.append(dict(state_lag_frames=lag, **score))
                results.append(
                    dict(
                        landmark=label,
                        body=body,
                        calibration=calibration,
                        ranking=sorted(scores, key=lambda value: value["heldout_mse_px2"]),
                    )
                )
    report = dict(
        source_files={
            key: dict(
                path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            for key, path in vars(args).items()
            if key not in ("render", "output") and path is not None
        },
        render_sha256=hashlib.sha256((args.render / "result.json").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        camera_mount_max_matrix_difference=mount_error,
        intrinsic=intrinsic.tolist(),
        camera_mounts_torso_to_ros=np.asarray(mounts).tolist(),
        online_sensor_config=online_sensor,
        online_episode_metadata=online_metadata,
        video_frames=indices.tolist(),
        train_frames=indices[train].tolist(),
        heldout_frames=indices[~train].tolist(),
        pose_source=(
            "Online: frame0=final stabilized state; frame i+1=policy env_step i afterstate"
            if online_poses is not None
            else "Identity-matched RAW78 observation state and base pose"
        ),
        results=results,
        scope="CPU native FK, no rendering, model inference or physics rollout. Positive lag "
        "pairs video frame i with state i+lag. Fixed frames30..89 and alternating5-frame "
        "fit/holdout blocks for all lags. A constant3D body-fixed landmark is fitted per lag. "
        "All candidate bodies are reported; tracked surface identity, camera calibration, "
        "mesh/joint conventions and limited motion can affect rankings. Four imperfect "
        "snapshot calibrations are evaluated separately, not assumed identical. This is not an "
        "absolute synchronization proof or full-dataset audit.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                mount_error=mount_error,
                results=[
                    dict(
                        landmark=r["landmark"],
                        body=r["body"],
                        calibration=r["calibration"],
                        top=r["ranking"][:3],
                    )
                    for r in results
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
