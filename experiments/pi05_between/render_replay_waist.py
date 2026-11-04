"""用官方环境渲染保存的回放qpos及仅腰部修正快照；不运行策略或MJ物理。"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "trace", "expert-requests", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--renderer-gpu", type=int, required=True)
    parser.add_argument("--dds-domain-id", type=int, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
    os.environ.setdefault("SIMPLE_ISAAC_LOG_LEVEL", "warning")
    sys.argv = [
        sys.argv[0],
        f"--/renderer/activeGpu={args.renderer_gpu}",
        "--/renderer/multiGpu/enabled=False",
        f"--/plugins/carb.tasking.plugin/threadCount={args.cpu_threads}",
        f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={args.cpu_threads}",
    ]
    import gymnasium as gym
    import mujoco
    import numpy as np
    import simple.envs  # noqa: F401 -- registers native environments
    from PIL import Image
    from simple.cli.eval_decoupled_wbc import _make_sonic_config

    metadata = (args.dataset / "meta/episodes.jsonl").read_bytes()
    episode = next(
        row for row in map(json.loads, metadata.splitlines()) if row["episode_index"] == 0
    )
    config = episode["environment_config"]
    config = json.loads(config) if isinstance(config, str) else config
    request_content = args.expert_requests.read_bytes()
    requests = {row["frame"]: row for row in map(json.loads, request_content.splitlines())}
    previous, snapshots = None, {}
    trace_content = args.trace.read_bytes()
    for row in map(json.loads, trace_content.splitlines()):
        if row["kind"] == "env_step":
            previous = row
        elif row["kind"] == "model_chunk":
            if previous is None:
                raise ValueError("Missing preceding snapshot")
            if row["step"] in snapshots:
                raise ValueError("Repeated model request step")
            snapshots[row["step"]] = previous
    if sorted(requests) != [0, 30, 60, 90] or set(requests) != set(snapshots):
        raise ValueError("Expected matching four recorded frames")
    sonic = _make_sonic_config()
    sonic["DOMAIN_ID"], sonic["INTERFACE"] = args.dds_domain_id, "lo"
    env = gym.make(
        "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
        sim_mode="mujoco_isaac",
        headless=True,
        sonic_config=sonic,
        disable_env_checker=True,
    )
    records = []
    try:
        env.reset(options={"state_dict": config})
        native = env.unwrapped
        addresses = [
            native.mjModel.jnt_qposadr[native.mjModel.joint(name).id]
            for name in ("waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint")
        ]
        for frame, request in sorted(requests.items()):
            original = np.asarray(snapshots[frame]["qpos"])
            if original.shape != native.mjData.qpos.shape:
                raise ValueError("Saved qpos and reconstructed scene dimensions differ")
            for mode in ("replay", "expert_waist"):
                qpos = original.copy()
                if mode == "expert_waist":
                    qpos[addresses] = np.asarray(request["expert_state"])[28:31]
                native.mjData.qpos[:] = qpos
                native.mjData.qvel[:] = 0
                mujoco.mj_forward(native.mjModel, native.mjData)
                # 只同步并渲染同一静态姿态；不调用env.step或mujoco.step。
                for _ in range(3):
                    native.isaac.step(native.mujoco)
                observation = native._get_obs()
                observed = observation["joint_qpos"][[13, 14, 12]]
                error = float(np.max(np.abs(observed - qpos[addresses])))
                if error > 1e-4 or not np.array_equal(native.mjData.qpos, qpos):
                    raise ValueError("Static render changed the requested waist or MuJoCo qpos")
                images, cameras = {}, {}
                for name in ("head_stereo_left", "head_stereo_right"):
                    path = args.output / f"{mode}-frame-{frame:04d}-{name}.png"
                    Image.fromarray(observation[name]).save(path)
                    images[name] = dict(
                        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                    )
                    position, quaternion = native.isaac.cameras[name].get_world_pose()
                    cameras[name] = dict(position=position.tolist(), quaternion=quaternion.tolist())
                records.append(
                    dict(
                        frame=frame,
                        mode=mode,
                        waist_error_rad=error,
                        images=images,
                        cameras=cameras,
                    )
                )
                (args.output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        report = dict(
            trace_sha256=hashlib.sha256(trace_content).hexdigest(),
            requests_sha256=hashlib.sha256(request_content).hexdigest(),
            episode_metadata_sha256=hashlib.sha256(metadata).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            records=records,
            limitation="Static snapshots with replay floating base retained. Training full base trajectory unavailable; waist correction does not restore the entire recorded training state. No policy, MuJoCo dynamics, success-rate test or material modification.",
        )
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("STATIC_WAIST_RENDER_COMPLETE", len(records), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
