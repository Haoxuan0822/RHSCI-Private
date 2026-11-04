"""原生CPU物理回放身份匹配RAW的全关节目标，检验PD语义与帧偏移。"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "raw-info", "manifest", "dataset", "trace", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dds-domain-id", type=int, required=True)
    parser.add_argument("--initial-state", choices=("raw", "trace"), default="raw")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.manifest.read_text())["context"]["source_binding"]["0"]["raw_source"]
    if hashlib.sha256(args.raw.read_bytes()).hexdigest() != source["raw_sha256"]:
        raise ValueError("Wrong source trajectory")
    if hashlib.sha256(args.raw_info.read_bytes()).hexdigest() != source["raw_metadata_sha256"]:
        raise ValueError("Wrong source joint metadata")
    import gymnasium as gym
    import mujoco
    import numpy as np
    import pyarrow.parquet as pq
    import simple.envs  # noqa: F401 -- native task registration
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from simple.cli.eval_decoupled_wbc import _make_sonic_config
    from simple.core.action import ActionCmd
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

    info = json.loads(args.raw_info.read_text())
    table = pq.read_table(args.raw)
    states = np.asarray(table["observation.state"].to_pylist())
    actions = np.asarray(table["action"].to_pylist())
    bases = np.asarray(table["observation.base_pose"].to_pylist())
    base_velocities = np.asarray(table["observation.base_vel"].to_pylist())
    if not 1 <= args.steps < len(states):
        raise ValueError("Need a following recorded state for every physical step")
    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    trace = [json.loads(line) for line in args.trace.read_text().splitlines()]
    replay_targets = [
        row["q"] for row in trace if row["kind"] == "wbc_action" and row["phase"] == "policy"
    ]
    if len(replay_targets) < args.steps:
        raise ValueError("Insufficient recorded WBC targets for the paired control")
    stabilized = [row for row in trace if row["kind"] == "env_step" and row["phase"] == "stabilize"]
    physical = [row for row in trace if row["kind"] == "env_step" and row["phase"] == "policy"]
    if len(stabilized) < 2 or len(physical) < args.steps:
        raise ValueError("Missing initial or baseline physical trajectory")
    episode = json.loads((args.dataset / "meta/episodes.jsonl").read_text().splitlines()[0])
    config = episode["environment_config"]
    config = json.loads(config) if isinstance(config, str) else config
    sonic = _make_sonic_config()
    sonic["DOMAIN_ID"], sonic["INTERFACE"] = args.dds_domain_id, "lo"
    SonicLocoManipEnv._get_obs = lambda env: {
        "joint_qpos": np.asarray(list(env.task.robot.get_robot_qpos().values()))
    }
    env = gym.make(
        "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
        sim_mode="mujoco",
        headless=True,
        sonic_config=sonic,
        disable_env_checker=True,
    )
    results = []
    try:
        env.reset(options={"state_dict": config})
        native = env.unwrapped
        model, data, robot = native.mjModel, native.mjData, native.task.robot
        if abs(model.opt.timestep - 0.005) > 1e-10 or native.control_decimal != 4:
            raise ValueError("Native simulation cadence differs from 200Hz/50Hz")
        addresses = [model.jnt_qposadr[model.joint(name).id] for name in state_names]
        dofs = [model.jnt_dofadr[model.joint(name).id] for name in state_names]
        torso = model.body("torso_link").id
        waist_ids = [state_names.index(f"waist_{axis}_joint") for axis in ("roll", "pitch", "yaw")]
        reference = mujoco.MjData(model)
        recorded_rotations = []
        for frame in range(args.steps + 1):
            reference.qpos[:] = data.qpos
            reference.qpos[:7] = bases[frame]
            reference.qpos[addresses] = states[frame]
            mujoco.mj_forward(model, reference)
            recorded_rotations.append(reference.xmat[torso].reshape(3, 3).copy())
        data.qpos[:7] = bases[0]
        data.qpos[addresses] = states[0]
        data.qvel[:] = 0
        data.qvel[:6] = base_velocities[0]
        data.qvel[dofs] = (states[1] - states[0]) / 0.02
        if args.initial_state == "trace":
            if len(stabilized[-1]["qpos"]) != model.nq:
                raise ValueError("Trace/native scene dimensions differ")
            data.qpos[:] = stabilized[-1]["qpos"]
            mujoco.mj_differentiatePos(
                model, data.qvel, 0.02, np.asarray(stabilized[-2]["qpos"]), data.qpos
            )
        mujoco.mj_forward(model, data)
        initial_base = data.qpos[:7].copy()
        recorded_physical = np.asarray([row["qpos"] for row in physical[: args.steps]])[
            :, addresses
        ]
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        initial = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, data, initial, state_spec)
        wbc_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
        named_indices = [action_names.index(name) for name in wbc_model.joint_names]
        modes = (("raw_all", 0), ("raw_all", 1), ("replay_final", 0), ("raw_lower_only", 0))
        lower_indices = wbc_model.get_joint_group_indices("lower_body")
        if len(lower_indices) != 15:
            raise ValueError("Expected legs12 and waist3 in lower-body group")
        for mode, offset in modes:
            mujoco.mj_setState(model, data, initial, state_spec)
            mujoco.mj_forward(model, data)
            measured, positions, angle_errors, contacts = [], [], [], set()
            for step in range(args.steps):
                target = actions[step + offset, named_indices]
                if mode in ("replay_final", "raw_lower_only"):
                    recorded_target = np.asarray(replay_targets[step]).copy()
                    if recorded_target.shape != target.shape:
                        raise ValueError("WBC target dimensions differ")
                    if mode == "raw_lower_only":
                        recorded_target[lower_indices] = target[lower_indices]
                    target = recorded_target
                command = ActionCmd(
                    "decoupled_wbc",
                    target_q=wbc_model.get_body_actuated_joints(target),
                    left_hand_q=wbc_model.get_hand_actuated_joints(target, side="left"),
                    right_hand_q=wbc_model.get_hand_actuated_joints(target, side="right"),
                )
                for _ in range(native.control_decimal):
                    robot.apply_action(command)
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                measured.append(data.qpos[addresses].copy())
                positions.append(data.qpos[:7].copy())
                delta = recorded_rotations[step + 1].T @ data.xmat[torso].reshape(3, 3)
                angle_errors.append(
                    float(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
                )
                for contact in data.contact:
                    contacts.add(
                        tuple(
                            model.body(model.geom(g).bodyid.item()).name
                            for g in (contact.geom1, contact.geom2)
                        )
                    )
            measured, positions = np.asarray(measured), np.asarray(positions)
            result = dict(
                mode=mode,
                action_offset=offset,
                joint_rmse_rad=np.sqrt(
                    np.mean((measured - states[1 : args.steps + 1]) ** 2, axis=0)
                ).tolist(),
                waist_rmse_rad=np.sqrt(
                    np.mean(
                        (measured[:, waist_ids] - states[1 : args.steps + 1, waist_ids]) ** 2,
                        axis=0,
                    )
                ).tolist(),
                base_xyz_rmse_m=np.sqrt(
                    np.mean((positions[:, :3] - bases[1 : args.steps + 1, :3]) ** 2, axis=0)
                ).tolist(),
                torso_rotation_error_deg=angle_errors,
                original_trace_waist_max_abs_rad=float(
                    np.max(np.abs(measured[:, waist_ids] - recorded_physical[:, waist_ids]))
                ),
                request_frames={
                    str(frame): dict(
                        base_qpos=positions[frame - 1].tolist(),
                        torso_rotation_error_deg=angle_errors[frame - 1],
                    )
                    for frame in (30, 60, 90, 120)
                    if frame <= args.steps
                },
                contact_body_pairs=sorted(contacts),
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in ("torso_rotation_error_deg", "contact_body_pairs")
                    }
                ),
                flush=True,
            )
        report = dict(
            inputs={
                key: dict(
                    path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
                for key, path in vars(args).items()
                if isinstance(path, Path) and key not in ("dataset", "output")
            },
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            steps=args.steps,
            initial_state=args.initial_state,
            initial_base_qpos=initial_base.tolist(),
            joint_names=state_names,
            physics_dt=model.opt.timestep,
            control_decimal=native.control_decimal,
            results=results,
            limitation="Native MJ/PD only, no VLA or lower-body RL policy. Raw initialization uses recorded base velocity and joint velocity estimated from frames0/1; trace initialization uses full final stabilized qpos and estimated velocity from its last two snapshots. Saved lower targets do not recompute feedback. Original-trace waist error must validate the trace-initialized baseline before interpreting interventions. Full target replay is not a proposed deployable 36D policy or old/new evaluation success comparison.",
        )
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
