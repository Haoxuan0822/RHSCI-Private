"""原生MJ/PD重放，单独核对腰PD或导航门限；不运行VLA或正式评测。"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "trace", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dds-domain-id", type=int, required=True)
    parser.add_argument("--closed-loop-wbc", action="store_true")
    parser.add_argument("--camera-render", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--diagnostic-ungated-navigation", action="store_true")
    args = parser.parse_args()
    if args.diagnostic_ungated_navigation and (not args.closed_loop_wbc or args.camera_render):
        parser.error("Navigation comparison requires closed-loop WBC and no camera comparison")
    if args.output.exists():
        raise FileExistsError(args.output)
    import gymnasium as gym
    import mujoco
    import numpy as np
    import pyarrow.parquet as pq
    import simple.envs  # noqa: F401 -- native registry
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from simple.cli.eval_decoupled_wbc import _make_sonic_config
    from simple.core.action import ActionCmd
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

    metadata = next(
        value
        for line in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()
        if (value := json.loads(line))["episode_index"] == args.episode
    )
    config = metadata["environment_config"]
    config = json.loads(config) if isinstance(config, str) else config
    content = args.trace.read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    previous = [r for r in rows if r["kind"] == "env_step" and r["phase"] == "stabilize"]
    targets = [r for r in rows if r["kind"] == "wbc_action" and r["phase"] == "policy"]
    physical = [r for r in rows if r["kind"] == "env_step" and r["phase"] == "policy"]
    if len(targets) != 120 or len(physical) != 120 or len(previous) < 2:
        raise ValueError("Expected stabilized complete120-step expert replay")
    actions = states = None
    if not args.diagnostic_ungated_navigation:
        table = pq.read_table(
            args.dataset / f"data/chunk-000/episode_{args.episode:06d}.parquet",
            columns=["action", "states"],
        )
        actions = np.asarray(table["action"].to_pylist())
        states = np.asarray(table["states"].to_pylist())
    sonic = _make_sonic_config()
    sonic["DOMAIN_ID"], sonic["INTERFACE"] = args.dds_domain_id, "lo"
    # 仅关闭图像读取；保留原生reset、场景构建、机器人PD和MJ物理实现。
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
            raise ValueError("Expected original200Hz physics and50Hz target rate")
        q0, qprev = (np.asarray(row["qpos"]) for row in previous[-2:][::-1])
        if q0.shape != data.qpos.shape:
            raise ValueError("Scene reconstruction dimensions differ")
        data.qpos[:] = q0
        mujoco.mj_differentiatePos(model, data.qvel, 0.02, qprev, q0)
        mujoco.mj_forward(model, data)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        initial = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, data, initial, state_spec)
        wbc = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
        agent = None
        native_lower = ungated_lower = None
        if args.closed_loop_wbc:
            import onnxruntime as ort
            import torch
            from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent

            # 限制原生ONNX CPU线程；不改模型/计算精度或控制公式。
            original_session = ort.InferenceSession
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            ort.InferenceSession = lambda *a, **kw: original_session(*a, sess_options=options, **kw)
            sonic["enable_waist"] = True
            agent = SonicDecoupledWbcAgent(robot, sonic)
            if args.diagnostic_ungated_navigation:
                import ast
                import inspect
                import subprocess
                import textwrap
                from types import MethodType

                from decoupled_wbc.control.policy import g1_gear_wbc_policy as module

                native_lower = agent._wbc_policy.lower_body_policy.get_action
                source = subprocess.check_output(
                    ["git", "show", "5b304e4^:control/policy/g1_gear_wbc_policy.py"],
                    cwd=Path(module.__file__).resolve().parents[2],
                    text=True,
                )
                cls = next(
                    node
                    for node in ast.parse(source).body
                    if isinstance(node, ast.ClassDef) and node.name == "G1GearWbcPolicy"
                )
                method = next(
                    node
                    for node in cls.body
                    if isinstance(node, ast.FunctionDef) and node.name == "get_action"
                )
                current = inspect.getsource(module.G1GearWbcPolicy.get_action)
                expected = textwrap.dedent(current).replace(" or np.abs(vyaw_flag) < 0.1", "", 1)
                expected_method = ast.parse(expected).body[0]
                # inspect.dedent仅改变docstring缩进；比较实际方法体，保留完整控制逻辑。
                for function in (method, expected_method):
                    if (
                        isinstance(function.body[0], ast.Expr)
                        and isinstance(function.body[0].value, ast.Constant)
                        and isinstance(function.body[0].value.value, str)
                    ):
                        function.body.pop(0)
                if ast.dump(method) != ast.dump(expected_method):
                    raise ValueError("Historical method is not exactly the one-line gate change")
                namespace = dict(vars(module))
                exec(
                    compile(
                        ast.Module(body=[method], type_ignores=[]),
                        "<diagnostic-native-parent-gate-only>",
                        "exec",
                    ),
                    namespace,
                )
                ungated_lower = MethodType(
                    namespace["get_action"], agent._wbc_policy.lower_body_policy
                )
        policy_goals = [r for r in rows if r["kind"] == "wbc_goal" and r["phase"] == "policy"]
        waist_names = ("waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint")
        addresses = [model.jnt_qposadr[model.joint(name).id] for name in waist_names]
        recorded = np.asarray([row["qpos"] for row in physical])[:, addresses]
        modes = (
            ("recorded_wbc", "ungated_navigation")
            if args.diagnostic_ungated_navigation
            else ("recorded_wbc", "dataset_direct_pd")
        )
        for mode in modes:
            if agent is not None:
                if native_lower is not None:
                    agent._wbc_policy.lower_body_policy.get_action = (
                        ungated_lower if mode == "ungated_navigation" else native_lower
                    )
                first_goal = next(row["goal"] for row in rows if row["kind"] == "wbc_goal")
                agent._wbc_policy.reset(init_time=first_goal["timestamp"] - 1)
                agent._wbc_policy.lower_body_policy.use_policy_action = True
                current_q = np.asarray(previous[0]["qpos"])
                previous_q = current_q.copy()
                pending = None
                # 以trace稳定姿态暖起原生下身历史；速度只可由相邻qpos估计。
                for warm_row in rows:
                    if warm_row["phase"] == "policy":
                        break
                    if warm_row["kind"] == "wbc_goal":
                        pending = warm_row["goal"]
                    elif warm_row["kind"] == "wbc_action":
                        data.qpos[:] = current_q
                        mujoco.mj_differentiatePos(model, data.qvel, 0.02, previous_q, current_q)
                        mujoco.mj_forward(model, data)
                        agent._wbc_policy.set_observation(
                            agent._build_wbc_observation(robot.prepare_obs())
                        )
                        agent._wbc_policy.set_goal(pending)
                        agent._wbc_policy.get_action(time=warm_row["query_time"])
                    elif warm_row["kind"] == "env_step":
                        previous_q, current_q = current_q, np.asarray(warm_row["qpos"])
                agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros(1)
            mujoco.mj_setState(model, data, initial, state_spec)
            mujoco.mj_forward(model, data)
            measured, positions = [], []
            for step, row in enumerate(targets):
                q = np.array(row["q"], copy=True)
                if agent is not None:
                    agent._wbc_policy.set_observation(
                        agent._build_wbc_observation(robot.prepare_obs())
                    )
                    agent._wbc_policy.set_goal(policy_goals[step]["goal"])
                    q = agent._wbc_policy.get_action(time=row["query_time"])["q"].copy()
                if mode == "dataset_direct_pd":
                    for name, value in zip(waist_names, actions[step, 28:31], strict=True):
                        q[wbc.dof_index(name)] = value
                command = ActionCmd(
                    "decoupled_wbc",
                    target_q=wbc.get_body_actuated_joints(q),
                    left_hand_q=wbc.get_hand_actuated_joints(q, side="left"),
                    right_hand_q=wbc.get_hand_actuated_joints(q, side="right"),
                )
                for _ in range(native.control_decimal):
                    robot.apply_action(command)
                    mujoco.mj_step(model, data)
                measured.append(data.qpos[addresses].copy())
                positions.append(data.qpos.copy())
            measured = np.asarray(measured)
            error = measured - states[1:121, 28:31] if states is not None else None
            replay_error = measured - recorded
            result = dict(
                mode=mode,
                expert_state_rmse_rad=(
                    np.sqrt(np.mean(error**2, axis=0)).tolist() if error is not None else None
                ),
                recorded_physics_rmse_rad=np.sqrt(np.mean(replay_error**2, axis=0)).tolist(),
                recorded_physics_max_abs_rad=float(np.abs(replay_error).max()),
                request_frames={
                    str(frame): dict(
                        measured=measured[frame - 1].tolist(),
                        expert=states[frame, 28:31].tolist() if states is not None else None,
                        original=recorded[frame - 1].tolist(),
                    )
                    for frame in (30, 60, 90, 120)
                },
                measured_waist=measured.tolist(),
                full_qpos=np.asarray(positions).tolist(),
            )
            if args.camera_render is not None:
                render = json.loads((args.camera_render / "result.json").read_text())
                saved = {
                    row["frame"]: row for row in render["records"] if row["mode"] == "expert_camera"
                }
                snapshots, last = {}, None
                for trace_row in rows:
                    if trace_row["kind"] == "env_step":
                        last = trace_row
                    elif trace_row["kind"] == "model_chunk":
                        snapshots[trace_row["step"]] = last
                torso = model.body("torso_link").id
                reference_data = mujoco.MjData(model)
                camera_results = []
                for frame in (30, 60, 90):
                    desired_q = np.array(snapshots[frame]["qpos"], copy=True)
                    desired_q[:7] = saved[frame]["base_qpos"]
                    desired_q[addresses] = states[frame, 28:31]
                    reference_data.qpos[:] = desired_q
                    mujoco.mj_forward(model, reference_data)
                    desired_rotation = reference_data.xmat[torso].reshape(3, 3).copy()
                    reference_data.qpos[:] = positions[frame - 1]
                    mujoco.mj_forward(model, reference_data)
                    actual_rotation = reference_data.xmat[torso].reshape(3, 3)
                    angle = np.degrees(
                        np.arccos(
                            np.clip((np.trace(desired_rotation.T @ actual_rotation) - 1) / 2, -1, 1)
                        )
                    )
                    camera_results.append(dict(frame=frame, torso_rotation_error_deg=float(angle)))
                result["estimated_training_camera_rotation"] = camera_results
            results.append(result)
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k not in ("measured_waist", "full_qpos")}
                ),
                flush=True,
            )
        report = dict(
            trace_sha256=hashlib.sha256(content).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            closed_loop_wbc=args.closed_loop_wbc,
            diagnostic_ungated_navigation=args.diagnostic_ungated_navigation,
            episode=args.episode,
            results=results,
            limitation="No VLA, rendering or success-rate test. "
            + (
                "Native lower controller recomputes leg targets; recorded high-level goals retained. "
                if args.closed_loop_wbc
                else "Other body/hand targets remain recorded feedforward values. "
            )
            + "Initial velocity estimated from last two stabilized trace positions; baseline "
            "physical replay error must be checked. Direct-PD agreement alone does not prove "
            "the original recording source or every model failure. Navigation comparison "
            "retains recorded VLA goals; it does not test how VLA would react to changed images.",
        )
        if args.diagnostic_ungated_navigation:
            from scipy.spatial.transform import Rotation

            baseline, changed = [np.asarray(result["full_qpos"]) for result in results]
            actual = np.asarray([row["qpos"] for row in physical])
            joint_addresses = {
                name: int(model.jnt_qposadr[model.joint(name).id]) for name in robot.joint_names
            }
            arm_addresses = [
                address
                for name, address in joint_addresses.items()
                if name.startswith(
                    (
                        "left_shoulder_",
                        "right_shoulder_",
                        "left_elbow_",
                        "right_elbow_",
                        "left_wrist_",
                        "right_wrist_",
                    )
                )
            ]
            if len(arm_addresses) != 14:
                raise ValueError("Need the fourteen named arm joints")
            all_addresses = list(joint_addresses.values())
            rotations = [
                Rotation.from_quat(value[:, [4, 5, 6, 3]]) for value in (baseline, changed)
            ]
            torso_angles = []
            torso = model.body("torso_link").id
            temporary = mujoco.MjData(model)
            for left, right in zip(baseline, changed, strict=True):
                matrices = []
                for position in (left, right):
                    temporary.qpos[:] = position
                    mujoco.mj_forward(model, temporary)
                    matrices.append(temporary.xmat[torso].reshape(3, 3).copy())
                torso_angles.append(
                    float(
                        np.degrees(
                            np.arccos(
                                np.clip((np.trace(matrices[0].T @ matrices[1]) - 1) / 2, -1, 1)
                            )
                        )
                    )
                )
            report["navigation_physics_comparison"] = dict(
                joint_qpos_addresses=joint_addresses,
                native_replay_joint_max_abs_error_rad=float(
                    np.abs(baseline[:, all_addresses] - actual[:, all_addresses]).max()
                ),
                changed_vs_native_arm_max_abs_rad=float(
                    np.abs(baseline[:, arm_addresses] - changed[:, arm_addresses]).max()
                ),
                changed_vs_native_base_position_max_mm=float(
                    np.linalg.norm(baseline[:, :3] - changed[:, :3], axis=-1).max() * 1000
                ),
                changed_vs_native_base_rotation_max_deg=float(
                    np.degrees((rotations[0].inv() * rotations[1]).magnitude()).max()
                ),
                changed_vs_native_torso_rotation_deg=torso_angles,
                changed_vs_native_torso_rotation_max_deg=max(torso_angles),
            )
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
