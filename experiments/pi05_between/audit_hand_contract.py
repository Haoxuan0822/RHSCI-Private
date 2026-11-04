"""用真实MJCF、WBC索引及原生PD核对历史手指兼容组合；不推进物理或修改控制器。"""

import argparse
import ast
import base64
import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.utils.gear_wbc_utils import load_config
from simple.core.action import ActionCmd

HISTORICAL_REF = "bfffc460c0d857ae447f18be2529447ea612a11c"
HISTORICAL_SHA = "f30f8dbe2f9e1aa22e946072865e2520019fb360e991861348e893ee8be5495d"


def execute_nodes(nodes, namespace):
    """直接执行选中的原函数/常量，绕过只在完整Kit启动时需要的模块导入。"""
    tree = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<native-hand-audit>", "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simple-source", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    historical = base64.b64decode(
        subprocess.check_output(
            [
                "gh",
                "api",
                f"repos/enterx-x/Psi0/contents/scripts/eval_wbc_hand_order.py?ref={HISTORICAL_REF}",
                "--jq",
                ".content",
            ]
        )
    )
    if hashlib.sha256(historical).hexdigest() != HISTORICAL_SHA:
        raise ValueError("Authenticated historical source changed")
    old = {}
    exec(compile(historical, "<authenticated-historical-hand-adapter>", "exec"), old)
    robot_path = args.simple_source / "src/simple/robots/g1_sonic.py"
    agent_path = args.simple_source / "src/simple/baselines/psi0_decoupled_wbc.py"
    xml_path = args.simple_source / "data/robots/g1_sonic/g1_29dof_with_hand.xml"
    observation_path = args.simple_source / "src/simple/agents/sonic_decoupled_wbc_agent.py"
    config_path = (
        args.simple_source
        / "third_party/decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml"
    )
    lower_path = (
        args.simple_source / "third_party/decoupled_wbc/control/policy/g1_gear_wbc_policy.py"
    )
    if Path(inspect.getfile(G1GearWbcPolicy)).resolve() != lower_path.resolve():
        raise ValueError("Imported lower controller differs from requested native source")
    robot_source, agent_source = robot_path.read_bytes(), agent_path.read_bytes()
    controller_type = type("ControllerTypeForNativePDAssertion", (), {})
    namespace = dict(
        np=np, mujoco=mujoco, ActionCmd=ActionCmd, WholeBodyEEFController=controller_type
    )
    tree = ast.parse(robot_source)
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.endswith("JOINTS") for target in node.targets
        )
    ]
    owner = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1Sonic"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_action"
    )
    execute_nodes([*constants, method], namespace)
    execute_nodes(
        [
            node
            for node in ast.parse(agent_source).body
            if isinstance(node, ast.FunctionDef) and node.name == "from_psi0_upper_joints"
        ],
        namespace,
    )
    mjmodel = mujoco.MjModel.from_xml_path(str(xml_path))
    wbc = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    observation_owner = next(
        node
        for node in ast.parse(observation_path.read_bytes()).body
        if isinstance(node, ast.ClassDef) and node.name == "SonicDecoupledWbcAgent"
    )
    execute_nodes(
        [
            node
            for node in observation_owner.body
            if isinstance(node, ast.FunctionDef) and node.name == "_build_wbc_observation"
        ],
        namespace,
    )
    native_class = type("NativePD", (), {"apply_action": namespace["apply_action"]})
    historical_class = type("HistoricalPD", (), {"apply_action": namespace["apply_action"]})
    historical_agent_class = type(
        "HistoricalObservation", (), {"_build_wbc_observation": namespace["_build_wbc_observation"]}
    )
    old["install_on_classes"](historical_class, historical_agent_class)

    def robot(cls):
        result = cls()
        result.mjModel, result.mjData = mjmodel, mujoco.MjData(mjmodel)
        result.joint_names = namespace["WHOLE_BODY_JOINTS"]
        result.controller = controller_type()
        for side in ("left", "right"):
            setattr(
                result,
                side + "_hand_index",
                np.array(
                    [
                        index
                        for index in range(mjmodel.njnt)
                        if side + "_hand" in mjmodel.joint(index).name
                    ]
                ),
            )
        result.body_joint_index = np.array(
            [
                index
                for index in range(mjmodel.njnt)
                if mjmodel.joint(index).name in result.joint_names[:29]
            ]
        )
        result.num_body_dof, result.num_hand_dof = 29, 7
        result.qpos_offset, result.qvel_offset = 7, 6
        result.sonic_config = {"FREE_BASE": False}
        result.torques, result.torque_limit = np.zeros(43), 1000.0
        return result

    native, legacy = robot(native_class), robot(historical_class)
    native_agent = SimpleNamespace(robot=native, _dwbc_robot_model=wbc)
    historical_agent = historical_agent_class()
    historical_agent.robot, historical_agent._dwbc_robot_model = legacy, wbc
    config, _ = load_config(str(config_path))

    def lower():
        policy = G1GearWbcPolicy.__new__(G1GearWbcPolicy)
        policy.robot_model, policy.config = wbc, config
        policy.gait_indices, policy.freq_cmd = torch.zeros(1), config["freq_cmd"]
        policy.cmd, policy.height_cmd = config["cmd_init"].copy(), config["height_cmd"]
        policy.roll_cmd, policy.pitch_cmd, policy.yaw_cmd = config["rpy_cmd"]
        policy.action = np.zeros(config["num_actions"])
        return policy

    native_lower, historical_lower = lower(), lower()
    trace_content = args.trace.read_bytes()
    rows = [json.loads(line) for line in trace_content.splitlines()]
    frames = {
        row["step"]: row for row in rows if row["kind"] == "env_step" and row["phase"] == "policy"
    }
    upper_names = native.joint_names[15:]
    differences, feedback_errors, network_input_errors, example = [], [], [], None
    for row in rows:
        if row["kind"] != "model_chunk" or row["phase"] != "policy":
            continue
        for offset, action in enumerate(row["actions"]):
            step = row["step"] + offset
            if step not in frames:
                continue
            upper = namespace["from_psi0_upper_joints"](np.asarray(action)[:28])
            old_upper = old["legacy_upper_targets"](upper, legacy)
            commands = []
            for values in (upper, old_upper):
                q = np.zeros(wbc.num_dofs)
                for name, value in zip(upper_names, values, strict=True):
                    q[wbc.dof_index(name)] = value
                commands.append(
                    ActionCmd(
                        "decoupled_wbc",
                        target_q=wbc.get_body_actuated_joints(q),
                        left_hand_q=wbc.get_hand_actuated_joints(q, side="left"),
                        right_hand_q=wbc.get_hand_actuated_joints(q, side="right"),
                    )
                )
            for obj, command in zip((native, legacy), commands, strict=True):
                obj.mjData.qpos[:] = frames[step]["qpos"][: mjmodel.nq]
                obj.mjData.qvel[:] = 0  # 同输入静态PD核对；trace未记录速度，不冒称现场力矩重放。
                obj.apply_action(command)
            differences.append(float(np.max(np.abs(native.torques - legacy.torques))))
            for side in ("left", "right"):
                physical, canonical = old["names"](native, side)
                indices = getattr(native, side + "_hand_index")
                observed = native.mjData.qpos[indices + 6]
                feedback_errors.append(
                    float(np.max(np.abs(observed - old["reorder"](observed, physical, canonical))))
                )
            sim_obs = dict(
                body_q=native.mjData.qpos[native.body_joint_index + 6],
                body_dq=np.zeros(29),
                floating_base_pose=native.mjData.qpos[:7],
                floating_base_vel=np.zeros(6),
            )
            for side in ("left", "right"):
                sim_obs[side + "_hand_q"] = native.mjData.qpos[
                    getattr(native, side + "_hand_index") + 6
                ]
            native_obs = namespace["_build_wbc_observation"](native_agent, sim_obs)
            historical_obs = historical_agent._build_wbc_observation(sim_obs)
            native_input, _ = native_lower.compute_observation(native_obs)
            historical_input, _ = historical_lower.compute_observation(historical_obs)
            network_input_errors.append(float(np.max(np.abs(native_input - historical_input))))
            if example is None:
                example = {
                    side: dict(
                        zip(
                            old["names"](native, side)[0],
                            commands[0][side + "_hand_q"].tolist(),
                            strict=True,
                        )
                    )
                    for side in ("left", "right")
                }
    if not differences or max(differences) != 0 or max(network_input_errors) != 0:
        raise ValueError("Historical compatible targets do not reproduce native PD")
    result = dict(
        scope="Same raw predicted targets and physical positions, zero velocities, no interpolation or simulation. "
        "Historical legacy-action conversion plus named PD; not named-PD-only. "
        "No closed-loop success-rate counterfactual.",
        historical_ref=HISTORICAL_REF,
        historical_sha256=HISTORICAL_SHA,
        sources={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                robot_path,
                agent_path,
                xml_path,
                observation_path,
                config_path,
                lower_path,
            )
        },
        trace_sha256=hashlib.sha256(trace_content).hexdigest(),
        compared_policy_actions=len(differences),
        maximum_native_vs_historical_pd_torque_error_nm=max(differences),
        maximum_physical_vs_canonical_feedback_slot_difference_rad=max(feedback_errors),
        maximum_native_vs_historical_lower_network_input_error=max(network_input_errors),
        lower_network_input_dimensions=len(native_input),
        names={
            side: dict(zip(("physical", "canonical"), old["names"](native, side), strict=True))
            for side in ("left", "right")
        },
        first_raw_action_physical_hand_targets=example,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
