"""Opt-in tracing around the existing official policy and simulation methods."""

import functools
import hashlib
import json
import os
import time


def install_control_trace(
    path,
    *,
    control_clock="wall",
    execute_horizon=None,
    legacy_navigation=False,
    camera_timing=False,
):
    """默认只记录；诊断可显式对照时钟、执行前缀或旧导航语义。"""
    import numpy as np
    from PIL import Image
    from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("x", buffering=1)
    context = {"step": 0, "phase": "stabilize"}
    if control_clock == "simulation":
        install_simulation_clock(SonicLocoManipEnv)
    navigation = None
    if legacy_navigation:
        from legacy_navigation import install

        navigation = install()

    def encode(value):
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        raise TypeError(type(value).__name__)

    def emit(kind, **values):
        stream.write(
            json.dumps(
                {"kind": kind, "wall_time": time.monotonic(), **context, **values},
                default=encode,
            )
            + "\n"
        )

    original_init = Psi0DecoupledWbcAgent.__init__

    @functools.wraps(original_init)
    def init(agent, *args, **kwargs):
        original_init(agent, *args, **kwargs)
        original_query = agent.client.query_action
        original_goal = agent._wbc_policy.set_goal
        original_action = agent._wbc_policy.get_action
        lower = agent._wbc_policy.lower_body_policy
        original_lower_action = lower.get_action

        def lower_action(*args, **kwargs):
            emit(
                "lower_input",
                navigate_cmd=args[4] if len(args) > 4 else kwargs.get("interpolated_navigate_cmd"),
                previous_velocity=lower.cmd.copy(),
                previous_heading=lower.target_yaw_cmd,
                floating_base_pose=lower.observation["floating_base_pose"].copy(),
            )
            result = original_lower_action(*args, **kwargs)
            emit("lower_output", velocity=lower.cmd.copy(), navigate_cmd=result["navigate_cmd"])
            return result

        lower.get_action = lower_action
        # 两路诊断都记录实际站/走网络调用；包装原网络，不替换其计算。
        for attribute, label in (("policy_1", "stand"), ("policy_2", "walk")):
            original_network = getattr(lower, attribute)

            def run_network(*args, network=original_network, mode=label, **kwargs):
                emit("lower_policy", mode=mode, command=lower.cmd.copy())
                return network(*args, **kwargs)

            setattr(lower, attribute, run_network)

        def query(*args, **kwargs):
            images = args[0]
            image_path = path.parent / f"observation-step-{context['step']:04d}.png"
            Image.fromarray(images["rgb_head_stereo_left"]).save(image_path)
            start = time.monotonic()
            result = original_query(*args, **kwargs)
            predicted = result[0]
            executed = predicted if execute_horizon is None else predicted[:execute_horizon]
            if execute_horizon is not None and (
                len(predicted) != 30 or len(executed) != execute_horizon
            ):
                raise ValueError(
                    "Execute-horizon diagnostic requires a complete 30-step model chunk"
                )
            emit(
                "model_chunk",
                duration=time.monotonic() - start,
                actions=executed,
                predicted_actions=predicted,
                observed_state=args[2],
                instruction=args[1],
                image_path=str(image_path),
                image_sha256=(
                    hashlib.sha256(images["rgb_head_stereo_left"].tobytes()).hexdigest()
                    if camera_timing
                    else None
                ),
            )
            return result if execute_horizon is None else (executed, *result[1:])

        def goal(value):
            emit("wbc_goal", goal=value)
            return original_goal(value)

        def action(*args, **kwargs):
            result = original_action(*args, **kwargs)
            emit("wbc_action", query_time=kwargs.get("time"), q=result["q"])
            return result

        agent.client.query_action = query
        agent._wbc_policy.set_goal = goal
        agent._wbc_policy.get_action = action

    original_get_action = Psi0DecoupledWbcAgent.get_action

    @functools.wraps(original_get_action)
    def get_action(agent, *args, **kwargs):
        context.update(step=agent._global_step_idx, phase="policy")
        start = time.monotonic()
        result = original_get_action(agent, *args, **kwargs)
        emit("agent_action", duration=time.monotonic() - start)
        return result

    original_step = SonicLocoManipEnv.step

    @functools.wraps(original_step)
    def step(env, action):
        start = time.monotonic()
        before = float(env.mujoco.mjData.time)
        result = original_step(env, action)
        cameras = None
        if camera_timing:
            cameras = {}
            for name, camera in env.isaac.cameras.items():
                # 与本机Isaac 4.5 Camera获取图像参考帧的实现一致，只读取现有annotator。
                frame = int(camera._fabric_time_annotator.get_data()["referenceTimeNumerator"])
                cached = camera.get_current_frame()
                cameras[name] = dict(
                    direct_frame=frame,
                    direct_sim_time=float(
                        camera._core_nodes_interface.get_sim_time_at_swh_frame(frame)
                    ),
                    callback_frame=int(cached["rendering_frame"]),
                    callback_sim_time=float(cached["rendering_time"]),
                    isaac_world_time=float(env.isaac.world.current_time),
                    image_sha256=hashlib.sha256(result[0][name].tobytes()).hexdigest(),
                )
        emit(
            "env_step",
            duration=time.monotonic() - start,
            sim_before=before,
            sim_after=float(env.mujoco.mjData.time),
            qpos=env.mujoco.mjData.qpos.copy(),
            observed_joints=result[0]["joint_qpos"],
            cameras=cameras,
            objects={
                name: np.concatenate((body.xpos, body.xquat))
                for name, body in env.mujoco.mj_objects.items()
            },
        )
        return result

    Psi0DecoupledWbcAgent.__init__ = init
    Psi0DecoupledWbcAgent.get_action = get_action
    SonicLocoManipEnv.step = step
    emit(
        "trace_installed",
        changes_control=(
            control_clock != "wall" or execute_horizon not in (None, 30) or legacy_navigation
        ),
        control_clock=control_clock,
        execute_horizon=execute_horizon,
        navigation=navigation,
        camera_timing=camera_timing,
        modified_policy_switch_behavior=os.environ.get("MODIFIED_POLICY_SWITCH_BEHAVIOR", "0"),
    )


def install_simulation_clock(env_class):
    """只替换控制模块的 time 引用，HTTP、日志和全局标准库墙钟保持原样。"""
    from types import SimpleNamespace

    from decoupled_wbc.control.policy import g1_decoupled_whole_body_policy
    from simple.agents import sonic_decoupled_wbc_agent
    from simple.baselines import psi0_decoupled_wbc

    origin = time.monotonic()
    active = {}

    def now():
        env = active.get("env")
        mujoco = getattr(env, "mujoco", None)
        return origin + (float(mujoco.mjData.time) if mujoco is not None else 0.0)

    clock = SimpleNamespace(monotonic=now)
    sonic_decoupled_wbc_agent.time = clock
    psi0_decoupled_wbc.time = clock
    # 复合策略的 last_goal_time 与显式 get_action(time=...) 必须同域。
    g1_decoupled_whole_body_policy.time_module = clock
    original_reset = env_class.reset

    @functools.wraps(original_reset)
    def reset(env, *args, **kwargs):
        active["env"] = env
        return original_reset(env, *args, **kwargs)

    env_class.reset = reset
