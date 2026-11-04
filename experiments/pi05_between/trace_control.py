"""Opt-in tracing around the existing official policy and simulation methods."""

import functools
import json
import time


def install_control_trace(path, *, control_clock="wall"):
    """默认只记录；显式 simulation 仅在独立诊断中替换 WBC 时间来源。"""
    import numpy as np
    from PIL import Image
    from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("x", buffering=1)
    context = {"step": 0, "phase": "stabilize"}
    if control_clock == "simulation":
        install_simulation_clock(SonicLocoManipEnv)

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

        def query(*args, **kwargs):
            images = args[0]
            image_path = path.parent / f"observation-step-{context['step']:04d}.png"
            Image.fromarray(images["rgb_head_stereo_left"]).save(image_path)
            start = time.monotonic()
            result = original_query(*args, **kwargs)
            emit(
                "model_chunk",
                duration=time.monotonic() - start,
                actions=result[0],
                observed_state=args[2],
                instruction=args[1],
                image_path=str(image_path),
            )
            return result

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
        emit(
            "env_step",
            duration=time.monotonic() - start,
            sim_before=before,
            sim_after=float(env.mujoco.mjData.time),
            qpos=env.mujoco.mjData.qpos.copy(),
            observed_joints=result[0]["joint_qpos"],
            objects={
                name: np.concatenate((body.xpos, body.xquat))
                for name, body in env.mujoco.mj_objects.items()
            },
        )
        return result

    Psi0DecoupledWbcAgent.__init__ = init
    Psi0DecoupledWbcAgent.get_action = get_action
    SonicLocoManipEnv.step = step
    emit("trace_installed", changes_control=control_clock != "wall", control_clock=control_clock)


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
