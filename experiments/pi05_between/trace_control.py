"""Opt-in tracing around the existing official policy and simulation methods."""

import functools
import json
import time


def install_control_trace(path):
    """仅记录真实动作、WBC 目标和仿真步，不替换控制器或修改时间。"""
    import numpy as np
    from PIL import Image
    from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("x", buffering=1)
    context = {"step": 0, "phase": "stabilize"}

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
        )
        return result

    Psi0DecoupledWbcAgent.__init__ = init
    Psi0DecoupledWbcAgent.get_action = get_action
    SonicLocoManipEnv.step = step
    emit("trace_installed", changes_control=False)
