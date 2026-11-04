"""沿用 SIMPLE 原生 agent/HTTP 客户端，只替换发往策略的测量值。"""

from functools import wraps

import numpy as np

from rhsci.inference.measurement_noise import MeasurementNoise, MeasurementNoiseConfig


def install(
    config: MeasurementNoiseConfig | None,
    *,
    episode_start: int = 0,
    agent_class=None,
    state_slices=None,
) -> None:
    """仅开启时包装原生入口；每个 agent 独立保存缓存与随机流。"""
    if config is None:
        return
    if type(episode_start) is not int or episode_start < 0:
        raise ValueError("State noise episode start must be a nonnegative integer")
    if agent_class is None:
        from simple.baselines.psi0_decoupled_wbc import STATE_SLICES, Psi0DecoupledWbcAgent

        agent_class, state_slices = Psi0DecoupledWbcAgent, STATE_SLICES
    if state_slices is None:
        raise ValueError("State noise requires the native SIMPLE measured-joint slices")
    identity = (config, episode_start)
    installed = getattr(agent_class, "_rhsci_state_noise_config", None)
    if installed is not None:
        if installed != identity:
            raise ValueError("Cannot change installed state noise configuration in one process")
        return
    original_init = agent_class.__init__
    original_reset = agent_class.reset
    original_action = agent_class.get_action

    @wraps(original_init)
    def init(agent, *args, **kwargs):
        original_init(agent, *args, **kwargs)
        agent._rhsci_measurement_noise = MeasurementNoise(config)
        agent._rhsci_noise_next_episode = episode_start
        original_query = agent.client.query_action

        @wraps(original_query)
        def query(image_dict, instruction, state_dict, condition_dict, *args, **kwargs):
            noisy_state = dict(state_dict)
            noisy_state["states"] = agent._rhsci_measurement_noise.apply_to_state(
                state_dict["states"]
            )
            return original_query(
                image_dict, instruction, noisy_state, condition_dict, *args, **kwargs
            )

        # 只包装该 agent 的客户端实例，其他 HTTP 客户端不受影响。
        agent.client.query_action = query

    @wraps(original_reset)
    def reset(agent, *args, **kwargs):
        result = original_reset(agent, *args, **kwargs)
        agent._rhsci_measurement_noise.reset(agent._rhsci_noise_next_episode)
        agent._rhsci_noise_next_episode += 1
        return result

    @wraps(original_action)
    def get_action(agent, observation, *args, **kwargs):
        qpos = observation["joint_qpos"]
        # 与原生 HTTP state 共用关节切片；腰部顺序仍为 roll/pitch/yaw。
        measured = np.concatenate(
            [qpos[start:end] for _, start, end in state_slices] + [qpos[[13, 14, 12]]]
        ).astype(np.float32)
        agent._rhsci_measurement_noise.update(measured, float(agent.robot.mjData.time))
        return original_action(agent, observation, *args, **kwargs)

    agent_class.__init__ = init
    agent_class.reset = reset
    agent_class.get_action = get_action
    agent_class._rhsci_state_noise_config = identity
