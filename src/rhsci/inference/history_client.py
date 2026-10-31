"""复用既有逐控制步因果采样方法，为带星号RHSCI请求附加四帧RGB。"""

from collections import deque
from functools import wraps

import numpy as np

OFFSETS = (-6, -4, -2, 0)
PROTOCOL = "rsci-control-rgb-50hz-v1"


class CausalRgbHistory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=7)
        self.step = -1

    def append(self, frame):
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("RHSCI历史要求uint8 HWC RGB")
        if self.frames and frame.shape != self.frames[-1].shape:
            raise ValueError("同episode相机尺寸不能变化")
        # Isaac可能复用同一buffer，须在下一控制步前复制；不额外触发渲染。
        self.frames.append(frame.copy())
        self.step += 1

    def condition(self):
        if not self.frames:
            raise RuntimeError("RGB历史为空")
        frames = list(self.frames)
        current = len(frames) - 1
        return dict(
            rsci_rgb_frames=np.stack([frames[max(0, current + t)] for t in OFFSETS]),
            rsci_history_valid=np.array([self.step + t >= 0 for t in OFFSETS], dtype=bool),
            rsci_frame_offsets=np.array(OFFSETS, dtype=np.int64),
            rsci_history_protocol=PROTOCOL,
            rsci_history_fps=50,
        )


def install(agent_class=None, client_class=None):
    """进程内包装实际客户端；spawn入口重复安装幂等，不修改SIMPLE源码。"""
    if agent_class is None:
        from simple.baselines.client import HttpActionClient
        from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent

        agent_class, client_class = Psi0DecoupledWbcAgent, HttpActionClient
    if getattr(agent_class, "_rsci_history_installed", False):
        return
    action, reset, query = agent_class.get_action, agent_class.reset, client_class.query_action

    @wraps(action)
    def with_frame(self, observation, *args, **kwargs):
        if not np.isclose(self._control_dt, 1 / 50, rtol=0, atol=1e-9):
            raise ValueError("RHSCI历史要求50Hz控制步，与训练帧时钟一致")
        if not hasattr(self, "_rsci_rgb_history"):
            self._rsci_rgb_history = CausalRgbHistory()
        self._rsci_rgb_history.append(observation["head_stereo_left"])
        self.client._rsci_rgb_history = self._rsci_rgb_history
        return action(self, observation, *args, **kwargs)

    @wraps(reset)
    def reset_history(self, *args, **kwargs):
        result = reset(self, *args, **kwargs)
        self._rsci_rgb_history = CausalRgbHistory()
        self.client._rsci_rgb_history = self._rsci_rgb_history
        return result

    @wraps(query)
    def with_history(self, image_dict, instruction, state_dict, condition_dict, *args, **kwargs):
        history = getattr(self, "_rsci_rgb_history", None)
        if history is None:
            raise RuntimeError("动作请求前必须记录当前控制步RGB")
        condition = dict(condition_dict or {})
        condition.update(history.condition())
        if not np.array_equal(condition["rsci_rgb_frames"][-1], image_dict["rgb_head_stereo_left"]):
            raise ValueError("历史当前帧与实际请求图像不一致")
        return query(self, image_dict, instruction, state_dict, condition, *args, **kwargs)

    agent_class.get_action, agent_class.reset = with_frame, reset_history
    client_class.query_action = with_history
    agent_class._rsci_history_installed = True
