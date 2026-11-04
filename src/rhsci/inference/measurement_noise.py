"""评测测量扰动：只处理原始弧度关节值，不依赖模型或仿真引擎。"""

import argparse
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from rhsci.data.state_noise import MEASURED_STATE_DIM, validate_seed

NOISE_TYPES = ("gaussian", "zero-offset", "delay", "dropout")
NOISE_LEVELS = ("low", "medium", "high", "extreme")
ANGLE_DEGREES = (5.0, 15.0, 30.0, 60.0)
DELAY_SECONDS = (0.1, 0.3, 0.6, 1.2)
DROPOUT_SECONDS = (0.2, 0.6, 1.2, 2.4)
DROPOUT_PERIOD_SECONDS = 4.0
TIME_TOLERANCE = 1e-9


@dataclass(frozen=True)
class MeasurementNoiseConfig:
    kind: str
    level: str
    seed: int = 0

    def __post_init__(self):
        if self.kind not in NOISE_TYPES or self.level not in NOISE_LEVELS:
            raise ValueError("State noise requires a supported type and level")
        validate_seed(self.seed)

    @property
    def magnitude(self) -> float:
        index = NOISE_LEVELS.index(self.level)
        if self.kind in ("gaussian", "zero-offset"):
            return math.radians(ANGLE_DEGREES[index])
        if self.kind == "delay":
            return DELAY_SECONDS[index]
        return DROPOUT_SECONDS[index]

    def metadata(self) -> dict:
        return {
            "enabled": True,
            "type": self.kind,
            "level": self.level,
            "seed": self.seed,
            "magnitude": self.magnitude,
            "unit": "rad" if self.kind in ("gaussian", "zero-offset") else "s",
            "measured_slice": [0, MEASURED_STATE_DIM],
            "clock": "simulation-time-since-first-policy-observation",
            "dropout_period_seconds": DROPOUT_PERIOD_SECONDS if self.kind == "dropout" else None,
            "protocol": "client-measurement-noise-v1",
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-noise", action="store_true", help="Enable client measurement noise"
    )
    parser.add_argument("--state-noise-type", choices=NOISE_TYPES)
    parser.add_argument("--state-noise-level", choices=NOISE_LEVELS)
    parser.add_argument("--state-noise-seed", type=int, help="Independent noise seed; default 0")


def config_from_arguments(args: argparse.Namespace) -> MeasurementNoiseConfig | None:
    """关闭时返回 None；类型、等级和种子都只能在显式开启时提供。"""
    if not args.state_noise:
        if any(
            value is not None
            for value in (args.state_noise_type, args.state_noise_level, args.state_noise_seed)
        ):
            raise ValueError("State noise options require --state-noise")
        return None
    if args.state_noise_type is None or args.state_noise_level is None:
        raise ValueError("--state-noise requires both --state-noise-type and --state-noise-level")
    return MeasurementNoiseConfig(
        args.state_noise_type,
        args.state_noise_level,
        0 if args.state_noise_seed is None else args.state_noise_seed,
    )


def command_arguments(config: MeasurementNoiseConfig | None) -> list[str]:
    """worker 只转发已验证的配置，关闭时不向原客户端命令添加参数。"""
    if config is None:
        return []
    return [
        "--state-noise",
        "--state-noise-type",
        config.kind,
        "--state-noise-level",
        config.level,
        "--state-noise-seed",
        str(config.seed),
    ]


class MeasurementNoise:
    def __init__(self, config: MeasurementNoiseConfig):
        self.config = config
        self.origin = None
        self.last_time = None
        self.history = deque()
        self.measured = None
        self.offset = None
        self.generator = None

    def reset(self, episode: int) -> None:
        """每轮清空缓存，并用独立的 episode 随机流重新采样零位偏移。"""
        if type(episode) is not int or episode < 0:
            raise ValueError("State noise episode must be a nonnegative integer")
        self.origin = self.last_time = self.measured = None
        self.history.clear()
        self.generator = np.random.default_rng(np.random.SeedSequence([self.config.seed, episode]))
        self.offset = None
        if self.config.kind == "zero-offset":
            bound = self.config.magnitude
            self.offset = self.generator.uniform(-bound, bound, MEASURED_STATE_DIM)

    def update(self, measured: np.ndarray, simulation_time: float) -> None:
        """每个策略控制步记录一次测量；动作块内没有 HTTP 请求时也必须调用。"""
        measured = np.asarray(measured)
        if (
            measured.shape != (MEASURED_STATE_DIM,)
            or not np.issubdtype(measured.dtype, np.floating)
            or not np.isfinite(measured).all()
        ):
            raise ValueError("Expected 31 finite floating-point measured joints in radians")
        if self.generator is None:
            raise RuntimeError("Reset state noise before recording an episode")
        if not math.isfinite(simulation_time) or simulation_time < 0:
            raise ValueError("Measurement simulation time must be finite and nonnegative")
        if self.last_time is not None and simulation_time <= self.last_time:
            raise ValueError("Measurement simulation time must increase; reset between episodes")
        if self.origin is None:
            self.origin = simulation_time
        elapsed = simulation_time - self.origin
        self.last_time = simulation_time
        kind = self.config.kind
        magnitude = self.config.magnitude

        if kind == "gaussian":
            self.measured = (
                measured + self.generator.normal(0.0, magnitude, MEASURED_STATE_DIM)
            ).astype(measured.dtype)
        elif kind == "zero-offset":
            self.measured = (measured + self.offset).astype(measured.dtype)
        elif kind == "delay":
            self.history.append((elapsed, measured.copy()))
            target = elapsed - magnitude
            # 保留目标时间之前的最新帧；历史不足时使用首帧。缓存有界。
            while len(self.history) > 1 and self.history[1][0] <= target + TIME_TOLERANCE:
                self.history.popleft()
            self.measured = self.history[0][1]
        else:
            # 第 4/8/12... 秒开始中断；恢复边界当步立即接收最新测量。
            cycle = math.floor((elapsed + TIME_TOLERANCE) / DROPOUT_PERIOD_SECONDS)
            phase = elapsed - cycle * DROPOUT_PERIOD_SECONDS
            dropping = cycle >= 1 and phase < magnitude - TIME_TOLERANCE
            if not dropping:
                self.measured = measured.copy()

    def apply_to_state(self, state: np.ndarray) -> np.ndarray:
        """仅复制并替换 HTTP state 前31维，保留高度、尾部及真实观测。"""
        state = np.asarray(state)
        if (
            state.ndim != 2
            or state.shape[0] != 1
            or state.shape[1] < 32
            or not np.issubdtype(state.dtype, np.floating)
            or not np.isfinite(state).all()
        ):
            raise ValueError("Expected finite floating-point SIMPLE state with shape (1, >=32)")
        if self.measured is None:
            raise RuntimeError(
                "Record the current control-step measurement before querying actions"
            )
        result = state.copy()
        result[0, :MEASURED_STATE_DIM] = self.measured
        return result
