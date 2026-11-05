"""客户端测量噪声的单位、时间、随机流和原生请求边界。"""

import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rhsci.inference.measurement_noise import (
    MeasurementNoise,
    MeasurementNoiseConfig,
    add_arguments,
    command_arguments,
    config_from_arguments,
)
from rhsci.inference.state_noise_client import install


class MeasurementNoiseTests(unittest.TestCase):
    def noise(self, kind, level="low", seed=17, episode=0):
        noise = MeasurementNoise(MeasurementNoiseConfig(kind, level, seed))
        noise.reset(episode)
        return noise

    def test_switch_validation_and_worker_forwarding(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        self.assertIsNone(config_from_arguments(parser.parse_args([])))
        self.assertEqual(command_arguments(None), [])
        for argv in (
            ["--state-noise"],
            ["--state-noise-type", "gaussian"],
            ["--state-noise-level", "low"],
            ["--state-noise-seed", "0"],
            ["--state-noise", "--state-noise-type", "delay"],
            ["--state-noise", "--state-noise-level", "low"],
            [
                "--state-noise",
                "--state-noise-type",
                "delay",
                "--state-noise-level",
                "low",
                "--state-noise-seed",
                "-1",
            ],
        ):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                config_from_arguments(parser.parse_args(argv))
        for kind in ("gaussian", "zero-offset", "delay", "dropout"):
            for level in ("low", "medium", "high", "extreme"):
                config = MeasurementNoiseConfig(kind, level, 73)
                actual = config_from_arguments(parser.parse_args(command_arguments(config)))
                self.assertEqual(config, actual)

    def test_gaussian_all_levels_independent_joints_steps_and_global_rng(self):
        global_rng = np.random.get_state()
        for level, degrees in zip(("low", "medium", "high", "extreme"), (5, 15, 30, 60)):
            with self.subTest(level=level):
                noise = self.noise("gaussian", level)
                expected_rng = np.random.default_rng(np.random.SeedSequence([17, 0]))
                samples = []
                for step in range(3000):
                    noise.update(np.zeros(31, np.float32), step * 0.02)
                    expected = expected_rng.normal(0, np.deg2rad(degrees), 31).astype(np.float32)
                    np.testing.assert_array_equal(noise.measured, expected)
                    samples.append(noise.measured)
                samples = np.stack(samples)
                self.assertAlmostEqual(samples.std(), np.deg2rad(degrees), delta=0.01)
                self.assertLess(abs(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]), 0.05)
        after = np.random.get_state()
        self.assertEqual(global_rng[0], after[0])
        np.testing.assert_array_equal(global_rng[1], after[1])
        self.assertEqual(global_rng[2:], after[2:])

    def test_offset_is_per_joint_per_episode_and_repeatable(self):
        for level, degrees in zip(("low", "medium", "high", "extreme"), (5, 15, 30, 60)):
            noise = self.noise("zero-offset", level)
            offset = noise.offset.copy()
            self.assertTrue((abs(offset) <= np.deg2rad(degrees)).all())
            self.assertEqual(len(np.unique(offset)), 31)
            for step in range(3):
                noise.update(np.full(31, step, np.float64), step * 0.02)
                np.testing.assert_allclose(noise.measured - step, offset, atol=1e-15)
            noise.reset(1)
            self.assertFalse(np.array_equal(offset, noise.offset))
            noise.reset(0)
            np.testing.assert_array_equal(offset, noise.offset)

    def test_delay_all_levels_padding_and_bounded_history(self):
        for level, lag in zip(("low", "medium", "high", "extreme"), (5, 15, 30, 60)):
            noise = self.noise("delay", level)
            for step in range(501):
                # 非零起始时间，模拟站稳之后才开始采样。
                noise.update(np.full(31, step, np.float32), 3.2 + step * 0.02)
                np.testing.assert_array_equal(noise.measured, max(0, step - lag))
                self.assertLessEqual(len(noise.history), lag + 2)
            noise.reset(1)
            noise.update(np.full(31, -1, np.float32), 1.0)
            np.testing.assert_array_equal(noise.measured, -1)

    def test_dropout_all_levels_freeze_and_immediate_recovery(self):
        for level, length in zip(("low", "medium", "high", "extreme"), (10, 30, 60, 120)):
            noise = self.noise("dropout", level)
            for step in range(601):
                state = np.full((1, 32), step, np.float32)
                noise.update(state[0, :31], 3.2 + step * 0.02)
                cycle, phase = divmod(step, 200)
                expected = cycle * 200 - 1 if cycle >= 1 and phase < length else step
                np.testing.assert_array_equal(noise.apply_to_state(state)[0, :31], expected)
            noise.reset(1)
            noise.update(np.full(31, -1, np.float32), 0.0)
            np.testing.assert_array_equal(noise.apply_to_state(np.zeros((1, 32)))[0, :31], -1)

    def test_only_measured_state_is_replaced_without_clipping_or_input_mutation(self):
        state = np.arange(36, dtype=np.float32)[None]
        original = state.copy()
        noise = self.noise("zero-offset", "extreme")
        noise.update(state[0, :31], 0)
        output = noise.apply_to_state(state)
        self.assertEqual(output.dtype, state.dtype)
        self.assertEqual(output.shape, state.shape)
        np.testing.assert_array_equal(state, original)
        np.testing.assert_array_equal(output[:, 31:], state[:, 31:])
        np.testing.assert_array_equal(
            output[0, :31], (state[0, :31] + noise.offset).astype(np.float32)
        )
        output[:] = 0
        self.assertTrue(noise.measured.any())

    def test_invalid_measurements_and_clock_rejected(self):
        noise = self.noise("delay")
        for values in (np.zeros(30), np.zeros(31, int), np.full(31, np.nan)):
            with self.assertRaises(ValueError):
                noise.update(values, 0)
        noise.update(np.zeros(31), 1)
        for time in (1, 0.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                noise.update(np.zeros(31), time)


def client_fixture(horizon):
    """保留原生的 state 分组及按动作块请求方式，不引入物理环境。"""
    slices = [
        ("hand", 29, 32),
        ("middle", 34, 36),
        ("index", 32, 34),
        ("right_hand", 36, 43),
        ("left_arm", 15, 22),
        ("right_arm", 22, 29),
    ]

    class Client:
        def __init__(self):
            self.requests = []

        def query_action(self, images, instruction, state_dict, conditions, **kwargs):
            self.requests.append((images, instruction, state_dict, conditions, kwargs))
            return "native-action"

    class Agent:
        def __init__(self):
            self.client = Client()
            self.robot = SimpleNamespace(mjData=SimpleNamespace(time=0.0))

        def reset(self):
            self.step = 0

        def get_action(self, observation):
            self.last_observation = observation
            qpos = observation["joint_qpos"]
            state = np.concatenate(
                [qpos[start:end] for _, start, end in slices] + [qpos[[13, 14, 12]], [0.74]]
            ).astype(np.float32)[None]
            if self.step % horizon == 0:
                self.client.query_action(
                    observation["image"],
                    "task",
                    {"states": state},
                    {"unchanged": True},
                    history={"reset": self.step == 0},
                    dataset="simple",
                )
            self.step += 1
            return "native-action"

    return Agent, slices


class StateNoiseClientTests(unittest.TestCase):
    def test_dropout_hits_each_period_at_both_native_chunk_lengths(self):
        for horizon in (16, 30):
            for level, duration in zip(("low", "medium", "high", "extreme"), (.2, .6, 1.2, 2.4)):
                with self.subTest(horizon=horizon, level=level):
                    agent_class, slices = client_fixture(horizon)
                    install(MeasurementNoiseConfig("dropout", level), agent_class=agent_class, state_slices=slices)
                    agent = agent_class()
                    agent.reset()
                    last_success = None
                    cycle = 0
                    until = -1
                    hits = set()
                    for step in range(1201):
                        elapsed = step * .02
                        agent.robot.mjData.time = 3.2 + elapsed
                        qpos = np.arange(43, dtype=np.float32) + step
                        observation = {"joint_qpos": qpos, "image": "raw-image"}
                        agent.get_action(observation)
                        np.testing.assert_array_equal(qpos, np.arange(43) + step)
                        if step % horizon:
                            continue
                        current_cycle = step // 200
                        if current_cycle > cycle:
                            cycle, until = current_cycle, elapsed + duration
                        if elapsed < until - 1e-9:
                            expected = last_success
                            hits.add(cycle)
                        else:
                            last_success = expected = step
                        state = agent.client.requests[-1][2]["states"]
                        self.assertEqual(state[0, 0], 29 + expected)
                        self.assertAlmostEqual(state[0, 31], .74, places=6)
                        self.assertIs(agent.last_observation, observation)
                    self.assertEqual(hits, set(range(1, 7)))

    def test_real_cli_rejects_invalid_switches_before_loading_simulation(self):
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        # 即使 simple 激活脚本已清空 PYTHONPATH，CLI 也能解析并拒绝无效配置。
        environment.pop("PYTHONPATH", None)
        entries = (
            (
                "eval_decoupled_wbc.py",
                [
                    "simple/Task-v0",
                    "psi0_decoupled_wbc",
                    "--port",
                    "8120",
                    "--data-dir",
                    "unused",
                    "--eval-dir",
                    "unused",
                    "--renderer-gpu",
                    "0",
                    "--dds-domain-id",
                    "100",
                ],
            ),
            ("run_evaluation.py", ["--gpu", "0", "--variant", "STATE", "--level", "dr-level-0"]),
        )
        for entry, base in entries:
            for switches, expected in (
                (["--state-noise"], "requires both"),
                (["--state-noise-type", "delay"], "require --state-noise"),
            ):
                with self.subTest(entry=entry, switches=switches):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(root / "experiments/pi05_between" / entry),
                            *base,
                            *switches,
                        ],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(expected, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_disabled_preserves_methods_and_observations(self):
        agent_class, slices = client_fixture(16)
        before = (agent_class.__init__, agent_class.get_action, agent_class.reset)
        install(None, agent_class=agent_class, state_slices=slices)
        self.assertEqual(before, (agent_class.__init__, agent_class.get_action, agent_class.reset))
        agent = agent_class()
        agent.reset()
        self.assertFalse(hasattr(agent, "_rhsci_measurement_noise"))
        agent.get_action({"joint_qpos": np.arange(43), "image": "raw-image"})
        request = agent.client.requests[-1]
        np.testing.assert_array_equal(
            request[2]["states"][0, :14], [29, 30, 31, 34, 35, 32, 33, 36, 37, 38, 39, 40, 41, 42]
        )

    def test_control_step_delay_is_independent_of_http_chunk_length(self):
        for horizon in (16, 30):
            agent_class, slices = client_fixture(horizon)
            config = MeasurementNoiseConfig("delay", "medium")
            install(config, episode_start=4, agent_class=agent_class, state_slices=slices)
            original = agent_class.get_action
            install(config, episode_start=4, agent_class=agent_class, state_slices=slices)
            self.assertIs(original, agent_class.get_action)
            agent = agent_class()
            other_agent = agent_class()
            agent.reset()
            other_agent.reset()
            for step in range(91):
                agent.robot.mjData.time = 3.2 + step * 0.02
                qpos = np.arange(43, dtype=np.float32) + step
                observation = {"joint_qpos": qpos, "image": "raw-image"}
                self.assertEqual(agent.get_action(observation), "native-action")
                self.assertIs(agent.last_observation, observation)
                np.testing.assert_array_equal(qpos, np.arange(43) + step)
                if step % horizon == 0:
                    request = agent.client.requests[-1]
                    expected = max(0, step - 15)
                    self.assertEqual(request[2]["states"][0, 0], 29 + expected)
                    np.testing.assert_array_equal(
                        request[2]["states"][0, 28:31], np.array([13, 14, 12]) + expected
                    )
                    self.assertAlmostEqual(request[2]["states"][0, 31], 0.74, places=6)
                    self.assertEqual(request[:2], ("raw-image", "task"))
                    self.assertEqual(request[3], {"unchanged": True})
                    self.assertEqual(request[4]["dataset"], "simple")
            self.assertIsNone(other_agent._rhsci_measurement_noise.measured)
            agent.reset()
            agent.robot.mjData.time = 0
            agent.get_action({"joint_qpos": np.zeros(43, np.float32), "image": "reset-image"})
            self.assertEqual(agent.client.requests[-1][2]["states"][0, 0], 0)
            self.assertEqual(agent._rhsci_noise_next_episode, 6)


if __name__ == "__main__":
    unittest.main()
