"""CPU request/response coverage for the production PSI0 serving entry."""

import json
import time
import unittest
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace

import numpy as np
import torch

from psi.deploy.helpers import RequestMessage, convert_numpy_in_dict, numpy_deserialize
from psi.deploy.psi0_serve_simple import Server
from rsci.data.state_noise import InferenceStateNoise
from rsci.inference.state_noise import RequestStateNoise


class ServingTests(unittest.TestCase):
    def test_request_noise_is_independent_of_global_rng_and_previous_episode_length(self):
        from rsci.inference.action_noise import PROTOCOL, RTC_PROTOCOL, install_server

        def make_server(rtc):
            def sample(**kwargs):
                result = torch.randn(1, 3, 2) + torch.randn_like(torch.zeros(1, 3, 2))
                return result + kwargs.get("prev_actions", torch.zeros_like(result)) * 0.1

            server = Server.__new__(Server)
            server.model = SimpleNamespace(
                rsci_spec=SimpleNamespace(name="NO-STATE"),
                rsci_history_from_request=lambda *args: {},
                predict_action=sample,
                predict_action_with_training_rtc_flow=sample,
            )
            server.model_transform = SimpleNamespace(
                resize=lambda: lambda x: x, center_crop=lambda: lambda x: x
            )
            server.maxmin = SimpleNamespace(
                normalize_state=False, pad_state_dim=36, denormalize=lambda x: x
            )
            server.state_noise = RequestStateNoise()
            server.device, server.enable_rtc = torch.device("cpu"), rtc
            server.previous_action = None
            server.Tp, server.Ta, server.Da, server.rtc_max_delay = 3, 2, 2, 1
            server.count, server.last_serve_time = 0, time.monotonic()
            return install_server(server, enabled=True, base_seed=73, rtc=rtc)

        for rtc in (False, True):
            outputs = []
            for global_seed, length in ((4, 1), (900, 3)):
                server = make_server(rtc)
                torch.manual_seed(global_seed)
                observed = []
                for episode, count in ((0, length), (3, 2)):
                    for index in range(count):
                        torch.randn(100 * (index + 1))
                        before = torch.get_rng_state().clone()
                        request = RequestMessage(
                            image={"rgb_head_stereo_left": np.zeros((4, 4, 3), np.uint8)},
                            instruction="test",
                            state={},
                            condition={},
                            gt_action=[],
                            dataset_name="fixture",
                            timestamp="0",
                            history={
                                "psi_eval_deterministic": dict(
                                    protocol=RTC_PROTOCOL if rtc else PROTOCOL,
                                    episode=episode,
                                    episode_seed=73 + episode,
                                    inference_index=index,
                                )
                            },
                        )
                        response = json.loads(server.predict_action(request.serialize()).body)
                        action = convert_numpy_in_dict(response, numpy_deserialize)["action"]
                        self.assertTrue(torch.equal(before, torch.get_rng_state()))
                        if episode == 3:
                            observed.append(action)
                outputs.append(observed)
            np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_noise_cli_default_bare_and_explicit_values(self):
        import tyro
        from pydantic import ValidationError

        from psi.config.config import ServerConfig

        base = ["--run-dir", "/runs/STATE/task", "--ckpt-step", "100", "--policy", "psi0"]
        flag = "--state-inference-measurement-noise-std-rad"
        cfg = tyro.cli(
            ServerConfig, args=base + ["--deterministic-eval", "--eval-seed", "73", "--rtc"]
        )
        self.assertTrue(cfg.deterministic_eval)
        self.assertTrue(cfg.rtc)
        self.assertEqual(cfg.eval_seed, 73)
        for extra, expected in (
            ([], 0.0),
            ([flag], 0.02),
            ([flag, "--state-noise-seed", "73"], 0.02),
            ([flag, "0.03"], 0.03),
            ([f"{flag}=0.04"], 0.04),
        ):
            with self.subTest(extra=extra):
                cfg = tyro.cli(ServerConfig, args=base + extra)
                self.assertEqual(cfg.state_inference_measurement_noise_std_rad, expected)
                if "--state-noise-seed" in extra:
                    self.assertEqual(cfg.state_noise_seed, 73)
        for value in ("-0.01", "nan", "inf", "invalid"):
            with self.subTest(value=value), redirect_stderr(StringIO()):
                with self.assertRaises((ValidationError, SystemExit, ValueError)):
                    tyro.cli(ServerConfig, args=base + [flag, value])

    def test_action_conversion_normalization_and_rtc_history(self):
        calls = []

        class Policy:
            rsci_spec = SimpleNamespace(name="VGQ-VLM*")

            def rsci_history_from_request(self, condition, *args):
                self_history = condition["rsci_history_valid"]
                return {"history_valid": torch.from_numpy(self_history.copy())[None]}

            def predict_action(self, **kwargs):
                calls.append(("first", kwargs))
                return torch.ones(1, 3, 2)

            def predict_action_with_training_rtc_flow(self, **kwargs):
                calls.append(("rtc", kwargs))
                return torch.full((1, 3, 2), 2.0)

        server = Server.__new__(Server)
        server.state_noise = RequestStateNoise()
        server.model = Policy()
        server.model_transform = SimpleNamespace(
            resize=lambda: lambda x: x, center_crop=lambda: lambda x: x
        )
        server.maxmin = SimpleNamespace(
            normalize_state=False, pad_state_dim=36, denormalize=lambda x: x * 2
        )
        server.device = torch.device("cpu")
        server.enable_rtc = True
        server.previous_action = None
        server.Tp, server.Ta, server.Da, server.rtc_max_delay = 3, 2, 2, 1
        server.count, server.last_serve_time = 0, time.monotonic()
        request = RequestMessage(
            image={"rgb_head_stereo_left": np.zeros((4, 4, 3), np.uint8)},
            instruction="test",
            history={},
            state={"states": np.full((1, 36), 999, np.float32)},
            condition={"rsci_history_valid": np.array([False, False, False, True])},
            gt_action=[],
            dataset_name="simple",
            timestamp="0",
        )
        for expected in (2, 4):
            response = server.predict_action(request.serialize())
            decoded = convert_numpy_in_dict(json.loads(response.body), numpy_deserialize)
            np.testing.assert_array_equal(decoded["action"], np.full((2, 2), expected))
        self.assertEqual([name for name, _ in calls], ["first", "rtc"])
        for _, inputs in calls:
            self.assertFalse(inputs["states"].any())
            self.assertEqual(inputs["history_valid"].tolist(), [[False, False, False, True]])
        torch.testing.assert_close(
            calls[1][1]["prev_actions"], torch.tensor([[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
        )

    def test_state_noise_is_applied_to_raw_measurements_before_normalization(self):
        calls, raw_seen = [], []

        def normalize(value):
            raw_seen.append(value.copy())
            return value / 2

        def predict(**kwargs):
            calls.append(kwargs)
            return torch.zeros(1, 3, 2)

        server = Server.__new__(Server)
        server.model = SimpleNamespace(
            rsci_spec=SimpleNamespace(name="STATE"),
            rsci_history_from_request=lambda *args: {},
            predict_action=predict,
        )
        server.model_transform = SimpleNamespace(
            resize=lambda: lambda x: x, center_crop=lambda: lambda x: x
        )
        server.maxmin = SimpleNamespace(
            normalize_state=True,
            pad_state_dim=36,
            normalize_state_func=normalize,
            denormalize=lambda value: value,
        )
        server.state_noise = RequestStateNoise(0.02, 19, "STATE")
        server.inference_state_noise = InferenceStateNoise()
        server.device, server.enable_rtc = torch.device("cpu"), False
        server.Tp, server.Ta, server.Da, server.count = 3, 2, 2, 0
        raw = np.zeros((1, 36), np.float32)
        raw[0, 31] = 0.74
        request = RequestMessage(
            image={"rgb_head_stereo_left": np.zeros((4, 4, 3), np.uint8)},
            instruction="test",
            history={"psi_eval_deterministic": dict(episode=3, inference_index=2)},
            state={"states": raw},
            condition={},
            gt_action=[],
            dataset_name="simple",
            timestamp="0",
        )
        rng = torch.get_rng_state().clone()
        for _ in range(2):
            response = json.loads(server.predict_action(request.serialize()).body)
            self.assertIn("action", response)
        np.testing.assert_array_equal(raw_seen[0], raw_seen[1])
        self.assertTrue(np.all(raw_seen[0][0, :31] != 0))
        np.testing.assert_array_equal(raw_seen[0][0, 31:], raw[0, 31:])
        torch.testing.assert_close(calls[0]["states"][0], torch.from_numpy(raw_seen[0]) / 2)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
