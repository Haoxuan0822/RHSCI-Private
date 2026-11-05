"""CPU request/response coverage for the production PSI0 serving entry."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from psi.deploy.helpers import RequestMessage, convert_numpy_in_dict, numpy_deserialize
from psi.deploy.psi0_serve_simple import Server


class ServingTests(unittest.TestCase):
    def test_native_psi_server_rejects_wrong_method_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "checkpoints/ckpt_20").mkdir(parents=True)
            (run / "run_config.json").write_text("{}")
            config = SimpleNamespace(
                model=SimpleNamespace(rsci_variant="VGQ-AE-Mo"),
                data=SimpleNamespace(rsci_ardy_offset_frames=0),
            )
            with patch("psi.rhsci.config.load_run_config", return_value=config), patch(
                "psi.models.psi0.Psi0Model.from_pretrained"
            ) as load, self.assertRaisesRegex(ValueError, "differs from checkpoint"):
                Server("psi0", run, 20, device="cpu", expected_variant="VGQ-AE")
            load.assert_not_called()

    def test_native_cli_accepts_mo_evaluation_identity(self):
        import tyro
        from psi.config.config import ServerConfig

        config = tyro.cli(ServerConfig, args=[
            "--run-dir", "/runs/mo", "--ckpt-step", "20", "--policy", "psi0",
            "--expected-variant", "VGQ-AE-Mo",
        ])
        self.assertEqual(config.expected_variant, "VGQ-AE-Mo")

    def test_server_config_has_no_measurement_noise_options(self):
        from psi.config.config import ServerConfig

        self.assertNotIn("state_inference_measurement_noise_std_rad", ServerConfig.model_fields)
        self.assertNotIn("state_noise_seed", ServerConfig.model_fields)

    def test_request_noise_is_independent_of_global_rng_and_previous_episode_length(self):
        from rhsci.inference.action_noise import PROTOCOL, RTC_PROTOCOL, install_server

        def make_server(rtc):
            def sample(**kwargs):
                result = torch.randn(1, 3, 2) + torch.randn_like(torch.zeros(1, 3, 2))
                return result + kwargs.get("prev_actions", torch.zeros_like(result)) * 0.1

            server = Server.__new__(Server)
            server.model = SimpleNamespace(
                rsci_spec=SimpleNamespace(name="NO-STATE"),
                rhsci_history_from_request=lambda *args: {},
                predict_action=sample,
                predict_action_with_training_rtc_flow=sample,
            )
            server.model_transform = SimpleNamespace(
                resize=lambda: lambda x: x, center_crop=lambda: lambda x: x
            )
            server.maxmin = SimpleNamespace(
                normalize_state=False, pad_state_dim=36, denormalize=lambda x: x
            )
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

    def test_action_conversion_normalization_and_rtc_history(self):
        calls = []

        class Policy:
            rsci_spec = SimpleNamespace(name="VGQ-VLM*")

            def rhsci_history_from_request(self, condition, *args):
                self_history = condition["rsci_history_valid"]
                return {"history_valid": torch.from_numpy(self_history.copy())[None]}

            def predict_action(self, **kwargs):
                calls.append(("first", kwargs))
                return torch.ones(1, 3, 2)

            def predict_action_with_training_rtc_flow(self, **kwargs):
                calls.append(("rtc", kwargs))
                return torch.full((1, 3, 2), 2.0)

        server = Server.__new__(Server)
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
