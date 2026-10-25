"""Checkpoint admission and SIMPLE HTTP inference without full model weights."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsci.inference.checkpoints import discover, inspect_run


class CheckpointTests(unittest.TestCase):
    def make_run(self, root):
        run = Path(root) / "run"
        run.mkdir()
        source = "src/rsci/backbones/pi05/processing.py"
        contract = dict(
            model="VGQ-AE",
            world_size=1,
            horizon=30,
            code={source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()},
        )
        (run / "launch_contract.json").write_text(json.dumps(contract))
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        for step in (200, 700):
            checkpoint = run / f"checkpoint-{step}"
            checkpoint.mkdir()
            names = (
                "rsci_config.json",
                "optimizer.pt",
                "scheduler.pt",
                "training_args.bin",
                "rng_state.pth",
                "model.safetensors",
                "trainer_state.json",
            )
            for name in names:
                (checkpoint / name).write_text("fixture")
            (checkpoint / "trainer_state.json").write_text(json.dumps(dict(global_step=step)))
            files = {
                name: dict(
                    size=(checkpoint / name).stat().st_size,
                    sha256=hashlib.sha256((checkpoint / name).read_bytes()).hexdigest(),
                )
                for name in names
            }
            (checkpoint / "RSCI_COMPLETE.json").write_text(
                json.dumps(
                    dict(
                        schema=2,
                        numerical_validation="model-and-optimizer-finite-v1",
                        step=step,
                        world_size=1,
                        contract=digest,
                        files=files,
                    )
                )
            )
        return run

    def test_latest_complete_and_explicit_checkpoint_without_fixed_final_step(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.make_run(folder)
            (run / "checkpoint-800").mkdir()
            self.assertEqual(inspect_run(run, "pi05")[1].name, "checkpoint-700")
            self.assertEqual(inspect_run(run, "pi05", step=200)[1].name, "checkpoint-200")
            (run / "checkpoint-700/model.safetensors").write_text("truncated")
            self.assertEqual(inspect_run(run, "pi05")[1].name, "checkpoint-200")
            with self.assertRaisesRegex(ValueError, "No complete"):
                inspect_run(run, "pi05", step=700)
            rows = discover(folder, "pi05")
            self.assertEqual(len(rows["ready"]), 1)
            self.assertFalse(rows["rejected"])

    def test_rejects_changed_sources_and_checkpoint_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.make_run(folder)
            contract = json.loads((run / "launch_contract.json").read_text())
            contract["code"] = {"../outside": "invalid"}
            (run / "launch_contract.json").write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                inspect_run(run, "pi05")
            self.assertEqual(len(discover(folder, "pi05")["rejected"]), 1)
        with tempfile.TemporaryDirectory() as folder:
            run = self.make_run(folder)
            checkpoint = run / "checkpoint-700"
            checkpoint.rename(run / "hidden")
            checkpoint.symlink_to(run / "hidden", target_is_directory=True)
            self.assertEqual(inspect_run(run, "pi05")[1].name, "checkpoint-200")

    def test_historical_sources_are_verified_in_explicit_training_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.make_run(folder)
            training = Path(folder) / "training"
            training.mkdir()
            script = training / "Pi05_RSCI.sh"
            script.write_text("historical launcher")
            contract = json.loads((run / "launch_contract.json").read_text())
            contract["code"] = {script.name: hashlib.sha256(script.read_bytes()).hexdigest()}
            (run / "launch_contract.json").write_text(json.dumps(contract))
            digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            for path in run.glob("checkpoint-*/RSCI_COMPLETE.json"):
                marker = json.loads(path.read_text())
                marker["contract"] = digest
                path.write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                inspect_run(run, "pi05")
            self.assertEqual(
                inspect_run(run, "pi05", training_source_root=training)[1].name, "checkpoint-700"
            )
            for path in run.glob("checkpoint-*/RSCI_COMPLETE.json"):
                marker = json.loads(path.read_text())
                marker["schema"] = 1
                marker.pop("numerical_validation")
                path.write_text(json.dumps(marker))
            self.assertEqual(
                inspect_run(run, "pi05", training_source_root=training)[1].name, "checkpoint-700"
            )
            (run / "checkpoint-700/model.safetensors").write_text("truncated")
            self.assertEqual(
                inspect_run(run, "pi05", training_source_root=training)[1].name, "checkpoint-200"
            )
            script.write_text("changed")
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                inspect_run(run, "pi05", training_source_root=training)


class NativeActionNoiseTests(unittest.TestCase):
    def test_failed_native_response_does_not_advance_request_index(self):
        import torch
        from fastapi.responses import JSONResponse

        from rsci.inference.action_noise import PROTOCOL, install_server

        failed = [True]

        def predict(payload):
            value = torch.randn(1, 16, 36)
            if failed[0]:
                return JSONResponse({"status": "failed"}, status_code=400)
            return JSONResponse({"action": value.tolist()})

        server = install_server(SimpleNamespace(predict_action=predict), enabled=True, base_seed=73)
        payload = {
            "history": {
                "psi_eval_deterministic": dict(
                    protocol=PROTOCOL, episode=0, episode_seed=73, inference_index=0
                )
            }
        }
        with self.assertRaisesRegex(ValueError, "not advanced"):
            server.predict_action(payload)
        failed[0] = False
        self.assertEqual(server.predict_action(payload).status_code, 200)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            server.predict_action(payload)

    def test_native_action_stream_is_independent_of_explicit_state_noise_generator(self):
        import torch

        from rsci.data.state_noise import InferenceStateNoise
        from rsci.inference.action_noise import noise_context

        values = []
        global_before = torch.get_rng_state().clone()
        for state_enabled in (False, True):
            traces = []
            with noise_context(7346786445294623666, traces):
                if state_enabled:
                    InferenceStateNoise().sample(torch.zeros(31), seed=73)
                values.append(torch.randn(size=(1, 16, 128), dtype=torch.bfloat16))
            self.assertEqual(len(traces), 1)
        torch.testing.assert_close(*values, rtol=0, atol=0)
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))

    def test_native_response_without_captured_action_noise_is_rejected(self):
        from fastapi.responses import JSONResponse

        from rsci.inference.action_noise import PROTOCOL, install_server

        server = install_server(
            SimpleNamespace(predict_action=lambda payload: JSONResponse({"action": [[0.0]]})),
            enabled=True,
        )
        with self.assertRaisesRegex(ValueError, "No native action noise"):
            server.predict_action(
                {
                    "history": {
                        "psi_eval_deterministic": dict(
                            protocol=PROTOCOL, episode=0, episode_seed=0, inference_index=0
                        )
                    }
                }
            )


class Pi05HTTPTests(unittest.TestCase):
    def test_pi05_loads_all_saved_shards_strictly(self):
        import torch
        from safetensors.torch import save_file
        from torch import nn

        from rsci.backbones.pi05.serve import Server

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tokenizer = root / "tokenizer.model"
            tokenizer.write_text("fixture")
            contract = dict(
                model="VGQ-AE",
                dataset="fixture",
                tokenizer=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            )
            (root / "rsci_config.json").write_text(
                json.dumps(
                    dict(
                        variant="VGQ-AE",
                        horizon=30,
                        action_dim=36,
                        vlm="frozen",
                        normalization="simple-quantile-zero-span-clipped-v2",
                    )
                )
            )
            expected = dict(weight=torch.full((2, 2), 7.0), bias=torch.full((2,), 8.0))
            save_file({"weight": expected["weight"]}, root / "first.safetensors")
            save_file({"bias": expected["bias"]}, root / "second.safetensors")
            index = dict(weight_map={"weight": "first.safetensors", "bias": "second.safetensors"})
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            with (
                patch(
                    "rsci.backbones.pi05.model.Pi05RSCI",
                    side_effect=lambda *a, **k: nn.Linear(2, 2),
                ),
                patch("rsci.backbones.pi05.processing.Processor"),
                patch("rsci.data.ardy_sidecar.dataset_content_fingerprint", return_value="fixture"),
            ):
                server = Server(root, contract, root, tokenizer, "cpu")
                self.assertFalse(server.model.training)
                for name, value in expected.items():
                    torch.testing.assert_close(server.model.state_dict()[name], value)
                save_file({"unexpected": expected["bias"]}, root / "second.safetensors")
                with self.assertRaises(RuntimeError):
                    Server(root, contract, root, tokenizer, "cpu")

    def test_current_frame_protocol_and_all_action_dimensions_across_requests(self):
        self.check_current_frame_protocol("VGQ-AE", 0.0)

    def test_state_noise_reaches_processor_through_http(self):
        self.check_current_frame_protocol("STATE", 0.02)

    def check_current_frame_protocol(self, variant, scale):
        import numpy as np
        import torch
        from fastapi.testclient import TestClient

        from psi.deploy.helpers import RequestMessage, ResponseMessage
        from rsci.backbones.pi05.processing import Processor
        from rsci.backbones.pi05.serve import Server
        from rsci.data.state_noise import InferenceStateNoise
        from rsci.inference.http_server import create_app
        from rsci.inference.state_noise import RequestStateNoise

        model = SimpleNamespace(calls=0)

        def sample_actions(images, tokens, mask, num_steps):
            self.assertEqual(tuple(images.shape), (1, 1, 3, 224, 224))
            self.assertEqual(num_steps, 10)
            model.calls += 1
            return torch.full((1, 30, 36), model.calls / 10)

        model.sample_actions = sample_actions
        processor = Processor.__new__(Processor)
        processor.variant = variant
        processor.state_measurement_noise_std_rad = 0.0
        processor.inference_state_noise = InferenceStateNoise()
        processor.stats = {
            "action": dict(q01=[-1.0] * 36, q99=[1.0] * 36),
            "states": dict(q01=[-2.0] * 32, q99=[2.0] * 32),
        }
        prompts = []

        def encode(prompt, **kwargs):
            prompts.append(prompt)
            return [1, 2]

        processor.tokenizer = SimpleNamespace(encode=encode)
        server = Server.__new__(Server)
        server.model, server.processor = model, processor
        server.device, server.variant = torch.device("cpu"), variant
        server.state_noise = RequestStateNoise(scale, 73, variant)
        state = np.zeros((1, 32), np.float32)
        state[0, 31] = 0.74
        request = RequestMessage(
            image={"rgb_head_stereo_left": np.zeros((16, 16, 3), np.uint8)},
            instruction="pick",
            history={"psi_eval_deterministic": {"episode": 2, "inference_index": 7}},
            state={"states": state},
            condition={},
            gt_action=[],
            dataset_name="fixture",
            timestamp="0",
        )
        with TestClient(create_app(server)) as client:
            rng_before = torch.get_rng_state().clone()
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            previous = None
            for _ in range(2):
                response = client.post("/act", json=request.serialize())
                self.assertEqual(response.status_code, 200, response.text)
                action = ResponseMessage.deserialize(response.json()).action
                self.assertEqual(action.shape, (24, 36))
                np.testing.assert_allclose(action, model.calls / 10, atol=1e-6)
                if previous is not None:
                    self.assertTrue(np.all(action[:, -4:] != previous[:, -4:]))
                previous = action
            if scale > 0:
                self.assertEqual(prompts[0], prompts[1])
                bins = np.asarray(prompts[0].split("State: ")[1].split(";")[0].split(), int)
                clean = (
                    np.digitize(
                        processor.normalize(state[0], "states"), np.linspace(-1, 1, 257)[:-1]
                    )
                    - 1
                )
                self.assertTrue(np.any(bins[:31] != clean[:31]))
                self.assertEqual(bins[31], clean[31])
                np.testing.assert_array_equal(request.state["states"], state)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            request.image = {}
            self.assertEqual(client.post("/act", json=request.serialize()).status_code, 422)

    def test_gr00t_uses_saved_checkpoint_with_current_frame_and_native_horizon(self):
        from rsci.backbones.gr00t.serve import build_server

        modality = {"video": SimpleNamespace(delta_indices=[-6, -4, -2, 0])}
        policy = SimpleNamespace(
            modality_configs=modality,
            processor=SimpleNamespace(training=False),
            model=SimpleNamespace(training=False),
        )
        server = SimpleNamespace(modality_configs=modality, policy=SimpleNamespace(policy=policy))
        configurations = []

        def native_server(cfg):
            configurations.append(cfg)
            return server

        module = SimpleNamespace(Server=native_server, ServerConfig=SimpleNamespace)
        data = SimpleNamespace(TAG="fixture")
        with patch.dict(
            sys.modules,
            {
                "gr00t.deploy.gr00t_serve_simple": module,
                "rsci.backbones.gr00t.data": data,
            },
        ):
            self.assertIs(build_server(Path("/models/checkpoint-700"), "cpu"), server)
            self.assertEqual(configurations[0].model_path, "/models/checkpoint-700")
            self.assertEqual(configurations[0].action_exec_horizon, 16)
            self.assertEqual(modality["video"].delta_indices, [0])
            policy.model.training = True
            with self.assertRaisesRegex(ValueError, "eval mode"):
                build_server(Path("/models/checkpoint-700"), "cpu")


if __name__ == "__main__":
    unittest.main()
