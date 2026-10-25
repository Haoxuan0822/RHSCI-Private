"""Raw-radian noise reaches π0.5 state tokens without changing action inputs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_pi05_rsci import TinyBackbone, fixture

from rsci.backbones.pi05.model import Pi05RSCI
from rsci.backbones.pi05.processing import Processor
from rsci.backbones.pi05.train import parser, resolve
from rsci.inference.state_noise import RequestStateNoise


class Tokenizer:
    def encode(self, text, **kwargs):
        self.prompt = text
        return list(range(len(text)))


class StateNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "meta").mkdir()
        (self.root / "meta/stats.json").write_text(
            json.dumps(
                {
                    key: dict(q01=[-bound] * size, q99=[bound] * size)
                    for key, size, bound in (("states", 32, 2.0), ("action", 36, 1.0))
                }
            )
        )

    def processor(self, scale=0.0):
        with patch("sentencepiece.SentencePieceProcessor", return_value=Tokenizer()):
            return Processor(self.root, "unused", "STATE", state_measurement_noise_std_rad=scale)

    def sample(self):
        return dict(
            text="move",
            state=np.linspace(-0.5, 0.5, 32, dtype=np.float32),
            actions=np.zeros((30, 36), np.float32),
            images=np.zeros((1, 4, 6, 3), np.uint8),
        )

    def test_training_noise_changes_only_measured_values_before_state_tokenization(self):
        processor = self.processor(0.2)
        value = self.sample()
        raw = value["state"].copy()
        clean = self.processor()(value)
        with patch("torch.randn_like", side_effect=lambda ref: torch.ones_like(ref)) as draw:
            actual = processor(value)
        self.assertEqual(tuple(draw.call_args.args[0].shape), (31,))
        noisy = raw.copy()
        noisy[:31] += 0.2
        bins = np.digitize(processor.normalize(noisy, "states"), np.linspace(-1, 1, 257)[:-1]) - 1
        self.assertEqual(
            processor.tokenizer.prompt, f"Task: move, State: {' '.join(map(str, bins))};\nAction: "
        )
        clean_bins = (
            np.digitize(processor.normalize(raw, "states"), np.linspace(-1, 1, 257)[:-1]) - 1
        )
        self.assertTrue(np.all(bins[:31] != clean_bins[:31]))
        self.assertEqual(bins[31], clean_bins[31])
        np.testing.assert_array_equal(value["state"], raw)
        for key in ("actions", "images"):
            torch.testing.assert_close(actual[key], clean[key], rtol=0, atol=0)
        self.assertAlmostEqual(actual["state_noise_std_rad"].item(), 0.2)

    def test_default_off_and_seeded_inference_preserve_action_rng(self):
        processor = self.processor()
        value = self.sample()
        rng = torch.get_rng_state().clone()
        clean = processor(value)
        self.assertNotIn("state_noise_std_rad", clean)
        first = processor(value, state_inference_measurement_noise_std_rad=0.2, state_noise_seed=9)
        second = processor(value, state_inference_measurement_noise_std_rad=0.2, state_noise_seed=9)
        torch.testing.assert_close(first["tokens"], second["tokens"], rtol=0, atol=0)
        # Token IDs here encode prompt length; compare the actual prompt for the noise effect.
        prompt = processor.tokenizer.prompt
        processor(value)
        self.assertNotEqual(processor.tokenizer.prompt, prompt)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_historical_quantile_preprocessing_is_preserved(self):
        with patch("sentencepiece.SentencePieceProcessor", return_value=Tokenizer()):
            processor = Processor(self.root, "unused", "STATE", normalization="simple-quantile-v1")
        state = np.full(32, 5.0, np.float32)
        expected = (state + 2) / (4 + 1e-6) * 2 - 1
        np.testing.assert_array_equal(processor.normalize(state, "states"), expected)
        self.assertTrue((self.processor().normalize(state, "states") == 1).all())

    def test_http_action_noise_is_independent_of_prior_global_rng(self):
        from psi.deploy.helpers import RequestMessage, ResponseMessage
        from rsci.backbones.pi05.serve import Server
        from rsci.inference.action_noise import PROTOCOL, RequestActionNoise, derive_seed

        # Golden seeds from the local evaluation protocol, independent of server RNG.
        for episode_seed, index, expected in (
            (0, 0, 7070836379803831727),
            (76, 0, 7346786445294623666),
            (76, 1, 8702962673278848276),
            (123, 42, 1975257683876589514),
        ):
            self.assertEqual(derive_seed(episode_seed, index), expected)

        model, data = fixture("STATE")
        model.eval()
        batch = {key: data[key][0] for key in ("tokens", "token_mask")}
        batch["images"] = data["images"][0, -1:]

        class FixedProcessor:
            stats = {"action": dict(q01=[-1.0] * 36, q99=[1.0] * 36)}

            def __call__(self, sample):
                return batch

        server = Server.__new__(Server)
        server.model, server.processor = model, FixedProcessor()
        server.device, server.variant = torch.device("cpu"), "STATE"
        request = RequestMessage(
            image={"rgb_head_stereo_left": np.zeros((4, 6, 3), np.uint8)},
            instruction="move",
            state={"states": np.zeros((1, 32), np.float32)},
            history={
                "psi_eval_deterministic": dict(
                    protocol=PROTOCOL, episode=3, episode_seed=76, inference_index=0
                )
            },
            condition={},
            gt_action=[],
            dataset_name="fixture",
            timestamp="0",
        )
        outputs = []
        for seed in (7, 900):
            torch.manual_seed(seed)
            server.action_noise = RequestActionNoise(True, 73)
            before = torch.get_rng_state().clone()
            response = server.predict_action(request.serialize())
            outputs.append(ResponseMessage.deserialize(json.loads(response.body)).action)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            with self.assertRaisesRegex(ValueError, "consecutive"):
                server.predict_action(request.serialize())
        np.testing.assert_array_equal(*outputs)
        reference = torch.Generator().manual_seed(derive_seed(76, 0))
        expected = torch.randn((1, 30, 36), generator=reference)
        noise = RequestActionNoise(True, 73).sample(request.serialize(), torch.device("cpu"))
        torch.testing.assert_close(noise, expected, rtol=0, atol=0)

    def test_model_rejects_training_without_matching_preprocessing_noise(self):
        _, data = fixture("STATE")
        data["images"] = data["images"][:, -1:]
        model = Pi05RSCI(
            "STATE",
            TinyBackbone(),
            vlm_width=64,
            ae_width=32,
            compile_inference=False,
            state_measurement_noise_std_rad=0.02,
        )
        with self.assertRaisesRegex(ValueError, "matching raw-state noise"):
            model(**data)
        data["state_noise_std_rad"] = torch.full((2,), 0.02)
        loss = model(**data)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.action_out_proj.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.rsci_config["state_measurement_noise_std_rad"], 0.02)

    def test_cli_rejects_invalid_noise_and_separates_output_paths(self):
        base = ["--task", "example", "--model", "STATE", "--dataset", "/datasets/example"]
        clean = resolve(parser().parse_args(base))
        noisy = resolve(parser().parse_args(base + ["--state-measurement-noise-std-rad"]))
        self.assertNotEqual(clean["output"], noisy["output"])
        for value in ("-0.1", "nan", "inf"):
            with self.assertRaises(ValueError):
                resolve(parser().parse_args(base + ["--state-measurement-noise-std-rad", value]))
        with self.assertRaises(ValueError):
            resolve(
                parser().parse_args(
                    [
                        "--task",
                        "example",
                        "--model",
                        "VGQ-AE",
                        "--state-measurement-noise-std-rad",
                    ]
                )
            )

    def test_request_seed_is_episode_indexed_and_rejects_non_state(self):
        source = RequestStateNoise(0.02, 13, "STATE")
        payload = dict(history=dict(psi_eval_deterministic=dict(episode=2, inference_index=7)))
        seed = source.request_seed(payload)
        self.assertEqual(seed, RequestStateNoise(0.02, 13, "STATE").request_seed(payload))
        payload["history"]["psi_eval_deterministic"]["episode"] = 3
        self.assertNotEqual(seed, source.request_seed(payload))
        with self.assertRaises(ValueError):
            RequestStateNoise(0.02, 13, "VGQ-AE")
        for seed in (-1, True, 2**63):
            with self.assertRaises(ValueError):
                RequestStateNoise(0.02, seed, "STATE")


if __name__ == "__main__":
    unittest.main()
