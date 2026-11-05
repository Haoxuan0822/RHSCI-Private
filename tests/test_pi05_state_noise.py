"""π0.5 state preprocessing and independent native action sampling."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_pi05_rsci import fixture

from openpi.rhsci.config import build_argument_parser
from openpi.rhsci.processing import Processor


class Tokenizer:
    def encode(self, text, **kwargs):
        self.prompt = text
        return list(range(len(text)))


class StatePreprocessingTests(unittest.TestCase):
    def test_training_measurement_noise_option_is_removed(self):
        with self.assertRaises(SystemExit):
            build_argument_parser().parse_args(
                ["--task", "example", "--model", "STATE", "--state-measurement-noise-std-rad", "0.02"]
            )

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

    def processor(self):
        with patch("sentencepiece.SentencePieceProcessor", return_value=Tokenizer()):
            return Processor(self.root, "unused", "STATE")

    def sample(self):
        return dict(
            text="move",
            state=np.linspace(-0.5, 0.5, 32, dtype=np.float32),
            actions=np.zeros((30, 36), np.float32),
            images=np.zeros((1, 4, 6, 3), np.uint8),
        )

    def test_historical_quantile_preprocessing_is_preserved(self):
        with patch("sentencepiece.SentencePieceProcessor", return_value=Tokenizer()):
            processor = Processor(self.root, "unused", "STATE", normalization="simple-quantile-v1")
        state = np.full(32, 5.0, np.float32)
        expected = (state + 2) / (4 + 1e-6) * 2 - 1
        np.testing.assert_array_equal(processor.normalize(state, "states"), expected)
        self.assertTrue((self.processor().normalize(state, "states") == 1).all())

    def test_http_action_noise_is_independent_of_prior_global_rng(self):
        from openpi.rhsci.serve import Server
        from psi.deploy.helpers import RequestMessage, ResponseMessage
        from rhsci.inference.action_noise import PROTOCOL, RequestActionNoise, derive_seed

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
            self.assertEqual(outputs[-1].shape, (30, 36))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            with self.assertRaisesRegex(ValueError, "consecutive"):
                server.predict_action(request.serialize())
        np.testing.assert_array_equal(*outputs)
        reference = torch.Generator().manual_seed(derive_seed(76, 0))
        expected = torch.randn((1, 30, 36), generator=reference)
        noise = RequestActionNoise(True, 73).sample(request.serialize(), torch.device("cpu"))
        torch.testing.assert_close(noise, expected, rtol=0, atol=0)



if __name__ == "__main__":
    unittest.main()
