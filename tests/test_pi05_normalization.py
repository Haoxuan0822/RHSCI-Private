"""稀有动作/STATE 不被退化分位数抹除，部署使用断点保存的相同范围。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from openpi.rhsci.normalization import NORMALIZATION_VERSION, build_normalization_ranges
from openpi.rhsci.processing import Processor
from rhsci.runtime.contracts import reuse_run_contract


def statistics():
    return {
        key: dict(
            q01=[0.74, 0, -1, 1],
            q99=[0.74, 0, 1, 1.00001],
            min=[0.42, 0, -2, 0.8],
            max=[0.74, 0, 2, 1.2],
        )
        for key in ("action", "states")
    }


class NormalizationTests(unittest.TestCase):
    def processor(self):
        processor = Processor.__new__(Processor)
        processor.normalization = NORMALIZATION_VERSION
        processor.stats = statistics()
        processor.normalization_ranges = build_normalization_ranges(processor.stats)
        return processor

    def test_rare_height_and_near_equal_quantiles_roundtrip(self):
        processor = self.processor()
        before = copy.deepcopy(processor.stats)
        raw = np.array([[0.42, 0, -1, 0.8], [0.74, 0, 1, 1.2]], np.float32)
        for key in ("action", "states"):
            normalized = processor.normalize(raw, key)
            np.testing.assert_array_equal(normalized, [[-1, 0, -1, -1], [1, 0, 1, 1]])
            np.testing.assert_allclose(processor.denormalize(normalized, key), raw, atol=1e-7)
            # 真正恒定的维度始终还原常量，正常分位数维度仍保持裁剪。
            self.assertEqual(processor.denormalize(np.ones(4), key)[1], 0)
            self.assertEqual(processor.normalize([0.74, 0, 2, 1.2], key)[2], 1)
        self.assertEqual(processor.stats, before)

    def test_missing_or_invalid_extrema_cannot_silently_erase_variation(self):
        for field, value in (("min", None), ("max", [0.5, 0, 2, 1.2])):
            stats = statistics()
            if value is None:
                del stats["action"][field]
            else:
                stats["action"][field] = value
            with self.assertRaises(ValueError):
                build_normalization_ranges(stats)

    def test_saved_ranges_are_used_and_training_change_needs_new_run(self):
        ranges = self.processor().normalization_ranges
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            stats = statistics()
            stats["action"]["min"][0] = 0.1
            (root / "meta/stats.json").write_text(json.dumps(stats))
            with patch("sentencepiece.SentencePieceProcessor"):
                processor = Processor(root, "unused", "STATE", normalization_ranges=ranges)
            self.assertEqual(processor.normalization, NORMALIZATION_VERSION)
            self.assertAlmostEqual(processor.denormalize([-1, 0, 0, 0])[0], 0.42, places=6)
            saved = {"normalization": "simple-quantile-zero-span-clipped-v2"}
            (root / "launch_contract.json").write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "new run-tag"):
                reuse_run_contract(
                    {"normalization": NORMALIZATION_VERSION, "normalization_ranges": ranges}, root
                )

    def test_training_contract_records_ranges_and_can_explicitly_resume_legacy(self):
        from openpi.rhsci.config import build_argument_parser
        from openpi.rhsci.contract import build_training_contract
        from openpi.rhsci.normalization import LEGACY_NORMALIZATION

        args = build_argument_parser().parse_args(["--task", "custom", "--model", "STATE"])
        self.assertEqual(args.normalization, NORMALIZATION_VERSION)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/stats.json").write_text(json.dumps(statistics()))
            tokenizer = root / "tokenizer"
            tokenizer.write_text("test")
            paths = {
                "dataset": root,
                "tokenizer": tokenizer,
                "checkpoint": root / "initial",
                "input_cache": None,
                "output": root,
            }
            with (
                patch(
                    "rhsci.data.ardy_sidecar.dataset_content_fingerprint", return_value="dataset"
                ),
                patch("rhsci.data.input_cache.cache_identity", return_value=None),
            ):
                current, _ = build_training_contract(args, paths)
                self.assertEqual(
                    current["normalization_ranges"], build_normalization_ranges(statistics())
                )
                args.normalization = LEGACY_NORMALIZATION
                legacy, digest = build_training_contract(args, paths)
                self.assertEqual(legacy["recipe"], "SIMPLE-pi05-default-v4")
                self.assertNotIn("normalization_ranges", legacy)
                (root / "launch_contract.json").write_text(json.dumps(legacy))
                self.assertEqual(build_training_contract(args, paths), (legacy, digest))
                args.normalization = NORMALIZATION_VERSION
                with self.assertRaisesRegex(ValueError, "new run-tag"):
                    build_training_contract(args, paths)

    def test_legacy_v1_v2_are_not_reinterpreted_as_corrected_training(self):
        processor = self.processor()
        for version in ("simple-quantile-v1", "simple-quantile-zero-span-clipped-v2"):
            processor.normalization = version
            old = processor.denormalize(np.zeros(4))
            self.assertAlmostEqual(old[0], 0.7400005, places=7)
            if version.endswith("v2"):
                self.assertEqual(processor.normalize([0.42, 0, 0, 1], "action")[0], 0)

    def test_state_tokens_and_deployed_actions_keep_both_heights(self):
        import torch

        from openpi.rhsci.serve import Server
        from rhsci.inference.protocol import RequestMessage, ResponseMessage

        processor = Processor.__new__(Processor)
        processor.normalization = NORMALIZATION_VERSION
        processor.variant = "STATE"
        processor.state_measurement_noise_std_rad = 0.0
        processor.stats = {
            key: {
                name: [-1.0 if name in ("q01", "min") else 1.0] * width
                for name in ("q01", "q99", "min", "max")
            }
            for key, width in (("action", 36), ("states", 32))
        }
        for stats in processor.stats.values():
            for name in ("q01", "q99", "max"):
                stats[name][31] = 0.74
            stats["min"][31] = 0.42
        processor.normalization_ranges = build_normalization_ranges(processor.stats)
        prompts = []
        processor.tokenizer = SimpleNamespace(
            encode=lambda text, **kw: prompts.append(text) or [1, 2]
        )
        server = Server.__new__(Server)
        server.processor = processor
        server.device, server.variant = torch.device("cpu"), "STATE"
        for height, normalized in ((0.42, -1), (0.74, 1)):
            prediction = torch.zeros(1, 30, 36)
            prediction[:, :, 31] = normalized
            server.model = SimpleNamespace(sample_actions=lambda *args, **kw: prediction)
            state = np.zeros((1, 32), np.float32)
            state[:, 31] = height
            request = RequestMessage(
                image={"rgb_head_stereo_left": np.zeros((2, 2, 3), np.uint8)},
                instruction="move",
                state={"states": state},
                history={},
                condition={},
                gt_action=[],
                dataset_name="fixture",
                timestamp="0",
            )
            response = server.predict_action(request.serialize())
            actions = ResponseMessage.deserialize(json.loads(response.body)).action
            np.testing.assert_allclose(actions[:, 31], height, atol=1e-7)
        bins = [int(text.split("State: ")[1].split(";")[0].split()[-1]) for text in prompts]
        self.assertEqual(bins, [0, 255])


if __name__ == "__main__":
    unittest.main()
