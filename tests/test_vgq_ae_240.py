"""240 ms 变体仅改变视觉取帧：核对数据边界、原模型梯度与单帧推理。"""

import importlib
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rhsci.data.visual_history import visual_history_offsets
from test_vgq_ae_motion import sidecar


def assert_same_parameters(test, reference, widened):
    """同一随机种子下参数逐项相等，证明没有新增头或复制共享模块。"""
    test.assertEqual(reference.state_dict().keys(), widened.state_dict().keys())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, widened.state_dict()[name], rtol=0, atol=0)


def dataset_fixture(dataset_class, variant, length=50):
    """真实 raw_sample 路径；缓存用像素值暴露实际取帧索引及 episode。"""
    dataset = dataset_class.__new__(dataset_class)
    dataset.variant, dataset.reader = variant, sidecar(length)
    dataset.ends = (length - 29) * np.array([1, 2])
    dataset.episodes = [dict(episode_index=index, length=length) for index in (0, 1)]
    dataset.tasks = {0: "move"}
    dataset._pid = dataset._cache_pid = os.getpid()
    dataset.path = lambda *args: Path("unused.mp4")
    dataset._table = lambda episode: dict(
        action=np.arange(length * 36, dtype=np.float32).reshape(length, 36),
        task_index=np.zeros(length, dtype=int),
    )
    cache = SimpleNamespace(
        frames=lambda episode, indices: np.broadcast_to(
            (100 * episode + np.array(indices, np.uint8))[:, None, None, None],
            (len(indices), 2, 2, 3),
        ).copy()
    )
    dataset.input_cache = dataset._input_cache = cache
    return dataset


def check_dataset(test, dataset_class, *, gr=False, ego=False):
    offsets = np.array(visual_history_offsets(test.variant))
    span = -offsets[0]
    current = dataset_fixture(dataset_class, test.variant, length=80)
    reference = dataset_fixture(dataset_class, "VGQ-AE", length=80)
    for episode in (0, 1):
        for frame in (0, 6, span - 1, span, span + 1):
            index = 51 * episode + frame
            sample, original = current.raw_sample(index), reference.raw_sample(index)
            if gr:
                values = sample.metadata
                pixels = np.array(sample.images["rs_view"])
            else:
                values, pixels = sample, sample["images"]
            requested = frame + offsets
            np.testing.assert_array_equal(
                pixels[:, 0, 0, 0], 100 * episode + np.maximum(requested, 0)
            )
            prefix = "" if ego else "rsci_"
            valid = values[prefix + "history_valid"]
            np.testing.assert_array_equal(valid, requested >= 0)
            target = current.reader.get(episode, frame)
            np.testing.assert_array_equal(values[prefix + "ardy_target"], target.ardy_body_latent)
            # GR 在模型/累积计数处联合 history mask；其它路径在数据侧合并。
            test.assertEqual(bool(values[prefix + "ardy_valid"] and valid.all()), frame >= span)
            test.assertFalse(any("motion" in key for key in values))
            if gr:
                for key in sample.actions:
                    np.testing.assert_array_equal(sample.actions[key], original.actions[key])
                test.assertFalse(sample.states)
            else:
                np.testing.assert_array_equal(sample["actions"], original["actions"])
                test.assertTrue(sample.get("state") is None and "states" not in sample)
            # 旧变体仍采用六帧跨度，不受新增方法影响。
            old_pixels = np.array(original.images["rs_view"]) if gr else original["images"]
            np.testing.assert_array_equal(
                old_pixels[:, 0, 0, 0],
                100 * episode + np.maximum(frame + np.array([-6, -4, -2, 0]), 0),
            )


class ConfigurationTests(unittest.TestCase):
    def test_all_launchers_select_separate_variant_without_motion_options(self):
        from rhsci.data.ardy_motion import motion_config

        for package in ("psi", "openpi", "gr00t", "human_plan", "lambda0"):
            module = "training_config" if package == "psi" else "config"
            parser = importlib.import_module(f"{package}.rhsci.{module}").build_argument_parser()
            flag = "--variant" if package == "lambda0" else "--model"
            args = parser.parse_args(["--task", "fixture", flag, "VGQ-AE-240"])
            self.assertEqual(motion_config(args), {})
        self.assertEqual(visual_history_offsets("VGQ-AE-240"), (-12, -8, -4, 0))
        self.assertEqual(visual_history_offsets("VGQ-AE-Mo"), (-6, -4, -2, 0))


class PsiTests(unittest.TestCase):
    variant = "VGQ-AE-240"

    def test_visual_timestamps_and_history_validity(self):
        from psi.config.transform import SimpleRepackTransform

        old = SimpleRepackTransform(rsci_variant="VGQ-AE")
        widened = SimpleRepackTransform(rsci_variant=self.variant)
        offsets = np.array(visual_history_offsets(self.variant))
        span = -offsets[0]
        stamps = widened.delta_timestamps(50)
        self.assertEqual(stamps["observation.images.egocentric"], (offsets / 50).tolist())
        for key in ("states", "action"):
            self.assertEqual(stamps[key], old.delta_timestamps(50)[key])
        for frame in (6, span - 1, span):
            target = sidecar(80).get(0, frame)
            raw = dict(
                states=torch.full((1, 32), float("nan")),
                action=torch.ones(30, 36),
                action_is_pad=torch.zeros(30, dtype=torch.bool),
                task="move",
                ardy_target=target.ardy_body_latent,
                ardy_valid=target.ardy_loss_mask,
                **{
                    "observation.images.egocentric": torch.rand(4, 3, 4, 6),
                    "observation.images.egocentric_is_pad": frame + offsets < 0,
                },
            )
            sample = widened(raw)
            self.assertEqual(sample["ardy_valid"], frame >= span)
            self.assertFalse(sample["states"].any())
            np.testing.assert_array_equal(sample["ardy_target"], target.ardy_body_latent)
        from psi.rhsci.training_config import build_argument_parser, resolve_training_plan

        for option in (["--no-reader"], ["--ardy-offset", "1"]):
            args = build_argument_parser().parse_args(
                ["--task", "fixture", "--model", self.variant, *option]
            )
            with self.assertRaisesRegex(ValueError, "retains the Reader"):
                resolve_training_plan(args)

    def test_existing_query_gradients_and_current_only_inference(self):
        from test_rsci import inputs, policy, sample
        from psi0_fixtures import trainer

        torch.set_num_threads(1)
        torch.manual_seed(37)
        _, reference = policy("VGQ-AE")
        torch.manual_seed(37)
        config, model = policy(self.variant)
        assert_same_parameters(self, reference, model)
        data = sample()
        losses = trainer(config).forward_and_loss(model, data)
        losses["loss_ardy"].backward()
        self.assertGreater(model.action_header.rsci.state_query.grad.abs().sum(), 0)
        model.eval()
        current = inputs(data, auxiliary=False)
        for key in ("history_vlm_inputs", "history_rgb_images", "history_valid"):
            current.pop(key)
        with patch.object(
            model, "_rhsci_query_latent", side_effect=AssertionError("inference Query")
        ):
            self.assertTrue(torch.isfinite(model(**current).action).all())


class PiTests(unittest.TestCase):
    variant = "VGQ-AE-240"

    def test_sampling_and_unchanged_current_targets(self):
        from openpi.rhsci.data import SimpleRHSCIDataset

        check_dataset(self, SimpleRHSCIDataset)

    def test_existing_network_and_single_frame_sampling(self):
        from test_pi05_rsci import fixture

        torch.set_num_threads(1)
        model, batch = fixture(self.variant)
        assert_same_parameters(self, fixture("VGQ-AE")[0], model)
        self.assertEqual(model.rsci_config["history"], list(visual_history_offsets(self.variant)))
        result = model(**batch)
        self.assertNotIn("motion_loss", result)
        result["loss"].backward()
        self.assertGreater(model.query.grad.abs().sum(), 0)
        with patch.object(model.reader, "forward", side_effect=AssertionError("inference Reader")):
            self.assertEqual(
                model.sample_actions(
                    batch["images"][:, -1:], batch["tokens"], batch["token_mask"], num_steps=1
                ).shape,
                (2, 30, 36),
            )


class GrTests(unittest.TestCase):
    variant = "VGQ-AE-240"

    def test_sampling_and_modality_keep_inference_current_only(self):
        from gr00t.rhsci.data import SimpleRHSCIDataset, TAG, modality

        check_dataset(self, SimpleRHSCIDataset, gr=True)
        self.assertEqual(
            modality(self.variant)[TAG.value]["video"].delta_indices,
            list(visual_history_offsets(self.variant)),
        )
        self.assertEqual(
            modality(self.variant, training=False)[TAG.value]["video"].delta_indices, [0]
        )

    def test_existing_dit_and_history_mask(self):
        from test_gr00t_rsci import batch, model

        torch.set_num_threads(1)
        torch.manual_seed(37)
        reference = model("VGQ-AE")
        torch.manual_seed(37)
        widened = model(self.variant)
        assert_same_parameters(self, reference, widened)
        data = batch()
        output = widened(data)
        self.assertNotIn("motion_loss", output)
        output["ardy_loss"].backward()
        self.assertGreater(widened.action_head.rsci_query.query.grad.abs().sum(), 0)
        data["rsci_history_valid"][:, 0] = False
        data["rsci_ardy_target"].fill_(float("nan"))
        self.assertEqual(widened(data)["ardy_loss"].item(), 0)
        current = {key: value for key, value in data.items() if not key.startswith("rsci_")}
        with patch.object(
            widened.action_head.rsci_query, "forward", side_effect=AssertionError("inference Query")
        ):
            self.assertIn("action_pred", widened.get_action(current))
        with self.assertRaisesRegex(ValueError, "Reader"):
            model(self.variant, memory_mode="full-vlm")


class EgoTests(unittest.TestCase):
    variant = "VGQ-AE-240"

    def test_sampling_and_unchanged_current_targets(self):
        from human_plan.rhsci.data import SimpleDataset

        check_dataset(self, SimpleDataset, ego=True)

    def test_existing_decoder_and_current_only_inference(self):
        from egovla.test_egovla import Backbone, batch
        from human_plan.rhsci.model import EgoVLARHSCI

        torch.set_num_threads(1)
        torch.manual_seed(37)
        reference = EgoVLARHSCI(Backbone(), variant="VGQ-AE", vlm_training="frozen")
        torch.manual_seed(37)
        model = EgoVLARHSCI(Backbone(), variant=self.variant, vlm_training="frozen")
        assert_same_parameters(self, reference, model)
        self.assertEqual(model.rsci_config["history"], list(visual_history_offsets(self.variant)))
        data = batch()
        output = model(**data)
        self.assertNotIn("motion_loss", output)
        output["loss"].backward()
        self.assertGreater(model.query.grad.abs().sum(), 0)
        model.eval()
        current = {key: data[key] for key in ("images", "input_ids", "labels", "attention_mask")}
        current["images"] = current["images"][:, -1:]
        with patch.object(model.reader, "forward", side_effect=AssertionError("inference Reader")):
            self.assertEqual(model.sample_actions(**current).shape, (2, 30, 36))


class LambdaTests(unittest.TestCase):
    variant = "VGQ-AE-240"

    def test_video_span_and_padded_history_preserve_native_action_sample(self):
        from lambda0.rhsci.data import RHSCIPsiSimpleHandler, SafePsiSimpleHandler

        handler = RHSCIPsiSimpleHandler.__new__(RHSCIPsiSimpleHandler)
        handler.variant, handler.reader, handler.motion_delta = self.variant, sidecar(80), None
        offsets = np.array(visual_history_offsets(self.variant))
        span = -offsets[0]
        handler.storage_layout, handler.fps = "episode_v2", 50
        handler._episode_paths = lambda episode: (Path("unused.parquet"), Path("unused.mp4"))
        for frame in (0, 6, span - 1, span, span + 1):
            native = dict(
                proprio=torch.ones(32),
                proprio_valid_mask=torch.ones(32, dtype=torch.bool),
                action=torch.randn(50, 36),
            )

            def decode(path, *, start_frame, length, fps):
                self.assertEqual(start_frame, max(0, frame - span))
                self.assertEqual(length, frame - start_frame + 1)
                return np.broadcast_to(
                    np.arange(start_frame, start_frame + length, dtype=np.uint8)[
                        :, None, None, None
                    ],
                    (length, 2, 2, 3),
                ).copy()

            with (
                patch.object(SafePsiSimpleHandler, "_build_sample", return_value=native),
                patch("lambda0.rhsci.data.decode_video_segment", side_effect=decode),
            ):
                result = handler._build_sample(
                    item=dict(episode_index=1),
                    index=frame,
                    image_aug=lambda image: torch.tensor(np.array(image).copy()),
                )
            np.testing.assert_array_equal(
                result["history"][:, 0, 0, 0], np.maximum(frame + offsets[:-1], 0)
            )
            np.testing.assert_array_equal(
                result["ardy_target"], handler.reader.get(1, frame).ardy_body_latent
            )
            self.assertEqual(bool(result["ardy_valid"]), frame >= span)
            self.assertIs(result["action"], native["action"])
            self.assertFalse(result["proprio"].any())

    def test_shared_expert_without_motion_head(self):
        spec = importlib.util.spec_from_file_location(
            "lambda_240_fixtures", Path(__file__).parent / "lambda0/test_integration.py"
        )
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        torch.set_num_threads(1)
        torch.manual_seed(37)
        reference = fixtures.tiny("VGQ-AE")
        torch.manual_seed(37)
        model = fixtures.tiny(self.variant)
        model.install_auxiliary()
        assert_same_parameters(self, reference, model)
        data = fixtures.inputs(model)
        data.update(
            history=[fixtures.vlm_inputs() for _ in range(3)],
            history_valid=torch.ones(2, 4, dtype=torch.bool),
            ardy_target=torch.randn(2, 128),
            ardy_valid=torch.ones(2, dtype=torch.bool),
        )
        output = model(**data)
        output["loss"].backward()
        self.assertGreater(model.rsci_query.grad.abs().sum(), 0)
        self.assertFalse(hasattr(model, "rsci_motion"))
        self.assertTrue(torch.isfinite(model.last_ardy_loss))
        model.eval()
        current = fixtures.inputs(model)
        for key in ("action", "noise_seed", "action_expert_noise_samples", "rtc_max_delay"):
            current.pop(key)
        with patch.object(
            model.rsci_reader, "forward", side_effect=AssertionError("inference Reader")
        ):
            actions = model.generate_actions(
                **current, steps=2, generator=torch.Generator().manual_seed(4)
            )
        self.assertTrue(torch.isfinite(actions).all())
