"""Mo-240 沿用原双头检查，另核对240ms视觉与120ms差分相互独立。"""

import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import test_vgq_ae_motion as motion
from rhsci.data.ardy_motion import ardy_motion_target, motion_config
from rhsci.data.visual_history import visual_history_offsets
from test_vgq_ae_240 import assert_same_parameters, dataset_fixture

VARIANT = "VGQ-AE-Mo-240"


def check_motion_samples(test, dataset_class, *, gr=False, ego=False):
    dataset = dataset_fixture(dataset_class, test.variant, length=80)
    offsets = np.array(visual_history_offsets(test.variant))
    span = -offsets[0]
    for delta in (test.delta, 2):
        dataset.ardy_delta_frames = delta
        for episode in (0, 1):
            for frame in (0, test.delta, 8, span - 1, span, span + 1):
                sample = dataset.raw_sample(episode * 51 + frame)
                values = sample.metadata if gr else sample
                images = np.array(sample.images["rs_view"]) if gr else sample["images"]
                prefix = "" if ego else "rsci_"
                requested = frame + offsets
                np.testing.assert_array_equal(
                    images[:, 0, 0, 0], 100 * episode + np.maximum(requested, 0)
                )
                expected, valid = ardy_motion_target(dataset.reader, episode, frame, delta)
                np.testing.assert_array_equal(values[prefix + "motion_target"], expected)
                np.testing.assert_array_equal(
                    values[prefix + "ardy_target"],
                    dataset.reader.get(episode, frame).ardy_body_latent,
                )
                history_valid = values[prefix + "history_valid"]
                test.assertEqual(
                    bool(values[prefix + "motion_valid"] and history_valid.all()),
                    bool(valid and frame >= span),
                )
                # π/Ego直接使用此mask；不能只在测试或模型外临时合并历史有效性。
                if not gr:
                    test.assertEqual(
                        bool(values[prefix + "motion_valid"]), bool(valid and frame >= span)
                    )


class ConfigurationTests(unittest.TestCase):
    def test_all_launchers_keep_default_and_explicit_motion_settings(self):
        for package in ("psi", "openpi", "gr00t", "human_plan", "lambda0"):
            module = "training_config" if package == "psi" else "config"
            parser = importlib.import_module(f"{package}.rhsci.{module}").build_argument_parser()
            flag = "--variant" if package == "lambda0" else "--model"
            args = parser.parse_args(["--task", "fixture", flag, VARIANT])
            self.assertEqual(motion_config(args), dict(ardy_delta_frames=6, motion_weight=0.1))
            args = parser.parse_args(
                [
                    "--task",
                    "fixture",
                    flag,
                    VARIANT,
                    "--ardy-delta-frames",
                    "2",
                    "--motion-weight",
                    "0.3",
                ]
            )
            self.assertEqual(motion_config(args), dict(ardy_delta_frames=2, motion_weight=0.3))
        self.assertEqual(visual_history_offsets(VARIANT), (-12, -8, -4, 0))
        self.assertEqual(visual_history_offsets("VGQ-AE-Mo"), (-6, -4, -2, 0))


class PsiMotion240Tests(motion.PsiMotionTests):
    variant = VARIANT
    delta = 6

    def test_sampling_and_saved_motion_configuration(self):
        from psi.config.transform import SimpleRepackTransform
        from psi.rhsci.contract import build_training_contract
        from psi.rhsci.training_config import build_argument_parser, resolve_training_plan

        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--task",
                "fixture",
                "--model",
                self.variant,
                "--ardy-delta-frames",
                "2",
                "--motion-weight",
                "0.3",
            ]
        )
        plan = resolve_training_plan(args)
        config = plan.config
        contract = build_training_contract(config, 1)
        self.assertEqual(contract["model"]["rsci_variant"], self.variant)
        self.assertEqual(contract["model"]["rsci_motion_weight"], 0.3)
        self.assertEqual(contract["data"]["rsci_ardy_delta_frames"], 2)
        repack = config.data.transform.repack
        offsets = np.array(visual_history_offsets(self.variant))
        span = -offsets[0]
        self.assertEqual(
            repack.delta_timestamps(50)["observation.images.egocentric"], (offsets / 50).tolist()
        )
        for frame in (8, span - 1, span):
            target, valid = ardy_motion_target(motion.sidecar(80), 0, frame, 2)
            raw = dict(
                states=torch.zeros(1, 32),
                action=torch.ones(30, 36),
                action_is_pad=torch.zeros(30, dtype=torch.bool),
                task="move",
                ardy_target=motion.sidecar(80).get(0, frame).ardy_body_latent,
                ardy_valid=True,
                motion_target=target,
                motion_valid=valid,
                **{
                    "observation.images.egocentric": torch.rand(4, 3, 4, 6),
                    "observation.images.egocentric_is_pad": frame + offsets < 0,
                },
            )
            packed = SimpleRepackTransform(rsci_variant=self.variant)(raw)
            self.assertEqual(packed["motion_valid"], frame >= span)
            self.assertEqual(packed["ardy_valid"], frame >= span)
        for option in (["--no-reader"], ["--ardy-offset", "1"]):
            args = parser.parse_args(["--task", "fixture", "--model", self.variant, *option])
            with self.assertRaisesRegex(ValueError, "retains the Reader"):
                resolve_training_plan(args)


class PiMotion240Tests(motion.PiMotionTests):
    variant = VARIANT
    delta = 6

    def test_visual_history_and_independent_difference_interval(self):
        from openpi.rhsci.data import SimpleRHSCIDataset

        check_motion_samples(self, SimpleRHSCIDataset)

    def test_parameter_layout_and_saved_sampling_metadata_match_mo(self):
        from test_pi05_rsci import fixture

        model, _ = fixture(self.variant)
        assert_same_parameters(self, fixture("VGQ-AE-Mo")[0], model)
        self.assertEqual(model.rsci_config["history"], list(visual_history_offsets(self.variant)))
        self.assertEqual(model.rsci_config["ardy_delta_frames"], self.delta)
        self.assertEqual(model.rsci_config["variant"], self.variant)


class GrMotion240Tests(motion.GrMotionTests):
    variant = VARIANT
    delta = 6

    def test_visual_history_and_independent_difference_interval(self):
        from gr00t.rhsci.data import SimpleRHSCIDataset

        check_motion_samples(self, SimpleRHSCIDataset, gr=True)


class EgoMotion240Tests(motion.EgoMotionTests):
    variant = VARIANT
    delta = 6

    def test_visual_history_and_independent_difference_interval(self):
        from human_plan.rhsci.data import SimpleDataset

        check_motion_samples(self, SimpleDataset, ego=True)


class LambdaMotion240Tests(motion.LambdaMotionTests):
    variant = VARIANT
    delta = 6

    def test_video_span_and_short_delta_require_complete_visual_history(self):
        from lambda0.rhsci.data import RHSCIPsiSimpleHandler, SafePsiSimpleHandler

        handler = RHSCIPsiSimpleHandler.__new__(RHSCIPsiSimpleHandler)
        handler.variant, handler.reader = self.variant, motion.sidecar(80)
        span = -visual_history_offsets(self.variant)[0]
        handler.storage_layout, handler.fps = "episode_v2", 50
        handler._episode_paths = lambda episode: (Path("unused.parquet"), Path("unused.mp4"))
        for delta in (self.delta, 2):
            handler.motion_delta = delta
            for frame in (0, self.delta, 8, span - 1, span, span + 1):
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
                expected, valid = ardy_motion_target(handler.reader, 1, frame, delta)
                np.testing.assert_array_equal(result["motion_target"], expected)
                self.assertEqual(bool(result["motion_valid"]), bool(valid and frame >= span))
                self.assertEqual(bool(result["ardy_valid"]), frame >= span)
                self.assertIs(result["action"], native["action"])
