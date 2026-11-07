"""480ms复用240ms的完整数据/前向/梯度检查，Mo默认回看240ms。"""

import importlib
import unittest

import numpy as np
import torch

import test_vgq_ae_240 as visual
import test_vgq_ae_motion_240 as motion
from rhsci.data.ardy_motion import ardy_motion_target, motion_config
from rhsci.data.visual_history import visual_history_offsets
from test_vgq_ae_motion import sidecar


class ConfigurationTests(unittest.TestCase):
    def test_visual_and_motion_defaults_are_independent_and_saved(self):
        for package in ("psi", "openpi", "gr00t", "human_plan", "lambda0"):
            module = "training_config" if package == "psi" else "config"
            parser = importlib.import_module(f"{package}.rhsci.{module}").build_argument_parser()
            flag = "--variant" if package == "lambda0" else "--model"
            for variant in ("VGQ-AE-480", "VGQ-AE-Mo-480"):
                args = parser.parse_args(["--task", "fixture", flag, variant])
                expected = dict(ardy_delta_frames=12, motion_weight=0.1) if "Mo" in variant else {}
                self.assertEqual(motion_config(args), expected)
                self.assertEqual(visual_history_offsets(variant), (-24, -16, -8, 0))
            args = parser.parse_args(
                ["--task", "fixture", flag, "VGQ-AE-Mo-480", "--ardy-delta-frames", "6"]
            )
            self.assertEqual(motion_config(args)["ardy_delta_frames"], 6)

        from psi.rhsci.training_config import build_argument_parser, resolve_training_plan
        from psi.rhsci.contract import build_training_contract

        args = build_argument_parser().parse_args(
            ["--task", "fixture", "--model", "VGQ-AE-Mo-480"]
        )
        config = resolve_training_plan(args).config
        contract = build_training_contract(config, 1)
        self.assertEqual(contract["data"]["rsci_ardy_delta_frames"], 12)
        self.assertEqual(contract["model"]["rsci_variant"], "VGQ-AE-Mo-480")

    def test_motion_uses_past_twelve_frames_and_both_endpoint_masks(self):
        reader = sidecar(80)
        target, valid = ardy_motion_target(reader, 1, 24, 12)
        self.assertTrue(valid)
        np.testing.assert_array_equal(target, np.full(128, 24**2 - 12**2, np.float32))
        reader._episodes[1]["ardy_loss_mask"][12] = False
        _, valid = ardy_motion_target(reader, 1, 24, 12)
        self.assertFalse(valid)


class PsiTests(visual.PsiTests):
    variant = "VGQ-AE-480"


class PiTests(visual.PiTests):
    variant = "VGQ-AE-480"


class GrTests(visual.GrTests):
    variant = "VGQ-AE-480"


class EgoTests(visual.EgoTests):
    variant = "VGQ-AE-480"


class LambdaTests(visual.LambdaTests):
    variant = "VGQ-AE-480"


class PsiMotionTests(motion.PsiMotion240Tests):
    variant = "VGQ-AE-Mo-480"
    delta = 12

    def test_collated_motion_labels_reach_real_training_loss(self):
        from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether
        from psi0_fixtures import trainer
        from test_rsci import policy, sample

        def unbatch(value, index):
            if isinstance(value, list):
                return [unbatch(item, index) for item in value]
            if isinstance(value, dict):
                return {key: unbatch(item, index) for key, item in value.items()}
            return value[index]

        data = sample()
        data.update(motion_target=torch.randn(2, 128), motion_valid=torch.ones(2, dtype=torch.bool))
        instances = [unbatch(data, index) for index in range(2)]
        # 原生collator的actions_mask输入是numpy，模型/损失侧仍接收tensor。
        for instance in instances:
            instance["actions_mask"] = instance["actions_mask"].numpy()
        batch = PaddedCollatorForTogether(64, 0)(instances)
        torch.testing.assert_close(batch["motion_target"], data["motion_target"])
        torch.testing.assert_close(batch["motion_valid"], data["motion_valid"])
        config, model = policy(self.variant)
        losses = trainer(config).forward_and_loss(model, batch)
        losses["loss_motion"].backward()
        self.assertGreater(model.action_header.rsci.motion_head[-1].weight.grad.abs().sum(), 0)


class PiMotionTests(motion.PiMotion240Tests):
    variant = "VGQ-AE-Mo-480"
    delta = 12


class GrMotionTests(motion.GrMotion240Tests):
    variant = "VGQ-AE-Mo-480"
    delta = 12

    def test_native_configuration_preserves_default_and_explicit_delta(self):
        from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

        self.assertEqual(Gr00tN1d6Config(rsci_variant=self.variant).rsci_ardy_delta_frames, 12)
        explicit = Gr00tN1d6Config(rsci_variant=self.variant, rsci_ardy_delta_frames=6)
        self.assertEqual(explicit.rsci_ardy_delta_frames, 6)


class EgoMotionTests(motion.EgoMotion240Tests):
    variant = "VGQ-AE-Mo-480"
    delta = 12


class LambdaMotionTests(motion.LambdaMotion240Tests):
    variant = "VGQ-AE-Mo-480"
    delta = 12

    def test_training_backend_defaults_to_twelve_frames(self):
        from lambda0.rhsci.runtime import RHSCITrainingBackend

        backend = RHSCITrainingBackend(None, self.variant, {})
        self.assertEqual(backend.motion_config["ardy_delta_frames"], 12)
