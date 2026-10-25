"""Validate model recipes at the public PSI0 training boundary."""

import sys
import unittest
from unittest.mock import patch

from psi.config.model_psi0 import Psi0ModelConfig
from rsci.backbones.psi0 import train


class TrainingGateTests(unittest.TestCase):
    def test_frozen_vlm_nonstar_rejected_before_config_or_assets(self):
        for variant in ("VGQ-VLM", "EGQ-VLM"):
            with (
                patch.object(sys, "argv", ["entry", "--model", variant, "--task", "G1Test-v0"]),
                patch.object(train, "resolved") as resolve,
            ):
                with self.assertRaisesRegex(ValueError, "no ARDY supervision path"):
                    train.main()
                resolve.assert_not_called()

    def test_supported_models_reach_normal_resolution(self):
        for variant, mode in (
            ("VGQ-VLM", "lora"),
            ("EGQ-VLM", "lora"),
            ("VGQ-AE", "frozen"),
            ("VGQ-VLM*", "frozen"),
        ):
            argv = ["entry", "--model", variant, "--task", "G1Test-v0", "--vlm", mode, "--dry-run"]
            with (
                patch.object(sys, "argv", argv),
                patch.object(train, "resolved", side_effect=RuntimeError("resolved")) as resolve,
            ):
                with self.assertRaisesRegex(RuntimeError, "resolved"):
                    train.main()
                resolve.assert_called_once()
                self.assertEqual(sys.argv, argv)

    def test_frozen_configuration_roundtrip(self):
        for variant in ("VGQ-VLM", "EGQ-VLM"):
            cfg = Psi0ModelConfig(rsci_variant=variant, n_conditions=0)
            self.assertEqual(
                Psi0ModelConfig.model_validate(cfg.model_dump()).rsci_vlm_mode, "frozen"
            )
