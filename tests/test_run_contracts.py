"""验证兼容源码编辑可恢复，但数据/模型/优化器配方变化仍会被识别。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rhsci.runtime.contracts import reuse_run_contract


class RunContractTests(unittest.TestCase):
    def test_rhsci_environment_names_preserve_legacy_launch_settings(self):
        """旧启动环境仍可读取；新名称优先，且读取过程不修改调用者环境。"""
        from rhsci.runtime.environment import get_env

        with patch.dict(os.environ, {"RSCI_INPUT_CACHE": "/legacy"}, clear=True):
            before = dict(os.environ)
            self.assertEqual(get_env("RHSCI_INPUT_CACHE"), "/legacy")
            self.assertEqual(dict(os.environ), before)
            os.environ["RHSCI_INPUT_CACHE"] = "/current"
            self.assertEqual(get_env("RHSCI_INPUT_CACHE"), "/current")
            self.assertEqual(get_env("RSCI_INPUT_CACHE"), "/current")
            os.environ["RHSCI_INPUT_CACHE"] = ""
            self.assertEqual(get_env("RHSCI_INPUT_CACHE"), "")
            self.assertEqual(get_env("CUDA_VISIBLE_DEVICES", "0"), "0")

    def test_native_assets_do_not_require_private_receipts(self):
        from human_plan.rhsci.preflight import assets
        from lambda0.rhsci.assets import verify_assets

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in (
                "config.json",
                "llm/config.json",
                "vision_tower/config.json",
                "mm_projector/config.json",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            (root / "model.safetensors").write_bytes(b"weight inventory fixture")
            self.assertEqual(assets(root)["schema"], "egovla-native-local-v1")
            self.assertIn("model.safetensors", verify_assets(root)["files"])
            self.assertFalse((root / "RSCI_ASSETS.json").exists())

    def test_source_and_runtime_changes_reuse_saved_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            saved = dict(
                model="STATE",
                dataset="fixed-data",
                optimizer={"lr": 1e-4},
                code={"old.py": "sha"},
                versions={"torch": "old"},
                initialization_stat={"st_mtime_ns": 1},
            )
            record = output / "launch_contract.json"
            record.write_text(json.dumps(saved))
            before = record.read_bytes()
            current = dict(model="STATE", dataset="fixed-data", optimizer={"lr": 1e-4})
            self.assertEqual(reuse_run_contract(current, output), saved)
            self.assertEqual(record.read_bytes(), before)
            for field, value in (
                ("model", "VGQ-AE"),
                ("dataset", "different"),
                ("optimizer", {"lr": 2e-4}),
            ):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "configuration"):
                    reuse_run_contract({**current, field: value}, output)

    def test_lambda_signature_reuses_saved_identity_without_source_or_lock_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            signature = dict(
                variant="VGQ-AE",
                config={"seed": 0},
                source={"model.py": "old"},
                upstream_patch="old",
                environment={"versions": {"torch": "old"}},
            )
            (Path(folder) / "rsci_run.json").write_text(json.dumps(signature))
            self.assertEqual(
                reuse_run_contract(
                    dict(variant="VGQ-AE", config={"seed": 0}), folder, "rsci_run.json"
                ),
                signature,
            )


if __name__ == "__main__":
    unittest.main()
