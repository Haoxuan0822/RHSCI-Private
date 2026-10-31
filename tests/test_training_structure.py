"""入口兼容及缓存选择跨子进程传递的CPU回归，不加载模型。"""

import importlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from rsci.data.cache_config import InputCacheConfig, cache_arguments
from rsci.paths import BACKBONE_MODULES, CODE_ROOT, worker_arguments


class TrainingStructureTests(unittest.TestCase):
    def test_model_modules_resolve_inside_native_packages(self):
        expected = {
            "psi0": "src/psi/rsci",
            "pi05": "src/openpi/rsci",
            "gr00t": "src/gr00t/gr00t/rsci",
            "egovla": "src/egovla/human_plan/rsci",
            "act": "src/act/rsci",
            "lambda0": "src/openlambda/lambda0/lambda0/rsci",
        }
        # 验证实际导入位置，防止独立环境中的旧 editable 包遮蔽本仓库。
        for backbone, relative in expected.items():
            with self.subTest(backbone=backbone):
                module = importlib.import_module(BACKBONE_MODULES[backbone] + ".train")
                self.assertEqual(Path(module.__file__).resolve().parent, CODE_ROOT / relative)
                if backbone in ("pi05", "gr00t"):
                    contract = importlib.import_module(BACKBONE_MODULES[backbone] + ".contract")
                    self.assertEqual(contract.CODE_ROOT, CODE_ROOT)
                else:
                    self.assertEqual(module.CODE, CODE_ROOT)
        self.assertFalse((CODE_ROOT / "src/rsci/backbones").exists())

    def test_native_package_entry_points_reexport_same_functions(self):
        for backbone in ("act", "pi05", "egovla", "gr00t"):
            with self.subTest(backbone=backbone):
                module = BACKBONE_MODULES[backbone]
                entry = importlib.import_module(f"{module}.train")
                config = importlib.import_module(f"{module}.config")
                worker = importlib.import_module(f"{module}.worker")
                self.assertIs(entry.build_argument_parser, config.build_argument_parser)
                self.assertIs(entry.resolve_training_paths, config.resolve_training_paths)
                self.assertIs(entry.build_training_arguments, config.build_training_arguments)
                self.assertIs(entry.run_training_worker, worker.run_training_worker)

    def test_cli_freezes_legacy_cache_default_and_worker_role(self):
        for backbone, env in (
            ("act", "RSCI_ACT_SPACE_CACHE"),
            ("pi05", "RSCI_INPUT_CACHE"),
            ("egovla", "RSCI_INPUT_CACHE"),
            ("gr00t", "RSCI_GR00T_INPUT_CACHE"),
        ):
            with self.subTest(backbone=backbone), patch.dict(os.environ, {env: "/cache/first"}):
                config = importlib.import_module(f"{BACKBONE_MODULES[backbone]}.config")
                cli = config.build_argument_parser()
                base = ["--task", "fixture", "--dataset", "/dataset", "--model", "STATE"]
                args = cli.parse_args(base)
                paths = config.resolve_training_paths(args)
                cache = paths["input_cache"]
                self.assertEqual(cache.root, Path("/cache/first"))
                self.assertEqual(cache.verification, "full")
                os.environ[env] = "/cache/changed"
                child_args = cli.parse_args(base + cache_arguments(cache) + ["--worker"])
                child = config.resolve_training_paths(child_args)["input_cache"]
                self.assertEqual(child.root, cache.root)
                self.assertEqual(child.verification, "metadata")
                disabled = config.resolve_training_paths(
                    cli.parse_args(base + ["--no-input-cache"])
                )["input_cache"]
                self.assertIsNone(disabled.root)
                self.assertIsNone(
                    config.resolve_training_paths(cli.parse_args(base + cache_arguments(disabled)))[
                        "input_cache"
                    ].root
                )
                if backbone == "gr00t":
                    forwarded = worker_arguments(child_args, cli)
                    self.assertIn("--input-cache-root", forwarded)
                    replay = cli.parse_args(forwarded + cache_arguments(child))
                    self.assertEqual(replay.input_cache_root, child.root)

    def test_invalid_verification_rejected(self):
        with self.assertRaises(ValueError):
            InputCacheConfig("/cache", "unknown")
