"""入口兼容及缓存选择跨子进程传递的CPU回归，不加载模型。"""

import importlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from rsci.data.cache_config import InputCacheConfig, cache_arguments
from rsci.paths import worker_arguments


class TrainingStructureTests(unittest.TestCase):
    def test_legacy_entry_points_reexport_same_functions(self):
        for backbone in ("act", "pi05", "egovla", "gr00t"):
            with self.subTest(backbone=backbone):
                entry = importlib.import_module(f"rsci.backbones.{backbone}.train")
                config = importlib.import_module(f"rsci.backbones.{backbone}.config")
                worker = importlib.import_module(f"rsci.backbones.{backbone}.worker")
                self.assertIs(entry.parser, config.parser)
                self.assertIs(entry.resolve, config.resolve)
                self.assertIs(entry.training_arguments, config.training_arguments)
                self.assertIs(entry.train, worker.train)

    def test_cli_freezes_legacy_cache_default_and_worker_role(self):
        for backbone, env in (
            ("act", "RSCI_ACT_SPACE_CACHE"),
            ("pi05", "RSCI_INPUT_CACHE"),
            ("egovla", "RSCI_INPUT_CACHE"),
            ("gr00t", "RSCI_GR00T_INPUT_CACHE"),
        ):
            with self.subTest(backbone=backbone), patch.dict(os.environ, {env: "/cache/first"}):
                config = importlib.import_module(f"rsci.backbones.{backbone}.config")
                cli = config.parser()
                base = ["--task", "fixture", "--dataset", "/dataset", "--model", "STATE"]
                args = cli.parse_args(base)
                paths = config.resolve(args)
                cache = paths["input_cache"]
                self.assertEqual(cache.root, Path("/cache/first"))
                self.assertEqual(cache.verification, "full")
                os.environ[env] = "/cache/changed"
                child_args = cli.parse_args(base + cache_arguments(cache) + ["--worker"])
                child = config.resolve(child_args)["input_cache"]
                self.assertEqual(child.root, cache.root)
                self.assertEqual(child.verification, "metadata")
                disabled = config.resolve(cli.parse_args(base + ["--no-input-cache"]))[
                    "input_cache"
                ]
                self.assertIsNone(disabled.root)
                self.assertIsNone(
                    config.resolve(cli.parse_args(base + cache_arguments(disabled)))[
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
