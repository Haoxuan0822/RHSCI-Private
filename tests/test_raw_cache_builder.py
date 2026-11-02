"""缓存构建直接读取原生帧表，不需要 ACT 处理器或模型统计。"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rhsci.data.lerobot import load_raw_lerobot_dataset


class RawCacheBuilderTests(unittest.TestCase):
    def test_native_table_values_are_preserved_without_act_statistics(self):
        from psi.data.lerobot.compat import LeRobotDataset

        with tempfile.TemporaryDirectory(prefix="rhsci-raw-table-") as directory:
            root = Path(directory) / "fixture"
            native = LeRobotDataset.create(
                repo_id="fixture",
                root=root,
                fps=50,
                use_videos=False,
                features={
                    "states": {"dtype": "float32", "shape": (32,), "names": None},
                    "action": {"dtype": "float32", "shape": (36,), "names": None},
                },
            )
            for index in range(3):
                native.add_frame(
                    {
                        "states": np.full(32, index, dtype=np.float32),
                        "action": np.full(36, index + 10, dtype=np.float32),
                    },
                    task="fixture",
                )
            native.save_episode()
            loaded = load_raw_lerobot_dataset(root)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded.meta.fps, 50)
            self.assertIsNone(loaded.delta_indices)
            self.assertFalse((root / "meta/stats_psi0.json").exists())
            torch.testing.assert_close(
                torch.stack(loaded.hf_dataset["states"]),
                torch.arange(3, dtype=torch.float32)[:, None].expand(3, 32),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                torch.stack(loaded.hf_dataset["action"]),
                (torch.arange(3, dtype=torch.float32) + 10)[:, None].expand(3, 36),
                rtol=0,
                atol=0,
            )

    def test_cache_cli_does_not_import_any_backbone(self):
        script = Path(__file__).resolve().parents[1] / "scripts/data/build_raw_input_cache.py"
        code = """
import builtins, runpy, sys
original = builtins.__import__
def without_backbones(name, *args, **kwargs):
    if name.split('.')[0] in ('act', 'psi', 'gr00t', 'openpi', 'human_plan', 'lambda0'):
        raise AssertionError('cache CLI imported backbone: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = without_backbones
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(script)],
            capture_output=True,
            text=True,
            timeout=20,
            env={**os.environ, "PYTHONPATH": str(script.parents[2] / "src")},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cache-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
