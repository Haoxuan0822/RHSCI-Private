"""SIMPLE normalization preparation and reference sidecar provenance."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/data" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats_tool = load_tool("prepare_simple_stats")


class SimplePreparationTests(unittest.TestCase):
    def dataset(self, root):
        (root / "meta").mkdir()
        (root / "data/chunk-000").mkdir(parents=True)
        (root / "meta/info.json").write_text(
            json.dumps(
                {"fps": 50, "features": {"action": {"shape": [36]}, "states": {"shape": [32]}}}
            )
        )
        actions = np.arange(144, dtype=np.float32).reshape(4, 36)
        states = np.arange(128, dtype=np.float32).reshape(4, 32)
        for episode, indices in enumerate((slice(0, 2), slice(2, 4))):
            pq.write_table(
                pa.table({"action": actions[indices].tolist(), "states": states[indices].tolist()}),
                root / f"data/chunk-000/episode_{episode:06d}.parquet",
            )
        return actions, states

    def test_statistics_match_numeric_data_and_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actions, states = self.dataset(root)
            stats_tool.prepare_stats(root)
            before = {
                name: (root / "meta" / name).read_bytes()
                for name in ("stats.json", "stats_psi0.json")
            }
            self.assertEqual(before["stats.json"], before["stats_psi0.json"])
            stats = json.loads(before["stats.json"])
            for key, values in (("action", actions), ("states", states)):
                np.testing.assert_allclose(stats[key]["min"], values.min(axis=0))
                np.testing.assert_allclose(stats[key]["mean"], values.mean(axis=0))
                np.testing.assert_allclose(stats[key]["q99"], np.quantile(values, 0.99, axis=0))
            stats_tool.prepare_stats(root)
            for name, original in before.items():
                self.assertEqual((root / "meta" / name).read_bytes(), original)

    def test_conflict_leaves_both_files_untouched_and_overwrite_preserves_other_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.dataset(root)
            destination = root / "meta/stats_psi0.json"
            original = '{"action": {"min": [99]}, "custom": {"value": 1}}'
            destination.write_text(original)
            with self.assertRaisesRegex(FileExistsError, "Conflicting"):
                stats_tool.prepare_stats(root)
            self.assertEqual(destination.read_text(), original)
            self.assertFalse((root / "meta/stats.json").exists())
            stats_tool.prepare_stats(root, overwrite=True)
            self.assertEqual(json.loads(destination.read_text())["custom"], {"value": 1})

    def test_invalid_timing_or_numeric_data_is_rejected_without_writes(self):
        for invalid in ("fps", "shape", "nan"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                actions, states = self.dataset(root)
                if invalid == "fps":
                    info = json.loads((root / "meta/info.json").read_text())
                    info["fps"] = 30
                    (root / "meta/info.json").write_text(json.dumps(info))
                else:
                    if invalid == "nan":
                        actions[0, 0] = np.nan
                    else:
                        actions = actions[:, :-1]
                    pq.write_table(
                        pa.table({"action": actions.tolist(), "states": states.tolist()}),
                        root / "data/chunk-000/episode_000000.parquet",
                    )
                with self.assertRaises(ValueError):
                    stats_tool.prepare_stats(root)
                self.assertFalse((root / "meta/stats.json").exists())

    def test_sidecar_provenance_hashes_the_exported_module(self):
        builder = load_tool("build_simple_ardy_sidecars")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.dataset(root)
            (root / "meta/replay.txt").write_text("fixture")
            context = builder._build_context(root, root, {}, builder.ARDY_SOURCE_COMMIT)
            self.assertEqual(len(context["generator_sha256"]["ardy_sidecar.py"]), 64)


if __name__ == "__main__":
    unittest.main()
