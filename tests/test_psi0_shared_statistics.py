"""联合 bounds 的数据覆盖、验证隔离、启动传递和部署统计固定性。"""

import json
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from psi.config.transform import ActionStateTransform
from psi.rhsci.config import load_run_config
from psi.rhsci.normalization import merge_training_bounds, prepare_shared_statistics
from psi.rhsci.training_config import build_argument_parser, resolve_training_plan


class SharedStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="rhsci-shared-bounds-")
        self.root = Path(self.directory.name)
        self.write_stats("a", 0, 0)
        self.write_stats("b", -3, 2)
        self.write_stats("validation", -100, 100)

    def tearDown(self):
        self.directory.cleanup()

    def write_stats(self, repo, lo, hi):
        stats = {key: {"min": [lo] * size, "max": [hi] * size}
                 for key, size in (("action", 36), ("states", 32))}
        path = self.root / repo / "meta/stats_psi0.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats))
        return path

    def plan(self, *extra):
        args = build_argument_parser().parse_args([
            "--task", "joint", "--model", "STATE", "--data-root", str(self.root),
            "--train-repo-ids", "a", "b", "--val-repo-ids", "validation",
            "--output-dir", str(self.root / "output"), "--dry-run", *extra,
        ])
        return resolve_training_plan(args)

    def test_merge_extrema_covers_training_and_excludes_validation(self):
        stats = merge_training_bounds(self.root, ["a", "b"])
        self.assertEqual(stats["action"]["min"], [-3.] * 36)
        self.assertEqual(stats["states"]["max"], [2.] * 32)
        self.assertEqual([s["repo_id"] for s in stats["rhsci_shared_bounds"]["sources"]], ["a", "b"])
        for source in stats["rhsci_shared_bounds"]["sources"]:
            self.assertEqual(len(source["stats_sha256"]), 64)
        reverse = merge_training_bounds(self.root, ["b", "a"])
        self.assertEqual(stats["action"], reverse["action"])

    def test_native_roundtrip_preserves_second_task_and_previously_constant_dimension(self):
        plan = self.plan()
        destination = prepare_shared_statistics(plan.config)
        field = ActionStateTransform(stat_path=str(destination), action_norm_type="bounds",
                                     normalize_state=True)
        actions = np.stack([np.full(36, x, np.float32) for x in (-3, -1, 0, 2)])
        states = actions[:, :32].copy()
        result = field({"actions": actions.copy(), "states": states})
        np.testing.assert_allclose(field.reverse_call(result["actions"]), actions, atol=1e-6)
        self.assertTrue(np.all(np.abs(result["states"]) <= 1))
        self.assertNotEqual(result["actions"][0, 35], result["actions"][-1, 35])

    def test_plan_is_read_only_and_passes_shared_path_to_native_worker(self):
        plan = self.plan()
        path = Path(plan.config.data.transform.field.stat_path)
        self.assertFalse((self.root / "output").exists())
        self.assertIn(f"--data.transform.field.stat-path={path}", plan.command)
        self.assertTrue(path.is_absolute())
        prepare_shared_statistics(plan.config)
        self.assertTrue(path.is_file())
        # Worker重新解析命令时读取相同文件，父进程配置与worker的pad后统计一致。
        worker = self.plan().config.data.transform.field
        self.assertEqual(worker.action_min, plan.config.data.transform.field.action_min)
        self.assertEqual(worker.state_max, plan.config.data.transform.field.state_max)

    def test_check_cli_prepares_statistics_before_preflight_and_dry_run_writes_nothing(self):
        from psi.rhsci import train

        argv = ["train", "--task", "joint", "--model", "STATE", "--data-root", str(self.root),
                "--train-repo-ids", "a", "b", "--output-dir", str(self.root / "output")]
        with patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            with patch("sys.argv", argv + ["--dry-run"]), patch.object(train, "run_training_preflight") as preflight:
                train.main()
                preflight.assert_not_called()
                self.assertFalse((self.root / "output").exists())
            with patch("sys.argv", argv + ["--check"]), patch.object(train, "run_training_preflight") as preflight:
                train.main()
                preflight.assert_called_once()
                field = preflight.call_args.args[0].data.transform.field
                self.assertEqual(field.action_min, [-3.] * 36)
                self.assertTrue(Path(field.stat_path).is_file())

    def test_conflicting_existing_statistics_are_preserved(self):
        cfg = self.plan().config
        path = prepare_shared_statistics(cfg)
        before = path.read_bytes()
        prepare_shared_statistics(cfg)
        self.assertEqual(path.read_bytes(), before)
        self.write_stats("b", -4, 2)
        with self.assertRaisesRegex(ValueError, "statistics changed"):
            prepare_shared_statistics(cfg)
        self.assertEqual(path.read_bytes(), before)

    def test_explicit_and_single_task_paths_remain_unchanged(self):
        explicit = self.root / "a/meta/stats_psi0.json"
        cfg = self.plan("--stats-path", str(explicit)).config
        self.assertEqual(cfg.data.transform.field.stat_path, str(explicit))
        self.assertEqual(cfg.data.transform.field.action_max, [0.] * 36)
        cfg = self.plan("--train-repo-ids", "b").config
        self.assertEqual(cfg.data.transform.field.stat_path, "meta/stats_psi0.json")
        self.assertEqual(cfg.data.transform.field.action_min, [-3.] * 36)
        self.assertFalse((self.root / "output").exists())

    def test_saved_preprocessing_survives_removed_sources_and_changed_stats_file(self):
        cfg = self.plan().config
        path = prepare_shared_statistics(cfg)
        run = self.root / "saved"
        run.mkdir()
        (run / "argv.txt").write_text("train.py finetune_simple_psi0_config")
        (run / "run_config.json").write_text(cfg.model_dump_json())
        # 源统计消失、生成文件被外部改动，部署仍必须使用训练时嵌入的值。
        for repo in ("a", "b", "validation"):
            (self.root / repo / "meta/stats_psi0.json").unlink()
        path.write_text(json.dumps({"action": {"min": [-100.] * 36, "max": [100.] * 36}}))
        restored = load_run_config(run).data.transform.field
        self.assertEqual(restored.action_min, [-3.] * 36)
        self.assertEqual(restored.action_max, [2.] * 36)
        path.unlink()
        self.assertEqual(load_run_config(run).data.transform.field.state_min, [-3.] * 32 + [0.] * 4)

    def test_invalid_source_bounds_fail_before_publication(self):
        path = self.write_stats("b", 2, -3)
        with self.assertRaisesRegex(ValueError, "Invalid action bounds"):
            prepare_shared_statistics(self.plan().config)
        self.assertFalse((self.root / "output").exists())
        stats = json.loads(path.read_text())
        stats["action"]["min"] = [0.] * 35
        path.write_text(json.dumps(stats))
        with self.assertRaisesRegex(ValueError, "shape"):
            merge_training_bounds(self.root, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
