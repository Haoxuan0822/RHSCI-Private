"""暂停续测只计完整episode一次，且保留原片段。"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "experiments/pi05_between/summarize_results.py"
spec = importlib.util.spec_from_file_location("pi05_eval_summary", path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class EvaluationResumeTests(unittest.TestCase):
    def test_merge_interrupted_and_resumed_episode_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "stats/eval_stats.txt"
            first.parent.mkdir()
            original = "episode_0: True\nepisode_1: False\n"
            first.write_text(original)
            second = root / "resume-01/stats/eval_stats.txt"
            second.parent.mkdir(parents=True)
            second.write_text("episode_2: True\n")
            self.assertEqual(summary.read_round_stats(root), {0: True, 1: False, 2: True})
            self.assertEqual(first.read_text(), original)
            self.assertNotIn(3, summary.read_round_stats(root))

    def test_duplicate_episode_in_a_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for parent in (root, root / "resume-01"):
                (parent / "stats").mkdir(parents=True)
                (parent / "stats/eval_stats.txt").write_text("episode_0: True\n")
            with self.assertRaisesRegex(ValueError, "Duplicate resumed episode"):
                summary.read_round_stats(root)

    def test_empty_interrupted_attempt_does_not_add_a_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(summary.read_round_stats(root), {})
            (root / "resume-01/stats").mkdir(parents=True)
            (root / "resume-01/stats/eval_stats.txt").write_text("episode_0: True\n")
            self.assertEqual(summary.read_round_stats(root), {0: True})


if __name__ == "__main__":
    unittest.main()
