"""旧SIMPLE数值统计兼容：原生聚合、单库/多库路径及磁盘身份。"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from psi.data.lerobot.compat import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
    complete_numeric_episode_counts,
    simple_rgb_parquet_loading,
)


class EpisodeStatisticsTests(unittest.TestCase):
    def test_existing_counts_preserved_and_unknown_counts_rejected(self):
        count = np.array([3])
        statistics = {0: {"action": {"count": count}}}
        self.assertIs(complete_numeric_episode_counts(statistics, {})[0]["action"]["count"], count)
        with self.assertRaisesRegex(ValueError, "Cannot infer"):
            complete_numeric_episode_counts({0: {"observation.images.rgb": {}}}, {0: {"length": 8}})
        with self.assertRaisesRegex(ValueError, "Invalid episode length"):
            complete_numeric_episode_counts({0: {"action": {}}}, {0: {"length": 0}})

    def test_native_single_and_multi_aggregate_without_rewriting_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            originals = {}
            references = []
            for name, offset in (("a", 0), ("b", 10)):
                dataset = LeRobotDataset.create(
                    repo_id=name, root=root / name, fps=50, use_videos=False,
                    features={key: {"dtype": "float32", "shape": (width,), "names": None}
                              for key, width in (("states", 32), ("action", 36))},
                )
                for length, value in ((2, 1 + offset), (5, 4 + offset)):
                    for _ in range(length):
                        dataset.add_frame({"states": np.full(32, value, np.float32),
                                           "action": np.full(36, value, np.float32)}, task="move")
                    dataset.save_episode()
                references.append(copy.deepcopy(LeRobotDatasetMetadata(name, root / name).stats))
                path = root / name / "meta/episodes_stats.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows:
                    for key in ("states", "action"):
                        row["stats"][key].pop("count")
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                originals[path] = path.read_bytes()
                metadata = LeRobotDatasetMetadata(name, root / name)
                for key in ("states", "action"):
                    for field in ("min", "max", "mean", "std", "count"):
                        np.testing.assert_allclose(metadata.stats[key][field], references[-1][key][field])
                    np.testing.assert_allclose(metadata.stats[key]["mean"], (2 * (1 + offset) + 5 * (4 + offset)) / 7)
            single = LeRobotDataset("a", root=root / "a")
            multi = MultiLeRobotDataset(["a", "b"], root=root)
            self.assertEqual(len(single), 7)
            self.assertEqual(len(multi), 14)
            np.testing.assert_allclose(multi.stats["action"]["mean"], 22 / 7 + 5)
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)

            # 同一任务内异构的非必需列不能阻断RGB训练；动作/索引仍取原文件。
            import pyarrow as pa
            import pyarrow.parquet as pq
            extra = root / "a/data/chunk-000/episode_000001.parquet"
            table = pq.read_table(extra)
            pq.write_table(table.append_column("unused_source_field", pa.array([7] * len(table))), extra)
            with simple_rgb_parquet_loading():
                projected = LeRobotDataset("a", root=root / "a")
                joint = MultiLeRobotDataset(["a", "b"], root=root)
            self.assertEqual(len(projected), 7)
            self.assertEqual(len(joint), 14)
            self.assertNotIn("unused_source_field", projected.hf_dataset.column_names)
            self.assertEqual(float(projected[2]["action"][0]), 4.)
            self.assertEqual(int(joint[7]["episode_index"]), 0)


if __name__ == "__main__":
    unittest.main()
