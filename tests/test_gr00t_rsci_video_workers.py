"""验证预检后fork的真实H264随机寻帧；逐像素哈希防止仅消除异常却读错帧。"""

import hashlib
import json
import multiprocessing
import pickle
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from rsci.backbones.gr00t.data import SimpleRSCIDataset


def image_hash(sample):
    return hashlib.sha256(np.stack(sample.images["rs_view"]).tobytes()).hexdigest()


class VideoProbe(Dataset):
    """只返回身份和哈希，使用生产raw_sample路径，不涉及VLM或CUDA。"""

    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        return index, image_hash(self.dataset.raw_sample(index))


@unittest.skipUnless(shutil.which("ffmpeg"), "Real H264 regression requires ffmpeg")
class VideoWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        meta = cls.root / "meta"
        meta.mkdir()
        (meta / "info.json").write_text(
            json.dumps(
                dict(
                    fps=50,
                    total_episodes=1,
                    total_frames=64,
                    data_path="episode_{episode_index:06d}.parquet",
                    video_path="episode_{episode_index:06d}.mp4",
                )
            )
        )
        (meta / "episodes.jsonl").write_text(json.dumps(dict(episode_index=0, length=64)) + "\n")
        (meta / "tasks.jsonl").write_text(json.dumps(dict(task_index=0, task="test")) + "\n")
        pq.write_table(
            pa.table(
                dict(
                    action=np.zeros((64, 36)).tolist(),
                    states=np.zeros((64, 32)).tolist(),
                    episode_index=[0] * 64,
                    frame_index=list(range(64)),
                    task_index=[0] * 64,
                )
            ),
            cls.root / "episode_000000.parquet",
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=50",
                "-frames:v",
                "64",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-g",
                "16",
                str(cls.root / "episode_000000.mp4"),
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "Linux fork regression")
    def test_preflight_then_eight_fork_workers_matches_serial_frames(self):
        for variant in ("NO-STATE", "VGQ-AE"):
            with self.subTest(variant=variant):
                # 只测RGB缓存；VGQ的正式教师校验另有独立测试，不伪造标签。
                dataset = SimpleRSCIDataset(self.root, "NO-STATE")
                dataset.variant = variant
                reference = {i: image_hash(dataset.raw_sample(i)) for i in range(len(dataset))}
                # 与生产preflight相同：父进程先读取0/6，再fork worker。
                dataset.raw_sample(0)
                dataset.raw_sample(6)
                indices = np.random.default_rng(292285).integers(0, len(dataset), size=512)
                observed = 0
                for positions, hashes in DataLoader(
                    VideoProbe(dataset, indices),
                    batch_size=8,
                    num_workers=8,
                    multiprocessing_context="fork",
                ):
                    for index, digest in zip(positions.tolist(), hashes):
                        self.assertEqual(digest, reference[index])
                        observed += 1
                self.assertEqual(observed, 512)

    def test_pickle_discards_handles_and_rebuilds_correct_frames(self):
        dataset = SimpleRSCIDataset(self.root, "NO-STATE")
        expected = image_hash(dataset.raw_sample(6))
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertFalse(restored._decoders)
        self.assertIsNone(restored._cache_pid)
        self.assertEqual(image_hash(restored.raw_sample(6)), expected)

    def test_decode_error_reports_exact_sample_and_preserves_cause(self):
        dataset = SimpleRSCIDataset(self.root, "NO-STATE")
        dataset.raw_sample(0)
        original = RuntimeError("injected decoder failure")
        with patch.object(type(dataset._decoders[0]), "get_frames_at", side_effect=original):
            with self.assertRaisesRegex(
                RuntimeError, r"episode_000000.mp4 episode=0 frame=12 indices=\[12\] sample=12"
            ) as caught:
                dataset.raw_sample(12)
        self.assertIs(caught.exception.__cause__, original)


if __name__ == "__main__":
    unittest.main()
