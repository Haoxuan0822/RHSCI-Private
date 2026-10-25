import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rsci.backbones.act.video_cache import VideoCache


class FakeDecoder:
    opens = 0
    reads = 0

    def __init__(self, *args, **kwargs):
        FakeDecoder.opens += 1
        self.metadata = SimpleNamespace(average_fps=50)

    def get_frames_at(self, indices):
        FakeDecoder.reads += 1
        return SimpleNamespace(
            data=torch.stack([torch.full((3, 2, 2), i, dtype=torch.uint8) for i in indices]),
            pts_seconds=torch.tensor([i / 50 for i in indices], dtype=torch.float64),
        )


class CacheTests(unittest.TestCase):
    def test_guardian_stop_does_not_reenter_popen_wait_lock(self):
        import signal
        import subprocess
        import sys

        code = """
import subprocess, sys
from rsci.backbones.act.launch import wait_worker
p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
print(p.pid, flush=True)
wait_worker(p)
"""
        guard = subprocess.Popen(
            [sys.executable, "-u", "-c", code],
            stdout=subprocess.PIPE,
            text=True,
            env=dict(
                os.environ,
                PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src")
                + os.pathsep
                + os.environ.get("PYTHONPATH", ""),
            ),
        )
        child = int(guard.stdout.readline())
        try:
            import time

            time.sleep(0.1)
            guard.send_signal(signal.SIGTERM)
            self.assertEqual(guard.wait(timeout=10), 143)
            self.assertFalse(Path(f"/proc/{child}").exists())
        finally:
            if guard.poll() is None:
                guard.kill()
                guard.wait()
            guard.stdout.close()

    def test_exact_reordered_duplicates_rng_and_mutation(self):
        with (
            tempfile.NamedTemporaryFile() as f,
            patch("torchcodec.decoders.VideoDecoder", FakeDecoder),
        ):
            FakeDecoder.opens = FakeDecoder.reads = 0
            cache = VideoCache()
            rng = torch.get_rng_state().clone()
            first = cache.decode(f.name, [0, 0.04, 0.04, 0.1], 0.001)
            first.zero_()
            second = cache.decode(f.name, [0.1, 0.04, 0], 0.001)
            self.assertTrue(
                torch.equal(
                    second,
                    torch.stack(
                        [torch.full((3, 2, 2), i, dtype=torch.uint8) for i in [5, 2, 0]]
                    ).float()
                    / 255,
                )
            )
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual((FakeDecoder.opens, FakeDecoder.reads), (1, 1))
            with self.assertRaises(AssertionError):
                cache.decode(f.name, [0.031], 0.00001)

    def test_bounds_fork_pickle_and_file_identity(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch("torchcodec.decoders.VideoDecoder", FakeDecoder),
        ):
            paths = [Path(root) / str(i) for i in range(3)]
            for p in paths:
                p.touch()
            cache = VideoCache(max_decoders=1, max_frame_bytes=12)
            for p in paths:
                cache.decode(p, [0, 0.02], 0.001)
            self.assertEqual(len(cache.decoders), 1)
            self.assertLessEqual(cache.frame_bytes, 12)
            restored = pickle.loads(pickle.dumps(cache))
            self.assertFalse(restored.decoders)
            cache.pid = -1
            cache.decode(paths[0], [0.0], 0.001)
            self.assertEqual(cache.pid, os.getpid())
            paths[0].write_bytes(b"changed")
            cache.decode(paths[0], [0.0, 0.02, 0.0], 0.001)
            self.assertEqual(len(cache.decoders), 1)


if __name__ == "__main__":
    unittest.main()
