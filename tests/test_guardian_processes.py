"""Exercise guardian termination through real ACT and EgoVLA child processes."""

import os
import select
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GuardianProcessTests(unittest.TestCase):
    def test_guardians_forward_termination_and_reap_workers(self):
        code = """
import importlib, pathlib, subprocess, sys
from types import SimpleNamespace
from unittest.mock import patch
module = importlib.import_module(sys.argv[1])
real_popen = subprocess.Popen
def worker(*args, **kwargs):
    process = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                         start_new_session=True)
    print(process.pid, file=sys.stderr, flush=True)
    return process
with patch.object(module, 'latest', return_value=None), \
     patch.object(module, 'checkpoint_ok', return_value=False), \
     patch('subprocess.Popen', side_effect=worker):
    module.guardian(SimpleNamespace(gpus='0', max_steps=2),
                    {'output': pathlib.Path(sys.argv[2])}, {}, 'test')
"""
        for backbone in ("act", "egovla"):
            with self.subTest(backbone=backbone), tempfile.TemporaryDirectory() as temp:
                module = f"rsci.backbones.{backbone}." + (
                    "launch" if backbone == "act" else "train"
                )
                env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
                guard = subprocess.Popen(
                    [sys.executable, "-u", "-c", code, module, temp],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    start_new_session=True,
                )
                child = None
                try:
                    self.assertTrue(
                        select.select([guard.stderr], [], [], 10)[0],
                        "guardian startup timed out",
                    )
                    child = int(guard.stderr.readline())
                    # The guardian is waiting inside Popen.wait when SIGTERM arrives.
                    import time

                    time.sleep(0.1)
                    guard.send_signal(signal.SIGTERM)
                    _, errors = guard.communicate(timeout=10)
                    self.assertEqual(guard.returncode, 143, errors)
                    self.assertFalse(Path(f"/proc/{child}").exists())
                finally:
                    if child is not None and Path(f"/proc/{child}").exists():
                        try:
                            os.killpg(child, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if guard.poll() is None:
                        guard.kill()
                    guard.communicate()


if __name__ == "__main__":
    unittest.main()
