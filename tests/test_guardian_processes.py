"""Exercise guardian termination through actual child processes for four backbones."""

import os
import select
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rhsci.paths import BACKBONE_MODULES

ROOT = Path(__file__).resolve().parents[1]


class GuardianProcessTests(unittest.TestCase):
    def test_guardians_forward_termination_and_reap_workers(self):
        code = """
import importlib, pathlib, subprocess, sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from rhsci.data.cache_config import InputCacheConfig
module = importlib.import_module(sys.argv[1])
real_popen = subprocess.Popen
def worker(*args, **kwargs):
    process = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                         start_new_session=True)
    print(process.pid, file=sys.stderr, flush=True)
    return process
with ExitStack() as stack:
    name = 'latest' if hasattr(module, 'latest') else 'latest_checkpoint'
    stack.enter_context(patch.object(module, name, return_value=None))
    stack.enter_context(patch.object(module, 'checkpoint_ok', return_value=False))
    stack.enter_context(patch('subprocess.Popen', side_effect=worker))
    if hasattr(module, 'build_training_contract'):
        stack.enter_context(patch.object(module, 'build_training_contract',
                                         return_value=({}, 'abcd')))
    args = SimpleNamespace(gpus='0', max_steps=2, resume=None, guardian_fd=0,
                           nccl_library=None)
    paths = {'output': pathlib.Path(sys.argv[2]), 'input_cache': InputCacheConfig()}
    if module.__name__.startswith(('gr00t.', 'openpi.')):
        module.run_training_guardian(args, paths, 'abcd')
    else:
        module.run_training_guardian(args, paths, {}, 'abcd')
"""
        for backbone in ("act", "egovla", "pi05", "gr00t"):
            with self.subTest(backbone=backbone), tempfile.TemporaryDirectory() as temp:
                module = f"{BACKBONE_MODULES[backbone]}." + (
                    "launch" if backbone == "act" else "guardian"
                )
                env = dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT / "src/egovla"))),
                    PYTHONDONTWRITEBYTECODE="1",
                )
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
