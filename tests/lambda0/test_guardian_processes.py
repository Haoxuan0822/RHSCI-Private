"""真实 CPU 子进程：λ₀ 守护退出前回收 worker/rank，锁覆盖整个训练生命周期。"""

import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARDIAN = """
import pathlib, subprocess, sys
from types import SimpleNamespace
from unittest.mock import patch
from lambda0.rhsci.launch import run_training_guardian
from lambda0.rhsci import launch
root = pathlib.Path(sys.argv[1])
real_popen = subprocess.Popen
def worker(command, **options):
    child = real_popen([sys.executable, '-u', str(root / 'worker.py'), str(root)], **options)
    (root / 'leader.json').write_text(str(child.pid))
    if (root / 'pause-spawn').exists():
        (root / 'spawn-ready').touch()
        import time
        time.sleep(0.5)
    return child
real_atomic_json = launch.atomic_json
def record(path, value):
    if (root / 'pause-record').exists():
        (root / 'record-ready').touch()
        import time
        time.sleep(0.5)
    return real_atomic_json(path, value)
real_cleanup = launch.cleanup
def cleanup(tracked):
    if (root / 'pause-cleanup').exists():
        (root / 'cleanup-ready').touch()
        import time
        time.sleep(0.5)
    return real_cleanup(tracked)
with patch('subprocess.Popen', side_effect=worker), patch.object(launch, 'atomic_json', record), patch.object(launch, 'cleanup', cleanup):
    run_training_guardian(SimpleNamespace(gpus='0', max_steps=2), [], root)
"""
WORKER = """
import json, os, pathlib, signal, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
rank = subprocess.Popen([sys.executable, '-u', '-c', '''
import signal, sys, time
def stop(*args):
    print('rank stopping', flush=True)
    time.sleep(0.3)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
print('rank ready', flush=True)
while True: time.sleep(0.1)
'''])
(root / 'workers.json').write_text(json.dumps([os.getpid(), rank.pid]))
while True: time.sleep(0.1)
"""


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def lock_available(root):
    with (root / "guardian.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


class GuardianProcessTests(unittest.TestCase):
    def start_guardian(self, root):
        return subprocess.Popen(
            [sys.executable, "-u", "-c", GUARDIAN, str(root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            start_new_session=True,
        )

    def wait_ready(self, root, guardian):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if guardian.poll() is not None:
                self.fail(f"Guardian exited during startup: {guardian.communicate()}")
            if (root / "workers.json").is_file() and any(
                "rank ready" in path.read_text() for path in root.glob("attempt-*.log")
            ):
                return json.loads((root / "workers.json").read_text())
            time.sleep(0.02)
        self.fail("Worker did not start")

    def clean_guardian(self, guardian, workers):
        if workers:
            try:
                os.killpg(workers[0], signal.SIGKILL)
            except ProcessLookupError:
                pass
        if guardian.poll() is None:
            guardian.kill()
        guardian.communicate(timeout=5)

    def test_stop_reaps_workers_before_releasing_lock(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=signum), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "worker.py").write_text(WORKER)
                guardian = self.start_guardian(root)
                workers = []
                try:
                    workers = self.wait_ready(root, guardian)
                    self.assertFalse(lock_available(root))
                    duplicate = self.start_guardian(root)
                    _, errors = duplicate.communicate(timeout=5)
                    self.assertNotEqual(duplicate.returncode, 0, errors)
                    self.assertIn("BlockingIOError", errors)
                    guardian.send_signal(signum)
                    deadline = time.monotonic() + 10
                    while guardian.poll() is None and time.monotonic() < deadline:
                        if lock_available(root):
                            self.assertFalse(any(alive(pid) for pid in workers))
                        time.sleep(0.02)
                    output, errors = guardian.communicate(timeout=5)
                    self.assertEqual(guardian.returncode, 128 + signum, errors)
                    self.assertFalse(any(alive(pid) for pid in workers))
                    self.assertTrue(lock_available(root))
                    self.assertIn("rank ready", output)
                finally:
                    self.clean_guardian(guardian, workers)

    def test_killed_guardian_does_not_allow_duplicate_while_worker_lives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "worker.py").write_text(WORKER)
            guardian = self.start_guardian(root)
            workers = []
            try:
                workers = self.wait_ready(root, guardian)
                guardian.kill()
                guardian.communicate(timeout=5)
                self.assertTrue(alive(workers[0]))
                self.assertFalse(lock_available(root))
            finally:
                self.clean_guardian(guardian, workers)

    def test_stop_during_spawn_or_record_still_cleans_workers(self):
        for phase in ("spawn", "record"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "worker.py").write_text(WORKER)
                (root / f"pause-{phase}").touch()
                guardian = self.start_guardian(root)
                workers = []
                try:
                    workers = self.wait_ready(root, guardian)
                    self.assertTrue((root / f"{phase}-ready").exists())
                    guardian.terminate()
                    _, errors = guardian.communicate(timeout=10)
                    self.assertEqual(guardian.returncode, 143, errors)
                    self.assertFalse(any(alive(pid) for pid in workers))
                    self.assertTrue(lock_available(root))
                finally:
                    self.clean_guardian(guardian, workers)

    def test_worker_record_failure_reaps_child_before_unlocking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "worker.py").write_text(WORKER)
            (root / "worker.json").mkdir()  # 模拟启动后写记录失败。
            guardian = self.start_guardian(root)
            workers = []
            try:
                _, errors = guardian.communicate(timeout=10)
                self.assertNotEqual(guardian.returncode, 0)
                self.assertIn("IsADirectoryError", errors)
                workers = [int((root / "leader.json").read_text())]
                self.assertFalse(alive(workers[0]))
                self.assertTrue(lock_available(root))
            finally:
                self.clean_guardian(guardian, workers)

    def test_successful_worker_exit_also_cleans_remaining_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, _ = WORKER.rsplit("while True: time.sleep(0.1)", 1)
            script += (
                "(root / 'training_complete.json').write_text(json.dumps({'global_step': 2}))\n"
            )
            (root / "worker.py").write_text(script)
            guardian = self.start_guardian(root)
            workers = []
            try:
                _, errors = guardian.communicate(timeout=10)
                self.assertEqual(guardian.returncode, 0, errors)
                workers = json.loads((root / "workers.json").read_text())
                self.assertFalse(any(alive(pid) for pid in workers))
                self.assertTrue(lock_available(root))
            finally:
                self.clean_guardian(guardian, workers)

    def test_stop_during_exit_cleanup_does_not_release_lock_early(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, _ = WORKER.rsplit("while True: time.sleep(0.1)", 1)
            script += (
                "(root / 'training_complete.json').write_text(json.dumps({'global_step': 2}))\n"
            )
            (root / "worker.py").write_text(script)
            (root / "pause-cleanup").touch()
            guardian = self.start_guardian(root)
            workers = []
            try:
                deadline = time.monotonic() + 10
                while not (root / "cleanup-ready").exists():
                    if guardian.poll() is not None or time.monotonic() > deadline:
                        self.fail("Guardian did not reach exit cleanup")
                    time.sleep(0.01)
                workers = json.loads((root / "workers.json").read_text())
                self.assertFalse(lock_available(root))
                guardian.terminate()
                _, errors = guardian.communicate(timeout=10)
                self.assertEqual(guardian.returncode, 143, errors)
                self.assertFalse(any(alive(pid) for pid in workers))
                self.assertTrue(lock_available(root))
            finally:
                self.clean_guardian(guardian, workers)


if __name__ == "__main__":
    unittest.main()
