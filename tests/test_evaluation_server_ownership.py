"""An evaluation must not accept another policy's listener during GPU handoff."""

import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

directory = Path(__file__).resolve().parents[1] / "experiments/pi05_between"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("evaluation_worker_ownership", directory / "run_evaluation.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ServerOwnershipTests(unittest.TestCase):
    def test_foreign_listener_is_not_our_model(self):
        script = "import socket,time; s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); print(s.getsockname()[1],flush=True); time.sleep(30)"
        process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        try:
            port = int(process.stdout.readline())
            self.assertTrue(worker.owns_listener(process.pid, port))
            self.assertFalse(worker.owns_listener(os.getpid(), port))
        finally:
            process.terminate()
            process.wait(timeout=5)
            process.stdout.close()
        self.assertFalse(worker.owns_listener(process.pid, port))

    def test_own_listener_survives_transient_proc_fds(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            self.assertTrue(worker.owns_listener(os.getpid(), listener.getsockname()[1]))

    def test_reserved_gpu_rejected_before_server_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            leases = root / "evals/.gpu-worker-leases"
            leases.mkdir(parents=True)
            with (leases / "gpu0-slot0.lock").open("a") as lease:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(worker, "ROOT", root), patch.object(worker.subprocess, "Popen") as launch:
                    with self.assertRaises(BlockingIOError):
                        worker.worker(argparse.Namespace(gpu=0, slot=0))
                    launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
