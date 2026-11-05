"""An evaluation must not accept another policy's listener during GPU handoff."""

import argparse
import fcntl
import importlib.util
import json
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
    def test_mo_method_is_forwarded_to_each_native_server_before_launch(self):
        class StopBeforeLaunch(Exception):
            pass

        for backbone in ("pi05", "gr00t", "psi0"):
            with self.subTest(backbone=backbone), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "evals").mkdir()
                args = worker.build_argument_parser().parse_args([
                    "--gpu", "0", "--variant", "VGQ-AE-Mo", "--level", "dr-level-0",
                    "--backbone", backbone, "--run-dir", str(root / "model"),
                    "--output-root", str(root / "results"),
                ])
                with patch.object(worker, "ROOT", root), patch.object(
                    worker.subprocess, "Popen", side_effect=StopBeforeLaunch
                ) as launch:
                    with self.assertRaises(StopBeforeLaunch):
                        worker.worker(args)
                command = launch.call_args.args[0]
                self.assertEqual(command[command.index("--expected-variant") + 1], "VGQ-AE-Mo")

    def test_resume_noise_rejects_switch_type_level_seed_and_protocol_changes(self):
        from rhsci.inference.measurement_noise import MeasurementNoiseConfig

        original = MeasurementNoiseConfig("gaussian", "low", 17)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = {"state_noise_config": original.metadata()}
            worker.validate_resume_noise(root, previous, original)
            for changed in (
                None,
                MeasurementNoiseConfig("zero-offset", "low", 17),
                MeasurementNoiseConfig("gaussian", "high", 17),
                MeasurementNoiseConfig("gaussian", "low", 18),
            ):
                with self.subTest(config=changed), self.assertRaisesRegex(ValueError, "noise configuration differs"):
                    worker.validate_resume_noise(root, previous, changed)
            with self.assertRaisesRegex(ValueError, "noise configuration differs"):
                worker.validate_resume_noise(root, {}, original)
            dropout = MeasurementNoiseConfig("dropout", "low")
            legacy = dropout.metadata() | {"protocol": "client-measurement-noise-v1"}
            with self.assertRaisesRegex(ValueError, "noise configuration differs"):
                worker.validate_resume_noise(root, {"state_noise_config": legacy}, dropout)

    def test_resume_checks_every_segment_and_accepts_historical_clean_records(self):
        from rhsci.inference.measurement_noise import MeasurementNoiseConfig

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "repeat-01/resume-01/effective_eval_config.json"
            segment.parent.mkdir(parents=True)
            segment.write_text("{}")
            worker.validate_resume_noise(root, {}, None)
            clean = {"state_noise_config": {"enabled": False}}
            noise = MeasurementNoiseConfig("delay", "medium")
            segment.write_text(json.dumps({"state_noise_config": noise.metadata()}))
            with self.assertRaisesRegex(ValueError, "resume-01"):
                worker.validate_resume_noise(root, clean, None)
            segment.write_text(json.dumps({"state_noise": True}))
            with self.assertRaisesRegex(ValueError, "Missing saved"):
                worker.validate_resume_noise(root, clean, None)

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
