"""CPU 子进程验证公共守护恢复、无进展退出和骨干命令绑定。"""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rhsci.data.cache_config import InputCacheConfig
from rhsci.runtime.guardian import run_checkpoint_guardian


class GuardianRecoveryTests(unittest.TestCase):
    def run_guardian(self, output, command, *, initial_resume=None, validate=lambda: None):
        def latest():
            points = list(output.glob("checkpoint-*"))
            return max(points, key=lambda point: int(point.name.split("-")[-1]), default=None)

        def checkpoint_ok(path, **kwargs):
            return (path / "complete").is_file()

        run_checkpoint_guardian(
            output,
            max_steps=2,
            resume=initial_resume,
            latest_checkpoint=latest,
            checkpoint_ok=checkpoint_ok,
            validate_run=validate,
            build_command=command,
            label="CPU",
        )

    def test_failed_worker_resumes_new_checkpoint_and_preserves_arguments(self):
        code = """
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
resume = sys.argv[2]
step = 1 if resume == 'None' else 2
point = root / ('checkpoint-' + str(step))
point.mkdir()
(point / 'complete').touch()
with (root / 'resumes.jsonl').open('a') as stream:
    stream.write(json.dumps(resume) + '\\n')
sys.exit(1 if step == 1 else 0)
"""
        with tempfile.TemporaryDirectory(prefix="rhsci guardian ") as directory:
            output = Path(directory)
            self.run_guardian(
                output, lambda resume: [sys.executable, "-c", code, str(output), str(resume)]
            )
            resumes = [
                json.loads(line) for line in (output / "resumes.jsonl").read_text().splitlines()
            ]
            self.assertEqual(resumes, ["None", str(output / "checkpoint-1")])
            self.assertEqual(
                json.loads((output / "training_started.json").read_text())["attempt"], 2
            )

    def test_failure_without_checkpoint_never_restarts_from_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "restart from zero"):
                self.run_guardian(
                    output, lambda resume: [sys.executable, "-c", "raise SystemExit(1)"]
                )
            self.assertEqual(
                json.loads((output / "training_started.json").read_text())["attempt"], 1
            )

    def test_existing_checkpoint_without_progress_is_not_replayed_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            point = output / "checkpoint-1"
            point.mkdir()
            (point / "complete").touch()
            with self.assertRaisesRegex(RuntimeError, "repeated replay"):
                self.run_guardian(
                    output, lambda resume: [sys.executable, "-c", "raise SystemExit(1)"]
                )
            self.assertEqual(
                json.loads((output / "training_started.json").read_text())["attempt"], 1
            )

    def test_bad_or_foreign_checkpoint_never_starts_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            foreign = output / "foreign" / "checkpoint-1"
            foreign.mkdir(parents=True)
            (foreign / "complete").touch()
            command = unittest.mock.Mock()
            for point in (foreign, output / "checkpoint-1"):
                with (
                    self.subTest(point=point),
                    self.assertRaisesRegex(ValueError, "verified complete"),
                ):
                    self.run_guardian(output, command, initial_resume=point)
            command.assert_not_called()

    def test_numerical_failure_or_existing_started_run_never_starts_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("NUMERICAL_FAILURE.json", "training_started.json"):
                marker = output / name
                marker.write_text("{}")
                command = unittest.mock.Mock()
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    self.run_guardian(output, command)
                command.assert_not_called()
                marker.unlink()

    def test_adapters_forward_parsed_options_and_only_selected_resume(self):
        for package, port in (("openpi", 32000), ("gr00t", 31000)):
            with self.subTest(package=package), tempfile.TemporaryDirectory() as directory:
                module = importlib.import_module(f"{package}.rhsci.guardian")
                args = module.build_argument_parser().parse_args(
                    [
                        "--task",
                        "fixture",
                        "--model",
                        "STATE",
                        "--dataset",
                        directory,
                        "--resume=old-checkpoint",
                        "--gpus",
                        "0,1",
                    ]
                )
                args.guardian_fd = 7
                output = Path(directory)
                with patch.object(module, "run_checkpoint_guardian") as shared:
                    module.run_training_guardian(
                        args, {"output": output, "input_cache": InputCacheConfig()}, "abcd"
                    )
                options = shared.call_args.kwargs
                command = options["build_command"](output / "checkpoint-1")
                self.assertIn(f"{package}.rhsci.train", command)
                self.assertIn(f"--master_port={port + int('abcd', 16) % 1000}", command)
                self.assertIn("--nproc_per_node=2", command)
                self.assertEqual(command.count("--worker"), 1)
                self.assertEqual(command.count("--resume"), 1)
                self.assertNotIn("--resume=old-checkpoint", command)
                self.assertEqual(
                    command[command.index("--resume") + 1], str(output / "checkpoint-1")
                )
                self.assertEqual(options["pass_fds"], (7,))


if __name__ == "__main__":
    unittest.main()
