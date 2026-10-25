"""Public launch commands, argument boundaries and custom dataset admission."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("baseline_cli", ROOT / "scripts/train/baseline.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
from types import SimpleNamespace

from rsci.paths import dataset_paths, with_nccl, worker_arguments


class TrainingCLITests(unittest.TestCase):
    def test_all_native_commands_resolve_to_vendored_entries(self):
        configs = {
            "psi0": "finetune_simple_psi0_config",
            "act": "simple_act_config",
            "dp": "simple_dp_config",
            "pi05": "pi05_droid",
            "internvla-m1": "/tmp/training config.yaml",
            "lambda0": "/tmp/experiment.json",
        }
        for backbone in cli.BACKBONES:
            with self.subTest(backbone=backbone):
                arguments = ["--backbone", backbone, "--gpus", "2,5"]
                if backbone in configs:
                    arguments += ["--config", configs[backbone]]
                command, cwd, env = cli.plan(cli.parser().parse_args(arguments + ["--", "--help"]))
                self.assertIn("--nproc-per-node=2", command)
                self.assertEqual(command[-1], "--help")
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,5")
                self.assertTrue(cwd.is_dir())
                for argument in command:
                    if argument.startswith(str(ROOT)) and argument.endswith(".py"):
                        self.assertTrue(Path(argument).is_file(), argument)

    def test_rsci_interpreter_and_arguments_reach_the_launcher(self):
        for backbone in cli.ADAPTED:
            arguments = [
                "--backbone",
                backbone,
                "--mode",
                "rsci",
                "--python",
                "/tmp/my python",
                "--gpus",
                "3",
                "--",
                "--task",
                "custom",
                "--output-dir",
                "/tmp/run with spaces",
            ]
            command, _, env = cli.plan(cli.parser().parse_args(arguments))
            self.assertTrue(Path(command[1]).is_file())
            self.assertEqual(
                command[2:],
                ["--gpus", "3", "--task", "custom", "--output-dir", "/tmp/run with spaces"],
            )
            self.assertIn("/tmp/my python", env.values())

    def test_beingh05_native_paths_and_unavailable_rsci(self):
        args = cli.parser().parse_args(
            [
                "--backbone",
                "beingh05",
                "--config",
                "/tmp/my data.yaml",
                "--args-file",
                str(ROOT / "configs/train/beingh05.json"),
                "--",
                "--output_dir",
                "/tmp/my run",
            ]
        )
        command, cwd, env = cli.plan(args)
        self.assertEqual(cwd, ROOT / "src/beingh05")
        self.assertIn(str(cwd), env["PYTHONPATH"].split(os.pathsep))
        index = command.index("--dataset_config_file")
        self.assertEqual(command[index + 1], "/tmp/my data.yaml")
        self.assertEqual(command[-2:], ["--output_dir", "/tmp/my run"])
        self.assertEqual(command[command.index("--action_chunk_length") + 1], "16")
        self.assertFalse((cwd / "configs/rng_state.pkl").exists())
        with self.assertRaisesRegex(ValueError, "no RHSCI adapter"):
            cli.plan(cli.parser().parse_args(["--backbone", "beingh05", "--mode", "rsci"]))

    def test_native_exec_preserves_cwd_environment_and_spaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "fake python"
            record = root / "record.json"
            executable.write_text(
                "#!" + sys.executable + "\nimport json, os, sys\n"
                'from pathlib import Path\nPath(os.environ["TRAIN_CLI_RECORD"]).write_text('
                'json.dumps(dict(args=sys.argv[1:], cwd=os.getcwd(), gpu=os.environ["CUDA_VISIBLE_DEVICES"])))\n'
            )
            executable.chmod(0o700)
            command = [
                sys.executable,
                "-B",
                str(ROOT / "scripts/train/baseline.py"),
                "--backbone",
                "h-rdt",
                "--python",
                str(executable),
                "--gpus",
                "6",
                "--",
                "--output_dir",
                "/tmp/run with spaces",
            ]
            subprocess.run(command, env={**os.environ, "TRAIN_CLI_RECORD": str(record)}, check=True)
            actual = json.loads(record.read_text())
            self.assertEqual(actual["cwd"], str(ROOT / "src/h_rdt"))
            self.assertEqual(actual["gpu"], "6")
            self.assertEqual(actual["args"][-2:], ["--output_dir", "/tmp/run with spaces"])
            record.unlink()
            subprocess.run(
                command[: command.index("--")] + ["--dry-run"] + command[command.index("--") :],
                env={**os.environ, "TRAIN_CLI_RECORD": str(record)},
                capture_output=True,
                check=True,
            )
            self.assertFalse(record.exists())

    def test_invalid_topology_and_unadapted_rsci_are_rejected(self):
        for flags in (
            ["--gpus", "0,0"],
            ["--nproc-per-node", "0"],
            ["--gpus", "1", "--nproc-per-node", "2"],
            ["--nnodes", "2", "--node-rank", "2"],
            ["--mode", "rsci"],
            ["--master-port", "22"],
        ):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                cli.plan(
                    cli.parser().parse_args(
                        ["--backbone", "act", "--config", "simple_act_config", *flags]
                    )
                )

    def test_native_argument_file_preserves_token_boundaries(self):
        for backbone in ("act", "dp"):
            args = cli.parser().parse_args(
                [
                    "--backbone",
                    backbone,
                    "--config",
                    f"simple_{backbone}_config",
                    "--args-file",
                    str(ROOT / f"configs/train/{backbone}.json"),
                    "--",
                    "--exp",
                    "my task",
                ]
            )
            command, _, _ = cli.plan(args)
            self.assertEqual(command[-2:], ["--exp", "my task"])
            self.assertIn("--data.transform.field.normalize-state", command)
            self.assertIn("--standalone", command)

    def test_custom_datasets_require_explicit_query_labels_and_safe_task_names(self):
        args = SimpleNamespace(
            task="my_task", dataset=Path("/tmp/my dataset"), sidecar=None, model="VGQ-AE"
        )
        with self.assertRaisesRegex(ValueError, "--sidecar"):
            dataset_paths(args)
        args.sidecar = Path("/tmp/labels")
        paths = dataset_paths(args)
        self.assertEqual(paths["dataset"], args.dataset)
        args.task = "../escape"
        with self.assertRaises(ValueError):
            dataset_paths(args)

    def test_task_names_never_select_private_dataset_paths(self):
        for task in ("example", "hug", "oven"):
            args = SimpleNamespace(task=task, dataset=None, sidecar=None, model="STATE")
            with self.assertRaisesRegex(ValueError, "--dataset"):
                dataset_paths(args)
            args.dataset = Path("/data/new")
            self.assertEqual(dataset_paths(args)["task"], task)
            self.assertEqual(dataset_paths(args)["dataset"], args.dataset)

    def test_environment_nccl_default_and_explicit_override(self):
        with patch.dict(os.environ, RSCI_NCCL_LIBRARY="", RSCI_NCCL_VERSION="2.31.2"):
            command = ["python", "-m", "torch.distributed.run", "train.py"]
            self.assertEqual(with_nccl(command), command)
            wrapped = with_nccl(command, "/runtime/libnccl.so.2")
            self.assertIn("/runtime/libnccl.so.2", wrapped)
            self.assertEqual(wrapped[-len(command) :], command)

    @unittest.skipUnless(shutil.which("cc"), "C compiler required for the loader fixture")
    def test_explicit_nccl_is_preloaded_in_the_child_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library = root / "libnccl.so.fixture"
            source = root / "nccl.c"
            source.write_text("int ncclGetVersion(int *v) { *v = 22703; return 0; }\n")
            subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
            # The fixture exercises the real dynamic loader without importing CUDA.
            (root / "torch.py").write_text(
                "from types import SimpleNamespace\n"
                "__version__ = 'fixture'\n"
                "cuda = SimpleNamespace(nccl=SimpleNamespace(version=lambda: (2, 27, 3)))\n"
            )
            child = [
                sys.executable,
                "-c",
                (
                    "import ctypes; value=ctypes.c_int(); "
                    "assert ctypes.CDLL(None).ncclGetVersion(ctypes.byref(value)) == 0; "
                    "assert value.value == 22703; print('CHILD_PRELOADED')"
                ),
            ]
            environment = {**os.environ, "PYTHONPATH": folder, "PYTHONDONTWRITEBYTECODE": "1"}
            with patch.dict(os.environ, RSCI_NCCL_LIBRARY=str(library)):
                with patch.dict(os.environ):
                    os.environ.pop("RSCI_NCCL_VERSION", None)
                    command = with_nccl(child, expected_version="2.27.3")
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("CHILD_PRELOADED", result.stdout)
            self.assertIn("runtime=22703", result.stdout)
            wrong = command.copy()
            wrong[wrong.index("--expected-version") + 1] = "2.31.2"
            result = subprocess.run(wrong, env=environment, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("CHILD_PRELOADED", result.stdout)

    def test_worker_forwarding_keeps_custom_paths_but_removes_control_flags(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=Path)
        parser.add_argument("--max-steps", type=int)
        parser.add_argument("--guardian-fd", type=int)
        parser.add_argument("--dry-run", action="store_true")
        args = parser.parse_args(
            [
                "--dataset",
                "/tmp/data with spaces",
                "--max-steps",
                "123",
                "--guardian-fd",
                "4",
                "--dry-run",
            ]
        )
        self.assertEqual(
            worker_arguments(args, parser),
            ["--dataset", "/tmp/data with spaces", "--max-steps", "123"],
        )


if __name__ == "__main__":
    unittest.main()
