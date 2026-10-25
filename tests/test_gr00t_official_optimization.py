"""对照官方TrainingArguments创建真实优化器/调度器，验证生产配置。"""

import copy
import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from gr00t.configs.training.training_config import TrainingConfig
from transformers import Trainer, TrainingArguments

from rsci.backbones.gr00t.train import parser, resolve, training_arguments


class OfficialOptimizationTests(unittest.TestCase):
    def test_custom_dataset_output_and_step_budget(self):
        cli = parser().parse_args(
            [
                "--task",
                "custom",
                "--dataset",
                "/datasets/custom",
                "--sidecar",
                "/labels/custom",
                "--output-dir",
                "/runs/custom",
                "--max-steps",
                "2000",
            ]
        )
        paths = resolve(cli)
        self.assertEqual(paths["dataset"], Path("/datasets/custom"))
        self.assertEqual(paths["output"], Path("/runs/custom"))
        from rsci.paths import worker_arguments

        forwarded = worker_arguments(cli, parser())
        self.assertIn("--dataset", forwarded)
        self.assertIn("/datasets/custom", forwarded)
        self.assertIn("/runs/custom", forwarded)
        self.assertEqual(forwarded[forwarded.index("--max-steps") + 1], "2000")
        from transformers import TrainingArguments

        def cpu_arguments(**kwargs):
            return TrainingArguments(**{**kwargs, "use_cpu": True, "bf16": False, "tf32": False})

        cpu_arguments.__dataclass_fields__ = TrainingArguments.__dataclass_fields__
        with patch("transformers.TrainingArguments", new=cpu_arguments):
            self.assertEqual(training_arguments(cli, paths).max_steps, 2000)

    def test_real_optimizer_and_schedule_match_official_experiment(self):
        native = TrainingConfig()
        with tempfile.TemporaryDirectory() as directory:
            cli = parser().parse_args(
                [
                    "--dataset",
                    "/datasets/test",
                    "--sidecar",
                    "/labels/test",
                    "--task",
                    "hug",
                    "--batch-size",
                    "8",
                    "--gradient-accumulation-steps",
                    "2",
                    "--num-workers",
                    "0",
                ]
            )
            with patch(
                "transformers.TrainingArguments",
                side_effect=lambda **kwargs: TrainingArguments(
                    **{**kwargs, "use_cpu": True, "bf16": False, "tf32": False}
                ),
            ) as factory:
                factory.__dataclass_fields__ = TrainingArguments.__dataclass_fields__
                actual = training_arguments(cli, {"output": Path(directory) / "actual"})
            # 仅测试设备/精度选择用CPU；实际生产优化参数完全保留。
            actual = dataclasses.replace(actual, use_cpu=True, bf16=False, tf32=False)
            expected = TrainingArguments(
                output_dir=directory + "/official",
                use_cpu=True,
                report_to="none",
                max_steps=20000,
                learning_rate=native.learning_rate,
                lr_scheduler_type=native.lr_scheduler_type,
                weight_decay=native.weight_decay,
                warmup_ratio=native.warmup_ratio,
                max_grad_norm=native.max_grad_norm,
                optim=native.optim,
            )
            for key in (
                "learning_rate",
                "weight_decay",
                "adam_beta1",
                "adam_beta2",
                "adam_epsilon",
                "optim",
            ):
                self.assertEqual(getattr(actual, key), getattr(expected, key))
            self.assertEqual(actual.max_steps, 20000)
            self.assertEqual(actual.get_warmup_steps(20000), 1000)
            self.assertEqual(actual.gradient_accumulation_steps, 2)
            model = torch.nn.Linear(3, 2)
            a = Trainer(model=copy.deepcopy(model), args=actual)
            b = Trainer(model=copy.deepcopy(model), args=expected)
            for trainer in (a, b):
                trainer.create_optimizer_and_scheduler(20000)
            for left, right in zip(a.optimizer.param_groups, b.optimizer.param_groups):
                for key in ("betas", "eps", "weight_decay", "lr"):
                    self.assertEqual(left[key], right[key])
            for step in (0, 1, 999, 10000, 19999, 20000):
                self.assertEqual(
                    a.lr_scheduler.lr_lambdas[0](step), b.lr_scheduler.lr_lambdas[0](step)
                )
            self.assertAlmostEqual(a.lr_scheduler.lr_lambdas[0](1000), 1.0)

    def test_new_output_namespace_preserves_original_runs(self):
        cli = parser().parse_args(
            ["--dataset", "/datasets/test", "--sidecar", "/labels/test", "--task", "between"]
        )
        path = resolve(cli)["output"]
        self.assertIn("runs/gr00t/between", str(path))
        self.assertEqual(cli.run_tag, "main")


if __name__ == "__main__":
    unittest.main()
