"""直接验证正式训练入口创建的优化器/调度器及默认编译配置。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_pi05_rsci import TinyBackbone

from rsci.backbones.pi05.model import Pi05RSCI
from rsci.backbones.pi05.train import parser, resolve, trainer_class, training_arguments


class OfficialRecipeTests(unittest.TestCase):
    def test_custom_dataset_output_and_step_budget(self):
        args = parser().parse_args(
            [
                "--task",
                "custom",
                "--dataset",
                "/datasets/custom",
                "--sidecar",
                "/labels/custom",
                "--output-dir",
                "/runs/custom",
                "--tokenizer",
                "/models/tokenizer.model",
                "--max-steps",
                "2000",
            ]
        )
        paths = resolve(args)
        self.assertEqual(paths["dataset"], Path("/datasets/custom"))
        self.assertEqual(paths["output"], Path("/runs/custom"))
        self.assertEqual(paths["tokenizer"], Path("/models/tokenizer.model"))
        from transformers import TrainingArguments

        with patch(
            "transformers.TrainingArguments",
            side_effect=lambda **kwargs: TrainingArguments(
                **{**kwargs, "use_cpu": True, "bf16": False, "tf32": False}
            ),
        ):
            self.assertEqual(training_arguments(args, paths).max_steps, 2000)
        from rsci.backbones.pi05.train import simple_lr_multiplier

        self.assertAlmostEqual(simple_lr_multiplier(2000, 2000), 1e-4)

    def test_actual_optimizer_and_schedule_match_simple_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parser().parse_args(
                [
                    "--dataset",
                    "/datasets/test",
                    "--sidecar",
                    "/labels/test",
                    "--task",
                    "between",
                    "--num-workers",
                    "0",
                ]
            )
            paths = resolve(args)
            self.assertEqual(args.run_tag, "main")
            self.assertIn("runs/pi05", str(paths["output"]))
            paths["output"] = Path(directory)
            from transformers import TrainingArguments

            with patch(
                "transformers.TrainingArguments",
                side_effect=lambda **kwargs: TrainingArguments(
                    **{**kwargs, "use_cpu": True, "bf16": False, "tf32": False}
                ),
            ):
                actual = training_arguments(args, paths)
            self.assertEqual(actual.max_steps, 40000)
            self.assertEqual(actual.per_device_train_batch_size * 8, 128)
            trainer = trainer_class()(model=torch.nn.Linear(3, 2), args=actual)
            trainer.create_optimizer_and_scheduler(40000)
            self.assertEqual(len(trainer.optimizer.param_groups), 1)
            group = trainer.optimizer.param_groups[0]
            self.assertEqual(group["betas"], (0.9, 0.95))
            self.assertEqual(group["eps"], 1e-8)
            self.assertEqual(group["weight_decay"], 1e-10)
            # 固定SIMPLE原生lr_schedule的起点/预热边界/末端，独立计算期望值。
            import math

            for step in (0, 1, 999, 1000, 20000, 39999, 40000):
                init = 1e-4 / 1001
                expected = (
                    init + (1e-4 - init) * step / 1000
                    if step < 1000
                    else 1e-8
                    + (1e-4 - 1e-8) * 0.5 * (1 + math.cos(math.pi * (step - 1000) / 39000))
                )
                self.assertAlmostEqual(
                    trainer.lr_scheduler.lr_lambdas[0](step) * 1e-4, expected, places=15
                )

    def test_default_sampling_compilation_and_explicit_opt_out(self):
        with patch("torch.compile", side_effect=lambda method, **kwargs: method) as compile_call:
            Pi05RSCI("NO-STATE", TinyBackbone(), vlm_width=64, ae_width=32)
            self.assertEqual(compile_call.call_args.kwargs, {"mode": "max-autotune"})
            compile_call.reset_mock()
            Pi05RSCI("NO-STATE", TinyBackbone(), vlm_width=64, ae_width=32, compile_inference=False)
            compile_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
