"""真实HF循环：不足累计组的完整epoch及跨epoch中途精确恢复。"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from human_plan.rsci.train import parser, trainer_type, training_arguments
from torch import nn
from transformers import set_seed


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.rsci_config = {"test": "partial-epoch"}

    def forward(self, x):
        prediction = nn.functional.dropout(self.linear(x), 0.2, self.training)
        loss = (prediction - x.square() * 0.01).square().mean()
        return {"loss": loss, "action_loss": loss.detach()}


class EpochTailTests(unittest.TestCase):
    @patch("human_plan.rsci.checkpoints.prune", lambda *a: None)
    def test_native_epoch_tail_and_mid_epoch_resume(self):
        torch.set_num_threads(1)

        class Recording(trainer_type()):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.seen = []

            def training_step(self, model, inputs, *a, **kw):
                self.seen.extend(inputs["x"].flatten().tolist())
                return super().training_step(model, inputs, *a, **kw)

        data = [{"x": torch.tensor([float(i)])} for i in range(9)]
        args = parser().parse_args(
            [
                "--task",
                "between",
                "--batch-size",
                "1",
                "--gradient-accumulation-steps",
                "4",
                "--num-train-epochs",
                "3",
                "--num-workers",
                "0",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            set_seed(21)
            initial = TinyPolicy()
            options = training_arguments(args, {"output": root / "continuous"}, cpu=True)
            options.gradient_checkpointing = False  # 微型线性模型无Transformer重算层。
            options.save_steps, options.disable_tqdm = 1, True
            first = Recording(
                model=copy.deepcopy(initial),
                args=options,
                train_dataset=data,
                launch_contract={"test": "partial-epoch"},
            )
            first.train()
            self.assertEqual(first.state.global_step, 9)  # ceil(9/4) × 3
            self.assertEqual(len(first.seen), 27)
            for offset in (0, 9, 18):
                self.assertEqual(sorted(first.seen[offset : offset + 9]), list(range(9)))
            expected_rng = torch.get_rng_state().clone()
            resumed_options = copy.deepcopy(options)
            resumed_options.output_dir = str(root / "resumed")
            second = Recording(
                model=copy.deepcopy(initial),
                args=resumed_options,
                train_dataset=data,
                launch_contract={"test": "partial-epoch"},
            )
            second.train(resume_from_checkpoint=str(root / "continuous/checkpoint-4"))
            self.assertEqual(second.seen, first.seen[13:])  # 第一epoch9批＋第二epoch4批
            self.assertEqual(second.state.global_step, 9)
            for name, parameter in second.model.named_parameters():
                torch.testing.assert_close(
                    parameter, dict(first.model.named_parameters())[name], rtol=0, atol=0
                )
            torch.testing.assert_close(
                second.optimizer.state_dict(), first.optimizer.state_dict(), rtol=0, atol=0
            )
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
