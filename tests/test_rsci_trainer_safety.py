"""使用真实FinetuneTrainer更新/保存/恢复；微型模型只替换昂贵的VLA前向。"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/train")]
from psi.trainers.trainer import Trainer
from rsci.backbones.psi0 import train as launch_rsci
from rsci.backbones.psi0.trainer import RSCIFinetuneTrainer


class Samples(Dataset):
    def __len__(self):
        return 43

    def __getitem__(self, index):
        return dict(
            input_ids=torch.tensor([index + 1]),
            pixel_values=torch.tensor([index / 100.0]),
            image_grid_thw=torch.tensor([[1, 1, 1]]),
        )


def tiny_forward(self, model, batch, *args, **kwargs):
    loss = model(batch["pixel_values"]).square().mean()
    if getattr(self, "inject_nan", False):
        loss = loss * float("nan")
    return {"loss": loss}


class TinyTrainer(RSCIFinetuneTrainer):
    forward_and_loss = tiny_forward


class TrainerTests(unittest.TestCase):
    def setup_trainer(self, directory, resume=None):
        args = launch_rsci.parser().parse_args(
            ["--model", "VGQ-AE", "--task", "Test-v0", "--gpus", "0"]
        )
        cfg = launch_rsci.resolved(args)[0]
        cfg.train.output_dir = directory
        cfg.train.mixed_precision = "no"
        cfg.train.gradient_accumulation_steps = 2
        cfg.train.train_batch_size = 4
        cfg.train.val_batch_size = 4
        cfg.train.num_workers = cfg.train.val_num_workers = 2
        cfg.train.optimizer_fused = cfg.train.pin_memory = cfg.train.non_blocking_transfer = False
        cfg.train.resume_from_checkpoint = resume
        cfg.log.report_to = None
        trainer = TinyTrainer.__new__(TinyTrainer)
        Trainer.__init__(trainer, cfg, torch.device("cpu"))
        trainer.noise_scheduler_name = cfg.model.noise_scheduler
        trainer.train_diffusion_steps = cfg.model.train_diffusion_steps
        trainer.tokenizer = SimpleNamespace(model_max_length=32, pad_token_id=0)
        trainer.model = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
        trainer.create_dataloaders(Samples(), Samples())
        trainer.prepare(Accelerator(cpu=True, gradient_accumulation_steps=2))
        trainer.resume_from_checkpoint()
        return trainer

    def shutdown(self, trainer):
        for loader in (trainer.train_dataloader, trainer.val_dataloader):
            iterator = getattr(loader, "_iterator", None)
            if iterator:
                iterator._shutdown_workers()

    def test_real_update_checkpoint_and_resume_match(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.setup_trainer(directory)
            first.next_epoch(0)
            iterator = iter(first.train_dataloader)
            for local in range(2):
                sync, _ = first.step(next(iterator), 0, local)
            self.assertTrue(sync)
            checkpoint = first.save_checkpoint(1)
            self.assertTrue((Path(checkpoint) / "custom_checkpoint_0.pkl").is_file())
            expected_ids = []
            for local in range(2, 4):
                batch = next(iterator)
                expected_ids.extend(batch["input_ids"][:, 0].tolist())
                first.step(batch, 1, local)
            expected = copy.deepcopy(first.accelerator.unwrap_model(first.model).state_dict())
            self.shutdown(first)
            second = self.setup_trainer(directory, checkpoint)
            self.assertEqual(second._data_progress.consumed, 2)
            second.next_epoch(0)
            resumed = iter(second.train_dataloader)
            for _ in range(2):
                next(resumed)
            actual_ids = []
            for local in range(2, 4):
                batch = next(resumed)
                actual_ids.extend(batch["input_ids"][:, 0].tolist())
                second.step(batch, 1, local)
            self.assertEqual(expected_ids, actual_ids)
            for key, value in expected.items():
                torch.testing.assert_close(
                    value,
                    second.accelerator.unwrap_model(second.model).state_dict()[key],
                    rtol=0,
                    atol=0,
                )
            self.shutdown(second)

    def test_nan_loss_and_inf_gradient_do_not_update(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.setup_trainer(directory)
            trainer.next_epoch(0)
            batch = next(iter(trainer.train_dataloader))
            initial = copy.deepcopy(trainer.accelerator.unwrap_model(trainer.model).state_dict())
            trainer.inject_nan = True
            with self.assertRaises(FloatingPointError):
                trainer.step(batch, 0, 0)
            trainer.inject_nan = False
            hook = next(trainer.model.parameters()).register_hook(lambda grad: grad * float("inf"))
            with self.assertRaises(FloatingPointError):
                trainer.step(batch, 0, 0)
            hook.remove()
            for key, value in initial.items():
                torch.testing.assert_close(
                    value, trainer.accelerator.unwrap_model(trainer.model).state_dict()[key]
                )
            self.shutdown(trainer)

    def test_bad_memory_is_rejected_before_any_checkpoint_write(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.setup_trainer(directory)
            with torch.no_grad():
                next(trainer.model.parameters()).fill_(float("nan"))
            with self.assertRaises(FloatingPointError):
                trainer.save_checkpoint(1)
            self.assertFalse(list(Path(directory).rglob("model.safetensors")))
            self.assertTrue((Path(directory) / "NUMERICAL_FAILURE.json").is_file())
            self.shutdown(trainer)


if __name__ == "__main__":
    unittest.main()
