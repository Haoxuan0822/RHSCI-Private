"""生产恢复循环跨epoch、梯度累计与连续训练逐元素一致。"""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments, set_seed

from gr00t.rhsci.trainer import create_resume_trainer_class


class Samples(Dataset):
    def __len__(self):
        return 19

    def __getitem__(self, identity):
        epoch, index = identity
        return dict(
            x=torch.tensor([float(index), float(epoch)]), labels=torch.tensor([index / 19.0])
        )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 8), torch.nn.Dropout(0.1), torch.nn.Linear(8, 1)
        )
        self.seen = []

    def forward(self, x, labels):
        self.seen.extend(x.detach().tolist())
        return dict(loss=(self.net(x) - labels).square().mean())


class StopAtStep(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_training_stop = True


class ResumeTests(unittest.TestCase):
    def test_production_trainer_finite_loss_gate(self):
        from gr00t.rhsci.trainer import create_trainer_class

        class LossModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, x):
                loss = self.weight * x
                return dict(loss=loss, action_loss=loss, action_mask=torch.ones_like(loss))

        with tempfile.TemporaryDirectory() as tmp:
            model = LossModel()
            args = TrainingArguments(output_dir=tmp, use_cpu=True, report_to="none")
            trainer = create_trainer_class()(model=model, args=args)
            loss = trainer.compute_loss(model, dict(x=torch.tensor(2.0)))
            loss.backward()
            torch.testing.assert_close(model.weight.grad, torch.tensor(2.0))
            with self.assertRaises(FloatingPointError):
                trainer.compute_loss(model, dict(x=torch.tensor(float("nan"))))
            self.assertTrue((Path(tmp) / "NUMERICAL_FAILURE.json").is_file())

    def test_mid_second_epoch_accumulation_resume_matches_continuous(self):
        self.compare_resume(accumulation=2, max_steps=12, stop=7, workers=0)

    def test_partial_accumulation_tail_resume_matches_continuous(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                self.compare_resume(accumulation=4, max_steps=9, stop=4, workers=workers)

    def test_epoch_budget_includes_partial_accumulation_group(self):
        self.compare_resume(accumulation=4, max_steps=-1, stop=4, workers=2)

    def compare_resume(self, *, accumulation, max_steps, stop, workers):
        torch.set_num_threads(1)
        expected_steps = 9 if max_steps < 0 else max_steps
        with tempfile.TemporaryDirectory() as temporary:

            def trainer(name, callbacks=()):
                set_seed(292285)
                arguments = TrainingArguments(
                    output_dir=str(Path(temporary) / name),
                    use_cpu=True,
                    max_steps=max_steps,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=accumulation,
                    learning_rate=1e-3,
                    save_steps=stop,
                    report_to="none",
                    disable_tqdm=True,
                    seed=292285,
                    dataloader_num_workers=workers,
                    dataloader_persistent_workers=workers > 0,
                    remove_unused_columns=False,
                    save_safetensors=True,
                )
                result = create_resume_trainer_class()(
                    model=Model(),
                    args=arguments,
                    train_dataset=Samples(),
                    callbacks=list(callbacks),
                )
                result.model_accepts_loss_kwargs = False
                return result

            full = trainer("full")
            full.train()
            self.assertEqual(full.state.global_step, expected_steps)
            expected_rng = torch.get_rng_state().clone()
            interrupted = trainer("resume", [StopAtStep(stop)])
            interrupted.train()
            self.assertEqual(interrupted.state.global_step, stop)
            resumed = trainer("resume")
            resumed.train(resume_from_checkpoint=str(Path(temporary) / f"resume/checkpoint-{stop}"))
            self.assertEqual(resumed.state.global_step, expected_steps)
            self.assertEqual(full.model.seen, interrupted.model.seen + resumed.model.seen)
            for name, parameter in full.model.state_dict().items():
                self.assertTrue(torch.equal(parameter, resumed.model.state_dict()[name]), name)
            torch.testing.assert_close(
                full.optimizer.state_dict(), resumed.optimizer.state_dict(), rtol=0, atol=0
            )
            self.assertEqual(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
