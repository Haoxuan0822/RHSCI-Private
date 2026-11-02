"""用实际 Trainer 和完整断点比较跨 epoch 的样本、随机数与优化状态。"""

import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import TrainerCallback, TrainingArguments, set_seed

from openpi.rhsci.checkpoints import checkpoint_ok
from openpi.rhsci.trainer import build_checkpoint_callback, create_trainer_class


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3))
        self.rsci_config = {"variant": "resume-test"}
        self.samples = []
        self.random_values = []

    def forward(self, index):
        self.samples.extend(index.tolist())
        noise = torch.rand(3) + random.random() + np.random.random()
        self.random_values.append(noise.clone())
        loss = ((self.weight * noise - index.float().mean() / 19) ** 2).mean()
        return {"loss": loss, "action_loss": loss}


class StopAtStep(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_save = True
            control.should_training_stop = True


class ResumeProgressTests(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right), (left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_complete_resume_matches_uninterrupted_later_epoch(self):
        for workers, stops in ((0, [13]), (2, [13, 21]), (0, [20])):
            with self.subTest(workers=workers, stops=stops):
                self.compare_resume(workers, stops)

    def test_accumulated_epoch_tail_matches_after_resume(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                self.compare_resume(workers, [4], accumulation=4, epochs=3, max_steps=-1)

    def compare_resume(self, workers, stops, accumulation=1, epochs=3, max_steps=24):
        with tempfile.TemporaryDirectory() as directory:

            def run(name, stop=None, resume=None):
                set_seed(42)
                model = RecordingModel()
                output = Path(directory) / name
                args = TrainingArguments(
                    output_dir=str(output),
                    max_steps=max_steps,
                    num_train_epochs=epochs,
                    gradient_accumulation_steps=accumulation,
                    per_device_train_batch_size=2,
                    learning_rate=1e-4,
                    seed=42,
                    data_seed=42,
                    use_cpu=True,
                    save_strategy="no",
                    report_to="none",
                    disable_tqdm=True,
                    remove_unused_columns=False,
                    dataloader_pin_memory=False,
                    dataloader_num_workers=workers,
                    dataloader_persistent_workers=workers > 0,
                )
                callbacks = [build_checkpoint_callback(output, 1, "resume-progress")]
                if stop is not None:
                    callbacks.append(StopAtStep(stop))
                trainer = create_trainer_class()(
                    model=model,
                    args=args,
                    train_dataset=[{"index": index} for index in range(19)],
                    callbacks=callbacks,
                )
                trainer.train(resume_from_checkpoint=str(resume) if resume else None)
                states = (
                    trainer.state.global_step,
                    copy.deepcopy(model.state_dict()),
                    copy.deepcopy(trainer.optimizer.state_dict()),
                    copy.deepcopy(trainer.lr_scheduler.state_dict()),
                    (torch.get_rng_state().clone(), random.getstate(), np.random.get_state()),
                )
                return model, states, output

            full, expected, _ = run("full")
            self.assertEqual(expected[0], 9 if accumulation == 4 else max_steps)
            checkpoint = None
            samples, noise = [], []
            for stop in [*stops, None]:
                partial, actual, output = run("partial", stop=stop, resume=checkpoint)
                samples.extend(partial.samples)
                noise.extend(partial.random_values)
                if stop is not None:
                    checkpoint = output / f"checkpoint-{stop}"
                    self.assertTrue(
                        checkpoint_ok(checkpoint, 1, "resume-progress", verify_hash=True)
                    )
            self.assertEqual(full.samples, samples)
            self.assert_nested_equal(full.random_values, noise)
            self.assert_nested_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
