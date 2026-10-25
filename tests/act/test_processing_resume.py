import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from rsci.backbones.act.data import IMAGE, ACTDataset, Processor
from rsci.backbones.act.train import parser, resolve, trainer_type, training_arguments


def stats():
    return {
        "action": {"min": [-2.0] * 36, "max": [2.0] * 36},
        "states": {"min": [-2.0] * 32, "max": [2.0] * 32},
    }


class Processing(unittest.TestCase):
    def test_zero_worker_sample_rng_preserves_model_stream_and_never_seeds_cuda(self):
        dataset = ACTDataset.__new__(ACTDataset)
        dataset.native = [dict(index=i) for i in range(3)]
        dataset.variant, dataset.seed, dataset.training = "STATE", 2026, True
        dataset.processor = lambda raw, variant, training: {"x": torch.rand(8)}
        before = torch.get_rng_state().clone()
        with patch("torch.cuda.manual_seed_all") as cuda_seed:
            first = dataset[(2, 1)]
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            torch.rand(17)
            after = torch.get_rng_state().clone()
            second = dataset[(2, 1)]
            torch.testing.assert_close(first["x"], second["x"], rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), after, rtol=0, atol=0)
            self.assertFalse(torch.equal(first["x"], dataset[(3, 1)]["x"]))
            cuda_seed.assert_not_called()

    def test_explicit_data_paths_and_state_without_sidecar(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ):
            args = parser().parse_args(
                [
                    "--task",
                    "custom",
                    "--model",
                    "STATE",
                    "--dataset",
                    temp,
                    "--output-dir",
                    temp + "/run",
                    "--torch-home",
                    temp + "/torch",
                ]
            )
            paths = resolve(args)
            self.assertEqual(paths["dataset"], Path(temp))
            self.assertIsNone(paths["sidecar"])
            self.assertEqual(paths["output"], Path(temp) / "run")
            self.assertEqual(os.environ["TORCH_HOME"], temp + "/torch")
            args.model = "EGQ-AE"
            with self.assertRaisesRegex(ValueError, "--sidecar"):
                resolve(args)

    def test_current_native_exact_shared_history_augmentation_and_inference(self):
        processor = Processor(stats())
        image = torch.rand(3, 240, 320)
        raw = {
            IMAGE: image[None].repeat(4, 1, 1, 1),
            "states": torch.randn(1, 32),
            "action": torch.randn(16, 36),
            "action_is_pad": torch.tensor([False] * 10 + [True] * 6),
            "task": "pick",
        }
        torch.manual_seed(23)
        sample = processor(raw, "EGQ-AE")
        torch.manual_seed(23)
        expected = processor.image(processor.field(processor.repack({**raw, IMAGE: image})))
        torch.testing.assert_close(
            sample["observation.images"], expected["observation.images"], rtol=0, atol=0
        )
        for frame in sample["history_images"]:
            torch.testing.assert_close(frame, sample["observation.images"][0], rtol=0, atol=0)
        self.assertNotIn("observation.state", sample)
        np.testing.assert_array_equal(sample["action_is_pad"], raw["action_is_pad"])
        from psi.utils import pt_to_pil

        state_sample = processor({**raw, IMAGE: image}, "STATE", training=False)
        inference = processor.inference(pt_to_pil(image, normalized=False), raw["states"].numpy())
        torch.testing.assert_close(
            inference["observation.images"][0], state_sample["observation.images"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            inference["observation.state"][0], state_sample["observation.state"], rtol=0, atol=0
        )
        np.testing.assert_allclose(
            processor.field.denormalize(sample["action"]), np.clip(raw["action"], -2, 2), atol=2e-7
        )

    def test_degenerate_bounds_finite_and_constant_state_zero(self):
        values = stats()
        values["action"]["min"][2] = values["action"]["max"][2] = 0
        values["states"]["min"][3] = values["states"]["max"][3] = 0.75
        processor = Processor(values)
        state = processor.field.normalize_state_func(np.ones(36, np.float32))
        self.assertEqual(state[3], 0)
        self.assertTrue(np.isfinite(state).all())
        actions = processor.field(
            {"actions": np.zeros((16, 36), np.float32), "states": np.ones((1, 36), np.float32)}
        )
        self.assertTrue(np.isfinite(actions["actions"]).all())


class Synthetic(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, index):
        epoch, index = index if isinstance(index, tuple) else (0, index)
        # 与生产数据相同的(epoch,index)增强随机性，worker预取不改变结果。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(3 + epoch * len(self) + index)
            return {
                "x": torch.randn(3),
                "y": torch.randn(2),
                "sample_id": torch.tensor(epoch * 100 + index),
            }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.dropout = torch.nn.Dropout(0.2)
        self.register_buffer("frozen_buffer", torch.tensor([7.0]))
        self.rsci_config = {"test": "resume-v1"}
        self.trace = []

    def forward(self, x, y, sample_id, **unused):
        self.trace.extend(sample_id.tolist())
        return {"loss": (self.linear(self.dropout(x)) - y).square().mean()}


def assert_state_equal(test, first, second):
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        test.assertEqual(set(first), set(second))
        for key in first:
            assert_state_equal(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_state_equal(test, a, b)
    else:
        test.assertEqual(first, second)


class Resume(unittest.TestCase):
    def test_real_hf_second_epoch_accumulation_four_exact_resume(self):
        self.run_resume(0)

    def test_persistent_workers_second_epoch_resume(self):
        self.run_resume(2)

    def run_resume(self, workers):
        from transformers import TrainerCallback, set_seed

        from rsci.backbones.act.checkpoints import checkpoint_ok, publish

        class StopAtFive(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 5:
                    control.should_save = True
                    control.should_training_stop = True

        args = parser().parse_args(
            [
                "--task",
                "hug",
                "--batch-size",
                "2",
                "--gradient-accumulation-steps",
                "4",
                "--num-workers",
                str(workers),
                "--max-steps",
                "8",
            ]
        )
        with tempfile.TemporaryDirectory() as temp:

            def build(name, callback=None):
                set_seed(55)
                arguments = training_arguments(args, {"output": Path(temp) / name}, cpu=True)
                arguments.eval_strategy = "no"
                arguments.disable_tqdm = True
                return trainer_type()(
                    model=TinyModel(),
                    args=arguments,
                    train_dataset=Synthetic(),
                    launch_contract={"test": "identical"},
                    callbacks=[callback] if callback else [],
                )

            continuous = build("continuous")
            continuous.train()
            interrupted = build("resume", StopAtFive())
            interrupted.train()
            self.assertEqual(interrupted.state.global_step, 5)
            path = Path(temp) / "resume/checkpoint-5"
            publish(path, 1, "test-digest")
            self.assertTrue(checkpoint_ok(path, 1, "test-digest", verify=True))
            resumed = build("resume")
            resumed.train(resume_from_checkpoint=str(path))
            self.assertEqual(continuous.model.trace, interrupted.model.trace + resumed.model.trace)
            assert_state_equal(self, continuous.model.state_dict(), resumed.model.state_dict())
            assert_state_equal(
                self, continuous.optimizer.state_dict(), resumed.optimizer.state_dict()
            )
            assert_state_equal(
                self, continuous.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict()
            )
            self.assertEqual(resumed.model.frozen_buffer.item(), 7.0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
