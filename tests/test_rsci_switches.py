"""Experimental switches: full-token memory, future supervision and raw state noise."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from test_rsci import inputs, policy, sample

from psi.config.transform import ActionStateTransform
from psi.data.lerobot.lerobot_ext import LeRobotDatasetWrapper
from psi.trainers.finetune import FinetuneTrainer
from rsci.backbones.psi0.model import RSCIQuery, RSCISpec, masked_auxiliary_loss
from rsci.backbones.psi0.train import contract, parser, resolved
from rsci.data.ardy_future import ardy_target_at_offset, supervision_counts
from rsci.data.ardy_sidecar import ArdySidecarReader
from rsci.data.state_noise import InferenceStateNoise, TrainingStateNoise, add_measurement_noise


def reader():
    result = ArdySidecarReader.__new__(ArdySidecarReader)
    result._episodes = {}
    for episode, length in ((0, 42), (1, 10)):
        result._episodes[episode] = dict(
            ardy_body_latent=np.broadcast_to(
                (1000 * episode + np.arange(length, dtype=np.float32))[:, None], (length, 128)
            ).copy(),
            teacher_valid=np.ones(length, dtype=bool),
            visual_history_valid=np.arange(length) >= 6,
            ardy_loss_mask=np.arange(length) >= 6,
            raw_episode_index=episode,
            raw_frame_index=np.arange(length),
        )
    return result


class ReaderSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_token_memory_preserves_masks_and_has_no_reader_parameters(self):
        query = RSCIQuery(RSCISpec("VGQ-AE"), 8, 8, 2, use_reader=False)
        frames = [torch.randn(2, count, 8, requires_grad=True) for count in (3, 4, 2, 5)]
        masks = [torch.ones(2, frame.shape[1], dtype=torch.bool) for frame in frames]
        masks[-1][1, -2:] = False
        valid = torch.tensor([[False, True, True, True], [True, True, True, True]])
        memory, mask = query.memory(frames, masks, valid)
        self.assertEqual(tuple(memory.shape), (2, 14, 8))
        self.assertFalse(mask[0, :3].any())
        self.assertFalse(mask[1, -2:].any())
        self.assertTrue(torch.equal(memory[~mask], torch.zeros_like(memory[~mask])))
        self.assertFalse(any("reader" in name for name, _ in query.named_parameters()))
        memory.sum().backward()
        self.assertEqual(frames[0].grad[0].abs().sum().item(), 0)
        self.assertGreater(frames[0].grad[1].abs().sum().item(), 0)
        self.assertEqual(frames[-1].grad[1, -2:].abs().sum().item(), 0)

    def test_all_query_variants_backpropagate_without_reader(self):
        for variant in ("VGQ-AE", "EGQ-AE", "VGQ-VLM", "EGQ-VLM", "VGQ-VLM*", "EGQ-VLM*"):
            with self.subTest(variant=variant):
                _, model = policy(
                    variant, mode="lora" if "VLM" in variant else "frozen", use_reader=False
                )
                value = sample()
                output = model(**inputs(value))
                loss = masked_auxiliary_loss(
                    output.ardy_prediction, value["ardy_target"], value["ardy_valid"]
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertGreater(model.action_header.rsci.state_query.grad.abs().sum().item(), 0)

    def test_reader_checkpoint_cannot_silently_load_into_no_reader(self):
        with_reader = RSCIQuery(RSCISpec("VGQ-AE"), 8, 8, 2)
        without = RSCIQuery(RSCISpec("VGQ-AE"), 8, 8, 2, use_reader=False)
        self.assertEqual(with_reader.schema_version.item(), 1)
        self.assertEqual(without.schema_version.item(), 2)
        with self.assertRaises(RuntimeError):
            without.load_state_dict(with_reader.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            with_reader.load_state_dict(without.state_dict(), strict=True)


class FutureArdyTests(unittest.TestCase):
    def test_exact_targets_teacher_masks_and_episode_tail(self):
        source = reader()
        for offset in (0, 1, 10, 30):
            target, valid = ardy_target_at_offset(source, 0, 6, offset)
            np.testing.assert_array_equal(
                target, source._episodes[0]["ardy_body_latent"][6 + offset]
            )
            self.assertTrue(valid)
            target[:] = -1
            self.assertTrue((source._episodes[0]["ardy_body_latent"][6 + offset] >= 0).all())
        for episode, frame, offset in ((0, 41, 1), (0, 12, 30), (1, 6, 30)):
            target, valid = ardy_target_at_offset(source, episode, frame, offset)
            self.assertFalse(valid)
            np.testing.assert_array_equal(target, np.zeros(128, np.float32))
        source._episodes[0]["ardy_loss_mask"][36] = False
        self.assertFalse(ardy_target_at_offset(source, 0, 6, 30)[1])
        self.assertEqual(
            supervision_counts(source, 30),
            dict(
                total_frames=52,
                valid_ardy_frames=5,
                tail_masked_frames=40,
                offset_frames=30,
            ),
        )
        for offset in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                ardy_target_at_offset(source, 0, 6, offset)

    def test_wrapper_changes_only_labels_and_keeps_tail_action_samples(self):
        samples = [
            dict(
                episode_index=0,
                frame_index=frame,
                action=torch.randn(3, 4),
                states=torch.ones(1, 32),
                rgb=torch.randn(4, 3, 2, 2),
            )
            for frame in (6, 41)
        ]
        wrapper = LeRobotDatasetWrapper.__new__(LeRobotDatasetWrapper)
        wrapper.base_dataset = copy.deepcopy(samples)
        wrapper.ardy_reader, wrapper.ardy_offset = reader(), 10
        for index, original in enumerate(samples):
            actual = wrapper[index]
            for key, value in original.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual[key], value)
                else:
                    self.assertEqual(actual[key], value)
            self.assertEqual(actual["ardy_valid"], index == 0)
        self.assertEqual(len(wrapper), 2)


class RawStateNoiseTests(unittest.TestCase):
    def test_training_rank_is_forwarded_and_validation_noise_is_disabled(self):
        field = ActionStateTransform(
            stat_path="/nonexistent", normalize_state=True, state_measurement_noise_std_rad=0.02
        )

        class Data:
            transform = SimpleNamespace(field=field, repack=SimpleNamespace(rsci_variant="STATE"))

            def __call__(self, split, transform_kwargs):
                return SimpleNamespace(split=split, transform_kwargs=transform_kwargs)

        trainer = FinetuneTrainer.__new__(FinetuneTrainer)
        trainer.cfg = SimpleNamespace(
            model=SimpleNamespace(rsci_variant="STATE", state_measurement_noise_std_rad=0.02),
            data=Data(),
        )
        trainer.accelerator = SimpleNamespace(process_index=3)
        trainer.vlm_processor = None
        train, validation = trainer.create_datasets()
        self.assertEqual(train.transform_kwargs["state_noise_rank"], 3)
        self.assertFalse(validation.transform_kwargs["state_measurement_noise"])
        self.assertNotIn("state_measurement_noise", train.transform_kwargs)
        trainer.cfg.model.state_measurement_noise_std_rad = 0.1
        with self.assertRaisesRegex(ValueError, "must match"):
            trainer.create_datasets()

    def test_raw_noise_precedes_normalization_and_preserves_height_and_padding(self):
        field = ActionStateTransform(
            stat_path="/nonexistent",
            normalize_state=True,
            state_min=[-2.0] * 36,
            state_max=[2.0] * 36,
            action_min=[-1.0] * 36,
            action_max=[1.0] * 36,
            state_measurement_noise_std_rad=0.02,
        )

        class Ones:
            def sample(self, reference, rank):
                return torch.ones_like(reference)

        field._state_noise = Ones()
        raw = np.zeros((1, 36), np.float32)
        raw[0, 31:] = [0.74, 0.0, 0.0, 0.0, 0.0]
        value = dict(states=raw.copy(), actions=np.zeros((1, 36), np.float32))
        actual = field(copy.deepcopy(value))["states"]
        np.testing.assert_allclose(actual[0, :31], 0.01, atol=1e-7)
        np.testing.assert_array_equal(actual[0, 31:], field.normalize_state_func(raw)[0, 31:])
        for kwargs in (dict(no_aug=True), dict(state_measurement_noise=False)):
            np.testing.assert_array_equal(
                field(copy.deepcopy(value), **kwargs)["states"], field.normalize_state_func(raw)
            )
        np.testing.assert_array_equal(value["states"], raw)

    def test_private_streams_preserve_action_rng_and_separate_ranks(self):
        reference = torch.zeros(2, 31)
        rng = torch.get_rng_state().clone()
        stream = TrainingStateNoise()
        left = stream.sample(reference, 0)
        right = stream.sample(reference, 1)
        self.assertFalse(torch.equal(left, right))
        torch.testing.assert_close(left, TrainingStateNoise().sample(reference, 0), rtol=0, atol=0)
        inference = InferenceStateNoise()
        torch.testing.assert_close(
            inference.sample(reference, 73), inference.sample(reference, 73), rtol=0, atol=0
        )
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        values = torch.zeros(2, 36)
        values[:, 31] = 0.74
        noisy = add_measurement_noise(values, inference.sample(reference, 73), 0.02)
        torch.testing.assert_close(noisy[:, 31:], values[:, 31:], rtol=0, atol=0)

    def test_launch_switches_are_persisted_and_separate_run_contracts(self):
        base = ["--task", "example", "--model", "VGQ-AE"]
        identities, outputs = [], []
        for extra in (
            [],
            ["--no-reader"],
            ["--ardy-offset", "10"],
            ["--no-reader", "--ardy-offset", "10"],
        ):
            cfg, command, output, ranks = resolved(parser().parse_args(base + extra))
            self.assertEqual(cfg.model.rsci_use_reader, "--no-reader" not in extra)
            self.assertEqual(
                cfg.data.rsci_ardy_offset_frames, 10 if "--ardy-offset" in extra else 0
            )
            identities.append(contract(cfg, ranks))
            outputs.append(output)
        self.assertEqual(len(set(outputs)), 4)
        self.assertEqual(len({str(value) for value in identities}), 4)
        cfg, _, _, _ = resolved(
            parser().parse_args(
                [
                    "--task",
                    "example",
                    "--model",
                    "STATE",
                    "--state-measurement-noise-std-rad",
                ]
            )
        )
        self.assertEqual(cfg.model.state_measurement_noise_std_rad, 0.02)
        self.assertEqual(cfg.data.transform.field.state_measurement_noise_std_rad, 0.02)
        for extra in (
            ["--no-reader"],
            ["--ardy-offset", "10"],
            ["--state-measurement-noise-std-rad", "nan"],
        ):
            with self.assertRaises(ValueError):
                resolved(parser().parse_args(["--task", "example", "--model", "STATE", *extra]))


if __name__ == "__main__":
    unittest.main()
