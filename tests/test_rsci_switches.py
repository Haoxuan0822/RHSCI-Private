"""Experimental switches: full-token memory, future supervision."""

import copy
import unittest

import numpy as np
import torch
from test_rsci import inputs, policy, sample

from psi.data.lerobot.lerobot_ext import LeRobotDatasetWrapper
from psi.rhsci.contract import build_training_contract
from psi.rhsci.model import RHSCIQuery, RHSCISpec, masked_auxiliary_loss
from psi.rhsci.training_config import build_argument_parser, resolve_training_plan
from rhsci.data.ardy_future import ardy_target_at_offset, supervision_counts
from rhsci.data.ardy_sidecar import ArdySidecarReader


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
        query = RHSCIQuery(RHSCISpec("VGQ-AE"), 8, 8, 2, use_reader=False)
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
        for variant in ("VGQ-AE", "EGQ-AE", "EGQ-VLM", "VGQ-VLM*", "EGQ-VLM*"):
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
        with_reader = RHSCIQuery(RHSCISpec("VGQ-AE"), 8, 8, 2)
        without = RHSCIQuery(RHSCISpec("VGQ-AE"), 8, 8, 2, use_reader=False)
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


class LaunchSwitchTests(unittest.TestCase):
    def test_training_measurement_noise_option_is_removed(self):
        with self.assertRaises(SystemExit):
            build_argument_parser().parse_args(
                ["--task", "example", "--model", "STATE", "--state-measurement-noise-std-rad", "0.02"]
            )

    def test_launch_switches_are_persisted_and_separate_run_contracts(self):
        base = ["--task", "example", "--model", "VGQ-AE"]
        identities, outputs = [], []
        for extra in (
            [],
            ["--no-reader"],
            ["--ardy-offset", "10"],
            ["--no-reader", "--ardy-offset", "10"],
        ):
            cfg, command, output, ranks = resolve_training_plan(
                build_argument_parser().parse_args(base + extra)
            )
            self.assertEqual(cfg.model.rsci_use_reader, "--no-reader" not in extra)
            self.assertEqual(
                cfg.data.rsci_ardy_offset_frames, 10 if "--ardy-offset" in extra else 0
            )
            identities.append(build_training_contract(cfg, ranks))
            outputs.append(output)
        self.assertEqual(len(set(outputs)), 4)
        self.assertEqual(len({str(value) for value in identities}), 4)
        for extra in (
            ["--no-reader"],
            ["--ardy-offset", "10"],
        ):
            with self.assertRaises(ValueError):
                resolve_training_plan(
                    build_argument_parser().parse_args(
                        ["--task", "example", "--model", "STATE", *extra]
                    )
                )


if __name__ == "__main__":
    unittest.main()
