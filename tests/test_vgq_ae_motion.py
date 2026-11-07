"""VGQ-AE-Mo: backward labels, shared Query gradients and unchanged action inference."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rhsci.data.ardy_motion import ardy_motion_target, motion_config
from rhsci.data.ardy_sidecar import ArdySidecarReader
from rhsci.runtime.losses import masked_ardy_loss


def sidecar(length=20):
    reader = ArdySidecarReader.__new__(ArdySidecarReader)
    reader._episodes = {}
    for episode in (0, 1):
        reader._episodes[episode] = dict(
            ardy_body_latent=np.broadcast_to(
                (1000 * episode + np.arange(length, dtype=np.float32) ** 2)[:, None], (length, 128)
            ).copy(),
            teacher_valid=np.ones(length, bool),
            visual_history_valid=np.arange(length) >= 6,
            ardy_loss_mask=np.arange(length) >= 6,
            raw_episode_index=episode,
            raw_frame_index=np.arange(length),
        )
    return reader


class LabelTests(unittest.TestCase):
    def test_all_five_launchers_accept_explicit_motion_settings(self):
        import importlib

        for package in ("psi", "openpi", "gr00t", "human_plan", "lambda0"):
            module = "training_config" if package == "psi" else "config"
            parser = importlib.import_module(f"{package}.rhsci.{module}").build_argument_parser()
            flag = "--variant" if package == "lambda0" else "--model"
            defaults = parser.parse_args(["--task", "fixture", flag, "VGQ-AE-Mo"])
            self.assertEqual(motion_config(defaults)["ardy_delta_frames"], 6)
            args = parser.parse_args(
                [
                    "--task",
                    "fixture",
                    flag,
                    "VGQ-AE-Mo",
                    "--ardy-delta-frames",
                    "4",
                    "--motion-weight",
                    "0.3",
                ]
            )
            self.assertEqual(motion_config(args), dict(ardy_delta_frames=4, motion_weight=0.3))
        with self.assertRaises(ValueError):
            motion_config(SimpleNamespace(model="VGQ-AE", ardy_delta_frames=4))

    def test_backward_difference_and_episode_boundary(self):
        reader = sidecar()
        self.assertFalse(ardy_motion_target(reader, 0, 11)[1])
        target, valid = ardy_motion_target(reader, 0, 12)
        self.assertTrue(valid)
        np.testing.assert_array_equal(target, np.full(128, 108, np.float32))
        for episode in (0, 1):
            target, valid = ardy_motion_target(reader, episode, 8, 2)
            self.assertTrue(valid)
            np.testing.assert_array_equal(target, np.full(128, 28, np.float32))
            target[:] = -1
            self.assertEqual(reader.get(episode, 8).ardy_body_latent[0], 1000 * episode + 64)
            for frame in (0, 1, 6, 7):
                target, valid = ardy_motion_target(reader, episode, frame, 2)
                self.assertFalse(valid)
                self.assertFalse(target.any())
        reader._episodes[0]["ardy_loss_mask"][6] = False
        self.assertFalse(ardy_motion_target(reader, 0, 8, 2)[1])
        reader._episodes[1]["ardy_body_latent"][8, 0] = np.nan
        with self.assertRaises(ValueError):
            ardy_motion_target(reader, 1, 8, 2)

    def test_configuration_and_empty_mask(self):
        self.assertEqual(motion_config(SimpleNamespace(model="VGQ-AE")), {})
        self.assertEqual(
            motion_config(
                SimpleNamespace(model="VGQ-AE-Mo", ardy_delta_frames=4, motion_weight=0.2)
            ),
            dict(ardy_delta_frames=4, motion_weight=0.2),
        )
        for delta, weight in (
            (0, 0.1),
            (-1, 0.1),
            (True, 0.1),
            (1.5, 0.1),
            (2, -1),
            (2, float("nan")),
        ):
            with self.assertRaises(ValueError):
                motion_config(
                    SimpleNamespace(
                        model="VGQ-AE-Mo", ardy_delta_frames=delta, motion_weight=weight
                    )
                )
        prediction = torch.randn(2, 128, requires_grad=True)
        loss = masked_ardy_loss(
            prediction, torch.full_like(prediction, float("nan")), torch.zeros(2, dtype=torch.bool)
        )
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertFalse(prediction.grad.any())


class PsiMotionTests(unittest.TestCase):
    variant = "VGQ-AE-Mo"

    def test_wrapper_and_repack_keep_motion_labels_out_of_observation(self):
        from psi.config.transform import SimpleRepackTransform
        from psi.data.lerobot.lerobot_ext import LeRobotDatasetWrapper

        reader = sidecar()
        raw = dict(
            episode_index=1,
            frame_index=8,
            **{
                "observation.images.egocentric": torch.rand(4, 3, 4, 6),
                "observation.images.egocentric_is_pad": np.zeros(4, bool),
            },
            states=torch.full((1, 32), float("nan")),
            action=torch.ones(2, 36),
            action_is_pad=torch.zeros(2, dtype=torch.bool),
            task="move",
        )
        wrapper = LeRobotDatasetWrapper.__new__(LeRobotDatasetWrapper)
        wrapper.base_dataset = [raw]
        wrapper.ardy_reader, wrapper.ardy_offset, wrapper.motion_delta = reader, 0, 2
        wrapper.ardy_readers = [reader]
        sample = wrapper[0]
        repack = SimpleRepackTransform(rsci_variant=self.variant, action_chunk_size=2)
        packed = repack(sample)
        np.testing.assert_array_equal(packed["motion_target"], np.full(128, 28, np.float32))
        self.assertTrue(packed["motion_valid"])
        np.testing.assert_array_equal(packed["states"], np.zeros((1, 32)))
        np.testing.assert_array_equal(sample["ardy_target"], reader.get(1, 8).ardy_body_latent)

    def test_two_heads_share_one_query_and_trainer_weights_losses(self):
        from test_rsci import inputs, policy, sample
        from psi0_fixtures import trainer

        torch.set_num_threads(1)
        config, model = policy(self.variant)
        batch = sample()
        batch.update(motion_target=torch.randn(2, 128), motion_valid=torch.tensor([False, True]))
        with patch.object(model, "_rhsci_query_latent", wraps=model._rhsci_query_latent) as query:
            losses = trainer(config).forward_and_loss(model, batch)
        self.assertEqual(query.call_count, 1)
        torch.testing.assert_close(
            losses["loss"],
            losses["loss_action"] + 0.1 * losses["loss_ardy"] + 0.1 * losses["loss_motion"],
        )
        losses["loss_motion"].backward()
        self.assertGreater(model.action_header.rsci.state_query.grad.abs().sum(), 0)
        self.assertGreater(model.action_header.rsci.motion_head[-1].weight.grad.abs().sum(), 0)
        self.assertIsNone(model.action_header.rsci.ardy_head[-1].weight.grad)
        model.eval()
        with patch.object(
            model, "_rhsci_query_latent", side_effect=AssertionError("inference Query")
        ):
            output = model(**inputs(batch, auxiliary=False))
        self.assertIsNone(output.motion_prediction)


class PiMotionTests(unittest.TestCase):
    variant = "VGQ-AE-Mo"

    def test_processor_collates_labels_without_adding_observation_fields(self):
        from openpi.rhsci.processing import Processor, collate

        processor = Processor.__new__(Processor)
        processor.variant = self.variant
        processor.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])
        processor.stats = {"action": dict(q01=[-1.0] * 36, q99=[1.0] * 36)}
        sample = dict(
            text="move",
            images=np.zeros((4, 4, 4, 3), np.uint8),
            actions=np.zeros((30, 36), np.float32),
            rsci_history_valid=np.ones(4, bool),
            rsci_ardy_target=np.zeros(128, np.float32),
            rsci_ardy_valid=True,
            rsci_motion_target=np.full(128, 28, np.float32),
            rsci_motion_valid=True,
        )
        batch = collate([processor(sample), processor(sample)])
        self.assertEqual(batch["motion_target"].shape, (2, 128))
        self.assertTrue(batch["motion_valid"].all())
        self.assertEqual(
            set(batch),
            {
                "images",
                "tokens",
                "token_mask",
                "actions",
                "history_valid",
                "ardy_target",
                "ardy_valid",
                "motion_target",
                "motion_valid",
            },
        )

    def test_total_gradients_roundtrip_and_current_only_inference(self):
        from test_pi05_rsci import fixture

        torch.set_num_threads(1)
        model, batch = fixture(self.variant)
        batch.update(motion_target=torch.randn(2, 128), motion_valid=torch.tensor([False, True]))
        output = model(**batch)
        torch.testing.assert_close(
            output["loss"].detach(),
            output["action_loss"] + 0.1 * output["ardy_loss"] + 0.1 * output["motion_loss"],
        )
        output["loss"].backward()
        self.assertGreater(model.motion_head[-1].weight.grad.abs().sum(), 0)
        self.assertGreater(model.query.grad.abs().sum(), 0)
        restored, _ = fixture(self.variant)
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        with patch.object(model.reader, "forward", side_effect=AssertionError("inference Reader")):
            self.assertEqual(
                model.sample_actions(
                    batch["images"][:, -1:], batch["tokens"], batch["token_mask"], num_steps=1
                ).shape,
                (2, 30, 36),
            )
        self.assertFalse(hasattr(fixture("VGQ-AE")[0], "motion_head"))


class GrMotionTests(unittest.TestCase):
    variant = "VGQ-AE-Mo"

    def test_shared_dit_backward_and_unchanged_action_path(self):
        from test_gr00t_rsci import batch, model

        torch.set_num_threads(1)
        policy = model(self.variant)
        inputs = batch()
        inputs.update(
            rsci_motion_target=torch.randn(2, 128), rsci_motion_valid=torch.tensor([False, True])
        )
        with patch.object(
            policy.action_head.rsci_query, "forward", wraps=policy.action_head.rsci_query.forward
        ) as query:
            output = policy(inputs)
        self.assertEqual(query.call_count, 1)
        output["motion_loss"].backward()
        self.assertGreater(policy.action_head.rsci_query.query.grad.abs().sum(), 0)
        self.assertGreater(policy.action_head.rsci_query.motion_head[-1].weight.grad.abs().sum(), 0)
        self.assertIsNone(policy.action_head.rsci_query.ardy_head[-1].weight.grad)
        restored = model(self.variant)
        restored.load_state_dict(policy.state_dict(), strict=True)
        with patch.object(
            policy.action_head.rsci_query, "forward", side_effect=AssertionError("inference Query")
        ):
            self.assertIn("action_pred", policy.get_action(inputs))
        self.assertFalse(hasattr(model("VGQ-AE").action_head.rsci_query, "motion_head"))


class EgoMotionTests(unittest.TestCase):
    variant = "VGQ-AE-Mo"

    def test_shared_decoder_and_motion_loss(self):
        from egovla.test_egovla import Backbone, batch
        from human_plan.rhsci.model import EgoVLARHSCI

        torch.set_num_threads(1)
        model = EgoVLARHSCI(Backbone(), variant=self.variant, vlm_training="frozen")
        inputs = batch()
        inputs.update(motion_target=torch.randn(2, 128), motion_valid=torch.tensor([False, True]))
        output = model(**inputs)
        torch.testing.assert_close(
            output["loss"].detach(),
            output["action_loss"] + 0.1 * output["ardy_loss"] + 0.1 * output["motion_loss"],
        )
        output["loss"].backward()
        self.assertGreater(model.motion_head[-1].weight.grad.abs().sum(), 0)
        self.assertGreater(model.query.grad.abs().sum(), 0)
        model.eval()
        current = {key: inputs[key] for key in ("images", "input_ids", "labels", "attention_mask")}
        current["images"] = current["images"][:, -1:]
        with patch.object(model.reader, "forward", side_effect=AssertionError("inference Reader")):
            self.assertEqual(model.sample_actions(**current).shape, (2, 30, 36))


class LambdaMotionTests(unittest.TestCase):
    variant = "VGQ-AE-Mo"

    def test_motion_head_uses_single_auxiliary_pass(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "lambda_motion_fixtures", Path(__file__).parent / "lambda0/test_integration.py"
        )
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        inputs, tiny, vlm_inputs = fixtures.inputs, fixtures.tiny, fixtures.vlm_inputs

        torch.set_num_threads(1)
        model = tiny(self.variant)
        model.config.rsci_ardy_delta_frames = 2
        model.config.rsci_motion_weight = 0.1
        model.install_auxiliary()
        data = inputs(model)
        data.update(
            history=[vlm_inputs() for _ in range(3)],
            history_valid=torch.ones(2, 4, dtype=torch.bool),
            ardy_target=torch.randn(2, 128),
            ardy_valid=torch.ones(2, dtype=torch.bool),
            motion_target=torch.randn(2, 128),
            motion_valid=torch.tensor([False, True]),
        )
        with patch.object(model, "auxiliary", wraps=model.auxiliary) as query:
            output = model(**data)
        self.assertEqual(query.call_count, 1)
        output["loss"].backward()
        self.assertGreater(model.rsci_motion[-1].weight.grad.abs().sum(), 0)
        self.assertGreater(model.rsci_query.grad.abs().sum(), 0)
        self.assertTrue(torch.isfinite(model.last_motion_loss))
        self.assertFalse(hasattr(tiny("VGQ-AE"), "rsci_motion"))
