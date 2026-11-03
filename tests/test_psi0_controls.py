"""四图 VGQ-VLM 与状态重建：实际 Qwen 前向、监督隔离和加载兼容。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np
from safetensors.torch import save_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import ActionStateTransform, DataTransform, SimpleRepackTransform
from psi.rhsci.config import load_run_config
from psi.rhsci.controls import control_spec, forward_loss, reconstruction_loss
from psi.rhsci.guardian import run_training_guardian
from psi.rhsci.training_config import build_argument_parser, resolve_training_plan
from test_rsci import inputs, policy, sample
from test_psi0_preprocessing import ImageTransform


def tiny_vlm():
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=32, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         head_dim=4, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 0]}),
        vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=8, patch_size=2, temporal_patch_size=2,
                           spatial_merge_size=2, num_position_embeddings=16,
                           deepstack_visual_indexes=[0]),
        image_token_id=29, video_token_id=27, vision_start_token_id=30, vision_end_token_id=28,
    )
    for component in (config, config.text_config, config.vision_config):
        component._attn_implementation = "sdpa"
    return Qwen3VLForConditionalGeneration(config)


def four_frame_sample():
    value = sample()
    frame = dict(input_ids=torch.tensor([[30, 29, 28, 1], [30, 29, 28, 2]]),
                 attention_mask=torch.ones(2, 4, dtype=torch.long),
                 pixel_values=torch.randn(2, 4, 24),
                 image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]))
    value.update(frame)
    value["history_vlm_inputs"] = [dict(frame, pixel_values=torch.randn(2, 4, 24)) for _ in range(3)]
    value["history_valid"].fill_(True)
    value["ardy_valid"].fill_(True)
    return value


class ControlTests(unittest.TestCase):
    def test_reconstruction_labels_survive_transform_without_entering_observation(self):
        field = ActionStateTransform(stat_path="/nonexistent/fixture-stats.json", normalize_state=True,
                                     pad_state_dim=36, pad_action_dim=36)
        field.populate_stats({"action": {"min": [-2.] * 36, "max": [2.] * 36},
                              "states": {"min": [-2.] * 32, "max": [2.] * 32}})
        transform = DataTransform(
            repack=SimpleRepackTransform(rsci_variant="STATE-RECONSTRUCTION", pad_state_dim=36, pad_action_dim=36),
            field=field, model=ImageTransform(img_aug=False))
        value = {"observation.images.egocentric": torch.rand(4, 3, 12, 20),
                 "observation.images.egocentric_is_pad": np.zeros(4, bool),
                 "states": torch.ones(1, 32), "action": torch.zeros(30, 36),
                 "action_is_pad": torch.zeros(30, dtype=torch.bool), "task": "Move",
                 "ardy_target": np.zeros(128, np.float32), "ardy_valid": True}
        result = transform(value, no_aug=True)
        np.testing.assert_array_equal(result["states"], np.zeros((1, 36)))
        np.testing.assert_array_equal(result["state_target"], np.full(31, .5))
        self.assertTrue(result["state_valid"].all())
        self.assertEqual(len(result["history_vlm_inputs"]), 3)
        self.assertNotIn("teacher_states", result)

    def test_direct_four_images_gradient_action_isolation_and_roundtrip(self):
        torch.manual_seed(32)
        cfg, model = policy("VGQ-VLM", "lora", vlm=tiny_vlm())
        self.assertFalse(hasattr(model.action_header, "rsci"))
        self.assertIsNone(model.action_header.obs_proj._obs_strat)
        value = four_frame_sample()
        # Query branch must carry all four raw image patches through the actual VLM.
        with patch.object(model.vlm_model, "forward", wraps=model.vlm_model.forward) as forward:
            output = model(**inputs(value))
        self.assertEqual(len(forward.call_args_list), 2)
        self.assertEqual(tuple(forward.call_args_list[-1].kwargs["image_grid_thw"].shape), (8, 3))
        self.assertEqual(tuple(output.ardy_prediction.shape), (2, 128))
        output.ardy_prediction.square().mean().backward()
        self.assertGreater(model.body_query.grad.abs().sum().item(), 0)
        visual_grads = [p.grad for n, p in model.named_parameters() if "visual" in n and "lora_B" in n]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in visual_grads))
        model.eval()
        with torch.no_grad():
            expected = model(**inputs(value, auxiliary=False)).action
            changed = dict(value, states=torch.randn_like(value["states"]),
                           history_vlm_inputs=None, history_valid=None)
            with patch("psi.rhsci.vlm_query.query_output", side_effect=AssertionError("inference Query")):
                actual = model(**inputs(changed, auxiliary=False)).action
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            restored = policy("VGQ-VLM", "lora", vlm=tiny_vlm())[1].eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(restored(**inputs(value, auxiliary=False)).action, expected, rtol=0, atol=0)
        # The public trainer uses action+ARDY, preserving masks and shared flow sampling.
        model.train()
        losses = forward_loss(model, value, control_spec(cfg), max_delay=2)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIn("loss_ardy", losses)

    def test_state_reconstruction_uses_reader_ae_and_valid_element_loss(self):
        cfg, model = policy("STATE-RECONSTRUCTION")
        self.assertTrue(model.action_header.rsci.use_reader)
        self.assertEqual(model.action_header.rsci.ardy_head[-1].out_features, 31)
        value = sample()
        value.update(state_target=torch.randn(2, 31), state_valid=torch.ones(2, 31, dtype=torch.bool))
        value["ardy_valid"].fill_(True)
        losses = forward_loss(model, value, control_spec(cfg), max_delay=2)
        losses["loss"].backward()
        self.assertGreater(model.action_header.rsci.state_query.grad.abs().sum().item(), 0)
        self.assertIn("loss_state", losses)
        self.assertNotIn("loss_ardy", losses)
        target = torch.full((2, 31), float("nan"))
        prediction = torch.randn(2, 31, requires_grad=True)
        empty = reconstruction_loss(prediction, target, torch.zeros_like(target, dtype=torch.bool))
        empty.backward()
        self.assertEqual(empty.item(), 0)
        self.assertEqual(prediction.grad.count_nonzero().item(), 0)

    def test_removed_methods_and_legacy_checkpoint_detection(self):
        for name in ("VGQ-VLM-DIRECT", "STATE-TEACHER-DISTILLATION"):
            with self.assertRaises(ValueError):
                Psi0ModelConfig(rsci_variant=name, n_conditions=0)
        args = build_argument_parser().parse_args(["--model", "VGQ-VLM", "--vlm", "lora", "--task", "fixture"])
        cfg = resolve_training_plan(args).config
        self.assertTrue(cfg.data.transform.model.img_aug)
        saved = cfg.model_dump(mode="json")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "argv.txt").write_text("train.py finetune_simple_psi0_config")
            config_file = root / "run_config.json"
            config_file.write_text(json.dumps(saved))
            weights = root / "model.safetensors"
            save_file({"action_header.rsci.schema_version": torch.tensor(1)}, weights)
            with self.assertRaisesRegex(ValueError, "Legacy VGQ-VLM Reader"):
                load_run_config(root, weights)
            save_file({"body_query": torch.zeros(1, 1, 8)}, weights)
            self.assertEqual(load_run_config(root, weights).model.rsci_variant, "VGQ-VLM")

            saved["model"]["rsci_variant"] = "VGQ-VLM-DIRECT"
            config_file.write_text(json.dumps(saved))
            self.assertEqual(load_run_config(root, weights).model.rsci_variant, "VGQ-VLM")

    def test_guardian_rejects_old_same_name_before_completed_skip(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            args = build_argument_parser().parse_args([
                "--model", "VGQ-VLM", "--vlm", "lora", "--task", "fixture",
                "--output-dir", str(output),
            ])
            plan = resolve_training_plan(args)
            run = output / "finetune/run"
            checkpoint = run / "checkpoints/ckpt_40000"
            checkpoint.mkdir(parents=True)
            (run / "argv.txt").write_text("train.py finetune_simple_psi0_config")
            (run / "run_config.json").write_text(json.dumps(plan.config.model_dump(mode="json")))
            save_file({"action_header.rsci.schema_version": torch.tensor(1)}, checkpoint / "model.safetensors")
            args.resume = checkpoint
            with patch("psi.rhsci.guardian.checkpoint_ok", return_value=True), \
                 patch("psi.rhsci.guardian.signal.signal"), \
                 patch("psi.rhsci.guardian.subprocess.Popen") as spawn:
                with self.assertRaisesRegex(ValueError, "Legacy VGQ-VLM Reader"):
                    run_training_guardian(args, *plan)
                spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
