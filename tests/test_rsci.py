"""RHSCI 小尺寸真实 AE/Qwen 解码器：结构、两种损失梯度、推理与恢复契约。"""

import copy
import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from psi0_fixtures import batch, trainer, vl_inputs
from torch import nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import SimpleRepackTransform
from psi.models.psi0 import ActionTransformerModel, Psi0Model
from rsci.backbones.psi0.model import VARIANTS, RSCISpec, masked_auxiliary_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))
from rsci.backbones.psi0 import train as launch_rsci


class SmallVLM(nn.Module):
    def __init__(self):
        super().__init__()
        config = Qwen3VLTextConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 0]},
        )
        config._attn_implementation = "sdpa"
        self.model = nn.Module()
        self.model.language_model = Qwen3VLTextModel(config)
        self.model.visual = nn.Linear(4, 8)
        self.lm_head = nn.Linear(8, 32, bias=False)

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        embeds = self.model.language_model.embed_tokens(input_ids)
        embeds = embeds + self.model.visual(pixel_values)[:, None]
        return SimpleNamespace(
            hidden_states=[
                self.model.language_model(
                    inputs_embeds=embeds, attention_mask=attention_mask, use_cache=False
                ).last_hidden_state
            ]
        )


class SmallDINO(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer("scale", torch.arange(1, 385).float())

    def forward(self, images):
        return (
            images.float().mean((1, 2, 3))[:, None, None].expand(-1, 2, 384)
            * self.scale[None, None]
            / 255
        )


def policy(variant, mode="frozen", vlm=None, gradient_checkpointing=False, use_reader=True):
    cfg = Psi0ModelConfig(
        rsci_variant=variant,
        rsci_vlm_mode=mode,
        n_conditions=0,
        hidden_dim=8,
        nhead=2,
        num_blocks=2,
        action_dim=2,
        action_chunk_size=2,
        odim=36,
        view_feature_dim=8,
        rtc=True,
        max_delay=2,
        gradient_checkpointing=gradient_checkpointing,
        rsci_use_reader=use_reader,
    )
    constructor = ActionTransformerModel

    def small_header(**kwargs):
        return constructor(
            attention_head_dim=4,
            num_attention_heads=2,
            caption_projection_dim=8,
            pooled_projection_dim=8,
            **kwargs,
        )

    with (
        patch("psi.models.psi0.ActionTransformerModel", side_effect=small_header),
        patch("rsci.backbones.psi0.rgb_encoder.DINOv2RGBAdapter", SmallDINO),
    ):
        model = Psi0Model(cfg, vlm if vlm is not None else SmallVLM())
    return cfg, model


def sample():
    value = batch()
    value["pixel_values"] = torch.randn(2, 4)
    value["history_rgb_images"] = torch.randint(0, 256, (2, 4, 3, 4, 6), dtype=torch.uint8)
    return value


def inputs(value, auxiliary=True):
    keys = (*vl_inputs(), "states", "history_vlm_inputs", "history_rgb_images", "history_valid")
    return {
        **{k: value[k] for k in keys},
        "action_samples": value["actions"],
        "timestep": torch.tensor([1.0, 2.0]),
        "compute_rsci_aux": auxiliary,
    }


def ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        for variant in ("VGQ-AE", "EGQ-VLM*"):
            cfg, model = policy(variant, "lora")
            ddp = torch.nn.parallel.DistributedDataParallel(model)
            for step in range(2):
                value = sample()
                value["ardy_valid"].fill_(rank == 0 and step == 0)
                if not value["ardy_valid"].any():
                    value["ardy_target"].fill_(float("nan"))
                losses = trainer(cfg).forward_and_loss(ddp, value)
                losses["loss"].backward()
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.requires_grad
                )
                ddp.zero_grad(set_to_none=True)
    finally:
        torch.distributed.destroy_process_group()


class RSCITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_structures_loss_gradients_and_roundtrip(self):
        for variant in VARIANTS:
            for mode in ("frozen", "lora"):
                with self.subTest(variant=variant, mode=mode):
                    cfg, model = policy(variant, mode)
                    value = sample()
                    output = model(**inputs(value))
                    self.assertEqual(output.action.shape, (2, 2, 2))
                    self.assertEqual(cfg.tune_vlm, mode == "lora")
                    lora = [p for n, p in model.vlm_model.named_parameters() if p.requires_grad]
                    self.assertEqual(bool(lora), mode == "lora")
                    self.assertTrue(
                        all(
                            "lora_" in n
                            for n, p in model.vlm_model.named_parameters()
                            if p.requires_grad
                        )
                    )
                    if RSCISpec(variant).query:
                        auxiliary = masked_auxiliary_loss(
                            output.ardy_prediction, value["ardy_target"], value["ardy_valid"]
                        )
                        auxiliary.backward(retain_graph=True)
                        self.assertGreater(
                            model.action_header.rsci.state_query.grad.abs().sum().item(), 0
                        )
                        if mode == "lora" and variant != "EGQ-AE":
                            self.assertTrue(
                                any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora)
                            )
                        model.zero_grad(set_to_none=True)
                    output.action.square().mean().backward()
                    if mode == "lora":
                        self.assertTrue(
                            any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora)
                        )
                    model.eval()
                    restored = policy(variant, mode)[1].eval()
                    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                    with torch.no_grad():
                        torch.testing.assert_close(
                            model(**inputs(value)).action, restored(**inputs(value)).action
                        )
                    self.assertFalse(hasattr(model, "depth_encoder"))

    def test_no_state_history_and_star_contract(self):
        for variant in VARIANTS[1:]:
            model = policy(variant)[1].eval()
            value = sample()
            first = model(**inputs(value))
            value["states"].fill_(float("nan"))
            second = model(**inputs(value))
            torch.testing.assert_close(first.action, second.action, rtol=0, atol=0)
            if not RSCISpec(variant).forward_latent:
                inference = inputs(value, False)
                for key in ("history_vlm_inputs", "history_rgb_images", "history_valid"):
                    inference.pop(key)
                with patch.object(
                    model,
                    "_rsci_query_latent",
                    side_effect=AssertionError("inference must skip Query"),
                ):
                    torch.testing.assert_close(
                        first.action, model(**inference).action, rtol=0, atol=0
                    )
            else:
                inference = inputs(value, False)
                inference.pop("history_valid")
                with self.assertRaisesRegex(ValueError, "history_valid"):
                    model(**inference)

    def test_star_history_changes_action_and_query_is_encoded_once(self):
        # 固定小模型初始化并恢复外部RNG，避免不同测试顺序令历史影响偶然低于FP32阈值。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = policy("VGQ-VLM*", "lora")[1].eval()
            value = sample()
        first = model(**inputs(value, False)).action
        value["history_vlm_inputs"] = [vl_inputs(9)] * 3
        second = model(**inputs(value, False)).action
        self.assertGreater((first - second).abs().max().item(), 1e-7)
        hidden = model.vlm_model(**vl_inputs()).hidden_states[-1]
        with patch.object(model, "_rsci_query_latent", wraps=model._rsci_query_latent) as query:
            appended, mask = model._rsci_inference_features(hidden, value["attention_mask"], value)
            self.assertEqual(query.call_count, 1)
            self.assertEqual(appended.shape[1], hidden.shape[1] + 1)
            self.assertEqual(mask.shape[1], hidden.shape[1] + 1)

    def test_loss_labels_empty_mask_and_ddp(self):
        cfg, model = policy("VGQ-AE", "lora")
        value = sample()
        value["ardy_target"].fill_(float("nan"))
        value["ardy_valid"].zero_()
        losses = trainer(cfg).forward_and_loss(model, value)
        self.assertEqual(losses["loss_ardy"].item(), 0)
        torch.testing.assert_close(
            losses["loss"], losses["loss_action"] + 0.1 * losses["loss_ardy"]
        )
        losses["loss"].backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.requires_grad
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                ddp_worker, args=(str(Path(directory) / "rendezvous"),), nprocs=2
            )

    def test_repack_and_launcher_matrix(self):
        for variant in VARIANTS:
            repack = SimpleRepackTransform(rsci_variant=variant, action_chunk_size=2)
            timestamps = repack.delta_timestamps(50)
            self.assertNotIn("observation.depth.egocentric", timestamps)
            if RSCISpec(variant).query:
                self.assertEqual(
                    timestamps["observation.images.egocentric"], [-0.12, -0.08, -0.04, 0]
                )
                raw = {
                    "observation.images.egocentric": torch.rand(4, 3, 4, 6),
                    "observation.images.egocentric_is_pad": torch.tensor(
                        [True, False, False, False]
                    ),
                    "states": torch.full((1, 32), float("nan")),
                    "action": torch.ones(2, 32),
                    "action_is_pad": torch.zeros(2, dtype=torch.bool),
                    "task": "test",
                    "ardy_target": np.zeros(128, np.float32),
                    "ardy_valid": True,
                }
                packed = repack(raw)
                self.assertFalse(packed["ardy_valid"])
                self.assertTrue(np.isfinite(packed["states"]).all())
                self.assertEqual(np.abs(packed["states"]).sum(), 0)
                self.assertIn(
                    "history_rgb_images" if variant.startswith("EGQ") else "history_vlm_images",
                    packed,
                )
            for mode in ("frozen", "lora"):
                args = launch_rsci.parser().parse_args(
                    [
                        "--model",
                        variant,
                        "--task",
                        "G1Test-v0",
                        "--vlm",
                        mode,
                        "--gpus",
                        "0",
                        "--dry-run",
                    ]
                )
                cfg, command, output, ranks = launch_rsci.resolved(args)
                self.assertEqual(cfg.model.rsci_variant, variant)
                self.assertEqual(cfg.model.tune_vlm, mode == "lora")
                self.assertIn("runs/psi0", str(output))
                self.assertFalse(hasattr(cfg.model, "use_depth"))
                self.assertIsNone(cfg.log.report_to)
        with self.assertRaises(ValueError):
            RSCISpec("EGQ-AE*")

    def test_custom_training_budget_and_environment_nccl(self):
        args = launch_rsci.parser().parse_args(
            [
                "--task",
                "custom",
                "--model",
                "STATE",
                "--gpus",
                "0",
                "--data-root",
                "/datasets",
                "--output-dir",
                "/runs/custom",
                "--max-steps",
                "2000",
            ]
        )
        config, command, output, ranks = launch_rsci.resolved(args)
        self.assertEqual(config.train.max_training_steps, 2000)
        self.assertEqual(str(output), "/runs/custom")
        self.assertNotIn("--library", command)
        self.assertIn("torch.distributed.run", command)
        args.nccl_library = Path("/runtime/libnccl.so.2")
        self.assertIn("/runtime/libnccl.so.2", launch_rsci.resolved(args)[1])

    def test_real_qwen_visual_lora_and_full_checkpoint_restore(self):
        from safetensors.torch import load_file, save_model
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        config = Qwen3VLConfig(
            text_config=dict(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=4,
                rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 0]},
                tie_word_embeddings=True,
            ),
            vision_config=dict(
                depth=2,
                hidden_size=16,
                intermediate_size=32,
                num_heads=2,
                out_hidden_size=8,
                patch_size=2,
                temporal_patch_size=2,
                spatial_merge_size=2,
                num_position_embeddings=16,
                deepstack_visual_indexes=[0],
            ),
            image_token_id=29,
            vision_start_token_id=30,
            vision_end_token_id=28,
            tie_word_embeddings=True,
        )
        for component in (config, config.text_config, config.vision_config):
            component._attn_implementation = "sdpa"
        cfg, model = policy(
            "VGQ-VLM*",
            "lora",
            Qwen3VLForConditionalGeneration(config).to(torch.bfloat16),
            gradient_checkpointing=True,
        )
        self.assertTrue(
            all(
                p.dtype == torch.float32
                for n, p in model.vlm_model.named_parameters()
                if "lora_" in n
            )
        )
        value = sample()
        value.update(
            input_ids=torch.tensor([[30, 29, 28, 1]] * 2),
            attention_mask=torch.ones(2, 4, dtype=torch.long),
            pixel_values=torch.randn(2, 4, 24),
            image_grid_thw=torch.tensor([[1, 2, 2]] * 2),
        )
        value["history_vlm_inputs"] = [{k: value[k] for k in vl_inputs()}] * 3
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(**inputs(value))
            loss = (
                output.action.float().square().mean()
                + output.ardy_prediction.float().square().mean()
            )
        loss.backward()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for n, p in model.vlm_model.named_parameters()
                if n.startswith("model.visual") and "lora_B" in n
            )
        )
        model.eval()
        constructor = ActionTransformerModel

        def small_header(**kwargs):
            return constructor(
                attention_head_dim=4,
                num_attention_heads=2,
                caption_projection_dim=8,
                pooled_projection_dim=8,
                **kwargs,
            )

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.safetensors")
            save_model(model, path)
            with (
                patch(
                    "transformers.AutoConfig.from_pretrained", return_value=copy.deepcopy(config)
                ),
                patch("transformers.AutoProcessor.from_pretrained", return_value=object()),
                patch("psi.models.psi0.ActionTransformerModel", side_effect=small_header),
            ):
                restored = Psi0Model._restore_rsci(load_file(path), cfg, "cpu")
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                torch.testing.assert_close(
                    model(**inputs(value)).action, restored(**inputs(value)).action, rtol=0, atol=0
                )

    def test_guard_recovery_contract_and_no_zero_restart(self):
        from safetensors.torch import save_file

        args = launch_rsci.parser().parse_args(
            ["--model", "VGQ-AE", "--task", "G1Test-v0", "--gpus", "0", "--dry-run"]
        )
        cfg, command, _, ranks = launch_rsci.resolved(args)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            # 无断点失败只能停止，不能重新从零跑第二次。
            failed = SimpleNamespace(pid=99999999, returncode=1, poll=lambda: 1)
            with patch.object(launch_rsci.subprocess, "Popen", return_value=failed) as spawn:
                with self.assertRaisesRegex(RuntimeError, "step zero"):
                    launch_rsci.guard(args, cfg, command, output, ranks)
                self.assertEqual(spawn.call_count, 1)
            with patch.object(launch_rsci.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "step zero"):
                    launch_rsci.guard(args, cfg, command, output, ranks)
                spawn.assert_not_called()
            run = output / "finetune/test"
            checkpoint = run / "checkpoints/ckpt_1000"
            checkpoint.mkdir(parents=True)
            (run / "run_config.json").write_text(cfg.model_dump_json())
            save_file({"weight": torch.ones(1)}, str(checkpoint / "model.safetensors"))
            for name in ("optimizer.bin", "scheduler.bin"):
                torch.save({}, checkpoint / name)
            (checkpoint / "random_states_0.pkl").write_bytes(b"fixture")
            self.assertEqual(launch_rsci.latest_checkpoint(output, 1), checkpoint)
            self.assertIsNone(launch_rsci.latest_checkpoint(output, 2))
            with patch.object(launch_rsci.subprocess, "Popen", return_value=failed) as spawn:
                with self.assertRaisesRegex(RuntimeError, "No new complete checkpoint"):
                    launch_rsci.guard(args, cfg, command, output, ranks)
                self.assertEqual(spawn.call_count, 1)
                self.assertIn(
                    f"--train.resume-from-checkpoint={checkpoint}", spawn.call_args.args[0]
                )
                self.assertIn("--train.skip-resumed-steps", spawn.call_args.args[0])
            saved = cfg.model_dump(mode="json")
            saved["model"]["rsci_variant"] = "EGQ-AE"
            (run / "run_config.json").write_text(json.dumps(saved))
            with patch.object(launch_rsci.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(ValueError, "do not match"):
                    launch_rsci.guard(args, cfg, command, output, ranks)
                spawn.assert_not_called()
            identity = launch_rsci.contract(cfg, ranks)
            changed = cfg.model_copy(deep=True)
            changed.model.rsci_vlm_mode = "lora"
            self.assertNotEqual(identity, launch_rsci.contract(changed, ranks))
            self.assertNotEqual(identity, launch_rsci.contract(cfg, 2))

    def test_serving_history_protocol(self):
        model = policy("EGQ-VLM*")[1].eval()
        model.device = "cpu"
        frames = np.zeros((4, 4, 6, 3), dtype=np.uint8)
        condition = dict(
            rsci_rgb_frames=frames,
            rsci_history_valid=np.array([False, False, True, True]),
            rsci_frame_offsets=np.array([-6, -4, -2, 0]),
        )
        history = model.rsci_history_from_request(condition, "test", None, frames[-1])
        self.assertEqual(history["history_rgb_images"].shape, (1, 4, 3, 4, 6))
        with self.assertRaisesRegex(ValueError, "must equal"):
            model.rsci_history_from_request(condition, "test", None, np.ones_like(frames[-1]))
        condition["rsci_frame_offsets"] = np.array([-3, -2, -1, 0])
        with self.assertRaisesRegex(ValueError, "offsets"):
            model.rsci_history_from_request(condition, "test", None, frames[-1])
        model = policy("EGQ-AE")[1]
        self.assertEqual(model.rsci_history_from_request({}, "test", None), {})

    def test_public_prediction_and_training_rtc_keep_star_once(self):
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from PIL import Image
        from transformers import BatchFeature

        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                return "test"

            def __call__(self, **kwargs):
                values = {key: value[:1].clone() for key, value in vl_inputs().items()}
                values["pixel_values"] = values["pixel_values"][0]
                return BatchFeature(values)

        for variant in ("EGQ-VLM*", "VGQ-AE"):
            model = policy(variant)[1].eval()
            model.vlm_processor, model.device = Processor(), "cpu"
            model.action_horizon, model.action_dim = 2, 2
            model.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
            history = sample() if variant.endswith("*") else {}
            history = {
                key: history[key][:1]
                for key in ("history_rgb_images", "history_valid")
                if key in history
            }
            for rtc in (False, True):
                with patch.object(
                    model, "_rsci_query_latent", wraps=model._rsci_query_latent
                ) as query:
                    args = dict(
                        observations=[[Image.new("RGB", (32, 32))]],
                        states=torch.full((1, 1, 36), float("nan")),
                        instructions=["test"],
                        num_inference_steps=3,
                        traj2ds=None,
                        **history,
                    )
                    if rtc:
                        action = model.predict_action_with_training_rtc_flow(
                            **args,
                            prev_actions=torch.zeros(1, 2, 2),
                            inference_delay=1,
                            max_delay=2,
                        )
                    else:
                        action = model.predict_action(**args)
                    self.assertEqual(action.shape, (1, 2, 2))
                    self.assertTrue(torch.isfinite(action).all())
                    self.assertEqual(query.call_count, int(variant.endswith("*")))

    def test_numerical_failure_blocks_even_explicit_resume_before_spawn(self):
        args = launch_rsci.parser().parse_args(
            ["--model", "VGQ-AE", "--task", "Test-v0", "--resume", "/unused/checkpoint"]
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "NUMERICAL_FAILURE.json").write_text("{}")
            with patch.object(launch_rsci.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "NUMERICAL_FAILURE"):
                    launch_rsci.guard(args, None, [], output, 1)
                spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
