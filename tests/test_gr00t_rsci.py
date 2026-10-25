"""RHSCI GR00T：真实小尺寸DiT的隔离、梯度、推理与完整断点协议验证。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.rsci_query import masked_ardy_loss
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from rsci.backbones.gr00t import checkpoints
from rsci.backbones.gr00t.data import GROUPS, TAG, modality


class TinyBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(64, 16).requires_grad_(False)

    def prepare_input(self, inputs):
        return BatchFeature(dict(inputs))

    def forward(self, inputs):
        ids = inputs["input_ids"]
        return BatchFeature(
            dict(
                backbone_features=self.embedding(ids),
                backbone_attention_mask=inputs["attention_mask"].bool(),
                image_mask=ids.remainder(2).bool(),
            )
        )


def model(variant):
    config = Gr00tN1d6Config(
        rsci_variant=variant,
        tune_top_llm_layers=0,
        max_state_dim=6,
        max_action_dim=4,
        action_horizon=3,
        hidden_size=8,
        input_embedding_dim=16,
        backbone_embedding_dim=16,
        max_num_embodiments=2,
        diffusion_model_cfg=dict(
            positional_embeddings=None,
            num_layers=4,
            num_attention_heads=2,
            attention_head_dim=8,
            norm_type="ada_norm",
            dropout=0.0,
            final_dropout=False,
            output_dim=8,
            interleave_self_attention=True,
        ),
    )
    with (
        patch("gr00t.model.gr00t_n1d6.gr00t_n1d6.get_backbone_cls", return_value=TinyBackbone),
        patch("gr00t.model.gr00t_n1d6.processing_gr00t_n1d6.Gr00tN1d6DataCollator"),
    ):
        return Gr00tN1d6(config).eval()


def batch():
    current = dict(
        input_ids=torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
    )
    return {
        **current,
        "state": torch.randn(2, 1, 6),
        "action": torch.randn(2, 3, 4),
        "action_mask": torch.ones(2, 3, 4),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
        "rsci_history": [copy.deepcopy(current) for _ in range(3)],
        "rsci_history_valid": torch.ones(2, 4, dtype=torch.bool),
        "rsci_ardy_target": torch.randn(2, 128),
        "rsci_ardy_valid": torch.ones(2, dtype=torch.bool),
    }


def ddp_worker(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(19)
        policy = model("VGQ-AE").train()
        distributed = torch.nn.parallel.DistributedDataParallel(
            policy, find_unused_parameters=False
        )
        optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-3)
        for _ in range(2):
            inputs = batch()
            inputs["rsci_ardy_valid"][:] = rank == 0
            optimizer.zero_grad(set_to_none=True)
            distributed(inputs)["loss"].backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in policy.parameters()
                if p.requires_grad
            )
            optimizer.step()
        checksum = torch.stack([p.detach().sum() for p in policy.parameters()]).sum()
        gathered = [torch.zeros_like(checksum) for _ in range(2)]
        dist.all_gather(gathered, checksum)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actions_ignore_state_history_query_and_teacher(self):
        policy, inputs = model("VGQ-AE"), batch()
        torch.manual_seed(123)
        reference = policy(copy.deepcopy(inputs))
        altered = copy.deepcopy(inputs)
        altered["state"].fill_(float("nan"))
        altered["rsci_ardy_target"] += 100
        for past in altered["rsci_history"]:
            past["input_ids"] += 6
        with torch.no_grad():
            policy.action_head.rsci_query.query.add_(
                10 * torch.randn_like(policy.action_head.rsci_query.query)
            )
        torch.manual_seed(123)
        result = policy(altered)
        torch.testing.assert_close(reference["action_loss"], result["action_loss"], rtol=0, atol=0)
        self.assertFalse(torch.allclose(reference["ardy_prediction"], result["ardy_prediction"]))

    def test_ardy_ignores_action_and_noise_and_updates_shared_dit(self):
        policy, inputs = model("VGQ-AE"), batch()
        first = policy(copy.deepcopy(inputs))
        altered = copy.deepcopy(inputs)
        altered["action"] *= 20
        torch.manual_seed(983)
        second = policy(altered)
        torch.testing.assert_close(
            first["ardy_prediction"], second["ardy_prediction"], rtol=0, atol=0
        )
        second["ardy_loss"].backward()
        shared = list(policy.action_head.model.parameters())
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in shared))
        self.assertTrue(all(p.grad is None for p in policy.backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.action_head.state_encoder.parameters()))
        self.assertTrue(
            all(not p.requires_grad for p in policy.action_head.state_encoder.parameters())
        )

    def test_inference_needs_neither_state_history_nor_query(self):
        policy, inputs = model("VGQ-AE"), batch()
        minimal = {key: inputs[key] for key in ("input_ids", "attention_mask", "embodiment_id")}
        with patch.object(
            policy.action_head.rsci_query,
            "forward",
            side_effect=AssertionError("Query executed during inference"),
        ):
            output = policy.get_action(minimal)
        self.assertEqual(tuple(output["action_pred"].shape), (2, 3, 4))
        self.assertTrue(torch.isfinite(output["action_pred"]).all())

    def test_no_state_and_state_paths_differ_as_expected(self):
        inputs = batch()
        for variant in ("STATE", "NO-STATE"):
            policy = model(variant)
            torch.manual_seed(1)
            first = policy(copy.deepcopy(inputs))["action_loss"]
            changed = copy.deepcopy(inputs)
            changed["state"] += 100 * torch.randn_like(changed["state"])
            torch.manual_seed(1)
            second = policy(changed)["action_loss"]
            if variant == "NO-STATE":
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                self.assertFalse(hasattr(policy.action_head, "rsci_query"))
            else:
                self.assertFalse(torch.allclose(first, second))

    def test_state_matches_native_path_with_identical_weights(self):
        native, baseline, inputs = model(None), model("STATE"), batch()
        baseline.load_state_dict(native.state_dict(), strict=True)
        torch.manual_seed(77)
        expected = native(copy.deepcopy(inputs))["action_loss"]
        torch.manual_seed(77)
        torch.testing.assert_close(expected, baseline(inputs)["action_loss"], rtol=0, atol=0)

    def test_added_modules_survive_pretrained_save_and_reload(self):
        policy, inputs = model("VGQ-AE"), batch()
        expected = policy(copy.deepcopy(inputs))["ardy_prediction"]
        with tempfile.TemporaryDirectory() as directory:
            policy.save_pretrained(directory)
            with (
                patch(
                    "gr00t.model.gr00t_n1d6.gr00t_n1d6.get_backbone_cls", return_value=TinyBackbone
                ),
                patch("gr00t.model.gr00t_n1d6.processing_gr00t_n1d6.Gr00tN1d6DataCollator"),
            ):
                restored, loading = Gr00tN1d6.from_pretrained(directory, output_loading_info=True)
            self.assertFalse(loading["missing_keys"] or loading["unexpected_keys"])
            torch.testing.assert_close(
                expected, restored.eval()(inputs)["ardy_prediction"], rtol=0, atol=0
            )

    def test_native_checkpoint_initializes_only_new_query_modules(self):
        native = model(None)
        config = copy.deepcopy(native.config)
        config.rsci_variant = "VGQ-AE"
        with tempfile.TemporaryDirectory() as directory:
            native.save_pretrained(directory)
            with (
                patch(
                    "gr00t.model.gr00t_n1d6.gr00t_n1d6.get_backbone_cls", return_value=TinyBackbone
                ),
                patch("gr00t.model.gr00t_n1d6.processing_gr00t_n1d6.Gr00tN1d6DataCollator"),
            ):
                restored, loading = Gr00tN1d6.from_pretrained(
                    directory, config=config, low_cpu_mem_usage=True, output_loading_info=True
                )
            self.assertTrue(loading["missing_keys"])
            self.assertTrue(
                all(key.startswith("action_head.rsci_query.") for key in loading["missing_keys"])
            )
            for key, value in native.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            query = restored.action_head.rsci_query
            self.assertTrue(
                all(torch.isfinite(parameter).all() for parameter in query.parameters())
            )
            self.assertGreater(query.query.std().item(), 0.01)
            self.assertGreater(query.reader.in_proj_weight.std().item(), 0.01)
            torch.testing.assert_close(
                query.ardy_head[0].weight, torch.ones_like(query.ardy_head[0].weight)
            )
            self.assertTrue(torch.isfinite(restored.eval()(batch())["loss"]))

    def test_invalid_history_keeps_loss_finite_and_graph_connected(self):
        policy, inputs = model("VGQ-AE"), batch()
        inputs["rsci_history_valid"][:, :3] = False
        inputs["rsci_ardy_target"].fill_(float("nan"))
        inputs["rsci_ardy_valid"][:] = False
        output = policy(inputs)
        self.assertEqual(output["ardy_loss"].item(), 0.0)
        output["loss"].backward()
        self.assertTrue(all(p.grad is not None for p in policy.action_head.rsci_query.parameters()))
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
        )

    def test_bf16_pretrained_loading_keeps_beta_sampling_fp32(self):
        # 复现真实入口的HF BF16加载上下文，而非仅在FP32构造完成后调用.to(BF16)。
        native = model(None)
        with tempfile.TemporaryDirectory() as directory:
            native.save_pretrained(directory)
            for variant in ("STATE", "NO-STATE", "VGQ-AE"):
                with self.subTest(variant=variant):
                    config = copy.deepcopy(native.config)
                    config.rsci_variant = variant
                    with (
                        patch(
                            "gr00t.model.gr00t_n1d6.gr00t_n1d6.get_backbone_cls",
                            return_value=TinyBackbone,
                        ),
                        patch("gr00t.model.gr00t_n1d6.processing_gr00t_n1d6.Gr00tN1d6DataCollator"),
                    ):
                        policy = Gr00tN1d6.from_pretrained(
                            directory,
                            config=config,
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True,
                        ).train()
                    for parameter in policy.parameters():
                        if parameter.requires_grad:
                            parameter.data = parameter.data.float()
                    distribution = policy.action_head.beta_dist
                    self.assertEqual(distribution.concentration1.dtype, torch.float32)
                    self.assertEqual(distribution.concentration1.device.type, "cpu")
                    sample = policy.action_head.sample_time(64, "cpu", torch.bfloat16)
                    self.assertEqual(sample.dtype, torch.bfloat16)
                    self.assertTrue(torch.isfinite(sample).all())
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        loss = policy(batch())["loss"]
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(
                        any(p.grad is not None for p in policy.action_head.model.parameters())
                    )
                    self.assertTrue(
                        all(
                            torch.isfinite(p.grad).all()
                            for p in policy.parameters()
                            if p.grad is not None
                        )
                    )

    def test_reader_preserves_image_text_and_time_layout(self):
        policy, inputs = model("VGQ-AE"), batch()
        frames = [policy.backbone(value) for value in inputs["rsci_history"]] + [
            policy.backbone(inputs)
        ]
        memory, valid, image = policy.action_head.rsci_query.memory(
            frames, inputs["rsci_history_valid"]
        )
        self.assertEqual(tuple(memory.shape), (2, 64, 16))
        self.assertEqual(int((valid & image).sum()), 64)
        self.assertEqual(int((valid & ~image).sum()), 64)

    def test_configuration_accepts_simple_top4_or_explicit_frozen(self):
        self.assertEqual(Gr00tN1d6Config(rsci_variant="VGQ-AE").tune_top_llm_layers, 4)
        self.assertEqual(
            Gr00tN1d6Config(rsci_variant="VGQ-AE", tune_top_llm_layers=0).tune_top_llm_layers, 0
        )
        with self.assertRaises(ValueError):
            Gr00tN1d6Config(rsci_variant="VGQ-AE", tune_llm=True)

    def test_simple_default_batch_follows_world_and_accumulation(self):
        from rsci.backbones.gr00t.train import parser, resolve

        for gpus, accumulation, batch in (
            ("0,1,2,3,4,5,6,7", 1, 3),
            ("0,1,2,3", 2, 3),
            ("0,1", 1, 12),
        ):
            args = parser().parse_args(
                [
                    "--dataset",
                    "/datasets/test",
                    "--sidecar",
                    "/labels/test",
                    "--task",
                    "hug",
                    "--gpus",
                    gpus,
                    "--gradient-accumulation-steps",
                    str(accumulation),
                ]
            )
            paths = resolve(args)
            self.assertEqual(args.batch_size, batch)
            self.assertEqual(args.vlm, "top4")
            self.assertEqual(paths["output"].name, "main")

    def test_native_top4_receives_gradients(self):
        from gr00t.model.modules.eagle_backbone import EagleBackbone

        backbone = EagleBackbone.__new__(EagleBackbone)
        torch.nn.Module.__init__(backbone)
        backbone.model = torch.nn.Module()
        backbone.model.language_model = torch.nn.Module()
        backbone.model.language_model.model = torch.nn.Module()
        layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(7)])
        backbone.model.language_model.model.layers = layers
        backbone.model.vision_model = torch.nn.Linear(8, 8)
        backbone.model.mlp1 = torch.nn.Linear(8, 8)
        backbone.set_trainable_parameters(False, False, 4)
        hidden = backbone.model.mlp1(backbone.model.vision_model(torch.randn(2, 8)))
        for layer in layers:
            hidden = layer(hidden)
        hidden.square().sum().backward()
        for index, layer in enumerate(layers):
            self.assertEqual(layer.weight.requires_grad, index >= 3)
            self.assertEqual(layer.weight.grad is not None, index >= 3)
        self.assertIsNone(backbone.model.vision_model.weight.grad)

    def test_masked_loss_is_valid_sample_mean(self):
        pred = torch.ones(2, 128, requires_grad=True)
        target = torch.zeros_like(pred)
        target[1] = float("nan")
        loss = masked_ardy_loss(pred, target, torch.tensor([True, False]))
        self.assertEqual(loss.item(), 1.0)
        loss.backward()
        self.assertEqual(pred.grad[1].abs().sum(), 0)

    def test_two_rank_training_with_one_rank_having_no_valid_teacher(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.start_processes(
                ddp_worker,
                args=(str(Path(directory) / "rendezvous"),),
                nprocs=2,
                join=True,
                start_method="spawn",
            )

    def test_inference_modality_is_one_frame_and_has_no_state(self):
        config = modality("VGQ-AE", training=False)["g1_loco_downstream"]
        self.assertEqual(config["video"].delta_indices, [0])
        self.assertEqual(config["state"].modality_keys, [])


class CheckpointTests(unittest.TestCase):
    def test_hf_trainer_writes_and_restores_real_optimizer_scheduler_rng(self):
        from transformers import Trainer, TrainingArguments

        with tempfile.TemporaryDirectory() as directory:
            training = TrainingArguments(
                output_dir=directory,
                max_steps=2,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                save_steps=1,
                logging_strategy="no",
                report_to="none",
                use_cpu=True,
                disable_tqdm=True,
                remove_unused_columns=False,
                save_safetensors=True,
            )
            examples = [{"inputs": batch()} for _ in range(4)]
            first = Trainer(
                model=model("VGQ-AE"),
                args=training,
                train_dataset=examples,
                data_collator=lambda rows: rows[0],
            )
            first.train()
            self.assertEqual(first.state.global_step, 2)
            self.assertEqual(first.state.epoch, 1.0)
            saved = Path(directory) / "checkpoint-2"
            for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
                (saved / name).write_text("{}")
            checkpoints.complete_checkpoint(saved, 1, "hf-integration")
            self.assertTrue(checkpoints.checkpoint_ok(saved, 1, "hf-integration", verify_hash=True))
            training.max_steps = 3
            restored = Trainer(
                model=model("VGQ-AE"),
                args=training,
                train_dataset=examples,
                data_collator=lambda rows: rows[0],
            )
            restored.train(resume_from_checkpoint=str(saved))
            self.assertEqual(restored.state.global_step, 3)
            self.assertTrue(
                all(state["step"].item() == 3 for state in restored.optimizer.state.values())
            )

    def checkpoint(self, root, step):
        path = root / f"checkpoint-{step}"
        path.mkdir()
        for name in (
            "config.json",
            "processor_config.json",
            "statistics.json",
            "embodiment_id.json",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
            "training_args.bin",
            "rng_state.pth",
            "model.safetensors",
        ):
            (path / name).write_bytes(json.dumps({"global_step": step}).encode())
        from safetensors.torch import save_file

        save_file({"weight": torch.ones(2, 2)}, str(path / "model.safetensors"))
        torch.save({"state": {0: {"exp_avg": torch.zeros(2, 2)}}}, path / "optimizer.pt")
        checkpoints.complete_checkpoint(path, 1, "fixture")
        return path

    def test_only_complete_matching_checkpoint_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.checkpoint(Path(directory), 1000)
            self.assertTrue(checkpoints.checkpoint_ok(path, 1, "fixture", verify_hash=True))
            self.assertFalse(checkpoints.checkpoint_ok(path, 2, "fixture"))
            self.assertFalse(checkpoints.checkpoint_ok(path, 1, "different"))
            (path / "optimizer.pt").write_bytes(b"bad")
            self.assertFalse(checkpoints.checkpoint_ok(path, 1, "fixture", verify_hash=True))

    def test_nonfinite_checkpoint_rejected_without_publishing_marker(self):
        from safetensors.torch import save_file

        for corrupt_model in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                path = self.checkpoint(Path(directory), 1000)
                (path / checkpoints.MARKER).unlink()
                if corrupt_model:
                    save_file(
                        {"weight": torch.tensor([float("nan")])}, str(path / "model.safetensors")
                    )
                else:
                    torch.save(
                        {"state": {0: {"exp_avg": torch.tensor([float("inf")])}}},
                        path / "optimizer.pt",
                    )
                with self.assertRaises(ValueError):
                    checkpoints.complete_checkpoint(path, 1, "fixture")
                self.assertFalse((path / checkpoints.MARKER).exists())
                self.assertTrue((path.parent / "NUMERICAL_FAILURE.json").exists())

    def test_nonfinite_loss_blocks_future_resume(self):
        from rsci.runtime.numerics import reject_numerical_failure, require_finite

        with tempfile.TemporaryDirectory() as directory:
            require_finite([torch.tensor(1.0)], directory, 0, "loss")
            with self.assertRaises(FloatingPointError):
                require_finite([torch.tensor(float("nan"))], directory, 1, "loss")
            with self.assertRaises(RuntimeError):
                reject_numerical_failure(directory)

    def test_keep_anchors_and_newest_rolling_only_after_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = self.checkpoint(root, 10000)
            old = self.checkpoint(root, 11000)
            current = root / "checkpoint-12000"
            current.mkdir()
            with self.assertRaises(ValueError):
                checkpoints.prune_rolling(root, current, 1, "fixture")
            self.assertTrue(old.exists())
            current.rmdir()
            current = self.checkpoint(root, 12000)
            checkpoints.prune_rolling(root, current, 1, "fixture")
            self.assertTrue(anchor.exists() and current.exists())
            self.assertFalse(old.exists())


class ProcessorAndServingTests(unittest.TestCase):
    def test_temporal_processor_keeps_conversations_separate_and_teacher_out(self):
        from gr00t.data.types import VLAStepData
        from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

        class EagleStub:
            tokenizer = SimpleNamespace(padding_side="left")

            def apply_chat_template(self, conversation, **kwargs):
                return conversation[0]["content"][0]["text"]

            def process_vision_info(self, conversations):
                return [
                    item["image"]
                    for conversation in conversations
                    for item in conversation[0]["content"]
                    if item["type"] == "image"
                ], None

            def __call__(self, text, images, **kwargs):
                # 只替代Eagle分词；真实状态/动作归一化、时序增强与collator均执行。
                ids = torch.tensor([[int(np.array(image).mean()), 1] for image in images])
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        stats = {
            TAG.value: {
                "state": {},
                "action": {
                    key: {
                        "min": [-1.0] * (stop - start),
                        "max": [1.0] * (stop - start),
                        "mean": [0.0] * (stop - start),
                        "std": [1.0] * (stop - start),
                    }
                    for key, (start, stop) in GROUPS.items()
                },
            }
        }
        with patch(
            "gr00t.model.gr00t_n1d6.processing_gr00t_n1d6.build_processor", return_value=EagleStub()
        ):
            processor = Gr00tN1d6Processor(
                modality("VGQ-AE"),
                statistics=stats,
                max_state_dim=128,
                max_action_dim=128,
                max_action_horizon=50,
                shortest_image_edge=24,
                crop_fraction=1.0,
                use_albumentations=True,
                rsci_variant="VGQ-AE",
            )
        sample = VLAStepData(
            images={
                "rs_view": [np.full((24, 32, 3), value, np.uint8) for value in (10, 20, 30, 40)]
            },
            states={},
            actions={
                key: np.zeros((16, stop - start), np.float32)
                for key, (start, stop) in GROUPS.items()
            },
            text="Pick box",
            embodiment=TAG,
            metadata=dict(
                rsci_history_valid=np.ones(4, dtype=bool),
                rsci_ardy_target=np.ones(128, np.float32),
                rsci_ardy_valid=True,
            ),
        )
        result = processor([{"type": "episode_step", "content": sample}])
        inputs = processor.collator([result])["inputs"]
        self.assertEqual(inputs["input_ids"][0, 0], 40)
        self.assertEqual(
            [frame["input_ids"][0, 0].item() for frame in inputs["rsci_history"]], [10, 20, 30]
        )
        self.assertEqual(inputs["action_mask"].sum(), 16 * 36)
        np.testing.assert_array_equal(inputs["state"], np.zeros((1, 1, 128)))
        for content in [result["vlm_content"], *result["rsci_history_content"]]:
            self.assertEqual(len(content["images"]), 1)
            self.assertEqual(content["text"], "pick box")
        processor.eval()
        sample.images["rs_view"] = sample.images["rs_view"][-1:]
        sample.actions, sample.metadata = {}, {}
        inference = processor([{"type": "episode_step", "content": sample}])
        self.assertNotIn("rsci_history_content", inference)
        self.assertNotIn("rsci_ardy_target", inference)
        decoded = processor.decode_action(np.zeros((1, 50, 128), np.float32), TAG, state={})
        self.assertEqual(sum(value.shape[-1] for value in decoded.values()), 36)
        self.assertTrue(
            all(value.shape[1] == 16 and np.isfinite(value).all() for value in decoded.values())
        )

    def test_simple_http_needs_no_state_and_returns_finite_36d(self):
        from gr00t.deploy.gr00t_serve_simple import Server

        from psi.deploy.helpers import RequestMessage, ResponseMessage

        server = Server.__new__(Server)
        server.modality_configs = modality("VGQ-AE", training=False)[TAG.value]
        server.action_exec_horizon = 16
        action = {
            key: np.zeros((1, 50, stop - start), np.float32)
            for key, (start, stop) in GROUPS.items()
        }

        def predict(observation):
            self.assertFalse(any(key.startswith("state.") for key in observation))
            self.assertEqual(observation["video.rs_view"].shape[:2], (1, 1))
            return action, {}

        server.policy = SimpleNamespace(get_action=predict)
        request = RequestMessage(
            image={"rs_view": np.zeros((24, 32, 3), np.uint8)},
            instruction="pick box",
            history={},
            state={},
            condition={},
            gt_action=[],
            dataset_name="test",
            timestamp="0",
        ).serialize()
        del request["state"]
        response = server.predict_action(request)
        self.assertEqual(response.status_code, 200, response.body)
        decoded = ResponseMessage.deserialize(json.loads(response.body))
        self.assertEqual(decoded.action.shape, (16, 36))
        action["height"].fill(np.nan)
        self.assertEqual(server.predict_action(request).status_code, 400)

    def test_native_serving_request_noise_survives_different_episode_lengths(self):
        from gr00t.deploy.gr00t_serve_simple import Server

        from psi.deploy.helpers import RequestMessage, ResponseMessage
        from rsci.backbones.gr00t.serve import build_server
        from rsci.inference.action_noise import PROTOCOL

        outputs = []
        for global_seed, prefix_length in ((2, 1), (900, 3)):
            server = Server.__new__(Server)
            server.modality_configs = modality("VGQ-AE", training=False)[TAG.value]
            server.action_exec_horizon = 16

            def sample(observation):
                values = torch.randn(size=(1, 50, 128), dtype=torch.bfloat16).float().numpy()
                return {key: values[..., start:stop] for key, (start, stop) in GROUPS.items()}, {}

            native_policy = SimpleNamespace(
                modality_configs=server.modality_configs,
                processor=SimpleNamespace(training=False),
                model=SimpleNamespace(training=False),
            )
            server.policy = SimpleNamespace(policy=native_policy, get_action=sample)
            with patch("gr00t.deploy.gr00t_serve_simple.Server", return_value=server):
                build_server(Path("/models/fixture"), "cpu", deterministic_eval=True, eval_seed=73)
            torch.manual_seed(global_seed)
            target = []
            for episode, count in ((0, prefix_length), (3, 2)):
                for index in range(count):
                    torch.randn(10 * (index + 1))
                    before = torch.get_rng_state().clone()
                    request = RequestMessage(
                        image={"rs_view": np.zeros((24, 32, 3), np.uint8)},
                        instruction="pick",
                        state={},
                        condition={},
                        gt_action=[],
                        dataset_name="fixture",
                        timestamp="0",
                        history={
                            "psi_eval_deterministic": dict(
                                protocol=PROTOCOL,
                                episode=episode,
                                episode_seed=73 + episode,
                                inference_index=index,
                            )
                        },
                    ).serialize()
                    response = server.predict_action(request)
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    if episode == 3:
                        target.append(ResponseMessage.deserialize(json.loads(response.body)).action)
            outputs.append(target)
            with self.assertRaisesRegex(ValueError, "consecutive"):
                server.predict_action(request)
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_nested_policy_accepts_missing_state_and_discards_extra_state(self):
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy.__new__(Gr00tPolicy)
        policy.modality_configs = modality("VGQ-AE", training=False)[TAG.value]
        policy.language_key = "annotation.human.task_description"
        value = {
            "video": {"rs_view": np.zeros((1, 1, 24, 32, 3), np.uint8)},
            "language": {policy.language_key: [["pick box"]]},
        }
        policy.check_observation(value)
        self.assertEqual(policy._unbatch_observation(value)[0]["state"], {})
        value["state"] = {"leaked": np.full((1, 1, 32), np.nan)}
        self.assertEqual(policy._unbatch_observation(value)[0]["state"], {})


if __name__ == "__main__":
    unittest.main()
