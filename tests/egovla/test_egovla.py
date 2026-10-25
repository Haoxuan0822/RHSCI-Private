"""真实官方AE/图文展开与微型Qwen验证；不下载或初始化全尺寸模型。"""

import ast
import copy
import importlib.util
import json
import re
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from rsci.backbones.egovla.checkpoints import (
    atomic_json,
    checkpoint_ok,
    load_trainables,
    prune,
    publish,
)
from rsci.backbones.egovla.model import EgoVLARSCI, decode_latents, masked_ardy_loss
from rsci.backbones.egovla.processing import Collator, Processor
from rsci.backbones.egovla.train import (
    CODE,
    assets,
    completion_callback,
    parser,
    resolve,
    trainer_type,
    training_arguments,
)


def native_decoder(width=16):
    spec = importlib.util.spec_from_file_location(
        "official_ego_transformer",
        CODE / "src/egovla/VILA/llava/model/ego_vla_decoder/transformer.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TransformerActionVector(width, 32, 36, False, False)


def native_multimodal():
    # 从已读取的官方源码抽取原方法，避免CPU测试导入无关CUDA/MANO依赖；函数本身不改。
    path = CODE / "src/egovla/VILA/llava/model/llava_arch.py"
    tree = ast.parse(path.read_text())
    method = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "LlavaMetaForCausalLM"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_inputs_labels_for_multimodal"
    )
    scope = dict(
        torch=torch,
        warnings=warnings,
        IMAGE_TOKEN_INDEX=-200,
        IGNORE_INDEX=-100,
        get_pg_manager=lambda: None,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method.name]


class Backbone(nn.Module):
    prepare_inputs_labels_for_multimodal = native_multimodal()

    def __init__(self):
        super().__init__()
        from transformers import Qwen2Config, Qwen2Model

        self.config = SimpleNamespace(
            traj_decoder_type="transformer_action_vector",
            action_output_dim=36,
            hidden_size=16,
            use_proprio=False,
            input_placeholder_end_token_idx=696,
            input_placeholder_start_token_idx=756,
        )
        self.llm = nn.Module()
        self.llm.model = Qwen2Model(
            Qwen2Config(
                vocab_size=1024,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                use_cache=False,
                attention_dropout=0.0,
                _attn_implementation="sdpa",
            )
        )
        self.llm.config = self.llm.model.config
        self.llm.config.tokenizer_padding_side = "right"
        self.llm.config.tokenizer_model_max_length = 4096
        self.vision = nn.Linear(3, 16)
        self.projector = nn.Linear(16, 16)
        self.traj_decoder = nn.Module()
        self.traj_decoder.decoder = native_decoder()

    @property
    def device(self):
        return next(self.parameters()).device

    def get_llm(self):
        return self.llm

    def get_vision_tower(self):
        return self.vision

    def get_mm_projector(self):
        return self.projector

    def get_traj_decoder(self):
        return self.traj_decoder

    def encode_images(self, images):
        features = self.projector(self.vision(images.mean((-1, -2))))
        return features[:, None].expand(-1, 4, -1)


def batch(size=2):
    ids = torch.tensor([[1, 20, -200, 21] + [756] * 30] * size)
    labels = torch.full_like(ids, -100)
    labels[:, -30:] = 756
    return dict(
        images=torch.randn(size, 4, 3, 4, 4),
        input_ids=ids,
        labels=labels,
        attention_mask=torch.ones_like(ids, dtype=torch.bool),
        actions=torch.randn(size, 30, 36),
        action_mask=torch.ones(size, 30, 36, dtype=torch.bool),
        history_valid=torch.tensor([[False, False, False, True], [True, True, True, True]][:size]),
        ardy_target=torch.randn(size, 128),
        ardy_valid=torch.ones(size, dtype=torch.bool),
    )


class TinyTokenizer:
    """仅替代词表资产；提示模板/占位/图像处理均调用生产官方函数。"""

    vocab_size = 1024
    pad_token_id = 0
    bos_token_id = 1
    padding_side = "right"

    def decode(self, tokens):
        return "".join(f"<tok{token}>" for token in tokens)

    def __call__(self, text):
        tokens = [1] + [
            int(value[4:-1]) if value.startswith("<tok") else 2 + ord(value) % 150
            for value in re.findall(r"<tok\d+>|.", text, re.DOTALL)
        ]
        return SimpleNamespace(input_ids=tokens)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.model = EgoVLARSCI(Backbone(), vlm_training="frozen")
        self.batch = batch()

    def test_official_decoder_exact_outputs_and_gradients(self):
        decoder = native_decoder().eval()
        latent = torch.randn(2, 30, 16, requires_grad=True)
        memory = torch.randn(2, 6, 16, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        original = decoder(latent.flatten(0, 1), dict(proprio=torch.randn(2, 32)), memory, mask)[
            "pred"
        ].reshape(2, 30, 36)
        replacement = decoder.output_projection(decode_latents(decoder, latent, memory, mask))
        torch.testing.assert_close(original, replacement, rtol=0, atol=0)
        parameters = (
            latent,
            memory,
            *tuple(
                parameter
                for name, parameter in decoder.named_parameters()
                if not name.startswith("proprio_projection")
            ),
        )
        gradients = torch.autograd.grad(original.square().sum(), parameters, retain_graph=True)
        new_gradients = torch.autograd.grad(replacement.square().sum(), parameters)
        for old, new in zip(gradients, new_gradients):
            torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_ardy_updates_shared_action_layers_without_vlm_or_action_head(self):
        self.model.eval()
        current = self.model.encode(
            self.batch["images"][:, -1],
            self.batch["input_ids"],
            self.batch["labels"],
            self.batch["attention_mask"],
        )
        frames = [
            self.model.encode(
                self.batch["images"][:, i],
                self.batch["input_ids"],
                self.batch["labels"],
                self.batch["attention_mask"],
            )
            for i in range(3)
        ] + [current]
        memory, mask = self.model.reader(frames, self.batch["history_valid"])
        hidden = decode_latents(
            self.model.decoder, self.model.query.expand(2, -1, -1), memory, mask
        )[:, 0]
        loss = masked_ardy_loss(
            self.model.ardy_head(hidden), self.batch["ardy_target"], self.batch["ardy_valid"]
        )
        loss.backward()
        self.assertGreater(self.model.decoder.layers[0].linear1.weight.grad.abs().sum(), 0)
        self.assertGreater(self.model.query.grad.abs().sum(), 0)
        self.assertIsNone(self.model.decoder.output_projection[0].weight.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in self.model.backbone.get_llm().parameters())
        )

    def test_history_cannot_enter_action_branch(self):
        self.model.eval()
        first = self.model(**self.batch)
        changed = {**self.batch, "images": self.batch["images"].clone()}
        changed["images"][:, :3] += 500
        second = self.model(**changed)
        torch.testing.assert_close(first["action_loss"], second["action_loss"], rtol=0, atol=0)
        self.assertNotEqual(first["ardy_loss"].item(), second["ardy_loss"].item())

    def test_one_current_encode_and_four_independent_frames(self):
        shapes = []
        handle = self.model.backbone.vision.register_forward_pre_hook(
            lambda module, values: shapes.append(values[0].shape)
        )
        self.model(**self.batch)
        handle.remove()
        self.assertEqual(shapes, [torch.Size([2, 3])] * 4)

    def test_inference_does_not_touch_reader_query_or_history(self):
        self.model.eval()
        self.model.reader.forward = lambda *args: self.fail(
            "Reader ran during plain VGQ-AE inference"
        )
        inputs = {
            key: self.batch[key] for key in ("images", "input_ids", "labels", "attention_mask")
        }
        inputs["images"] = inputs["images"][:, -1:]
        first = self.model.sample_actions(**inputs)
        self.model.query.data.add_(10000)
        second = self.model.sample_actions(**inputs)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(first.shape, (2, 30, 36))

    def test_masked_invalid_target_and_all_invalid_are_finite(self):
        prediction = torch.randn(2, 128, requires_grad=True)
        target = torch.full((2, 128), float("nan"))
        loss = masked_ardy_loss(prediction, target, torch.zeros(2, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(prediction.grad.abs().sum().item(), 0)

    def test_frozen_eval_and_ae_native_dropout(self):
        self.model.train()
        self.assertFalse(self.model.backbone.llm.training)
        self.assertFalse(self.model.backbone.vision.training)
        self.assertTrue(self.model.decoder.training)
        self.assertEqual(self.model.decoder.layers[0].dropout.p, 0.1)
        self.assertFalse(self.model.decoder.proprio_projection[0].weight.requires_grad)

    def test_incomplete_history_and_wrong_placeholders_rejected(self):
        changed = {**self.batch, "history_valid": torch.zeros(2, 4, dtype=torch.bool)}
        with self.assertRaises(ValueError):
            self.model(**changed)
        changed = {**self.batch, "labels": torch.full_like(self.batch["labels"], -100)}
        with self.assertRaises(ValueError):
            self.model(**changed)


class FullVLMTests(unittest.TestCase):
    def test_checkpointed_four_frame_losses_and_gradients_match(self):
        torch.manual_seed(13)
        reference = EgoVLARSCI(Backbone()).train()
        recomputed = copy.deepcopy(reference)
        recomputed.gradient_checkpointing_enable()
        self.assertTrue(recomputed.backbone.llm.model.is_gradient_checkpointing)
        inputs = batch()
        torch.manual_seed(27)
        expected = reference(**inputs)
        expected["loss"].backward()
        torch.manual_seed(27)
        actual = recomputed(**inputs)
        actual["loss"].backward()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        for (name, left), (_, right) in zip(
            reference.named_parameters(), recomputed.named_parameters()
        ):
            self.assertEqual(left.grad is None, right.grad is None, name)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)

    def test_native_siglip_checkpointing_gradients_match(self):
        from llava.model.multimodal_encoder.siglip import SiglipVisionModel
        from transformers import SiglipVisionConfig

        config = SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
            attention_dropout=0.0,
        )
        config._attn_implementation = "sdpa"
        vision = SiglipVisionModel(config).train()
        recomputed = copy.deepcopy(vision)
        recomputed.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        images = torch.randn(2, 3, 28, 28)
        vision(images).last_hidden_state.square().mean().backward()
        recomputed(images).last_hidden_state.square().mean().backward()
        self.assertTrue(recomputed.is_gradient_checkpointing)
        for (name, left), (_, right) in zip(
            vision.named_parameters(), recomputed.named_parameters()
        ):
            self.assertEqual(left.grad is None, right.grad is None, name)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)

    def test_both_losses_update_vision_projector_and_language(self):
        for branch in ("action", "ardy"):
            torch.manual_seed(7)
            model = EgoVLARSCI(Backbone()).eval()
            inputs = batch()
            images = inputs["images"].requires_grad_()
            current = model.encode(
                images[:, -1], inputs["input_ids"], inputs["labels"], inputs["attention_mask"]
            )
            if branch == "action":
                loss = (model.action_prediction(current) - inputs["actions"]).square().mean()
            else:
                frames = [
                    model.encode(
                        images[:, i],
                        inputs["input_ids"],
                        inputs["labels"],
                        inputs["attention_mask"],
                    )
                    for i in range(3)
                ] + [current]
                memory, mask = model.reader(frames, inputs["history_valid"])
                latent = decode_latents(model.decoder, model.query.expand(2, -1, -1), memory, mask)[
                    :, 0
                ]
                loss = masked_ardy_loss(
                    model.ardy_head(latent), inputs["ardy_target"], inputs["ardy_valid"]
                )
            loss.backward()
            for module in (model.backbone.vision, model.backbone.projector, model.backbone.llm):
                self.assertGreater(
                    sum(
                        p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None
                    ),
                    0,
                    branch,
                )
            if branch == "action":
                self.assertEqual(images.grad[:, :3].abs().sum().item(), 0)
            else:
                self.assertGreater(images.grad[1, :3].abs().sum().item(), 0)
                self.assertIsNone(model.decoder.output_projection[0].weight.grad)

    def test_native_vlm_precision_mode_and_explicit_frozen_output(self):
        model = EgoVLARSCI(Backbone().bfloat16())
        self.assertEqual(next(model.backbone.llm.parameters()).dtype, torch.bfloat16)
        self.assertEqual(next(model.backbone.vision.parameters()).dtype, torch.bfloat16)
        self.assertTrue(
            all(p.dtype == torch.float32 for p in model.decoder.parameters() if p.requires_grad)
        )
        self.assertTrue(model.backbone.llm.training)
        self.assertTrue(model.backbone.vision.training)
        model.eval()
        self.assertFalse(model.backbone.llm.training)
        full = parser().parse_args(
            ["--dataset", "/datasets/test", "--sidecar", "/labels/test", "--task", "hug"]
        )
        frozen = parser().parse_args(
            [
                "--dataset",
                "/datasets/test",
                "--sidecar",
                "/labels/test",
                "--task",
                "hug",
                "--vlm-training",
                "frozen",
            ]
        )
        self.assertEqual(full.vlm_training, "full")
        self.assertNotEqual(resolve(full)["output"], resolve(frozen)["output"])
        self.assertTrue(
            training_arguments(full, resolve(full), cpu=True).ddp_find_unused_parameters
        )
        self.assertFalse(
            training_arguments(frozen, resolve(frozen), cpu=True).ddp_find_unused_parameters
        )


class OfficialStateTests(unittest.TestCase):
    def test_checkpoint_preserves_state_projection_and_rejects_other_variant(self):
        model = EgoVLARSCI(Backbone(), variant="STATE")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            trainer_type()(
                model=model,
                args=training_arguments(
                    parser().parse_args(
                        [
                            "--dataset",
                            "/datasets/test",
                            "--sidecar",
                            "/labels/test",
                            "--task",
                            "hug",
                        ]
                    ),
                    dict(output=path),
                    cpu=True,
                ),
                launch_contract={},
            )._save(path)
            restored = EgoVLARSCI(Backbone(), variant="STATE")
            load_trainables(restored, path)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    torch.testing.assert_close(
                        dict(restored.named_parameters())[name], parameter, rtol=0, atol=0
                    )
            with self.assertRaises(ValueError):
                load_trainables(EgoVLARSCI(Backbone()), path)

    def test_native_state_decoder_loss_and_state_gradient(self):
        torch.manual_seed(9)
        model = EgoVLARSCI(Backbone(), variant="STATE").eval()
        sample = batch()
        images = sample["images"][:, -1:]
        states = torch.randn(2, 32, requires_grad=True)
        current = model.encode(
            images[:, 0], sample["input_ids"], sample["labels"], sample["attention_mask"]
        )
        features, mask, selected = current
        expected = model.decoder(features[selected], dict(proprio=states), features, mask)[
            "pred"
        ].reshape(2, 30, 36)
        torch.testing.assert_close(
            model.action_prediction(current, states), expected, rtol=0, atol=0
        )
        loss = model(
            images,
            sample["input_ids"],
            sample["labels"],
            sample["attention_mask"],
            sample["actions"],
            sample["action_mask"],
            states=states,
        )
        self.assertNotIn("ardy_loss", loss)
        loss["loss"].backward()
        self.assertGreater(states.grad.abs().sum(), 0)
        self.assertGreater(model.decoder.proprio_projection[0].weight.grad.abs().sum(), 0)
        self.assertFalse(hasattr(model, "reader"))
        self.assertFalse(hasattr(model, "query"))
        torch.testing.assert_close(
            model.sample_actions(
                images,
                sample["input_ids"],
                sample["labels"],
                sample["attention_mask"],
                states=states,
            ),
            expected,
            rtol=0,
            atol=0,
        )
        with self.assertRaises(ValueError):
            model.sample_actions(
                images, sample["input_ids"], sample["labels"], sample["attention_mask"]
            )

    def test_custom_dataset_and_training_budget(self):
        args = parser().parse_args(
            [
                "--task",
                "custom",
                "--model",
                "STATE",
                "--dataset",
                "/datasets/custom",
                "--output-dir",
                "/runs/custom",
                "--max-steps",
                "2000",
            ]
        )
        paths = resolve(args)
        self.assertEqual(paths["dataset"], Path("/datasets/custom"))
        self.assertEqual(paths["output"], Path("/runs/custom"))
        self.assertEqual(training_arguments(args, paths, cpu=True).max_steps, 2000)

    def test_epoch_budget_and_shared_recipe(self):
        # 45046有效窗口；8卡×16得到352 microbatch/epoch，累计4 -> 88更新/epoch。
        with tempfile.TemporaryDirectory() as temp:
            import os

            root = Path(temp)
            meta = root / "dataset/meta"
            meta.mkdir(parents=True)
            (meta / "episodes.jsonl").write_text(json.dumps(dict(length=45075)) + "\n")
            with patch.dict(os.environ, PSI_RUNTIME_ROOT=temp):
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
                os.environ.pop("NCCL_SHM_DISABLE", None)
                recipes = []
                for variant in ("VGQ-AE", "STATE"):
                    args = parser().parse_args(
                        [
                            "--dataset",
                            str(meta.parent),
                            "--sidecar",
                            "/labels/test",
                            "--task",
                            "hug",
                            "--gpus",
                            "0,1,2,3,4,5,6,7",
                            "--model",
                            variant,
                            "--preset",
                            "simple-115epoch",
                            "--batch-size",
                            "16",
                            "--gradient-accumulation-steps",
                            "4",
                        ]
                    )
                    paths = resolve(args)
                    self.assertEqual(
                        os.environ["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True"
                    )
                    self.assertEqual(os.environ["NCCL_SHM_DISABLE"], "0")
                    self.assertEqual(args.max_steps, 10120)
                    actual = training_arguments(args, paths, cpu=True)
                    self.assertEqual(actual.max_steps, -1)
                    self.assertEqual(actual.num_train_epochs, 115)
                    recipes.append(
                        (
                            actual.learning_rate,
                            actual.per_device_train_batch_size,
                            actual.gradient_accumulation_steps,
                            actual.lr_scheduler_type,
                        )
                    )
                self.assertEqual(recipes[0], recipes[1])
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                os.environ["NCCL_SHM_DISABLE"] = "1"
                args.max_steps = None
                resolve(args)
                self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:128")
                self.assertEqual(os.environ["NCCL_SHM_DISABLE"], "1")

    def test_epoch_preset_rejects_a_different_budget_or_global_batch(self):
        base = ["--task", "example", "--preset", "simple-115epoch"]
        for overrides in (
            ["--gpus", "0,1"],
            ["--gpus", "0,1,2,3,4,5,6,7", "--max-steps", "40000"],
            ["--gpus", "0,1,2,3,4,5,6,7", "--vlm-training", "frozen"],
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve(parser().parse_args(base + overrides))


class ProcessorTests(unittest.TestCase):
    def setUp(self):
        from transformers import SiglipImageProcessor

        self.processor = Processor(
            TinyTokenizer(), SiglipImageProcessor(size=dict(height=16, width=16))
        )

    def test_official_prompt_actions_raw_and_rgb(self):
        sample = dict(
            images=np.full((4, 12, 20, 3), 100, dtype=np.uint8),
            text="carry the box",
            actions=np.full((30, 36), 50.0, dtype=np.float32),
            history_valid=np.array([0, 0, 0, 1]),
            ardy_target=np.zeros(128, dtype=np.float32),
            ardy_valid=True,
        )
        output = self.processor(sample)
        self.assertEqual(output["actions"].max().item(), 50.0)
        self.assertEqual(output["images"].shape, (4, 3, 16, 16))
        self.assertEqual(
            (
                output["input_ids"] == self.processor.action_tokenizer.input_placeholder_token_idx
            ).sum(),
            30,
        )
        sample["actions"] *= -100
        second = self.processor(sample)
        torch.testing.assert_close(output["input_ids"], second["input_ids"], rtol=0, atol=0)
        self.assertFalse(any("proprio" in key or "state" in key for key in output))

    def test_collator_and_inference_use_same_current(self):
        image = np.full((12, 20, 3), 50, dtype=np.uint8)
        sample = dict(
            images=np.stack([image] * 4),
            text="lift",
            actions=np.zeros((30, 36), dtype=np.float32),
            history_valid=np.ones(4, dtype=bool),
            ardy_target=np.zeros(128, dtype=np.float32),
            ardy_valid=True,
        )
        item = self.processor(sample)
        other = self.processor({**sample, "text": "lift the large box"})
        collated = Collator(0)([item, other])
        infer = self.processor.inference(image, "lift")
        torch.testing.assert_close(collated["images"][:1, -1:], infer["images"])
        self.assertTrue((collated["labels"][~collated["attention_mask"]] == -100).all())

    def test_simple_inference_ignores_state_and_preserves_numpy_protocol(self):
        from rsci.backbones.egovla.serve import decode_numpy, encode_numpy, predict

        model = EgoVLARSCI(Backbone()).eval()
        image = np.full((12, 20, 3), 50, dtype=np.uint8)
        payload = dict(image=dict(rgb_head_stereo_left=image), instruction="lift")
        expected = predict(model, self.processor, payload)
        actual = predict(
            model, self.processor, {**payload, "state": {"states": np.full(32, np.nan)}}
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(decode_numpy(encode_numpy(actual)), expected)

    def test_state_processor_and_server_require_actual_state(self):
        from rsci.backbones.egovla.serve import predict

        image = np.full((12, 20, 3), 50, dtype=np.uint8)
        state = np.arange(32, dtype=np.float32) / 32
        item = self.processor(
            dict(
                images=image[None],
                text="lift",
                actions=np.zeros((30, 36), dtype=np.float32),
                states=state,
            )
        )
        self.assertEqual(
            set(item), {"images", "input_ids", "labels", "actions", "action_mask", "states"}
        )
        torch.testing.assert_close(item["states"], torch.from_numpy(state))
        model = EgoVLARSCI(Backbone(), variant="STATE").eval()
        payload = dict(
            image=dict(rgb_head_stereo_left=image), instruction="lift", state=dict(states=state)
        )
        first = predict(model, self.processor, payload)
        second = predict(model, self.processor, {**payload, "state": dict(states=state + 10)})
        self.assertFalse(np.array_equal(first, second))
        with self.assertRaises(KeyError):
            predict(model, self.processor, {k: v for k, v in payload.items() if k != "state"})


class RecipeAndCheckpointTests(unittest.TestCase):
    @patch("rsci.backbones.egovla.checkpoints.prune", lambda *args: None)
    def test_resume_mid_later_epochs_with_accumulation(self):
        from torch.utils.data import Dataset
        from transformers import set_seed

        class Indexed(Dataset):
            def __init__(self):
                self.samples = batch(1)

            def __len__(self):
                return 8

            def __getitem__(self, index):
                sample = {key: value[0].clone() for key, value in self.samples.items()}
                sample["actions"] += index * 0.17
                sample["ardy_target"] += index * 0.11
                return sample

        class RecordingTrainer(trainer_type()):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.seen = []

            def training_step(self, model, inputs, *args, **kwargs):
                self.seen.extend(inputs["actions"][:, 0, 0].tolist())
                return super().training_step(model, inputs, *args, **kwargs)

        with tempfile.TemporaryDirectory() as root:
            args = parser().parse_args(
                [
                    "--dataset",
                    "/datasets/test",
                    "--sidecar",
                    "/labels/test",
                    "--task",
                    "between",
                    "--batch-size",
                    "1",
                    "--gradient-accumulation-steps",
                    "4",
                    "--num-workers",
                    "0",
                    "--max-steps",
                    "6",
                ]
            )
            set_seed(23)
            initial, dataset = EgoVLARSCI(Backbone()), Indexed()
            arguments = training_arguments(args, dict(output=Path(root) / "continuous"), cpu=True)
            arguments.save_steps, arguments.disable_tqdm = 1, True
            continuous = RecordingTrainer(
                model=copy.deepcopy(initial),
                args=arguments,
                train_dataset=dataset,
                launch_contract={"test": "later-epochs"},
            )
            continuous.train()
            expected = {
                name: parameter.detach().clone()
                for name, parameter in continuous.model.named_parameters()
                if parameter.requires_grad
            }
            expected_rng = torch.get_rng_state().clone()
            for step in (2, 3, 4, 5):
                with self.subTest(resume_step=step):
                    resumed_args = copy.deepcopy(arguments)
                    resumed_args.output_dir = str(Path(root) / f"resumed-{step}")
                    resumed = RecordingTrainer(
                        model=copy.deepcopy(initial),
                        args=resumed_args,
                        train_dataset=dataset,
                        launch_contract={"test": "later-epochs"},
                    )
                    resumed.train(
                        resume_from_checkpoint=str(Path(root) / "continuous" / f"checkpoint-{step}")
                    )
                    self.assertEqual(resumed.seen, continuous.seen[step * 4 :])
                    self.assertEqual(resumed.state.global_step, 6)
                    for name, parameter in resumed.model.named_parameters():
                        if parameter.requires_grad:
                            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
                    torch.testing.assert_close(
                        resumed.optimizer.state_dict(),
                        continuous.optimizer.state_dict(),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(
                        resumed.lr_scheduler.state_dict(), continuous.lr_scheduler.state_dict()
                    )
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_skip_preserves_epoch_in_two_rank_batch_shards_without_global_patch(self):
        from accelerate import skip_first_batches
        from accelerate.data_loader import prepare_data_loader
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import Trainer

        from rsci.runtime.hf_resume import epoch_safe_training_loop, skip_preserving_epoch

        native = Trainer._inner_training_loop
        repaired = epoch_safe_training_loop(Trainer)
        self.assertIs(repaired.__code__, native.__code__)
        self.assertIs(native.__globals__["skip_first_batches"], skip_first_batches)
        self.assertIs(repaired.__globals__["skip_first_batches"], skip_preserving_epoch)
        for rank in (0, 1):

            def loader():
                return prepare_data_loader(
                    DataLoader(TensorDataset(torch.arange(32)), batch_size=2, shuffle=True),
                    num_processes=2,
                    process_index=rank,
                    use_seedable_sampler=True,
                    data_seed=292285,
                )

            for epoch in (1, 2):
                with self.subTest(rank=rank, epoch=epoch):
                    full = loader()
                    full.set_epoch(epoch)
                    expected = [batch[0].tolist() for batch in full]
                    resumed = loader()
                    resumed.set_epoch(epoch)
                    actual = [batch[0].tolist() for batch in skip_preserving_epoch(resumed, 3)]
                    self.assertEqual(actual, expected[3:])

    def test_asset_inventory_rejects_new_loader_preferred_weights(self):
        from transformers import Qwen2Config, Qwen2Model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "base"
            model = Qwen2Model(
                Qwen2Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                )
            )
            with torch.no_grad():
                model.get_input_embeddings().weight.fill_(0.1)
            model.save_pretrained(root, safe_serialization=False)
            for part in ("llm", "vision_tower", "mm_projector"):
                (root / part).mkdir()
                atomic_json(root / part / "config.json", {})
            assets(root, register=True)
            before = (
                Qwen2Model.from_pretrained(root, local_files_only=True)
                .get_input_embeddings()
                .weight.detach()
            )
            with torch.no_grad():
                model.get_input_embeddings().weight.fill_(0.2)
            model.save_pretrained(Path(directory) / "replacement", safe_serialization=True)
            shutil.copyfile(
                Path(directory) / "replacement/model.safetensors", root / "model.safetensors"
            )
            after = (
                Qwen2Model.from_pretrained(root, local_files_only=True)
                .get_input_embeddings()
                .weight.detach()
            )
            self.assertFalse(torch.equal(before, after))  # 实际HF loader确实优先读取新增权重。
            for verify in (True, False):
                with self.subTest(verify=verify), self.assertRaises(ValueError):
                    assets(root, verify=verify)

    def test_asset_inventory_covers_nested_hidden_indexes_and_removed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "config.json",
                "llm/config.json",
                "vision_tower/config.json",
                "mm_projector/config.json",
            ):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                atomic_json(root / name, {})
            torch.save({"weight": torch.ones(1)}, root / "pytorch_model.bin")
            assets(root, register=True)
            assets(root)  # 原目录可以重复验收，根清单自身不会被误算为新增文件。
            for name in (
                "llm/model.safetensors.index.json",
                "llm/tokenizer_config.json",
                ".hidden/weights.bin",
            ):
                with self.subTest(addition=name):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(path, {})
                    for verify in (True, False):
                        with self.assertRaisesRegex(ValueError, "inventory changed"):
                            assets(root, verify=verify)
                    path.unlink()
            (root / "llm/config.json").unlink()
            with self.assertRaisesRegex(ValueError, "removed=.*llm/config.json"):
                assets(root)

    def test_asset_hash_rejects_same_size_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "config.json",
                "llm/config.json",
                "vision_tower/config.json",
                "mm_projector/config.json",
            ):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                atomic_json(root / name, {})
            torch.save({"weight": torch.ones(1)}, root / "pytorch_model.bin")
            assets(root, register=True)
            path = root / "pytorch_model.bin"
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                assets(root)

    def test_official_recipe_and_actual_constant_schedule(self):
        with tempfile.TemporaryDirectory() as root:
            args = parser().parse_args(
                [
                    "--dataset",
                    "/datasets/test",
                    "--sidecar",
                    "/labels/test",
                    "--task",
                    "between",
                    "--num-workers",
                    "0",
                ]
            )
            paths = resolve(args)
            paths["output"] = Path(root)
            parameters = training_arguments(args, paths, cpu=True)
            trainer = trainer_type()(
                model=EgoVLARSCI(Backbone()), args=parameters, launch_contract={}
            )
            trainer.create_optimizer_and_scheduler(40000)
            self.assertEqual(trainer.optimizer.defaults["betas"], (0.9, 0.999))
            self.assertEqual(trainer.optimizer.defaults["eps"], 1e-8)
            for step in (0, 1, 1200, 20000, 40000):
                self.assertEqual(trainer.lr_scheduler.lr_lambdas[0](step), 1.0)
            self.assertEqual(parameters.learning_rate, 2e-5)
            self.assertEqual(parameters.weight_decay, 0.0)
            self.assertEqual(parameters.gradient_accumulation_steps, 4)

    @patch("rsci.backbones.egovla.checkpoints.prune", lambda *args: None)
    def test_hf_full_resume_model_adam_rng_and_step(self):
        from torch.utils.data import Dataset
        from transformers import set_seed

        class Fixed(Dataset):
            def __init__(self):
                self.samples = batch(1)

            def __len__(self):
                return 8

            def __getitem__(self, index):
                sample = {key: value[0].clone() for key, value in self.samples.items()}
                sample["actions"] += index * 0.17
                sample["ardy_target"] += index * 0.11
                return sample

        with tempfile.TemporaryDirectory() as root:
            args = parser().parse_args(
                [
                    "--dataset",
                    "/datasets/test",
                    "--sidecar",
                    "/labels/test",
                    "--task",
                    "between",
                    "--batch-size",
                    "1",
                    "--gradient-accumulation-steps",
                    "1",
                    "--num-workers",
                    "0",
                    "--max-steps",
                    "2",
                ]
            )
            paths = dict(output=Path(root))
            set_seed(11)
            initial = EgoVLARSCI(Backbone())
            dataset = Fixed()
            arguments = training_arguments(args, paths, cpu=True)
            arguments.save_steps = 1
            arguments.disable_tqdm = True
            value, digest = {"test": True}, "fixture"
            trainer = trainer_type()(
                model=copy.deepcopy(initial),
                args=arguments,
                train_dataset=dataset,
                launch_contract=value,
                callbacks=[completion_callback(Path(root), 1, digest)],
            )
            trainer.train()
            expected = {
                name: parameter.detach().clone()
                for name, parameter in trainer.model.named_parameters()
                if parameter.requires_grad
            }
            restored = trainer_type()(
                model=copy.deepcopy(initial),
                args=arguments,
                train_dataset=dataset,
                launch_contract=value,
            )
            restored.train(resume_from_checkpoint=str(Path(root) / "checkpoint-1"))
            self.assertEqual(restored.state.global_step, 2)
            for name, parameter in restored.model.named_parameters():
                if parameter.requires_grad:
                    torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
            self.assertTrue(checkpoint_ok(Path(root) / "checkpoint-2", 1, digest, verify=True))
            with patch(
                "rsci.backbones.egovla.checkpoints.validate_numerics",
                side_effect=FloatingPointError("fixture"),
            ):
                self.assertFalse(checkpoint_ok(Path(root) / "checkpoint-2", 1, digest, verify=True))
            # 验证真实清理函数仅删除已验收的旧滚动点，保留每10000步里程碑。
            for step in (10000, 20000):
                destination = Path(root) / f"checkpoint-{step}"
                shutil.copytree(
                    Path(root) / ("checkpoint-2" if step == 10000 else "checkpoint-10000"),
                    destination,
                )
                state = json.loads((destination / "trainer_state.json").read_text())
                state["global_step"] = step
                atomic_json(destination / "trainer_state.json", state)
                publish(destination, 1, digest)
                prune(Path(root), destination, 1, digest)
            self.assertFalse((Path(root) / "checkpoint-1").exists())
            self.assertFalse((Path(root) / "checkpoint-2").exists())
            self.assertTrue(checkpoint_ok(Path(root) / "checkpoint-10000", 1, digest, verify=True))


if __name__ == "__main__":
    unittest.main()
