"""真实官方 AE/flow/processor/数据边界的 CPU 验收，不代表 CUDA 长跑。"""

import copy
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from lambda0.models.action_expert import LayerwiseActionExpert
from lambda0.models.heterogeneous_action_space import HeterogeneousActionSpace
from lambda0.models.modeling_lambda0 import Lambda0
from torch import nn

from rsci.backbones.lambda0.data import ResumableLoader, local_rng
from rsci.backbones.lambda0.model import Lambda0RSCI, masked_ardy_loss
from rsci.backbones.lambda0.runtime import (
    RSCITrainingBackend,
    checkpoint_receipt,
    verify_checkpoint,
)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 16)
        self.image = nn.Linear(3, 16)

    def forward(self, input_ids, pixel_values, **kwargs):
        feature = self.embedding(input_ids) + self.image(pixel_values).unsqueeze(1)
        return SimpleNamespace(hidden_states=(feature, feature.tanh()))


def tiny(variant="STATE"):
    model = Lambda0RSCI.__new__(Lambda0RSCI)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vlm_hidden_size=16,
        expert_hidden_size=16,
        expert_num_layers=2,
        state_conditioning_mode="discrete_vlm"
        if variant == "STATE"
        else "continuous_action_expert",
        state_discrete_bins=256,
        rsci_variant=variant,
        prediction_type="velocity",
        num_domains=8,
        num_actions=50,
    )
    model.vlm = TinyVLM()
    model.num_actions, model.use_proprio = 50, variant == "STATE"
    model.state_conditioning_mode = model.config.state_conditioning_mode
    model.action_space = HeterogeneousActionSpace(hidden_size=16, dim_time=8)
    model.max_action_dim, model.max_proprio_dim = (
        model.action_space.max_action_dim,
        model.action_space.max_proprio_dim,
    )
    model.action_expert = LayerwiseActionExpert(16, 16, 2, 4, 2.0, 0.0, 8, 2, 50)
    if variant == "VGQ-AE":
        model.install_auxiliary()
    return model


def vlm_inputs(batch=2):
    return dict(
        input_ids=torch.randint(0, 20, (batch, 5)),
        attention_mask=torch.ones(batch, 5, dtype=torch.long),
        pixel_values=torch.randn(batch, 3),
        image_grid_thw=torch.ones(batch, 3, dtype=torch.long),
        mm_token_type_ids=torch.zeros(batch, 5, dtype=torch.int),
    )


def inputs(model):
    return dict(
        **vlm_inputs(),
        proprio=torch.randn(2, model.max_proprio_dim),
        action=torch.randn(2, 50, model.max_action_dim),
        action_space_id=torch.tensor([3, 3]),
        embodiment_id=torch.tensor([7, 7]),
        noise_seed=917,
        action_expert_noise_samples=2,
        rtc_max_delay=8,
    )


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_backend_loading_is_explicit_and_does_not_change_native_engine(self):
        from lambda0.training import engine

        from rsci.backbones.lambda0.optimization import native_arguments

        operations = (
            "load_model_and_processor",
            "prepare_inputs",
            "create_dataloader",
            "build_optimizer",
            "save_training_checkpoint",
            "load_training_checkpoint",
        )
        original = {name: getattr(engine, name) for name in operations}
        backends = [
            RSCITrainingBackend(engine, variant, {"variant": variant})
            for variant in ("STATE", "VGQ-AE")
        ]
        for backend in backends:
            args = native_arguments({"models": "fixture", "train_metas_path": "fixture"})
            args.require_fast_linear_attention = False
            args.gradient_checkpointing = False
            args.action_expert_gradient_checkpointing = False
            args.reset_action_space_ids = []
            model = tiny("STATE")
            with (
                patch.object(Lambda0RSCI, "from_pretrained_or_vlm", return_value=model) as load,
                patch.object(engine.Lambda0Processor, "from_pretrained", return_value=object()),
            ):
                actual, _ = backend.load_model_and_processor(args)
            load.assert_called_once()
            self.assertIs(actual, model)
            self.assertIs(type(actual), Lambda0RSCI)
            self.assertEqual(actual.config.rsci_variant, backend.variant)
            self.assertEqual(actual.config.rsci_contract, backend.contract)
            if backend.variant == "VGQ-AE":
                self.assertFalse(actual.use_proprio)
                self.assertTrue(hasattr(actual, "rsci_reader"))
            for operation, function in original.items():
                self.assertIs(getattr(engine, operation), function)
        self.assertEqual([backend.variant for backend in backends], ["STATE", "VGQ-AE"])

    def test_serving_normalization_is_explicit_and_keeps_native_defaults(self):
        from fastapi.testclient import TestClient
        from lambda0.inference import server

        from rsci.backbones.lambda0.contracts import SAFE_NORMALIZATION

        model = tiny("STATE").eval()
        spec = model.action_space.specs[3]
        normalizer = SimpleNamespace(
            action_space=spec.name,
            action_dim=spec.action_dim,
            proprio_dim=spec.proprio_dim,
            coordinate_contract=server.SIMPLE_COORDINATE_CONTRACT,
            normalization_contract=SAFE_NORMALIZATION,
            dataset_name="fixture",
        )
        original = server.QUANTILE_NORMALIZATION_CONTRACT
        with self.assertRaisesRegex(ValueError, "normalization contract"):
            server.build_app(model, None, torch.device("cpu"), normalizer=normalizer)
        app = server.build_app(
            model,
            None,
            torch.device("cpu"),
            normalizer=normalizer,
            normalization_contract=SAFE_NORMALIZATION,
        )
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["normalization_dataset"], "fixture")
        self.assertEqual(server.QUANTILE_NORMALIZATION_CONTRACT, original)

    def test_state_native_forward_parity(self):
        model = tiny()
        data = inputs(model)
        native = Lambda0.forward(model, **data)
        actual = model(**data)
        for key in native:
            torch.testing.assert_close(native[key], actual[key], rtol=0, atol=0)

    def test_vgq_shared_gradients_and_state_independence(self):
        model = tiny("VGQ-AE")
        data = inputs(model)
        history = [vlm_inputs() for _ in range(3)]
        extra = dict(
            history=history,
            history_valid=torch.ones(2, 4, dtype=torch.bool),
            ardy_target=torch.randn(2, 128),
            ardy_valid=torch.ones(2, dtype=torch.bool),
        )
        current = Lambda0.forward_vlm(model, **{key: data[key] for key in vlm_inputs()})
        prediction = model.auxiliary(
            current, history, extra["history_valid"], data["embodiment_id"]
        )
        masked_ardy_loss(prediction, extra["ardy_target"], extra["ardy_valid"]).backward()
        auxiliary = {
            name
            for name, value in model.named_parameters()
            if value.grad is not None and value.grad.abs().sum() > 0
        }
        self.assertTrue(any(name.startswith("action_expert.blocks") for name in auxiliary))
        self.assertTrue(any(name.startswith("vlm.") for name in auxiliary))
        model.zero_grad(set_to_none=True)
        Lambda0.forward(model, **data)["loss"].backward()
        action = {
            name
            for name, value in model.named_parameters()
            if value.grad is not None and value.grad.abs().sum() > 0
        }
        self.assertTrue(any(name.startswith("action_expert.blocks") for name in auxiliary & action))
        model.zero_grad(set_to_none=True)
        first = model(**data, **extra)["loss"]
        data["proprio"] = torch.randn_like(data["proprio"]) * 1000
        second = model(**data, **extra)["loss"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_inference_removes_reader_and_state(self):
        model = tiny("VGQ-AE").eval()
        data = inputs(model)
        for key in ("action", "noise_seed", "action_expert_noise_samples", "rtc_max_delay"):
            data.pop(key)

        def forbidden(*args, **kwargs):
            raise AssertionError("Reader must not execute in inference")

        model.rsci_reader.forward = forbidden
        first = model.generate_actions(**data, steps=2, generator=torch.Generator().manual_seed(4))
        data["proprio"] *= 1000
        second = model.generate_actions(**data, steps=2, generator=torch.Generator().manual_seed(4))
        self.assertEqual(first.shape, (2, 50, 36))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_official_simple_http_protocol_and_denormalization(self):
        from fastapi.testclient import TestClient
        from lambda0.data.normalization import DatasetNormalizer
        from lambda0.inference.server import build_app, decode_numpy_envelope, encode_numpy_envelope

        location = os.environ.get("RSCI_TEST_LAMBDA0_STATS")
        if not location:
            self.skipTest("Set RSCI_TEST_LAMBDA0_STATS to a released normalization JSON")
        stats = Path(location)
        if not stats.is_file():
            self.skipTest("Local normalization assets are not installed")
        model = tiny("VGQ-AE").eval()

        class Processor:
            def __call__(self, images, language_instruction, **kwargs):
                return dict(
                    input_ids=torch.tensor([[1, 2, 3]]),
                    attention_mask=torch.ones(1, 3, dtype=torch.long),
                    pixel_values=torch.zeros(1, 3),
                    image_grid_thw=torch.ones(1, 3, dtype=torch.long),
                    mm_token_type_ids=torch.zeros(1, 3, dtype=torch.int),
                )

        normalizer = DatasetNormalizer(stats)
        app = build_app(
            model,
            Processor(),
            torch.device("cpu"),
            default_embodiment_id=7,
            default_action_space_id=3,
            normalizer=normalizer,
            inference_seed=97,
            inference_steps=2,
        )
        client = TestClient(app)
        payload = dict(
            image=dict(
                rgb_head_stereo_left=encode_numpy_envelope(np.zeros((32, 32, 3), dtype=np.uint8))
            ),
            state=dict(states=encode_numpy_envelope(np.zeros((1, 32), dtype=np.float32))),
            instruction="pick up the box",
            history=dict(session_id="test-session", step_index=0),
        )
        response = client.post("/act", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        first = np.asarray(decode_numpy_envelope(response.json()["action"]))
        self.assertEqual(first.shape, (50, 36))
        self.assertTrue(np.isfinite(first).all())
        payload["state"]["states"] = encode_numpy_envelope(np.full((1, 32), 7.0, dtype=np.float32))
        response = client.post("/act", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        np.testing.assert_array_equal(first, decode_numpy_envelope(response.json()["action"]))
        for bad in (dict(instruction=""), dict(action_space_id=2), dict(image={})):
            malformed = {**payload, **bad}
            self.assertEqual(client.post("/act", json=malformed).status_code, 400)
        payload["state"]["states"] = encode_numpy_envelope(
            np.full((1, 32), np.inf, dtype=np.float32)
        )
        self.assertEqual(client.post("/act", json=payload).status_code, 400)
        payload["state"]["states"] = encode_numpy_envelope(np.zeros((1, 28), dtype=np.float32))
        self.assertEqual(client.post("/act", json=payload).status_code, 400)

    def test_action_window_and_paired_augmentation_at_episode_edges(self):
        from lambda0.data.dataset import InfiniteDataReader

        from rsci.backbones.lambda0.contracts import RUNTIME
        from rsci.backbones.lambda0.data import RSCIPsiSimpleHandler, SafePsiSimpleHandler

        location = os.environ.get("RSCI_TEST_LAMBDA0_BUNDLE")
        if not location:
            self.skipTest("Set RSCI_TEST_LAMBDA0_BUNDLE to a prepared train bundle")
        bundle = Path(location)
        if not bundle.is_dir():
            self.skipTest("Local benchmark assets are not installed")
        reader = InfiniteDataReader(
            str(bundle), data_root=str(RUNTIME / "data"), training=True, image_max_pixels=4096
        )
        meta = next(iter(reader.metas.values()))
        native, adapted = SafePsiSimpleHandler(meta, 1), RSCIPsiSimpleHandler(meta, 1)
        item = meta["datalist"][0]
        _, raw_actions, _, _ = native._load_episode(item)
        for index in (0, 6, item["length"] - 1):
            kwargs = dict(
                num_actions=50, image_aug=reader.image_aug, lang_aug_map=None, sample_index=index
            )
            with local_rng(437):
                reference = native.sample_episode(0, **kwargs)
            with local_rng(437):
                actual = adapted.sample_episode(0, **kwargs)
            for key in ("action", "action_valid_mask", "image_input", "image_mask"):
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
            expected_indices = np.minimum(np.arange(index + 1, index + 51), item["length"] - 1)
            np.testing.assert_allclose(
                native.normalizer.denormalize_action(actual["action"].numpy()),
                raw_actions[expected_indices],
                atol=1e-6,
                rtol=1e-6,
            )
            self.assertTrue(actual["action_valid_mask"].all())
            torch.testing.assert_close(
                actual["history_valid"], torch.tensor(np.array([-6, -4, -2, 0]) + index >= 0)
            )
            if index == 0:
                for historical in actual["history"]:
                    torch.testing.assert_close(historical, actual["image_input"][0], rtol=0, atol=0)
            self.assertFalse(actual["proprio_valid_mask"].any())

    def test_overflowing_finite_gradient_norm_is_rejected(self):
        from accelerate import Accelerator

        parameter = nn.Parameter(torch.zeros(8))
        parameter.grad = torch.full_like(parameter, 1e30)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        # 有限梯度的总范数仍可溢出；调用实际训练器保护，在优化器更新之前拒绝。
        from lambda0.training.engine import clip_gradients

        model = nn.ParameterList([parameter])
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite gradient norm"):
            clip_gradients(Accelerator(cpu=True, mixed_precision="no"), model, 1.0)
        self.assertFalse(parameter.any())

    def test_guardian_does_not_restart_from_zero_or_numeric_failure(self):
        from rsci.backbones.lambda0.launch import guardian

        for message, expected in (
            (b"worker crashed", "No verified complete checkpoint"),
            (b"non-finite total gradient norm", "Non-retryable failure"),
        ):
            with tempfile.TemporaryDirectory() as root:
                process = SimpleNamespace(pid=123, poll=lambda: 1, wait=lambda: 1)

                def spawn(command, **kwargs):
                    kwargs["stdout"].write(message)
                    return process

                with patch(
                    "rsci.backbones.lambda0.launch.subprocess.Popen", side_effect=spawn
                ) as mock:
                    with self.assertRaisesRegex(RuntimeError, expected):
                        guardian(SimpleNamespace(gpus="0"), [], Path(root))
                    self.assertEqual(mock.call_count, 1)

    def test_serving_rejects_foreign_processor_statistics_and_source(self):
        from rsci.backbones.lambda0 import serve
        from rsci.backbones.lambda0.contracts import sha256

        source = Path(serve.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("config.json", "state.json", "training_config.json"):
                (root / name).write_text("{}")
            (root / "training_state").mkdir()
            stats = root / "data_bundle/stats/test.json"
            stats.parent.mkdir(parents=True)
            stats.write_text("{}")
            provenance = dict(
                source={p.name: sha256(p) for p in sorted(source.glob("*.py"))},
                upstream_patch=sha256(source.parents[3] / "baselines/lambda0/upstream.patch"),
            )
            (root / "rsci_run.json").write_text(json.dumps(provenance))
            checkpoint_receipt(root, {})
            external = root.parent / (root.name + "-wrong.json")
            try:
                external.write_text('{"wrong": true}')
                argv = ["serve", "--model", str(root), "--normalization", str(external)]
                with patch("sys.argv", argv + ["--processor", "/foreign"]):
                    with self.assertRaisesRegex(ValueError, "processor"):
                        serve.main()
                with patch("sys.argv", argv):
                    with self.assertRaisesRegex(ValueError, "statistics"):
                        serve.main()
                provenance["source"] = {}
                (root / "rsci_run.json").write_text(json.dumps(provenance))
                checkpoint_receipt(root, {})
                with patch("sys.argv", argv):
                    with self.assertRaisesRegex(ValueError, "source"):
                        serve.main()
            finally:
                external.unlink(missing_ok=True)

    def test_masked_nan_and_empty_ardy(self):
        prediction = torch.randn(2, 128, requires_grad=True)
        target = torch.randn_like(prediction)
        target[0] = float("nan")
        valid = torch.tensor([False, True])
        loss = masked_ardy_loss(prediction, target, valid)
        torch.testing.assert_close(loss, (prediction[1] - target[1]).square().mean())
        loss.backward()
        self.assertFalse(prediction.grad[0].any())
        self.assertEqual(
            masked_ardy_loss(prediction, target, torch.zeros(2, dtype=torch.bool)).item(), 0.0
        )

    def test_launcher_coordination_is_not_a_checkpoint_or_training_run(self):
        from rsci.backbones.lambda0.train import guard_output

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "guardian.lock").touch()
            (root / "attempt-1.log").touch()
            guard_output(SimpleNamespace(resume="none"), root)
            (root / "rsci_run.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                guard_output(SimpleNamespace(resume="none"), root)
            with self.assertRaisesRegex(ValueError, "no verified complete checkpoint"):
                guard_output(SimpleNamespace(resume="auto"), root)

    def test_auxiliary_rejects_broadcast_targets_and_missing_current_frame(self):
        prediction = torch.randn(2, 128)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            masked_ardy_loss(prediction, torch.randn(1, 128), torch.ones(2, dtype=torch.bool))
        model = tiny("VGQ-AE")
        with self.assertRaisesRegex(ValueError, "current frame"):
            model.auxiliary({}, [], torch.zeros(2, 4, dtype=torch.bool), torch.tensor([7, 7]))

    def test_receipt_detects_corruption_and_config_changes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in ("config.json", "state.json", "training_config.json"):
                (root / name).write_text("{}")
            (root / "training_state").mkdir()
            (root / "training_state/optimizer.bin").write_bytes(b"finite state fixture")
            checkpoint_receipt(root, {"variant": "VGQ-AE"})
            verify_checkpoint(root, {"variant": "VGQ-AE"})
            with self.assertRaises(ValueError):
                verify_checkpoint(root, {"variant": "STATE"})
            (root / "config.json").write_text("bad")
            with self.assertRaises(ValueError):
                verify_checkpoint(root, {"variant": "VGQ-AE"})

    def test_real_data_prefetch_resume_and_rng_independence(self):
        from lambda0.data.dataset import InfiniteDataReader

        from rsci.backbones.lambda0.contracts import RUNTIME

        location = os.environ.get("RSCI_TEST_LAMBDA0_BUNDLE")
        if not location:
            self.skipTest("Set RSCI_TEST_LAMBDA0_BUNDLE to a prepared train bundle")
        bundle = Path(location)
        if not bundle.is_dir():
            self.skipTest("Local benchmark assets are not installed")

        def loader(workers):
            reader = InfiniteDataReader(
                str(bundle),
                data_root=str(RUNTIME / "data"),
                training=True,
                sampling_seed=87,
                image_max_pixels=4096,
            )
            for meta in reader.metas.values():
                meta["handler"] = "rsci.backbones.lambda0.data:RSCIPsiSimpleHandler"
            return ResumableLoader(reader, 2, workers)

        running = loader(2)
        iterator = iter(running)
        for _ in range(7):
            next(iterator)
        cursor = running.state_dict()
        expected = [next(iterator) for _ in range(3)]
        resumed = loader(0)
        resumed.load_state_dict(cursor)
        restored = iter(resumed)
        for reference in expected:
            actual = next(restored)
            for key in (
                "action",
                "proprio",
                "history",
                "history_valid",
                "ardy_target",
                "ardy_valid",
            ):
                torch.testing.assert_close(
                    actual[key], reference[key], rtol=0, atol=0, equal_nan=True
                )
            for actual_image, expected_image in zip(
                actual["image_input"], reference["image_input"]
            ):
                torch.testing.assert_close(actual_image, expected_image, rtol=0, atol=0)
        state = torch.get_rng_state().clone()
        with local_rng(987):
            torch.rand(20)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        del iterator

    def test_native_accelerate_checkpoint_restores_optimizer_rng_and_cursor(self):
        from accelerate import Accelerator
        from lambda0.data.dataset import InfiniteDataReader
        from lambda0.training import engine

        from rsci.backbones.lambda0.contracts import RUNTIME

        location = os.environ.get("RSCI_TEST_LAMBDA0_BUNDLE")
        if not location:
            self.skipTest("Set RSCI_TEST_LAMBDA0_BUNDLE to a prepared train bundle")
        bundle = Path(location)
        if not bundle.is_dir():
            self.skipTest("Local benchmark assets are not installed")
        reader = InfiniteDataReader(
            str(bundle),
            data_root=str(RUNTIME / "data"),
            training=True,
            sampling_seed=82,
            image_max_pixels=4096,
        )
        for meta in reader.metas.values():
            meta["handler"] = "rsci.backbones.lambda0.data:SafePsiSimpleHandler"
        loader = ResumableLoader(reader, 1, 0)
        for _ in range(3):
            next(iter(loader))
        model = tiny()

        # 此测试使用微型 VLM；只替换其 HF 配置写出，Accelerate/原生保存与恢复完整执行。
        def save_model(self, destination, **kwargs):
            (Path(destination) / "config.json").write_text("{}")
            torch.save(self.state_dict(), Path(destination) / "tiny-model.bin")

        model.save_pretrained = types.MethodType(save_model, model)

        class Processor:
            def save_pretrained(self, destination):
                (Path(destination) / "processor_config.json").write_text("{}")

        args = engine.get_args_parser().parse_args(
            ["--models", "tiny", "--train_metas_path", str(bundle)]
        )
        args.gradient_accumulation_steps = 1
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        accelerator.register_for_checkpointing(loader)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model, optimizer = accelerator.prepare(model, optimizer)
        model(**inputs(model))["loss"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        with tempfile.TemporaryDirectory() as root:
            log = SimpleNamespace(info=lambda *args: None)
            checkpoint = engine.save_training_checkpoint(
                accelerator, model, None, Processor(), Path(root), 3, args, log
            )
            expected_rng = torch.rand(5)
            expected_batch = next(iter(loader))["action"]
            with torch.no_grad():
                next(model.parameters()).add_(7)
            restored = engine.load_training_checkpoint(accelerator, checkpoint, args)
            from rsci.backbones.lambda0.runtime import verify_restored_state

            verify_restored_state(accelerator)
            self.assertEqual(restored, 3)
            self.assertEqual(loader.consumed, 3)
            torch.testing.assert_close(torch.rand(5), expected_rng, rtol=0, atol=0)
            torch.testing.assert_close(next(iter(loader))["action"], expected_batch, rtol=0, atol=0)
            for key, value in expected_model.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            for key, state in expected_optimizer["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        optimizer.state_dict()["state"][key][name], value, rtol=0, atol=0
                    )

    def test_official_optimizer_and_schedule_preserved(self):
        from lambda0.training import engine

        reference_model = tiny("VGQ-AE")
        args = SimpleNamespace(
            weight_decay=1e-10,
            betas=[0.9, 0.95],
            fused_optimizer=False,
            learning_rate=5e-5,
            vlm_lr_coef=1.0,
            conditioning_lr_coef=1.0,
            freeze_steps=0,
            warmup_steps=0,
            new_module_warmup_steps=0,
            iters=40000,
            min_lr_ratio=0.01,
            use_cosine_decay=True,
        )
        original = engine.build_optimizer
        reference = original(reference_model, args)
        backend = RSCITrainingBackend(engine, "VGQ-AE", {})
        actual = backend.build_optimizer(reference_model, args)
        self.assertEqual(reference.defaults, actual.defaults)
        for step in (0, 1000, 20000, 39999):
            engine.update_group_lrs(reference, step, args)
            engine.update_group_lrs(actual, step, args)
            self.assertEqual(
                [group["lr"] for group in reference.param_groups],
                [group["lr"] for group in actual.param_groups],
            )
        self.assertAlmostEqual(actual.param_groups[0]["lr"], 5e-7, places=15)


if __name__ == "__main__":
    unittest.main()
