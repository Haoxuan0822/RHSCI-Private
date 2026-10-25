"""使用真实微型 Gemma 层验证掩码、梯度、KV只读及推理分支隔离。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaModel

from rsci.backbones.pi05.checkpoints import (
    atomic_json,
    checkpoint_ok,
    complete_checkpoint,
    prune_rolling,
)
from rsci.backbones.pi05.model import Pi05RSCI
from rsci.backbones.pi05.processing import Processor


def ddp_worker(rank, rendezvous):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        model, batch = fixture()
        # 一个rank全无效，另一个rank只有一条有效；覆盖全局mask归一化与unused风险。
        if rank == 0:
            batch["ardy_valid"].zero_()
        model = torch.nn.parallel.DistributedDataParallel(model, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(**batch)["loss"].backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.requires_grad
            )
            optimizer.step()
    finally:
        dist.destroy_process_group()


def trainer_ddp_worker(rank, rendezvous, output, port):
    import os

    import torch.distributed as dist
    from transformers import TrainingArguments

    from rsci.backbones.pi05.processing import collate
    from rsci.backbones.pi05.train import complete_callback, trainer_class

    # CPU 多 rank 默认设备 cpu:0 无法由 torch2.7 恢复；仅测试显式选择标准 cpu。
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        ACCELERATE_USE_CPU="true",
        ACCELERATE_TORCH_DEVICE="cpu",
    )
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        model, batch = fixture()
        samples = [{key: value[index] for key, value in batch.items()} for index in range(2)]
        arguments = TrainingArguments(
            output_dir=output,
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            use_cpu=True,
            ddp_backend="gloo",
            ddp_find_unused_parameters=False,
            optim="adamw_torch_fused",
            save_steps=2,
            report_to="none",
            disable_tqdm=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        trainer = trainer_class()(
            model=model,
            args=arguments,
            train_dataset=samples,
            data_collator=collate,
            callbacks=[complete_callback(Path(output), 2, "ddp-trainer")],
        )
        trainer.train()
        checkpoint = Path(output) / "checkpoint-2"
        assert checkpoint_ok(checkpoint, 2, "ddp-trainer", verify_hash=True)
        assert trainer.accelerator.ddp_handler.gradient_as_bucket_view
        restored, _ = fixture()
        arguments.max_steps = 3
        trainer = trainer_class()(
            model=restored,
            args=arguments,
            train_dataset=samples,
            data_collator=collate,
            callbacks=[complete_callback(Path(output), 2, "ddp-trainer")],
        )
        trainer.train(resume_from_checkpoint=str(checkpoint))
        assert trainer.state.global_step == 3
    finally:
        dist.destroy_process_group()


def nonfinite_ddp_worker(rank, rendezvous, output):
    import torch.distributed as dist

    from rsci.runtime.numerics import require_finite

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        try:
            require_finite([torch.tensor(float("nan") if rank else 1.0)], output, 3, "loss")
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Every rank must reject another rank's nonfinite loss")
    finally:
        dist.destroy_process_group()


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        def model(width, adaptive):
            return GemmaModel(
                GemmaConfig(
                    vocab_size=128,
                    hidden_size=width,
                    intermediate_size=width * 2,
                    num_hidden_layers=2,
                    num_attention_heads=8,
                    num_key_value_heads=1,
                    head_dim=8,
                    use_adarms=adaptive,
                    adarms_cond_dim=width if adaptive else None,
                )
            )

        self.paligemma = nn.Module()
        self.paligemma.language_model = model(64, False)
        self.paligemma.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=64, num_hidden_layers=2)
        )
        self.paligemma.model = SimpleNamespace(language_model=self.paligemma.language_model)
        self.paligemma.image = nn.Linear(3, 64)
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = model(32, True)
        self.gemma_expert.model.embed_tokens = None

    def embed_image(self, image):
        return self.paligemma.image(image.flatten(2).transpose(1, 2))

    def embed_language_tokens(self, tokens):
        return self.paligemma.language_model.embed_tokens(tokens)


def fixture(variant="VGQ-AE"):
    torch.manual_seed(3)
    model = Pi05RSCI(variant, TinyBackbone(), vlm_width=64, ae_width=32, compile_inference=False)
    batch = dict(
        images=torch.randn(2, 4, 3, 2, 2),
        tokens=torch.tensor([[1, 2, 0], [1, 2, 3]]),
        token_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        actions=torch.randn(2, 30, 36),
        history_valid=torch.tensor([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=torch.bool),
        ardy_target=torch.randn(2, 128),
        ardy_valid=torch.tensor([0, 1], dtype=torch.bool),
    )
    return model, batch


class ModelTests(unittest.TestCase):
    def test_state_and_no_state_baselines_train_without_auxiliary(self):
        for variant in ("STATE", "NO-STATE"):
            model, batch = fixture(variant)
            batch["images"] = batch["images"][:, -1:]
            for key in ("history_valid", "ardy_target", "ardy_valid"):
                del batch[key]
            output = model(**batch)
            self.assertNotIn("ardy_loss", output)
            output["loss"].backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))

    def test_official_tied_embedding_and_action_expansion(self):
        from safetensors.torch import save_file

        model, _ = fixture()
        state = {
            name: value.clone()
            for name, value in model.state_dict().items()
            if not name.startswith(("reader.", "query", "ardy_head."))
        }
        embedding = "paligemma_with_expert.paligemma.language_model.embed_tokens.weight"
        state["paligemma_with_expert.paligemma.lm_head.weight"] = state.pop(embedding)
        state["action_in_proj.weight"] = state["action_in_proj.weight"][:, :32].contiguous()
        state["action_out_proj.weight"] = state["action_out_proj.weight"][:32].contiguous()
        state["action_out_proj.bias"] = state["action_out_proj.bias"][:32].contiguous()
        with tempfile.TemporaryDirectory() as directory:
            save_file(state, str(Path(directory) / "model.safetensors"))
            restored, _ = fixture()
            restored.load_initial(directory)
            torch.testing.assert_close(
                restored.state_dict()[embedding],
                state["paligemma_with_expert.paligemma.lm_head.weight"],
            )
            torch.testing.assert_close(
                restored.action_in_proj.weight[:, 32:], state["action_in_proj.weight"][:, -4:]
            )
            torch.testing.assert_close(
                restored.action_out_proj.weight[32:], state["action_out_proj.weight"][-4:]
            )

    def test_cached_action_matches_original_joint_forward(self):
        import math

        from torch.nn import functional as F

        from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
        from rsci.backbones.pi05.model import time_embedding

        model, batch = fixture()
        model.eval()
        native = model.paligemma_with_expert
        _, mask, cache = model.encode(batch["images"][:, -1], batch["tokens"], batch["token_mask"])
        suffix = torch.randn(2, 16, 32)
        time = torch.tensor([0.2, 0.8])
        actual = model.expert(suffix, time, mask, cache)
        cond = F.silu(model.time_mlp_out(F.silu(model.time_mlp_in(time_embedding(time, 32)))))
        prefix = torch.cat(
            (
                native.embed_image(batch["images"][:, -1]),
                native.embed_language_tokens(batch["tokens"]) * math.sqrt(64),
            ),
            1,
        )
        full_mask = torch.cat((mask, torch.ones(2, 16, dtype=torch.bool)), 1)
        attention = torch.zeros(2, 1, full_mask.shape[1], full_mask.shape[1])
        attention.masked_fill_(~full_mask[:, None, None, :], torch.finfo(torch.float32).min)
        attention[:, :, : mask.shape[1], mask.shape[1] :] = torch.finfo(torch.float32).min
        positions = full_mask.long().cumsum(-1) - 1
        outputs, _ = PaliGemmaWithExpertModel.forward(
            native,
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            adarms_cond=[None, cond],
        )
        torch.testing.assert_close(actual, outputs[1], rtol=2e-5, atol=2e-5)

    def test_real_ae_backward_and_frozen_vlm(self):
        model, batch = fixture()
        output = model(**batch)
        output["loss"].backward()
        for parameter in (
            model.query,
            model.reader.queries,
            model.ardy_head[-1].weight,
            model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(
            all(p.grad is None for p in model.paligemma_with_expert.paligemma.parameters())
        )

    def test_history_never_conditions_actions(self):
        model, batch = fixture()
        torch.manual_seed(12)
        a = model(**batch)
        batch["images"][:, :3] += 10
        torch.manual_seed(12)
        b = model(**batch)
        torch.testing.assert_close(a["action_loss"], b["action_loss"], rtol=0, atol=0)
        self.assertNotEqual(a["ardy_loss"].item(), b["ardy_loss"].item())

    def test_invalid_history_and_target_are_masked(self):
        model, batch = fixture()
        torch.manual_seed(12)
        a = model(**batch)
        batch["images"][0, :3] += 100
        batch["ardy_target"][0] += 100
        torch.manual_seed(12)
        b = model(**batch)
        torch.testing.assert_close(a["loss"], b["loss"], rtol=0, atol=0)

    def test_inference_never_uses_auxiliary_and_cache_unchanged(self):
        model, batch = fixture()
        model.eval()

        def forbidden(*a, **kw):
            raise AssertionError("Auxiliary branch entered at inference")

        model.reader.forward = forbidden
        _, mask, cache = model.encode(batch["images"][:, -1], batch["tokens"], batch["token_mask"])
        lengths = [key.shape[2] for key, value in cache]
        model.expert(torch.randn(2, 16, 32), torch.ones(2), mask, cache)
        model.expert(torch.randn(2, 16, 32), torch.ones(2), mask, cache)
        self.assertEqual(lengths, [key.shape[2] for key, value in cache])
        result = model.sample_actions(
            batch["images"][:, -1:], batch["tokens"], batch["token_mask"], num_steps=2
        )
        self.assertEqual(result.shape, (2, 30, 36))
        self.assertTrue(torch.isfinite(result).all())

    def test_all_invalid_ardy_still_connects_ddp_graph(self):
        model, batch = fixture()
        batch["ardy_valid"].zero_()
        output = model(**batch)
        output["loss"].backward()
        self.assertEqual(output["ardy_loss"].item(), 0)
        self.assertIsNotNone(model.query.grad)
        self.assertEqual(model.query.grad.abs().sum().item(), 0)


class ProcessingTests(unittest.TestCase):
    def test_quantile_zero_span_outliers_and_repeated_calls(self):
        import numpy as np

        processor = object.__new__(Processor)
        processor.stats = {"action": dict(q01=[0.0, -2.0, 1.0], q99=[0.0, 2.0, 1.00001])}
        original = json.dumps(processor.stats)
        values = np.array([[1.5, -20.0, 1.0], [-0.5, 20.0, 1.00002]], dtype=np.float32)
        expected = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        for _ in range(3):
            np.testing.assert_array_equal(processor.normalize(values, "action"), expected)
        self.assertEqual(json.dumps(processor.stats), original)
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            processor.normalize(values, "action")

    def test_no_state_text_leak(self):
        import numpy as np

        for variant in ("STATE", "NO-STATE", "VGQ-AE"):
            processor = object.__new__(Processor)
            processor.variant = variant
            processor.state_measurement_noise_std_rad = 0.0
            seen = []
            processor.tokenizer = SimpleNamespace(
                encode=lambda prompt, **kw: seen.append(prompt) or [1, 2]
            )
            processor.stats = {
                key: dict(q01=[-1] * width, q99=[1] * width)
                for key, width in (("states", 32), ("action", 36))
            }
            sample = dict(
                text="do task",
                state=np.ones(32),
                images=np.zeros((4, 2, 2, 3), np.uint8),
                actions=np.zeros((30, 36), np.float32),
                rsci_history_valid=[0, 0, 0, 1],
                rsci_ardy_target=np.zeros(128, np.float32),
                rsci_ardy_valid=False,
            )
            processor(sample)
            self.assertEqual("State:" in seen[0], variant == "STATE")


class CheckpointTests(unittest.TestCase):
    def test_nonfinite_model_and_optimizer_never_publish_complete(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint-1"
            path.mkdir()
            for name in ("rsci_config.json", "scheduler.pt", "training_args.bin", "rng_state.pth"):
                (path / name).write_bytes(b"test")
            atomic_json(path / "trainer_state.json", dict(global_step=1))
            torch.save({"state": {0: {"exp_avg": torch.zeros(2)}}}, path / "optimizer.pt")
            save_file({"weight": torch.tensor([float("nan")])}, str(path / "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "Nonfinite checkpoint tensor"):
                complete_checkpoint(path, 1, "recipe")
            self.assertFalse((path / "RSCI_COMPLETE.json").exists())
            save_file({"weight": torch.zeros(1)}, str(path / "model.safetensors"))
            torch.save(
                {"state": {0: {"exp_avg": torch.tensor([float("inf")])}}}, path / "optimizer.pt"
            )
            with self.assertRaisesRegex(ValueError, "optimizer"):
                complete_checkpoint(path, 1, "recipe")
            self.assertFalse((path / "RSCI_COMPLETE.json").exists())

    def test_production_trainer_two_rank_accumulation_save_and_resume(self):
        import socket

        import torch.multiprocessing as mp

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                trainer_ddp_worker,
                args=(str(Path(directory) / "rendezvous"), directory, port),
                nprocs=2,
                join=True,
            )

    def test_real_two_rank_ddp(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)

    def test_real_trainer_save_and_resume(self):
        from transformers import TrainingArguments

        from rsci.backbones.pi05.processing import collate
        from rsci.backbones.pi05.train import complete_callback, trainer_class

        Trainer = trainer_class()
        model, batch = fixture()
        samples = [{key: value[index] for key, value in batch.items()} for index in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(
                output_dir=directory,
                max_steps=2,
                per_device_train_batch_size=2,
                use_cpu=True,
                save_steps=2,
                report_to="none",
                disable_tqdm=True,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                dataloader_num_workers=2,
                dataloader_persistent_workers=True,
                optim="adamw_torch_fused",
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=samples,
                data_collator=collate,
                callbacks=[complete_callback(Path(directory), 1, "real-trainer")],
            )
            trainer.model_accepts_loss_kwargs = False
            trainer.train()
            self.assertEqual(trainer._rsci_loaders, [])
            path = Path(directory) / "checkpoint-2"
            self.assertTrue(checkpoint_ok(path, 1, "real-trainer", verify_hash=True))
            resumed, _ = fixture()
            args.max_steps = 3
            trainer = Trainer(
                model=resumed, args=args, train_dataset=samples, data_collator=collate
            )
            trainer.model_accepts_loss_kwargs = False
            trainer.train(resume_from_checkpoint=str(path))
            self.assertEqual(trainer.state.global_step, 3)
            self.assertEqual(trainer._rsci_loaders, [])

    def test_missing_rng_corruption_and_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (1000, 2000, 10000):
                path = root / f"checkpoint-{step}"
                path.mkdir()
                for name in (
                    "rsci_config.json",
                    "optimizer.pt",
                    "scheduler.pt",
                    "training_args.bin",
                    "rng_state.pth",
                    "model.safetensors",
                ):
                    (path / name).write_bytes(b"test")
                from safetensors.torch import save_file

                save_file({"weight": torch.zeros(1)}, str(path / "model.safetensors"))
                torch.save({"state": {0: {"exp_avg": torch.zeros(1)}}}, path / "optimizer.pt")
                atomic_json(path / "trainer_state.json", dict(global_step=step))
                complete_checkpoint(path, 1, "recipe")
            current = root / "checkpoint-10000"
            self.assertTrue(checkpoint_ok(current, 1, "recipe", verify_hash=True))
            (current / "rng_state.pth").write_bytes(b"oops")
            self.assertFalse(checkpoint_ok(current, 1, "recipe", verify_hash=True))
            (current / "rng_state.pth").write_bytes(b"test")
            prune_rolling(root, current, 1, "recipe")
            self.assertTrue(current.exists())
            self.assertFalse((root / "checkpoint-1000").exists())
            self.assertFalse((root / "checkpoint-2000").exists())


class NumericalTests(unittest.TestCase):
    def test_one_bad_rank_stops_all_ranks(self):
        import torch.multiprocessing as mp

        from rsci.runtime.numerics import FAILURE, reject_numerical_failure

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                nonfinite_ddp_worker,
                args=(str(Path(directory) / "rendezvous"), directory),
                nprocs=2,
                join=True,
            )
            record = json.loads((Path(directory) / FAILURE).read_text())
            self.assertEqual(record["stage"], "loss")
            with self.assertRaisesRegex(RuntimeError, "refusing automatic resume"):
                reject_numerical_failure(directory)

    def test_production_trainer_rejects_loss_and_backward_overflow_before_update(self):
        from transformers import TrainingArguments

        from rsci.backbones.pi05.train import trainer_class
        from rsci.runtime.numerics import FAILURE

        class InfiniteBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, grad):
                return torch.full_like(grad, float("inf"))

        class BadModel(nn.Module):
            def __init__(self, bad_loss):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.bad_loss = bad_loss

            def forward(self, value):
                loss = (
                    self.weight * float("nan")
                    if self.bad_loss
                    else InfiniteBackward.apply(self.weight)
                )
                return dict(loss=loss, action_loss=loss)

        for bad_loss, stage in ((True, "loss"), (False, "gradient norm")):
            with tempfile.TemporaryDirectory() as directory:
                model = BadModel(bad_loss)
                before = model.weight.detach().clone()
                args = TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    use_cpu=True,
                    report_to="none",
                    disable_tqdm=True,
                    remove_unused_columns=False,
                    dataloader_pin_memory=False,
                )
                trainer = trainer_class()(model=model, args=args, train_dataset=[{"value": 1.0}])
                with self.assertRaisesRegex(FloatingPointError, stage):
                    trainer.train()
                torch.testing.assert_close(model.weight, before)
                self.assertEqual(trainer.state.global_step, 0)
                self.assertEqual(
                    json.loads((Path(directory) / FAILURE).read_text())["stage"], stage
                )


class QueueTests(unittest.TestCase):
    def test_preflight_rejects_other_tasks_stale_run_and_zero_restart(self):
        from rsci.backbones.pi05.train import parser, validate_run

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = parser().parse_args(
                ["--dataset", "/datasets/test", "--sidecar", "/labels/test", "--task", "between"]
            )
            paths = {"output": output}
            atomic_json(output / "launch_contract.json", {"batch": 8})
            with self.assertRaisesRegex(ValueError, "different configuration"):
                validate_run(args, paths, {"batch": 16}, "digest")
            atomic_json(output / "launch_contract.json", {"batch": 16})
            atomic_json(output / "training_started.json", {"attempt": 1})
            with self.assertRaisesRegex(RuntimeError, "refusing restart from zero"):
                validate_run(args, paths, {"batch": 16}, "digest")
            self.assertIsNone(
                validate_run(args, paths, {"batch": 16}, "digest", allow_started=True)
            )


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
