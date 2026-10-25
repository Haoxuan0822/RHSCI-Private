"""挑战真实验证循环、RTC边界及双rank不均等验证，不替代CUDA训练。"""

import copy
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_integration import inputs, tiny, vlm_inputs
from torch import nn

from rsci.backbones.lambda0.model import masked_ardy_loss
from rsci.backbones.lambda0.runtime import check_finite, verify_restored_state


def validation_fixture(model, folder):
    data = inputs(model)
    for key in ("noise_seed", "action_expert_noise_samples", "rtc_max_delay"):
        data.pop(key)
    data["action_valid_mask"] = model.action_space.action_mask(
        data["action_space_id"], 50, torch.device("cpu")
    )
    if model.config.rsci_variant == "VGQ-AE":
        data.update(
            history=[vlm_inputs() for _ in range(3)],
            history_valid=torch.ones(2, 4, dtype=torch.bool),
            ardy_target=torch.randn(2, 128),
            ardy_valid=torch.tensor([True, False]),
        )
    stats = Path(folder) / "statistics.json"
    stats.write_text(json.dumps({"action": {"scale": [2.0] * 36}}))
    meta = {"fixture": {"normalization_path": str(stats)}}
    batch = dict(prepared=data, dataset_name=["fixture", "fixture"])
    return batch, meta


class Loader:
    def __init__(self, batches, metas):
        self.batches = batches
        self.dataset = SimpleNamespace(metas=metas)

    def __iter__(self):
        return iter(self.batches)


def distributed_worker(rank, rendezvous, output, fixture_folder):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        # rank0没有有效ARDY；全局梯度仍须等于拼接batch的有效样本均值。
        query = nn.Parameter(torch.tensor(0.25))
        target = torch.ones(2, 128) * (rank + 1)
        valid = torch.tensor([False, False] if rank == 0 else [True, False])
        loss = masked_ardy_loss(query.expand(2, 128), target, valid)
        loss.backward()
        gradient = query.grad.clone()
        dist.all_reduce(gradient)
        gradient /= 2
        torch.testing.assert_close(gradient, torch.tensor(2 * (0.25 - 2)), atol=1e-6, rtol=1e-6)
        rejected = False
        try:
            check_finite([torch.tensor(float("inf") if rank == 1 else 1.0)], "distributed fixture")
        except FloatingPointError:
            rejected = True
        assert rejected
        # 单episode holdout下某些rank根本无数据，不能在VGQ前向内部做collective。
        from accelerate import Accelerator
        from lambda0.training import engine

        accelerator = Accelerator(cpu=True, mixed_precision="no")

        # 直接使用gloo归约，避免测试进程由环境隐式选择NCCL。
        def reduce(tensor, reduction="sum"):
            value = tensor.clone()
            dist.all_reduce(value)
            return value

        accelerator.reduce = reduce
        accelerator.log = lambda *args, **kwargs: None
        torch.manual_seed(97)
        model = tiny("VGQ-AE")
        rank_folder = Path(fixture_folder) / f"fixture-{rank}"
        rank_folder.mkdir()
        batch, metas = validation_fixture(model, rank_folder)
        loader = Loader([batch] if rank == 1 else [], metas)
        with patch.object(
            engine, "prepare_inputs", side_effect=lambda batch, *args: batch["prepared"]
        ):
            logs = engine.run_validation(
                accelerator,
                model,
                None,
                {"holdout": loader},
                0,
                4,
                91,
                0,
                action_chunk_mse=True,
                action_chunk_samples=1,
                action_chunk_steps=2,
            )
        assert logs["validation/holdout/samples"] == 2
        assert model.training
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps(dict(gradient=gradient.item(), logs=logs))
        )
    finally:
        dist.destroy_process_group()


class ValidationFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_native_validation_determinism_physical_scale_and_training_mode(self):
        from accelerate import Accelerator
        from lambda0.training import engine

        accelerator = Accelerator(cpu=True, mixed_precision="no")
        for variant in ("STATE", "VGQ-AE"):
            with tempfile.TemporaryDirectory() as folder:
                model = tiny(variant)
                batch, metas = validation_fixture(model, folder)
                loader = Loader([batch], metas)
                with patch.object(
                    engine, "prepare_inputs", side_effect=lambda batch, *args: batch["prepared"]
                ):
                    first = engine.run_validation(
                        accelerator,
                        model,
                        None,
                        {"holdout": loader},
                        0,
                        4,
                        91,
                        0,
                        action_chunk_mse=True,
                        action_chunk_samples=1,
                        action_chunk_steps=2,
                    )
                    second = engine.run_validation(
                        accelerator,
                        model,
                        None,
                        {"holdout": loader},
                        0,
                        4,
                        91,
                        0,
                        action_chunk_mse=True,
                        action_chunk_samples=1,
                        action_chunk_steps=2,
                    )
                self.assertEqual(first, second)
                self.assertTrue(model.training)
                self.assertEqual(first["validation/holdout/samples"], 2)
                self.assertAlmostEqual(
                    first["validation/holdout/original_scale_mse"],
                    4 * first["validation/holdout/action_chunk_mse"],
                    places=5,
                )
                self.assertTrue(
                    all(torch.isfinite(torch.tensor(value)) for value in first.values())
                )

    def test_explicit_preparation_callback_is_used_for_model_and_ema(self):
        from accelerate import Accelerator
        from lambda0.training import engine

        model = tiny("VGQ-AE")
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        with tempfile.TemporaryDirectory() as folder:
            batch, metas = validation_fixture(model, folder)
            loader = Loader([batch], metas)
            with (
                patch.object(accelerator, "log"),
                patch.object(
                    engine, "prepare_inputs", side_effect=AssertionError("native callback")
                ),
                patch("builtins.print"),
            ):
                from unittest.mock import Mock

                prepare = Mock(side_effect=lambda batch, *_: batch["prepared"])
                result = engine.run_validation(
                    accelerator,
                    model,
                    None,
                    {"holdout": loader},
                    0,
                    1,
                    91,
                    1,
                    ema_policy=copy.deepcopy(model),
                    prepare_batch=prepare,
                )
            self.assertEqual(prepare.call_count, 2)
            self.assertIn("validation/holdout/loss_total", result)
            self.assertIn("validation_ema/holdout/loss_total", result)

    def test_validation_loss_includes_flow_and_weighted_ardy(self):
        from lambda0.models.modeling_lambda0 import Lambda0

        model = tiny("VGQ-AE").eval()
        data = inputs(model)
        extra = dict(
            history=[vlm_inputs() for _ in range(3)],
            history_valid=torch.ones(2, 4, dtype=torch.bool),
            ardy_target=torch.randn(2, 128) * 100,
            ardy_valid=torch.tensor([True, False]),
        )
        native = Lambda0.forward(model, **data)
        actual = model(**data, **extra)
        torch.testing.assert_close(actual["loss"], native["loss"] + 0.1 * model.last_ardy_loss)
        for key in native.keys() - {"loss"}:
            torch.testing.assert_close(actual[key], native[key], rtol=0, atol=0)
        self.assertGreater(model.last_ardy_loss.item(), 0)

    def test_training_and_validation_share_the_total_loss_contract(self):
        from lambda0.models.modeling_lambda0 import Lambda0

        model = tiny("VGQ-AE")
        model._rsci_current = None
        target = torch.full((2, 128), 100.0)
        target[1] = float("nan")
        for training in (True, False):
            model.train(training)
            with (
                patch.object(
                    Lambda0,
                    "forward",
                    side_effect=lambda *args, **kwargs: {"loss": torch.tensor(3.35)},
                ),
                patch.object(model, "auxiliary", return_value=torch.zeros(2, 128)),
            ):
                losses = model(
                    history=[{}] * 3,
                    history_valid=torch.ones(2, 4, dtype=torch.bool),
                    embodiment_id=torch.tensor([7, 7]),
                    ardy_target=target,
                    ardy_valid=torch.tensor([True, False]),
                )
            self.assertAlmostEqual(losses["loss"].item(), 1003.35, places=3)
            self.assertEqual(model.last_ardy_loss.item(), 10000.0)

    def test_native_rtc_prefix_is_preserved_with_replicas_and_no_auxiliary_inference(self):
        from lambda0.models.modeling_lambda0 import Lambda0

        model = tiny("VGQ-AE").eval()
        data = inputs(model)
        for key in ("action", "noise_seed", "action_expert_noise_samples", "rtc_max_delay"):
            data.pop(key)
        prefix = torch.randn(2, 8, 36)
        lengths = torch.tensor([0, 8])

        def forbidden(*args, **kwargs):
            raise AssertionError("Inference must not enter auxiliary branch")

        model.auxiliary = forbidden
        kwargs = dict(**data, action_prefix=prefix, prefix_lengths=lengths, steps=3, num_samples=2)
        first = model.generate_actions(**kwargs, generator=torch.Generator().manual_seed(97))
        reference = Lambda0.generate_actions(
            model, **kwargs, generator=torch.Generator().manual_seed(97)
        )
        torch.testing.assert_close(first, reference, rtol=0, atol=0)
        torch.testing.assert_close(first[1, :8], prefix[1], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(first).all())
        for malformed in (torch.tensor([-1, 8]), torch.tensor([0, 9])):
            with self.assertRaisesRegex(ValueError, "prefix_lengths"):
                model.generate_actions(**{**kwargs, "prefix_lengths": malformed})

    def test_flow_time_and_rtc_masks_match_native_contract_on_padded_rows(self):
        from lambda0.models.flow_matching import sample_flow_conditions

        valid = torch.zeros(64, 50, 36, dtype=torch.bool)
        valid[:32, :1] = True
        valid[32:, :7] = True
        rng = torch.get_rng_state().clone()
        for no_prefix in (0.0, 0.5, 1.0):
            first = sample_flow_conditions(valid, 8, no_prefix, torch.Generator().manual_seed(87))
            second = sample_flow_conditions(valid, 8, no_prefix, torch.Generator().manual_seed(87))
            for actual, reference in zip(first, second):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            time, prefix = first
            self.assertTrue(((time >= 0.001) & (time <= 1)).all())
            self.assertFalse(prefix[:32].any())
            self.assertTrue((prefix.sum(-1)[32:] <= 6).all())
            self.assertFalse((prefix & ~valid.any(-1)).any())
            self.assertTrue((valid.any(-1) & ~prefix).any(-1).all())
            if no_prefix == 1.0:
                self.assertFalse(prefix.any())
            if no_prefix == 0.0:
                self.assertTrue(prefix[32:].any(-1).all())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_restored_nonfinite_model_adam_and_ema_are_rejected(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        ema = copy.deepcopy(model)
        accelerator = SimpleNamespace(
            _models=[model], _optimizers=[optimizer], _custom_objects=[ema]
        )
        verify_restored_state(accelerator)
        for tensor, message in (
            (model.weight, "model"),
            (optimizer.state[model.weight]["exp_avg"], "Adam"),
            (ema.weight, "EMA"),
        ):
            with torch.no_grad():
                original = tensor.clone()
                tensor.fill_(float("nan"))
            with self.assertRaisesRegex(FloatingPointError, message):
                verify_restored_state(accelerator)
            with torch.no_grad():
                tensor.copy_(original)

    def test_two_gloo_ranks_with_empty_validation_rank_and_global_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(
                distributed_worker,
                args=(str(Path(folder) / "rendezvous"), folder, folder),
                nprocs=2,
                join=True,
            )
            reports = [
                json.loads((Path(folder) / f"rank-{rank}.json").read_text()) for rank in range(2)
            ]
            self.assertEqual(reports[0], reports[1])


if __name__ == "__main__":
    unittest.main()
