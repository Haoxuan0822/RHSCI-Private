"""真实微型Gemma：执行方式的输出、全梯度、更新、恢复及推理对照。"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_pi05_rsci import fixture


def speed_ddp_worker(rank, rendezvous):
    import test_pi05_rsci

    original = test_pi05_rsci.fixture

    def optimized_fixture():
        model, batch = original()
        model.execution = "both"
        return model, batch

    test_pi05_rsci.fixture = optimized_fixture
    test_pi05_rsci.ddp_worker(rank, rendezvous)


def trajectory(model, batch, bf16=False, resume=False):
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=1e-10,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / 1001)
    torch.manual_seed(193)
    records = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            result = model(**batch)
        result["loss"].backward()
        gradients = {
            n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }
        assert all(torch.isfinite(g).all() for g in gradients.values())
        assert all(p.grad is None for p in model.paligemma_with_expert.paligemma.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        records.append(
            dict(
                output={k: v.detach().clone() for k, v in result.items()},
                gradients=gradients,
                weights={
                    n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
                },
                optimizer=copy.deepcopy(optimizer.state_dict()),
                scheduler=scheduler.state_dict(),
                rng=torch.get_rng_state().clone(),
            )
        )
        if resume and step == 0:
            # 实际序列化优化器/调度器/RNG，不用重新从零开始冒充恢复。
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "resume.pt"
                torch.save(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        rng=torch.get_rng_state(),
                    ),
                    path,
                )
                saved = torch.load(path, weights_only=True)
            model = copy.deepcopy(model)
            model.load_state_dict(saved["model"])
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad), fused=True
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / 1001)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            torch.set_rng_state(saved["rng"])
    return records


class SpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_nested_close(self, a, b, rtol=0, atol=0):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_close(a[key], b[key], rtol, atol)
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_close(x, y, rtol, atol)
        else:
            self.assertEqual(a, b)

    def test_query_recompute_removal_is_exact_over_three_updates(self):
        for bf16 in (False, True):
            base, batch = fixture()
            changed = copy.deepcopy(base)
            changed.execution = "query"
            a = trajectory(base, batch, bf16)
            b = trajectory(changed, batch, bf16)
            self.assert_nested_close(a, b)

    def test_history_order_masks_and_frozen_eval(self):
        base, batch = fixture()
        base.train()
        self.assertFalse(base.paligemma_with_expert.paligemma.training)
        old = base.encode_history(batch["images"][:, :3], batch["tokens"], batch["token_mask"])
        base.execution = "history"
        new = base.encode_history(batch["images"][:, :3], batch["tokens"], batch["token_mask"])
        self.assert_nested_close(old, new, rtol=2e-5, atol=2e-6)
        # 同一batch内另一条轨迹的历史变化不能串到第一条轨迹。
        altered = batch["images"][:, :3].clone()
        altered[1] += 20
        changed = base.encode_history(altered, batch["tokens"], batch["token_mask"])
        for a, b in zip(new, changed):
            torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)

    def test_history_and_combination_gradients_and_updates(self):
        for bf16 in (False, True):
            base, batch = fixture()
            expected = trajectory(copy.deepcopy(base), batch, bf16)
            for profile in ("history", "both"):
                other = copy.deepcopy(base)
                other.execution = profile
                actual = trajectory(other, batch, bf16)
                # BF16批量形状可能引起舍入差异；明确容差而不声称逐位相同。
                tolerance = dict(rtol=0.03, atol=5e-4) if bf16 else dict(rtol=3e-4, atol=2e-6)
                for a, b in zip(expected, actual):
                    self.assert_nested_close(a["output"], b["output"], **tolerance)
                    self.assert_nested_close(a["gradients"], b["gradients"], **tolerance)
                    self.assert_nested_close(a["weights"], b["weights"], rtol=1e-5, atol=2e-6)
                    self.assert_nested_close(a["rng"], b["rng"])

    def test_all_profiles_resume_exactly(self):
        base, batch = fixture()
        for profile in ("reference", "history", "query", "both"):
            model = copy.deepcopy(base)
            model.execution = profile
            self.assert_nested_close(
                trajectory(copy.deepcopy(model), batch), trajectory(model, batch, resume=True)
            )

    def test_inference_does_not_use_training_optimizations(self):
        base, batch = fixture()
        base.eval()
        noise = torch.randn(2, 30, 36)
        images = batch["images"][:, -1:]
        expected = base.sample_actions(
            images, batch["tokens"], batch["token_mask"], num_steps=2, noise=noise
        )
        for profile in ("history", "query", "both"):
            base.execution = profile
            actual = base.sample_actions(
                images, batch["tokens"], batch["token_mask"], num_steps=2, noise=noise
            )
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_profile_is_part_of_run_path_and_config(self):
        from rsci.backbones.pi05.train import parser, resolve

        for profile in ("reference", "history", "query", "both"):
            args = parser().parse_args(
                [
                    "--dataset",
                    "/datasets/test",
                    "--sidecar",
                    "/labels/test",
                    "--task",
                    "between",
                    "--execution",
                    profile,
                ]
            )
            path = resolve(args)["output"]
            self.assertIn("runs/pi05", str(path))
            self.assertIn(profile, path.parts)

    def test_combined_profile_two_rank_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                speed_ddp_worker, args=(str(Path(directory) / "gloo"),), nprocs=2, join=True
            )

    def test_busy_gpu_admission_does_not_initialize_cuda(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "speed_probe", Path(__file__).resolve().parents[1] / "scripts/validate_pi05_speed.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from types import SimpleNamespace

        for status in ("16826, 32607, 99", "0, 32607, 98", "0, 24000, 0"):
            with (
                patch.object(module.subprocess, "run", return_value=SimpleNamespace(stdout=status)),
                patch("torch.cuda.init") as initialize,
            ):
                with self.assertRaisesRegex(RuntimeError, "no CUDA test started"):
                    module.admit_gpu(0)
                initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
