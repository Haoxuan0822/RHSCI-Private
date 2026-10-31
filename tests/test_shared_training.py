"""公共训练实现的数值、分布式归一化及旧断点格式回归。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel

from gr00t.rsci import checkpoints as gr00t
from openpi.rsci import checkpoints as pi05
from rsci.data.sampling import EpochRandomSampler, EpochSampler
from rsci.runtime.losses import masked_ardy_loss, masked_auxiliary_loss


def loss_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        layer = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(1)
        model = DistributedDataParallel(layer)
        values = torch.tensor([[1.0], [9.0]]) if rank == 0 else torch.tensor([[2.0], [3.0]])
        target = torch.tensor([[0.0], [float("nan")]]) if rank == 0 else torch.zeros(2, 1)
        valid = torch.tensor([True, rank == 1])
        masked_auxiliary_loss(model(values), target, valid).backward()
        # 三个有效样本的集中式 MSE 梯度：2×(1²+2²+3²)/3。
        torch.testing.assert_close(layer.weight.grad, torch.tensor([[28.0 / 3]]))
    finally:
        dist.destroy_process_group()


class SharedTrainingTests(unittest.TestCase):
    def test_uneven_rank_valid_counts_match_global_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(loss_worker, args=("file://" + str(Path(directory) / "gloo"),), nprocs=2)

    def test_empty_labels_preserve_zero_gradient_and_valid_labels_contribute(self):
        prediction = torch.ones(3, 128, requires_grad=True)
        target = torch.full_like(prediction, float("nan"))
        empty = masked_ardy_loss(prediction, target, torch.zeros(3, dtype=torch.bool))
        empty.backward()
        self.assertEqual(empty.item(), 0)
        self.assertEqual(prediction.grad.abs().sum().item(), 0)
        prediction.grad = None
        target[1] = 0
        loss = masked_ardy_loss(prediction, target, torch.tensor([False, True, False]))
        loss.backward()
        self.assertEqual(loss.item(), 1)
        self.assertEqual(prediction.grad[[0, 2]].abs().sum().item(), 0)
        torch.testing.assert_close(prediction.grad[1], torch.full((128,), 2 / 128))

    def test_sampler_epoch_recovery_is_independent_of_global_rng(self):
        dataset = list(range(23))
        sampler = EpochSampler(dataset, 7)
        sampler.set_epoch(4)
        state = torch.get_rng_state().clone()
        expected = list(sampler)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.rand(100)
        restored = EpochSampler(dataset, 7)
        restored.set_epoch(4)
        self.assertEqual(list(restored), expected)
        plain = EpochRandomSampler(dataset, 7)
        plain.set_epoch(4)
        self.assertEqual(list(plain), [index for epoch, index in expected])
        self.assertEqual(sorted(plain), dataset)

    def test_pi05_and_gr00t_keep_schema_two_and_their_own_metadata(self):
        for protocol, metadata in [
            (pi05, ["rsci_config.json"]),
            (
                gr00t,
                ["config.json", "processor_config.json", "statistics.json", "embodiment_id.json"],
            ),
        ]:
            with (
                self.subTest(protocol=protocol.__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "checkpoint-1000"
                path.mkdir()
                for name in [
                    *metadata,
                    "scheduler.pt",
                    "training_args.bin",
                    "rng_state_0.pth",
                    "rng_state_1.pth",
                ]:
                    (path / name).write_bytes(b"fixture")
                save_file({"weight": torch.ones(2)}, str(path / "part.safetensors"))
                (path / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {"weight": "part.safetensors"}})
                )
                torch.save({"exp_avg": torch.zeros(2)}, path / "optimizer.pt")
                (path / "trainer_state.json").write_text(json.dumps({"global_step": 1000}))
                marker = protocol.complete_checkpoint(path, 2, "recipe")
                self.assertEqual(
                    set(marker),
                    {"schema", "numerical_validation", "step", "world_size", "contract", "files"},
                )
                self.assertEqual(marker["schema"], 2)
                self.assertEqual(marker["world_size"], 2)
                self.assertTrue(set(metadata).issubset(marker["files"]))
                self.assertTrue(protocol.checkpoint_ok(path, 2, "recipe", verify_hash=True))
                (path / "rng_state_1.pth").unlink()
                self.assertFalse(protocol.checkpoint_ok(path, 2, "recipe"))
