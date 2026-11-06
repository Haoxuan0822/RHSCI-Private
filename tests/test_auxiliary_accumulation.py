"""同一有效批的双头梯度应与微批拆分、空标签及 DDP 分布无关。"""

import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from rhsci.runtime.auxiliary_accumulation import (
    AuxiliaryAccumulationTrainerMixin,
    attach_auxiliary_normalization,
    normalized_auxiliary_batches,
)
from rhsci.runtime.losses import masked_auxiliary_loss


def batches():
    return [
        dict(
            value=torch.tensor([[float(i + 1)]]),
            ardy_valid=torch.tensor([i == 0]),
            motion_valid=torch.tensor([i >= 2]),
        )
        for i in range(4)
    ]


def auxiliary_loss(prediction, batch):
    normalization = batch.get("auxiliary_normalization") or {}
    return sum(
        weight
        * masked_auxiliary_loss(
            prediction,
            torch.zeros_like(prediction),
            batch[f"{name}_valid"],
            normalization_denominator=normalization.get(name),
        )
        for name, weight in (("ardy", 0.1), ("motion", 0.3))
    )


def distributed_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        layer = nn.Linear(1, 1, bias=False)
        nn.init.ones_(layer.weight)
        model = DistributedDataParallel(layer)
        group = batches()[rank * 2 : rank * 2 + 2]
        attach_auxiliary_normalization(group, "cpu", 2)
        for batch in group:
            (auxiliary_loss(model(batch["value"]), batch) / 2).backward()
        # 0.1 * 2*1² + 0.3 * 2*(3²+4²)/2
        torch.testing.assert_close(layer.weight.grad, torch.tensor([[7.7]]))
    finally:
        dist.destroy_process_group()


class AuxiliaryAccumulationTests(unittest.TestCase):
    def test_partition_and_empty_masks(self):
        for sizes, divisor in (([4], 1), ([2, 2], 2), ([1, 1, 1, 1], 4), ([2, 2], 4)):
            with self.subTest(sizes=sizes, divisor=divisor):
                source = batches()
                group, start = [], 0
                for size in sizes:
                    group.append(
                        {
                            key: torch.cat([b[key] for b in source[start : start + size]])
                            for key in source[0]
                        }
                    )
                    start += size
                attach_auxiliary_normalization(group, "cpu", divisor)
                value = torch.tensor(1.0, requires_grad=True)
                for batch in group:
                    (auxiliary_loss(value * batch["value"], batch) / divisor).backward()
                torch.testing.assert_close(value.grad, torch.tensor(7.7))
        group = batches()
        for batch in group:
            batch["ardy_valid"].fill_(False)
            batch["motion_valid"].fill_(False)
        attach_auxiliary_normalization(group, "cpu", 4)
        value = torch.tensor(1.0, requires_grad=True)
        for batch in group:
            (auxiliary_loss(value * batch["value"], batch) / 4).backward()
        self.assertEqual(value.grad.item(), 0)

    def test_ddp_uneven_masks_and_empty_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                distributed_worker, args=("file://" + str(Path(directory) / "gloo"),), nprocs=2
            )

    def test_gr00t_counts_joint_history_validity(self):
        group = [
            {
                "inputs": dict(
                    rsci_ardy_valid=torch.ones(2, dtype=torch.bool),
                    rsci_motion_valid=torch.ones(2, dtype=torch.bool),
                    rsci_history_valid=torch.tensor([[False, True], [True, True]]),
                )
            }
        ]
        attach_auxiliary_normalization(group, "cpu", 4)
        self.assertEqual(group[0]["inputs"]["auxiliary_normalization"]["ardy"].item(), 0.25)

    def test_lambda_backend_keeps_denominators_out_of_native_processor(self):
        from types import SimpleNamespace
        from lambda0.rhsci.runtime import RHSCITrainingBackend

        received = []

        def native_prepare(batch, *unused):
            received.append(batch)
            return dict(value=batch["value"])

        backend = RHSCITrainingBackend(
            SimpleNamespace(prepare_inputs=native_prepare), "VGQ-AE-Mo", {}
        )
        group = backend.prepare_training_batches(iter(batches()), 4, "cpu")
        first = backend.prepare_inputs(group[0], None, "cpu", None)
        self.assertNotIn("auxiliary_normalization", received[0])
        self.assertNotIn("ardy_valid", received[0])
        self.assertEqual(first["auxiliary_normalization"]["ardy"].item(), 0.25)
        self.assertEqual(first["auxiliary_normalization"]["motion"].item(), 0.5)

    def test_psi_prefetch_preserves_sync_and_loader_cleanup(self):
        from accelerate import Accelerator
        from torch.utils.data import DataLoader

        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        previous_loader = accelerator.gradient_state.active_dataloader
        model = nn.Linear(1, 1)
        loader = accelerator.prepare(DataLoader(batches()[:3], batch_size=None))
        flags = []
        for batch in normalized_auxiliary_batches(loader, 2, accelerator.device):
            with accelerator.accumulate(model):
                flags.append(accelerator.sync_gradients)
        self.assertEqual(flags, [False, True, True])
        self.assertIs(accelerator.gradient_state.active_dataloader, previous_loader)

    def test_actual_hf_training_update_matches_full_batch(self):
        from transformers import Trainer, TrainingArguments

        class TrainerWithAuxiliary(AuxiliaryAccumulationTrainerMixin, Trainer):
            pass

        class EpochDataset(list):
            def __getitem__(self, index):
                return super().__getitem__(index[1] if isinstance(index, tuple) else index)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def forward(self, value, ardy_valid, motion_valid, auxiliary_normalization=None):
                batch = dict(
                    ardy_valid=ardy_valid.reshape(-1),
                    motion_valid=motion_valid.reshape(-1),
                    auxiliary_normalization=auxiliary_normalization,
                )
                loss = auxiliary_loss(self.weight * value.reshape(-1, 1), batch)
                return dict(
                    loss=loss,
                    action_loss=loss.detach() * 0,
                    action_mask=torch.ones(1),
                    ardy_loss=loss.detach(),
                    motion_loss=loss.detach() * 0,
                )

        factory = os.environ.get("RHSCI_TEST_TRAINER")
        if factory:
            from importlib import import_module

            TrainerWithAuxiliary = import_module(factory).create_trainer_class()
        weights = []
        for micro, accumulation in ((4, 1), (2, 2), (1, 4), (2, 4)):
            with tempfile.TemporaryDirectory() as directory:
                model = Model()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                trainer = TrainerWithAuxiliary(
                    model=model,
                    train_dataset=EpochDataset(batches()),
                    optimizers=(
                        optimizer,
                        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
                    ),
                    args=TrainingArguments(
                        output_dir=directory,
                        use_cpu=True,
                        report_to="none",
                        save_strategy="no",
                        max_steps=1,
                        per_device_train_batch_size=micro,
                        gradient_accumulation_steps=accumulation,
                        max_grad_norm=0,
                        dataloader_pin_memory=False,
                        disable_tqdm=True,
                    ),
                )
                trainer.model_accepts_loss_kwargs = False
                trainer.train()
                weights.append(model.weight.detach().clone())
        for value in weights:
            torch.testing.assert_close(value, torch.tensor(0.923))


if __name__ == "__main__":
    unittest.main()
