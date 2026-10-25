"""真实Accelerate保存恢复、坏断点与多worker采样回归；不占GPU。"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))
import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset

from rsci.backbones.psi0.checkpoints import checkpoint_ok
from rsci.backbones.psi0.trainer import DataProgress, EpochRandomSampler, finite_guard


def distributed_failure_worker(rank, rendezvous, output):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(train=SimpleNamespace(output_dir=output)),
        accelerator=SimpleNamespace(device=torch.device("cpu"), is_main_process=rank == 0),
        global_step=7,
    )
    caught = []
    for stage, bad_rank in [("loss", 1), ("backward", 0)]:
        try:
            finite_guard(trainer, [torch.tensor(float("nan") if rank == bad_rank else 1.0)], stage)
        except FloatingPointError:
            caught.append(stage)
    (Path(output) / f"rank-{rank}.json").write_text(__import__("json").dumps(caught))
    dist.destroy_process_group()


class SafetyTests(unittest.TestCase):
    def test_two_rank_failure_is_collective(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                distributed_failure_worker,
                args=(directory + "/rendezvous", directory),
                nprocs=2,
                join=True,
            )
            for rank in range(2):
                self.assertEqual(
                    __import__("json").loads((Path(directory) / f"rank-{rank}.json").read_text()),
                    ["loss", "backward"],
                )

    def checkpoint(self, root):
        save_file({"weight": torch.ones(4)}, str(root / "model.safetensors"))
        torch.save({"state": {0: {"exp_avg": torch.ones(4)}}}, root / "optimizer.bin")
        torch.save({"last_epoch": 1}, root / "scheduler.bin")
        torch.save({"seed": 1}, root / "random_states_0.pkl")

    def test_checkpoint_nan_model_and_inf_optimizer_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkpoint(root)
            self.assertTrue(checkpoint_ok(root, 1))
            save_file({"weight": torch.tensor([float("nan")])}, str(root / "model.safetensors"))
            self.assertFalse(checkpoint_ok(root, 1))
            self.checkpoint(root)
            torch.save(
                {"state": {0: {"exp_avg": torch.tensor([float("inf")])}}}, root / "optimizer.bin"
            )
            self.assertFalse(checkpoint_ok(root, 1))
            self.checkpoint(root)
            self.assertTrue(checkpoint_ok(root, 1))

    def test_actual_accelerate_resume_persistent_workers_across_epochs(self):
        def setup():
            accelerator = Accelerator(cpu=True)
            data = TensorDataset(torch.arange(43))
            sampler = EpochRandomSampler(data, 292285)
            loader = DataLoader(
                data,
                batch_size=4,
                sampler=sampler,
                num_workers=2,
                persistent_workers=True,
                drop_last=True,
                generator=torch.Generator().manual_seed(292285),
            )
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
            progress = DataProgress(292285, len(loader), 4, accelerator.num_processes)
            accelerator.register_for_checkpointing(progress)
            return accelerator, loader, sampler, progress

        with tempfile.TemporaryDirectory() as directory:
            first, loader, sampler, progress = setup()
            for epoch in range(2):
                loader.set_epoch(epoch)
                progress.epoch = epoch
                iterator = iter(loader)
                if epoch == 0:
                    list(iterator)
                else:
                    for index in range(3):
                        next(iterator)
                        progress.consumed = index + 1
            first.save_state(directory)
            expected = [next(iterator)[0].tolist() for _ in range(3)]
            second, restored, sampler2, cursor = setup()
            second.load_state(directory)
            self.assertTrue(cursor.restored)
            self.assertEqual((cursor.epoch, cursor.consumed), (1, 3))
            restored.set_epoch(cursor.epoch)
            resumed = iter(restored)
            for _ in range(cursor.consumed):
                next(resumed)
            self.assertEqual(expected, [next(resumed)[0].tolist() for _ in range(3)])
            # epoch边界保存须指向下一轮，不能重播最后一轮。
            progress.consumed = len(loader)
            self.assertEqual(progress.state_dict()["epoch"], 2)
            self.assertEqual(progress.state_dict()["consumed"], 0)
            for value in (loader, restored):
                if getattr(value, "_iterator", None):
                    value._iterator._shutdown_workers()

    def test_data_contract_mismatch_rejected(self):
        progress = DataProgress(4, 10, 4, 1)
        state = progress.state_dict()
        other = DataProgress(4, 10, 8, 1)
        with self.assertRaisesRegex(ValueError, "contract"):
            other.load_state_dict(state)

    def test_nonfinite_loss_or_gradient_stops_before_update(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = SimpleNamespace(
                cfg=SimpleNamespace(train=SimpleNamespace(output_dir=directory)),
                accelerator=SimpleNamespace(device=torch.device("cpu"), is_main_process=True),
                global_step=3,
            )
            for stage, values in [
                ("loss", [torch.tensor(float("nan"))]),
                ("backward", [torch.tensor([1.0, float("inf")])]),
            ]:
                with self.assertRaises(FloatingPointError):
                    finite_guard(trainer, values, stage)
                self.assertTrue((Path(directory) / "NUMERICAL_FAILURE.json").is_file())


if __name__ == "__main__":
    unittest.main()
