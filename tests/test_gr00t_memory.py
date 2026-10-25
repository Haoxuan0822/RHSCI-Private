"""真实小DiT双调用梯度、两rank生产HF/Accelerate分片Adam及断点恢复。"""

import copy
import os
import socket
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_gr00t_rsci import batch, model
from transformers import Trainer, TrainerCallback, TrainingArguments

from rsci.backbones.gr00t import checkpoints
from rsci.backbones.gr00t.memory import (
    ShardedOptimizerTrainerMixin,
    enable_dit_recomputation,
    unwrap_zero,
)


class MemoryTrainer(ShardedOptimizerTrainerMixin, Trainer):
    pass


class StopAtTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True


def hf_worker(rank, directory, port):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        OMP_NUM_THREADS="1",
    )
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(37)
        initial = model("VGQ-AE")
        rows = [{"inputs": batch()} for _ in range(16)]
        for row in rows:
            row["inputs"]["rsci_ardy_valid"][:] = rank == 0

        def create(name, memory, stop=False):
            args = TrainingArguments(
                output_dir=str(Path(directory) / name),
                use_cpu=True,
                max_steps=4,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                save_steps=2,
                logging_strategy="no",
                report_to="none",
                disable_tqdm=True,
                remove_unused_columns=False,
                ddp_find_unused_parameters=False,
                optim="adamw_torch_fused",
                learning_rate=1e-4,
                weight_decay=1e-5,
                adam_beta1=0.9,
                adam_beta2=0.999,
                warmup_steps=1,
                seed=19,
            )
            policy = copy.deepcopy(initial)
            if memory:
                enable_dit_recomputation(policy)
            return (MemoryTrainer if memory else Trainer)(
                model=policy,
                args=args,
                train_dataset=rows,
                data_collator=lambda items: items[0],
                callbacks=[StopAtTwo()] if stop else [],
            )

        reference = create("reference", False)
        reference.train()
        uninterrupted = create("uninterrupted", True)
        uninterrupted.train()
        zero = unwrap_zero(uninterrupted.optimizer)
        # 每卡只持有一部分动量，所有份额之和等于原生完整状态数量。
        local = sum(
            value.numel()
            for state in zero.optim.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.ndim > 0
        )
        count = torch.tensor(local)
        dist.all_reduce(count)
        full = sum(
            value.numel()
            for state in reference.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.ndim > 0
        )
        assert 0 < local < full and count.item() == full
        assert not uninterrupted.optimizer.device_placement
        assert not zero._all_state_dicts
        for left, right in zip(reference.model.parameters(), uninterrupted.model.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)

        interrupted = create("resume", True, stop=True)
        interrupted.train()
        saved = Path(directory) / "resume/checkpoint-2"
        if rank == 0:
            for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
                (saved / name).write_text("{}")
            checkpoints.complete_checkpoint(saved, 2, "zero-resume")
            assert checkpoints.checkpoint_ok(saved, 2, "zero-resume", verify_hash=True)
        dist.barrier()
        restored = create("resume", True)
        restored.train(resume_from_checkpoint=str(saved))
        assert restored.state.global_step == 4
        for left, right in zip(uninterrupted.model.parameters(), restored.model.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert restored.lr_scheduler.state_dict() == uninterrupted.lr_scheduler.state_dict()
        left, right = unwrap_zero(restored.optimizer), unwrap_zero(uninterrupted.optimizer)
        for p, q in zip(restored.model.parameters(), uninterrupted.model.parameters()):
            a, b = left.optim.state.get(p, {}), right.optim.state.get(q, {})
            assert a.keys() == b.keys()
            for name in a:
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        print(f"ZERO_HF_RESUME_OK rank={rank} local_state_elements={local} full={full}", flush=True)
    finally:
        dist.destroy_process_group()


class MemoryTests(unittest.TestCase):
    def test_gpu_gate_refuses_busy_or_missing_devices(self):
        from scripts.check_gr00t_memory_devices import require_idle_devices

        idle = "\n".join(f"{index}, 18" for index in range(8))
        with patch("subprocess.check_output", return_value=idle):
            require_idle_devices("0,1,2,3,4,5,6,7")
        for rows in (idle.replace("3, 18", "3, 18900"), "\n".join(idle.splitlines()[:-1])):
            with (
                patch("subprocess.check_output", return_value=rows),
                self.assertRaises(RuntimeError),
            ):
                require_idle_devices("0,1,2,3,4,5,6,7")

    def test_shared_dit_recomputation_preserves_dropout_masks_and_all_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(83)
        plain = model("VGQ-AE").train()
        for module in plain.action_head.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.2
        recompute = copy.deepcopy(plain)
        enable_dit_recomputation(recompute)
        inputs = batch()
        for policy in (plain, recompute):
            torch.manual_seed(41)
            output = policy(copy.deepcopy(inputs))
            output["loss"].backward()
        # 包含两次共享DiT调用、Reader memory梯度、时间条件以及dropout反向重算。
        for (name, a), (_, b) in zip(plain.named_parameters(), recompute.named_parameters()):
            if a.requires_grad:
                self.assertIsNotNone(a.grad, name)
                self.assertIsNotNone(b.grad, name)
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
        minimal = {key: inputs[key] for key in ("input_ids", "attention_mask", "embodiment_id")}
        torch.manual_seed(44)
        a = plain.eval().get_action(copy.deepcopy(minimal))["action_pred"]
        torch.manual_seed(44)
        b = recompute.eval().get_action(copy.deepcopy(minimal))["action_pred"]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_two_rank_hf_accumulation_save_resume_matches_native_adam(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as connection:
            connection.bind(("127.0.0.1", 0))
            port = connection.getsockname()[1]
            connection.close()
            mp.start_processes(
                hf_worker, args=(directory, port), nprocs=2, join=True, start_method="spawn"
            )


if __name__ == "__main__":
    unittest.main()
