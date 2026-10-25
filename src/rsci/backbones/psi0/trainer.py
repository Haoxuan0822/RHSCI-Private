"""PSI0数值检查、可恢复采样进度与训练器集成。"""

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from psi.trainers.finetune import FinetuneTrainer
from rsci.runtime.checkpoints import check_tensors

from .checkpoints import numerical_checkpoint_ok


class EpochRandomSampler(Sampler):
    """按seed/epoch重建排列，独立于worker预取及DataLoader generator消耗。"""

    def __init__(self, data_source, seed):
        self.data_source, self.seed, self.epoch = data_source, int(seed), 0

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self), generator=generator).tolist()


class DataProgress:
    def __init__(self, seed, batches, batch_size, world_size):
        self.contract = dict(
            schema=1,
            seed=seed,
            batches=batches,
            batch_size=batch_size,
            world_size=world_size,
            sampler="seed-plus-epoch-randperm-v1",
        )
        self.epoch, self.consumed = 0, 0
        self.restored = False

    def state_dict(self):
        # 在epoch末保存时，记录下一epoch的起点。
        epoch, consumed = self.epoch, self.consumed
        if consumed == self.contract["batches"]:
            epoch, consumed = epoch + 1, 0
        return dict(contract=self.contract, epoch=epoch, consumed=consumed)

    def load_state_dict(self, state):
        if state["contract"] != self.contract:
            raise ValueError("Data progress contract changed; exact sample resume refused")
        self.epoch, self.consumed = int(state["epoch"]), int(state["consumed"])
        if self.epoch < 0 or not 0 <= self.consumed < self.contract["batches"]:
            raise ValueError("Invalid saved data cursor")
        self.restored = True


def failure(trainer, stage, reason):
    """各rank先汇合失败状态，再一起退出，不能让单rank提前异常挂死其它rank。"""
    output = Path(trainer.cfg.train.output_dir)
    if trainer.accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        marker = output / "NUMERICAL_FAILURE.json"
        pending = marker.with_suffix(".pending")
        pending.write_text(
            json.dumps(dict(stage=stage, step=getattr(trainer, "global_step", None), reason=reason))
            + "\n"
        )
        pending.replace(marker)
    raise FloatingPointError(f"NUMERICAL_FAILURE {stage}: {reason}")


def finite_guard(trainer, values, stage):
    flags = []
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() or value.is_complex():
                flags.append(torch.isfinite(value.detach()).all().to(trainer.accelerator.device))
        elif isinstance(value, float):
            flags.append(torch.tensor(math.isfinite(value), device=trainer.accelerator.device))
    healthy = (
        torch.stack(flags).all() if flags else torch.tensor(True, device=trainer.accelerator.device)
    )
    bad = (~healthy).to(torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
    if bad.item():
        failure(trainer, stage, "nonfinite loss/gradient/update detected on at least one rank")


class RSCIFinetuneTrainer(FinetuneTrainer):
    """PSI0 trainer with deterministic sampling and validated checkpoint boundaries."""

    def create_dataloaders(self, train_dataset, val_dataset):
        super().create_dataloaders(train_dataset, val_dataset)
        old = self.train_dataloader
        sampler = EpochRandomSampler(train_dataset, 42 if self.cfg.seed is None else self.cfg.seed)
        kwargs = dict(
            dataset=old.dataset,
            batch_size=old.batch_size,
            sampler=sampler,
            num_workers=old.num_workers,
            collate_fn=old.collate_fn,
            drop_last=old.drop_last,
            pin_memory=old.pin_memory,
            worker_init_fn=old.worker_init_fn,
            persistent_workers=old.persistent_workers,
            generator=torch.Generator().manual_seed(42 if self.cfg.seed is None else self.cfg.seed),
        )
        if old.num_workers:
            kwargs["prefetch_factor"] = old.prefetch_factor
        self.train_dataloader = DataLoader(**kwargs)
        self._epoch_sampler = sampler
        return self.train_dataloader, self.val_dataloader

    def prepare(self, accelerator):
        if self.train_cfg.data_parallel != "ddp":
            raise ValueError("RHSCI PSI0 training supports DDP only")
        result = super().prepare(accelerator)
        self._data_progress = DataProgress(
            42 if self.cfg.seed is None else self.cfg.seed,
            len(self.train_dataloader),
            self.device_train_batch_size,
            accelerator.num_processes,
        )
        accelerator.register_for_checkpointing(self._data_progress)
        return result

    def validate_training_values(self, values, stage):
        finite_guard(self, values, stage)

    def next_epoch(self, epoch):
        super().next_epoch(epoch)
        self._epoch_sampler.set_epoch(epoch)
        self._data_progress.epoch = epoch
        if hasattr(self.train_dataloader, "set_epoch"):
            self.train_dataloader.set_epoch(epoch)
        if hasattr(self.train_dataloader, "iteration"):
            self.train_dataloader.iteration = epoch

    def step(self, batch, global_step, local_step):
        result = super().step(batch, global_step, local_step)
        self._data_progress.consumed = local_step + 1
        return result

    @property
    def num_steps_per_epoch(self):
        return max(math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps), 1)

    def resume_from_checkpoint(self):
        path = self.cfg.train.resume_from_checkpoint
        if path:
            checkpoint = Path(path)
            if not checkpoint.name.startswith("ckpt_"):
                raise ValueError("RHSCI resume requires an explicit checkpoint directory")
            if not numerical_checkpoint_ok(checkpoint):
                raise ValueError("Nonfinite or invalid checkpoint; resume refused")
            if not (checkpoint / "custom_checkpoint_0.pkl").is_file():
                raise ValueError(
                    "Checkpoint has no saved data progress; sample-position resume impossible"
                )
        result = super().resume_from_checkpoint()
        if path:
            epoch = result[0] // self.num_steps_per_epoch
            if not self._data_progress.restored or self._data_progress.epoch != epoch:
                raise ValueError("Saved data epoch disagrees with optimizer progress")
        return result

    def resumed_batches(self, initial_global_step):
        if self._data_progress.restored:
            return self._data_progress.consumed
        return super().resumed_batches(initial_global_step)

    def save_checkpoint(self, global_step):
        # Validate before FinetuneTrainer writes model files or prunes old checkpoints.
        error = None
        try:
            check_tensors(self.accelerator.unwrap_model(self.model).state_dict(), "model")
            check_tensors(self.optimizer.state_dict(), "optimizer")
        except ValueError as exc:
            error = str(exc)
        finite_guard(self, [float("nan") if error else 0.0], "checkpoint-memory")
        return super().save_checkpoint(global_step)

    def validate_saved_checkpoint(self, path):
        valid = numerical_checkpoint_ok(path) if self.accelerator.is_main_process else True
        finite_guard(self, [0.0 if valid else float("nan")], "checkpoint-disk")
