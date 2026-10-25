from __future__ import annotations
from typing import Union, Optional, List, Any, TYPE_CHECKING
from pathlib import Path
import torch
import math
from copy import copy
from abc import ABC, abstractmethod
if TYPE_CHECKING:
    from psi.config.config import LaunchConfig, TrainConfig, LoggingConfig
    from psi.config.config import  DataConfig
from accelerate import Accelerator
from transformers.trainer_utils import PredictionOutput
from transformers.optimization import get_scheduler
from torch.optim import Optimizer
from psi.utils import snake_to_pascal
import os
import re
import importlib
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import datetime
import accelerate
import random
import shutil

from psi.utils import initialize_overwatch
overwatch = initialize_overwatch(__name__)



def resumed_dataloader_batches(
    initial_global_step: int,
    num_optimizer_steps_per_epoch: int,
    gradient_accumulation_steps: int,
) -> int:
    """把当前 epoch 内已完成的 optimizer step 换算成需要跳过的 microbatch 数。"""
    if initial_global_step < 0:
        raise ValueError("initial_global_step must be non-negative")
    if num_optimizer_steps_per_epoch <= 0:
        raise ValueError("num_optimizer_steps_per_epoch must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    optimizer_steps_in_epoch = initial_global_step % num_optimizer_steps_per_epoch
    return optimizer_steps_in_epoch * gradient_accumulation_steps



def worker_init_fn(worker_id):
    # print(f"worker_init_fn called by worker {worker_id}")
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

class Trainer(ABC):
    cfg: LaunchConfig
    model: Any
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler

    train_dataset: Dataset
    val_dataset: Optional[Dataset]

    def __init__(self, cfg: LaunchConfig, device: Union[torch.device, int]):
        self.device = torch.device(device)
        self.cfg = cfg

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        if cfg.train.mixed_precision == "fp16":
            self.dtype = torch.float16
        elif cfg.train.mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # avoid duplicate run names in a real training run
        # should read the timestamp from command line args instead
        # self.timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
        self.timestamp = self.cfg.timestamp

    @classmethod
    def instantiate(
        cls, cfg: LaunchConfig, device: Union[torch.device, int]
    ) -> "Trainer":
        if cfg.train.name == "finetune" and getattr(cfg.model, "rsci_variant", None):
            from rsci.backbones.psi0.trainer import RSCIFinetuneTrainer

            return RSCIFinetuneTrainer(cfg, device)
        trainer_name = cfg.train.name
        try:
            parts = re.split(r"[-_]", trainer_name)
            trainer_name = "_".join([p.lower() for p in parts])
            module = importlib.import_module(f"psi.trainers.{trainer_name}")
            trainer_clazz = getattr(module, f"{snake_to_pascal(trainer_name)}Trainer")
        except Exception as e:
            raise ValueError(
                f"fail to import {trainer_name} from psi.trainers"
            ) from e

        return trainer_clazz(cfg, device)

    @property
    def train_cfg(self) -> TrainConfig:
        return self.cfg.train

    @property
    def log_cfg(self) -> LoggingConfig:
        return self.cfg.log

    @property
    def data_cfg(self) -> DataConfig:
        return self.cfg.data

    @property
    def hf_token(self):
        return self.cfg.train.hf_token.read_text().strip() \
            if isinstance(self.cfg.train.hf_token, Path) \
            else os.environ[self.cfg.train.hf_token]

    def get_fsdp_plugin(self) -> accelerate.utils.FullyShardedDataParallelPlugin | None: ...

    def create_datasets(self) -> tuple[Dataset, Dataset|None]:
        ...

    def create_dataloaders(
        self, train_dataset, val_dataset
    ) -> tuple[DataLoader, DataLoader|None]: ...

    def create_optimizer(self):
        optimizer_kwargs = dict(self.cfg.train.lr_scheduler_kwargs)
        if self.cfg.train.optimizer_foreach is not None:
            optimizer_kwargs["foreach"] = self.cfg.train.optimizer_foreach
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            **optimizer_kwargs, # type: ignore
        )

    def create_scheduler(
        self, num_training_steps: int | None = None, optimizer: Optimizer | None = None
    ):
        if num_training_steps is None:
            num_training_steps = self.max_training_steps

        # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            self.lr_scheduler = get_scheduler(
                name=self.cfg.train.lr_scheduler_type,
                optimizer=optimizer if optimizer is not None else self.optimizer,
                num_warmup_steps=self.num_warmup_steps * self.world_size,
                num_training_steps=num_training_steps * self.world_size,
                scheduler_specific_kwargs=self.cfg.train.scheduler_specific_kwargs,
            )
        else:
            self.lr_scheduler = accelerate.utils.DummyScheduler(
                optimizer, total_num_steps=self.max_training_steps, warmup_num_steps=self.num_warmup_steps
            ) # type: ignore

        return self.lr_scheduler

    def log(
        self, metrics: dict[str, Any], start_time: Optional[float] = None
    ) -> None:
        wandb_dict = copy(metrics)
        for k,v in list(wandb_dict.items()):
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                wandb_dict[k] = v.item()
        # wandb_dict.update({"train/grad_norm": grad_norm})
        # wandb_dict.update(self.get_log_kv())
        self.accelerator.log(wandb_dict, step=self.global_step) # WANDB logging

    def create_optimizer_and_scheduler(self, num_training_steps: int | None = None):
        optimizer = self.create_optimizer()
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=optimizer
        )
    def compute_loss(
        self, batch#, return_outputs=False#, num_items_in_batch=None
    ) -> dict: ...

    def training_step(
        self,
        # model: nn.Module,
        batch: dict[str, Union[torch.Tensor, Any]],
    ) -> tuple[bool, dict[str, Any]]: ...

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]: ...

    def evaluate(
        self,
        # eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # ignore_keys: Optional[list[str]] = None,
        # metric_key_prefix: str = "eval",
    ) -> dict[str, float] | None:
        ...

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput: ...

    @property
    def world_size(self) -> int:
        """Returns the number of processes in the current distributed group."""
        return overwatch.world_size()

    @property
    def device_train_batch_size(self) -> int:
        """Returns the training batch size per process."""
        return self.cfg.train.train_batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        """Returns the number of gradient accumulation steps."""
        return self.cfg.train.gradient_accumulation_steps

    @property
    def global_train_batch_size(self) -> int:
        """Returns the global training batch size."""
        return (
            self.device_train_batch_size
            * self.world_size
            * self.gradient_accumulation_steps
        )

    @property
    def len_train_dataset(self) -> int:
        """Returns the length of the training dataset."""
        if hasattr(self, "_len_train_dataset") and self._len_train_dataset: # type: ignore
            return self._len_train_dataset # type: ignore
        return len(self.train_dataset) # type: ignore

    @len_train_dataset.setter
    def len_train_dataset(self, length: int):
        self._len_train_dataset = length

    @property
    def len_val_dataset(self) -> int:
        """Returns the length of the training dataset."""
        if hasattr(self, "_len_train_dataset") and self._len_val_dataset: # type: ignore
            return self._len_val_dataset # type: ignore
        return len(self.val_dataset) # type: ignore

    @len_val_dataset.setter
    def len_val_dataset(self, length: int):
        self._len_val_dataset = length

    @property
    def len_train_dataloader(self) -> int:

        if getattr(self.cfg.model, "rsci_variant", None) and hasattr(self, "train_dataloader"):  # RHSCI同样按实际DDP分片续训。
            if hasattr(self, "_prepared_loader_length"):
                return self._prepared_loader_length
            raw_length = len(self.train_dataloader)
            return (raw_length // self.world_size if self.train_dataloader.drop_last
                    else math.ceil(raw_length / self.world_size))

        total = self.len_train_dataset / (
            self.device_train_batch_size * self.world_size
        )
        if hasattr(self, "train_dataloader_drop_last") and self.train_dataloader_drop_last: # type: ignore
            return math.floor(total)  # drop the last incomplete batch
        else:
            return math.ceil(total)  # include the last incomplete batch


    @property
    def len_val_dataloader(self) -> int:
        if not hasattr(self, "val_dataloader") or self.val_dataloader is None:
            return 0
        total = self.len_val_dataset / (
            self.device_train_batch_size * self.world_size
        )
        if hasattr(self, "val_dataloader_drop_last") and self.val_dataloader_drop_last: # type: ignore
            return math.floor(total)  # drop the last incomplete batch
        else:
            return math.ceil(total)  # include the last incomplete batch

    @property
    def num_steps_per_epoch(self) -> int:
        """number of global steps (sync gradients) per epoch"""
        # assert self.len_train_dataloader > self.gradient_accumulation_steps
        return max(self.len_train_dataloader // self.gradient_accumulation_steps, 1)

    @property
    def max_training_steps(self) -> int:
        """Returns the maximum number of training steps."""
        if self.cfg.train.max_training_steps is not None:
            return self.cfg.train.max_training_steps
        else:
            assert self.cfg.train.num_train_epochs is not None
            return self.num_steps_per_epoch * self.world_size * self.cfg.train.num_train_epochs

    @property
    def max_training_epochs(self) -> int:
        return math.ceil(self.max_training_steps / self.num_steps_per_epoch)

    @property
    def num_warmup_steps(self) -> int:
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.cfg.train.warmup_steps
            if self.cfg.train.warmup_steps is not None
            else math.ceil(self.max_training_steps * self.cfg.train.warmup_ratio)  # type: ignore
        )
        return warmup_steps # because accelerate divides the steps by world_size when preparing the scheduler

    @property
    @abstractmethod
    def task_run_name(self) -> str:
        """Returns the task-specific run name."""

    @property
    def run_name(self) -> str:
        run_name = (
            f"{self.cfg.exp}{self.task_run_name}"
            f".b{self.global_train_batch_size}.gpus{overwatch.world_size()}"
        )

        run_name = f"{run_name}.{self.timestamp}"
        if self.cfg.debug:
            run_name = f"debug-{run_name}"
        return run_name

    @property
    def project_dir(self) -> str:
        """Returns the project directory for saving checkpoints and logs."""
        return os.path.join(
            self.cfg.train.output_dir, self.cfg.train.name, self.run_name
        )

    def next_epoch(self, epoch):
        """called between epochs, e.g. for resetting the distributed samplers"""
        if hasattr(self, "train_sampler") and self.train_sampler is not None: # type: ignore
            self.train_sampler.set_epoch(epoch) # type: ignore

    @abstractmethod
    def init_models(self):
        """Initialize the models for training."""

    # @abstractmethod
    def set_train(self):
        self.model.train()

    # @abstractmethod
    def set_eval(self):
        self.model.eval()

    # @abstractmethod
    # def create_optimizers(self):
    #     ...

    # @abstractmethod
    # def create_lr_schedulers(self):
    #     ...

    # @abstractmethod
    # def train_one_step(self, batch_input, global_step, local_step, accelerator=None):
    #     ...

    def step(self, batch_input, global_step, local_step) -> tuple[bool, dict[str, Any]]:
        """ Perform a single training step. """

        self.local_step = local_step
        self.global_step = global_step

        sync_gradients, losses = self.training_step(batch_input)
        return sync_gradients, losses

    @abstractmethod
    def prepare(self, accelerator: Accelerator) -> DataLoader:
        self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )

        self.train_dataloader = accelerator.prepare(self.train_dataloader)

        if self.cfg.train.overfit_single_batch:
            overwatch.warning("Overfitting a single batch: reusing first batch every step. set cfg.data.image_aug = False for true memorization.")
            first_batch = next(iter(self.train_dataloader))
            class SingleBatchLoader:
                def __iter__(self):
                    while True:
                        yield first_batch
                def __len__(self):
                    return 1
            self.train_dataloader = SingleBatchLoader()


        # FIXME if self.train_dataloader.dataset is IterableDataset:
        # assert (
        #     abs(len(self.train_dataloader) - self.len_train_dataloader)
        #     <= 1  # because of drop_last option is lost in prepare.
        # ), f"check calculations again, {len(self.train_dataloader)} != {self.len_train_dataloader}"

        val_dataloader = getattr(self, "val_dataloader", None)
        if val_dataloader is not None: # not using if self.val_dataloader to avoid DataLoader.__len__() being called on iterable dataset
            self.val_dataloader = accelerator.prepare(self.val_dataloader)
        """ NOTE:
            We have to manually record how many steps we have accumulated gradients
            because the accelerator does not do it for us.
        """
        self.accelerator = accelerator
        return self.train_dataloader # type: ignore

    def validate_training_values(self, values, stage):
        """Extension point for numerical validation around optimizer updates."""

    def validate_saved_checkpoint(self, path):
        """Validate a completed checkpoint before retention removes older ones."""

    def resumed_batches(self, initial_global_step):
        return resumed_dataloader_batches(
            initial_global_step, self.num_steps_per_epoch, self.gradient_accumulation_steps
        )

    def save_checkpoint(self, global_step: int) -> str | None:
        save_dir = os.path.join(self.project_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_dir = os.path.join(save_dir, f"ckpt_{global_step}")

        self.accelerator.save_state(ckpt_dir)
        self.accelerator.wait_for_everyone()
        self.validate_saved_checkpoint(ckpt_dir)


        anchor_steps = self.train_cfg.checkpoint_anchor_steps
        if anchor_steps is not None:
            # 所有 rank 同时验收完整 DDP 状态，防止单卡报错时其他 rank 卡在 barrier。
            required = ["model.safetensors", "optimizer.bin", "scheduler.bin"]
            required += [f"random_states_{rank}.pkl" for rank in range(self.accelerator.num_processes)]
            missing = [name for name in required if not os.path.isfile(os.path.join(ckpt_dir, name))
                       or os.path.getsize(os.path.join(ckpt_dir, name)) == 0]
            if missing:
                raise RuntimeError(f"Incomplete checkpoint {ckpt_dir}: {missing}")
        if self.accelerator.is_main_process:
            if anchor_steps is not None:
                for entry in os.scandir(save_dir):
                    if not entry.is_dir() or not entry.name.startswith("ckpt_"):
                        continue
                    suffix = entry.name.removeprefix("ckpt_")
                    if not suffix.isdecimal():
                        continue
                    step = int(suffix)
                    if step < global_step and step % anchor_steps != 0:
                        try:
                            shutil.rmtree(entry.path)
                            overwatch.info(f"Removed old rolling checkpoint: {entry.path}")
                        except OSError as exc:
                            overwatch.warning(f"Retained old rolling checkpoint {entry.path}: {exc}")
            else:
                # Keep only the latest max_checkpoints_to_keep checkpoints
                max_to_keep = self.train_cfg.max_checkpoints_to_keep or 100
                if max_to_keep is not None and max_to_keep > 0:
                    # List all checkpoint directories matching ckpt_*
                    ckpt_dirs = [d for d in os.listdir(save_dir) if d.startswith("ckpt_") and os.path.isdir(os.path.join(save_dir, d))]
                    # Extract step numbers and sort by step (assume ckpt_{step})
                    def extract_step(d):
                        try:
                            return int(d.split("ckpt_")[-1])
                        except Exception:
                            return -1
                    ckpt_dirs_sorted = sorted(ckpt_dirs, key=extract_step, reverse=True)
                    # Remove older checkpoints if exceeding max_to_keep
                    for old_ckpt in ckpt_dirs_sorted[max_to_keep:]:
                        old_ckpt_path = os.path.join(save_dir, old_ckpt)
                        if "0000" in old_ckpt:
                            # force keeping ckpt saved every 10k
                            continue
                        try:
                            shutil.rmtree(old_ckpt_path)
                            overwatch.info(f"Removed old checkpoint: {old_ckpt_path}")
                        except Exception as e:
                            overwatch.warning(f"Failed to remove old checkpoint {old_ckpt_path}: {e}")


        self.accelerator.wait_for_everyone()
        return ckpt_dir

    def resume_from_checkpoint(self) -> tuple[int, Optional[str]]:
        """ resume from a checkpoint if specified in the config.
            the checkpoint path can be either:
            1) full path to a checkpoint folder, e.g.
               .runs/trainer_name/run_name/checkpoints/ckpt_xxxxxx
            3) or .runs/trainer_name/run_name (latest)
        """
        if self.cfg.train.resume_from_checkpoint is None:
            return 0, None


        load_path = None  # 最终传给 Accelerate 的 checkpoint 目录。
        resume_path = self.cfg.train.resume_from_checkpoint.rstrip(os.sep)  # 去掉末尾斜杠，便于判断目录名。
        resume_name = os.path.basename(resume_path)  # 取得路径最后一段，例如 ckpt_5000。
        if resume_name.startswith("ckpt_"):
            load_path = resume_path  # 用户已传完整 checkpoint 路径，直接使用，不再次拼接。
            path = resume_name  # 从目录名中提取后续需要恢复的 step。
        else:
            if os.path.exists(f"{self.cfg.train.resume_from_checkpoint}/checkpoints"):
                # Get the most recent checkpoint
                dirs = [  # 只枚举符合 checkpoint 命名规则的目录。
                    directory  # 保留 ckpt_N 目录名。
                    for directory in os.listdir(
                        f"{self.cfg.train.resume_from_checkpoint}/checkpoints"
                    )
                    if directory.startswith("ckpt_")  # 排除 notes、日志等非 checkpoint 目录。
                ]
                dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))  # 按数字 step 排序，不能按字符串排序。
                path = dirs[-1] if len(dirs) > 0 else None  # run 目录模式下选择最新 checkpoint。
            else:
                path = None

        if path is None:
            overwatch.critical(
                f"Checkpoint '{self.cfg.train.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self.cfg.train.resume_from_checkpoint = None
            initial_global_step = 0
            load_path = None
        else:
            if load_path is None:  # 只有传入 run 目录时才需要拼接 checkpoints/ckpt_N。
                load_path = os.path.join(self.cfg.train.resume_from_checkpoint, "checkpoints", path)  # 形成真实加载路径。
            overwatch.info(f"Resuming from checkpoint {load_path}")
            self.accelerator.load_state(load_path)
            initial_global_step = int(path.split("_")[1])  # ckpt_N 已表示恢复到第 N 步，不再额外加 1。


        return initial_global_step, load_path

    @property
    def lr(self):
        return self.get_lr()

    def get_lr(self):
        return self.lr_scheduler.get_last_lr()[0]

    def get_total_grad_norm(self, params=None, norm_type=2):
        if "DeepSpeedEngine" in self.model.__class__.__name__:
            grad_norm = self.model.get_global_grad_norm()
            return grad_norm
        else:
            total_norm = 0.0
            if params is None:
                params = self.model.parameters()
            for p in params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type

            total_norm = total_norm ** (1. / norm_type)
            return total_norm

    def unwrap_model(self):
        # Function for unwrapping if model was compiled with `torch.compile`.
        model = self.accelerator.unwrap_model(self.model)
        from diffusers.utils.torch_utils import is_compiled_module

        model = model._orig_mod if is_compiled_module(model) else model
        # songlin: revert original forward which is changed by accelerate.prepare to always return fp32
        # if hasattr(model, "_original_forward"):
        #     model.forward = model._original_forward
        return model

    # def log(
    #     self, ):
    #     ...

    # def evaluate(self, global_step: int, accelerator: Accelerator):
    #     ...

    def finalize(self):

        from rsci.data.loader_cleanup import close_dataloaders
        closed = close_dataloaders(
            getattr(self, 'train_dataloader', None),
            getattr(self, 'val_dataloader', None),
        )
        overwatch.info(f"Closed {closed} persistent DataLoader iterator(s).")
