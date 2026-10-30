"""pi05: 骨干Trainer、数值门禁与完整断点回调。"""

from .checkpoints import atomic_json
from .config import simple_lr_multiplier

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def trainer_class():
    """正式训练和 CPU 保存/恢复验证复用同一 Trainer，避免测试绕过生产封装。"""
    import torch
    import torch.distributed as dist
    from transformers import Trainer

    from rsci.data.loader_cleanup import close_dataloaders
    from rsci.runtime.numerics import require_finite

    class RSCITrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.model_accepts_loss_kwargs = False
            self._rsci_loaders = []
            clip = self.accelerator.clip_grad_norm_

            def checked_clip(*args, **kwargs):
                norm = clip(*args, **kwargs)
                # 检查原裁剪返回的范数，覆盖 loss 有限但反向溢出的情况；不重复扫描梯度。
                require_finite(
                    [norm], self.args.output_dir, self.state.global_step, "gradient norm"
                )
                return norm

            self.accelerator.clip_grad_norm_ = checked_clip

        def get_train_dataloader(self):
            loader = super().get_train_dataloader()
            self._rsci_loaders.append(loader)
            return loader

        def train(self, *a, **kw):
            try:
                return super().train(*a, **kw)
            finally:
                # 沿用 Psi0 已有清理，避免持久视频 worker 在解释器退出时 abort。
                close_dataloaders(*self._rsci_loaders)
                self._rsci_loaders.clear()

        def _wrap_model(self, *a, **kw):
            wrapped = super()._wrap_model(*a, **kw)
            if self.accelerator.ddp_handler is not None:
                self.accelerator.ddp_handler.gradient_as_bucket_view = True
            return wrapped

        def _rotate_checkpoints(self, *a, **kw):
            pass

        def create_optimizer(self):
            if self.optimizer is None:
                # 原生Torch脚本对所有可训练参数使用同一衰减，含偏置和范数。
                self.optimizer = torch.optim.AdamW(
                    (p for p in self.model.parameters() if p.requires_grad),
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                    weight_decay=self.args.weight_decay,
                    fused=True,
                )
            return self.optimizer

        def create_scheduler(self, num_training_steps, optimizer=None):
            if self.lr_scheduler is None:
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer or self.optimizer,
                    lr_lambda=lambda step: simple_lr_multiplier(step, self.args.max_steps),
                )
                self._created_lr_scheduler = True
            return self.lr_scheduler

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            output = model(**inputs)
            require_finite(
                [output[key] for key in ("loss", "action_loss", "ardy_loss") if key in output],
                self.args.output_dir,
                self.state.global_step,
                "loss",
            )
            if self.state.global_step % 10 == 0:
                keys = [key for key in ("action_loss", "ardy_loss") if key in output]
                values = torch.stack([output[key].float() for key in keys])
                if dist.is_initialized():
                    dist.all_reduce(values)
                    values /= dist.get_world_size()
                self.log(dict(zip(keys, values.cpu().tolist())))
            return (output["loss"], output) if return_outputs else output["loss"]

    return RSCITrainer


def complete_callback(output, world, digest):
    """所有 rank 写完 RNG 后才发布完整标记；验证与生产共用回调。"""
    import torch.distributed as dist
    from transformers import TrainerCallback

    from .checkpoints import complete_checkpoint, prune_rolling

    class CompleteCallback(TrainerCallback):
        def on_save(self, arguments, state, control, model=None, **kw):
            if dist.is_initialized():
                dist.barrier()
            if state.is_world_process_zero:
                target = output / f"checkpoint-{state.global_step}"
                atomic_json(target / "rsci_config.json", model.rsci_config)
                complete_checkpoint(target, world, digest)
                prune_rolling(output, target, world, digest)
            if dist.is_initialized():
                dist.barrier()

    return CompleteCallback()
