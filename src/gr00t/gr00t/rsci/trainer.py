"""gr00t: 骨干Trainer、数值门禁与完整断点回调。"""


# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def resume_trainer_type():
    from transformers import Trainer

    from rsci.runtime.hf_resume import epoch_safe_training_loop

    class ResumeTrainer(Trainer):
        _inner_training_loop = epoch_safe_training_loop(Trainer)

        def _get_train_sampler(self, train_dataset=None):
            from .data import EpochSampler

            return EpochSampler(
                self.train_dataset if train_dataset is None else train_dataset, self.args.seed
            )

        def get_train_dataloader(self):
            import torch

            loader = super().get_train_dataloader()
            # worker重建/跳batch不能消耗动作噪声与dropout使用的主线程RNG。
            base = getattr(loader, "base_dataloader", loader)
            base.generator = torch.Generator().manual_seed(self.args.seed)
            return loader

    return ResumeTrainer


def trainer_type():
    """装配原生分片优化器、采样恢复和数值门禁，不加载模型资产。"""
    import torch
    import torch.distributed as dist

    from rsci.runtime.numerics import require_finite

    from .memory import ShardedOptimizerTrainerMixin

    class RSCITrainer(ShardedOptimizerTrainerMixin, resume_trainer_type()):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clip = self.accelerator.clip_grad_norm_

            def checked_clip(*args, **kwargs):
                norm = clip(*args, **kwargs)
                require_finite(
                    [norm], self.args.output_dir, self.state.global_step, "gradient norm"
                )
                return norm

            self.accelerator.clip_grad_norm_ = checked_clip

        def _wrap_model(self, *arguments, **kwargs):
            wrapped = super()._wrap_model(*arguments, **kwargs)
            # HF Trainer在_wrap_model中重建ddp_handler；须在此后设置才会真正生效。
            if self.accelerator.ddp_handler is not None:
                self.accelerator.ddp_handler.gradient_as_bucket_view = True
            return wrapped

        def _rotate_checkpoints(self, *unused, **kwargs):
            pass  # 完整标记写入后才按本实验契约删除上一份滚动断点。

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            output = model(**inputs)
            require_finite(
                [output[key] for key in ("loss", "action_loss", "ardy_loss") if key in output],
                self.args.output_dir,
                self.state.global_step,
                "loss",
            )
            if self.state.global_step % self.args.logging_steps == 0:
                logs = {
                    "action_loss": (
                        output["action_loss"].sum() / output["action_mask"].sum().clamp_min(1)
                    )
                    .detach()
                    .float()
                }
                if "ardy_loss" in output:
                    logs["ardy_loss"] = output["ardy_loss"].detach().float()
                values = torch.stack(list(logs.values()))
                if dist.is_initialized():
                    dist.all_reduce(values)
                    values /= dist.get_world_size()
                self.log(dict(zip(logs, values.cpu().tolist())))
            return (output["loss"], output) if return_outputs else output["loss"]

    return RSCITrainer


def completion_callback(args, paths, processor, world, digest):
    """完整保存单帧推理处理器与断点后，才轮换旧滚动点。"""
    import torch.distributed as dist
    from transformers import TrainerCallback

    from .checkpoints import complete_checkpoint, prune_rolling
    from .data import modality

    class CompleteCallback(TrainerCallback):
        def on_save(self, arguments, state, control, **kwargs):
            if dist.is_initialized():
                dist.barrier()
            if state.is_world_process_zero:
                path = paths["output"] / f"checkpoint-{state.global_step}"
                # 推理配置只要求单帧；训练恢复时从固定配方重建四帧处理器。
                training_modality = processor.modality_configs
                processor.modality_configs = modality(args.model, training=False)
                processor.save_pretrained(path)
                processor.modality_configs = training_modality
                complete_checkpoint(path, world, digest)
                prune_rolling(paths["output"], path, world, digest)
            if dist.is_initialized():
                dist.barrier()

    return CompleteCallback()
