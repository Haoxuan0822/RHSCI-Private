"""gr00t: 骨干Trainer、数值门禁与完整断点回调。"""


def create_resume_trainer_class():
    """返回保留 epoch、采样顺序与数据消费位置的 Trainer 子类，不实例化模型。"""
    from rhsci.runtime.hf_resume import (
        isolate_worker_rng,
        tail_safe_training_loop,
        tail_safe_training_values,
    )
    from transformers import Trainer

    class ResumeTrainer(Trainer):
        _inner_training_loop = tail_safe_training_loop(Trainer)
        set_initial_training_values = tail_safe_training_values(Trainer)

        def _get_train_sampler(self, train_dataset=None):
            from .data import EpochSampler

            return EpochSampler(
                self.train_dataset if train_dataset is None else train_dataset, self.args.seed
            )

        def get_train_dataloader(self):
            return isolate_worker_rng(super().get_train_dataloader(), self.args.seed)

    return ResumeTrainer


def create_trainer_class():
    """延迟导入骨干依赖并返回 Trainer 子类；此处不实例化 Trainer 或加载模型权重。

    装配原生分片优化器、采样恢复和数值门禁，不加载模型资产。"""
    from rhsci.runtime.numerics import require_finite
    import torch
    import torch.distributed as dist

    from .memory import ShardedOptimizerTrainerMixin

    class RHSCITrainer(ShardedOptimizerTrainerMixin, create_resume_trainer_class()):
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
                [output[key] for key in ("loss", "action_loss", "ardy_loss", "motion_loss") if key in output],
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
                if "motion_loss" in output:
                    logs["motion_loss"] = output["motion_loss"].detach().float()
                values = torch.stack(list(logs.values()))
                if dist.is_initialized():
                    dist.all_reduce(values)
                    values /= dist.get_world_size()
                self.log(dict(zip(logs, values.cpu().tolist())))
            return (output["loss"], output) if return_outputs else output["loss"]

    return RHSCITrainer


def build_checkpoint_callback(args, paths, processor, world, digest):
    """返回完整断点回调；等待各 rank 写完状态后再验证并发布完整标记。

    完整保存单帧推理处理器与断点后，才轮换旧滚动点。"""
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
