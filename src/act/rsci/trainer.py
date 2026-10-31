"""act: 骨干Trainer、数值门禁与完整断点回调。"""

import json
from pathlib import Path

from .checkpoints import atomic_json

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def require_finite(values, output, step, stage):
    import torch
    import torch.distributed as dist

    valid = torch.stack([torch.isfinite(value.detach()).all() for value in values]).all().int()
    if dist.is_initialized():
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        if not dist.is_initialized() or dist.get_rank() == 0:
            atomic_json(Path(output) / "NUMERICAL_FAILURE.json", dict(step=step, stage=stage))
        raise FloatingPointError(f"Nonfinite {stage}; automatic restart disabled")


def create_optimizer(model, *, cpu=False):
    import torch

    # 必须单组所有参数；HF默认的bias/LayerNorm免衰减分组不等价于SIMPLE原生构造。
    return torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6, fused=not cpu
    )


def trainer_type():
    import torch
    from transformers import Trainer

    from rsci.runtime.hf_resume import epoch_safe_training_loop

    from .checkpoints import finite_state, load_model_state
    from .data import EpochSampler

    class ACTTrainer(Trainer):
        _inner_training_loop = epoch_safe_training_loop(Trainer)

        def __init__(self, *args, launch_contract=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.launch_contract = launch_contract
            # 模型返回的是每微批mean；必须由HF按累计次数缩放，不能因**batch误判为自带分母。
            self.model_accepts_loss_kwargs = False

        def _get_train_sampler(self, train_dataset=None):
            return EpochSampler(
                train_dataset if train_dataset is not None else self.train_dataset, self.args.seed
            )

        def create_optimizer(self):
            if self.optimizer is None:
                self.optimizer = create_optimizer(self.model, cpu=self.args.use_cpu)
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            require_finite(
                list(outputs.values()), self.args.output_dir, self.state.global_step, "loss"
            )
            return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

        def training_step(self, model, inputs, *args, **kwargs):
            loss = super().training_step(model, inputs, *args, **kwargs)
            gradients = [
                p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None
            ]
            require_finite(
                [torch.stack(gradients).norm()],
                self.args.output_dir,
                self.state.global_step,
                "gradient",
            )
            return loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad(), self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
                # 单独报告真正z=0动作误差；不得把posterior重建误差冒充推理误差。
                native = self.accelerator.unwrap_model(model)
                prediction = native.predict_action(inputs)
                mask = ~inputs["action_is_pad"]
                error = (prediction.float() - inputs["action"].float()).abs()
                require_finite(
                    [prediction, error], self.args.output_dir, self.state.global_step, "validation"
                )
                valid_error = error[mask]
                normalized = valid_error.mean() if valid_error.numel() else error.sum() * 0
                physical = self.eval_dataset.processor.field.denormalize_L1_action_err(
                    valid_error.cpu()
                ).mean()
                self._eval_action_errors.append(
                    (normalized.detach(), physical.to(loss.device), mask.sum())
                )
            return loss.detach(), None, None

        def evaluate(self, *args, **kwargs):
            self._eval_action_errors = []
            result = super().evaluate(*args, **kwargs)
            if self._eval_action_errors:
                totals = torch.stack(
                    [torch.stack((a * n, b * n, n)) for a, b, n in self._eval_action_errors]
                ).sum(0)
                totals = self.accelerator.reduce(totals, reduction="sum")
                metrics = {
                    "eval/z0_l1": (totals[0] / totals[2].clamp_min(1)).item(),
                    "eval/z0_physical_l1": (totals[1] / totals[2].clamp_min(1)).item(),
                }
                self.log(metrics)
                result.update(metrics)
            return result

        def _wrap_model(self, *args, **kwargs):
            model = super()._wrap_model(*args, **kwargs)
            if self.accelerator.ddp_handler is not None:
                self.accelerator.ddp_handler.gradient_as_bucket_view = True
            return model

        def _save(self, output_dir=None, state_dict=None):
            from safetensors.torch import save_file

            output = Path(output_dir or self.args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            state = {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            }
            if not finite_state(state):
                atomic_json(
                    Path(self.args.output_dir) / "NUMERICAL_FAILURE.json",
                    dict(step=self.state.global_step, stage="model-before-save"),
                )
                raise FloatingPointError("Nonfinite model before checkpoint save")
            save_file(state, str(output / "model.safetensors"))
            atomic_json(output / "rsci_config.json", model.rsci_config)
            atomic_json(output / "launch_contract.json", self.launch_contract)
            torch.save(self.args, output / "training_args.bin")

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            path = Path(resume_from_checkpoint)
            if json.loads((path / "launch_contract.json").read_text()) != self.launch_contract:
                raise ValueError("Resume training contract changed")
            load_model_state(self.accelerator.unwrap_model(model or self.model), path)

        def _load_optimizer_and_scheduler(self, checkpoint):
            super()._load_optimizer_and_scheduler(checkpoint)
            if checkpoint is not None and not finite_state(self.optimizer.state_dict()):
                raise FloatingPointError("Nonfinite restored optimizer")

    return ACTTrainer


def completion_callback(output, world, digest):
    import torch.distributed as dist
    from transformers import TrainerCallback

    from .checkpoints import prune, publish

    class Complete(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if dist.is_initialized():
                dist.barrier()
            if state.is_world_process_zero:
                try:
                    publish(output / f"checkpoint-{state.global_step}", world, digest)
                    prune(output, output / f"checkpoint-{state.global_step}", world, digest)
                except FloatingPointError:
                    atomic_json(
                        output / "NUMERICAL_FAILURE.json",
                        dict(step=state.global_step, stage="checkpoint"),
                    )
                    raise
            if dist.is_initialized():
                dist.barrier()

    return Complete()
