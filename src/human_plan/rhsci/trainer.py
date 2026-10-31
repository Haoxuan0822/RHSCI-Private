"""egovla: 骨干Trainer、数值门禁与完整断点回调。"""

import json
from functools import partial
from pathlib import Path

from rhsci.runtime.hf_callbacks import build_checkpoint_callback as shared_checkpoint_callback
from rhsci.runtime.numerics import require_finite as require_finite

from . import checkpoints as checkpoint_protocol
from .checkpoints import atomic_json

build_checkpoint_callback = partial(shared_checkpoint_callback, protocol=checkpoint_protocol)

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def create_trainer_class():
    """延迟导入骨干依赖并返回 Trainer 子类；此处不实例化 Trainer 或加载模型权重。"""
    import torch
    from transformers import Trainer

    from .checkpoints import load_trainables
    from .resume_tail import tail_safe_training_loop

    class RHSCITrainer(Trainer):
        _inner_training_loop = tail_safe_training_loop(Trainer)

        def __init__(self, *args, launch_contract=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.launch_contract = launch_contract

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            require_finite(
                list(outputs.values()), self.args.output_dir, self.state.global_step, "loss"
            )
            if model.training and self.state.global_step % 10 == 0:
                self.log({name: value.item() for name, value in outputs.items() if name != "loss"})
            return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

        def _wrap_model(self, *args, **kwargs):
            model = super()._wrap_model(*args, **kwargs)
            # Trainer在此创建DDP kwargs，Accelerate随后才真正包装模型。
            # 梯度直接使用通信bucket存储，避免额外保留一份全量VLM梯度。
            if self.accelerator.ddp_handler is not None:
                self.accelerator.ddp_handler.gradient_as_bucket_view = True
            return model

        def training_step(self, model, inputs, *args, **kwargs):
            loss = super().training_step(model, inputs, *args, **kwargs)
            gradients = [
                parameter.grad for parameter in model.parameters() if parameter.grad is not None
            ]
            norms = torch.stack([gradient.detach().float().norm() for gradient in gradients])
            require_finite([norms.norm()], self.args.output_dir, self.state.global_step, "gradient")
            return loss

        def _save(self, output_dir=None, state_dict=None):
            from safetensors.torch import save_file

            output = Path(output_dir or self.args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            state = {
                name: parameter.detach().cpu().contiguous()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            save_file(state, str(output / "model.safetensors"))
            atomic_json(output / "rsci_config.json", model.rsci_config)
            atomic_json(output / "launch_contract.json", self.launch_contract)
            torch.save(self.args, output / "training_args.bin")

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            # 基础模型已从同一SHA资产重新构造；包括解冻VLM的所有可训练参数逐键恢复。
            target = self.accelerator.unwrap_model(model or self.model)
            path = Path(resume_from_checkpoint)
            if json.loads((path / "launch_contract.json").read_text()) != self.launch_contract:
                raise ValueError("Resume contract changed")
            load_trainables(target, path)

    return RHSCITrainer


def configure_vlm_execution(model, args):
    """核验实际组件，而非仅检查Trainer字段；重算与是否训练参数相互独立。"""
    modules = dict(
        vision=model.backbone.get_vision_tower(),
        projector=model.backbone.get_mm_projector(),
        language=model.backbone.get_llm(),
    )
    enabled = args.gradient_checkpointing and args.vlm_training == "full"
    if enabled:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        for module in (
            modules["language"].model,
            getattr(modules["vision"], "vision_tower", modules["vision"]),
        ):
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()
    actual = dict(
        language=bool(modules["language"].model.is_gradient_checkpointing),
        vision=bool(
            getattr(
                getattr(modules["vision"], "vision_tower", modules["vision"]),
                "is_gradient_checkpointing",
                False,
            )
        ),
    )
    if any(value != enabled for value in actual.values()):
        raise RuntimeError(f"Actual VLM recomputation differs: {actual}, expected={enabled}")
    counts = {}
    for name, module in modules.items():
        parameters = list(module.parameters())
        if any(p.requires_grad != (args.vlm_training == "full") for p in parameters):
            raise RuntimeError(f"Official VLM trainability differs: {name}")
        counts[name] = dict(
            total=sum(p.numel() for p in parameters),
            trainable=sum(p.numel() for p in parameters if p.requires_grad),
        )
    print(
        "EGOVLA_VLM_EXECUTION " + json.dumps(dict(recomputation=actual, parameters=counts)),
        flush=True,
    )
    return actual, counts
