"""Explicit RHSCI backend for the upstream Lambda0 training engine."""

import json
import os
import shutil
from pathlib import Path

import torch

from .contracts import sha256


def check_finite(values, label, distributed=True):
    tensors = [
        value for value in values if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]
    if not tensors:
        return
    # Adam 的 step 标量可在CPU；先按设备归约，避免每个动量张量触发GPU→CPU同步。
    devices = {value.device for value in tensors}
    reference = next((device for device in devices if device.type == "cuda"), tensors[0].device)
    flags = [
        torch.stack([torch.isfinite(value).all() for value in tensors if value.device == device])
        .all()
        .to(reference)
        for device in devices
    ]
    valid = torch.stack(flags).all().int()
    if distributed and torch.distributed.is_initialized():
        torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
    if not bool(valid):
        raise FloatingPointError(f"Nonfinite {label}; do not resume this state")


def tree_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tree_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tree_tensors(child)


def verify_restored_state(accelerator):
    """摘要一致不等于数值有效；对实际回载后的模型、Adam和EMA再次验收。"""
    for model in accelerator._models:
        check_finite(model.parameters(), "restored model parameters")
    for optimizer in accelerator._optimizers:
        check_finite(tree_tensors(optimizer.state), "restored Adam state")
    for item in accelerator._custom_objects:
        if isinstance(item, torch.nn.Module):
            check_finite(item.parameters(), "restored EMA parameters")


def checkpoint_receipt(path, contract):
    path = Path(path)
    for name in ("config.json", "state.json", "training_config.json", "training_state"):
        if not (path / name).exists():
            raise ValueError(f"Incomplete checkpoint: {path / name}")
    files = {
        file.relative_to(path).as_posix(): dict(bytes=file.stat().st_size, sha256=sha256(file))
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name != "rsci_complete.json"
    }
    temporary = path / ".rsci_complete.tmp"
    temporary.write_text(json.dumps(dict(contract=contract, files=files), indent=2))
    os.replace(temporary, path / "rsci_complete.json")


def verify_checkpoint(path, contract):
    path = Path(path).resolve()
    receipt = json.loads((path / "rsci_complete.json").read_text())
    if receipt["contract"] != contract:
        raise ValueError("RHSCI checkpoint contract/config/source mismatch")
    actual = {
        p.relative_to(path).as_posix()
        for p in path.rglob("*")
        if p.is_file() and p.name != "rsci_complete.json"
    }
    if actual != set(receipt["files"]):
        raise ValueError("Checkpoint inventory mismatch")
    for name, record in receipt["files"].items():
        file = path / name
        if file.stat().st_size != record["bytes"] or sha256(file) != record["sha256"]:
            raise ValueError(f"Checkpoint digest mismatch: {name}")
    return path


class RSCITrainingBackend:
    """Adapt native training operations without replacing module globals."""

    def __init__(self, native, variant, contract):
        self.native = native
        self.variant = variant
        self.contract = contract

    def load_model_and_processor(self, args):
        # 先严格加载官方 Stage-II 原始权重，再构造研究模块，避免 missing-key 静默放宽。
        from .model import Lambda0RSCI

        model, processor = self.native.load_model_and_processor(args, model_class=Lambda0RSCI)
        model.config.rsci_variant = self.variant
        model.config.rsci_contract = self.contract
        if self.variant == "VGQ-AE":
            model.use_proprio = model.config.use_proprio = False
            model.state_conditioning_mode = model.config.state_conditioning_mode = (
                "continuous_action_expert"
            )
            model.install_auxiliary()
            args.use_proprio = False
            args.state_conditioning_mode = "continuous_action_expert"
        return model, processor

    def prepare_inputs(self, batch, processor, device, policy):
        batch = dict(batch)
        history = batch.pop("history", None)
        extra = {
            key: batch.pop(key)
            for key in ("history_valid", "ardy_target", "ardy_valid")
            if key in batch
        }
        result = self.native.prepare_inputs(batch, processor, device, policy)
        if history is not None:
            result["history"] = []
            for index in range(3):
                inputs = processor.prepare_batch(
                    history[:, index : index + 1],
                    torch.ones(history.shape[0], 1, dtype=torch.bool),
                    batch["language_instruction"],
                    state_conditioning_mode="continuous_action_expert",
                )
                result["history"].append({key: value.to(device) for key, value in inputs.items()})
        result.update({key: value.to(device) for key, value in extra.items()})
        return result

    def create_dataloader(self, **kwargs):
        from .data import ResumableLoader, RSCICollator

        if not kwargs["training"]:
            loader = self.native.create_dataloader(**kwargs)
            # 验证仍复用官方数据序列，仅让 collator 保留辅助标签。
            loader.collate_fn = RSCICollator()
            for meta in loader.dataset.metas.values():
                if self.variant == "VGQ-AE":
                    meta["handler"] = "rsci.backbones.lambda0.data:RSCIPsiSimpleHandler"
                elif meta["normalization_contract"] != "quantile_q01_q99_v1":
                    meta["handler"] = "rsci.backbones.lambda0.data:SafePsiSimpleHandler"
            return loader
        from lambda0.data.dataset import InfiniteDataReader

        keys = {
            key: value for key, value in kwargs.items() if key not in ("batch_size", "num_workers")
        }
        reader = InfiniteDataReader(**keys)
        for meta in reader.metas.values():
            if self.variant == "VGQ-AE":
                meta["handler"] = "rsci.backbones.lambda0.data:RSCIPsiSimpleHandler"
            elif meta["normalization_contract"] != "quantile_q01_q99_v1":
                meta["handler"] = "rsci.backbones.lambda0.data:SafePsiSimpleHandler"
        return ResumableLoader(reader, kwargs["batch_size"], kwargs["num_workers"])

    def build_optimizer(self, model, args):
        result = self.native.build_optimizer(model, args)
        from .optimization import optimizer_report

        optimizer_report(model, result)

        def before(opt, positional, keywords):
            check_finite((p.grad for p in model.parameters() if p.grad is not None), "gradients")

        def after(opt, positional, keywords):
            check_finite(model.parameters(), "model parameters")
            check_finite(tree_tensors(opt.state), "Adam state")

        result.register_step_pre_hook(before)
        result.register_step_post_hook(after)
        return result

    def save_training_checkpoint(
        self, accelerator, model, ema_model, processor, output_dir, global_step, args, logger
    ):
        check_finite(model.parameters(), "checkpoint model")
        if ema_model is not None:
            check_finite(ema_model.parameters(), "checkpoint EMA")
        destination = self.native.save_training_checkpoint(
            accelerator, model, ema_model, processor, output_dir, global_step, args, logger
        )
        if accelerator.is_main_process:
            provenance = Path(output_dir) / "rsci_run.json"
            if provenance.is_file():
                shutil.copy2(provenance, destination / "rsci_run.json")
            checkpoint_receipt(destination, self.contract)
            verify_checkpoint(destination, self.contract)
            # 只清理此 run 的已验收滚动断点，永久整万步保留；先验新再清旧。
            for previous in Path(output_dir).glob("ckpt-*"):
                try:
                    step = int(previous.name[5:])
                except ValueError:
                    continue
                if (
                    step < global_step
                    and step % 10000
                    and (previous / "rsci_complete.json").is_file()
                ):
                    shutil.rmtree(previous)
        accelerator.wait_for_everyone()
        return destination

    def load_training_checkpoint(self, accelerator, path, args):
        verify_checkpoint(path, self.contract)
        step = self.native.load_training_checkpoint(accelerator, path, args)
        verify_restored_state(accelerator)
        return step
