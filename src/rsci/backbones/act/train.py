"""SIMPLE ACT配方；统一STATE/NO-STATE/EGQ-AE，独立完整恢复与数值门禁。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

from rsci.paths import add_data_arguments, dataset_paths, with_nccl

from .checkpoints import atomic_json, checkpoint_ok, latest, sha256

CODE = Path(__file__).resolve().parents[4]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(p)
    p.add_argument("--task", required=True)
    p.add_argument(
        "--torch-home",
        type=Path,
        help="External Torch cache containing the ImageNet ResNet18 weights",
    )
    p.add_argument("--nccl-library", type=Path, help="Optional explicitly validated NCCL runtime")
    p.add_argument("--model", choices=("STATE", "NO-STATE", "EGQ-AE"), default="EGQ-AE")
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--run-tag", default="main")
    p.add_argument("--check-data", action="store_true")
    p.add_argument(
        "--check", action="store_true", help="随机初始化全尺寸CPU检查，不下载ImageNet权重、不训练"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--guardian", action="store_true", help=argparse.SUPPRESS)
    return p


def resolve(args):
    if (
        not re.fullmatch(r"\d+(,\d+)*", args.gpus)
        or len(set(args.gpus.split(","))) != len(args.gpus.split(","))
        or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag)
        or min(args.batch_size, args.gradient_accumulation_steps, args.max_steps) < 1
        or args.num_workers < 0
    ):
        raise ValueError("Invalid GPU list, tag, batch, accumulation, steps or workers")
    runtime = Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data")).resolve()
    paths = dataset_paths(args)
    paths["output"] = (
        args.output_dir
        or runtime / "runs/act" / args.task / args.model / f"seed-{args.seed}" / args.run_tag
    )
    args.torch_home = (
        args.torch_home or Path(os.environ.get("TORCH_HOME", str(runtime / "cache/act/torch")))
    ).resolve()
    os.environ["TORCH_HOME"] = str(args.torch_home)
    return paths


def initialization():
    """禁止torchvision隐式联网；由已有本地ImageNet文件初始化，不改变其它模型环境。"""
    import torch

    path = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    if not path.is_file():
        raise FileNotFoundError(
            f"Official ImageNet ResNet18 initialization missing: {path}; no automatic download"
        )
    digest = sha256(path)
    if not digest.startswith("f37072fd"):
        raise ValueError("ImageNet ResNet18 SHA prefix mismatch")
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def check_data(args, paths):
    import numpy as np
    import pyarrow.parquet as pq

    from .data import ACTDataset

    dataset = ACTDataset(paths["dataset"], paths["sidecar"], args.model, args.seed)
    info = json.loads((paths["dataset"] / "meta/info.json").read_text())
    episodes = [
        json.loads(line)
        for line in (paths["dataset"] / "meta/episodes.jsonl").read_text().splitlines()
        if line
    ]
    if info["total_episodes"] != len(episodes) or len(dataset) != sum(
        episode["length"] for episode in episodes
    ):
        raise ValueError("Dataset metadata and frame inventory disagree")
    scanned = 0
    for path in sorted((paths["dataset"] / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["action", "states"]).to_pydict()
        action = np.asarray(table["action"], np.float32)
        state = np.asarray(table["states"], np.float32)
        if (
            action.shape != (len(state), 36)
            or state.shape[1:] != (32,)
            or not (np.isfinite(action).all() and np.isfinite(state).all())
        ):
            raise ValueError(f"Invalid raw action/state: {path}")
        normalized = dataset.processor.field(
            dict(actions=action, states=np.pad(state, ((0, 0), (0, 4))))
        )
        if not (
            np.isfinite(normalized["actions"]).all() and np.isfinite(normalized["states"]).all()
        ):
            raise ValueError(f"Nonfinite native normalization: {path}")
        scanned += len(action)
    if scanned != len(dataset):
        raise ValueError("Parquet inventory does not match dataset length")
    first_length = episodes[0]["length"]
    for index in sorted(
        {
            0,
            min(6, len(dataset) - 1),
            first_length - 1,
            min(first_length, len(dataset) - 1),
            len(dataset) - 1,
        }
    ):
        sample = dataset[index]
        if sample["action"].shape != (16, 36) or sample["observation.images"].shape != (
            1,
            3,
            224,
            224,
        ):
            raise ValueError("Native ACT action/image shape mismatch")
        if index in (first_length - 1, len(dataset) - 1) and sample["action_is_pad"].sum() != 15:
            raise ValueError("Episode tail padding changed")
    print(
        f"ACT_DATA_CHECK_OK task={args.task} model={args.model} samples={len(dataset)} scanned={scanned}; no training started",
        flush=True,
    )
    return dataset


def contract(args, paths):
    from rsci.data.ardy_sidecar import dataset_content_fingerprint

    files = list((CODE / "src/rsci/backbones/act").glob("*.py")) + [
        CODE / name
        for name in (
            "src/act/models/act.py",
            "src/psi/config/transform.py",
            "src/psi/config/augmentation.py",
            "src/psi/data/lerobot/compat.py",
            "src/rsci/data/ardy_sidecar.py",
            "src/psi/utils/utils.py",
            "src/rsci/runtime/hf_resume.py",
            "src/rsci/runtime/artifacts.py",
            "src/rsci/runtime/processes.py",
            "src/rsci/paths.py",
            "scripts/train/act_rsci.sh",
        )
    ]
    return dict(
        schema="act-rsci-v1",
        variant=args.model,
        task=paths["task"],
        dataset=str(paths["dataset"]),
        fingerprint=dataset_content_fingerprint(paths["dataset"]),
        sidecar_manifest=sha256(paths["sidecar"] / "manifest.json")
        if args.model == "EGQ-AE"
        else None,
        initialization=initialization(),
        stats=sha256(paths["dataset"] / "meta/stats_psi0.json"),
        normalization=json.loads((paths["dataset"] / "meta/stats_psi0.json").read_text()),
        batch=args.batch_size,
        accumulation=args.gradient_accumulation_steps,
        world=len(args.gpus.split(",")),
        workers=args.num_workers,
        seed=args.seed,
        steps=args.max_steps,
        optimizer=dict(
            lr=1e-4,
            betas=[0.95, 0.999],
            eps=1e-8,
            weight_decay=1e-6,
            groups="all-parameters-one-group",
            fused=True,
            clip=None,
            warmup=1000,
            scheduler="cosine",
            scheduler_kwargs={},
        ),
        processing="native-act-bounds-nearest-center-color-v1-epoch-index-rng",
        execution=dict(
            nccl_runtime=nccl_runtime(),
            bf16=True,
            tf32=False,
            pin_memory=True,
            non_blocking=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2,
            ddp_gradient_as_bucket_view=True,
            flash_attention_extension=False,
        ),
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "accelerate",
                "lerobot",
                "torchcodec",
                "datasets",
                "av",
            )
        },
        code={str(path.relative_to(CODE)): sha256(path) for path in files},
    )


def digest_contract(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def nccl_runtime():
    """Bind the explicitly preloaded runtime; never replace a shared installation."""
    name = os.environ.get("RSCI_NCCL_LIBRARY")
    if name is None:
        return None
    import ctypes

    path = Path(name).resolve(strict=True)
    version = ctypes.c_int()
    library = ctypes.CDLL(None)
    expected = tuple(map(int, os.environ.get("RSCI_NCCL_VERSION", "2.27.3").split(".")))
    expected = expected[0] * 10000 + expected[1] * 100 + expected[2]
    if library.ncclGetVersion(ctypes.byref(version)) != 0 or version.value != expected:
        raise RuntimeError(f"ACT requires the selected preloaded NCCL runtime {expected}")
    mapped = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libnccl.so" in line
    }
    if mapped != {path}:
        raise RuntimeError(f"Unexpected loaded NCCL libraries: {mapped}")
    return dict(path=str(path), sha256=sha256(path), version=version.value)


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


def training_arguments(args, paths, *, cpu=False):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(paths["output"]),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=1e-4,
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=1e-6,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs={},
        max_grad_norm=0.0,
        bf16=not cpu,
        tf32=False,
        use_cpu=cpu,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=not cpu,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers else None,
        dataloader_drop_last=True,
        accelerator_config={"use_seedable_sampler": False, "non_blocking": not cpu},
        ddp_find_unused_parameters=False,
        save_steps=1000,
        save_total_limit=None,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=1,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )


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


def train(args, paths, value, digest):
    import torch
    from torch.utils.data import Subset

    from psi.utils import seed_everything

    from .data import ACTDataset
    from .model import RSCIACT

    world = len(args.gpus.split(","))
    if int(os.environ.get("WORLD_SIZE", 1)) != world:
        raise ValueError("Launch world size differs from training contract")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    seed_everything(args.seed)
    model = RSCIACT(args.model)
    dataset = ACTDataset(paths["dataset"], paths["sidecar"], args.model, args.seed)
    # 原生ACT验证数据没有传no_aug=True，实际复用训练增强；这里保留这一行为。
    validation = ACTDataset(
        paths["dataset"], paths["sidecar"], args.model, args.seed, training=True
    )
    # 上游验证按每rank最多20批；Subset避免中断prepared loader残留梯度状态。
    validation = Subset(validation, range(min(len(validation), 20 * 16 * world)))
    validation.processor = dataset.processor
    trainer = trainer_type()(
        model=model,
        args=training_arguments(args, paths),
        train_dataset=dataset,
        eval_dataset=validation,
        launch_contract=value,
        callbacks=[completion_callback(paths["output"], world, digest)],
    )
    resume = latest(paths["output"], world, digest)
    print(
        f"ACT_TRAIN resume={resume} global_batch={args.batch_size * world * args.gradient_accumulation_steps}",
        flush=True,
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if not checkpoint_ok(paths["output"] / f"checkpoint-{args.max_steps}", world, digest):
        trainer._save_checkpoint(trainer.model, trial=None)
        completion_callback(paths["output"], world, digest).on_save(
            trainer.args, trainer.state, trainer.control
        )
    if trainer.is_world_process_zero():
        atomic_json(
            paths["output"] / "TRAINING_COMPLETE.json",
            dict(step=trainer.state.global_step, contract=digest),
        )


def main():
    args = parser().parse_args()
    if args.nccl_library:
        os.environ["RSCI_NCCL_LIBRARY"] = str(args.nccl_library.resolve(strict=True))
    if not (args.worker or args.dry_run or args.check or args.check_data) and not os.environ.get(
        "RSCI_NCCL_PRELOADED"
    ):
        command = [sys.executable, "-m", "rsci.backbones.act.train", *sys.argv[1:]]
        wrapped = with_nccl(command, args.nccl_library, expected_version="2.27.3")
        if wrapped != command:
            os.execvpe(wrapped[0], wrapped, {**os.environ, "RSCI_NCCL_PRELOADED": "1"})
    paths = resolve(args)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2), flush=True)
    if args.dry_run:
        world = len(args.gpus.split(","))
        print(
            f"ACT_RECIPE batch={args.batch_size} world={world} accumulation={args.gradient_accumulation_steps} "
            f"global_batch={args.batch_size * world * args.gradient_accumulation_steps} seed={args.seed} "
            f"AdamW=1e-4/.95/.999/wd1e-6 warmup=1000 cosine_steps={args.max_steps} chunk=16"
        )
        return
    if args.check or args.check_data:
        dataset = check_data(args, paths)
        if args.check:
            import torch

            from .model import RSCIACT, official_config

            model = RSCIACT(args.model, official_config(args.model, pretrained=False)).eval()
            batch = {key: value[None] for key, value in dataset[min(6, len(dataset) - 1)].items()}
            with torch.no_grad():
                outputs = model(**batch)
                prediction = model.predict_action(batch)
            if not all(torch.isfinite(x).all() for x in outputs.values()) or prediction.shape != (
                1,
                16,
                36,
            ):
                raise ValueError("Full ACT CPU model check failed")
            print(
                f"ACT_CPU_CHECK_OK parameters={sum(p.numel() for p in model.parameters())}; random init, no training"
            )
        return
    if args.worker:
        value = contract(args, paths)
        digest = digest_contract(value)
        train(args, paths, value, digest)
    else:
        from .launch import guardian

        guardian(args, paths)


if __name__ == "__main__":
    main()
