"""SIMPLE EgoVLA VGQ-AE 训练入口：官方模型/配方，RHSCI 数据及安全恢复。"""

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
    p.add_argument("--model", choices=("VGQ-AE", "STATE"), default="VGQ-AE")
    p.add_argument(
        "--vlm-training",
        choices=("full", "frozen"),
        default="full",
        help="默认按SIMPLE全量微调视觉/投影/语言；frozen仅作显式消融",
    )
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--preset", choices=("simple-115epoch",))
    length = p.add_mutually_exclusive_group()
    length.add_argument("--max-steps", type=int)
    length.add_argument("--num-train-epochs", type=int)
    p.add_argument("--seed", type=int, default=292285)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--run-tag", default="main")
    p.add_argument(
        "--check-data", action="store_true", help="只核对正式数据，不需要 EgoVLA 权重/环境"
    )
    p.add_argument("--check", action="store_true", help="完整CPU初始化和处理器预检，不训练")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--register-assets", action="store_true", help="为已下载的本地基础权重生成SHA清单，不下载"
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--guardian", action="store_true", help=argparse.SUPPRESS)
    return p


def resolve(args):
    if args.preset == "simple-115epoch":
        if args.max_steps is not None or args.num_train_epochs not in (None, 115):
            raise ValueError("simple-115epoch requires 115 epochs, without --max-steps")
        if (
            args.batch_size != 16
            or args.gradient_accumulation_steps != 4
            or len(args.gpus.split(",")) != 8
            or args.vlm_training != "full"
        ):
            raise ValueError("simple-115epoch requires full VLM, 8 GPUs, batch 16, accumulation 4")
        args.num_train_epochs = 115
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    if args.max_steps is None:
        args.max_steps = 40000
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
    paths.update(
        checkpoint=(
            args.checkpoint or runtime / "cache/checkpoints/EgoVLA/SIMPLE-pretrain"
        ).resolve(),
        output=args.output_dir
        or runtime
        / "runs/egovla"
        / args.task
        / args.model
        / args.vlm_training
        / f"seed-{args.seed}"
        / args.run_tag,
    )
    if args.num_train_epochs is not None:
        if args.num_train_epochs < 1:
            raise ValueError("Epochs must be positive")
        import math

        episodes = [
            json.loads(line)
            for line in (paths["dataset"] / "meta/episodes.jsonl").read_text().splitlines()
            if line
        ]
        samples = sum(max(0, e["length"] - 29) for e in episodes)
        batches = math.ceil(samples / (args.batch_size * len(args.gpus.split(","))))
        # TF4.53.2的epoch预算以floor(batch/accum)计，末尾不足仍执行更新。
        # 要求整除，避免名义100epoch却提前停止；不要静默近似。
        if batches % args.gradient_accumulation_steps:
            raise ValueError(
                "Epoch recipe has a partial accumulation; use explicit steps or an exactly dividing accumulation"
            )
        args.max_steps = args.num_train_epochs * (batches // args.gradient_accumulation_steps)
    return paths


def asset_files(root):
    """冻结基础目录的完整文件集合；不能漏过加载优先级更高的新增文件。"""
    result = {}
    for path in root.rglob("*"):
        if path == root / "RSCI_ASSETS.json":
            continue
        if path.is_symlink() and (path.is_dir() or not path.is_file()):
            raise ValueError(f"Unsupported asset directory/broken symlink: {path}")
        if path.is_file():
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Asset escapes frozen base directory: {path}")
            result[path.relative_to(root).as_posix()] = path
    return result


def assets(root, register=False, verify=True):
    root = Path(root).resolve()
    manifest_path = root / "RSCI_ASSETS.json"
    if register:
        if manifest_path.exists():
            raise FileExistsError("Existing asset manifest is immutable; verify it instead")
        for name in (
            "config.json",
            "llm/config.json",
            "vision_tower/config.json",
            "mm_projector/config.json",
        ):
            if not (root / name).is_file():
                raise FileNotFoundError(f"Missing official component: {root / name}")
        names = asset_files(root)
        if not any(path.suffix in (".safetensors", ".bin") for path in names.values()):
            raise ValueError("No pretrained tensor files")
        atomic_json(
            manifest_path,
            dict(
                schema="egovla-base-local-v1",
                files={
                    name: dict(size=path.stat().st_size, sha256=sha256(path))
                    for name, path in names.items()
                },
            ),
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"EgoVLA assets not prepared: {manifest_path}; no automatic download"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "egovla-base-local-v1":
        raise ValueError("Unsupported EgoVLA asset manifest")
    actual = asset_files(root)
    recorded = manifest.get("files")
    if not isinstance(recorded, dict) or not recorded or set(actual) != set(recorded):
        added = (
            sorted(set(actual) - set(recorded or {}))
            if isinstance(recorded, dict)
            else sorted(actual)
        )
        removed = sorted(set(recorded or {}) - set(actual)) if isinstance(recorded, dict) else []
        raise ValueError(f"Asset file inventory changed: added={added}, removed={removed}")
    for name, record in manifest["files"].items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root)
            or path.stat().st_size != record["size"]
            or (verify and sha256(path) != record["sha256"])
        ):
            raise ValueError(f"Asset size/SHA mismatch: {name}")
    return manifest


def check_data(paths, variant="VGQ-AE"):
    import numpy as np

    from .data import OFFSETS, SimpleDataset

    dataset = SimpleDataset(paths["dataset"], paths["sidecar"], variant=variant)

    for index in (0, 6, int(dataset.ends[0]) - 1, int(dataset.ends[0])):
        sample = dataset.raw_sample(index)
        if variant == "VGQ-AE":
            np.testing.assert_array_equal(
                sample["history_valid"], OFFSETS + sample["frame_index"] >= 0
            )
        if sample["actions"].shape != (30, 36) or any(
            "depth" in key or (variant == "VGQ-AE" and "state" in key) for key in sample
        ):
            raise ValueError("Invalid action window or unexpected state/depth")
    print(
        f"EGOVLA_DATA_CHECK_OK task={paths['task']} samples={len(dataset)}; no training started",
        flush=True,
    )
    return dataset


def contract(args, paths, manifest):
    from rsci.data.ardy_sidecar import dataset_content_fingerprint

    sources = sorted((CODE / "src/rsci/backbones/egovla").glob("*.py"))
    sources += sorted((CODE / "src/egovla").rglob("*.py"))
    sources += [
        CODE / "scripts/train/egovla.sh",
        CODE / "src/rsci/paths.py",
        CODE / "src/egovla/pyproject.toml",
        CODE / "src/egovla/uv.lock",
    ]
    sources += sorted((CODE / "src/rsci/runtime").glob("*.py"))
    sources += sorted((CODE / "src/rsci/data").glob("*.py"))
    sources = sorted(set(sources))
    return dict(
        schema="egovla-vgq-ae-v1",
        task=paths["task"],
        model=args.model,
        vlm=args.vlm_training,
        dataset=str(paths["dataset"]),
        dataset_fingerprint=dataset_content_fingerprint(paths["dataset"]),
        sidecar_manifest=sha256(paths["sidecar"] / "manifest.json")
        if args.model == "VGQ-AE"
        else None,
        initialization=str(paths["checkpoint"]),
        assets=manifest,
        batch=args.batch_size,
        accumulation=args.gradient_accumulation_steps,
        world=len(args.gpus.split(",")),
        workers=args.num_workers,
        seed=args.seed,
        steps=args.max_steps,
        epochs=args.num_train_epochs,
        attention=dict(language="sdpa", vision="sdpa"),
        gradient_checkpointing=dict(
            enabled=args.vlm_training == "full", use_reentrant=False, scope="language-and-vision"
        ),
        execution=dict(
            optimizer="adamw_torch_fused",
            pin_memory=True,
            non_blocking=True,
            gradient_as_bucket_view=True,
            cuda_allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers else None,
            nccl_shm_disable=os.environ.get("NCCL_SHM_DISABLE", "0"),
            nccl_library=os.environ.get("RSCI_NCCL_LIBRARY"),
            nccl_library_sha256=sha256(Path(os.environ["RSCI_NCCL_LIBRARY"]))
            if os.environ.get("RSCI_NCCL_LIBRARY")
            else None,
        ),
        sampler="accelerate-seedable-epoch-shuffle-v2-preserve-skip-epoch",
        optimizer=dict(
            lr=2e-5,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0.0,
            clip=1.0,
            scheduler="constant",
            warmup_ratio=0.03,
            warmup_effective=False,
        ),
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "accelerate", "torchcodec")
        },
        code={str(path.relative_to(CODE)): sha256(path) for path in sources},
    )


def digest_contract(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


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


def trainer_type():
    import torch
    from transformers import Trainer

    from rsci.runtime.hf_resume import epoch_safe_training_loop

    from .checkpoints import load_trainables

    class RSCITrainer(Trainer):
        _inner_training_loop = epoch_safe_training_loop(Trainer)

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

    return RSCITrainer


def training_arguments(args, paths, cpu=False):
    from transformers import TrainingArguments

    # constant 不是 constant_with_warmup；原 SIMPLE 脚本虽然写了3%，实际不执行预热。
    return TrainingArguments(
        output_dir=str(paths["output"]),
        max_steps=-1 if args.num_train_epochs else (args.max_steps or 40000),
        num_train_epochs=args.num_train_epochs or 3,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        max_grad_norm=1.0,
        # 迁移Psi0执行加速，保留EgoVLA自身AdamW超参数；CPU验收使用非fused实现。
        optim="adamw_torch" if cpu else "adamw_torch_fused",
        bf16=not cpu,
        tf32=not cpu,
        use_cpu=cpu,
        gradient_checkpointing=args.vlm_training == "full",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=not cpu,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        remove_unused_columns=False,
        # 全量VLM时词表head不参与next_token_loss=0，与官方一样需要unused参数检测。
        ddp_find_unused_parameters=args.vlm_training == "full",
        save_steps=1000,
        save_total_limit=None,
        accelerator_config={"use_seedable_sampler": True, "non_blocking": not cpu},
        save_safetensors=True,
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
                target = output / f"checkpoint-{state.global_step}"
                try:
                    publish(target, world, digest)
                    prune(output, target, world, digest)
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

    if os.environ.get("RSCI_NCCL_LIBRARY"):
        # Torch的version查询可能反映编译版本，必须核对实际全局动态符号。
        import ctypes

        actual_nccl = ctypes.c_int()
        result = ctypes.CDLL(None).ncclGetVersion(ctypes.byref(actual_nccl))
        major, minor, patch = map(int, os.environ.get("RSCI_NCCL_VERSION", "2.27.3").split("."))
        expected = major * 10000 + minor * 100 + patch
        if result != 0 or actual_nccl.value != expected:
            raise RuntimeError(f"Expected NCCL {expected}; got {actual_nccl.value}")
        print(f"EGOVLA_NCCL_RUNTIME version={actual_nccl.value}", flush=True)
    from transformers import set_seed

    from .data import SimpleDataset
    from .native import load_native
    from .processing import Collator

    world = len(args.gpus.split(","))
    if int(os.environ.get("WORLD_SIZE", 1)) != world:
        raise ValueError("Launch world size differs from contract")
    # 先绑定rank，避免所有rank初始化/allocator意外占用GPU0。
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    set_seed(args.seed)
    model, processor = load_native(
        paths["checkpoint"], vlm_training=args.vlm_training, variant=args.model
    )
    dataset = SimpleDataset(paths["dataset"], paths["sidecar"], processor, variant=args.model)
    trainer = trainer_type()(
        model=model,
        args=training_arguments(args, paths),
        train_dataset=dataset,
        data_collator=Collator(processor.tokenizer.pad_token_id),
        launch_contract=value,
        callbacks=[completion_callback(paths["output"], world, digest)],
    )
    if args.num_train_epochs:
        actual_batches = len(trainer.get_train_dataloader())
        if (
            actual_batches * args.num_train_epochs
            != args.max_steps * args.gradient_accumulation_steps
        ):
            raise ValueError("Actual distributed loader differs from the epoch step budget")
    resume = latest(paths["output"], world, digest)
    print(
        f"EGOVLA_TRAIN resume={resume} global_batch={args.batch_size * args.gradient_accumulation_steps * world}",
        flush=True,
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    # 非1000整数步的显式短训练同样保存完整HF状态和RNG；不把裸权重当完成。
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


def guardian(args, paths, value, digest):
    import fcntl
    import subprocess

    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "guardian.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = output / "launch_contract.json"
        if previous.exists() and json.loads(previous.read_text()) != value:
            raise ValueError("Output belongs to a different training contract; use a new run-tag")
        atomic_json(previous, value)
        atomic_json(output / "guardian.json", dict(pid=os.getpid(), contract=digest))
        world = len(args.gpus.split(","))
        attempts_at_step = {}
        for attempt in range(1, 4):
            if (output / "NUMERICAL_FAILURE.json").exists():
                raise RuntimeError("Numerical failure; no automatic restart")
            resume = latest(output, world, digest)
            if resume is not None and not checkpoint_ok(resume, world, digest, verify=True):
                raise RuntimeError("Resume checkpoint failed complete SHA/numerical validation")
            if checkpoint_ok(output / f"checkpoint-{args.max_steps}", world, digest, verify=True):
                print("EGOVLA_TRAINING_COMPLETE", flush=True)
                return
            if resume is None and (attempt > 1 or (output / "STARTED.json").exists()):
                raise RuntimeError("No complete checkpoint; refusing automatic restart from zero")
            step = int(resume.name.split("-")[-1]) if resume else 0
            attempts_at_step[step] = attempts_at_step.get(step, 0) + 1
            if attempts_at_step[step] > 2:
                raise RuntimeError("Repeated failure without checkpoint progress")
            atomic_json(output / "STARTED.json", dict(attempt=attempt, resume=step))
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={world}",
                "-m",
                "rsci.backbones.egovla.train",
                *[arg for arg in sys.argv[1:] if arg != "--guardian"],
                "--worker",
            ]
            print(f"EGOVLA attempt={attempt} resume={resume}", flush=True)
            process = subprocess.Popen(
                with_nccl(command, expected_version="2.27.3"), start_new_session=True
            )

            from rsci.runtime.processes import wait_worker

            result = wait_worker(process)
            if result == 0 and checkpoint_ok(
                output / f"checkpoint-{args.max_steps}", world, digest, verify=True
            ):
                return
        raise RuntimeError("Training retries exhausted; inspect guardian.log")


def main():
    args = parser().parse_args()
    paths = resolve(args)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2), flush=True)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "preset": args.preset,
                    "max_steps": args.max_steps,
                    "num_train_epochs": args.num_train_epochs,
                    "global_batch": args.batch_size
                    * args.gradient_accumulation_steps
                    * len(args.gpus.split(",")),
                    "cuda_allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                    "nccl_shm_disable": os.environ.get("NCCL_SHM_DISABLE", "0"),
                },
                indent=2,
            )
        )
        return
    if args.check_data:
        check_data(paths, args.model)
        return
    manifest = assets(paths["checkpoint"], register=args.register_assets, verify=not args.worker)
    if args.register_assets:
        print("EGOVLA_ASSETS_REGISTERED; no training started")
        return
    if not args.worker:
        check_data(paths, args.model)
    value = contract(args, paths, manifest)
    digest = digest_contract(value)
    if args.check:
        from .native import load_native
        from .processing import Collator

        model, processor = load_native(
            paths["checkpoint"], vlm_training=args.vlm_training, variant=args.model
        )
        if args.vlm_training == "full":
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if not (
                model.backbone.get_llm().model.is_gradient_checkpointing
                and model.backbone.get_vision_tower().vision_tower.is_gradient_checkpointing
            ):
                raise RuntimeError("Native language/vision checkpointing did not activate")
        dataset = check_data(paths, args.model)
        dataset.processor = processor
        batch = Collator(processor.tokenizer.pad_token_id)([dataset[0], dataset[6]])
        if batch["actions"].shape != (2, 30, 36):
            raise ValueError("Invalid model batch")
        print(
            "EGOVLA_PARAMETERS "
            + json.dumps(
                dict(
                    vlm_training=args.vlm_training,
                    total=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    vlm_trainable=sum(
                        p.numel()
                        for module in (
                            model.backbone.get_llm(),
                            model.backbone.get_vision_tower(),
                            model.backbone.get_mm_projector(),
                        )
                        for p in module.parameters()
                        if p.requires_grad
                    ),
                )
            ),
            flush=True,
        )
        print("EGOVLA_MODEL_PROCESSOR_CHECK_OK; no training started")
    elif args.worker:
        expected = json.loads((paths["output"] / "launch_contract.json").read_text())
        if expected != value:
            raise ValueError("Worker source/data/recipe contract changed")
        train(args, paths, value, digest)
    elif args.guardian:
        guardian(args, paths, value, digest)
    else:
        raise RuntimeError("Use scripts/train/egovla.sh for the detached guardian and live log")


if __name__ == "__main__":
    main()
