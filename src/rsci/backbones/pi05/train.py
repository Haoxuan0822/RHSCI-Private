"""π0.5 RHSCI：统一配方、真实预检、完整断点和后台故障恢复。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from rsci.data.state_noise import STATE_NOISE_LOCATION, validate_scale
from rsci.paths import add_data_arguments, dataset_paths, with_nccl

from .checkpoints import atomic_json, checkpoint_ok, latest_checkpoint, sha256


def simple_lr_multiplier(step, max_steps=40000):
    """与 SIMPLE train_pytorch.py 一致，返回学习率相对1e-4的倍率。"""
    import math

    if step < 1000:
        return 1 / 1001 + (1 - 1 / 1001) * step / 1000
    progress = min(1.0, (step - 1000) / max(1, max_steps - 1000))
    return 1e-4 + (1 - 1e-4) * 0.5 * (1 + math.cos(math.pi * progress))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(p)
    p.add_argument(
        "--nccl-library",
        type=Path,
        help="Optional alternate NCCL library; otherwise use the environment runtime",
    )
    p.add_argument("--model", choices=("STATE", "NO-STATE", "VGQ-AE"), default="VGQ-AE")
    p.add_argument(
        "--state-measurement-noise-std-rad",
        type=float,
        nargs="?",
        const=0.02,
        default=0.0,
        help="Training noise before state normalization/tokenization; STATE only",
    )
    p.add_argument("--task", required=True)
    p.add_argument("--vlm", choices=("frozen",), default="frozen")
    p.add_argument(
        "--execution",
        choices=("reference", "history", "query", "both"),
        default="reference",
        help="执行方式验证；默认保留原路径，history合并历史batch，query关闭Query重算",
    )
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    p.add_argument("--batch-size", type=int, default=16, help="每卡微批量")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--run-tag", default="main")
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--seed", type=int, default=292285)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--check", action="store_true", help="真实数据/处理器/权重文件检查，不启动训练")
    p.add_argument(
        "--check-data", action="store_true", help="只检查RGB/动作/正式ARDY，不要求模型权重"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--guardian-fd", type=int, help=argparse.SUPPRESS)
    return p


def resolve(args):
    import re

    validate_scale(args.state_measurement_noise_std_rad, args.model)

    if not re.fullmatch(r"\d+(,\d+)*", args.gpus) or len(set(args.gpus.split(","))) != len(
        args.gpus.split(",")
    ):
        raise ValueError("GPU list must contain unique device indices")
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag)
        or min(args.batch_size, args.gradient_accumulation_steps) < 1
        or args.num_workers < 0
    ):
        raise ValueError("Invalid run-tag/batch/accumulation/workers")
    runtime = Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data"))
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    paths = dataset_paths(args)
    paths.update(
        checkpoint=args.checkpoint or runtime / "cache/checkpoints/openpi/pi05_droid",
        tokenizer=args.tokenizer or runtime / "assets/paligemma_tokenizer.model",
        output=args.output_dir
        or runtime
        / "runs/pi05"
        / args.task
        / args.model
        / args.vlm
        / f"seed-{args.seed}"
        / args.execution
        / args.run_tag,
    )
    if args.output_dir is None and args.state_measurement_noise_std_rad:
        paths["output"] = paths["output"].with_name(
            paths["output"].name + f"-noise-{args.state_measurement_noise_std_rad:g}rad"
        )
    return paths


def preflight(args, paths):
    import numpy as np

    from .data import SimpleRSCIDataset
    from .processing import Processor, collate

    if not args.check_data:
        from safetensors import safe_open

        asset_manifest = paths["checkpoint"] / "RSCI_ASSETS.json"
        if not asset_manifest.is_file():
            raise ValueError(
                "Missing RSCI_ASSETS.json; register local weights and tokenizer with "
                "python -m rsci.backbones.pi05.prepare_assets --checkpoint PATH --tokenizer PATH; "
                "see baselines/pi05/README.md"
            )
        manifest = json.loads(asset_manifest.read_text())
        for name, record in manifest["files"].items():
            path = paths["checkpoint"] / name
            if path.stat().st_size != record["size"]:
                raise ValueError(f"Initialization file size mismatch: {path}")
            if not args.worker and args.guardian_fd is None and sha256(path) != record["sha256"]:
                raise ValueError(f"Initialization SHA256 mismatch: {path}")
        if sha256(paths["tokenizer"]) != manifest["tokenizer_sha256"]:
            raise ValueError("Tokenizer SHA256 mismatch")
        with safe_open(paths["checkpoint"] / "model.safetensors", framework="pt") as stream:
            if "action_in_proj.weight" not in stream.keys():
                raise ValueError("Not an official π0.5 checkpoint")
        processor = Processor(
            paths["dataset"],
            paths["tokenizer"],
            args.model,
            state_measurement_noise_std_rad=args.state_measurement_noise_std_rad,
        )
    else:
        processor = None
    dataset = SimpleRSCIDataset(paths["dataset"], args.model, processor, paths["sidecar"])

    for index in (0, 6, int(dataset.ends[0]) - 1, int(dataset.ends[0]), 6):
        sample = dataset.raw_sample(index)
        if args.model == "VGQ-AE":
            np.testing.assert_array_equal(
                sample["rsci_history_valid"], np.array([-6, -4, -2, 0]) + sample["frame_index"] >= 0
            )
        if (sample["state"] is not None) != (args.model == "STATE"):
            raise ValueError("Unexpected state dependency")
    if processor:
        batch = collate([dataset[0], dataset[6]])
        if batch["actions"].shape != (2, 30, 36) or not np.isfinite(batch["actions"].numpy()).all():
            raise ValueError("Invalid normalized action shape/value")
    print(
        f"PI05_DATA_CHECK_OK task={args.task} model={args.model} samples={len(dataset)}; no training started",
        flush=True,
    )
    return processor, dataset


def contract(args, paths):
    from rsci.data.ardy_sidecar import dataset_content_fingerprint

    from .augmentation import AUGMENTATION_VERSION

    code = Path(__file__).resolve().parents[4]
    sources = list((code / "src/rsci/backbones/pi05").glob("*.py"))
    sources += list((code / "src/openpi/models_pytorch/transformers_replace").rglob("*.py"))
    sources += [
        code / "src/openpi/models_pytorch/gemma_pytorch.py",
        code / "src/rsci/data/ardy_sidecar.py",
        code / "src/openpi/models_pytorch/preprocessing_pytorch.py",
    ]
    sources += [
        code / "src/rsci/data/loader_cleanup.py",
        code / "scripts/with_nccl_runtime.py",
        code / "scripts/train/pi05.sh",
        code / "src/rsci/paths.py",
    ]
    sources += sorted((code / "src/rsci/runtime").glob("*.py"))
    sources += sorted((code / "src/rsci/data").glob("*.py"))
    sources = sorted(set(sources))
    identity = dict(
        schema=2,
        model=args.model,
        vlm=args.vlm,
        task=args.task,
        execution=args.execution,
        recipe="SIMPLE-pi05-default-v4",
        optimizer=dict(betas=[0.9, 0.95], eps=1e-8, weight_decay=1e-10, clip_norm=1),
        scheduler="SIMPLE-cosine-start-peak/1001-end-1e-8",
        inference_compile="max-autotune",
        batch=args.batch_size,
        accumulation=args.gradient_accumulation_steps,
        world_size=len(args.gpus.split(",")),
        seed=args.seed,
        workers=args.num_workers,
        steps=args.max_steps,
        lr=1e-4,
        warmup=1000,
        horizon=30,
        ardy_weight=0.1,
        preprocessing="RGB-bilinear-pad224-quantile-zero-span-clipped-v2",
        state_measurement_noise_std_rad=args.state_measurement_noise_std_rad,
        state_noise_location=STATE_NOISE_LOCATION,
        training_augmentation=AUGMENTATION_VERSION,
        dataset=dataset_content_fingerprint(paths["dataset"]),
        initialization=str(paths["checkpoint"].resolve()),
        initialization_stat={
            key: getattr((paths["checkpoint"] / "model.safetensors").stat(), key)
            for key in ("st_size", "st_mtime_ns", "st_ctime_ns")
        },
        asset_manifest=sha256(paths["checkpoint"] / "RSCI_ASSETS.json"),
        tokenizer=sha256(paths["tokenizer"]),
        code={str(p.relative_to(code)): sha256(p) for p in sources},
        dependencies={
            key: importlib.metadata.version(key)
            for key in ("torch", "transformers", "accelerate", "torchcodec", "sentencepiece")
        },
    )
    if args.model == "VGQ-AE":
        identity["sidecar"] = sha256(paths["sidecar"] / "manifest.json")
    return identity, hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def validate_run(args, paths, identity, digest, *, allow_started=False):
    """Reject incompatible runs and unsafe resumes before starting a worker."""
    from rsci.runtime.numerics import reject_numerical_failure

    reject_numerical_failure(paths["output"])
    record = paths["output"] / "launch_contract.json"
    if record.exists() and json.loads(record.read_text()) != identity:
        raise ValueError(
            f"{args.task}: existing run has different configuration; use a new run-tag"
        )
    resume = args.resume or latest_checkpoint(paths["output"], len(args.gpus.split(",")), digest)
    if resume is not None:
        if resume.resolve().parent != paths["output"].resolve() or not checkpoint_ok(
            resume, len(args.gpus.split(",")), digest, verify_hash=True
        ):
            raise ValueError(f"{args.task}: incomplete, corrupt, or mismatched resume checkpoint")
    elif not allow_started and (paths["output"] / "training_started.json").exists():
        raise RuntimeError(
            f"{args.task}: no complete checkpoint; refusing restart from zero; use a new run-tag for an intentional fresh run"
        )
    return resume


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


def training_arguments(args, paths):
    """生产与CPU配方核验共用，优化配置不再套用Psi0参数。"""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(paths["output"]),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=1e-4,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        weight_decay=1e-10,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1,
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        save_steps=1000,
        save_total_limit=None,
        save_safetensors=True,
        logging_steps=10,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers else None,
        accelerator_config={"non_blocking": True},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )


def train(args, paths, processor, dataset, digest):
    import torch
    from transformers import set_seed

    from .model import Pi05RSCI
    from .processing import collate

    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)
    model = Pi05RSCI(
        args.model,
        execution=args.execution,
        state_measurement_noise_std_rad=args.state_measurement_noise_std_rad,
    )
    model.load_initial(paths["checkpoint"])
    world = len(args.gpus.split(","))
    identity = contract(args, paths)[0]
    resume = validate_run(args, paths, identity, digest, allow_started=True)
    training = training_arguments(args, paths)
    trainer = trainer_class()(
        model=model,
        args=training,
        train_dataset=dataset,
        data_collator=collate,
        callbacks=[complete_callback(paths["output"], world, digest)],
    )
    print(
        f"PI05_SPEED BF16/FP32-trainable; SDPA; layer-checkpoint; frozen-VLM-KV; fused-AdamW; "
        f"TF32; pin/nonblocking/persistent; DDP bucket views; global_batch={args.batch_size * world * args.gradient_accumulation_steps}",
        flush=True,
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if trainer.is_world_process_zero() and not checkpoint_ok(
        paths["output"] / f"checkpoint-{args.max_steps}", world, digest
    ):
        raise RuntimeError("Missing complete final checkpoint")


def guardian(args, paths, digest):
    import signal
    import subprocess
    import time

    world = len(args.gpus.split(","))
    child = None
    previous_step = -1

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for attempt in range(4):
        from rsci.runtime.numerics import reject_numerical_failure

        reject_numerical_failure(paths["output"])
        if contract(args, paths)[1] != digest:
            raise ValueError("Source/data/dependencies changed; refusing automatic resume")
        resume = (
            args.resume
            if attempt == 0 and args.resume
            else latest_checkpoint(paths["output"], world, digest)
        )
        if resume:
            if resume.resolve().parent != paths["output"].resolve() or not checkpoint_ok(
                resume, world, digest, verify_hash=True
            ):
                raise ValueError("Resume must be complete and from this experiment")
            step = int(resume.name.split("-")[-1])
            if step >= args.max_steps:
                return
            if attempt and step <= previous_step:
                raise RuntimeError("No new complete checkpoint; refusing repeated replay")
            previous_step = step
        elif attempt or (paths["output"] / "training_started.json").exists():
            raise RuntimeError(
                "No complete checkpoint; refusing automatic restart from zero; inspect the earlier worker error"
            )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world}",
            f"--master_port={32000 + int(digest[:4], 16) % 1000}",
            "--module",
            "rsci.backbones.pi05.train",
            *[value for value in sys.argv[1:] if not value.startswith("--guardian-fd")],
            "--worker",
        ]
        # 不把父命令的旧 resume 带入下一次恢复。
        if "--resume" in command:
            index = command.index("--resume")
            del command[index : index + 2]
        if resume:
            command += ["--resume", str(resume)]
        command = with_nccl(command, args.nccl_library)
        atomic_json(
            paths["output"] / "training_started.json",
            dict(attempt=attempt + 1, command=command, time=time.time()),
        )
        print(f"PI05 attempt={attempt + 1} resume={resume}", flush=True)
        child = subprocess.Popen(
            command,
            start_new_session=True,
            pass_fds=(args.guardian_fd,),
            env={
                key: value
                for key, value in os.environ.items()
                if key not in ("NCCL_PROTO", "NCCL_ALGO")
            },
        )
        code_result = child.wait()
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if code_result == 0 and checkpoint_ok(
            paths["output"] / f"checkpoint-{args.max_steps}", world, digest
        ):
            return
    raise RuntimeError("Recovery attempts exhausted")


def main():
    import fcntl
    import subprocess
    import time

    args = parser().parse_args()
    paths = resolve(args)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2), flush=True)
    if args.dry_run:
        return
    if args.check_data:
        preflight(args, paths)
        return
    from .prepare_assets import transformers_overlay

    transformers_overlay()
    processor, dataset = preflight(args, paths)
    identity, digest = contract(args, paths)
    # 权重/数据检查之外，提前核对通信运行库及已有 run；不创建训练记录。
    if not args.worker and args.guardian_fd is None:
        if args.nccl_library or os.environ.get("RSCI_NCCL_LIBRARY"):
            subprocess.run(with_nccl([sys.executable, "-c", "pass"], args.nccl_library), check=True)
        validate_run(args, paths, identity, digest)
    if args.check:
        print(
            f"PI05_RSCI_CHECK_OK task={args.task} model={args.model}; data/assets/NCCL/run checked; no training started",
            flush=True,
        )
        return
    if args.worker or args.guardian_fd is not None:
        if json.loads((paths["output"] / "launch_contract.json").read_text()) != identity:
            raise ValueError("Launch contract mismatch")
        if args.worker:
            train(args, paths, processor, dataset, digest)
        else:
            guardian(args, paths, digest)
        return
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "guardian.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = output / "launch_contract.json"
        if record.exists() and json.loads(record.read_text()) != identity:
            raise ValueError("Existing run has different configuration; use a new run-tag")
        atomic_json(record, identity)
        log_path = output / "guardian.log"
        with log_path.open("ab", buffering=0) as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "rsci.backbones.pi05.train",
                    *sys.argv[1:],
                    f"--guardian-fd={lock.fileno()}",
                ],
                start_new_session=True,
                stdout=log,
                stderr=log,
                pass_fds=(lock.fileno(),),
            )
        print(f"PI05 guardian PID={child.pid}; log={log_path}; Ctrl+C仅退出日志查看", flush=True)
        with log_path.open() as stream:
            try:
                while child.poll() is None:
                    text = stream.read()
                    if text:
                        print(text, end="", flush=True)
                    time.sleep(0.3)
                print(stream.read(), end="", flush=True)
            except KeyboardInterrupt:
                # Closing the log viewer leaves the guardian running.
                raise SystemExit(130)
        if child.returncode:
            raise SystemExit(child.returncode)
        if not checkpoint_ok(
            output / f"checkpoint-{args.max_steps}", len(args.gpus.split(",")), digest
        ):
            raise RuntimeError("Guardian exited without complete 40k checkpoint")


if __name__ == "__main__":
    main()
