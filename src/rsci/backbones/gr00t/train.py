"""GR00T RHSCI训练worker：官方模型/处理器，map数据集，完整HF断点。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from rsci.paths import add_data_arguments, dataset_paths, with_nccl, worker_arguments


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(p)
    p.add_argument(
        "--nccl-library",
        type=Path,
        help="Optional alternate NCCL library; otherwise use the environment runtime",
    )
    p.add_argument("--model", choices=("STATE", "NO-STATE", "VGQ-AE"), default="VGQ-AE")
    p.add_argument("--task", required=True)
    p.add_argument(
        "--vlm",
        choices=("top4", "frozen"),
        default="top4",
        help="SIMPLE默认训练顶部4层；frozen为显式消融",
    )
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    p.add_argument(
        "--batch-size", type=int, default=None, help="默认按卡数和累计自动得到全局batch24"
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="每次优化更新累计的微批次数；改变后使用新run-tag",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--run-tag", default="main")
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=292285)
    p.add_argument("--checkpoint", type=Path, help="官方初始化目录；默认数据盘GR00T-N1.6-3B")
    p.add_argument("--resume", type=Path, help="本实验完整checkpoint；默认自动寻找")
    p.add_argument("--check", action="store_true", help="真实数据/CPU处理器预检，不加载完整模型")
    p.add_argument(
        "--check-data",
        action="store_true",
        help="仅真实RGB/动作/正式ARDY及episode身份检查，无需Eagle依赖",
    )
    p.add_argument("--dry-run", action="store_true", help="仅打印参数和路径")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--guardian-fd", type=int, help=argparse.SUPPRESS)
    return p


def resolve(args):
    import re

    if not re.fullmatch(r"\d+(,\d+)*", args.gpus) or len(set(args.gpus.split(","))) != len(
        args.gpus.split(",")
    ):
        raise ValueError("--gpus must list unique comma-separated device indices")
    if args.batch_size is None:
        divisor = len(args.gpus.split(",")) * args.gradient_accumulation_steps
        if divisor < 1 or 24 % divisor:
            raise ValueError("Global batch 24 must be divisible by world_size × accumulation")
        args.batch_size = 24 // divisor
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag)
        or args.batch_size < 1
        or args.num_workers < 0
        or args.gradient_accumulation_steps < 1
    ):
        raise ValueError("Invalid run-tag/batch/workers")
    runtime = Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data"))
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    paths = dataset_paths(args)
    paths.update(
        checkpoint=args.checkpoint or runtime / "cache/checkpoints/GR00T-N1.6-3B",
        output=args.output_dir
        or runtime
        / "runs/gr00t"
        / args.task
        / args.model
        / args.vlm
        / f"seed-{args.seed}"
        / args.run_tag,
    )
    return paths


def optimization_kwargs():
    """从原生训练配置和当前锁定HF版本取值，不复用Psi0的优化参数。"""
    from gr00t.configs.training.training_config import TrainingConfig
    from transformers import TrainingArguments

    native = TrainingConfig()
    values = {
        key: getattr(native, key)
        for key in (
            "learning_rate",
            "lr_scheduler_type",
            "weight_decay",
            "warmup_ratio",
            "warmup_steps",
            "max_grad_norm",
            "optim",
        )
    }
    # 官方experiment没有覆盖这些值，因此沿用同版本HF TrainingArguments默认值。
    values.update(
        {
            key: TrainingArguments.__dataclass_fields__[key].default
            for key in ("adam_beta1", "adam_beta2", "adam_epsilon")
        }
    )
    return values


def build_processor(args, paths):
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

    from .data import modality, statistics

    config = json.loads((paths["checkpoint"] / "config.json").read_text())
    settings = json.loads((paths["checkpoint"] / "processor_config.json").read_text())[
        "processor_kwargs"
    ]
    settings.update(
        modality_configs=modality(args.model),
        statistics=statistics(paths["dataset"], args.model),
        rsci_variant=args.model,
        use_relative_action=False,
        max_state_dim=config["max_state_dim"],
        max_action_dim=config["max_action_dim"],
        max_action_horizon=config["action_horizon"],
        use_albumentations=True,
        color_jitter_params=dict(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
    )
    return Gr00tN1d6Processor(
        **settings,
        transformers_loading_kwargs={"trust_remote_code": True, "local_files_only": True},
    )


def preflight(args, paths):
    from .data import SimpleRSCIDataset

    checkpoint = paths["checkpoint"]
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if not (checkpoint / shard).is_file():
            raise FileNotFoundError(checkpoint / shard)
    processor = build_processor(args, paths)
    dataset = SimpleRSCIDataset(paths["dataset"], args.model, processor, paths["sidecar"])

    # 起始无效历史及完整历史各一个真实样本，包含官方Eagle分词/图像处理。
    batch = processor.collator([dataset[0], dataset[6]])["inputs"]
    if batch["action"].shape != (
        2,
        config_horizon := processor.max_action_horizon,
        processor.max_action_dim,
    ):
        raise ValueError("Invalid padded action shape")
    if args.model == "VGQ-AE":
        if (
            len(batch["rsci_history"]) != 3
            or batch["rsci_history_valid"][0].all()
            or not batch["rsci_history_valid"][1].all()
        ):
            raise ValueError("Invalid chronological history mask")
    print(
        f"GR00T_RSCI_CHECK_OK task={args.task} model={args.model} samples={len(dataset)} padded_horizon={config_horizon}; no training started",
        flush=True,
    )
    return processor, dataset


def contract(args, paths):
    from rsci.data.ardy_sidecar import dataset_content_fingerprint

    from .checkpoints import sha256

    code = Path(__file__).resolve().parents[4]
    sources = list((code / "src/rsci/backbones/gr00t").glob("*.py"))
    sources += list((code / "src/gr00t/gr00t/model/gr00t_n1d6").glob("*.py"))
    sources += list(
        (code / "src/gr00t/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2").glob("*.py")
    )
    sources += [
        code / "src/gr00t/gr00t/model/modules/eagle_backbone.py",
        code / "src/gr00t/gr00t/model/modules/dit.py",
        code / "src/gr00t/gr00t/configs/model/gr00t_n1d6.py",
        code / "src/rsci/data/ardy_sidecar.py",
        code / "baselines/gr00t/requirements.txt",
        code / "src/gr00t/gr00t/configs/training/training_config.py",
        code / "scripts/train/gr00t.sh",
        code / "src/rsci/paths.py",
    ]
    sources += sorted((code / "src/rsci/runtime").glob("*.py"))
    sources += sorted((code / "src/rsci/data").glob("*.py"))
    sources = sorted(set(sources))
    identity = dict(
        schema=2,
        recipe="SIMPLE-GR00T-zero1-recompute-v1",
        model=args.model,
        vlm=args.vlm,
        task=args.task,
        batch=args.batch_size,
        memory_optimization="zero1-fp32-adam-and-nonreentrant-dit-v1",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_size=len(args.gpus.split(",")),
        seed=args.seed,
        workers=args.num_workers,
        max_steps=args.max_steps,
        optimization=optimization_kwargs(),
        ardy_weight=0.1,
        tune_top_llm_layers=4 if args.vlm == "top4" else 0,
        color_jitter=dict(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        valid_action_horizon=16,
        history_offsets=[-6, -4, -2, 0],
        reader_layout="8image+8text-per-frame",
        dataset_fingerprint=dataset_content_fingerprint(paths["dataset"]),
        initialization=str(paths["checkpoint"].resolve()),
        initialization_config=sha256(paths["checkpoint"] / "config.json"),
        initialization_index=sha256(paths["checkpoint"] / "model.safetensors.index.json"),
        sources={str(path.relative_to(code)): sha256(path) for path in sources},
        dependencies={
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "diffusers",
                "torchcodec",
                "albumentations",
                "accelerate",
            )
        },
    )
    if args.model == "VGQ-AE":
        identity["sidecar_manifest"] = sha256(paths["sidecar"] / "manifest.json")
    return identity, hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def check_training_environment():
    import importlib.util

    expected = {
        "torch": "2.7.0",
        "transformers": "4.51.3",
        "diffusers": "0.35.1",
        "torchcodec": "0.4.0",
    }
    failures = []
    for name, version in expected.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = "missing"
        if installed.split("+")[0] != version:
            failures.append(f"{name}=={version} (found {installed})")
    if importlib.util.find_spec("flash_attn") is None:
        failures.append("flash-attn==2.7.4.post1")
    if failures:
        raise RuntimeError(
            "GR00T training environment is not ready: "
            + ", ".join(failures)
            + "; see baselines/gr00t/README.md"
        )


def train(args, paths, processor, dataset, digest, *, validation_steps=0):
    import torch
    import torch.distributed as dist
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
    from transformers import Trainer, TrainerCallback, set_seed

    from rsci.runtime.numerics import reject_numerical_failure, require_finite

    from .checkpoints import checkpoint_ok, complete_checkpoint, latest_checkpoint, prune_rolling
    from .data import modality
    from .memory import ShardedOptimizerTrainerMixin, enable_dit_recomputation

    reject_numerical_failure(paths["output"])

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)
    config = json.loads((paths["checkpoint"] / "config.json").read_text())
    config.update(
        rsci_variant=args.model,
        rsci_ardy_weight=0.1,
        tune_llm=False,
        tune_visual=False,
        tune_top_llm_layers=4 if args.vlm == "top4" else 0,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        use_relative_action=False,
        load_bf16=True,
    )
    # 新增模块允许missing；原预训练骨干/AE任何键缺失都拒绝启动，不重置整个动作专家。
    model, loading = Gr00tN1d6.from_pretrained(
        str(paths["checkpoint"]),
        config=Gr00tN1d6Config(**config),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        output_loading_info=True,
    )
    if (
        any(not name.startswith("action_head.rsci_query.") for name in loading["missing_keys"])
        or loading["unexpected_keys"]
        or loading.get("mismatched_keys")
    ):
        raise ValueError(f"Unexpected pretrained loading mismatch: {loading}")
    # tune_llm=False 仍允许顶部4层；核验实际参数，避免误冻结或误解冻。
    layers = model.backbone.model.language_model.model.layers
    expected = (
        {id(p) for layer in layers[-4:] for p in layer.parameters()}
        if args.vlm == "top4"
        else set()
    )
    actual = {id(p) for p in model.backbone.parameters() if p.requires_grad}
    if actual != expected:
        raise ValueError("Backbone trainable set does not match top4/frozen recipe")
    # AE权重保留fp32以免小更新被bf16量化吞掉，前向仍使用bf16 autocast。
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    enable_dit_recomputation(model)
    world = len(args.gpus.split(","))
    resume = (
        None
        if validation_steps
        else args.resume or latest_checkpoint(paths["output"], world, digest)
    )
    if resume and (
        resume.resolve().parent != paths["output"].resolve()
        or not checkpoint_ok(resume, world, digest, verify_hash=True)
    ):
        raise ValueError("Resume checkpoint is incomplete, corrupt, or belongs to a different run")

    class RSCITrainer(ShardedOptimizerTrainerMixin, Trainer):
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

    class MemoryValidationCallback(TrainerCallback):
        def on_train_begin(self, arguments, state, control, **kwargs):
            if not isinstance(trainer.model_wrapped, torch.nn.parallel.DistributedDataParallel):
                raise RuntimeError("Memory validation must use real multi-GPU DDP")
            if not trainer.model_wrapped.gradient_as_bucket_view:
                raise RuntimeError("DDP gradient bucket views did not take effect")

        def on_step_begin(self, arguments, state, control, **kwargs):
            import time

            self.started_at = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, arguments, state, control, **kwargs):
            import time

            from .memory import unwrap_zero

            torch.cuda.synchronize()
            optimizer = unwrap_zero(trainer.optimizer)
            local_state_bytes = sum(
                value.numel() * value.element_size()
                for parameter_state in optimizer.optim.state.values()
                for value in parameter_state.values()
                if isinstance(value, torch.Tensor)
            )
            print(
                "GR00T_MEMORY "
                + json.dumps(
                    dict(
                        rank=dist.get_rank(),
                        step=state.global_step,
                        vlm=args.vlm,
                        micro_batch=args.batch_size,
                        accumulation=args.gradient_accumulation_steps,
                        optimizer="zero1-adamw-fp32",
                        dit_recomputation=model.action_head.model.gradient_checkpointing,
                        local_optimizer_state_gib=local_state_bytes / 2**30,
                        update_seconds=time.perf_counter() - self.started_at,
                        allocated_gib=torch.cuda.memory_allocated() / 2**30,
                        reserved_gib=torch.cuda.memory_reserved() / 2**30,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                        gradient_as_bucket_view=trainer.model_wrapped.gradient_as_bucket_view,
                    )
                ),
                flush=True,
            )

    training = training_arguments(args, paths, validation_steps=validation_steps)
    if local_rank == 0:
        print(
            "GR00T_RSCI_OPTIMIZATION " + json.dumps(optimization_kwargs(), sort_keys=True),
            flush=True,
        )
        print(
            "GR00T_RSCI_SPEED: BF16 autocast; trainable FP32; TF32; native FlashAttention2; "
            f"ZeRO-1 fused AdamW; DiT layer recomputation; workers/rank={args.num_workers}; pin_memory; non_blocking; "
            f"persistent_workers={args.num_workers > 0}; "
            f"prefetch={2 if args.num_workers else 'none'}; current-frame VLM reuse; "
            f"micro_batch={args.batch_size}; accumulation={args.gradient_accumulation_steps}; "
            f"global_batch={args.batch_size * world * args.gradient_accumulation_steps}; "
            "DDP gradient bucket views; VLM features only",
            flush=True,
        )
    trainer = RSCITrainer(
        model=model,
        args=training,
        train_dataset=dataset,
        data_collator=processor.collator,
        callbacks=[MemoryValidationCallback() if validation_steps else CompleteCallback()],
    )
    # 自定义loss已按微批次归一化，由Trainer按累积次数缩放梯度；不使用语言token计数。
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if validation_steps:
        if trainer.state.global_step != validation_steps:
            raise RuntimeError("Memory validation did not complete all optimizer steps")
        print(
            f"GR00T_MEMORY_VALIDATION_OK rank={dist.get_rank()} steps={validation_steps}",
            flush=True,
        )
        return
    final = paths["output"] / f"checkpoint-{args.max_steps}"
    if trainer.is_world_process_zero() and not checkpoint_ok(final, world, digest):
        raise RuntimeError("Training returned without a complete final checkpoint")
    print(f"GR00T_RSCI_FINISHED {final}", flush=True)


def training_arguments(args, paths, *, validation_steps=0):
    """生产和CPU回归共用同一TrainingArguments构造入口。"""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(paths["output"]),
        max_steps=validation_steps or args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        **optimization_kwargs(),
        bf16=True,
        tf32=True,
        logging_steps=1 if validation_steps else 10,
        save_steps=1000,
        save_strategy="no" if validation_steps else "steps",
        save_total_limit=None,
        save_safetensors=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers else None,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
        ignore_data_skip=False,
        report_to="none",
        accelerator_config={"non_blocking": True},
    )


def guardian(args, paths, digest):
    import signal
    import subprocess
    import sys
    import time

    from .checkpoints import checkpoint_ok, latest_checkpoint

    output = paths["output"]
    world = len(args.gpus.split(","))
    previous_step = -1
    child = None

    def stop_child(signum, frame):
        # 只清理由本守护创建的torchrun进程组，不扫描/终止其他训练和评测。
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_child)
    signal.signal(signal.SIGINT, stop_child)
    for attempt in range(4):
        from rsci.runtime.numerics import reject_numerical_failure

        reject_numerical_failure(output)
        _, current_digest = contract(args, paths)
        if current_digest != digest:
            raise ValueError("Source/data/environment changed; refusing recovery")
        resume = (
            args.resume
            if attempt == 0 and args.resume
            else latest_checkpoint(output, world, digest)
        )
        started = (output / "training_started.json").exists()
        if resume:
            if resume.resolve().parent != output.resolve() or not checkpoint_ok(
                resume, world, digest, verify_hash=True
            ):
                raise ValueError("Resume must be a verified complete checkpoint from this run")
            step = int(resume.name.split("-")[-1])
            if step >= args.max_steps:
                print(f"Already complete: {resume}", flush=True)
                return
            if attempt and step <= previous_step:
                raise RuntimeError("No new complete checkpoint; refusing repeated replay")
            previous_step = step
        elif started or attempt:
            raise RuntimeError("No complete checkpoint; refusing automatic restart from zero")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world}",
            f"--master_port={31000 + int(digest[:4], 16) % 1000}",
            "--module",
            "rsci.backbones.gr00t.train",
            "--worker",
            *worker_arguments(args, parser()),
        ]
        if resume:
            command += ["--resume", str(resume)]
        command = with_nccl(command, args.nccl_library)
        from .checkpoints import atomic_json

        atomic_json(
            output / "training_started.json",
            {"attempt": attempt + 1, "command": command, "time": time.time()},
        )
        print(f"GR00T attempt {attempt + 1}: {command}", flush=True)
        child = subprocess.Popen(
            command,
            start_new_session=True,
            env={
                key: value
                for key, value in os.environ.items()
                if key not in ("NCCL_PROTO", "NCCL_ALGO")
            },
            pass_fds=(args.guardian_fd,),
        )
        result = child.wait()
        # torchrun正常清理各rank；若异常退出，补清该次进程组，仍保留其他任务。
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if result == 0 and checkpoint_ok(output / f"checkpoint-{args.max_steps}", world, digest):
            return
    raise RuntimeError("GR00T recovery attempts exhausted")


def main():
    import fcntl
    import subprocess
    import sys
    import time

    from .checkpoints import atomic_json

    args = parser().parse_args()
    paths = resolve(args)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2), flush=True)
    if args.dry_run:
        print(json.dumps(optimization_kwargs(), sort_keys=True), flush=True)
        return
    if args.check_data:
        import numpy as np

        from .data import SimpleRSCIDataset

        dataset = SimpleRSCIDataset(paths["dataset"], args.model, sidecar=paths["sidecar"])

        # 混乱次序、第一集边界与下一集开头，验证不会跨episode读取历史/标签。
        indices = [6, 0, int(dataset.ends[0]), int(dataset.ends[0]) - 1, 6]
        for index in indices:
            sample = dataset.raw_sample(index)
            if args.model == "VGQ-AE":
                frame = sample.metadata["frame_index"]
                if not np.array_equal(
                    sample.metadata["rsci_history_valid"], np.array([-6, -4, -2, 0]) + frame >= 0
                ):
                    raise ValueError("History crossed episode boundary")
                target = dataset.reader.get(sample.metadata["episode_index"], frame)
                np.testing.assert_array_equal(
                    sample.metadata["rsci_ardy_target"], target.ardy_body_latent
                )
            if bool(sample.states) != (args.model == "STATE"):
                raise ValueError("Unexpected state dependency")
        print(
            f"GR00T_RSCI_DATA_OK task={args.task} model={args.model} samples={len(dataset)}; no model loaded",
            flush=True,
        )
        return
    if args.guardian_fd is not None:
        identity, digest = contract(args, paths)
        if json.loads((paths["output"] / "launch_contract.json").read_text()) != identity:
            raise ValueError("Guardian launch contract mismatch")
        guardian(args, paths, digest)
        return
    check_training_environment()
    processor, dataset = preflight(args, paths)
    if args.check:
        return
    identity, digest = contract(args, paths)
    if args.worker:
        if json.loads((paths["output"] / "launch_contract.json").read_text()) != identity:
            raise ValueError("Worker launch contract mismatch")
        train(args, paths, processor, dataset, digest)
        return
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != len(args.gpus.split(",")):
        raise ValueError("Visible CUDA devices do not match --gpus")
    output = paths["output"]
    from rsci.runtime.numerics import reject_numerical_failure

    reject_numerical_failure(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "guardian.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = output / "launch_contract.json"
        if record.exists() and json.loads(record.read_text()) != identity:
            raise ValueError("Existing run has a different contract; use a new run-tag")
        if not record.exists():
            atomic_json(record, identity)
        log_path = output / "guardian.log"
        with log_path.open("ab", buffering=0) as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "rsci.backbones.gr00t.train",
                    *sys.argv[1:],
                    f"--guardian-fd={lock.fileno()}",
                ],
                start_new_session=True,
                stdout=log,
                stderr=log,
                pass_fds=(lock.fileno(),),
            )
        print(f"GR00T guardian PID={child.pid}; log={log_path}; Ctrl+C仅退出日志查看", flush=True)
        with log_path.open() as log:
            try:
                while child.poll() is None:
                    line = log.readline()
                    if line:
                        print(line, end="", flush=True)
                    else:
                        time.sleep(0.2)
                print(log.read(), end="", flush=True)
            except KeyboardInterrupt:
                return
        if child.returncode:
            raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
