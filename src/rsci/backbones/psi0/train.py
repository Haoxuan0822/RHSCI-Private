"""PSI0 RHSCI 统一训练、只读预检和独立断点守护；不下载资产。"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from rsci.backbones.psi0.defaults import training_args as original_args
from rsci.data.state_noise import validate_scale
from rsci.runtime.processes import cleanup, process_table, remember_tree

from .checkpoints import checkpoint_ok

CODE = Path(__file__).resolve().parents[4]
VARIANTS = ("STATE", "NO-STATE", "VGQ-AE", "EGQ-AE", "VGQ-VLM", "EGQ-VLM", "VGQ-VLM*", "EGQ-VLM*")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", choices=VARIANTS, required=True, help="带星号的方案请用引号，例如 'VGQ-VLM*'"
    )
    p.add_argument("--task", required=True, help="本地 SIMPLE repo/task 名")
    p.add_argument("--vlm", choices=("frozen", "lora"), default="frozen")
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    p.add_argument("--port", type=int)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--sidecar-root", type=Path)
    p.add_argument("--vlm-path", type=Path)
    p.add_argument("--ae-path", type=Path)
    p.add_argument("--dino-path", type=Path)
    p.add_argument("--batch-size", type=int, default=16, help="每卡 batch size")
    p.add_argument(
        "--no-reader", action="store_true", help="Use full visual tokens for query memory"
    )
    p.add_argument(
        "--ardy-offset", type=int, default=0, help="Future ARDY target offset in 50 Hz frames"
    )
    p.add_argument(
        "--state-measurement-noise-std-rad",
        type=float,
        nargs="?",
        const=0.02,
        default=0.0,
        help="Training noise before normalization; flag alone selects 0.02 rad; STATE only",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--output-dir", type=Path)
    p.add_argument(
        "--nccl-library",
        type=Path,
        help="Optional validated NCCL shared library; otherwise use the environment runtime",
    )
    p.add_argument("--seed", type=int, default=292285)
    p.add_argument("--run-tag", default="main", help="源码/超参数改变时选择新的独立运行名")
    p.add_argument(
        "--resume", type=Path, help="完整 RHSCI checkpoint；默认自动查找同组最新完整断点"
    )
    modes = p.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run", action="store_true", help="解析并打印配置，不加载样本/模型或启动训练"
    )
    modes.add_argument(
        "--check", action="store_true", help="检查本地模型、数据与正式 ARDY 标签，不启动训练"
    )
    p.add_argument("--guardian-fd", type=int, help=argparse.SUPPRESS)
    return p


def resolved(args):
    noise = validate_scale(args.state_measurement_noise_std_rad, args.model)
    if args.ardy_offset < 0 or (
        (args.ardy_offset or args.no_reader) and args.model in ("STATE", "NO-STATE")
    ):
        raise ValueError("Future ARDY and --no-reader require an RHSCI Query variant")
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", args.task)
        or not re.fullmatch(r"\d+(,\d+)*", args.gpus)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag)
    ):
        raise ValueError(
            "task must be a simple repo name; gpus must be comma-separated device indices"
        )
    if (
        len(set(args.gpus.split(","))) != len(args.gpus.split(","))
        or args.batch_size < 1
        or args.num_workers < 1
        or args.max_steps < 1
    ):
        raise ValueError("GPU indices must be unique, batch-size and num-workers positive")
    runtime = Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data")).resolve()
    model_slug = args.model.replace("*", "-star")
    if args.no_reader:
        model_slug += "-no-reader"
    if args.ardy_offset:
        model_slug += f"-F{args.ardy_offset}"
    if noise:
        model_slug += f"-noise-{noise:g}rad"
    group = f"{args.task}/{model_slug}/{args.vlm}/seed-{args.seed}/{args.run_tag}"
    output = args.output_dir or runtime / "runs/psi0" / group
    local = runtime / "cache/checkpoints/psi0"
    updates = {
        "exp": f"RHSCI-{args.task}-{args.model.replace('*', '-star')}-{args.vlm}-{args.run_tag}",
        "seed": args.seed,
        "train.output-dir": output,
        "train.train_batch_size": args.batch_size,
        "train.max_training_steps": args.max_steps,
        "train.num-workers": args.num_workers,
        "log.report_to": "None",
        # Reuse the validated optimizer and transfer settings.
        "train.val-num-workers": args.num_workers,
        "train.prefetch-factor": 2,
        "data.root_dir": args.data_root or runtime / "data/simple",
        "data.train-repo-ids": args.task,
        "model.model_name_or_path": args.vlm_path
        or local / "pre.fast.1by1.2601091803.ckpt.ego200k.he30k",
        "model.pretrained-action-header-path": args.ae_path
        or local / "postpre.1by1.pad36.2601131206.ckpt.he30k",
        "model.rsci-variant": args.model,
        "model.rsci-vlm-mode": args.vlm,
        "model.state-measurement-noise-std-rad": noise,
        "data.transform.field.state-measurement-noise-std-rad": noise,
        "data.rsci-ardy-offset-frames": args.ardy_offset,
    }
    query = args.model not in ("STATE", "NO-STATE")
    if query:
        updates["data.rsci-ardy-sidecar-root"] = (
            args.sidecar_root or runtime / "data/simple-ardy-sidecars" / args.task
        )
    if args.model.startswith("EGQ-"):
        updates["model.rsci-rgb-encoder-path"] = (
            args.dino_path or runtime / "cache/checkpoints/dinov2-small"
        )
    # Apply the model, dataset and output choices to the native training defaults.
    old = original_args(runtime)
    values = [old[0]]
    excluded = {"--model.no-tune-vlm", "--train.skip-resumed-steps"}
    for item in old[1:]:
        key = item.split("=", 1)[0].removeprefix("--")
        if (
            item in excluded
            or key in updates
            or key.startswith("data.rsci-ardy-sidecar-root")
            or key.startswith("train.resume-from-checkpoint")
        ):
            continue
        values.append(item)
    updates["data.transform.repack.rsci-variant"] = args.model
    values += [f"--{key}={value}" for key, value in updates.items()]
    if args.no_reader:
        values.append("--model.no-rsci-use-reader")
    values += ["--train.optimizer-fused", "--train.pin-memory", "--train.non-blocking-transfer"]
    # 严禁环境中旧实验的 resume 隐式接入当前实验。
    import tyro

    from psi.config.train.finetune_simple_psi0_config import DynamicLaunchConfig

    cfg = tyro.cli(
        DynamicLaunchConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=values[1:]
    )
    cfg.data.transform.repack.rsci_variant = args.model
    if cfg.auto_tag_run:
        raise ValueError("RHSCI must not auto-commit unrelated workspace edits")
    ranks = len(args.gpus.split(","))
    port = args.port or 29600 + int(hashlib.sha256(group.encode()).hexdigest()[:4], 16) % 1000
    if not 1024 <= port <= 65535:
        raise ValueError("Invalid distributed port")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={ranks}",
        f"--master_port={port}",
        str(CODE / "scripts/train.py"),
        *values,
    ]
    if args.nccl_library:
        command = [
            sys.executable,
            str(CODE / "scripts/with_nccl_runtime.py"),
            "--library",
            str(args.nccl_library),
            "--",
            *command,
        ]
    return cfg, command, output, ranks


def contract(cfg, ranks):
    data = cfg.model_dump(mode="json")
    train = dict(data["train"])
    for key in ("resume_from_checkpoint", "skip_resumed_steps"):
        train.pop(key, None)
    # 源码也属于恢复契约，结构/数据/训练参数不同不得续接。
    digest = hashlib.sha256()
    for name in (
        "src/rsci/backbones/psi0/model.py",
        "src/psi/models/psi0.py",
        "src/psi/config/model_psi0.py",
        "src/psi/config/transform.py",
        "src/psi/config/data_lerobot.py",
        "src/psi/trainers/finetune.py",
        "src/rsci/backbones/psi0/train.py",
        "src/psi/data/lerobot/lerobot_ext.py",
        "src/psi/trainers/trainer.py",
        "scripts/train.py",
        "src/rsci/backbones/psi0/rgb_encoder.py",
        "src/psi/trainers/qwen3vl_mixin.py",
        "src/rsci/runtime/profiling.py",
        "scripts/with_nccl_runtime.py",
        "src/rsci/backbones/psi0/defaults.py",
        "src/rsci/runtime/processes.py",
        "src/rsci/backbones/psi0/trainer.py",
        "src/rsci/backbones/psi0/checkpoints.py",
        "src/rsci/runtime/checkpoints.py",
    ):
        digest.update((CODE / name).read_bytes())
    for directory in ("src/rsci/runtime", "src/rsci/data"):
        for path in sorted((CODE / directory).glob("*.py")):
            digest.update(path.relative_to(CODE).as_posix().encode())
            digest.update(path.read_bytes())
    return dict(
        schema=1,
        model=data["model"],
        data=data["data"],
        train=train,
        seed=data["seed"],
        world_size=ranks,
        source_sha256=digest.hexdigest(),
    )


def preflight(cfg):
    from transformers import AutoProcessor

    from psi.data.lerobot import LeRobotDatasetWrapper
    from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether

    vlm = Path(cfg.model.model_name_or_path)
    required = [
        vlm / "config.json",
        Path(cfg.model.pretrained_action_header_path) / "action_header.safetensors",
        Path(cfg.data.root_dir) / cfg.data.train_repo_ids[0] / "meta/stats_psi0.json",
    ]
    if not any(
        (vlm / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")
    ):
        raise ValueError(f"Missing local VLM weights: {vlm}")
    if cfg.model.rsci_variant.startswith("EGQ-"):
        dino = Path(cfg.model.rsci_rgb_encoder_path)
        required += [dino / "config.json"]
        if not any((dino / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            raise ValueError(f"Missing local DINOv2 weights: {dino}")
    for path in required:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing local asset: {path}")
    # wrapper 执行正式教师、数据指纹及逐 episode sidecar 长度门禁。
    wrapper = LeRobotDatasetWrapper(cfg.data)
    if cfg.data.rsci_ardy_offset_frames:
        from rsci.data.ardy_future import supervision_counts

        counts = supervision_counts(wrapper.ardy_reader, cfg.data.rsci_ardy_offset_frames)
        if not counts["valid_ardy_frames"]:
            raise ValueError("No valid future ARDY targets in this dataset")
        print("RSCI_ARDY_SUPERVISION " + json.dumps(counts), flush=True)
    if cfg.model.rsci_variant:
        if "observation.depth.egocentric" in wrapper.base_dataset.hf_dataset.column_names:
            raise ValueError("RHSCI RGB-only reader must exclude unused depth column")
    processor = AutoProcessor.from_pretrained(str(vlm), local_files_only=True)
    sample = cfg.data.transform(
        wrapper[min(6, len(wrapper) - 1)], vlm_processor=processor, no_aug=True
    )
    batch = PaddedCollatorForTogether(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id
    )([sample])
    if "depths" in batch:
        raise ValueError("RHSCI must not request depth")
    if cfg.model.rsci_variant not in ("STATE", "NO-STATE"):
        if "ardy_target" not in batch or batch["history_valid"].shape != (1, 4):
            raise ValueError("Invalid four-frame ARDY sample")
        if cfg.model.rsci_variant.startswith("VGQ-") and len(batch["history_vlm_inputs"]) != 3:
            raise ValueError("Invalid VGQ historical inputs")
        if cfg.model.rsci_variant.startswith("EGQ-") and batch["history_rgb_images"].shape[1:3] != (
            4,
            3,
        ):
            raise ValueError("Invalid EGQ historical inputs")
    print(
        "RHSCI local assets, formal dataset gate and one CPU preprocessing/collation sample passed; no training started.",
        flush=True,
    )


def latest_checkpoint(output, ranks):
    if any(Path(output).rglob("NUMERICAL_FAILURE.json")):
        raise RuntimeError("NUMERICAL_FAILURE: automatic recovery is disabled")
    candidates = [
        path
        for path in output.glob("finetune/*/checkpoints/ckpt_*")
        if path.name.removeprefix("ckpt_").isdigit() and checkpoint_ok(path, ranks)
    ]
    return max(
        candidates,
        key=lambda path: (int(path.name.removeprefix("ckpt_")), path.stat().st_mtime),
        default=None,
    )


def guard(args, cfg, command, output, ranks):
    if any(Path(output).rglob("NUMERICAL_FAILURE.json")):
        raise RuntimeError("NUMERICAL_FAILURE: automatic recovery is disabled")
    tracked = {}

    def stop(signum, frame):
        cleanup(tracked)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous_step = -1
    for attempt in range(4):
        identity_path = output / "launch_contract.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != contract(cfg, ranks):
            raise ValueError(
                "RHSCI source/parameters changed during training; refusing recovery with a different contract"
            )
        resume = args.resume if attempt == 0 and args.resume else latest_checkpoint(output, ranks)
        runs_exist = (
            any(output.glob("finetune/*/run_config.json"))
            or (output / "training_started.json").exists()
        )
        if resume:
            if not checkpoint_ok(resume, ranks) or output.resolve() not in resume.resolve().parents:
                raise ValueError(
                    "Resume must be a complete checkpoint in this experiment's output directory"
                )
            saved = json.loads((resume.parent.parent / "run_config.json").read_text())
            saved_model = saved["model"]
            if (
                saved_model != cfg.model_dump(mode="json")["model"]
                or saved["data"] != cfg.model_dump(mode="json")["data"]
            ):
                raise ValueError(
                    "Checkpoint model/data do not match the resolved RHSCI configuration"
                )
            step = int(resume.name.removeprefix("ckpt_"))
            if step >= cfg.train.max_training_steps:
                print(f"Complete: {resume}", flush=True)
                return
            if attempt > 0 and step <= previous_step:
                raise RuntimeError(
                    "No new complete checkpoint since previous recovery; refusing repeated replay"
                )
            previous_step = step
        elif runs_exist or attempt > 0:
            raise RuntimeError(
                "No complete checkpoint; refusing to restart this experiment from step zero"
            )
        launch = list(command)
        if resume:
            launch += [f"--train.resume-from-checkpoint={resume}", "--train.skip-resumed-steps"]
        print(f"Attempt {attempt + 1}: {shlex.join(launch)}", flush=True)
        with (output / "training_started.json").open("w") as marker:
            json.dump(dict(attempt=attempt + 1, command=launch, started_at=time.time()), marker)
        child = subprocess.Popen(
            launch,
            cwd=CODE,
            start_new_session=True,
            env={k: v for k, v in os.environ.items() if k not in ("NCCL_PROTO", "NCCL_ALGO")},
            pass_fds=(args.guardian_fd,) if args.guardian_fd is not None else (),
        )
        tracked = {}
        try:
            remember_tree(child.pid, tracked, process_table(), own_group=child.pid)
            while child.poll() is None:
                remember_tree(child.pid, tracked, process_table(), own_group=child.pid)
                time.sleep(1)
        finally:
            remember_tree(child.pid, tracked, process_table(), own_group=child.pid)
            cleanup(tracked)
        if child.returncode == 0:
            checkpoint = latest_checkpoint(output, ranks)
            if (
                checkpoint
                and int(checkpoint.name.removeprefix("ckpt_")) >= cfg.train.max_training_steps
            ):
                print(f"Complete: {checkpoint}", flush=True)
                return
            raise RuntimeError("Training exited without a complete final checkpoint")
    raise RuntimeError("RHSCI recovery attempts exhausted")


def main():
    args = parser().parse_args()
    from rsci.backbones.psi0.contract import validate_rsci_training

    validate_rsci_training(args.model, args.vlm)
    cfg, command, output, ranks = resolved(args)
    print(shlex.join(command), flush=True)
    print(
        json.dumps(
            dict(
                model=args.model, task=args.task, vlm=args.vlm, output=str(output), world_size=ranks
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["PSI_RSCI_TRAIN"] = "1"  # 此入口完全使用显式配置，不要求复制其他工作树的 .env。
    if args.guardian_fd is not None:
        guard(args, cfg, command, output, ranks)
        return
    preflight(cfg)
    if args.check:
        return
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != ranks:
        raise ValueError("Visible CUDA GPUs do not match --gpus")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "guardian.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = contract(cfg, ranks)
        path = output / "launch_contract.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError(
                "Existing experiment has different source/parameters; use a separate output namespace"
            )
        if not path.exists():
            with path.open("x") as handle:
                json.dump(identity, handle, indent=2)
        log_path = output / "guardian.log"
        with log_path.open("ab", buffering=0) as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "rsci.backbones.psi0.train",
                    *sys.argv[1:],
                    f"--guardian-fd={lock.fileno()}",
                ],
                cwd=CODE,
                stdout=log,
                stderr=log,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
        print(
            f"RHSCI guardian PID={child.pid}; log={log_path}; Ctrl+C only closes log viewing.",
            flush=True,
        )
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
