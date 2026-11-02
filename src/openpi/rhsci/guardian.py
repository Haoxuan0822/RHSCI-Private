"""PI05 守护适配：原生 worker 命令、配方与断点校验。"""

import sys
from functools import partial

from rhsci.data.cache_config import cache_arguments
from rhsci.paths import with_nccl, worker_arguments
from rhsci.runtime.guardian import run_checkpoint_guardian

from .checkpoints import checkpoint_ok, latest_checkpoint
from .config import build_argument_parser
from .contract import build_training_contract


def run_training_guardian(args, paths, digest):
    """绑定本骨干配置后执行公共重试循环；锁与断点格式沿用既有入口。"""
    world = len(args.gpus.split(","))

    def validate_run():
        if build_training_contract(args, paths)[1] != digest:
            raise ValueError("Training configuration/data changed; refusing recovery")

    def build_command(resume):
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world}",
            f"--master_port={32000 + int(digest[:4], 16) % 1000}",
            "--module",
            "openpi.rhsci.train",
            *worker_arguments(args, build_argument_parser()),
            "--worker",
            *cache_arguments(paths["input_cache"]),
        ]
        if resume:
            command += ["--resume", str(resume)]
        return with_nccl(command, args.nccl_library)

    run_checkpoint_guardian(
        paths["output"],
        max_steps=args.max_steps,
        resume=args.resume,
        latest_checkpoint=partial(latest_checkpoint, paths["output"], world, digest),
        checkpoint_ok=lambda path, **options: checkpoint_ok(path, world, digest, **options),
        validate_run=validate_run,
        build_command=build_command,
        label="PI05",
        pass_fds=(args.guardian_fd,),
    )
