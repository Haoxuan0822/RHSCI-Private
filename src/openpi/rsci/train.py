"""pi05: 兼容命令入口，负责流程编排。"""

import json
import os
import sys

from rsci.data.cache_config import cache_arguments
from rsci.paths import with_nccl as with_nccl

from .checkpoints import atomic_json as atomic_json
from .checkpoints import checkpoint_ok as checkpoint_ok
from .config import parser as parser
from .config import resolve as resolve
from .config import simple_lr_multiplier as simple_lr_multiplier
from .config import training_arguments as training_arguments
from .contract import contract as contract
from .guardian import guardian as guardian
from .preflight import preflight as preflight
from .preflight import validate_run as validate_run
from .trainer import complete_callback as complete_callback
from .trainer import trainer_class as trainer_class
from .worker import train as train

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


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
                    "openpi.rsci.train",
                    *sys.argv[1:],
                    f"--guardian-fd={lock.fileno()}",
                    *cache_arguments(paths["input_cache"]),
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
