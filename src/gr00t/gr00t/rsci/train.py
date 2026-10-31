"""GR00T RHSCI 训练入口：编排参数解析、预检与守护/worker 角色。"""

import json

from rsci.data.cache_config import cache_arguments

from .config import build_argument_parser as build_argument_parser
from .config import build_training_arguments as build_training_arguments
from .config import optimization_kwargs as optimization_kwargs
from .config import resolve_training_paths as resolve_training_paths
from .contract import build_training_contract as build_training_contract
from .guardian import run_training_guardian as run_training_guardian
from .preflight import build_processor as build_processor
from .preflight import check_training_environment as check_training_environment
from .preflight import run_training_preflight as run_training_preflight
from .preflight import validate_training_data
from .trainer import create_resume_trainer_class as create_resume_trainer_class
from .worker import run_training_worker as run_training_worker

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def main():
    import fcntl
    import subprocess
    import sys
    import time

    from .checkpoints import atomic_json

    args = build_argument_parser().parse_args()
    paths = resolve_training_paths(args)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2), flush=True)
    if args.dry_run:
        print(json.dumps(optimization_kwargs(), sort_keys=True), flush=True)
        return
    if args.check_data:
        validate_training_data(args, paths)
        return
    if args.guardian_fd is not None:
        identity, digest = build_training_contract(args, paths)
        if json.loads((paths["output"] / "launch_contract.json").read_text()) != identity:
            raise ValueError("Guardian launch contract mismatch")
        run_training_guardian(args, paths, digest)
        return
    check_training_environment()
    processor, dataset = run_training_preflight(args, paths)
    if args.check:
        return
    identity, digest = build_training_contract(args, paths)
    if args.worker:
        if json.loads((paths["output"] / "launch_contract.json").read_text()) != identity:
            raise ValueError("Worker launch contract mismatch")
        run_training_worker(args, paths, processor, dataset, digest)
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
                    "gr00t.rsci.train",
                    *sys.argv[1:],
                    f"--guardian-fd={lock.fileno()}",
                    *cache_arguments(paths["input_cache"]),
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
