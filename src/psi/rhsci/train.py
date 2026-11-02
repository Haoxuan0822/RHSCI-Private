"""PSI0 RHSCI 统一训练、只读预检和独立断点守护；不下载资产。"""

import fcntl
import json
import os
import shlex
import subprocess
import sys
import time

from rhsci.paths import CODE_ROOT as CODE_ROOT

from .contract import build_training_contract as build_training_contract
from .defaults import build_training_arguments as build_training_arguments
from .guardian import latest_checkpoint as latest_checkpoint
from .guardian import run_training_guardian as run_training_guardian
from .preflight import run_training_preflight as run_training_preflight
from .training_config import VARIANTS as VARIANTS
from .training_config import build_argument_parser as build_argument_parser
from .training_config import resolve_training_plan as resolve_training_plan

CODE = CODE_ROOT


def main():
    args = build_argument_parser().parse_args()
    from psi.rhsci.contract import validate_rhsci_training

    validate_rhsci_training(args.model, args.vlm)
    cfg, command, output, ranks = resolve_training_plan(args)
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
    os.environ["PSI_RHSCI_TRAIN"] = "1"  # 此入口完全使用显式配置，不要求复制其他工作树的 .env。
    if args.guardian_fd is not None:
        run_training_guardian(args, cfg, command, output, ranks)
        return
    run_training_preflight(cfg)
    if args.check:
        return
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != ranks:
        raise ValueError("Visible CUDA GPUs do not match --gpus")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "guardian.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = build_training_contract(cfg, ranks)
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
                    "psi.rhsci.train",
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
