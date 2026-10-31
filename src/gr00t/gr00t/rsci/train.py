"""gr00t: 兼容命令入口，负责流程编排。"""

import json

from rsci.data.cache_config import cache_arguments

from .config import optimization_kwargs as optimization_kwargs
from .config import parser as parser
from .config import resolve as resolve
from .config import training_arguments as training_arguments
from .contract import contract as contract
from .guardian import guardian as guardian
from .preflight import build_processor as build_processor
from .preflight import check_training_environment as check_training_environment
from .preflight import preflight as preflight
from .trainer import resume_trainer_type as resume_trainer_type
from .worker import train as train

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


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

        dataset = SimpleRSCIDataset(
            paths["dataset"],
            args.model,
            sidecar=paths["sidecar"],
            seed=args.seed,
            cache=paths["input_cache"],
        )

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
