"""EgoVLA RHSCI 训练入口：编排参数解析、预检与守护/worker 角色。"""

import json
import os

from .config import build_argument_parser as build_argument_parser
from .config import build_training_arguments as build_training_arguments
from .config import resolve_training_paths as resolve_training_paths
from .contract import CODE as CODE
from .contract import build_training_contract as build_training_contract
from .contract import compute_contract_digest as compute_contract_digest
from .guardian import run_training_guardian as run_training_guardian
from .preflight import asset_files as asset_files
from .preflight import assets as assets
from .preflight import check_data as check_data
from .preflight import validate_model_and_worker_preprocessing
from .trainer import build_checkpoint_callback as build_checkpoint_callback
from .trainer import configure_vlm_execution as configure_vlm_execution
from .trainer import create_trainer_class as create_trainer_class
from .trainer import require_finite as require_finite
from .worker import run_training_worker as run_training_worker

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def main():
    args = build_argument_parser().parse_args()
    paths = resolve_training_paths(args)
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
    manifest = assets(paths["checkpoint"], register=args.register_assets, verify=args.check)
    if args.register_assets:
        print("EGOVLA_ASSETS_REGISTERED; no training started")
        return
    if not args.worker:
        check_data(paths, args.model)
    value = build_training_contract(args, paths, manifest)
    digest = compute_contract_digest(value)
    if args.check:
        validate_model_and_worker_preprocessing(args, paths)
    elif args.worker:
        expected = json.loads((paths["output"] / "launch_contract.json").read_text())
        if expected != value:
            raise ValueError("Worker source/data/recipe contract changed")
        run_training_worker(args, paths, value, digest)
    elif args.guardian:
        run_training_guardian(args, paths, value, digest)
    else:
        raise RuntimeError("Use scripts/train/egovla.sh for the detached guardian and live log")


if __name__ == "__main__":
    main()
