"""ACT RHSCI 训练入口：编排参数解析、预检与守护/worker 角色。"""

import json
import os
import sys

from rhsci.paths import with_nccl as with_nccl
from rhsci.runtime.environment import get_env

from .config import build_argument_parser as build_argument_parser
from .config import build_training_arguments as build_training_arguments
from .config import resolve_training_paths as resolve_training_paths
from .contract import CODE as CODE
from .contract import build_training_contract as build_training_contract
from .contract import compute_contract_digest as compute_contract_digest
from .preflight import check_data as check_data
from .preflight import initialization as initialization
from .preflight import nccl_runtime as nccl_runtime
from .trainer import build_checkpoint_callback as build_checkpoint_callback
from .trainer import create_optimizer as create_optimizer
from .trainer import create_trainer_class as create_trainer_class
from .trainer import require_finite as require_finite
from .worker import run_training_worker as run_training_worker


def main():
    args = build_argument_parser().parse_args()
    if args.nccl_library:
        os.environ["RHSCI_NCCL_LIBRARY"] = str(args.nccl_library.resolve(strict=True))
    if not (args.worker or args.dry_run or args.check or args.check_data) and not get_env(
        "RHSCI_NCCL_PRELOADED"
    ):
        command = [sys.executable, "-m", "act.rhsci.train", *sys.argv[1:]]
        wrapped = with_nccl(command, args.nccl_library, expected_version="2.27.3")
        if wrapped != command:
            os.execvpe(wrapped[0], wrapped, {**os.environ, "RHSCI_NCCL_PRELOADED": "1"})
    paths = resolve_training_paths(args)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2), flush=True)
    if args.dry_run:
        world = len(args.gpus.split(","))
        print(
            f"ACT_RECIPE batch={args.batch_size} world={world} accumulation={args.gradient_accumulation_steps} "
            f"global_batch={args.batch_size * world * args.gradient_accumulation_steps} seed={args.seed} "
            f"AdamW=1e-4/.95/.999/wd1e-6 warmup=1000 cosine_steps={args.max_steps} chunk=16"
        )
        return
    if args.check or args.check_data:
        dataset = check_data(args, paths)
        if args.check:
            import torch

            from .model import RHSCIACT, official_config

            model = RHSCIACT(args.model, official_config(args.model, pretrained=False)).eval()
            batch = {key: value[None] for key, value in dataset[min(6, len(dataset) - 1)].items()}
            with torch.no_grad():
                outputs = model(**batch)
                prediction = model.predict_action(batch)
            if not all(torch.isfinite(x).all() for x in outputs.values()) or prediction.shape != (
                1,
                16,
                36,
            ):
                raise ValueError("Full ACT CPU model check failed")
            print(
                f"ACT_CPU_CHECK_OK parameters={sum(p.numel() for p in model.parameters())}; random init, no training"
            )
        return
    if args.worker:
        value = build_training_contract(args, paths)
        digest = compute_contract_digest(value)
        run_training_worker(args, paths, value, digest)
    else:
        from .launch import run_training_guardian

        run_training_guardian(args, paths)


if __name__ == "__main__":
    main()
