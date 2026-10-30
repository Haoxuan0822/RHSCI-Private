"""act: 兼容命令入口，负责流程编排。"""

import json
import os
import sys

from rsci.paths import with_nccl as with_nccl

from .config import parser as parser
from .config import resolve as resolve
from .config import training_arguments as training_arguments
from .contract import CODE as CODE
from .contract import contract as contract
from .contract import digest_contract as digest_contract
from .preflight import check_data as check_data
from .preflight import initialization as initialization
from .preflight import nccl_runtime as nccl_runtime
from .trainer import completion_callback as completion_callback
from .trainer import create_optimizer as create_optimizer
from .trainer import require_finite as require_finite
from .trainer import trainer_type as trainer_type
from .worker import train as train

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def main():
    args = parser().parse_args()
    if args.nccl_library:
        os.environ["RSCI_NCCL_LIBRARY"] = str(args.nccl_library.resolve(strict=True))
    if not (args.worker or args.dry_run or args.check or args.check_data) and not os.environ.get(
        "RSCI_NCCL_PRELOADED"
    ):
        command = [sys.executable, "-m", "rsci.backbones.act.train", *sys.argv[1:]]
        wrapped = with_nccl(command, args.nccl_library, expected_version="2.27.3")
        if wrapped != command:
            os.execvpe(wrapped[0], wrapped, {**os.environ, "RSCI_NCCL_PRELOADED": "1"})
    paths = resolve(args)
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

            from .model import RSCIACT, official_config

            model = RSCIACT(args.model, official_config(args.model, pretrained=False)).eval()
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
        value = contract(args, paths)
        digest = digest_contract(value)
        train(args, paths, value, digest)
    else:
        from .launch import guardian

        guardian(args, paths)


if __name__ == "__main__":
    main()
