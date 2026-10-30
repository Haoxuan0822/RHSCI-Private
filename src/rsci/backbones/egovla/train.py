"""egovla: 兼容命令入口，负责流程编排。"""

import json
import os

from .config import parser as parser
from .config import resolve as resolve
from .config import training_arguments as training_arguments
from .contract import CODE as CODE
from .contract import contract as contract
from .contract import digest_contract as digest_contract
from .guardian import guardian as guardian
from .preflight import asset_files as asset_files
from .preflight import assets as assets
from .preflight import check_data as check_data
from .trainer import completion_callback as completion_callback
from .trainer import configure_vlm_execution as configure_vlm_execution
from .trainer import require_finite as require_finite
from .trainer import trainer_type as trainer_type
from .worker import train as train

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def main():
    args = parser().parse_args()
    paths = resolve(args)
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
    manifest = assets(paths["checkpoint"], register=args.register_assets, verify=not args.worker)
    if args.register_assets:
        print("EGOVLA_ASSETS_REGISTERED; no training started")
        return
    if not args.worker:
        check_data(paths, args.model)
    value = contract(args, paths, manifest)
    digest = digest_contract(value)
    if args.check:
        from .native import load_native
        from .processing import Collator

        model, processor = load_native(
            paths["checkpoint"], vlm_training=args.vlm_training, variant=args.model
        )
        configure_vlm_execution(model, args)
        dataset = check_data(paths, args.model)
        dataset.processor = processor
        batch = Collator(processor.tokenizer.pad_token_id)([dataset[0], dataset[6]])
        if batch["actions"].shape != (2, 30, 36):
            raise ValueError("Invalid model batch")
        # 主进程预处理通过不代表worker能用；真实spawn读取同样两帧，
        # 核对模板、动作标签、RGB/ARDY全字段，且不运行任何训练更新。
        import torch
        from torch.utils.data import DataLoader, Subset

        worker_loader = DataLoader(
            Subset(dataset, [0, 6]),
            batch_size=2,
            num_workers=1,
            multiprocessing_context="spawn",
            timeout=120,
            collate_fn=Collator(processor.tokenizer.pad_token_id),
        )
        (worker_batch,) = list(worker_loader)
        if worker_batch.keys() != batch.keys():
            raise ValueError("Worker preprocessing keys differ from parent")
        for key in batch:
            torch.testing.assert_close(
                worker_batch[key],
                batch[key],
                rtol=0,
                atol=0,
                msg=f"Worker preprocessing mismatch: {key}",
            )
        print("EGOVLA_SPAWN_PROCESSOR_CHECK_OK; no training started", flush=True)
        print(
            "EGOVLA_PARAMETERS "
            + json.dumps(
                dict(
                    vlm_training=args.vlm_training,
                    total=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    vlm_trainable=sum(
                        p.numel()
                        for module in (
                            model.backbone.get_llm(),
                            model.backbone.get_vision_tower(),
                            model.backbone.get_mm_projector(),
                        )
                        for p in module.parameters()
                        if p.requires_grad
                    ),
                )
            ),
            flush=True,
        )
        print("EGOVLA_MODEL_PROCESSOR_CHECK_OK; no training started")
    elif args.worker:
        expected = json.loads((paths["output"] / "launch_contract.json").read_text())
        if expected != value:
            raise ValueError("Worker source/data/recipe contract changed")
        train(args, paths, value, digest)
    elif args.guardian:
        guardian(args, paths, value, digest)
    else:
        raise RuntimeError("Use scripts/train/egovla.sh for the detached guardian and live log")


if __name__ == "__main__":
    main()
