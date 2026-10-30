"""egovla: 单个训练worker的装配与执行。"""

import os

from .checkpoints import atomic_json, checkpoint_ok, latest
from .config import training_arguments
from .trainer import completion_callback, configure_vlm_execution, trainer_type

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def train(args, paths, value, digest):
    import torch

    if os.environ.get("RSCI_NCCL_LIBRARY"):
        # Torch的version查询可能反映编译版本，必须核对实际全局动态符号。
        import ctypes

        actual_nccl = ctypes.c_int()
        result = ctypes.CDLL(None).ncclGetVersion(ctypes.byref(actual_nccl))
        major, minor, patch = map(int, os.environ.get("RSCI_NCCL_VERSION", "2.27.3").split("."))
        expected = major * 10000 + minor * 100 + patch
        if result != 0 or actual_nccl.value != expected:
            raise RuntimeError(f"Expected NCCL {expected}; got {actual_nccl.value}")
        print(f"EGOVLA_NCCL_RUNTIME version={actual_nccl.value}", flush=True)
    from transformers import set_seed

    from .data import SimpleDataset
    from .native import load_native
    from .processing import Collator

    world = len(args.gpus.split(","))
    if int(os.environ.get("WORLD_SIZE", 1)) != world:
        raise ValueError("Launch world size differs from contract")
    # 先绑定rank，避免所有rank初始化/allocator意外占用GPU0。
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    set_seed(args.seed)
    model, processor = load_native(
        paths["checkpoint"], vlm_training=args.vlm_training, variant=args.model
    )
    configure_vlm_execution(model, args)
    dataset = SimpleDataset(
        paths["dataset"],
        paths["sidecar"],
        processor,
        variant=args.model,
        cache=paths["input_cache"],
    )
    trainer = trainer_type()(
        model=model,
        args=training_arguments(args, paths),
        train_dataset=dataset,
        data_collator=Collator(processor.tokenizer.pad_token_id),
        launch_contract=value,
        callbacks=[completion_callback(paths["output"], world, digest)],
    )
    if args.num_train_epochs:
        actual_batches = len(trainer.get_train_dataloader())
        import math

        if (
            math.ceil(actual_batches / args.gradient_accumulation_steps) * args.num_train_epochs
            != args.max_steps
        ):
            raise ValueError("Actual distributed loader differs from the epoch step budget")
    resume = latest(paths["output"], world, digest)
    print(
        f"EGOVLA_TRAIN resume={resume} global_batch={args.batch_size * args.gradient_accumulation_steps * world}",
        flush=True,
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    # 非1000整数步的显式短训练同样保存完整HF状态和RNG；不把裸权重当完成。
    if not checkpoint_ok(paths["output"] / f"checkpoint-{args.max_steps}", world, digest):
        trainer._save_checkpoint(trainer.model, trial=None)
        completion_callback(paths["output"], world, digest).on_save(
            trainer.args, trainer.state, trainer.control
        )
    if trainer.is_world_process_zero():
        atomic_json(
            paths["output"] / "TRAINING_COMPLETE.json",
            dict(step=trainer.state.global_step, contract=digest),
        )
