"""pi05: 单个训练worker的装配与执行。"""

from rhsci.runtime.environment import get_env

from .checkpoints import checkpoint_ok
from .config import build_training_arguments
from .contract import build_training_contract
from .preflight import validate_run
from .trainer import build_checkpoint_callback, create_trainer_class

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def run_training_worker(args, paths, processor, dataset, digest):
    """在当前分布式 worker 内装配模型、数据与 Trainer，恢复完整状态并执行训练。"""
    import torch
    from transformers import set_seed

    from .model import Pi05RHSCI
    from .processing import collate

    rank = int(get_env("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)
    model = Pi05RHSCI(
        args.model,
        execution=args.execution,
        state_measurement_noise_std_rad=args.state_measurement_noise_std_rad,
    )
    model.load_initial(paths["checkpoint"])
    model.rsci_config["normalization"] = processor.normalization
    if hasattr(processor, "normalization_ranges"):
        model.rsci_config["normalization_ranges"] = processor.normalization_ranges
    world = len(args.gpus.split(","))
    identity = build_training_contract(args, paths)[0]
    resume = validate_run(args, paths, identity, digest, allow_started=True)
    training = build_training_arguments(args, paths)
    trainer = create_trainer_class()(
        model=model,
        args=training,
        train_dataset=dataset,
        data_collator=collate,
        callbacks=[build_checkpoint_callback(paths["output"], world, digest)],
    )
    print(
        f"PI05_SPEED BF16/FP32-trainable; SDPA; layer-checkpoint; frozen-VLM-KV; fused-AdamW; "
        f"TF32; pin/nonblocking/persistent; DDP bucket views; global_batch={args.batch_size * world * args.gradient_accumulation_steps}",
        flush=True,
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if trainer.is_world_process_zero() and not checkpoint_ok(
        paths["output"] / f"checkpoint-{args.max_steps}", world, digest
    ):
        raise RuntimeError("Missing complete final checkpoint")
