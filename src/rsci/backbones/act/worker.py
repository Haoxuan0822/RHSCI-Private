"""act: 单个训练worker的装配与执行。"""

import os

from .checkpoints import atomic_json, checkpoint_ok, latest
from .config import training_arguments
from .trainer import completion_callback, trainer_type

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def train(args, paths, value, digest):
    import torch
    from torch.utils.data import Subset

    from psi.utils import seed_everything

    from .data import ACTDataset
    from .model import RSCIACT

    world = len(args.gpus.split(","))
    if int(os.environ.get("WORLD_SIZE", 1)) != world:
        raise ValueError("Launch world size differs from training contract")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    seed_everything(args.seed)
    model = RSCIACT(args.model)
    dataset = ACTDataset(
        paths["dataset"],
        paths["sidecar"],
        args.model,
        args.seed,
        cache=paths["input_cache"],
    )
    # 原生ACT验证数据没有传no_aug=True，实际复用训练增强；这里保留这一行为。
    validation = ACTDataset(
        paths["dataset"],
        paths["sidecar"],
        args.model,
        args.seed,
        training=True,
        cache=paths["input_cache"],
    )
    # 上游验证按每rank最多20批；Subset避免中断prepared loader残留梯度状态。
    validation = Subset(validation, range(min(len(validation), 20 * 16 * world)))
    validation.processor = dataset.processor
    trainer = trainer_type()(
        model=model,
        args=training_arguments(args, paths),
        train_dataset=dataset,
        eval_dataset=validation,
        launch_contract=value,
        callbacks=[completion_callback(paths["output"], world, digest)],
    )
    resume = latest(paths["output"], world, digest)
    print(
        f"ACT_TRAIN resume={resume} global_batch={args.batch_size * world * args.gradient_accumulation_steps}",
        flush=True,
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
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
