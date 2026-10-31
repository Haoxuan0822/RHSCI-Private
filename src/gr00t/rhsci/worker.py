"""gr00t: 单个训练worker的装配与执行。"""

import json

from rhsci.runtime.environment import get_env

from .config import build_training_arguments, optimization_kwargs
from .trainer import build_checkpoint_callback, create_trainer_class

# 中文说明：仅拆分职责；模型、优化器和断点格式由各骨干独立定义。


def run_training_worker(args, paths, processor, dataset, digest, *, validation_steps=0):
    """在当前分布式 worker 内装配模型、数据与 Trainer，恢复完整状态并执行训练。"""
    import torch
    import torch.distributed as dist
    from transformers import TrainerCallback, set_seed

    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
    from rhsci.runtime.numerics import reject_numerical_failure

    from .checkpoints import checkpoint_ok, latest_checkpoint
    from .memory import enable_dit_recomputation

    reject_numerical_failure(paths["output"])

    local_rank = int(get_env("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)
    config = json.loads((paths["checkpoint"] / "config.json").read_text())
    config.update(
        rsci_variant=args.model,
        rsci_ardy_weight=0.1,
        tune_llm=False,
        tune_visual=False,
        tune_top_llm_layers=4 if args.vlm == "top4" else 0,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        use_relative_action=False,
        load_bf16=True,
    )
    # 新增模块允许missing；原预训练骨干/AE任何键缺失都拒绝启动，不重置整个动作专家。
    model, loading = Gr00tN1d6.from_pretrained(
        str(paths["checkpoint"]),
        config=Gr00tN1d6Config(**config),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        output_loading_info=True,
    )
    if (
        any(not name.startswith("action_head.rsci_query.") for name in loading["missing_keys"])
        or loading["unexpected_keys"]
        or loading.get("mismatched_keys")
    ):
        raise ValueError(f"Unexpected pretrained loading mismatch: {loading}")
    # tune_llm=False 仍允许顶部4层；核验实际参数，避免误冻结或误解冻。
    layers = model.backbone.model.language_model.model.layers
    expected = (
        {id(p) for layer in layers[-4:] for p in layer.parameters()}
        if args.vlm == "top4"
        else set()
    )
    actual = {id(p) for p in model.backbone.parameters() if p.requires_grad}
    if actual != expected:
        raise ValueError("Backbone trainable set does not match top4/frozen recipe")
    # AE权重保留fp32以免小更新被bf16量化吞掉，前向仍使用bf16 autocast。
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    enable_dit_recomputation(model)
    world = len(args.gpus.split(","))
    resume = (
        None
        if validation_steps
        else args.resume or latest_checkpoint(paths["output"], world, digest)
    )
    if resume and (
        resume.resolve().parent != paths["output"].resolve()
        or not checkpoint_ok(resume, world, digest, verify_hash=True)
    ):
        raise ValueError("Resume checkpoint is incomplete, corrupt, or belongs to a different run")

    class MemoryValidationCallback(TrainerCallback):
        def on_train_begin(self, arguments, state, control, **kwargs):
            if not isinstance(trainer.model_wrapped, torch.nn.parallel.DistributedDataParallel):
                raise RuntimeError("Memory validation must use real multi-GPU DDP")
            if not trainer.model_wrapped.gradient_as_bucket_view:
                raise RuntimeError("DDP gradient bucket views did not take effect")

        def on_step_begin(self, arguments, state, control, **kwargs):
            import time

            self.started_at = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, arguments, state, control, **kwargs):
            import time

            from .memory import unwrap_zero

            torch.cuda.synchronize()
            optimizer = unwrap_zero(trainer.optimizer)
            local_state_bytes = sum(
                value.numel() * value.element_size()
                for parameter_state in optimizer.optim.state.values()
                for value in parameter_state.values()
                if isinstance(value, torch.Tensor)
            )
            print(
                "GR00T_MEMORY "
                + json.dumps(
                    dict(
                        rank=dist.get_rank(),
                        step=state.global_step,
                        vlm=args.vlm,
                        micro_batch=args.batch_size,
                        accumulation=args.gradient_accumulation_steps,
                        optimizer="zero1-adamw-fp32",
                        dit_recomputation=model.action_head.model.gradient_checkpointing,
                        local_optimizer_state_gib=local_state_bytes / 2**30,
                        update_seconds=time.perf_counter() - self.started_at,
                        allocated_gib=torch.cuda.memory_allocated() / 2**30,
                        reserved_gib=torch.cuda.memory_reserved() / 2**30,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                        gradient_as_bucket_view=trainer.model_wrapped.gradient_as_bucket_view,
                    )
                ),
                flush=True,
            )

    training = build_training_arguments(args, paths, validation_steps=validation_steps)
    if local_rank == 0:
        print(
            "GR00T_RSCI_OPTIMIZATION " + json.dumps(optimization_kwargs(), sort_keys=True),
            flush=True,
        )
        print(
            "GR00T_RSCI_SPEED: BF16 autocast; trainable FP32; TF32; native FlashAttention2; "
            f"ZeRO-1 fused AdamW; DiT layer recomputation; workers/rank={args.num_workers}; pin_memory; non_blocking; "
            f"persistent_workers={args.num_workers > 0}; "
            f"prefetch={2 if args.num_workers else 'none'}; current-frame VLM reuse; "
            f"micro_batch={args.batch_size}; accumulation={args.gradient_accumulation_steps}; "
            f"global_batch={args.batch_size * world * args.gradient_accumulation_steps}; "
            "DDP gradient bucket views; VLM features only",
            flush=True,
        )
    trainer = create_trainer_class()(
        model=model,
        args=training,
        train_dataset=dataset,
        data_collator=processor.collator,
        callbacks=[
            MemoryValidationCallback()
            if validation_steps
            else build_checkpoint_callback(args, paths, processor, world, digest)
        ],
    )
    # 自定义loss已按微批次归一化，由Trainer按累积次数缩放梯度；不使用语言token计数。
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if validation_steps:
        if trainer.state.global_step != validation_steps:
            raise RuntimeError("Memory validation did not complete all optimizer steps")
        print(
            f"GR00T_MEMORY_VALIDATION_OK rank={dist.get_rank()} steps={validation_steps}",
            flush=True,
        )
        return
    final = paths["output"] / f"checkpoint-{args.max_steps}"
    if trainer.is_world_process_zero() and not checkpoint_ok(final, world, digest):
        raise RuntimeError("Training returned without a complete final checkpoint")
    print(f"GR00T_RSCI_FINISHED {final}", flush=True)
