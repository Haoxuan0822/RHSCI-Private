"""HF 保存回调的同步、完整断点发布和轮换。"""

from pathlib import Path

from rhsci.runtime.artifacts import atomic_json


def build_checkpoint_callback(output, world, digest, *, protocol):
    """返回 ACT/EgoVLA 共用回调；protocol 提供各自的 publish/prune 实现。"""
    import torch.distributed as dist
    from transformers import TrainerCallback

    output = Path(output)

    class Complete(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if dist.is_initialized():
                dist.barrier()
            if state.is_world_process_zero:
                target = output / f"checkpoint-{state.global_step}"
                try:
                    protocol.publish(target, world, digest)
                    protocol.prune(output, target, world, digest)
                except FloatingPointError:
                    atomic_json(
                        output / "NUMERICAL_FAILURE.json",
                        dict(step=state.global_step, stage="checkpoint"),
                    )
                    raise
            if dist.is_initialized():
                dist.barrier()

    return Complete()
