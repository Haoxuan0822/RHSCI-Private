"""动作采样的精度契约；网络混合精度与积分状态精度分别控制。"""

from diffusers import FlowMatchEulerDiscreteScheduler


def step_action_scheduler(scheduler, *, model_output, timestep, sample, **kwargs):
    """Flow速度和累积状态保持FP32，其他scheduler保留原输入与返回契约。"""
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        model_output = model_output.float()
        sample = sample.float()
    return scheduler.step(model_output=model_output, timestep=timestep, sample=sample, **kwargs)
