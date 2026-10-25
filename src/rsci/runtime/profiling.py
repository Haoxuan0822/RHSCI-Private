"""不改变模型权重结构的训练执行优化。"""

import time

import torch


def timed_batches(loader):
    """测量next含worker等待/搬运的主机时间，迭代器启动归入首批。"""
    started = time.perf_counter()
    iterator = iter(loader)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            return
        yield batch, time.perf_counter() - started
        started = time.perf_counter()


class StageTimer:
    """CPU wall time与CUDA stream事件；不强制同步，避免改变流水线。"""

    def __init__(self, enabled, device):
        self.enabled = enabled
        self.cuda = enabled and torch.device(device).type == "cuda"
        self.stages = {}

    def start(self, name):
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True) if self.cuda else None
        if event is not None:
            event.record()
        self.stages[name] = [time.perf_counter(), event, None, None]

    def end(self, name):
        if not self.enabled:
            return
        stage = self.stages[name]
        stage[2] = time.perf_counter() - stage[0]
        if self.cuda:
            stage[3] = torch.cuda.Event(enable_timing=True)
            stage[3].record()

    def metrics(self):
        metrics = {}
        for name, (_, begin, wall, end) in self.stages.items():
            if wall is not None:
                metrics[f"timing/{name}_host_s"] = wall
            if end is not None and end.query():
                metrics[f"timing/{name}_gpu_s"] = begin.elapsed_time(end) / 1000
        return metrics
