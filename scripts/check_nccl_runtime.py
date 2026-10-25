"""独立通信验收，不加载模型或启动训练。

用 torchrun 启动，并由调用方用 timeout 限制整个进程组的运行时间。
"""

import argparse
import ctypes
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class BufferBroadcastProbe(torch.nn.Module):
    """复现控制器的 34 个 int64 状态及混合 dtype，保留默认每次前向广播。"""

    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float64, device=device))
        for name, count in (("version", 1), ("updates", 1), ("closed", 16), ("recovery", 16)):
            self.register_buffer(name, torch.zeros(count, dtype=torch.int64, device=device))
        self.register_buffer("enabled", torch.tensor(True, device=device))
        self.register_buffer("coefficient", torch.tensor(0.01, dtype=torch.float64, device=device))

    def forward(self, value):
        return self.weight * value


def check_ddp(device, iterations):
    rank, world = dist.get_rank(), dist.get_world_size()
    model = BufferBroadcastProbe(device)
    ddp = DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None
    )
    reference = torch.tensor([0.25], dtype=torch.float64, device=device)
    # 等量样本的全局均值参考；每轮故意让非零 rank buffer 不同，验证广播没有被跳过。
    mean_square = world * (world + 1) * (2 * world + 1) / (6 * world)
    mean_input = (world + 1) / 2
    for step in range(iterations):
        with torch.no_grad():
            for buffer in model.buffers():
                buffer.fill_(step + 1 if rank == 0 else 0)
        ddp.zero_grad(set_to_none=True)
        output = ddp(torch.tensor([rank + 1.0], dtype=torch.float64, device=device))
        for buffer in model.buffers():
            expected = bool(step + 1) if buffer.dtype == torch.bool else step + 1
            assert torch.all(buffer == expected).item(), (rank, step, buffer)
        # 交替目标保持非零梯度，避免长测后参数衰减到零导致数值验收失去判别力。
        target = 1.0 if step % 2 else -1.0
        ((output - target).square().sum() / 2).backward()
        expected_gradient = reference * mean_square - target * mean_input
        torch.testing.assert_close(model.weight.grad, expected_gradient, rtol=1e-12, atol=1e-12)
        with torch.no_grad():
            model.weight.add_(model.weight.grad, alpha=-0.01)
            reference.add_(expected_gradient, alpha=-0.01)
        torch.testing.assert_close(model.weight, reference, rtol=1e-12, atol=1e-12)
        if rank == 0 and (step + 1) % 1000 == 0:
            print(f"PASS DDP mixed-buffer broadcasts + gradient/update step={step + 1}", flush=True)
    if rank == 0:
        print(
            f"PASS DDP mixed-buffer broadcasts + gradient/update iterations={iterations}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--ddp-steps", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1 or args.ddp_steps < 1:
        parser.error("iterations and ddp-steps must be positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank) if args.backend == "nccl" else torch.device("cpu")
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
    version = "not used (CPU Gloo)"
    if args.backend == "nccl":
        # torch.cuda.nccl.version() 返回构建版本；预加载替换后必须查询运行库而非头文件版本。
        runtime_version = ctypes.c_int()
        library = ctypes.CDLL("libnccl.so.2")
        library.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.ncclGetVersion.restype = ctypes.c_int
        if library.ncclGetVersion(ctypes.byref(runtime_version)) != 0:
            raise RuntimeError("ncclGetVersion failed")
        version = f"runtime={runtime_version.value}, torch_build={torch.cuda.nccl.version()}"
    print(
        f"rank={local_rank} torch={torch.__version__} NCCL={version} "
        f"protocol={os.environ.get('NCCL_PROTO', 'auto')}",
        flush=True,
    )
    dist.init_process_group(args.backend, timeout=timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        # 覆盖小消息、512 KiB 阈值和较大 DDP 同步消息；检查结果而非只检查不崩溃。
        for count in (1, 65536, 131072, 4194304):
            for iteration in range(args.iterations):
                value = torch.full((count,), rank + 1.0, device=device, dtype=torch.float32)
                print(
                    f"rank={rank} all_reduce bytes={count * 4} iteration={iteration} START",
                    flush=True,
                )
                dist.all_reduce(value)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                assert torch.all(value == world * (world + 1) / 2).item()
                value.fill_(rank + 1.0)
                dist.broadcast(value, src=0)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                assert torch.all(value == 1).item()
                del value
            if rank == 0:
                print(f"PASS all_reduce+broadcast bytes={count * 4}", flush=True)
        check_ddp(device, args.ddp_steps)
        dist.barrier()
        print(f"rank={rank} ALL TESTS PASSED", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
