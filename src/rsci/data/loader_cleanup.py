"""训练结束时显式回收持久化DataLoader，不依赖解释器析构顺序。"""

import time
import warnings


def close_dataloaders(*loaders):
    """不创建新iterator；解开Accelerate包装，重复引用只关闭一次。"""
    seen_loaders, seen_iterators = set(), set()
    pending = list(loaders)
    closed = 0
    while pending:
        loader = pending.pop()
        if loader is None or id(loader) in seen_loaders:
            continue
        seen_loaders.add(id(loader))
        base = getattr(loader, "base_dataloader", None)
        if base is not None and base is not loader:
            pending.append(base)
            continue  # 包装器转发读取属性，赋值却不转发，必须清理真正的底层loader。
        iterator = getattr(loader, "_iterator", None)
        if iterator is None or id(iterator) in seen_iterators:
            continue
        seen_iterators.add(id(iterator))
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is None:
            continue  # 单进程或未使用的loader没有需要回收的worker。
        workers = list(getattr(iterator, "_workers", ()))
        try:
            # PyTorch没有公开的close接口；此方法按官方顺序关闭pin线程、队列及worker。
            shutdown()
        except Exception as error:
            warnings.warn(f"DataLoader退出清理异常：{error}", RuntimeWarning)
        finally:
            # PyTorch的异常回收可能仅terminate未join；只回收该iterator记录的子进程。
            deadline = time.monotonic() + 5
            for worker in workers:
                worker.join(timeout=max(0, deadline - time.monotonic()))
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=1)
                if worker.is_alive():
                    warnings.warn(f"DataLoader worker {worker.pid}尚未退出", RuntimeWarning)
            loader._iterator = None
        closed += 1
    return closed
