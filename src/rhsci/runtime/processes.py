"""Scoped child-process cleanup for training launchers."""

import logging
import os
import signal
import subprocess
import time
from pathlib import Path

LOG = logging.getLogger(__name__)


def _signal_worker_group(process, signum):
    if process is not None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


class _WorkerSignalHandler:
    """在创建子进程前接管停止信号，直至退出清理完成；不在信号回调里 wait。"""

    def __init__(self):
        self.process = None
        self.requested = None

    def stop(self, signum, frame):
        self.requested = signum
        _signal_worker_group(self.process, signal.SIGTERM)

    def __enter__(self):
        self.previous = {
            sig: signal.signal(sig, self.stop) for sig in (signal.SIGTERM, signal.SIGINT)
        }
        return self

    def __exit__(self, *exception):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


def _terminate_worker(process):
    """异常路径也等待直接子进程退出，超时后升级为 SIGKILL。"""
    _signal_worker_group(process, signal.SIGTERM)
    if process.poll() is None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _signal_worker_group(process, signal.SIGKILL)
            process.wait()


def run_worker_process(command, *, on_start=None, on_wait=None, on_exit=None, **options):
    """启动独立 worker 进程组；转发终止信号、回收子进程，返回退出码。

    on_start(process) 可记录子进程身份；on_wait() 在等待间隙读取日志或观察子孙。
    on_exit(process) 在回收直接子进程后清理模型专属资源/子孙。
    信号处理覆盖启动回调与退出清理；回调异常也清理，不另建监控进程。
    """
    with _WorkerSignalHandler() as termination:
        process = subprocess.Popen(command, start_new_session=True, **options)
        termination.process = process
        # 信号可能在 Popen 返回前到达；拿到子进程身份后补发。
        if termination.requested is not None:
            _signal_worker_group(process, signal.SIGTERM)
        try:
            if on_start is not None:
                on_start(process)
            result = _wait_worker(process, termination, grace=30, on_wait=on_wait)
        finally:
            try:
                _terminate_worker(process)
            finally:
                if on_exit is not None:
                    on_exit(process)
        if termination.requested is not None:
            raise SystemExit(128 + termination.requested)
        return result


def wait_worker(process, grace=30, *, on_wait=None):
    """Wait for a session-leading worker; forward signals without reentering Popen.wait."""
    with _WorkerSignalHandler() as termination:
        termination.process = process
        if termination.requested is not None:
            _signal_worker_group(process, signal.SIGTERM)
        return _wait_worker(process, termination, grace=grace, on_wait=on_wait)


def _wait_worker(process, termination, *, grace, on_wait):
    while True:
        if on_wait is not None:
            on_wait()
        try:
            result = process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            if termination.requested is None:
                continue
            try:
                result = process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                _signal_worker_group(process, signal.SIGKILL)
                result = process.wait()
        if termination.requested is not None:
            raise SystemExit(128 + termination.requested)
        return result


def process_table():
    """使用启动 tick 防 PID 重用；不根据 python 名称批量杀进程。"""
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            result[int(entry.name)] = (int(stat[1]), int(stat[2]), int(stat[19]), stat[0])
        except (OSError, ValueError, IndexError):
            continue
    return result


def remember_tree(root, tracked, table, own_group=None):
    """记录子孙身份，主进程退出后仍可识别此前发现的孤儿 worker。"""
    known = {pid for pid, birth in tracked.items() if pid in table and table[pid][2] == birth}
    if root in table and root in known:
        known.add(root)
    changed = True
    while changed:
        changed = False
        for pid, (parent, group, birth, state) in table.items():
            if state != "Z" and (parent in known or (own_group is not None and group == own_group)):
                if pid not in known:
                    known.add(pid)
                    changed = True
                tracked[pid] = birth


def survivors(tracked):
    table = process_table()
    return [
        pid
        for pid, birth in tracked.items()
        if pid in table and table[pid][2] == birth and table[pid][3] != "Z"
    ]


def cleanup(tracked, grace=10):
    """结束 CUDA 所属进程才能释放其显存；不用 GPU reset/empty_cache/全局 pkill。"""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        live = survivors(tracked)
        if not live:
            return
        LOG.warning("清理已确认的本轮残留 PID=%s signal=%s", live, sig.name)
        for pid in live:
            # 发信号前再次核对启动时间，不能误伤复用的 PID。
            if pid in survivors({pid: tracked[pid]}):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + grace
        while survivors(tracked) and time.monotonic() < deadline:
            time.sleep(0.2)
    if survivors(tracked):
        raise RuntimeError("本轮进程仍未退出，拒绝叠加启动新训练")
