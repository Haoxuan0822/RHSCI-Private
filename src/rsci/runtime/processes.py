"""Scoped child-process cleanup for training launchers."""

import logging
import os
import signal
import subprocess
import time
from pathlib import Path

LOG = logging.getLogger(__name__)


def wait_worker(process, grace=30):
    """Wait for a session-leading worker; forward signals without reentering Popen.wait."""
    requested = None

    def signal_group(signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    def stop(signum, frame):
        nonlocal requested
        requested = signum
        signal_group(signal.SIGTERM)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while True:
            try:
                result = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if requested is None:
                    continue
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    signal_group(signal.SIGKILL)
                    process.wait()
            if requested is not None:
                raise SystemExit(128 + requested)
            return result
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


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
