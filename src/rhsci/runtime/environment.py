"""RHSCI 环境变量读取；旧 RSCI 名称仅作为兼容输入。"""

import os


def get_env(name, default=None):
    """优先读取 RHSCI 名称，未设置时兼容旧 RSCI；不修改进程环境。"""
    canonical = name if "RHSCI" in name else name.replace("RSCI", "RHSCI")
    legacy = canonical.replace("RHSCI", "RSCI")
    return os.environ.get(canonical, os.environ.get(legacy, default))
