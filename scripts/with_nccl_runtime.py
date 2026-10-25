"""仅为子命令选择 NCCL，不修改虚拟环境和训练算法。"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--expected-version", default="2.31.2")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a child command is required after --")
    try:
        library = str(args.library.resolve(strict=True))
    except OSError as error:
        parser.error(str(error))
    # 动态加载器将空白和冒号视为多个库，不能静默加载错误路径。
    if any(c.isspace() or c == ":" for c in library):
        parser.error("LD_PRELOAD library path must not contain whitespace or ':'")
    env = os.environ.copy()
    env["LD_PRELOAD"] = library + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    # 新进程先加载库再导入 torch；检查 torch 实际调用的版本，不能相信 NCCL_VERSION 环境变量。
    probe = """
import ctypes, os, sys, torch
major, minor, patch = map(int, sys.argv[1].split('.'))
expected = major * 10000 + minor * 100 + patch
version = ctypes.c_int()
get_version = ctypes.CDLL(None).ncclGetVersion
get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
get_version.restype = ctypes.c_int
if get_version(ctypes.byref(version)) != 0:
    raise SystemExit('ncclGetVersion failed')
actual = version.value
mapped = {line.split()[-1] for line in open('/proc/self/maps') if 'libnccl.so' in line}
selected = os.path.realpath(sys.argv[2])
if actual != expected or selected not in {os.path.realpath(p) for p in mapped} or len(mapped) != 1:
    raise SystemExit(f'NCCL runtime rejected: actual={actual}, expected={expected}, loaded={sorted(mapped)}')
print(f'NCCL runtime verified: torch={torch.__version__} runtime={actual} torch_build={torch.cuda.nccl.version()} library={selected}', flush=True)
"""
    result = subprocess.run([sys.executable, "-c", probe, args.expected_version, library], env=env)
    if result.returncode:
        raise SystemExit(result.returncode)
    # 不改协议、buffer 广播、梯度规约、数据顺序或随机种子；由调用方显式选择命令。
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
