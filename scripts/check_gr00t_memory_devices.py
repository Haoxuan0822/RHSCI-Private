"""只读短测准入：拒绝挤占已有训练/评测，不自动停止任何进程。"""

import os
import re
import subprocess


def require_idle_devices(gpus=None):
    gpus = gpus or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    if not re.fullmatch(r"\d+(,\d+)*", gpus) or len(set(gpus.split(","))) != len(gpus.split(",")):
        raise ValueError("GPU indices must be distinct integers")
    selected = {int(index) for index in gpus.split(",")}
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    busy = []
    available = set()
    for row in rows.strip().splitlines():
        index, used = (int(value.strip()) for value in row.split(","))
        available.add(index)
        if index in selected and used > 1024:
            busy.append(f"GPU{index}={used}MiB")
    if not selected <= available:
        raise RuntimeError("Selected GPUs are unavailable")
    if busy:
        raise RuntimeError(
            "GR00T short check requires idle selected GPUs; preserved existing tasks: "
            + ", ".join(busy)
        )


def require_fresh_output():
    import argparse
    import os
    from pathlib import Path

    from rsci.backbones.gr00t.train import parser, resolve

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--steps", type=int, default=6)
    _, rest = probe.parse_known_args()
    args = parser().parse_args(rest)
    resolve(args)
    root = Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data"))
    output = args.output_dir or root / "validation/gr00t" / args.run_tag / args.task / args.vlm
    # 在torchrun之前检查，避免八rank各自检查目录造成创建竞态。
    if output.exists():
        raise ValueError("Validation directory already exists; use a new run-tag")


if __name__ == "__main__":
    import argparse

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--gpus")
    selected, _ = probe.parse_known_args()
    require_idle_devices(selected.gpus)
    require_fresh_output()
