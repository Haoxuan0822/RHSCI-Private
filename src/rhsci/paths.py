"""Dataset and output arguments shared by the LeRobot RHSCI adapters."""

import argparse
import re
import sys
from pathlib import Path
from typing import TypedDict

from rhsci.data.cache_config import InputCacheConfig
from rhsci.runtime.environment import get_env

# 代码根由共用模块定位，不依赖各骨干原生包不同的目录深度。
CODE_ROOT = Path(__file__).resolve().parents[2]
# 仅登记模块位置，不在共享工具导入时加载任何模型或可选依赖。
BACKBONE_MODULES = {
    "psi0": "psi.rhsci",
    "pi05": "openpi.rhsci",
    "gr00t": "gr00t.rhsci",
    "egovla": "human_plan.rhsci",
    "act": "act.rhsci",
    "lambda0": "lambda0.rhsci",
}


class DatasetPaths(TypedDict):
    """解析后的数据身份；STATE/NO-STATE 的 sidecar 可以为空。"""

    task: str
    dataset: Path
    sidecar: Path | None


class TrainingPaths(DatasetPaths):
    """四个 HF 骨干共用的数据、缓存选择及运行目录。"""

    input_cache: InputCacheConfig
    output: Path


class CheckpointTrainingPaths(TrainingPaths):
    """GR00T/EgoVLA 另需的初始化权重目录；不改变字典存取方式。"""

    checkpoint: Path


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """向调用方的解析器添加公共路径选项；不解析参数或访问数据。"""
    parser.add_argument(
        "--dataset", type=Path, help="LeRobot dataset directory containing meta/, data/ and videos/"
    )
    parser.add_argument(
        "--sidecar", type=Path, help="Formal ARDY sidecar directory for query variants"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Run directory; incompatible existing runs are rejected"
    )


def dataset_paths(args: argparse.Namespace) -> DatasetPaths:
    """返回解析后的 task/dataset/sidecar 字典；不修改参数、不读取数据文件。"""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.task):
        raise ValueError("task must be a plain identifier")
    dataset = getattr(args, "dataset", None)
    sidecar = getattr(args, "sidecar", None)
    if dataset is None:
        raise ValueError("Provide --dataset; task names do not select private dataset paths")
    if args.model not in ("STATE", "NO-STATE") and sidecar is None:
        raise ValueError("Query datasets require --sidecar")
    return DatasetPaths(
        task=args.task, dataset=dataset.resolve(), sidecar=sidecar.resolve() if sidecar else None
    )


def with_nccl(command, library=None, *, expected_version="2.31.2"):
    """Use the environment runtime unless an alternate NCCL library is explicit."""
    library = library or get_env("RHSCI_NCCL_LIBRARY")
    if not library:
        return command
    return [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts/with_nccl_runtime.py"),
        "--library",
        str(library),
        "--expected-version",
        get_env("RHSCI_NCCL_VERSION", expected_version),
        "--",
        *command,
    ]


def worker_arguments(args, argument_parser):
    """Forward parsed training options while keeping guardian/control flags private."""
    excluded = {
        "help",
        "worker",
        "guardian",
        "guardian_fd",
        "check",
        "check_data",
        "dry_run",
        "resume",
    }
    result = []
    for action in argument_parser._actions:
        value = getattr(args, action.dest, None)
        if action.dest in excluded or value is None or not action.option_strings:
            continue
        # 无参const选项可能与带值选项共用dest；仅转发实际匹配的选项。
        if action.nargs == 0 and not isinstance(value, bool):
            if getattr(action, "const", None) == value:
                result.append(action.option_strings[0])
            continue
        if isinstance(value, bool):
            if value:
                result.append(action.option_strings[0])
        elif isinstance(value, (list, tuple)):
            result += [action.option_strings[0], *map(str, value)]
        else:
            result += [action.option_strings[0], str(value)]
    return result
