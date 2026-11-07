"""原始输入缓存的显式配置；环境变量仅在命令解析边界兼容读取。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rhsci.runtime.environment import get_env


@dataclass(frozen=True)
class InputCacheConfig:
    """root为空时禁用；full核验全部SHA，metadata保留worker清单/stat门禁。"""

    root: Path | None = None
    verification: Literal["full", "metadata"] = "full"

    def __post_init__(self):
        if self.verification not in ("full", "metadata"):
            raise ValueError("Unknown input-cache verification mode")
        if self.root is not None:
            object.__setattr__(self, "root", Path(self.root).resolve())


def validate_cache_dataset_names(roots):
    """缓存沿用 <父目录>/<数据集名>，不同源不能映射到同一子目录。"""
    names = {}
    for root in roots:
        root = Path(root).resolve()
        previous = names.setdefault(root.name, root)
        if previous != root:
            raise ValueError(
                f"Input cache dataset-name collision: {previous} and {root}; "
                "use distinct dataset directory names"
            )


def add_input_cache_arguments(parser, *, env_var):
    # 只在入口读取旧变量；数据集/完整性工具不读取进程环境。
    options = parser.add_mutually_exclusive_group()
    options.add_argument(
        "--input-cache-root",
        type=Path,
        default=get_env(env_var) or None,
        help=f"原始uint8/小张量缓存根目录；兼容默认变量{env_var}",
    )
    options.add_argument(
        "--no-input-cache",
        dest="input_cache_root",
        action="store_const",
        const=None,
        help="显式禁用输入缓存，覆盖环境变量",
    )


def resolve_input_cache(args):
    # 角色显式决定验收级别，单卡worker也无需重复全量SHA；WORLD_SIZE不参与。
    return InputCacheConfig(
        args.input_cache_root, verification="metadata" if args.worker else "full"
    )


def cache_arguments(config):
    """把已解析的选择传给子进程，避免其重新依赖环境默认值。"""
    return ["--input-cache-root", str(config.root)] if config.root else ["--no-input-cache"]
