"""固定来源与本地任务；不触发依赖导入或下载。"""

import json
import os
from pathlib import Path

from rsci.paths import CODE_ROOT
from rsci.runtime.artifacts import sha256

UPSTREAM_COMMIT = "858b3837d42946ed90291d8da7ce6ce720335cac"
CODE = CODE_ROOT
UPSTREAM = CODE / "src/openlambda"
RUNTIME = Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data"))

CONTRACT = "lambda0-rsci-reader-current-v1"
SAFE_NORMALIZATION = "rsci_q01_q99_degenerate_minmax_v1"


def verify_upstream():
    # 源码随单一仓库发布，用完整文件集合及摘要核验固定来源，不依赖嵌套 Git。
    manifest = json.loads((UPSTREAM / "SOURCE_MANIFEST.json").read_text())
    if manifest["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("λ₀ source revision mismatch")
    # 包已平铺到 src/lambda0；清单保留上游路径，并显式记录其当前目录。
    trees = [
        ("", UPSTREAM),
        *[
            (prefix + "/", UPSTREAM / relative)
            for prefix, relative in manifest.get("relocated_roots", {}).items()
        ],
    ]
    actual = {
        prefix + str(path.relative_to(root)): sha256(path)
        for prefix, root in trees
        for path in root.rglob("*")
        if path.is_file()
        and not any(
            part in {".git", "__pycache__", ".pytest_cache"}
            for part in path.relative_to(root).parts
        )
        and path.name != "SOURCE_MANIFEST.json"
        and path.suffix != ".pyc"
    }
    if actual != manifest["files"]:
        raise ValueError("λ₀ source file set or SHA256 differs from the reviewed snapshot")
    return manifest["upstream_commit"]
