"""Restore exact legacy HSSD asset aliases without rewriting official USD files."""

import json
import os
import re
import time
from pathlib import Path

from pxr import Sdf

ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
records = []
for usd in (ROOT / "external/SIMPLE-official/data/scenes/hssd").glob("*/*.usd"):
    # USD crate 压缩字符串，必须通过 USD API 读取引用并解析相对路径。
    layer = Sdf.Layer.FindOrOpen(str(usd))
    content = "\n".join(
        os.path.abspath(usd.parent / reference) for reference in layer.GetExternalReferences()
    )
    aliases = set(re.findall(r"(/root/tmp|/tmp)/([a-zA-Z0-9]+)/(?:props|textures)/", content))
    # props 内的材质还通过 ../textures 引用纹理，两个目录必须同时恢复。
    for prefix, uid, folder in (
        (prefix, uid, folder) for prefix, uid in aliases for folder in ("props", "textures")
    ):
        target = usd.parent / folder
        if not target.is_dir():
            raise FileNotFoundError(target)
        alias = Path(prefix) / uid / folder
        preserved = None
        if alias.is_symlink() and not alias.exists():
            preserved = alias.with_name(folder + ".before-rhsci-restore-" + str(time.time_ns()))
            old_target = os.readlink(alias)
            alias.rename(preserved)
            print("PRESERVED_BROKEN_ALIAS", alias, old_target, preserved, flush=True)
        if not alias.exists():
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.symlink_to(target, target_is_directory=True)
            print("RESTORED_USD_ALIAS", alias, target, flush=True)
        records.append(
            dict(
                usd=str(usd),
                alias=str(alias),
                target=str(alias.resolve()),
                preserved=str(preserved) if preserved else None,
            )
        )
(ROOT / "logs/pi05-between-usd-aliases.json").write_text(json.dumps(records, indent=2) + "\n")
print("USD_ALIASES_CHECKED", len(records), flush=True)
