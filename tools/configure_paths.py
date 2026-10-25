"""将公开快照中的路径机械替换为当前机器路径；默认只预览。"""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    state_path = root / ".rsci-local-paths.json"
    old = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"code": "/opt/rsci", "data": "/opt/rsci-data"}
    )
    if not args.reset and args.data_root is None:
        parser.error("--data-root is required unless --reset is set")
    new = (
        {"code": "/opt/rsci", "data": "/opt/rsci-data"}
        if args.reset
        else {"code": str(root), "data": str(args.data_root.expanduser().resolve())}
    )
    # 路径可出现在 Python/JSON/Shell 字面量中，拒绝需要额外转义的字符。
    for value in new.values():
        if any(c in value for c in " \t\r\n\"'`$\\"):
            parser.error("paths must not contain spaces, quotes or shell metacharacters")
    if new["code"] == new["data"] or root == Path(new["data"]) or root in Path(new["data"]).parents:
        parser.error("data must be outside the source repository")
    manifest = json.loads((root / "PUBLIC_EXPORT_MANIFEST.json").read_text())
    names = set(manifest["files"]) - {"tools/configure_paths.py"}
    changed = 0
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            continue
        try:
            original = path.read_text()
        except UnicodeDecodeError:
            continue
        content = original
        # 先替换更长的数据路径，避免代码路径前缀吞掉 /opt/rsci-data。
        pairs = dict((old[k], new[k]) for k in ("data", "code"))
        pattern = re.compile(
            "|".join(re.escape(value) for value in sorted(pairs, key=len, reverse=True))
        )
        content = pattern.sub(lambda match: pairs[match.group(0)], content)
        if content != original:
            changed += 1
            if args.apply:
                path.write_text(content)
    if args.apply:
        state_path.write_text(json.dumps(new, indent=2) + "\n")
    print(
        json.dumps(
            {"apply": args.apply, "changed_files": changed, "paths": new}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
