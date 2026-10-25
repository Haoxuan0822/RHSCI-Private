"""检查公开源码中是否混入凭据、资产、外部链接或嵌套 Git 引用。"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    secrets = re.compile(
        rb"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})\b"
    )
    forbidden = {
        ".pt",
        ".pth",
        ".bin",
        ".safetensors",
        ".ckpt",
        ".onnx",
        ".parquet",
        ".arrow",
        ".mp4",
        ".avi",
        ".whl",
        ".pyc",
        ".so",
    }
    errors = []
    for component in ("src/gr00t", "src/egovla/VILA", "src/dreamzero"):
        try:
            provenance = json.loads((root / component / "SOURCE_MANIFEST.json").read_text())
            for name in provenance["license_source"]["files"]:
                path = root / component / name
                if (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    != provenance["files"][name]["sha256"]
                ):
                    errors.append(
                        f"{component}/{name}: component license differs from its source record"
                    )
        except (OSError, ValueError, KeyError, TypeError):
            errors.append(f"{component}: missing or invalid component license/source record")
    manifest = json.loads((root / "PUBLIC_EXPORT_MANIFEST.json").read_text())
    for name, record in manifest["files"].items():
        path = root / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["export_sha256"]
        ):
            errors.append(name + ": differs from the public source manifest")
    count = total = 0
    actual_sources = set()
    for current, dirs, files in os.walk(root):
        relative = Path(current).relative_to(root)
        for name in list(dirs):
            p = Path(current) / name
            if name == ".git":
                if relative != Path("."):
                    errors.append(str(p.relative_to(root)) + ": nested Git")
                dirs.remove(name)
            elif p.is_symlink():
                errors.append(str(p.relative_to(root)) + ": symlink")
                dirs.remove(name)
        for name in files:
            path = Path(current) / name
            label = str(path.relative_to(root))
            if label == ".rsci-local-paths.json":
                continue
            if label != "PUBLIC_EXPORT_MANIFEST.json":
                actual_sources.add(label)
            if path.is_symlink():
                errors.append(label + ": symlink")
                continue
            raw = path.read_bytes()
            count += 1
            total += len(raw)
            if name == ".git" or path.suffix.lower() in forbidden:
                errors.append(label + ": Git reference or runtime asset")
            if name.startswith(".env") and name not in {
                ".env.sample",
                ".env.example",
                ".env.aibox.example",
            }:
                errors.append(label + ": private environment file")
            if len(raw) >= 50_000_000:
                errors.append(label + ": file exceeds 50 MB")
            if (
                secrets.search(raw)
                or (b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----") in raw
                or (b"-----BEGIN " + b"RSA PRIVATE KEY-----") in raw
            ):
                errors.append(label + ": possible credential (content withheld)")
            if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
                errors.append(label + ": LFS asset pointer")
    if actual_sources != set(manifest["files"]):
        errors.append("source file inventory differs from the public source manifest")
    if (root / ".git").exists():
        index = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage"]).decode()
        if any(line.startswith("160000 ") for line in index.splitlines()):
            errors.append("index contains submodules/gitlinks")
        worktrees = subprocess.check_output(
            ["git", "-C", str(root), "worktree", "list", "--porcelain"]
        ).decode()
        if sum(line.startswith("worktree ") for line in worktrees.splitlines()) != 1:
            errors.append("repository must have exactly one worktree")
        if (root / ".git/objects/info/alternates").exists():
            errors.append("repository shares external Git objects")
    print(
        json.dumps({"files": count, "bytes": total, "errors": errors}, ensure_ascii=False, indent=2)
    )
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
