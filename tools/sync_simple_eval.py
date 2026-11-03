"""下载固定版本的 SIMPLE 评测资产；新旧发布分别保存，不覆盖历史实验。"""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST = Path(__file__).resolve().parents[1] / "configs/simple/eval_assets.json"


def matches_record(path, record):
    """按 HF LFS SHA256 或普通 Git blob SHA1 核对本地文件。"""
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    digest = hashlib.sha256() if "sha256" in record else hashlib.sha1()
    if "git_blob_sha1" in record:
        digest.update(f"blob {record['bytes']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == record.get("sha256", record.get("git_blob_sha1"))


def fetch_file(url, destination, record):
    """校验后原子发布下载文件，失败时移除本次临时文件。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        for attempt in range(3):
            try:
                with (
                    urllib.request.urlopen(url, timeout=90) as response,
                    temporary.open("wb") as out,
                ):
                    shutil.copyfileobj(response, out)
                if not matches_record(temporary, record):
                    raise ValueError(f"download checksum mismatch: {destination.name}")
                temporary.replace(destination)
                return
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    finally:
        temporary.unlink(missing_ok=True)


def extract_task(archive, destination, verify_only=False):
    """核对 ZIP 全部成员并在临时目录解包；已存在的任务必须逐文件一致。"""
    task = archive.stem
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != task
                or (member.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError(f"unexpected ZIP member: {member.filename}")
        if (destination / task).exists() or verify_only:
            for member in source.infolist():
                if member.is_dir():
                    continue
                path = destination / member.filename
                expected = source.read(member)  # ZIP CRC 也在此验证。
                if not path.is_file() or path.read_bytes() != expected:
                    raise ValueError(f"extracted file differs or is missing: {path}")
            return
        with tempfile.TemporaryDirectory(dir=destination) as directory:
            source.extractall(directory)
            (Path(directory) / task).rename(destination / task)


def sync_release(manifest, release_name, output_root, tasks=(), verify_only=False, workers=8):
    """同步一个独立发布；新版解压文件可复用旧 ZIP 内内容相同的文件。"""
    release = manifest["releases"][release_name]
    destination = output_root / release_name
    selected = {
        name: record
        for name, record in release["files"].items()
        if not tasks or name.split("/")[0].removesuffix(".zip") in tasks or "/" not in name
    }
    if tasks:
        available = {name.split("/")[0].removesuffix(".zip") for name in release["files"]}
        missing = set(tasks) - available
        if missing:
            raise ValueError(f"unknown tasks: {sorted(missing)}")
        if release["archives"]:
            selected = {name: rec for name, rec in selected.items() if Path(name).stem in tasks}
    if not verify_only:
        destination.mkdir(parents=True, exist_ok=True)

    def sync_one(item):
        name, record = item
        path = destination / ("archives" if release["archives"] else "") / name
        if not matches_record(path, record):
            if verify_only or path.exists():
                raise ValueError(f"file differs or is missing: {path}")
            cached = output_root / "zip-38e43f22" / name
            if not release["archives"] and matches_record(cached, record):
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached, path)
            else:
                url = (
                    f"https://huggingface.co/datasets/{release['repo']}/resolve/"
                    f"{release['revision']}/{release['prefix']}{name}"
                )
                fetch_file(url, path, record)
        if release["archives"]:
            extract_task(path, destination, verify_only)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(sync_one, selected.items()))
    return {"release": release_name, "files_verified": len(selected), "directory": str(destination)}


def main():
    manifest = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--release", choices=list(manifest["releases"]), default="snapshot-fb67cd9a"
    )
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            sync_release(manifest, args.release, args.output_root, args.task, args.verify_only)
        )
    )


if __name__ == "__main__":
    main()
