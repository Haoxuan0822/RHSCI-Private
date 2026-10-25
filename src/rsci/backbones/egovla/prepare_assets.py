"""下载SIMPLE quick-start指定的固定基础VLM，校验后独立解压，不启动训练。"""

import argparse
import os
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path

from .checkpoints import atomic_json, sha256
from .train import assets

REPO = "rchal97/egovla_base_vlm"
REVISION = "59a3400683ae07a77e348ac18153238a760d37b4"
FOLDER = "vila-qwen2-vl-1.5b-instruct-sft-20240830191953"
SIZE = 3152865216
SHA256 = "359163fdc2022e67857c953c2b2ec94556ce2ceb87a89cd56c3e4e1c94075e8d"


def main():
    import requests

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data")),
    )
    args = p.parse_args()
    base = args.runtime_root.resolve() / "cache/checkpoints/EgoVLA"
    base.mkdir(parents=True, exist_ok=True)
    target = base / "SIMPLE-pretrain"
    if target.exists():
        assets(target)
        print(f"EGOVLA_ASSETS_ALREADY_VERIFIED {target}", flush=True)
        return
    archive_dir = base / "downloads"
    archive_dir.mkdir(exist_ok=True)
    archive = archive_dir / (FOLDER + ".zip")
    if not archive.exists():
        pending = archive.with_suffix(".zip.partial")
        offset = pending.stat().st_size if pending.exists() else 0
        url = (
            f"https://huggingface.co/{REPO}/resolve/{REVISION}/{archive.name}"
            f"?download=true&fresh={time.time_ns()}"
        )
        with requests.get(
            url,
            headers={"Range": f"bytes={offset}-"} if offset else {},
            stream=True,
            timeout=(30, 120),
        ) as response:
            response.raise_for_status()
            if offset and response.status_code != 206:
                raise RuntimeError(
                    "Server refused partial resume; preserve partial and inspect before retry"
                )
            if offset and not response.headers.get("Content-Range", "").startswith(
                f"bytes {offset}-"
            ):
                raise RuntimeError("Unexpected resume byte range")
            written, reported = offset, offset
            with pending.open("ab" if offset else "xb") as stream:
                for block in response.iter_content(8 * 1024 * 1024):
                    stream.write(block)
                    written += len(block)
                    if written > SIZE:
                        raise ValueError("Download exceeds fixed official ZIP size")
                    if written - reported >= 256 * 1024 * 1024:
                        print(f"EGOVLA_DOWNLOAD {written}/{SIZE} bytes", flush=True)
                        reported = written
                stream.flush()
                os.fsync(stream.fileno())
        if pending.stat().st_size != SIZE or sha256(pending) != SHA256:
            raise ValueError("Official ZIP size/SHA verification failed; partial retained")
        pending.rename(archive)
    if archive.stat().st_size != SIZE or sha256(archive) != SHA256:
        raise ValueError("Existing archive does not match fixed official LFS record")
    stage = Path(tempfile.mkdtemp(prefix="SIMPLE-pretrain-stage-", dir=base))
    # 使用临时目录，CRC/布局全部通过才发布；拒绝ZIP目录逃逸、符号链接及覆盖。
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            destination = stage / member.filename
            if not destination.resolve().is_relative_to(stage) or stat.S_ISLNK(
                member.external_attr >> 16
            ):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    extracted = stage / FOLDER
    if not extracted.is_dir():
        raise ValueError(f"Unexpected ZIP root; stage retained: {stage}")
    atomic_json(
        extracted / "SOURCE.json",
        dict(
            repo=REPO,
            revision=REVISION,
            archive=archive.name,
            size=SIZE,
            sha256=SHA256,
            basis="SIMPLE examples/quick_start/psi.md; base VLM, not human-video-pretrained",
        ),
    )
    assets(extracted, register=True)
    extracted.rename(target)
    # 清单只含相对路径，改名后再验完整集合与SHA。
    manifest = assets(target)
    print(f"EGOVLA_ASSETS_READY {target} files={len(manifest['files'])}", flush=True)


if __name__ == "__main__":
    main()
