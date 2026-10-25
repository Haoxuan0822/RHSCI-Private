"""只提取官方 raw 的 metadata/parquet，不重复解压视频。"""

from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from pathlib import Path, PurePosixPath


def extract(archive: Path, output: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Official raw ZIP SHA256 mismatch")
    count = 0

    def unpack(source: zipfile.ZipFile) -> None:
        nonlocal count
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe ZIP member: {path}")
            if path.suffix == ".zip":
                # OpenFaucet 官方归档包含三批采集的嵌套 ZIP，逐批保持原路径。
                with zipfile.ZipFile(io.BytesIO(source.read(member))) as nested:
                    unpack(nested)
                continue
            if member.is_dir() or not ("meta" in path.parts or path.suffix == ".parquet"):
                continue
            target = output.joinpath(*path.parts)
            content = source.read(member)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f"Conflicting existing raw file: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # 使用独占写入，避免覆盖已有数据；ZIP CRC 同时由 read 检查。
                with target.open("xb") as stream:
                    stream.write(content)
            count += 1

    with zipfile.ZipFile(archive) as source:
        unpack(source)
    if count == 0:
        raise ValueError("No raw metadata/parquet found in archive")
    print(f"Verified and extracted {count} metadata/parquet members to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    extract(args.archive, args.output, args.sha256)
