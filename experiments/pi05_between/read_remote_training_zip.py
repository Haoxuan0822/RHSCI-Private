"""通过HTTP Range只读取官方训练ZIP的元数据和首条轨迹，不下载整个包。"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import aiohttp
import fsspec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--include-video", action="store_true")
    args = parser.parse_args()
    if args.episode < 0:
        parser.error("episode must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(url=args.url, files=[])
    with (
        fsspec.open(
            args.url,
            "rb",
            block_size=65536,
            client_kwargs={
                "trust_env": True,
                "timeout": aiohttp.ClientTimeout(total=60, connect=10),
            },
        ) as remote,
        ZipFile(remote) as archive,
    ):
        chunk = args.episode // 1000
        wanted = [
            "meta/info.json",
            "meta/episodes.jsonl",
            f"data/chunk-{chunk:03d}/episode_{args.episode:06d}.parquet",
        ]
        if args.include_video:
            wanted.append(
                f"videos/chunk-{chunk:03d}/observation.images.ego_view/"
                f"episode_{args.episode:06d}.mp4"
            )
        for suffix in wanted:
            members = [
                entry
                for entry in archive.infolist()
                if entry.filename.endswith("/" + suffix) or entry.filename == suffix
            ]
            if len(members) != 1:
                raise ValueError(f"Need a unique {suffix}: {[m.filename for m in members]}")
            entry = members[0]
            if entry.file_size > 150 * 1024**2:
                raise ValueError("Unexpectedly large selected member")
            content = archive.read(entry)
            path = args.output / Path(suffix).name
            with path.open("xb") as stream:
                stream.write(content)
            report["files"].append(
                dict(
                    member=entry.filename,
                    bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                    path=str(path),
                )
            )
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
