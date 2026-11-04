"""只下载并验证指定GR00T任务的官方旧评测初态包，不下载权重或场景资产。"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import requests

REVISION = "e78fb93cc28912a3031a10b8656d32d7f0a2b867"
TASKS = (
    "G1WholebodyOpenOvenTeleop-v0",
    "G1WholebodyHandoverTeleop-v0",
    "G1WholebodyOpenFaucetTeleop-v0",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    archive_root = args.output_root / "archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    endpoint = (
        f"https://huggingface.co/api/datasets/USC-PSI-Lab/psi-data/tree/{REVISION}/simple-eval"
    )
    response = session.get(endpoint, params={"limit": 100}, timeout=60)
    response.raise_for_status()
    remote = {row["path"]: row for row in response.json()}
    results = []
    for task in TASKS:
        filename = f"simple-eval/{task}.zip"
        source = remote[filename]
        archive = archive_root / f"{task}.zip"
        if not archive.exists():
            url = f"https://huggingface.co/datasets/USC-PSI-Lab/psi-data/resolve/{REVISION}/{filename}"
            response = session.get(url, timeout=120)
            response.raise_for_status()
            with archive.open("xb") as stream:
                stream.write(response.content)
        payload = archive.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != source["size"] or digest != source["lfs"]["oid"]:
            raise ValueError(f"Official archive identity mismatch: {archive}")
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                target = args.output_root / member.filename
                if not target.resolve().is_relative_to(args.output_root.resolve()):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                if member.is_dir():
                    continue
                content = package.read(member)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.read_bytes() != content:
                        raise ValueError(f"Existing old-package file differs: {target}")
                else:
                    with target.open("xb") as stream:
                        stream.write(content)
        counts = {}
        levels = sorted(path.name for path in (args.output_root / task).glob("dr-level-*"))
        if levels != ["dr-level-0", "dr-level-1", "dr-level-2"]:
            raise ValueError(f"Expected three old-package levels: {task}: {levels}")
        for level in levels:
            metadata = args.output_root / task / level / "meta/episodes.jsonl"
            records = [json.loads(line) for line in metadata.read_text().splitlines()]
            if sorted(row["episode_index"] for row in records) != list(range(10)):
                raise ValueError(f"Expected ten distinct initial states: {metadata}")
            counts[level] = len(records)
        row = dict(task=task, sha256=digest, bytes=len(payload), counts=counts)
        results.append(row)
        print(json.dumps(row), flush=True)
    receipt = dict(repository="USC-PSI-Lab/psi-data", revision=REVISION, tasks=results)
    (args.output_root / "gr00t-old-package-source.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
