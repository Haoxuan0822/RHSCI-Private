"""Extract verified priority archives outside the source checkout."""

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import zstandard

ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
TASK = "G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=["dataset", "state"])
args = parser.parse_args()
if args.kind == "dataset":
    archive = (
        ROOT
        / "hf/RHSCI/dataset-ea9d480f/train-data/between/simple-between-tables-new-training-h264--G1WholebodyLocomotionPickBetweenTablesTeleop-v0--2321af1753.tar.zst"
    )
    target = ROOT / "data/processed" / TASK
else:
    archive = (
        ROOT
        / "hf/RHSCI/model-59a1ea14/checkpoint/PI05/STATE/between/priority-rgb-reader-20261028-v1/checkpoint-40000.tar.zst"
    )
    target = ROOT / "checkpoints/pi05-STATE-between-40000"
target.mkdir(parents=True, exist_ok=False)
members = []
with archive.open("rb") as compressed:
    with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as source:
            for member in source:
                source.extract(member, target, filter="data")
                members.append({"name": member.name, "size": member.size})
                print("EXTRACTED", member.name, member.size, flush=True)
if args.kind == "state":
    contract = target / "launch_contract.json"
    if not contract.exists():
        candidates = list(target.rglob("launch_contract.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one saved contract, found: {candidates}")
        shutil.copy2(candidates[0], contract)
    assert (target / "checkpoint-40000/rsci_config.json").is_file()
else:
    assert (target / "meta/stats.json").is_file()
(ROOT / f"logs/pi05-between-{args.kind}-extraction.json").write_text(
    json.dumps(
        {
            "archive": str(archive),
            "destination": str(target),
            "members": members,
        },
        indent=2,
    )
    + "\n"
)
print("EXTRACTION_COMPLETE", args.kind, len(members), target, flush=True)
