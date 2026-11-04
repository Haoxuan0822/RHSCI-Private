"""Read historical GitHub episode manifests without downloading videos or model weights."""

import argparse
import base64
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def api(resource):
    result = subprocess.run(["gh", "api", resource], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="enterx-x/Psi0")
    parser.add_argument("--tree", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tree = api(f"repos/{args.repository}/git/trees/{args.tree}?recursive=1")
    if tree.get("truncated"):
        raise RuntimeError("Incomplete GitHub tree")
    entries = [item for item in tree["tree"] if item["path"].endswith("manifest.json")]

    def read(entry):
        blob = api(f"repos/{args.repository}/git/blobs/{entry['sha']}")
        raw = base64.b64decode(blob["content"])
        git_hash = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if git_hash != entry["sha"]:
            raise ValueError(f"Git blob identity mismatch: {entry['path']}")
        record = json.loads(raw)
        if record["checkpoint_sha256"] != args.checkpoint_sha256:
            raise ValueError(f"Different historical checkpoint: {entry['path']}")
        if type(record["success"]) is not bool:
            raise ValueError("Historical success must be an actual boolean")
        return {"path": entry["path"], "git_blob_sha1": git_hash, "record": record}

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(read, entries))
    identities = [
        (item["record"]["repeat"], item["record"]["internal_level"], item["record"]["episode"])
        for item in records
    ]
    expected = {
        (repeat, level, episode) for repeat in (1, 2) for level in range(3) for episode in range(10)
    }
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError(f"Historical grid incomplete or duplicated: {identities}")
    # 上传器的 stats_sha256 指 results/eval_stats.txt，不是模型的归一化统计。
    score_hashes = {}
    for item in records:
        record = item["record"]
        group = f"repeat-{record['repeat']}/level-{record['internal_level']}"
        score_hashes.setdefault(group, set()).add(record["stats_sha256"])
    if any(len(hashes) != 1 for hashes in score_hashes.values()):
        raise ValueError("Conflicting score file hashes within one historical group")
    report = {
        "repository": args.repository,
        "tree_sha": args.tree,
        "checkpoint_sha256": args.checkpoint_sha256,
        "score_file_sha256_by_group": {
            group: next(iter(hashes)) for group, hashes in score_hashes.items()
        },
        "stats_sha256_meaning": "Official results/eval_stats.txt, not normalization statistics",
        "successes_by_level": [
            sum(
                item["record"]["success"]
                for item in records
                if item["record"]["internal_level"] == level
            )
            for level in range(3)
        ],
        "completed_by_level": [20, 20, 20],
        "protocols": sorted({item["record"]["protocol"] for item in records}),
        "records": records,
        "scope": "Only the two uploaded rounds; not proof of all five historical rounds",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))


if __name__ == "__main__":
    main()
