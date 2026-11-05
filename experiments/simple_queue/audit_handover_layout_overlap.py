"""在 CPU 比较实际训练元数据与旧包递物初态，不启动模型或改动评测。"""

import argparse
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

TASK = "G1WholebodyHandoverTeleop-v0"
TRAIN_MEMBER = (
    f"current-data-metadata/simple-handover-new-training-h264-20261021/{TASK}/meta/episodes.jsonl"
)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def configs(payload):
    rows = [json.loads(line) for line in payload.splitlines()]
    for row in rows:
        value = row["environment_config"]
        row["config"] = json.loads(value) if isinstance(value, str) else value
    return rows


def core(config):
    """提取 reset 的三组空间位姿；不使用可能已被演示终态更新的 layout。"""
    state = config["dr_state_dict"]
    return {
        "robot": state["spatial"]["g1_sonic"],
        "target": state["spatial"][str(state["target"]["uid"])],
        "container": state["spatial"][f"container_{state['container']['uid']}"],
        "assets": {
            key: {field: state[key][field] for field in ("res_id", "uid")}
            for key in ("target", "container")
        },
    }


def scene_geometry(config):
    state = deepcopy(config["dr_state_dict"]["scene"])
    state["table"].pop("material", None)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("/root/workspace/Haoxuan/RHSCI-runtime"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = args.runtime
    archive = runtime / "evals/gr00t-state-input-audit-20261104/metadata.tar.zst"

    def read_member(name):
        return subprocess.check_output(
            [
                "tar",
                "--use-compress-program=zstd",
                "-xOf",
                str(archive),
                name,
            ]
        )

    payload = read_member(TRAIN_MEMBER)
    receipt = json.loads(
        read_member("restore-receipts/handover100-training-and-ardy-20261025.verified.json")
    )
    matches = [
        item
        for item in receipt["files"]
        if item["path"].endswith(
            f"simple-handover-new-training-h264-20261021/{TASK}/meta/episodes.jsonl"
        )
    ]
    if len(matches) != 1 or digest(payload) != matches[0]["sha256"]:
        raise ValueError("Training metadata does not match its restoration receipt")
    training = configs(payload)
    if len(training) != 100 or sum(row["length"] for row in training) != 47259:
        raise ValueError("Unexpected training dataset")
    contracts = {
        variant: json.loads(
            (
                runtime / "checkpoints/followup" / f"gr00t-{variant}-handover/launch_contract.json"
            ).read_text()
        )
        for variant in ("STATE", "VGQ-AE")
    }
    fingerprints = {value["dataset_fingerprint"] for value in contracts.values()}
    if fingerprints != {"0a9d8d34479a3e53729754e2f79ac6148348aa083a79972c4e26fa81e0b32cca"}:
        raise ValueError("Unexpected saved dataset fingerprint")
    report = {
        "training": {
            "archive": str(archive),
            "member": TRAIN_MEMBER,
            "sha256": digest(payload),
            "receipt_sha_verified": True,
            "episodes": len(training),
            "frames": 47259,
            "saved_dataset_fingerprint": fingerprints.pop(),
            "dr_levels": dict(Counter(row["config"]["metadata"]["dr_level"] for row in training)),
            "scene_names": dict(
                Counter(row["config"]["dr_state_dict"]["scene"]["name"] for row in training)
            ),
            "source_raw_unique_sha256": len({row["source_raw_sha256"] for row in training}),
        },
        "levels": {},
        "scope": (
            "Exact JSON numeric equality for saved reset spatial states/assets. "
            "Checks actual reset fields, not terminal layout actor poses. "
            "Scene/lighting/material equality reported separately; not pixel identity. "
            "Does not prove identical settled joint states, training RGB replay fidelity, "
            "or causal contribution to success. No model, GPU, or simulation started."
        ),
    }
    official_zip = runtime / f"eval-official-pre1102/archives/{TASK}.zip"
    official_sha = digest(official_zip.read_bytes())
    source_receipt = json.loads(
        (runtime / "eval-official-pre1102/gr00t-old-package-source.json").read_text()
    )
    expected = [row for row in source_receipt["tasks"] if row["task"] == TASK]
    if len(expected) != 1 or official_sha != expected[0]["sha256"]:
        raise ValueError("Official evaluation archive differs from source receipt")
    report["official_evaluation_archive"] = {
        "path": str(official_zip),
        "sha256": official_sha,
        "repository": source_receipt["repository"],
        "revision": source_receipt["revision"],
    }
    for level in range(3):
        path = runtime / f"eval-official-pre1102/{TASK}/dr-level-{level}/meta/episodes.jsonl"
        evaluation_payload = path.read_bytes()
        evaluation = configs(evaluation_payload)
        suffix = f"dr-level-{level}/meta/episodes.jsonl"
        with zipfile.ZipFile(official_zip) as package:
            members = [name for name in package.namelist() if name.endswith(suffix)]
            if len(members) != 1 or package.read(members[0]) != evaluation_payload:
                raise ValueError("Evaluation metadata differs from original official archive")
        if len(evaluation) != 10:
            raise ValueError("Unexpected evaluation episode count")
        episodes = []
        for row in evaluation:
            cfg = row["config"]
            candidates = [other for other in training if core(other["config"]) == core(cfg)]
            episode = {
                "episode": row["episode_index"],
                "core_exact_training_episodes": [other["episode_index"] for other in candidates],
                "scene_geometry_exact_training_episodes": [
                    other["episode_index"]
                    for other in training
                    if scene_geometry(other["config"]) == scene_geometry(cfg)
                ],
            }
            for field in ("lighting", "material", "distractors"):
                episode[f"{field}_exact_with_core_match"] = [
                    other["episode_index"]
                    for other in candidates
                    if other["config"]["dr_state_dict"].get(field)
                    == cfg["dr_state_dict"].get(field)
                ]
            episode["full_dr_state_exact_training_episodes"] = [
                other["episode_index"]
                for other in training
                if other["config"]["dr_state_dict"] == cfg["dr_state_dict"]
            ]
            episodes.append(episode)
        report["levels"][f"dr-level-{level}"] = {
            "path": str(path),
            "sha256": digest(evaluation_payload),
            "core_exact_matches": sum(
                bool(row["core_exact_training_episodes"]) for row in episodes
            ),
            "full_dr_state_exact_matches": sum(
                bool(row["full_dr_state_exact_training_episodes"]) for row in episodes
            ),
            "scene_geometry_matches": sum(
                bool(row["scene_geometry_exact_training_episodes"]) for row in episodes
            ),
            "episodes": episodes,
        }
        scores = {}
        for variant in contracts:
            root = (
                runtime
                / "evals/gr00t-zero-arms-pre1102-5cycles-20261104"
                / f"gr00t-{variant}-handover/{variant}/dr-level-{level}"
            )
            results = [
                json.loads(path.read_text())
                for path in sorted(root.glob("repeat-*/verified_results.json"))
            ]
            if len(results) != 5 or any(result["episodes"] != 10 for result in results):
                raise ValueError("Expected five complete evaluation rounds")
            scores[variant] = {
                "successes": sum(sum(result["stats"].values()) for result in results),
                "completed": 50,
            }
        report["levels"][f"dr-level-{level}"]["scores"] = scores
    source = runtime / "external/SIMPLE-official/src/simple"
    report["reset_source_sha256"] = {
        name: digest((source / name).read_bytes())
        for name in ("core/task.py", "dr/manager.py", "dr/spatial.py", "cli/eval_decoupled_wbc.py")
    }
    output = args.output or runtime / "evals/gr00t-handover-layout-overlap-20261105.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "levels": {
                    name: {key: value for key, value in data.items() if key != "episodes"}
                    for name, data in report["levels"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
