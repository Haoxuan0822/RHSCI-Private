"""Freeze only-old-protocol training cells as the authorized second phase."""

import json
from pathlib import Path
import subprocess

from plan import ROOT

STATE = ROOT / "evals/goal-latest-protocol-20261105"
RAW = Path("/tmp/rhsci-video-upload-GHEvq5/repo")
EXCLUDED_VARIANTS = {"VGQ-AE-D"}


def inventory():
    target = STATE / "legacy-retest-plan.json"
    if target.exists():
        saved = json.loads(target.read_text())
        runs = [run for run in saved["runs"] if run["variant"] not in EXCLUDED_VARIANTS]
        if runs != saved["runs"]:
            saved["runs"] = runs
            saved["excluded_variants"] = sorted(EXCLUDED_VARIANTS)
            saved["rule"] = "Training parentheses: historical-only; exclude ACT and VGQ-AE-D; preserve original historical scores"
            target.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n")
        return runs
    records = json.loads((ROOT / "hf/RHSCI/model-59a1ea14/DOWNLOAD_MANIFEST-3fdbb4c7.json").read_text())["files"]
    groups = {}
    for record in records:
        parts = record["rfilename"].split("/")
        if len(parts) < 5 or parts[0] != "checkpoint" or parts[1].startswith("_"):
            continue
        key = (parts[1].lower(), parts[2].strip("()"), {"betweentables": "between"}.get(parts[3], parts[3]))
        groups.setdefault(key, []).append(record)
    training = (RAW / "README.md").read_text().split("## 训练记录", 1)[1]
    backbone = None
    runs = []
    for line in training.splitlines():
        if line.startswith("### "):
            backbone = {"π0.5": "pi05", "GR00T": "gr00t", "Ψ0": "psi0"}.get(line[4:])
        if backbone is None or not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        if cells[0] in EXCLUDED_VARIANTS:
            continue
        for task, cell in zip(("between", "hug", "oven", "handover", "faucet"), cells[1:]):
            if not (cell.startswith("（") or cell.startswith("(")):
                continue
            key = (backbone, cells[0], task)
            if key not in groups:
                raise ValueError(f"No pinned HF assets for authorized retest: {key}")
            assets = groups[key]
            archives = [r["rfilename"] for r in assets if r["rfilename"].endswith(".tar.zst")]
            models = [r["rfilename"] for r in assets if r["rfilename"].endswith("model.safetensors")]
            if len(archives) + len(models) != 1:
                raise ValueError(f"Ambiguous checkpoint identity: {key}")
            run = dict(backbone=backbone, variant=cells[0], task=task, step=int(cell.strip("（）()").rstrip("K")) * 1000,
                       horizon=None, records=assets)
            if archives:
                run["archive"] = archives[0]
            else:
                run["model"] = models[0]
            runs.append(run)
    STATE.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(revision="3fdbb4c714d9ca70abf4b9e9bbe38cabe7781990",
                                      github_commit=subprocess.check_output(["git", "-C", str(RAW), "rev-parse", "HEAD"], text=True).strip(),
                                      rule="Training parentheses: historical-only; exclude ACT and VGQ-AE-D; preserve original historical scores",
                                      excluded_variants=sorted(EXCLUDED_VARIANTS), runs=runs), indent=2, ensure_ascii=False) + "\n")
    return runs


if __name__ == "__main__":
    for run in inventory():
        print(run["backbone"], run["variant"], run["task"], run["step"], run.get("archive", run.get("model")))
