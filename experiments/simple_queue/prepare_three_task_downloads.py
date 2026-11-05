"""清点并下载三个指定任务的推理资产，复用运行目录既有断点下载器。"""

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

import requests
from plan import ROOT

REVISION = "6921879a6bf52dea74670e9e20a341e50443aa1c"
TASKS = {"oven", "handover", "faucet"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    response = requests.get(
        f"https://huggingface.co/api/models/ds1x/RHSCI/revision/{REVISION}?blobs=true",
        timeout=60,
    )
    response.raise_for_status()
    metadata = response.json()
    assert metadata["sha"] == REVISION
    selected = []
    for record in metadata["siblings"]:
        path = Path(record["rfilename"])
        parts = path.parts
        if len(parts) < 5 or parts[0] != "checkpoint" or parts[3] not in TASKS:
            continue
        if parts[1] not in {"Gr00t", "PI05", "PSI0", "Psi0"}:
            continue
        # 推理只需模型及配置；不补下载单独发布的优化器、调度器、RNG。
        if path.name.startswith("random_states_") or path.name in {
            "optimizer.bin",
            "scheduler.bin",
        }:
            continue
        selected.append(record)
    destination = ROOT / "hf/RHSCI/model-59a1ea14"
    output = ROOT / "evals/three-task-downloads-20261105"
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(repo="ds1x/RHSCI", revision=REVISION, files=selected)
    (output / "DOWNLOAD_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for record in selected:
        path = destination / record["rfilename"]
        present = path.exists() and path.stat().st_size == record["size"]
        print("LOCAL" if present else "MISSING", record["rfilename"], record["size"], flush=True)
    if not args.download:
        return

    def download(record):
        relative = record["rfilename"]
        path = destination / relative
        # 已有GR00T归档已由当前评测准备流程校验，避免重复读取百GB。
        if (
            relative.startswith("checkpoint/Gr00t/")
            and path.exists()
            and path.stat().st_size == record["size"]
        ):
            return relative
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/ds1x/RHSCI/resolve/{REVISION}/{relative}"
        if record.get("lfs"):
            log = output / (relative.replace("/", "__") + ".log")
            with log.open("a") as stream:
                subprocess.run(
                    [
                        "/root/miniforge/bin/python3",
                        str(ROOT / "download_wheel.py"),
                        url,
                        str(path),
                        record["lfs"]["sha256"],
                        "--workers",
                        "8",
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        else:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id="ds1x/RHSCI",
                filename=relative,
                revision=REVISION,
                local_dir=destination,
                token=False,
            )
        if path.stat().st_size != record["size"]:
            raise ValueError(f"Incomplete file: {relative}")
        print("VERIFIED", relative, flush=True)
        return relative

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        verified = list(pool.map(download, selected))
    (output / "COMPLETE.json").write_text(
        json.dumps(dict(revision=REVISION, files=verified)) + "\n"
    )
    print("DOWNLOAD_COMPLETE", len(verified), flush=True)


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "0"
    main()
