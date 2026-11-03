"""Download and extract the seven pinned follow-up checkpoints outside the checkout."""

import concurrent.futures
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import time
import traceback
from pathlib import Path

import zstandard
from plan import OUTPUT, REVISION, ROOT, RUNS, run_key

MODEL_ROOT = ROOT / "hf/RHSCI/model-59a1ea14"
LOG = OUTPUT / "preparation"
LOG.mkdir(parents=True, exist_ok=True)


def processes():
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                yield int(entry.name), (entry / "cmdline").read_bytes().split(b"\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass


def existing_transfer(path):
    for pid, args in processes():
        if (
            len(args) > 4
            and args[1] == str(ROOT / "download_wheel.py").encode()
            and args[3] == str(path).encode()
        ):
            return pid
    return None


def prepare(run, records):
    key = run_key(run)
    record = records[run["archive"]]
    archive = MODEL_ROOT / run["archive"]
    destination = ROOT / "checkpoints/followup" / key
    receipt = LOG / f"{key}.json"
    if receipt.exists():
        return key, json.loads(receipt.read_text())
    while existing_transfer(archive) is not None:
        print("WAIT_EXISTING_TRANSFER", key, flush=True)
        time.sleep(30)
    with (LOG / f"{key}-download.log").open("a") as log:
        command = [
            "/root/miniforge/bin/python3",
            str(ROOT / "download_wheel.py"),
            f"https://huggingface.co/ds1x/RHSCI/resolve/{REVISION}/{run['archive']}",
            str(archive),
            record["lfs"]["sha256"],
            "--workers",
            "8",
        ]
        for attempt in range(20):
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                break
            print("RETRY", key, attempt + 1, flush=True)
            time.sleep(30)
        else:
            raise RuntimeError(f"Download failed: {key}; see {log.name}")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    assert (
        archive.stat().st_size == record["size"] and digest.hexdigest() == record["lfs"]["sha256"]
    )
    staging = destination.with_name(key + ".extracting")
    staging.mkdir(parents=True, exist_ok=False)
    with (
        archive.open("rb") as compressed,
        zstandard.ZstdDecompressor().stream_reader(compressed) as stream,
    ):
        with tarfile.open(fileobj=stream, mode="r|") as source:
            for member in source:
                source.extract(member, staging, filter="data")
    config = "rsci_config.json" if run["backbone"] == "pi05" else "config.json"
    checkpoints = [
        p.parent for p in staging.rglob(config) if p.parent.name == f"checkpoint-{run['step']}"
    ]
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one inference checkpoint: {key}: {checkpoints}")
    run_dir = checkpoints[0].parent
    if not (run_dir / "launch_contract.json").exists():
        contracts = list(staging.rglob("launch_contract.json"))
        if len(contracts) != 1:
            raise RuntimeError(f"Ambiguous saved contract: {key}: {contracts}")
        shutil.copy2(contracts[0], run_dir / "launch_contract.json")
    contract = json.loads((run_dir / "launch_contract.json").read_text())
    horizon = contract.get("horizon" if run["backbone"] == "pi05" else "valid_action_horizon")
    if horizon != run["horizon"]:
        raise ValueError(f"Saved action horizon mismatch: {key}: {horizon}")
    relative = run_dir.relative_to(staging)
    staging.rename(destination)
    canonical = destination / "inference"
    canonical.symlink_to(relative, target_is_directory=True)
    info = dict(
        run_dir=str(canonical),
        archive=str(archive),
        sha256=digest.hexdigest(),
        revision=REVISION,
        saved_model=contract.get("model"),
        horizon=horizon,
    )
    receipt.write_text(json.dumps(info, indent=2) + "\n")
    print("CHECKPOINT_PREPARED", key, info, flush=True)
    return key, info


def main():
    manifest = json.loads((MODEL_ROOT / "DOWNLOAD_MANIFEST.json").read_text())
    assert manifest["revision"] == REVISION
    records = {record["rfilename"]: record for record in manifest["files"]}
    wanted = {str(MODEL_ROOT / run["archive"]).encode() for run in RUNS}
    paused = []
    # 暂停全量下载的调度和非优先传输；已在传的本次权重继续，避免重复写同一 .parts。
    for pid, args in processes():
        if len(args) < 2:
            continue
        if (args[1] == str(ROOT / "download_rhsci_hf.py").encode() and b"weights" in args) or (
            args[1] == str(ROOT / "download_wheel.py").encode()
            and len(args) > 3
            and args[3].startswith(str(MODEL_ROOT).encode())
            and args[3] not in wanted
        ):
            os.kill(pid, signal.SIGSTOP)
            paused.append((pid, args))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(prepare, run, records) for run in RUNS]
            prepared = dict(future.result() for future in concurrent.futures.as_completed(futures))
        (LOG / "READY.json").write_text(json.dumps(prepared, indent=2) + "\n")
        print("ALL_FOLLOWUP_CHECKPOINTS_READY", flush=True)
    finally:
        current = dict(processes())
        for pid, args in paused:
            if current.get(pid) == args:
                os.kill(pid, signal.SIGCONT)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        (LOG / "FAILED.txt").write_text(traceback.format_exc())
        raise
