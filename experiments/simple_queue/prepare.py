"""Restore selected local checkpoints; the current plan enables no weight downloads."""

import argparse
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
from plan import ACTIVE_RUNS, DOWNLOAD_RUNS, PREPARATION, REVISION, ROOT, run_key

MODEL_ROOT = ROOT / "hf/RHSCI/model-59a1ea14"
LOG = PREPARATION
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
            and Path(os.fsdecode(args[1])).resolve() == (ROOT / "download_wheel.py").resolve()
            and Path(os.fsdecode(args[3])).resolve() == path.resolve()
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
    if run not in DOWNLOAD_RUNS:
        # 已下载的其他指定权重只做本地校验，不能因文件损坏触发重新下载。
        if not archive.is_file() or archive.stat().st_size != record["size"]:
            raise ValueError(f"Local archive missing or incomplete: {key}")
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["lfs"]["sha256"]:
                raise ValueError(f"Local archive SHA256 mismatch: {key}")
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
    if run["horizon"] is not None and horizon != run["horizon"]:
        raise ValueError(f"Saved action horizon mismatch: {key}: {horizon}")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"Invalid saved horizon: {key}: {horizon}")
    relative = run_dir.relative_to(staging)
    staging.rename(destination)
    canonical = destination / "inference"
    canonical.symlink_to(relative, target_is_directory=True)
    if run["backbone"] == "pi05" and (canonical / "dataset-meta").is_dir():
        processor = canonical / "processor-data"
        processor.mkdir(exist_ok=True)
        (processor / "meta").symlink_to("../dataset-meta", target_is_directory=True)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pi05-three-tasks", action="store_true", help="Prepare downloaded pi05 three-task assets"
    )
    parser.add_argument(
        "--hug-state", action="store_true", help="Only prepare the newly requested GR00T hug STATE"
    )
    parser.add_argument(
        "--remaining-gr00t",
        action="store_true",
        help="Resume only the requested handover/faucet VGQ-AE checkpoints",
    )
    args = parser.parse_args()
    manifest = json.loads((MODEL_ROOT / "DOWNLOAD_MANIFEST.json").read_text())
    assert manifest["revision"] == REVISION
    records = {record["rfilename"]: record for record in manifest["files"]}
    if args.pi05_three_tasks:
        from plan import PI05_THREE_RUNS

        pending = list(PI05_THREE_RUNS)
        while pending:
            for run in list(pending):
                archive = MODEL_ROOT / run["archive"]
                if not archive.exists() or existing_transfer(archive) is not None:
                    continue
                prepare(run, records)
                pending.remove(run)
            if pending:
                print("WAIT_PI05_DOWNLOADS", [run_key(run) for run in pending], flush=True)
                time.sleep(15)
        print("PI05_THREE_TASKS_READY", flush=True)
        return
    if args.remaining_gr00t:
        from plan import RUNS

        selected = [
            run
            for run in RUNS
            if run["backbone"] == "gr00t"
            and run["variant"] == "VGQ-AE"
            and run["task"] in ("handover", "faucet")
        ]
        DOWNLOAD_RUNS.extend(selected)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(prepare, run, records) for run in selected]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        print("REQUESTED_GR00T_CHECKPOINTS_READY", flush=True)
        return
    if args.hug_state:
        # 单任务追加准备，不改活动队列、不清理其他任务进程。
        run = dict(
            backbone="gr00t",
            variant="STATE",
            task="hug",
            step=20000,
            horizon=16,
            archive="checkpoint/Gr00t/STATE/hug/paper20k-fullvlm-rgb-spacecache-v1/checkpoint-20000.tar.zst",
        )
        DOWNLOAD_RUNS.append(run)
        prepare(run, records)
        return
    wanted = {str(MODEL_ROOT / run["archive"]).encode() for run in DOWNLOAD_RUNS}
    # 用户限定只下载本次评测权重：先结束全量重试器，再结束调度和无关传输。
    # 保留全部分片；本次需要且已在传的文件继续，避免重复写同一 .parts。
    running = list(processes())
    running.sort(key=lambda item: b"download_rhsci_hf_supervisor.py" not in b" ".join(item[1]))
    full_download_scripts = {
        str(ROOT / name).encode()
        for name in ("download_rhsci_hf.py", "download_rhsci_hf_supervisor.py")
    }
    for pid, args in running:
        if len(args) < 2:
            continue
        if (args[1] in full_download_scripts and b"weights" in args) or (
            args[1] == str(ROOT / "download_wheel.py").encode()
            and len(args) > 3
            and args[3].startswith(str(MODEL_ROOT).encode())
            and args[3] not in wanted
        ):
            try:
                os.kill(pid, signal.SIGTERM)
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(prepare, run, records) for run in ACTIVE_RUNS]
        prepared = dict(future.result() for future in concurrent.futures.as_completed(futures))
    (LOG / "READY.json").write_text(json.dumps(prepared, indent=2) + "\n")
    print("ALL_FOLLOWUP_CHECKPOINTS_READY", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        (LOG / "FAILED.txt").write_text(traceback.format_exc())
        raise
