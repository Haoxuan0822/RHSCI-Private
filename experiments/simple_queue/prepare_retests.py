"""Prepare authorized old-only retests from pinned checkpoint/processor assets."""

import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import zstandard

import prepare as assets
from legacy_inventory import inventory, STATE
from plan import PREPARATION, ROOT, run_key
from prepare_nostate import REVISION, prepare_psi0


def download(record):
    path = assets.MODEL_ROOT / record["rfilename"]
    lfs = record.get("lfs")
    if path.is_file() and path.stat().st_size == record["size"]:
        with path.open("rb") as stream:
            if not lfs or hashlib.file_digest(stream, "sha256").hexdigest() == lfs["sha256"]:
                return
    if lfs:
        subprocess.run(["/root/miniforge/bin/python3", str(ROOT / "download_wheel.py"),
                        f"https://huggingface.co/ds1x/RHSCI/resolve/{REVISION}/{record['rfilename']}",
                        str(path), lfs["sha256"], "--workers", "4"], check=True)
    else:
        subprocess.run([str(ROOT / ".venv-psi/bin/python"), "-m", "huggingface_hub.commands.huggingface_cli",
                        "download", "ds1x/RHSCI", record["rfilename"], "--revision", REVISION,
                        "--local-dir", str(assets.MODEL_ROOT)], check=True)


def old_processor():
    destination = ROOT / "checkpoints/legacy-processor-20261025"
    if (destination / "initialization-assets/config.json").is_file():
        return destination / "initialization-assets"
    archive = assets.MODEL_ROOT / "checkpoint/_metadata/20261025/all-16-checkpoint-configs-and-processors-20261025.tar.zst"
    records = json.loads((assets.MODEL_ROOT / "DOWNLOAD_MANIFEST-3fdbb4c7.json").read_text())["files"]
    download(next(r for r in records if r["rfilename"] == str(archive.relative_to(assets.MODEL_ROOT))))
    destination.mkdir(exist_ok=True)
    processor = destination / "initialization-assets"
    processor.mkdir(exist_ok=True)
    with archive.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as source:
            for member in source:
                if member.name.startswith("external/") and 3 <= int(member.name.split("/")[1]) <= 13:
                    member.name = Path(member.name).name
                    source.extract(member, processor, filter="data")
    return processor


def prepare_one(run, processor):
    key = run_key(run)
    receipt = PREPARATION / f"{key}.json"
    if receipt.exists():
        return
    if "archive" in run:
        record = next(r for r in run["records"] if r["rfilename"] == run["archive"])
        download(record)
        if run["backbone"] == "psi0":
            prepare_psi0(run, record)
        else:
            assets.prepare(run, {run["archive"]: record})
        return
    # Loose PSI0 checkpoints already carry embedded normalization; keep all saved fields.
    wanted = [r for r in run["records"] if Path(r["rfilename"]).name in ("model.safetensors", "run_config.json", "argv.txt", "launch_contract.json")]
    for record in wanted:
        download(record)
    checkpoint = assets.MODEL_ROOT / run["model"]
    original = checkpoint.parent.parent.parent
    config = json.loads((original / "run_config.json").read_text())
    horizon = config["model"]["action_chunk_size"]
    if horizon != config["model"]["action_exec_horizon"]:
        raise ValueError(f"Saved execution differs from prediction: {key}")
    destination = ROOT / "checkpoints/followup" / key
    destination.mkdir(exist_ok=True)
    if not (destination / "initialization-assets").exists():
        (destination / "initialization-assets").symlink_to(processor, target_is_directory=True)
    canonical = destination / "inference"
    canonical.mkdir(exist_ok=True)
    for name in ("run_config.json", "argv.txt"):
        link = canonical / name
        if not link.exists():
            link.symlink_to(original / name)
        elif link.resolve() != (original / name).resolve():
            raise ValueError(f"Unexpected existing run asset: {link}")
    if not (canonical / "checkpoints").exists():
        (canonical / "checkpoints").symlink_to(original / "checkpoints", target_is_directory=True)
    check = [str(ROOT / ".venv-psi/bin/python"), "-c",
                    "from pathlib import Path; from psi.rhsci.config import load_run_config; import sys; p=Path(sys.argv[1]); load_run_config(p,p/'checkpoints/ckpt_40000/model.safetensors')",
                    str(canonical)]
    if run["variant"] == "VGQ-AE-D":
        check = [str(ROOT / ".venv-psi/bin/python"), str(Path(__file__).with_name("serve_legacy_psi0.py")), "--check-config", str(canonical)]
    subprocess.run(check, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}, check=True)
    receipt.write_text(json.dumps(dict(run_dir=str(canonical), model=str(checkpoint), revision=REVISION,
                                      horizon=horizon, processor=str(processor)), indent=2) + "\n")
    print("LEGACY_CHECKPOINT_PREPARED", key, flush=True)


def main():
    lock = (STATE / "legacy-preparation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assets.REVISION = REVISION
    processor = old_processor()
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(prepare_one, run, processor): run for run in inventory()}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(dict(run=run_key(futures[future]), error=repr(error)))
                print("PREPARATION_FAILED", failures[-1], flush=True)
    (STATE / "legacy-preparation-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} retest assets require repair; ready models retained")


if __name__ == "__main__":
    main()
