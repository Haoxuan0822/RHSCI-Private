"""Prepare the five NO-STATE evaluations authorized on 2026-11-05."""

import concurrent.futures
import hashlib
import json
import tarfile
import time

from plan import PREPARATION, ROOT, run_key

REVISION = "3fdbb4c714d9ca70abf4b9e9bbe38cabe7781990"
RUNS = [
    dict(backbone="psi0", variant="NO-STATE", task=task, step=40000, horizon=30,
         archive=f"checkpoint/Psi0/NO-STATE/{task}/psi0-rgb-nostate-three-tasks-20261103-v1/checkpoint-40000.tar.zst")
    for task in ("oven", "handover", "faucet")
] + [
    dict(backbone="gr00t", variant="NO-STATE", task=task, step=20000, horizon=16,
         archive=f"checkpoint/Gr00t/NO-STATE/{task}/paper20k-nostate-spacecache-v1/checkpoint-20000.tar.zst")
    for task in ("between", "hug")
]


def prepare_psi0(run, record):
    """Keep the saved bundle unchanged and expose the native PSI0 run layout."""
    import zstandard
    import prepare as assets

    key = run_key(run)
    receipt = PREPARATION / f"{key}.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    archive = assets.MODEL_ROOT / run["archive"]
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record["lfs"]["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    destination = ROOT / "checkpoints/followup" / key
    staging = destination.with_name(key + ".extracting")
    staging.mkdir(exist_ok=False)
    with archive.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as source:
            for member in source:
                source.extract(member, staging, filter="data")
    configs = list(staging.rglob("run_config.json"))
    checkpoints = list(staging.rglob(f"ckpt_{run['step']}/model.safetensors"))
    argvs = list(staging.rglob("argv.txt"))
    if len(configs) != 1 or len(checkpoints) != 1 or len(argvs) != 1:
        raise ValueError(f"Ambiguous native PSI0 bundle: {key}")
    cfg = json.loads(configs[0].read_text())
    model = cfg["model"]
    horizon = model["action_chunk_size"]
    method = {"VGQ-VLM-DIRECT": "VGQ-VLM"}.get(model["rsci_variant"], model["rsci_variant"])
    if method != run["variant"] or (run["horizon"] is not None and horizon != run["horizon"]):
        raise ValueError(f"Saved PSI0 configuration mismatch: {key}")
    if model["action_exec_horizon"] != horizon:
        raise ValueError(f"Saved prediction/execution differ: {key}")
    canonical = staging / "inference"
    canonical.mkdir()
    for name, target in (("run_config.json", configs[0]), ("argv.txt", argvs[0])):
        (canonical / name).symlink_to("../" + str(target.relative_to(staging)))
    (canonical / "checkpoints").mkdir()
    (canonical / "checkpoints" / f"ckpt_{run['step']}").symlink_to("../../" + str(checkpoints[0].parent.relative_to(staging)), target_is_directory=True)
    staging.rename(destination)
    info = dict(run_dir=str(destination / "inference"), archive=str(archive), sha256=digest,
                revision=REVISION, saved_model=model, horizon=horizon)
    receipt.write_text(json.dumps(info, indent=2) + "\n")
    print("CHECKPOINT_PREPARED", key, flush=True)
    return info


def main():
    import prepare as assets

    manifest = json.loads((assets.MODEL_ROOT / "DOWNLOAD_MANIFEST.json").read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("Unexpected download manifest revision")
    records = {r["rfilename"]: r for r in manifest["files"]}
    assets.REVISION = REVISION
    pending = list(RUNS)
    active = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while pending or active:
            for future, run in list(active.items()):
                if future.done():
                    future.result()
                    del active[future]
            for run in list(pending):
                archive = assets.MODEL_ROOT / run["archive"]
                if not archive.is_file() or archive.stat().st_size != records[run["archive"]]["size"]:
                    continue
                if len(active) >= 3:
                    break
                future = pool.submit(prepare_psi0, run, records[run["archive"]]) if run["backbone"] == "psi0" else pool.submit(assets.prepare, run, records)
                active[future] = run
                pending.remove(run)
            if pending or active:
                time.sleep(10)


if __name__ == "__main__":
    main()
