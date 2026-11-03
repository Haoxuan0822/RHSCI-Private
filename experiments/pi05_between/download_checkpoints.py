"""Prioritize the two requested π0.5 inference checkpoints at the pinned HF revision."""

import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

root = Path("/root/workspace/Haoxuan/RHSCI-runtime")
destination = root / "hf/RHSCI/model-59a1ea14"
manifest = json.loads((destination / "DOWNLOAD_MANIFEST.json").read_text())
vgq = "checkpoint/pi05/VGQ-AE/betweentables/frozen/simple-default-v4-speed-both-v1/"
state = "checkpoint/PI05/STATE/between/priority-rgb-reader-20261028-v1/checkpoint-40000.tar.zst"
needed = {
    "model.safetensors",
    "rsci_config.json",
    "launch_contract.json",
    "checkpoint-manifest.json",
    "RSCI_COMPLETE.json",
}
records = [
    r
    for r in manifest["files"]
    if r["rfilename"] == state
    or (r["rfilename"].startswith(vgq) and Path(r["rfilename"]).name in needed)
    or r["rfilename"].endswith("/PI05-STATE-between-40k.json")
]
records.sort(key=lambda r: r["size"])
print("PRIORITY_FILES", len(records), "BYTES", sum(r["size"] for r in records), flush=True)


def fetch(record):
    path = destination / record["rfilename"]
    url = f"https://huggingface.co/ds1x/RHSCI/resolve/{manifest['revision']}/{record['rfilename']}"
    for attempt in range(20):
        try:
            if record.get("lfs"):
                subprocess.run(
                    [
                        "/root/miniforge/bin/python3",
                        str(root / "download_wheel.py"),
                        url,
                        str(path),
                        record["lfs"]["sha256"],
                        "--workers",
                        "12",
                    ],
                    check=True,
                )
            else:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                payload = response.content
                assert len(payload) == record["size"]
                assert (
                    hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
                    == record["blobId"]
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            print("PRIORITY_VERIFIED", record["rfilename"], flush=True)
            return
        except Exception as exc:
            print("RETRY", record["rfilename"], attempt, str(exc)[:200], flush=True)
            if attempt == 19:
                raise
            time.sleep(min(5 * (attempt + 1), 60))


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, records))
(root / "logs/pi05-between-priority-verified.json").write_text(
    json.dumps({"revision": manifest["revision"], "files": records}, indent=2) + "\n"
)
print("PI05_PRIORITY_COMPLETE", flush=True)
