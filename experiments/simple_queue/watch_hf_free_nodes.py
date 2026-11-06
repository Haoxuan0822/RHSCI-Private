"""每小时验证免费节点；恢复真实 HF 传输后下载新增权重，不改活动评测资产。"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote

import requests
import yaml

from plan import ROOT

STATE = ROOT / "evals/goal-latest-protocol-20261105/hf-free-watch"
BASE = ROOT / "hf/RHSCI/model-59a1ea14"
PROXY = "http://127.0.0.1:7890"
GROUP = "📥 下载专用"


def save(name, value):
    target = STATE / name
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)


def eligible(record):
    parts = record["rfilename"].split("/")
    return (
        len(parts) >= 5 and parts[0] == "checkpoint"
        and parts[1].lower() != "act" and parts[2].strip("()") != "VGQ-AE-D"
        and not parts[-1].startswith("random_states_")
        and parts[-1] not in {"optimizer.bin", "optimizer.pt", "scheduler.bin"}
    )


def identity(record):
    return record.get("lfs", {}).get("sha256", record.get("blobId"))


def check():
    config = yaml.safe_load(Path("/root/workspace/Haoxuan/mihomo/config/config.yaml").read_text())
    control = requests.Session()
    control.trust_env = False
    control.headers["Authorization"] = "Bearer " + str(config.get("secret", ""))
    endpoint = "http://127.0.0.1:9090/proxies/" + quote(GROUP, safe="")
    response = control.get(endpoint, timeout=10)
    response.raise_for_status()
    nodes = response.json()["all"]
    nodes = [n for n in nodes if "免费" in n]
    if not nodes:
        raise RuntimeError("No explicitly free nodes in download selector")
    attempts = []
    for node in nodes:
        selected = control.put(endpoint, json={"name": node}, timeout=10)
        selected.raise_for_status()
        session = requests.Session()
        session.trust_env = False
        session.proxies = {"http": PROXY, "https": PROXY}
        try:
            response = session.get("https://huggingface.co/api/models/ds1x/RHSCI?blobs=true", timeout=(8, 15))
            response.raise_for_status()
            metadata = response.json()
            revision = metadata["sha"]
            records = [r for r in metadata["siblings"] if eligible(r)]
            probe = min((r for r in records if r.get("lfs") and r["size"] >= 65536), key=lambda r: r["size"])
            url = f"https://huggingface.co/ds1x/RHSCI/resolve/{revision}/" + quote(probe["rfilename"], safe="/")
            with session.get(url, headers={"Range": "bytes=0-65535"}, timeout=(8, 15), stream=True) as transfer:
                transfer.raise_for_status()
                if transfer.status_code != 206 or not transfer.headers.get("Content-Range", "").startswith("bytes 0-65535/"):
                    raise RuntimeError("Range request was not honored")
                payload = b"".join(transfer.iter_content(65536))
                if len(payload) != 65536:
                    raise RuntimeError("Incomplete probe payload")
            old = json.loads((BASE / "DOWNLOAD_MANIFEST-3fdbb4c7.json").read_text())
            previous = {r["rfilename"]: r for r in old["files"]}
            changed = [r for r in records if r["rfilename"] not in previous or identity(r) != identity(previous[r["rfilename"]])]
            selection = dict(time=time.time(), repo="ds1x/RHSCI", revision=revision,
                             node=node, baseline_revision=old["revision"], files=changed)
            path = f"selection-{revision}.json"
            save(path, selection)
            save("last-check.json", dict(time=time.time(), available=True, node=node,
                                         revision=revision, new_or_changed_files=len(changed), attempts=attempts))
            return STATE / path
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as error:
            attempts.append(dict(node=node, error=type(error).__name__))
        finally:
            session.close()
    save("last-check.json", dict(time=time.time(), available=False, attempts=attempts))
    return None


def download(manifest):
    selection = json.loads(manifest.read_text())
    destination = ROOT / "hf/RHSCI" / ("model-" + selection["revision"])
    destination.mkdir(exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": PROXY, "https": PROXY}
    for record in selection["files"]:
        relative = record["rfilename"]
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/ds1x/RHSCI/resolve/{selection['revision']}/" + quote(relative, safe="/")
        if record.get("lfs"):
            subprocess.run(["/root/miniforge/bin/python3", str(ROOT / "download_wheel.py"),
                            url, str(target), record["lfs"]["sha256"], "--workers", "4"], check=True)
        else:
            import hashlib
            response = session.get(url, timeout=60)
            response.raise_for_status()
            payload = response.content
            digest = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            if len(payload) != record["size"] or digest != record["blobId"]:
                raise ValueError("Metadata digest mismatch")
            temporary = target.with_suffix(target.suffix + ".download")
            temporary.write_bytes(payload)
            temporary.replace(target)
        print("VERIFIED", relative, flush=True)
    save(f"complete-{selection['revision']}.json", dict(revision=selection["revision"],
         time=time.time(), files=len(selection["files"]), destination=str(destination)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-manifest", type=Path)
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    if args.download_manifest:
        download(args.download_manifest)
        return
    lock = (STATE / "watch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    while True:
        started = time.monotonic()
        try:
            selection = check()
            if selection and (child is None or child.poll() is not None):
                info = json.loads(selection.read_text())
                if info["files"] and not (STATE / f"complete-{info['revision']}.json").exists():
                    env = dict(os.environ, HTTP_PROXY=PROXY, HTTPS_PROXY=PROXY,
                               http_proxy=PROXY, https_proxy=PROXY, NO_PROXY="127.0.0.1,localhost")
                    with (STATE / "download.log").open("a") as log:
                        child = subprocess.Popen([sys.executable, "-u", __file__, "--download-manifest", str(selection)],
                                                 env=env, stdout=log, stderr=subprocess.STDOUT)
                    save("download-process.json", dict(pid=child.pid, revision=info["revision"], time=time.time()))
        except Exception as error:
            save("last-error.json", dict(time=time.time(), error=type(error).__name__))
        while time.monotonic() - started < 3600:
            time.sleep(min(60, 3600 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
