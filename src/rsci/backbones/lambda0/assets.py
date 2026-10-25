"""Stage-II 本地资产的完整官方文件集合与摘要验收，不自动下载。"""

import hashlib
import json
from pathlib import Path

from .contracts import sha256

REVISION = "2878ebd4a236514f41215d2a4c4af05ca3dd177a"


def verify_assets(root):
    root = Path(root)
    metadata = json.loads((root / "RSCI_HF_SOURCE.json").read_text())
    if metadata["sha"] != REVISION or metadata["id"] != "lambda-humanoid/lambda0-midtrain-100k":
        raise ValueError("Stage-II source identity mismatch")
    expected = {row["rfilename"]: row for row in metadata["siblings"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(root).parts
        and path.name not in ("RSCI_HF_SOURCE.json", "RSCI_ASSETS.json")
    }
    if actual != set(expected):
        raise ValueError("Stage-II asset inventory mismatch")
    records = {}
    for name, entry in expected.items():
        file = root / name
        if file.stat().st_size != entry["size"]:
            raise ValueError(f"Stage-II size mismatch: {name}")
        if entry.get("lfs"):
            digest = sha256(file)
            if digest != entry["lfs"]["sha256"]:
                raise ValueError(f"Stage-II LFS hash mismatch: {name}")
        else:
            contents = file.read_bytes()
            blob = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
            if blob != entry["blobId"]:
                raise ValueError(f"Stage-II Git blob mismatch: {name}")
            digest = hashlib.sha256(contents).hexdigest()
        records[name] = dict(bytes=file.stat().st_size, sha256=digest)
    return dict(repo="lambda-humanoid/lambda0-midtrain-100k", revision=REVISION, files=records)
