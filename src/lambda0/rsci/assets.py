"""读取本地 λ₀ 初始化资产；完整摘要核验仅用于显式资产检查。"""

import hashlib
import json
from pathlib import Path

from .contracts import sha256


def verify_assets(root, *, verify=False):
    """记录配置及已有权重；可选来源收据不限制用户选择其他原生初始化目录。"""
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    metadata_path = root / "RSCI_HF_SOURCE.json"
    if not metadata_path.is_file():
        weights = sorted(path for path in root.rglob("*.safetensors") if path.is_file())
        if not weights:
            raise FileNotFoundError(f"No model weights: {root}")
        return dict(
            config=config,
            files={
                path.relative_to(root).as_posix(): dict(bytes=path.stat().st_size)
                for path in weights
            },
        )
    metadata = json.loads(metadata_path.read_text())
    records = {}
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        file = root / name
        if not file.is_file() or file.stat().st_size != entry["size"]:
            raise ValueError(f"Initialization file missing or size mismatch: {name}")
        if entry.get("lfs"):
            digest = entry["lfs"]["sha256"]
            if verify and sha256(file) != digest:
                raise ValueError(f"Initialization LFS hash mismatch: {name}")
        else:
            contents = file.read_bytes()
            if verify:
                blob = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
                if blob != entry["blobId"]:
                    raise ValueError(f"Initialization Git blob mismatch: {name}")
            digest = hashlib.sha256(contents).hexdigest()
        records[name] = dict(bytes=file.stat().st_size, sha256=digest)
    return dict(repo=metadata["id"], revision=metadata["sha"], files=records)
