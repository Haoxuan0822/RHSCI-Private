"""推理只检查配置和模型权重；优化器、调度器与 RNG 属于续训状态。"""

import json
from pathlib import Path

from rhsci.runtime.artifacts import sha256


def validate_inference_weights(checkpoint, config="rsci_config.json", *, verify_hash=False):
    """检查单文件/分片权重非空；显式要求时核对已有清单中的模型 SHA。"""
    checkpoint = Path(checkpoint)
    json.loads((checkpoint / config).read_text())
    index = checkpoint / "model.safetensors.index.json"
    names = (
        [index.name, *sorted(set(json.loads(index.read_text())["weight_map"].values()))]
        if index.is_file()
        else ["model.safetensors"]
    )
    for name in names:
        path = checkpoint / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid model shard path: {name}")
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty model weights: {path}")
    marker = checkpoint / "RSCI_COMPLETE.json"
    if marker.is_file():
        records = json.loads(marker.read_text())["files"]
        for name in names:
            record = records.get(name)
            if record is not None and (
                (checkpoint / name).stat().st_size != record["size"]
                or (verify_hash and sha256(checkpoint / name) != record["sha256"])
            ):
                raise ValueError(f"Model weight checksum/size mismatch: {name}")
    return checkpoint
