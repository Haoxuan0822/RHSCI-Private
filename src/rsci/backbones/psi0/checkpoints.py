"""Structural and numerical validation of native PSI0 training checkpoints."""

import json
import pickle
import struct
import zipfile
from pathlib import Path

from rsci.runtime.checkpoints import check_tensors

_CACHE = {}


def validate_numerical_state(path):
    import torch
    from safetensors import safe_open

    path = Path(path)
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as stream:
        for key in stream.keys():
            check_tensors(stream.get_tensor(key), "model/" + key)
    for name in ("optimizer.bin", "scheduler.bin"):
        check_tensors(
            torch.load(path / name, map_location="cpu", weights_only=True, mmap=True), name
        )


def numerical_checkpoint_ok(path):
    """缓存仅复用同一inode/大小/mtime/ctime，文件变化必定重验；不写历史断点。"""
    from safetensors import SafetensorError

    path = Path(path)
    try:
        files = [path / name for name in ("model.safetensors", "optimizer.bin", "scheduler.bin")]
        signature = tuple(
            (str(p), p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
            for p in files
        )
        if signature in _CACHE:
            return _CACHE[signature]
        try:
            validate_numerical_state(path)
            result = True
        except (
            OSError,
            ValueError,
            RuntimeError,
            EOFError,
            TypeError,
            KeyError,
            pickle.UnpicklingError,
            SafetensorError,
        ):
            result = False
        _CACHE.clear() if len(_CACHE) > 128 else None
        _CACHE[signature] = result
        return result
    except OSError:
        return False


def checkpoint_ok(path, ranks=8):
    """结构及CPU mmap分块数值验收；不能替代完整恢复或全文件SHA校验。"""
    try:
        required = ["model.safetensors", "optimizer.bin", "scheduler.bin"]
        required += [f"random_states_{rank}.pkl" for rank in range(ranks)]
        if path.is_symlink() or any((path / name).is_symlink() for name in required):
            return False
        if any(
            not (path / name).is_file() or (path / name).stat().st_size == 0 for name in required
        ):
            return False
        with (path / "model.safetensors").open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 32 * 1024 * 1024:
                return False
            header = json.loads(stream.read(length))
        offsets = [value["data_offsets"] for key, value in header.items() if key != "__metadata__"]
        if not offsets or any(start < 0 or end < start for start, end in offsets):
            return False
        if (
            8 + length + max(end for _, end in offsets)
            != (path / "model.safetensors").stat().st_size
        ):
            return False
        for name in ("optimizer.bin", "scheduler.bin"):
            with zipfile.ZipFile(path / name) as archive:
                if not archive.namelist() or not any(
                    item.endswith("data.pkl") for item in archive.namelist()
                ):
                    return False
        return numerical_checkpoint_ok(path)
    except (OSError, ValueError, KeyError, TypeError, struct.error, zipfile.BadZipFile):
        return False
