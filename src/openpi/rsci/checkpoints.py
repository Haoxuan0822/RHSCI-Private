"""HF Trainer完整断点验收：先提交新断点，再回收本实验上一份滚动断点。"""

import json
import pickle
import shutil
from pathlib import Path

from rsci.runtime.artifacts import atomic_json, sha256
from rsci.runtime.hf_checkpoints import validate_numerical_state

MARKER = "RSCI_COMPLETE.json"


def required_files(path, world_size):
    names = [
        "rsci_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    ]
    names += (
        ["rng_state.pth"]
        if world_size == 1
        else [f"rng_state_{rank}.pth" for rank in range(world_size)]
    )
    if (path / "model.safetensors.index.json").exists():
        names += ["model.safetensors.index.json"]
        names += sorted(set(json.loads((path / names[-1]).read_text())["weight_map"].values()))
    else:
        names += ["model.safetensors"]
    if any(not (path / name).is_file() or not (path / name).stat().st_size for name in names):
        raise ValueError(f"Incomplete resumable HF checkpoint: {path}")
    return names


def complete_checkpoint(path, world_size, contract_digest):
    path = Path(path)
    names = required_files(path, world_size)
    step = int(path.name.removeprefix("checkpoint-"))
    if json.loads((path / "trainer_state.json").read_text())["global_step"] != step:
        raise ValueError("Checkpoint step disagrees with trainer state")
    try:
        validate_numerical_state(path, names)
    except ValueError as exc:
        # 保存阶段才暴露的非有限权重/动量也不能让守护自动回放旧断点。
        atomic_json(
            path.parent / "NUMERICAL_FAILURE.json",
            dict(step=step, stage="checkpoint", reason=str(exc)),
        )
        raise
    # 数值和文件验收均成功后再发布；坏断点不会触发旧健康滚动断点回收。
    marker = {
        "schema": 2,
        "numerical_validation": "model-and-optimizer-finite-v1",
        "step": step,
        "world_size": world_size,
        "contract": contract_digest,
        "files": {
            name: {"size": (path / name).stat().st_size, "sha256": sha256(path / name)}
            for name in names
        },
    }
    atomic_json(path / MARKER, marker)
    return marker


def checkpoint_ok(path, world_size, contract_digest, *, verify_hash=False):
    path = Path(path)
    try:
        marker = json.loads((path / MARKER).read_text())
        if (
            marker.get("schema") != 2
            or marker.get("numerical_validation") != "model-and-optimizer-finite-v1"
        ):
            return False
        required = required_files(path, world_size)
        if marker["world_size"] != world_size or marker["contract"] != contract_digest:
            return False
        if marker["step"] != int(path.name.removeprefix("checkpoint-")):
            return False
        if set(required) != set(marker["files"]):
            return False
        for name, record in marker["files"].items():
            if (path / name).stat().st_size != record["size"]:
                return False
            if verify_hash and sha256(path / name) != record["sha256"]:
                return False
        if verify_hash:
            validate_numerical_state(path, required)
        return True
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
    ):
        return False


def latest_checkpoint(output, world_size, digest):
    candidates = [
        path
        for path in Path(output).glob("checkpoint-*")
        if path.name.removeprefix("checkpoint-").isdigit()
        and checkpoint_ok(path, world_size, digest)
    ]
    return max(candidates, key=lambda path: int(path.name.split("-")[-1]), default=None)


def prune_rolling(output, current, world_size, digest):
    output, current = Path(output).resolve(), Path(current).resolve()
    if current.parent != output or not checkpoint_ok(current, world_size, digest):
        raise ValueError("Refusing cleanup before a complete new checkpoint")
    for old in output.glob("checkpoint-*"):
        if not old.name.removeprefix("checkpoint-").isdigit():
            continue
        # 只删本契约下、已完成且严格早于新断点的非整万滚动文件，不碰不完整现场。
        step = int(old.name.split("-")[-1])
        if (
            step < int(current.name.split("-")[-1])
            and step % 10000
            and checkpoint_ok(old, world_size, digest)
        ):
            if old.is_symlink() or old.resolve().parent != output:
                raise ValueError("Unsafe checkpoint cleanup target")
            shutil.rmtree(old)
