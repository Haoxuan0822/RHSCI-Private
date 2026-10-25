"""ACT完整模型（包括FrozenBN buffers）和HF完整状态；独立版本契约。"""

import json
import shutil
from pathlib import Path

from rsci.runtime.artifacts import atomic_json, sha256

MARKER = "RSCI_COMPLETE.json"


def finite_state(value):
    import torch

    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        return all(
            torch.isfinite(chunk).all().item() for chunk in value.reshape(-1).split(1024 * 1024)
        )
    if isinstance(value, dict):
        return all(finite_state(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(child) for child in value)
    return True


def required(world):
    return [
        "model.safetensors",
        "rsci_config.json",
        "launch_contract.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    ] + (["rng_state.pth"] if world == 1 else [f"rng_state_{rank}.pth" for rank in range(world)])


def validate_numerics(path):
    import torch
    from safetensors.torch import load_file

    if not finite_state(load_file(str(path / "model.safetensors"))) or not finite_state(
        torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    ):
        raise FloatingPointError("Nonfinite checkpoint model or Adam state")


def publish(path, world, digest):
    path = Path(path)
    step = int(path.name.removeprefix("checkpoint-"))
    if json.loads((path / "trainer_state.json").read_text())["global_step"] != step:
        raise ValueError("Checkpoint step mismatch")
    validate_numerics(path)
    files = {
        name: dict(size=(path / name).stat().st_size, sha256=sha256(path / name))
        for name in required(world)
    }
    if any(not record["size"] for record in files.values()):
        raise ValueError("Empty checkpoint payload")
    atomic_json(
        path / MARKER,
        dict(schema="act-full-state-v1", step=step, world=world, contract=digest, files=files),
    )


def checkpoint_ok(path, world, digest, verify=False):
    path = Path(path)
    try:
        marker = json.loads((path / MARKER).read_text())
        if (
            marker["schema"] != "act-full-state-v1"
            or marker["world"] != world
            or marker["contract"] != digest
            or marker["step"] != int(path.name.removeprefix("checkpoint-"))
            or set(marker["files"]) != set(required(world))
        ):
            return False
        for name, record in marker["files"].items():
            if record["size"] != (path / name).stat().st_size or not record["size"]:
                return False
            if verify and sha256(path / name) != record["sha256"]:
                return False
        if verify:
            validate_numerics(path)
        return True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, FloatingPointError):
        return False


def latest(output, world, digest):
    paths = [
        path
        for path in Path(output).glob("checkpoint-*")
        if path.name.removeprefix("checkpoint-").isdigit() and checkpoint_ok(path, world, digest)
    ]
    return max(paths, key=lambda path: int(path.name.split("-")[-1]), default=None)


def prune(output, current, world, digest):
    output, current = Path(output).resolve(), Path(current).resolve()
    if current.parent != output or not checkpoint_ok(current, world, digest, verify=True):
        raise ValueError("New checkpoint must be validated before rolling cleanup")
    for old in output.glob("checkpoint-*"):
        if not old.name.removeprefix("checkpoint-").isdigit():
            continue
        step = int(old.name.split("-")[-1])
        if (
            step < int(current.name.split("-")[-1])
            and step % 10000
            and checkpoint_ok(old, world, digest)
        ):
            if old.is_symlink() or old.resolve().parent != output:
                raise ValueError("Unsafe rolling checkpoint target")
            shutil.rmtree(old)


def load_model_state(model, path):
    from safetensors.torch import load_file

    state = load_file(str(Path(path) / "model.safetensors"))
    expected = model.state_dict()
    if set(state) != set(expected) or any(
        state[name].shape != value.shape for name, value in expected.items()
    ):
        raise ValueError("ACT full checkpoint keys/shapes mismatch")
    if json.loads(
        (Path(path) / "rsci_config.json").read_text()
    ) != model.rsci_config or not finite_state(state):
        raise ValueError("ACT checkpoint architecture/numerical mismatch")
    model.load_state_dict(state, strict=True)
