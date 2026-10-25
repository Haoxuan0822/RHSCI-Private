"""Discover complete RHSCI HF checkpoints without a private run registry."""

import hashlib
import importlib
import json
from pathlib import Path

from rsci.runtime.artifacts import sha256

BACKBONES = ("pi05", "gr00t")
ROOT = Path(__file__).resolve().parents[3]


def historical_pi05_checkpoint_ok(checkpoint, contract, digest, *, verify_hash):
    """Admit schema-1 inference after explicit original-source verification."""
    from rsci.backbones.pi05.checkpoints import required_files

    try:
        marker = json.loads((checkpoint / "RSCI_COMPLETE.json").read_text())
        step = int(checkpoint.name.removeprefix("checkpoint-"))
        if (
            marker.get("schema") != 1
            or marker["world_size"] != contract["world_size"]
            or marker["contract"] != digest
            or marker["step"] != step
            or json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] != step
        ):
            return False
        names = required_files(checkpoint, contract["world_size"])
        if set(names) != set(marker["files"]):
            return False
        for name in names:
            record = marker["files"][name]
            if (checkpoint / name).stat().st_size != record["size"]:
                return False
            if verify_hash and sha256(checkpoint / name) != record["sha256"]:
                return False
        if verify_hash:
            from rsci.runtime.hf_checkpoints import validate_numerical_state

            validate_numerical_state(checkpoint, names)
        return True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError):
        return False


def inspect_run(run, backbone, *, step=None, verify_hash=False, training_source_root=None):
    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported HTTP backbone: {backbone}")
    run = Path(run).resolve(strict=True)
    contract = json.loads((run / "launch_contract.json").read_text())
    sources = contract["code" if backbone == "pi05" else "sources"]
    source_root = (
        ROOT if training_source_root is None else Path(training_source_root).resolve(strict=True)
    )
    if training_source_root is not None and backbone != "pi05":
        raise ValueError("Historical training-source compatibility is supported for π0.5 only")
    for name, expected in sources.items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not (source_root / path).is_file()
            or sha256(source_root / path) != expected
        ):
            raise ValueError(f"Training/inference source mismatch: {name}")
    expected_horizon = 30 if backbone == "pi05" else 16
    if (
        contract.get("horizon" if backbone == "pi05" else "valid_action_horizon")
        != expected_horizon
    ):
        raise ValueError("Training action horizon differs from the serving protocol")
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    module = importlib.import_module(f"rsci.backbones.{backbone}.checkpoints")
    candidates = [run / f"checkpoint-{step}"] if step is not None else run.glob("checkpoint-*")
    candidates = sorted(
        (path for path in candidates if path.name.removeprefix("checkpoint-").isdigit()),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
        reverse=True,
    )
    for checkpoint in candidates:
        if checkpoint.is_symlink():
            continue
        if any(path.is_symlink() for path in checkpoint.rglob("*")):
            continue
        valid = module.checkpoint_ok(
            checkpoint, contract["world_size"], digest, verify_hash=verify_hash
        )
        if not valid and training_source_root is not None:
            valid = historical_pi05_checkpoint_ok(
                checkpoint, contract, digest, verify_hash=verify_hash
            )
        if valid:
            return contract, checkpoint
    raise ValueError("No complete checkpoint matches the training contract")


def discover(runs_root, backbone, *, training_source_root=None):
    ready, rejected = [], []
    for record in sorted(Path(runs_root).rglob("launch_contract.json")):
        try:
            contract, checkpoint = inspect_run(
                record.parent, backbone, training_source_root=training_source_root
            )
            ready.append(
                dict(run=str(record.parent), checkpoint=str(checkpoint), model=contract["model"])
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            rejected.append(dict(run=str(record.parent), reason=str(error)))
    return dict(ready=ready, rejected=rejected)
