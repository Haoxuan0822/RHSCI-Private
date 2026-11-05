"""Discover RHSCI model checkpoints for inference, independently of training state."""

import json
from pathlib import Path

from rhsci.runtime.artifacts import sha256

from .weights import validate_inference_weights

BACKBONES = ("pi05", "gr00t")
ROOT = Path(__file__).resolve().parents[3]


def validate_evaluation_variant(expected, actual, *, ardy_offset_frames=0):
    """测评标签必须匹配检查点方法；独立服务未指定期望时保持原行为。"""
    if expected is None:
        return
    actual = actual or "STATE"
    if ardy_offset_frames:
        actual = f"{actual}-F{ardy_offset_frames}"
    if expected != actual:
        raise ValueError(f"Evaluation variant {expected!r} differs from checkpoint {actual!r}")


def inspect_run(run, backbone, *, step=None, verify_hash=False, training_source_root=None):
    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported HTTP backbone: {backbone}")
    run = Path(run).resolve(strict=True)
    contract = json.loads((run / "launch_contract.json").read_text())
    # 旧参数仅用于显式审计历史来源；默认推理不绑定训练 checkout。
    if training_source_root is not None:
        sources = contract.get("code" if backbone == "pi05" else "sources", {})
        source_root = Path(training_source_root).resolve(strict=True)
        for name, expected in sources.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or sha256(source_root / path) != expected:
                raise ValueError(f"Training/inference source mismatch: {name}")
    expected_horizon = 30 if backbone == "pi05" else 16
    if (
        contract.get("horizon" if backbone == "pi05" else "valid_action_horizon")
        != expected_horizon
    ):
        raise ValueError("Training action horizon differs from the serving protocol")
    candidates = [run / f"checkpoint-{step}"] if step is not None else run.glob("checkpoint-*")
    candidates = sorted(
        (path for path in candidates if path.name.removeprefix("checkpoint-").isdigit()),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
        reverse=True,
    )
    for checkpoint in candidates:
        try:
            validate_inference_weights(
                checkpoint,
                config="config.json" if backbone == "gr00t" else "rsci_config.json",
                verify_hash=verify_hash,
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
        return contract, checkpoint
    raise ValueError("No complete model checkpoint is available")


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
