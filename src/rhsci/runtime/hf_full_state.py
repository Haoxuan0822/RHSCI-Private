"""四个 HF 骨干共用完整断点流程；文件清单、schema 和模型载入由骨干指定。"""

import json
import pickle
import shutil
from pathlib import Path

from rhsci.runtime.artifacts import atomic_json, sha256

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


class HFCheckpointProtocol:
    """共享文件/SHA/数值验收与保留策略，不解释模型权重含义。

    参数schema隔离不同骨干格式；validate在调用时解析骨干数值门禁。
    """

    def __init__(
        self,
        schema,
        validate,
        *,
        required_files=None,
        world_key="world",
        marker_fields=None,
        verify_before_prune=True,
        record_numerical_failure=False,
    ):
        self.schema = schema
        self.validate = validate
        self.required_files = required_files or (lambda path, world: required(world))
        self.world_key = world_key
        self.marker_fields = marker_fields or {}
        self.verify_before_prune = verify_before_prune
        self.record_numerical_failure = record_numerical_failure

    def publish(self, path, world, digest):
        path = Path(path)
        step = int(path.name.removeprefix("checkpoint-"))
        if json.loads((path / "trainer_state.json").read_text())["global_step"] != step:
            raise ValueError("Checkpoint step mismatch")
        names = self.required_files(path, world)
        try:
            self.validate(path)
        except ValueError as error:
            if self.record_numerical_failure:
                atomic_json(
                    path.parent / "NUMERICAL_FAILURE.json",
                    dict(step=step, stage="checkpoint", reason=str(error)),
                )
            raise
        files = {
            name: dict(size=(path / name).stat().st_size, sha256=sha256(path / name))
            for name in names
        }
        if any(not record["size"] for record in files.values()):
            raise ValueError("Empty checkpoint payload")
        marker = dict(
            schema=self.schema,
            step=step,
            contract=digest,
            files=files,
            **{self.world_key: world},
            **self.marker_fields,
        )
        atomic_json(path / MARKER, marker)
        return marker

    def checkpoint_ok(self, path, world, digest, verify=False, *, verify_hash=None):
        if verify_hash is not None:
            verify = verify_hash
        path = Path(path)
        try:
            marker = json.loads((path / MARKER).read_text())
            if (
                marker["schema"] != self.schema
                or marker[self.world_key] != world
                or marker["contract"] != digest
                or marker["step"] != int(path.name.removeprefix("checkpoint-"))
                or set(marker["files"]) != set(self.required_files(path, world))
                or any(marker.get(key) != value for key, value in self.marker_fields.items())
            ):
                return False
            for name, record in marker["files"].items():
                if record["size"] != (path / name).stat().st_size or not record["size"]:
                    return False
                if verify and sha256(path / name) != record["sha256"]:
                    return False
            if verify:
                self.validate(path)
            return True
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            EOFError,
            FloatingPointError,
            pickle.UnpicklingError,
        ):
            return False

    def latest(self, output, world, digest):
        paths = [
            path
            for path in Path(output).glob("checkpoint-*")
            if path.name.removeprefix("checkpoint-").isdigit()
            and self.checkpoint_ok(path, world, digest)
        ]
        return max(paths, key=lambda path: int(path.name.split("-")[-1]), default=None)

    def prune(self, output, current, world, digest):
        output, current = Path(output).resolve(), Path(current).resolve()
        if current.parent != output or not self.checkpoint_ok(
            current, world, digest, verify=self.verify_before_prune
        ):
            raise ValueError("New checkpoint must be validated before rolling cleanup")
        for old in output.glob("checkpoint-*"):
            if not old.name.removeprefix("checkpoint-").isdigit():
                continue
            step = int(old.name.split("-")[-1])
            if (
                step < int(current.name.split("-")[-1])
                and step % 10000
                and self.checkpoint_ok(old, world, digest)
            ):
                if old.is_symlink() or old.resolve().parent != output:
                    raise ValueError("Unsafe rolling checkpoint target")
                shutil.rmtree(old)


def model_weight_files(path):
    """返回原生 HF 单文件或分片权重清单，供训练完整状态校验使用。"""
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        return [index.name, *sorted(set(json.loads(index.read_text())["weight_map"].values()))]
    return ["model.safetensors"]


class HFTrainerCheckpointProtocol(HFCheckpointProtocol):
    """π0.5/GR00T 的 schema=2 格式；仅元数据文件集合随骨干变化。"""

    def __init__(self, metadata_files):
        self.metadata_files = tuple(metadata_files)
        super().__init__(
            2,
            self.validate_model_and_optimizer,
            required_files=self.required_files,
            world_key="world_size",
            marker_fields={"numerical_validation": "model-and-optimizer-finite-v1"},
            verify_before_prune=False,
            record_numerical_failure=True,
        )

    def required_files(self, path, world_size):
        path = Path(path)
        names = [
            *self.metadata_files,
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
        names += model_weight_files(path)
        if any(not (path / name).is_file() or not (path / name).stat().st_size for name in names):
            raise ValueError(f"Incomplete resumable HF checkpoint: {path}")
        return names

    def validate_model_and_optimizer(self, path):
        from rhsci.runtime.hf_checkpoints import validate_numerical_state

        validate_numerical_state(path, model_weight_files(path))
