"""Load saved PSI0 configuration without replaying obsolete training CLI options."""

import importlib
import json
import re
import shlex
from pathlib import Path


def restore_checkpoint_method(data, checkpoint):
    """从原专用方法元数据及权重结构恢复公开方法名；只读 header/小型 schema。"""
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    model = data["model"]
    metadata = checkpoint.parent / "controls.json"
    if metadata.is_file():
        recipe = json.loads(metadata.read_text())
        method = recipe["spec"]["method"]
        if method == "STATE-TEACHER-DISTILLATION":
            raise ValueError("State-teacher distillation has been removed")
        if method == "STATE-RECONSTRUCTION":
            if recipe.get("behavior", {}).get("auxiliary_placement") != "shared-ae-query-state31":
                raise ValueError("Legacy single-frame state reconstruction is unsupported")
            model["rsci_variant"] = method
        elif method == "VGQ-VLM":
            if recipe.get("behavior", {}).get("auxiliary_placement") != "direct-vlm-query":
                raise ValueError("Legacy VGQ-VLM Reader checkpoint is unsupported")
            model["rsci_variant"] = method
        model["rsci_ardy_weight"] = recipe["spec"]["auxiliary_weight"]
        if method == "VGQ-VLM":
            model["rsci_execution"] = recipe.get("speed_execution", {}).get("mode", "reference")
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        head = "action_header.rsci.ardy_head.1.weight"
        # 原四帧重建保存 native_variant=VGQ-AE；31维头同时提供结构上的独立证据。
        if model.get("rsci_variant") == "VGQ-AE" and head in keys:
            if weights.get_slice(head).get_shape()[0] == 31:
                model["rsci_variant"] = "STATE-RECONSTRUCTION"
        if model.get("rsci_variant") == "VGQ-VLM":
            if "body_query" not in keys:
                raise ValueError(
                    "Legacy VGQ-VLM Reader checkpoint is unsupported; "
                    "VGQ-VLM now uses four original images and body_query."
                )
        elif model.get("rsci_variant") not in (None, "STATE", "NO-STATE"):
            schema = weights.get_tensor("action_header.rsci.schema_version").item()
            if schema not in (1, 2):
                raise ValueError(f"Unsupported PSI0 Query checkpoint schema: {schema}")
            use_reader = schema == 1
            if "rsci_use_reader" in model and model["rsci_use_reader"] != use_reader:
                raise ValueError("Saved Reader setting disagrees with checkpoint schema")
            model["rsci_use_reader"] = use_reader
    if model.get("rsci_variant") == "STATE-RECONSTRUCTION":
        data["data"]["transform"]["repack"]["rsci_variant"] = "STATE-RECONSTRUCTION"


def load_run_config(run, checkpoint=None):
    run = Path(run)
    argv = shlex.split((run / "argv.txt").read_text())
    if len(argv) < 2 or not re.fullmatch(r"[A-Za-z0-9_]+", argv[1]):
        raise ValueError("Saved argv must identify a PSI0 configuration module")
    module = importlib.import_module(f"psi.config.train.{argv[1]}")
    data = json.loads((run / "run_config.json").read_text())
    model = data["model"]
    # 只兼容直接四图版本的旧保存名；早期同名 Reader 结构不能迁成新模型。
    if model.get("rsci_variant") == "VGQ-VLM-DIRECT":
        model["rsci_variant"] = "VGQ-VLM"
    for feature in ("use_depth", "v5_enabled", "psi0_ardy_aux", "depth_query_independent_rgb"):
        if model.get(feature):
            raise ValueError(f"Checkpoint uses an unsupported historical architecture: {feature}")
    aliases = (
        (data["data"], "v5_ardy_sidecar_root", "rsci_ardy_sidecar_root"),
        (model, "depth_rgb_encoder_path", "rsci_rgb_encoder_path"),
        (model, "depth_encoder_input_size", "rsci_encoder_input_size"),
    )
    for section, old, new in aliases:
        if old in section:
            if new in section and section[new] != section[old]:
                raise ValueError(f"Conflicting saved configuration fields: {old}, {new}")
            section[new] = section.pop(old)
    if checkpoint is not None:
        restore_checkpoint_method(data, checkpoint)
    return module.DynamicLaunchConfig.model_validate(data, context={"restore_preprocessing": True})
