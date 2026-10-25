"""Load saved PSI0 configuration without replaying obsolete training CLI options."""

import importlib
import json
import re
import shlex
from pathlib import Path


def load_run_config(run, checkpoint=None):
    run = Path(run)
    argv = shlex.split((run / "argv.txt").read_text())
    if len(argv) < 2 or not re.fullmatch(r"[A-Za-z0-9_]+", argv[1]):
        raise ValueError("Saved argv must identify a PSI0 configuration module")
    module = importlib.import_module(f"psi.config.train.{argv[1]}")
    data = json.loads((run / "run_config.json").read_text())
    model = data["model"]
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
    if checkpoint is not None and model.get("rsci_variant") not in (None, "STATE", "NO-STATE"):
        from safetensors import safe_open

        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            schema = weights.get_tensor("action_header.rsci.schema_version").item()
        if schema not in (1, 2):
            raise ValueError(f"Unsupported PSI0 Query checkpoint schema: {schema}")
        use_reader = schema == 1
        if "rsci_use_reader" in model and model["rsci_use_reader"] != use_reader:
            raise ValueError("Saved Reader setting disagrees with checkpoint schema")
        model["rsci_use_reader"] = use_reader
    return module.DynamicLaunchConfig.model_validate(data)
