"""Serve the archived retained-Query PSI0-D implementation with the current HTTP shell."""

import importlib.util
import sys
from pathlib import Path

from plan import REPO, ROOT

SOURCE = ROOT / "external/source-archive-20261025/worktrees/24-ARDY_PSCI_D/src"


def bootstrap():
    # This historical architecture retains its Query at inference. Do not alias it to VGQ-AE.
    sys.path.insert(0, str(SOURCE))
    sys.path.append(str(REPO / "src"))
    import psi

    psi.__path__.append(str(REPO / "src/psi"))
    from psi.models.psi0 import Psi0Model

    Psi0Model.rhsci_history_from_request = Psi0Model.rsci_history_from_request
    from psi.config.transform import ActionStateTransform

    original = ActionStateTransform.model_post_init

    def keep_saved_bounds(self, context):
        if context and context.get("restore_preprocessing"):
            if self.action_min is None or self.action_max is None:
                raise ValueError("Historical checkpoint is missing saved normalization")
            return
        return original(self, context)

    ActionStateTransform.model_post_init = keep_saved_bounds
    return Psi0Model


def main():
    bootstrap()
    if len(sys.argv) == 3 and sys.argv[1] == "--check-config":
        from psi.rhsci.config import load_run_config

        run = Path(sys.argv[2])
        checkpoint = run / "checkpoints/ckpt_40000/model.safetensors"
        cfg = load_run_config(run, checkpoint if checkpoint.exists() else None)
        if cfg.model.rsci_variant != "(VGQ-AE-D)":
            raise ValueError("Historical runtime is reserved for retained-Query PSI0-D")
        print("LEGACY_CONFIG_OK", cfg.model.rsci_variant, cfg.model.action_chunk_size)
        return
    # Use the maintained server's request handling, normalization and RNG defaults.
    spec = importlib.util.spec_from_file_location("legacy_psi0_http", REPO / "src/psi/deploy/psi0_serve_simple.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import tyro
    from psi.config.config import ServerConfig

    cfg = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    module.serve(cfg)


if __name__ == "__main__":
    main()
