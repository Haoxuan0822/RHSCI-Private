"""Register local π0.5 assets and install or verify the Transformers overlay."""

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

from rsci.runtime.artifacts import atomic_json, sha256


def register_assets(checkpoint, tokenizer, repo, revision):
    """Record a local inventory; this does not authenticate its upstream origin."""
    import sentencepiece
    from safetensors import safe_open

    checkpoint, tokenizer = Path(checkpoint).resolve(), Path(tokenizer).resolve()
    names = ("config.json", "model.safetensors")
    for name in names:
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            raise ValueError(f"Missing or empty π0.5 asset: {checkpoint / name}")
    json.loads((checkpoint / "config.json").read_text())
    with safe_open(checkpoint / "model.safetensors", framework="pt") as stream:
        if "action_in_proj.weight" not in stream.keys():
            raise ValueError("Expected an unsharded native π0.5 PyTorch checkpoint")
    sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
    manifest = dict(
        schema=1,
        repo=repo,
        revision=revision,
        verification="local-inventory; upstream origin supplied by the operator",
        files={
            name: dict(size=(checkpoint / name).stat().st_size, sha256=sha256(checkpoint / name))
            for name in names
        },
        tokenizer_sha256=sha256(tokenizer),
    )
    target = checkpoint / "RSCI_ASSETS.json"
    if target.exists():
        existing = json.loads(target.read_text())
        if any(existing.get(key) != manifest[key] for key in ("files", "tokenizer_sha256")):
            raise ValueError(
                "Existing asset inventory differs; use a separate checkpoint directory"
            )
        return existing
    atomic_json(target, manifest)
    return manifest


def transformers_overlay(install=False):
    if importlib.metadata.version("transformers") != "4.53.2":
        raise RuntimeError("π0.5 requires transformers==4.53.2 in its own environment")
    import transformers

    source = Path(__file__).resolve().parents[4] / "src/openpi/models_pytorch/transformers_replace"
    installed = Path(transformers.__file__).resolve().parent
    for path in sorted(source.rglob("*.py")):
        target = installed / path.relative_to(source)
        if install:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        if not target.is_file() or sha256(path) != sha256(target):
            raise RuntimeError(
                "π0.5 Transformers overlay is missing or differs; run "
                "python -m rsci.backbones.pi05.prepare_assets --install-transformers-patch "
                "in the dedicated environment"
            )
    return installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--repo", default="local")
    parser.add_argument("--revision", default="operator-supplied")
    parser.add_argument("--install-transformers-patch", action="store_true")
    parser.add_argument("--check-transformers-patch", action="store_true")
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.tokenizer):
        parser.error("--checkpoint and --tokenizer must be supplied together")
    if not (args.checkpoint or args.install_transformers_patch or args.check_transformers_patch):
        parser.error("Provide assets to register or select a Transformers patch operation")
    if args.install_transformers_patch or args.check_transformers_patch:
        path = transformers_overlay(install=args.install_transformers_patch)
        print(f"PI05_TRANSFORMERS_PATCH_OK {path}")
    if args.checkpoint:
        register_assets(args.checkpoint, args.tokenizer, args.repo, args.revision)
        print(f"PI05_ASSETS_OK {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
