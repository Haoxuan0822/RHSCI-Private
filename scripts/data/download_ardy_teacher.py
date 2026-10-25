#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsci.data.ardy_teacher_artifacts import (
    ARDY_G1_H52_ARTIFACTS,
    ARDY_MODEL_REPO,
    ARDY_MODEL_REVISION,
    verify_ardy_teacher_dir,
)

snapshot_download: Callable[..., str] | None = None


def _get_snapshot_download() -> Callable[..., str]:
    if snapshot_download is not None:
        return snapshot_download

    try:
        from huggingface_hub import snapshot_download as huggingface_snapshot_download
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "huggingface_hub is required to download the pinned ARDY teacher"
        ) from error
    return huggingface_snapshot_download


def download_ardy_teacher(
    output_dir: str | Path,
    *,
    expected_artifacts: Mapping[str, str] = ARDY_G1_H52_ARTIFACTS,
) -> dict[str, str]:
    """Download, verify, and atomically publish the pinned ARDY tokenizer artifacts."""
    destination = Path(output_dir)
    if destination.exists():
        if destination.is_dir():
            try:
                return verify_ardy_teacher_dir(destination, expected_artifacts=expected_artifacts)
            except ValueError as error:
                raise FileExistsError(
                    f"Refusing existing ARDY teacher destination {destination}: {error}"
                ) from error
        raise FileExistsError(
            f"Refusing existing ARDY teacher destination {destination}: not a directory"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    _get_snapshot_download()(
        repo_id=ARDY_MODEL_REPO,
        revision=ARDY_MODEL_REVISION,
        allow_patterns=list(ARDY_G1_H52_ARTIFACTS),
        local_dir=str(temporary_dir),
    )
    verified = verify_ardy_teacher_dir(temporary_dir, expected_artifacts=expected_artifacts)
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing ARDY teacher publication to existing destination {destination}"
        ) from error
    try:
        os.replace(temporary_dir, destination)
    except OSError as error:
        raise RuntimeError(
            "Unable to publish verified ARDY teacher; preserving temporary and "
            f"reserved directories for inspection: {temporary_dir}, {destination}"
        ) from error
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned ARDY G1 Horizon52 tokenizer artifacts."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verified = download_ardy_teacher(args.output_dir)
    print(f"Verified {len(verified)} ARDY teacher artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
