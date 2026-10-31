from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

ARDY_SOURCE_COMMIT = "693f74d13b3d04a0a22ce127ee79c929dd89756b"
ARDY_MODEL_REPO = "nvidia/ARDY-G1-RP-25FPS-Horizon52"
ARDY_MODEL_REVISION = "059b8007df0ba194a006a877b59a563955ac7b70"

ARDY_G1_H52_ARTIFACTS: Mapping[str, str] = {
    "config.yaml": "d34bcd2a43e5bade030e2779bd11e0c3d8ce7eeea6b636e7504efe9a404ec757",
    "tokenizer.safetensors": "6f2f75f738b8d6fb1b196d3253c7accc90494b9c12669b35968425fef54c8896",
    "stats/motion/mean.npy": "b6d2c72548475e9c82a80899429c955840796a06a82c88e646e7a0125f997407",
    "stats/motion/std.npy": "ee6069988035de4339b44fd229f169456d4bd86671fc4dbcfd293c4d41caa1a8",
    "stats/post_quantization/mean.npy": "78d5bbd3d22ca1866770a370f0fb9da15663cbbc63a0b8851f124dae78a2f8c0",
    "stats/post_quantization/std.npy": "289d21ca46aa8300cd213993a64efbd03d35440178de19b38cc8ee76d69499a1",
    "stats/pre_quantization/mean.npy": "14bf165f6cf79276bc45ea24d6101f320b9007854b2ddb86babdd057af287a8c",
    "stats/pre_quantization/std.npy": "413dbb9b736a41d033a5c0dbd0a632970b0e23737f39dd8eb0e859fe7a7e9556",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        while chunk := artifact_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_ardy_teacher_dir(
    root: str | Path,
    *,
    expected_artifacts: Mapping[str, str] = ARDY_G1_H52_ARTIFACTS,
) -> dict[str, str]:
    """Verify every approved ARDY artifact below ``root`` without network access."""
    root_path = Path(root)
    verified: dict[str, str] = {}
    for relative_path, expected_digest in sorted(expected_artifacts.items()):
        artifact_path = root_path / relative_path
        if not artifact_path.exists():
            raise ValueError(
                f"Missing approved artifact {relative_path}: "
                f"expected {expected_digest}, actual missing"
            )
        if not artifact_path.is_file():
            raise ValueError(
                f"Invalid approved artifact {relative_path}: "
                f"expected {expected_digest}, actual not a file"
            )

        actual_digest = _sha256_file(artifact_path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Hash mismatch for {relative_path}: "
                f"expected {expected_digest}, actual {actual_digest}"
            )
        verified[relative_path] = actual_digest
    return verified
