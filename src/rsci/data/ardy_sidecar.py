from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_VERSION = 1
CANDIDATE_TEACHER_STATUS = "candidate_unverified"
FORMAL_TEACHER_STATUS = "formal_verified"
TEACHER_VERIFICATION_STATUSES = {
    CANDIDATE_TEACHER_STATUS,
    FORMAL_TEACHER_STATUS,
}
SIDECAR_KEYS = {
    "ardy_body_latent",
    "teacher_valid",
    "visual_history_valid",
    "ardy_loss_mask",
    "raw_episode_index",
    "raw_frame_index",
}


def _normalized_joint_array(value: object, *, label: str) -> np.ndarray:
    """将关节序列规范为可精确比较的二维 43D 数值数组。"""
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 43:
        raise ValueError(f"{label} joints must be a two-dimensional 43D array, got {array.shape}")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{label} joints must have a real numeric dtype, got {array.dtype}")
    normalized = array.astype(np.float64, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError(f"{label} joints must contain only finite values")
    return normalized


def find_unique_episode_mapping(
    processed_joints: object,
    raw_candidates: Mapping[int, object],
) -> int:
    """以完整 43D 序列证明 processed[t] 唯一对应 raw[t+1]，返回 raw episode key。"""
    processed = _normalized_joint_array(processed_joints, label="processed")
    matches: list[int] = []
    candidate_frame_counts: dict[int, int] = {}
    for raw_episode_index, candidate_value in raw_candidates.items():
        raw = _normalized_joint_array(candidate_value, label=f"raw episode {raw_episode_index}")
        candidate_frame_counts[int(raw_episode_index)] = raw.shape[0]
        expected_raw_frames = processed.shape[0] + 1
        if raw.shape[0] != expected_raw_frames:
            continue
        if np.array_equal(processed, raw[1:]):
            matches.append(int(raw_episode_index))

    if not matches:
        raise ValueError(
            "Found no raw episode with exact processed[t] == raw[t+1] alignment; "
            f"expected raw frame count {processed.shape[0] + 1}, "
            f"candidate frame counts {candidate_frame_counts}"
        )
    if len(matches) != 1:
        raise ValueError(f"Found multiple raw episodes with exact alignment: {sorted(matches)}")
    return matches[0]


def window_indices(
    processed_frame: int,
) -> tuple[tuple[int, ...], tuple[int, ...], bool, bool]:
    """返回未裁剪的 teacher/视觉索引，并分别标记两个 episode 边界是否有效。"""
    if not isinstance(processed_frame, (int, np.integer)):
        raise TypeError(f"processed_frame must be an integer, got {type(processed_frame).__name__}")
    frame = int(processed_frame)
    raw_indices = (frame - 5, frame - 3, frame - 1, frame + 1)
    processed_indices = (frame - 6, frame - 4, frame - 2, frame)
    return raw_indices, processed_indices, raw_indices[0] >= 0, processed_indices[0] >= 0


def _canonical_context(context: object) -> tuple[dict[str, Any], bytes]:
    """复制 JSON context 并生成稳定字节；禁止 NaN 等非标准 JSON 值混入身份。"""
    if not isinstance(context, Mapping):
        raise TypeError("ARDY sidecar context must be a JSON mapping")
    try:
        encoded = json.dumps(
            context,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        copied = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"ARDY sidecar context must be JSON-serializable: {error}") from error
    if not isinstance(copied, dict):
        raise TypeError("ARDY sidecar context must serialize to a JSON object")
    return copied, encoded


def context_fingerprint(context: object) -> str:
    """返回 canonical JSON context 的 SHA256，供数据集和 checkpoint 精确绑定。"""
    _, encoded = _canonical_context(context)
    return hashlib.sha256(encoded).hexdigest()


def dataset_content_fingerprint(root: str | Path) -> str:
    """哈希会影响 ARDY 对齐的 metadata/parquet；允许数据目录跨机器搬迁。"""
    dataset_root = Path(root)
    files = sorted((dataset_root / "meta").glob("*")) + sorted(
        (dataset_root / "data").rglob("*.parquet")
    )
    files = [path for path in files if path.is_file()]
    if not files or not any(path.suffix == ".parquet" for path in files):
        raise ValueError(f"Dataset has no metadata/parquet files: {dataset_root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(dataset_root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_processed_dataset_context(
    processed_root: str | Path,
    context: Mapping[str, Any],
) -> None:
    """按任务名和内容哈希绑定本机数据，不要求 provenance 中的绝对路径相同。"""
    root = Path(processed_root).resolve()
    expected_dataset = context.get("dataset")
    if not isinstance(expected_dataset, str) or root.name != expected_dataset:
        raise ValueError(
            f"ARDY sidecar dataset {expected_dataset!r} does not match dataset directory {root.name!r}"
        )
    expected_fingerprint = context.get("processed_content_fingerprint")
    if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
        raise ValueError("ARDY sidecar context lacks a valid processed_content_fingerprint")
    actual_fingerprint = dataset_content_fingerprint(root)
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "ARDY sidecar processed dataset fingerprint mismatch: "
            f"expected {expected_fingerprint}, actual {actual_fingerprint}"
        )


def _validate_teacher_provenance_context(context: Mapping[str, Any]) -> None:
    """要求 teacher 身份可审计；formal 状态只能由外部一致性流程形成新 context。"""
    status = context.get("teacher_verification_status")
    if status not in TEACHER_VERIFICATION_STATUSES:
        raise ValueError(
            "ARDY sidecar context teacher_verification_status must be one of "
            f"{sorted(TEACHER_VERIFICATION_STATUSES)}, got {status!r}"
        )
    pipeline = context.get("teacher_pipeline")
    if not isinstance(pipeline, str) or not pipeline:
        raise ValueError("ARDY sidecar context teacher_pipeline must be a non-empty string")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_created_at(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("ARDY sidecar manifest created_at must be a UTC ISO8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("ARDY sidecar manifest created_at must be a UTC ISO8601 string") from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("ARDY sidecar manifest created_at must be a UTC ISO8601 string")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """经同目录临时文件发布 JSON；失败临时文件保留以诊断中断位置。"""
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(value, file, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _validate_sidecar_arrays(values: Mapping[str, object]) -> dict[str, np.ndarray]:
    """严格验收六个数组的 dtype/shape/mask 关系，返回不改变 dtype 的数组视图。"""
    keys = set(values)
    if keys != SIDECAR_KEYS:
        raise ValueError(
            f"ARDY sidecar arrays must be exactly {sorted(SIDECAR_KEYS)}, got {sorted(keys)}"
        )

    arrays = {name: np.asarray(value) for name, value in values.items()}
    latent = arrays["ardy_body_latent"]
    if latent.dtype != np.float32:
        raise ValueError(f"ardy_body_latent must have dtype float32, got {latent.dtype}")
    if latent.ndim != 2 or latent.shape[1] != 128:
        raise ValueError(f"ardy_body_latent must have shape [T, 128], got {latent.shape}")
    if latent.shape[0] <= 0:
        raise ValueError("ardy_body_latent frame count T must be positive")
    if not np.isfinite(latent).all():
        raise ValueError("ardy_body_latent must contain only finite values")
    frame_count = latent.shape[0]

    for name in ("teacher_valid", "visual_history_valid", "ardy_loss_mask"):
        array = arrays[name]
        if array.dtype != np.bool_:
            raise ValueError(f"{name} must have dtype bool, got {array.dtype}")
        if array.shape != (frame_count,):
            raise ValueError(f"{name} must have shape [{frame_count}], got {array.shape}")

    raw_episode_index = arrays["raw_episode_index"]
    if raw_episode_index.dtype != np.int64:
        raise ValueError(f"raw_episode_index must have dtype int64, got {raw_episode_index.dtype}")
    if raw_episode_index.shape != ():
        raise ValueError(
            f"raw_episode_index must be an int64 scalar, got {raw_episode_index.shape}"
        )

    raw_frame_index = arrays["raw_frame_index"]
    if raw_frame_index.dtype != np.int64:
        raise ValueError(f"raw_frame_index must have dtype int64, got {raw_frame_index.dtype}")
    if raw_frame_index.shape != (frame_count,):
        raise ValueError(
            f"raw_frame_index must have shape [{frame_count}], got {raw_frame_index.shape}"
        )

    expected_loss_mask = arrays["teacher_valid"] & arrays["visual_history_valid"]
    if not np.array_equal(arrays["ardy_loss_mask"], expected_loss_mask):
        raise ValueError("ardy_loss_mask must equal teacher_valid & visual_history_valid")
    return arrays


def _load_npz_arrays(path: Path, *, episode_index: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"ARDY sidecar episode {episode_index} artifact is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
    except Exception as error:
        raise ValueError(
            f"ARDY sidecar episode {episode_index} artifact is corrupt: {path}: {error}"
        ) from error
    try:
        return _validate_sidecar_arrays(arrays)
    except ValueError as error:
        raise ValueError(f"ARDY sidecar episode {episode_index} data invalid: {error}") from error


def _episode_artifact_path(root: Path, entry: Mapping[str, Any], episode_index: int) -> Path:
    relative = entry.get("path")
    expected = f"episode_{episode_index:06d}.npz"
    if relative != expected:
        raise ValueError(
            f"ARDY sidecar episode {episode_index} path must be {expected!r}, got {relative!r}"
        )
    return root / expected


def _validate_complete_artifact(
    root: Path, entry: Mapping[str, Any], episode_index: int
) -> dict[str, np.ndarray]:
    path = _episode_artifact_path(root, entry, episode_index)
    if not path.is_file():
        raise ValueError(f"ARDY sidecar episode {episode_index} artifact is missing: {path}")
    actual_hash = _sha256_file(path)
    if entry.get("sha256") != actual_hash:
        raise ValueError(
            f"ARDY sidecar episode {episode_index} hash mismatch: "
            f"expected {entry.get('sha256')}, actual {actual_hash}"
        )
    actual_size = path.stat().st_size
    if entry.get("size") != actual_size:
        raise ValueError(
            f"ARDY sidecar episode {episode_index} size mismatch: "
            f"expected {entry.get('size')}, actual {actual_size}"
        )
    arrays = _load_npz_arrays(path, episode_index=episode_index)
    if entry.get("frame_count") != arrays["ardy_body_latent"].shape[0]:
        raise ValueError(
            f"ARDY sidecar episode {episode_index} frame count mismatch: "
            f"expected {entry.get('frame_count')}, "
            f"actual {arrays['ardy_body_latent'].shape[0]}"
        )
    return arrays


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"ARDY sidecar manifest is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"ARDY sidecar manifest is corrupt: {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("ARDY sidecar manifest must be a JSON object")
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported ARDY sidecar manifest version: {manifest.get('version')!r}")
    _validate_created_at(manifest.get("created_at"))
    if not isinstance(manifest.get("context"), dict):
        raise ValueError("ARDY sidecar manifest context must be a JSON object")
    _validate_teacher_provenance_context(manifest["context"])
    if manifest.get("context_fingerprint") != context_fingerprint(manifest["context"]):
        raise ValueError("ARDY sidecar manifest context fingerprint does not match its context")
    if not isinstance(manifest.get("episodes"), dict):
        raise ValueError("ARDY sidecar manifest episodes must be a JSON object")
    for key, entry in manifest["episodes"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"ARDY sidecar episode {key} entry must be a JSON object")
        if entry.get("status") not in {"in_progress", "complete"}:
            raise ValueError(
                f"ARDY sidecar episode {key} has invalid status {entry.get('status')!r}"
            )
    return manifest


def _require_context(manifest: Mapping[str, Any], expected_context: object) -> dict[str, Any]:
    normalized, _ = _canonical_context(expected_context)
    if manifest["context"] != normalized:
        raise ValueError("ARDY sidecar manifest context does not match expected context")
    if manifest["context_fingerprint"] != context_fingerprint(normalized):
        raise ValueError(
            "ARDY sidecar manifest context fingerprint does not match expected context"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class ArdyTarget:
    """单个 processed frame 的只读 teacher 标签；不包含 raw pose 模型输入。"""

    ardy_body_latent: np.ndarray
    teacher_valid: bool
    visual_history_valid: bool
    ardy_loss_mask: bool
    raw_episode_index: int
    raw_frame_index: int


class ArdySidecarWriter:
    """按 episode 原子发布 NPZ，并以 manifest 状态机支持 fail-closed 续跑。"""

    def __init__(
        self,
        output_root: str | Path,
        context: Mapping[str, object],
        resume: bool = False,
    ) -> None:
        self.output_root = Path(output_root)
        self.manifest_path = self.output_root / "manifest.json"
        normalized_context, _ = _canonical_context(context)
        _validate_teacher_provenance_context(normalized_context)
        self.context = normalized_context
        self.context_fingerprint = context_fingerprint(normalized_context)

        if self.manifest_path.exists():
            manifest = _read_manifest(self.manifest_path)
            if not resume:
                raise ValueError(
                    f"ARDY sidecar manifest already exists at {self.manifest_path}; pass resume=True"
                )
            _require_context(manifest, normalized_context)
            self.manifest = manifest
            for key, entry in self.manifest["episodes"].items():
                if entry["status"] == "complete":
                    _validate_complete_artifact(self.output_root, entry, int(key))
            return

        if resume:
            raise ValueError(f"Cannot resume without ARDY sidecar manifest: {self.manifest_path}")
        if self.output_root.exists() and any(self.output_root.iterdir()):
            raise ValueError(
                f"Refusing nonempty ARDY sidecar output without manifest: {self.output_root}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest: dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "created_at": _utc_now(),
            "context": normalized_context,
            "context_fingerprint": self.context_fingerprint,
            "episodes": {},
        }
        _atomic_write_json(self.manifest_path, self.manifest)

    def is_episode_complete(self, episode_index: int) -> bool:
        """仅在 complete 项的 hash、size 和六个数组重新验收后返回真。"""
        episode = int(episode_index)
        entry = self.manifest["episodes"].get(str(episode))
        if entry is None or entry["status"] != "complete":
            return False
        _validate_complete_artifact(self.output_root, entry, episode)
        return True

    def write_episode(
        self,
        episode_index: int,
        *,
        ardy_body_latent: object,
        teacher_valid: object,
        visual_history_valid: object,
        ardy_loss_mask: object,
        raw_episode_index: object,
        raw_frame_index: object,
    ) -> Path:
        """先提交 in_progress，再验收并原子发布一个 episode 的六个数组。"""
        episode = int(episode_index)
        if episode < 0:
            raise ValueError(f"episode_index must be nonnegative, got {episode}")
        key = str(episode)
        path = self.output_root / f"episode_{episode:06d}.npz"
        supplied = {
            "ardy_body_latent": ardy_body_latent,
            "teacher_valid": teacher_valid,
            "visual_history_valid": visual_history_valid,
            "ardy_loss_mask": ardy_loss_mask,
            "raw_episode_index": raw_episode_index,
            "raw_frame_index": raw_frame_index,
        }
        latent_shape = np.asarray(ardy_body_latent).shape
        frame_count = int(latent_shape[0]) if latent_shape else 0
        existing = self.manifest["episodes"].get(key)
        if existing is not None and existing["status"] == "complete":
            raise ValueError(f"Refusing to overwrite complete ARDY sidecar episode {episode}")
        if existing is None:
            existing = {
                "status": "in_progress",
                "path": path.name,
                "frame_count": frame_count,
                "sha256": None,
                "size": None,
            }
            self.manifest["episodes"][key] = existing
            _atomic_write_json(self.manifest_path, self.manifest)
        elif existing.get("path") != path.name:
            raise ValueError(f"ARDY sidecar episode {episode} has incompatible in-progress path")

        arrays = _validate_sidecar_arrays(supplied)
        if path.exists():
            current = _load_npz_arrays(path, episode_index=episode)
            if any(not np.array_equal(current[name], arrays[name]) for name in SIDECAR_KEYS):
                raise ValueError(
                    f"Refusing to overwrite incompatible in-progress ARDY sidecar episode {episode}"
                )
        else:
            temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
            with temporary.open("xb") as file:
                np.savez(file, **arrays)
                file.flush()
                os.fsync(file.fileno())
            _load_npz_arrays(temporary, episode_index=episode)
            os.replace(temporary, path)

        existing.update(
            {
                "status": "complete",
                "frame_count": arrays["ardy_body_latent"].shape[0],
                "raw_episode_index": int(arrays["raw_episode_index"]),
                "teacher_valid_count": int(arrays["teacher_valid"].sum()),
                "visual_history_valid_count": int(arrays["visual_history_valid"].sum()),
                "ardy_loss_mask_count": int(arrays["ardy_loss_mask"].sum()),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
        _atomic_write_json(self.manifest_path, self.manifest)
        return path


class ArdySidecarReader:
    """启动时验收整个 sidecar 根，并按 processed episode/frame 返回 typed target。"""

    def __init__(
        self,
        root: str | Path,
        expected_context: Mapping[str, object],
        *,
        require_formal_teacher: bool = False,
    ) -> None:
        self.root = Path(root)
        self.manifest = _read_manifest(self.root / "manifest.json")
        _require_context(self.manifest, expected_context)
        self.teacher_verification_status = self.manifest["context"]["teacher_verification_status"]
        self.teacher_pipeline = self.manifest["context"]["teacher_pipeline"]
        if require_formal_teacher and self.teacher_verification_status != FORMAL_TEACHER_STATUS:
            raise ValueError(
                f"ARDY sidecar requires {FORMAL_TEACHER_STATUS} teacher, "
                f"got {self.teacher_verification_status}"
            )
        self._episodes: dict[int, dict[str, np.ndarray]] = {}
        for key, entry in self.manifest["episodes"].items():
            episode = int(key)
            if entry["status"] != "complete":
                raise ValueError(f"ARDY sidecar has in_progress episode {episode}")
            self._episodes[episode] = _validate_complete_artifact(self.root, entry, episode)

    def get(self, episode_index: int, frame_index: int) -> ArdyTarget:
        episode = int(episode_index)
        frame = int(frame_index)
        arrays = self._episodes.get(episode)
        if arrays is None:
            raise IndexError(f"ARDY sidecar episode {episode} is unavailable")
        frame_count = arrays["ardy_body_latent"].shape[0]
        if frame < 0 or frame >= frame_count:
            raise IndexError(
                f"ARDY sidecar episode {episode} frame {frame} is outside [0, {frame_count})"
            )
        return ArdyTarget(
            ardy_body_latent=arrays["ardy_body_latent"][frame].copy(),
            teacher_valid=bool(arrays["teacher_valid"][frame]),
            visual_history_valid=bool(arrays["visual_history_valid"][frame]),
            ardy_loss_mask=bool(arrays["ardy_loss_mask"][frame]),
            raw_episode_index=int(arrays["raw_episode_index"]),
            raw_frame_index=int(arrays["raw_frame_index"][frame]),
        )
