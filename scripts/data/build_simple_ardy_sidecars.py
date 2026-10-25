#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsci.data.ardy_sidecar import (
    CANDIDATE_TEACHER_STATUS,
    FORMAL_TEACHER_STATUS,
    ArdySidecarWriter,
    dataset_content_fingerprint,
    find_unique_episode_mapping,
    window_indices,
)
from rsci.data.ardy_teacher_artifacts import (
    ARDY_MODEL_REVISION,
    ARDY_SOURCE_COMMIT,
    verify_ardy_teacher_dir,
)

RAW_REVISION = "79baf58cf2147a4b975eab17cec761707b52df04"
RAW_ARCHIVE_SHA256 = "6bdd8113209b677ae9345c6e0a4016c537e4a6399375302192b4ed6368e02cb5"
PROCESSED_ARCHIVE_SHA256 = "4f91e23c34efa9b2743d2d554d3c4ca00db840e84c7f37f592f74eda92a0e85f"
DATASET_NAME = "G1WholebodyXMovePickTeleop-v0"
SOURCE_FPS = 50
TARGET_FPS = 25
DEFAULT_BATCH_SIZE = 64
TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS = 1e-6
TEACHER_PIPELINE = "tokenizer_only_single_patch"
FORMAL_REFERENCE_PROCESSED_EPISODE = 0
FORMAL_REFERENCE_RAW_EPISODE = 1
FORMAL_REFERENCE_LATENT_SHA256 = "f01b5207f4b9fe7cda25f577530e23287c01486957dfabcbaee88c13dd80cbd6"
G1_SKELETON_IDENTITY = "g1skel34"

G1_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_HAND_JOINT_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
G1_ARM_JOINT_NAMES = G1_BODY_JOINT_NAMES[15:]
PROCESSED_ACTION_MATCH_NAMES = G1_HAND_JOINT_NAMES + G1_ARM_JOINT_NAMES
ACTION_MATCH_ABSOLUTE_TOLERANCE = 1e-6
MIN_ACTION_MATCH_FRAMES = 8
G1_SKELETON_JOINT_NAMES = (
    "pelvis_skel",
    "left_hip_pitch_skel",
    "left_hip_roll_skel",
    "left_hip_yaw_skel",
    "left_knee_skel",
    "left_ankle_pitch_skel",
    "left_ankle_roll_skel",
    "left_toe_base",
    "right_hip_pitch_skel",
    "right_hip_roll_skel",
    "right_hip_yaw_skel",
    "right_knee_skel",
    "right_ankle_pitch_skel",
    "right_ankle_roll_skel",
    "right_toe_base",
    "waist_yaw_skel",
    "waist_roll_skel",
    "waist_pitch_skel",
    "left_shoulder_pitch_skel",
    "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel",
    "left_elbow_skel",
    "left_wrist_roll_skel",
    "left_wrist_pitch_skel",
    "left_wrist_yaw_skel",
    "left_hand_roll_skel",
    "right_shoulder_pitch_skel",
    "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel",
    "right_elbow_skel",
    "right_wrist_roll_skel",
    "right_wrist_pitch_skel",
    "right_wrist_yaw_skel",
    "right_hand_roll_skel",
)


class ArdyWindowAdapter:
    """把轻量 qpos 组装与重型官方 converter/tokenizer 隔离，便于纯测试注入。"""

    def __init__(self, converter: object, tokenizer: object) -> None:
        self.converter = converter
        self.tokenizer = tokenizer

    def tokenize_qpos_windows(self, qpos_windows: np.ndarray) -> np.ndarray:
        if (
            qpos_windows.dtype != np.float32
            or qpos_windows.ndim != 3
            or qpos_windows.shape[1:] != (4, 36)
        ):
            raise ValueError(
                f"qpos windows must be float32 [B, 4, 36], got {qpos_windows.dtype} {qpos_windows.shape}"
            )
        if not np.isfinite(qpos_windows).all():
            raise ValueError("qpos windows must contain only finite values")
        local_joint_rots, root_positions = self.converter.convert(qpos_windows)
        latent = np.asarray(self.tokenizer.tokenize(local_joint_rots, root_positions))
        expected = (qpos_windows.shape[0], 128)
        if latent.dtype != np.float32 or latent.shape != expected or not np.isfinite(latent).all():
            raise ValueError(
                f"official tokenizer output must be finite float32 {expected}, got {latent.dtype} {latent.shape}"
            )
        return latent


def validate_episode_timestamps(timestamps: object, *, label: str) -> np.ndarray:
    """要求 episode timestamp 为零基、逐帧严格对应 frame_index/50Hz 的浮点序列。"""
    values = np.asarray(timestamps)
    if values.ndim != 1:
        raise ValueError(f"{label} timestamps must be one-dimensional, got {values.shape}")
    if values.dtype.kind != "f":
        raise ValueError(f"{label} timestamps must have a floating-point dtype, got {values.dtype}")
    if values.size == 0:
        raise ValueError(f"{label} timestamps must not be empty")
    normalized = values.astype(np.float64, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError(f"{label} timestamps must contain only finite values")
    expected = np.arange(normalized.size, dtype=np.float64) / SOURCE_FPS
    if not np.allclose(
        normalized,
        expected,
        atol=TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS,
        rtol=0.0,
    ):
        max_error = float(np.max(np.abs(normalized - expected)))
        raise ValueError(
            f"{label} timestamps do not follow zero-based 50 Hz cadence; maximum error {max_error}"
        )
    return normalized


def validate_teacher_window_timestamps(
    timestamps: object,
    raw_indices: tuple[int, ...],
    *,
    label: str,
) -> None:
    """要求四个 teacher 样本按 raw 两帧步长相邻，即每步严格为 0.04 秒。"""
    values = np.asarray(timestamps)
    if len(raw_indices) != 4 or min(raw_indices) < 0 or max(raw_indices) >= values.shape[0]:
        raise ValueError(f"{label} teacher timestamp indices are outside the episode")
    selected = values[np.asarray(raw_indices, dtype=np.int64)].astype(np.float64, copy=False)
    expected_step = 2.0 / SOURCE_FPS
    if not np.allclose(
        np.diff(selected),
        expected_step,
        atol=TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS,
        rtol=0.0,
    ):
        raise ValueError(f"{label} teacher timestamps must be adjacent 0.04 second samples")


def build_episode_sidecar_arrays(
    *,
    processed_joints: object,
    raw_joint_candidates: dict[int, object],
    raw_base_candidates: dict[int, object],
    raw_timestamp_candidates: dict[int, object],
    processed_timestamps: object,
    adapter: ArdyWindowAdapter,
    batch_size: int = DEFAULT_BATCH_SIZE,
    raw_episode_index: int | None = None,
) -> tuple[int, dict[str, np.ndarray]]:
    """证明 episode 对齐后，以 raw base+wxyz+29 body joints 生成逐帧 ARDY 标签。"""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    processed = np.asarray(processed_joints)
    processed_time = validate_episode_timestamps(processed_timestamps, label="processed episode")
    if processed_time.shape != (processed.shape[0],):
        raise ValueError(
            "processed timestamps frame count does not match processed joint frame count"
        )
    if raw_episode_index is None:
        raw_episode_index = find_unique_episode_mapping(processed, raw_joint_candidates)
    elif raw_episode_index not in raw_joint_candidates:
        raise ValueError(f"Mapped raw episode {raw_episode_index} has no observation.state")
    raw_joints = np.asarray(raw_joint_candidates[raw_episode_index])
    if raw_episode_index not in raw_base_candidates:
        raise ValueError(f"raw episode {raw_episode_index} has no observation.base_pose")
    if raw_episode_index not in raw_timestamp_candidates:
        raise ValueError(f"raw episode {raw_episode_index} has no timestamp")
    raw_timestamps = validate_episode_timestamps(
        raw_timestamp_candidates[raw_episode_index],
        label=f"raw episode {raw_episode_index}",
    )
    if raw_timestamps.shape != (raw_joints.shape[0],):
        raise ValueError(
            f"raw episode {raw_episode_index} timestamp frame count does not match joints"
        )
    raw_base = np.asarray(raw_base_candidates[raw_episode_index])
    if raw_base.ndim != 2 or raw_base.shape != (raw_joints.shape[0], 7):
        raise ValueError(
            f"raw episode {raw_episode_index} base pose must be [{raw_joints.shape[0]}, 7], got {raw_base.shape}"
        )
    if raw_base.dtype.kind not in "iuf" or not np.isfinite(raw_base).all():
        raise ValueError(
            f"raw episode {raw_episode_index} base pose must be finite numeric xyz+wxyz"
        )
    quaternion_norms = np.linalg.norm(raw_base[:, 3:7].astype(np.float64), axis=1)
    if not np.allclose(quaternion_norms, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError(f"raw episode {raw_episode_index} base quaternion must be unit wxyz")

    frame_count = processed.shape[0]
    latent = np.zeros((frame_count, 128), dtype=np.float32)
    teacher_valid = np.zeros(frame_count, dtype=np.bool_)
    visual_history_valid = np.zeros(frame_count, dtype=np.bool_)
    windows: list[np.ndarray] = []
    target_frames: list[int] = []
    for frame in range(frame_count):
        raw_indices, _, teacher_ok, visual_ok = window_indices(frame)
        teacher_ok = teacher_ok and raw_indices[-1] < raw_joints.shape[0]
        teacher_valid[frame] = teacher_ok
        visual_history_valid[frame] = visual_ok
        if not teacher_ok:
            continue
        validate_teacher_window_timestamps(
            raw_timestamps,
            raw_indices,
            label=f"raw episode {raw_episode_index} processed frame {frame}",
        )
        indices = np.asarray(raw_indices, dtype=np.int64)
        qpos = np.concatenate([raw_base[indices], raw_joints[indices, :29]], axis=1)
        windows.append(qpos.astype(np.float32, copy=False))
        target_frames.append(frame)

    for start in range(0, len(windows), batch_size):
        batch = np.stack(windows[start : start + batch_size], axis=0)
        batch_latent = adapter.tokenize_qpos_windows(batch)
        for offset, value in enumerate(batch_latent):
            latent[target_frames[start + offset]] = value

    return raw_episode_index, {
        "ardy_body_latent": latent,
        "teacher_valid": teacher_valid,
        "visual_history_valid": visual_history_valid,
        "ardy_loss_mask": teacher_valid & visual_history_valid,
        "raw_episode_index": np.asarray(raw_episode_index, dtype=np.int64),
        "raw_frame_index": np.where(
            np.arange(1, frame_count + 1) < raw_joints.shape[0],
            np.arange(1, frame_count + 1),
            -1,
        ).astype(np.int64),
    }


def validate_formal_reference(
    processed_episode: int,
    raw_episode: int,
    arrays: dict[str, np.ndarray],
) -> None:
    """用固定真实样本摘要门禁 formal teacher，禁止只改 manifest 状态。"""
    if processed_episode != FORMAL_REFERENCE_PROCESSED_EPISODE:
        return
    if raw_episode != FORMAL_REFERENCE_RAW_EPISODE:
        raise ValueError(
            f"Formal reference processed episode 0 must map to raw episode 1, got {raw_episode}"
        )
    actual = hashlib.sha256(np.ascontiguousarray(arrays["ardy_body_latent"]).tobytes()).hexdigest()
    if actual != FORMAL_REFERENCE_LATENT_SHA256:
        raise ValueError(
            "Formal ARDY reference latent mismatch: "
            f"expected {FORMAL_REFERENCE_LATENT_SHA256}, actual {actual}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_root(value: str | Path, label: str) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{label} must be an existing directory: {root}")
    return root


def _read_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read dataset metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Dataset metadata must be an object: {path}")
    return value


def _feature(info: dict[str, Any], name: str, shape: list[int]) -> dict[str, Any]:
    feature = info.get("features", {}).get(name)
    if not isinstance(feature, dict) or feature.get("shape") != shape:
        raise ValueError(f"Dataset feature {name} must declare shape {shape}")
    return feature


def _validate_dataset_metadata(
    raw_root: Path, processed_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if raw_root.name != DATASET_NAME or processed_root.name != DATASET_NAME:
        raise ValueError(f"Task 2 is pinned to dataset {DATASET_NAME}")
    raw_info = _read_info(raw_root)
    processed_info = _read_info(processed_root)
    if raw_info.get("fps") != SOURCE_FPS or processed_info.get("fps") != SOURCE_FPS:
        raise ValueError("Raw and processed datasets must both declare 50 FPS")
    raw_state = _feature(raw_info, "observation.state", [43])
    raw_names = tuple(raw_state.get("names", ()))
    if len(raw_names) != 43 or raw_names[:29] != G1_BODY_JOINT_NAMES:
        raise ValueError("Raw observation.state must begin with the pinned 29 ARDY G1 body joints")
    raw_base = _feature(raw_info, "observation.base_pose", [7])
    if raw_base.get("names") != ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]:
        raise ValueError("Raw observation.base_pose must be xyz + quaternion wxyz")
    _feature(processed_info, "observation.leg_joints", [15])
    _feature(processed_info, "observation.arm_joints", [14])
    _feature(processed_info, "observation.hand_joints", [14])
    return raw_info, processed_info


def _source_commit(source_root: Path) -> str:
    if not (source_root / "ardy" / "__init__.py").is_file():
        raise ValueError(f"ARDY source root lacks ardy/__init__.py: {source_root}")
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Cannot verify ARDY source git commit at {source_root}: {result.stderr.strip()}"
        )
    commit = result.stdout.strip()
    if commit != ARDY_SOURCE_COMMIT:
        raise ValueError(
            f"ARDY source commit mismatch: expected {ARDY_SOURCE_COMMIT}, actual {commit}"
        )
    return commit


def _dataset_fingerprint(root: Path) -> str:
    return dataset_content_fingerprint(root)


def _load_episode_columns(root: Path, columns: tuple[str, ...]) -> dict[int, dict[str, np.ndarray]]:
    import pyarrow.parquet as pq

    rows: dict[int, dict[int, dict[str, Any]]] = {}
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet episodes found below {root / 'data'}")
    required = ("episode_index", "frame_index", *columns)
    for path in parquet_files:
        table = pq.ParquetFile(path).read(columns=list(required)).to_pylist()
        for row in table:
            episode = int(row["episode_index"])
            frame = int(row["frame_index"])
            episode_rows = rows.setdefault(episode, {})
            if frame in episode_rows:
                raise ValueError(f"Duplicate episode/frame in parquet: {episode}/{frame}")
            episode_rows[frame] = row

    episodes: dict[int, dict[str, np.ndarray]] = {}
    for episode, frame_rows in sorted(rows.items()):
        frame_indices = sorted(frame_rows)
        if frame_indices != list(range(len(frame_indices))):
            raise ValueError(f"Episode {episode} frame_index is not contiguous from zero")
        episode_values: dict[str, np.ndarray] = {}
        for column in columns:
            array = np.asarray([frame_rows[frame][column] for frame in frame_indices])
            if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
                raise ValueError(f"Episode {episode} column {column} must be finite numeric")
            episode_values[column] = array
        episodes[episode] = episode_values
    return episodes


def load_replay_status(replay_path: Path) -> dict[int, bool]:
    """严格读取官方 replay.txt；映射仍由不受物理漂移影响的 action 内容决定。"""
    try:
        lines = replay_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Cannot read official replay manifest {replay_path}: {error}") from error
    raw_status: dict[int, bool] = {}
    for line_number, line in enumerate(lines, start=1):
        parts = line.split(": ")
        if len(parts) != 2 or parts[1] not in {"True", "False"}:
            raise ValueError(f"Invalid replay manifest line {line_number}: {line!r}")
        try:
            raw_episode = int(parts[0])
        except ValueError as error:
            raise ValueError(f"Invalid replay episode at line {line_number}: {line!r}") from error
        if raw_episode in raw_status:
            raise ValueError(f"Duplicate raw episode {raw_episode} in replay manifest")
        raw_status[raw_episode] = parts[1] == "True"
    if sorted(raw_status) != list(range(len(raw_status))):
        raise ValueError("Replay manifest raw episode indices must be contiguous from zero")

    return raw_status


def find_action_episode_mapping(
    processed_actions: dict[int, np.ndarray],
    raw_actions: dict[int, np.ndarray],
    raw_action_names: tuple[str, ...],
    replay_status: dict[int, bool],
) -> dict[int, int]:
    """用 processed 前 28D 命令与 raw[t+1] 精确内容匹配；允许 processed 重复轨迹。"""
    if len(raw_action_names) != len(set(raw_action_names)):
        raise ValueError("Raw action joint names must be unique")
    try:
        raw_indices = [raw_action_names.index(name) for name in PROCESSED_ACTION_MATCH_NAMES]
    except ValueError as error:
        raise ValueError(f"Raw action schema lacks a required processed joint: {error}") from error

    mapping: dict[int, int] = {}
    for processed_episode, processed in sorted(processed_actions.items()):
        if processed.ndim != 2 or processed.shape[1] < len(PROCESSED_ACTION_MATCH_NAMES):
            raise ValueError(
                f"Processed episode {processed_episode} action must be [T, >=28], got {processed.shape}"
            )
        candidates: list[int] = []
        for raw_episode, raw in sorted(raw_actions.items()):
            if raw.ndim != 2 or raw.shape[1] != len(raw_action_names):
                raise ValueError(
                    f"Raw episode {raw_episode} action must be [T, {len(raw_action_names)}], got {raw.shape}"
                )
            overlap = min(processed.shape[0], raw.shape[0] - 1)
            if overlap < MIN_ACTION_MATCH_FRAMES:
                continue
            processed_commands = processed[:overlap, : len(PROCESSED_ACTION_MATCH_NAMES)]
            raw_commands = raw[1 : overlap + 1, raw_indices]
            if np.allclose(
                processed_commands,
                raw_commands,
                atol=ACTION_MATCH_ABSOLUTE_TOLERANCE,
                rtol=0.0,
            ):
                candidates.append(raw_episode)
        if len(candidates) != 1:
            raise ValueError(
                f"Processed episode {processed_episode} must match exactly one raw action sequence, "
                f"got {candidates}"
            )
        raw_episode = candidates[0]
        if replay_status.get(raw_episode) is not True:
            raise ValueError(
                f"Processed episode {processed_episode} maps to raw episode {raw_episode}, "
                "which replay.txt does not mark successful"
            )
        mapping[processed_episode] = raw_episode
    return mapping


class _OfficialMujocoConverter:
    """调用固定 ARDY 的官方 qpos loader，并用 MujocoQposConverter 做反向一致性门禁。"""

    def __init__(
        self,
        converter: object,
        legacy_skeleton: object,
        tokenizer_skeleton: object,
        load_csv,
    ) -> None:
        self.converter = converter
        self.legacy_skeleton = legacy_skeleton
        self.load_csv = load_csv
        self.legacy_joint_names = tuple(legacy_skeleton.get_bones_names())
        self.tokenizer_joint_names = tuple(tokenizer_skeleton.bone_order_names)
        if len(self.legacy_joint_names) != 30 or len(set(self.legacy_joint_names)) != 30:
            raise ValueError("Official legacy G1 skeleton must contain 30 unique root/body joints")
        if len(self.tokenizer_joint_names) != 34 or len(set(self.tokenizer_joint_names)) != 34:
            raise ValueError("Official tokenizer G1 skeleton must contain 34 unique joints")
        legacy_to_tokenizer = []
        for name in self.legacy_joint_names:
            matches = [
                index
                for index, candidate in enumerate(self.tokenizer_joint_names)
                if candidate == name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Legacy G1 joint {name!r} must map exactly once into tokenizer skeleton, "
                    f"got {len(matches)}"
                )
            legacy_to_tokenizer.append(matches[0])
        self.legacy_to_tokenizer = tuple(legacy_to_tokenizer)

    def convert(self, qpos_windows: np.ndarray):
        import torch

        batch_size = qpos_windows.shape[0]
        unitree_qpos = qpos_windows.reshape(-1, 36).copy()
        unitree_qpos[:, 3:7] = unitree_qpos[:, [4, 5, 6, 3]]  # 官方 loader 输入 xyzw。
        with tempfile.TemporaryDirectory(prefix="psi0-ardy-qpos-") as directory:
            csv_path = Path(directory) / "qpos.csv"
            np.savetxt(csv_path, unitree_qpos, delimiter=",")
            root_positions, local_joint_rots = self.load_csv(
                str(csv_path), self.legacy_skeleton, csv_format="retargeted_from_unitree"
            )
        root_positions = torch.as_tensor(root_positions, dtype=torch.float32)
        if root_positions.shape != (batch_size * 4, 3):
            raise ValueError(
                f"Official qpos loader root positions must be [{batch_size * 4}, 3], "
                f"got {tuple(root_positions.shape)}"
            )
        root_positions = (root_positions / 100.0).reshape(batch_size, 4, 3)
        legacy_joint_rots = torch.as_tensor(local_joint_rots, dtype=torch.float32)
        expected_legacy_shape = (batch_size * 4, len(self.legacy_joint_names), 3, 3)
        if tuple(legacy_joint_rots.shape) != expected_legacy_shape:
            raise ValueError(
                f"Official qpos loader rotations must be {expected_legacy_shape}, "
                f"got {tuple(legacy_joint_rots.shape)}"
            )
        legacy_joint_rots = legacy_joint_rots.reshape(batch_size, 4, 30, 3, 3)
        identity = torch.eye(3, dtype=legacy_joint_rots.dtype, device=legacy_joint_rots.device)
        tokenizer_joint_rots = identity.expand(batch_size, 4, 34, 3, 3).clone()
        for legacy_index, tokenizer_index in enumerate(self.legacy_to_tokenizer):
            tokenizer_joint_rots[:, :, tokenizer_index] = legacy_joint_rots[:, :, legacy_index]

        roundtrip = (
            self.converter.to_qpos(tokenizer_joint_rots, root_positions).detach().cpu().numpy()
        )
        if not np.allclose(roundtrip, qpos_windows, atol=1e-4, rtol=0.0):
            raise ValueError("Pinned MujocoQposConverter round-trip does not reproduce source qpos")
        return tokenizer_joint_rots, root_positions


class _OfficialTokenizer:
    def __init__(self, tokenizer: object, device: str) -> None:
        self.tokenizer = tokenizer
        self.device = device

    def tokenize(self, local_joint_rots, root_positions) -> np.ndarray:
        import torch

        local_joint_rots = local_joint_rots.to(device=self.device, dtype=torch.float32)
        root_positions = root_positions.to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            frame_mask = torch.ones(
                local_joint_rots.shape[:2], dtype=torch.bool, device=self.device
            )
            # 官方 canonicalize 的位置旋转只支持单窗口广播。逐窗口保持官方
            # 数学路径，再合批执行 tokenizer，避免修改或复刻官方实现。
            explicit = torch.cat(
                [
                    self.tokenizer.motion_rep(
                        local_joint_rots[index : index + 1],
                        root_positions[index : index + 1],
                        lengths=frame_mask[index : index + 1].sum(1),
                        to_normalize=True,
                        to_canonicalize=True,
                    )
                    for index in range(local_joint_rots.shape[0])
                ],
                dim=0,
            )
            latent = self.tokenizer.tokenize(explicit, motion_pad_mask=frame_mask)[:, 0]
        return latent.detach().cpu().to(torch.float32).numpy()


def _build_official_adapter_from_components(
    *,
    tokenizer: object,
    converter_type,
    legacy_skeleton_type,
    load_csv,
    device: str,
) -> ArdyWindowAdapter:
    """组装已导入的官方组件，并确保 tokenizer 与输入位于同一设备。"""
    tokenizer = tokenizer.to(device).eval()
    tokenizer_skeleton = tokenizer.motion_rep.skeleton
    actual_identity = getattr(tokenizer_skeleton, "name", None)
    actual_joint_order = tuple(getattr(tokenizer_skeleton, "bone_order_names", ()))
    if actual_identity != G1_SKELETON_IDENTITY or actual_joint_order != G1_SKELETON_JOINT_NAMES:
        raise ValueError(
            "Official tokenizer skeleton must match pinned g1skel34 identity and joint order"
        )
    converter = converter_type(tokenizer_skeleton)
    legacy_skeleton = legacy_skeleton_type()
    legacy_skeleton.load_from_xml(converter.xml_path)
    return ArdyWindowAdapter(
        _OfficialMujocoConverter(
            converter,
            legacy_skeleton,
            tokenizer_skeleton,
            load_csv,
        ),
        _OfficialTokenizer(tokenizer, device),
    )


def _load_official_adapter(source_root: Path, teacher_dir: Path) -> ArdyWindowAdapter:
    """所有路径与哈希门禁通过后，才导入 CUDA-heavy ARDY 模块。"""
    sys.path.insert(0, str(source_root))
    import torch
    from ardy.data_processing.skeleton import SkeletonG1, load_csv_g1_animation
    from ardy.exports.mujoco import MujocoQposConverter
    from ardy.model.loading import instantiate_from_dict
    from omegaconf import OmegaConf

    config = OmegaConf.load(teacher_dir / "config.yaml")
    config.checkpoint_dir = str(teacher_dir)
    resolved = OmegaConf.to_container(config, resolve=True)
    tokenizer = instantiate_from_dict(resolved["autoencoder"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _build_official_adapter_from_components(
        tokenizer=tokenizer,
        converter_type=MujocoQposConverter,
        legacy_skeleton_type=SkeletonG1,
        load_csv=load_csv_g1_animation,
        device=device,
    )


def _build_context(
    raw_root: Path,
    processed_root: Path,
    teacher_hashes: dict[str, str],
    source_commit: str,
    *,
    formal_verified: bool = False,
) -> dict[str, Any]:
    return {
        "dataset": DATASET_NAME,
        "raw_root": str(raw_root),
        "raw_revision": RAW_REVISION,
        "raw_archive_sha256": RAW_ARCHIVE_SHA256,
        "raw_content_fingerprint": _dataset_fingerprint(raw_root),
        "processed_root": str(processed_root),
        "processed_archive_sha256": PROCESSED_ARCHIVE_SHA256,
        "processed_content_fingerprint": _dataset_fingerprint(processed_root),
        "ardy_source_commit": source_commit,
        "ardy_model_revision": ARDY_MODEL_REVISION,
        "teacher_hashes": teacher_hashes,
        "teacher_verification_status": (
            FORMAL_TEACHER_STATUS if formal_verified else CANDIDATE_TEACHER_STATUS
        ),
        "teacher_pipeline": TEACHER_PIPELINE,
        "formal_reference": {
            "processed_episode": FORMAL_REFERENCE_PROCESSED_EPISODE,
            "raw_episode": FORMAL_REFERENCE_RAW_EPISODE,
            "ardy_body_latent_sha256": FORMAL_REFERENCE_LATENT_SHA256,
        },
        "body_joint_order": list(G1_BODY_JOINT_NAMES),
        "skeleton_identity": G1_SKELETON_IDENTITY,
        "skeleton_joint_order": list(G1_SKELETON_JOINT_NAMES),
        "quaternion_convention": "wxyz",
        "to_canonicalize": True,
        "qpos_conversion": "official_mujoco_qpos_converter_roundtrip",
        "tokenizer_context_policy": "single_patch_reset",
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "timestamp_policy": {
            "cadence": "zero_based_frame_index_div_fps",
            "source_fps": SOURCE_FPS,
            "teacher_stride_frames": 2,
            "teacher_step_seconds": 2.0 / SOURCE_FPS,
            "absolute_tolerance_seconds": TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS,
        },
        "raw_window": ["t-5", "t-3", "t-1", "t+1"],
        "processed_window": ["t-6", "t-4", "t-2", "t"],
        "episode_mapping_policy": "exact_hand_arm_action_t_plus_one",
        "episode_mapping_absolute_tolerance": ACTION_MATCH_ABSOLUTE_TOLERANCE,
        "episode_mapping_min_overlap_frames": MIN_ACTION_MATCH_FRAMES,
        "episode_mapping_allows_duplicate_raw": True,
        "replay_manifest_sha256": _sha256_file(raw_root / "meta" / "replay.txt"),
        "generator_sha256": {
            "ardy_sidecar.py": _sha256_file(
                PROJECT_ROOT / "src" / "rsci" / "data" / "ardy_sidecar.py"
            ),
            "build_simple_ardy_sidecars.py": _sha256_file(Path(__file__)),
        },
    }


def build_sidecars(args: argparse.Namespace) -> None:
    raw_root = _require_root(args.raw_root, "--raw-root")
    processed_root = _require_root(args.processed_root, "--processed-root")
    teacher_dir = _require_root(args.teacher_dir, "--teacher-dir")
    source_root = _require_root(args.ardy_source_root, "--ardy-source-root")
    raw_info, _ = _validate_dataset_metadata(raw_root, processed_root)
    teacher_hashes = verify_ardy_teacher_dir(teacher_dir)
    source_commit = _source_commit(source_root)
    context = _build_context(
        raw_root,
        processed_root,
        teacher_hashes,
        source_commit,
        formal_verified=args.formal_verified,
    )

    raw = _load_episode_columns(
        raw_root, ("observation.state", "observation.base_pose", "timestamp", "action")
    )
    processed = _load_episode_columns(
        processed_root,
        (
            "observation.leg_joints",
            "observation.arm_joints",
            "observation.hand_joints",
            "timestamp",
            "action",
        ),
    )
    replay_mapping = find_action_episode_mapping(
        {episode: values["action"] for episode, values in processed.items()},
        {episode: values["action"] for episode, values in raw.items()},
        tuple(raw_info["features"]["action"]["names"]),
        load_replay_status(raw_root / "meta" / "replay.txt"),
    )
    for episode, values in raw.items():
        values["timestamp"] = validate_episode_timestamps(
            values["timestamp"], label=f"raw episode {episode}"
        )
    for episode, values in processed.items():
        values["timestamp"] = validate_episode_timestamps(
            values["timestamp"], label=f"processed episode {episode}"
        )
    raw_joint_candidates = {episode: values["observation.state"] for episode, values in raw.items()}
    raw_base_candidates = {
        episode: values["observation.base_pose"] for episode, values in raw.items()
    }
    raw_timestamp_candidates = {episode: values["timestamp"] for episode, values in raw.items()}
    writer = ArdySidecarWriter(args.output_root, context, resume=args.resume)
    selected = sorted(processed)
    if args.episode_limit is not None:
        if args.episode_limit <= 0:
            raise ValueError("--episode-limit must be positive")
        selected = selected[: args.episode_limit]

    adapter = None
    for episode in selected:
        if writer.is_episode_complete(episode):
            continue
        if adapter is None:
            adapter = _load_official_adapter(source_root, teacher_dir)
        values = processed[episode]
        processed_joints = np.concatenate(
            [
                values["observation.leg_joints"],
                values["observation.arm_joints"],
                values["observation.hand_joints"],
            ],
            axis=1,
        )
        _, arrays = build_episode_sidecar_arrays(
            processed_joints=processed_joints,
            raw_joint_candidates=raw_joint_candidates,
            raw_base_candidates=raw_base_candidates,
            raw_timestamp_candidates=raw_timestamp_candidates,
            processed_timestamps=values["timestamp"],
            adapter=adapter,
            raw_episode_index=replay_mapping[episode],
        )
        if args.formal_verified:
            validate_formal_reference(episode, replay_mapping[episode], arrays)
        writer.write_episode(episode, **arrays)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build hash-validated ARDY G1 tokenizer sidecars for aligned SIMPLE episodes."
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--ardy-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument(
        "--formal-verified",
        action="store_true",
        help="Require the pinned real episode-0 latent digest and publish a formal teacher context.",
    )
    return parser


def main() -> None:
    build_sidecars(_parser().parse_args())


if __name__ == "__main__":
    main()
