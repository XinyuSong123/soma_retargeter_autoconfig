"""Deterministic runtime V3 target diagnostics."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

import numpy as np

from soma_retargeter.robotics.v3.spatial import rotation_error


TARGET_BUILDER_SOURCE = "target_builder.build_targets_from_source_semantic_frames"


def build_target_delta_diagnostics(
    *,
    robot_type: str,
    mode: str,
    clip_name: str,
    frame_count: int,
    semantic_names: Iterable[str],
    legacy_transforms: dict[str, np.ndarray],
    v3_transforms: dict[str, np.ndarray],
    capability_status: dict[str, str] | None = None,
    root_policy: dict | None = None,
    target_source: str = TARGET_BUILDER_SOURCE,
) -> dict:
    names = list(semantic_names)
    capability_status = capability_status or {}
    per_semantic = {}
    for semantic in names:
        legacy = legacy_transforms.get(semantic)
        v3 = v3_transforms.get(semantic)
        per_semantic[semantic] = _semantic_metrics(
            semantic,
            legacy,
            v3,
            frame_count=int(frame_count),
            capability_status=capability_status.get(semantic, "unknown"),
            target_source=target_source,
        )
    return {
        "schema_version": 1,
        "robot_type": str(robot_type),
        "mode": str(mode),
        "clip_name": str(clip_name),
        "frame_count": int(frame_count),
        "semantic_names": names,
        "legacy_target_available": {name: bool(name in legacy_transforms) for name in names},
        "v3_target_available": {name: bool(name in v3_transforms) for name in names},
        "per_semantic": per_semantic,
        "root_policy": root_policy or {},
        "capability_policy": {"status_by_semantic": {name: capability_status.get(name, "unknown") for name in names}},
    }


def write_deterministic_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _semantic_metrics(
    semantic: str,
    legacy: np.ndarray | None,
    v3: np.ndarray | None,
    *,
    frame_count: int,
    capability_status: str,
    target_source: str,
) -> dict:
    missing_reasons = []
    if legacy is None:
        missing_reasons.append("missing_legacy_target")
    if v3 is None:
        missing_reasons.append("missing_v3_target")
    if missing_reasons:
        return _skipped_metrics(capability_status, target_source, ",".join(missing_reasons))
    legacy_arr = _coerce_semantic_stack(legacy, semantic, "legacy", frame_count)
    v3_arr = _coerce_semantic_stack(v3, semantic, "v3", frame_count)
    finite_mask = np.isfinite(legacy_arr).all(axis=(1, 2)) & np.isfinite(v3_arr).all(axis=(1, 2))
    finite_count = int(np.count_nonzero(finite_mask))
    nan_count = int(frame_count - finite_count)
    if finite_count == 0:
        return _skipped_metrics(capability_status, target_source, "no_finite_target_pairs", nan_count=nan_count)
    legacy_finite = legacy_arr[finite_mask]
    v3_finite = v3_arr[finite_mask]
    translation = np.linalg.norm(v3_finite[:, :3, 3] - legacy_finite[:, :3, 3], axis=1)
    rotation = np.asarray(
        [rotation_error(legacy_finite[index, :3, :3], v3_finite[index, :3, :3]) for index in range(finite_count)],
        dtype=np.float64,
    )
    return {
        "translation_delta_mean": _stable_stat(float(np.mean(translation))),
        "translation_delta_max": _stable_stat(float(np.max(translation))),
        "translation_delta_p95": _stable_stat(float(np.percentile(translation, 95))),
        "rotation_delta_mean": _stable_stat(float(np.mean(rotation))),
        "rotation_delta_max": _stable_stat(float(np.max(rotation))),
        "rotation_delta_p95": _stable_stat(float(np.percentile(rotation, 95))),
        "finite_count": finite_count,
        "nan_count": nan_count,
        "skipped_reason": "",
        "target_source": target_source,
        "capability_status": capability_status,
    }


def _skipped_metrics(
    capability_status: str,
    target_source: str,
    skipped_reason: str,
    *,
    nan_count: int = 0,
) -> dict:
    return {
        "translation_delta_mean": None,
        "translation_delta_max": None,
        "translation_delta_p95": None,
        "rotation_delta_mean": None,
        "rotation_delta_max": None,
        "rotation_delta_p95": None,
        "finite_count": 0,
        "nan_count": int(nan_count),
        "skipped_reason": skipped_reason,
        "target_source": target_source,
        "capability_status": capability_status,
    }


def _coerce_semantic_stack(value: np.ndarray, semantic: str, label: str, frame_count: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (frame_count, 4, 4):
        raise ValueError(f"{label} target for {semantic} must be shaped ({frame_count}, 4, 4), got {arr.shape}")
    return arr


def _stable_stat(value: float) -> float:
    value = float(value)
    if abs(value) < 1e-15:
        return 0.0
    return round(value, 12)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
