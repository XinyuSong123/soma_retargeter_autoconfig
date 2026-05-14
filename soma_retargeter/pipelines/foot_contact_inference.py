from __future__ import annotations

from typing import Any

import numpy as np

try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal test envs
    wp = None


CONTACT_SCORE_NAMES = ("left_heel", "left_toe", "right_heel", "right_toe")
DEFAULT_NPZ_CONTACT_ORDER = CONTACT_SCORE_NAMES
DEFAULT_SOURCE_FOOT_JOINT_ALIASES = {
    "left_toe": ["LeftToeBase", "LeftToeEnd", "LeftToe"],
    "left_heel": ["LeftFoot"],
    "right_toe": ["RightToeBase", "RightToeEnd", "RightToe"],
    "right_heel": ["RightFoot"],
}
_MIN_HEIGHT_SCALE_M = 0.015
_MIN_VELOCITY_SCALE_MPS = 0.08


def _smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.clip(scores, 0.0, 1.0)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(scores, (window // 2, window - 1 - window // 2), mode="edge")
    return np.clip(np.convolve(padded, kernel, mode="valid"), 0.0, 1.0)


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _estimate_ground_height(traces: list[np.ndarray]) -> float:
    trace_z = [trace[:, 2] for trace in traces if len(trace)]
    if not trace_z:
        return 0.0
    z_values = np.concatenate(trace_z)
    return min(float(np.percentile(z_values, 2.0)), 0.0)


def _estimate_height_scale(positions: np.ndarray, ground_height: float) -> float:
    if len(positions) <= 1:
        return _MIN_HEIGHT_SCALE_M
    relative_z = np.maximum(positions[:, 2] - ground_height, 0.0)
    spread = float(np.percentile(relative_z, 90.0) - np.percentile(relative_z, 10.0))
    return float(np.clip(spread * 0.35, _MIN_HEIGHT_SCALE_M, 0.08))


def _horizontal_velocity(positions: np.ndarray, velocity_dt: float) -> np.ndarray:
    vel = np.zeros(len(positions), dtype=np.float32)
    if len(positions) > 1:
        vel[1:] = np.linalg.norm((positions[1:, :2] - positions[:-1, :2]) / max(velocity_dt, 1e-6), axis=1)
        vel[0] = vel[1]
    return vel


def _estimate_velocity_scale(positions: np.ndarray, ground_height: float, height_scale: float, velocity_dt: float) -> float:
    vel = _horizontal_velocity(positions, velocity_dt)
    if len(vel) == 0:
        return _MIN_VELOCITY_SCALE_MPS
    near_ground = positions[:, 2] <= ground_height + max(height_scale * 2.0, 0.02)
    sample = vel[near_ground] if np.any(near_ground) else vel
    moving = sample[sample > 1e-5]
    if len(moving) == 0:
        return _MIN_VELOCITY_SCALE_MPS
    return float(np.clip(np.percentile(moving, 75.0) * 0.5, _MIN_VELOCITY_SCALE_MPS, 1.0))


def _contact_score_from_positions(
    positions: np.ndarray,
    ground_height: float | None = None,
    velocity_dt: float = 1.0 / 60.0,
    *,
    contact_height_scale: float | None = None,
    contact_velocity_scale: float | None = None,
) -> np.ndarray:
    z = positions[:, 2]
    if ground_height is None:
        ground_height = float(np.percentile(z, 2.0)) if len(z) else 0.0
    height_scale = _numeric_or_none(contact_height_scale)
    if height_scale is None or height_scale <= 0.0:
        height_scale = _estimate_height_scale(positions, float(ground_height))
    velocity_scale = _numeric_or_none(contact_velocity_scale)
    if velocity_scale is None or velocity_scale <= 0.0:
        velocity_scale = _estimate_velocity_scale(positions, float(ground_height), height_scale, velocity_dt)
    vel = _horizontal_velocity(positions, velocity_dt)
    height_gate = np.exp(-np.maximum(z - float(ground_height), 0.0) / max(height_scale, 1e-6))
    vel_gate = np.exp(-vel / max(velocity_scale, 1e-6))
    return np.clip((height_gate * vel_gate).astype(np.float32), 0.0, 1.0)


def _normalized_aliases(source_foot_joint_aliases: dict[str, Any] | None) -> dict[str, list[str]]:
    aliases = {key: list(values) for key, values in DEFAULT_SOURCE_FOOT_JOINT_ALIASES.items()}
    if not source_foot_joint_aliases:
        return aliases
    for key, value in source_foot_joint_aliases.items():
        if key not in CONTACT_SCORE_NAMES:
            continue
        if isinstance(value, str):
            aliases[key] = [value]
        elif isinstance(value, (list, tuple)):
            aliases[key] = [str(item) for item in value if item is not None]
    return aliases


def _translation(transform) -> np.ndarray:
    if wp is not None:
        try:
            return np.array(wp.transform_get_translation(transform), dtype=np.float32)
        except Exception:
            pass
    values = np.asarray(transform, dtype=np.float32)
    if values.shape[0] < 3:
        raise ValueError(f"Cannot extract translation from transform with shape {values.shape}")
    return values[:3]


def infer_contacts_from_animation_buffer(
    buffer,
    root_tx=None,
    smoothing_window: int = 5,
    *,
    source_foot_joint_aliases: dict[str, Any] | None = None,
    contact_height_scale: float | None = None,
    contact_velocity_scale: float | None = None,
    ground_height_m: float | None = None,
) -> dict[str, np.ndarray]:
    skel = buffer.skeleton
    names = _normalized_aliases(source_foot_joint_aliases)

    def pick(name_list):
        for n in name_list:
            idx = skel.joint_index(n)
            if idx != -1:
                return idx
        return -1

    joint_indices = {k: pick(v) for k, v in names.items()}
    if any(v == -1 for v in joint_indices.values()):
        missing = [f"{k} aliases {names[k]!r}" for k, v in joint_indices.items() if v == -1]
        raise ValueError(f"Cannot infer contacts; missing source foot alias groups: {missing}")

    if root_tx is None and wp is not None:
        root_tx = wp.transform_identity()
    traces = {k: np.zeros((buffer.num_frames, 3), dtype=np.float32) for k in joint_indices}
    for frame in range(buffer.num_frames):
        if root_tx is None:
            gtx = buffer.compute_global_transforms(frame)
        else:
            gtx = buffer.compute_global_transforms(frame, root_tx=root_tx)
        for key, idx in joint_indices.items():
            traces[key][frame] = _translation(gtx[idx])

    ground = _numeric_or_none(ground_height_m)
    if ground is None:
        ground = _estimate_ground_height(list(traces.values()))
    dt = 1.0 / max(buffer.sample_rate, 1e-3)

    out = {}
    for key, pos in traces.items():
        out[f"{key}_contact_score"] = _smooth_scores(
            _contact_score_from_positions(
                pos,
                ground,
                dt,
                contact_height_scale=contact_height_scale,
                contact_velocity_scale=contact_velocity_scale,
            ),
            smoothing_window,
        )
    return out


def _normalize_contact_order(contact_order: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if contact_order is None:
        return DEFAULT_NPZ_CONTACT_ORDER
    order = tuple(str(name) for name in contact_order)
    unsupported = [name for name in order if name not in CONTACT_SCORE_NAMES]
    if unsupported:
        raise ValueError(f"Unsupported contact_order entries: {unsupported}; supported names are {CONTACT_SCORE_NAMES}")
    missing = [name for name in CONTACT_SCORE_NAMES if name not in order]
    if missing:
        raise ValueError(f"contact_order is missing required entries: {missing}")
    if len(set(order)) != len(order):
        raise ValueError(f"contact_order entries must be unique: {order}")
    return order


def contacts_from_npz_foot_contacts(
    foot_contacts: np.ndarray,
    smoothing_window: int = 5,
    *,
    contact_order: list[str] | tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Map NPZ foot-contact columns to named scores.

    When ``contact_order`` is omitted, the backward-compatible default order is
    ``[left_heel, left_toe, right_heel, right_toe]``.
    """

    order = _normalize_contact_order(contact_order)
    if foot_contacts.ndim != 2 or foot_contacts.shape[1] < len(order):
        raise ValueError(f"foot_contacts must have shape [T,{len(order)}+] in {list(order)} order")
    mapped = {
        f"{name}_contact_score": foot_contacts[:, column].astype(np.float32)
        for column, name in enumerate(order)
        if name in CONTACT_SCORE_NAMES
    }
    return {k: _smooth_scores(np.clip(v, 0.0, 1.0), smoothing_window) for k, v in mapped.items()}
