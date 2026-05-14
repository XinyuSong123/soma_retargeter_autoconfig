import copy
import math
from typing import Any

import numpy as np


DEBUG_CONFIG_KEY = "stance_width_diagnostics"
REPORT_SUFFIX = ".stance_width_report.json"

_FOOT_JOINTS = ("LeftFoot", "RightFoot")
_UPPER_BODY_ROTATION_JOINTS = ("Hips", "Pelvis", "Chest")
_POSITION_ONLY_JOINTS = ("Hips", "Pelvis", "LeftFoot", "RightFoot")
_ROOT_CANDIDATES = ("Hips", "Pelvis", "Root", "base_link")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def normalize_debug_config(raw_config: Any) -> dict[str, Any]:
    cfg = dict(raw_config) if isinstance(raw_config, dict) else {}
    lower_foot_rotation_weight = _as_optional_float(cfg.get("lower_foot_rotation_weight"))
    ablation_enabled = (
        _as_bool(cfg.get("disable_foot_rotation_tracking", False))
        or lower_foot_rotation_weight is not None
        or _as_bool(cfg.get("disable_upper_body_rotation_tracking", False))
        or _as_bool(cfg.get("position_only_lower_body_debug", False))
    )
    return {
        "enabled": _as_bool(cfg.get("enabled", False)) or ablation_enabled,
        "disable_foot_rotation_tracking": _as_bool(cfg.get("disable_foot_rotation_tracking", False)),
        "lower_foot_rotation_weight": lower_foot_rotation_weight,
        "disable_upper_body_rotation_tracking": _as_bool(
            cfg.get("disable_upper_body_rotation_tracking", False)
        ),
        "position_only_lower_body_debug": _as_bool(cfg.get("position_only_lower_body_debug", False)),
        "report_suffix": str(cfg.get("report_suffix", REPORT_SUFFIX) or REPORT_SUFFIX),
    }


def apply_debug_options_to_ik_map(
    ik_map: dict[str, Any],
    debug_config: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = normalize_debug_config(debug_config)
    effective = copy.deepcopy(ik_map)
    summary = {
        "enabled": cfg["enabled"],
        "options": cfg,
        "applied_changes": [],
    }
    if not cfg["enabled"]:
        return effective, summary

    def set_r_weight(joint_name: str, value: float):
        entry = effective.get(joint_name)
        if isinstance(entry, dict) and "r_weight" in entry:
            old_value = float(entry.get("r_weight", 0.0))
            entry["r_weight"] = float(value)
            summary["applied_changes"].append(
                {"joint": joint_name, "field": "r_weight", "old": old_value, "new": float(value)}
            )

    def scale_r_weight(joint_name: str, factor: float):
        entry = effective.get(joint_name)
        if isinstance(entry, dict) and "r_weight" in entry:
            old_value = float(entry.get("r_weight", 0.0))
            new_value = old_value * factor
            entry["r_weight"] = new_value
            summary["applied_changes"].append(
                {"joint": joint_name, "field": "r_weight", "old": old_value, "new": new_value}
            )

    if cfg["position_only_lower_body_debug"]:
        selected = {
            joint_name: entry
            for joint_name, entry in effective.items()
            if joint_name in _POSITION_ONLY_JOINTS
        }
        effective = selected
        summary["applied_changes"].append(
            {"mode": "position_only_lower_body_debug", "kept_joints": sorted(selected)}
        )
        for joint_name in list(effective):
            set_r_weight(joint_name, 0.0)
        return effective, summary

    if cfg["disable_foot_rotation_tracking"]:
        for joint_name in _FOOT_JOINTS:
            set_r_weight(joint_name, 0.0)
    elif cfg["lower_foot_rotation_weight"] is not None:
        for joint_name in _FOOT_JOINTS:
            scale_r_weight(joint_name, cfg["lower_foot_rotation_weight"])

    if cfg["disable_upper_body_rotation_tracking"]:
        for joint_name in _UPPER_BODY_ROTATION_JOINTS:
            set_r_weight(joint_name, 0.0)

    return effective, summary


def _clean_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def numeric_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    return {
        "count": int(len(arr)),
        "mean": _clean_float(np.mean(arr)),
        "min": _clean_float(np.min(arr)),
        "max": _clean_float(np.max(arr)),
        "std": _clean_float(np.std(arr)),
    }


def component_stats(values: Any, labels=("x", "y", "z")) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        label: numeric_stats(arr[:, idx])
        for idx, label in enumerate(labels)
        if idx < arr.shape[1]
    }


def _resolve_index(names: list[str], candidates: tuple[str, ...] | list[str]) -> tuple[int | None, str | None]:
    for name in candidates:
        if name in names:
            return names.index(name), name
    return None, None


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-12)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.array(q, copy=True)
    out[..., :3] *= -1.0
    return out


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _normalize_quat(q)
    xyz = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _quat_angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _normalize_quat(a)
    b = _normalize_quat(b)
    dots = np.abs(np.sum(a * b, axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    return 2.0 * np.arccos(dots)


def _angle_from_first(quats: np.ndarray) -> np.ndarray:
    if len(quats) == 0:
        return np.asarray([], dtype=np.float64)
    return _quat_angle_delta(quats, np.repeat(quats[:1], len(quats), axis=0))


def width_stats_from_named_transforms(
    transforms: Any,
    names: list[str],
    *,
    left_name: str = "LeftFoot",
    right_name: str = "RightFoot",
    root_candidates: tuple[str, ...] = _ROOT_CANDIDATES,
) -> dict[str, Any]:
    arr = np.asarray(transforms, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] < 7:
        return {"available": False, "reason": "transforms must have shape (frames, items, 7)"}

    left_idx, resolved_left = _resolve_index(names, (left_name,))
    right_idx, resolved_right = _resolve_index(names, (right_name,))
    if left_idx is None or right_idx is None:
        return {
            "available": False,
            "reason": "left/right foot transforms are missing",
            "left_name": resolved_left,
            "right_name": resolved_right,
        }

    root_idx, root_name = _resolve_index(names, root_candidates)
    left_pos = arr[:, left_idx, :3]
    right_pos = arr[:, right_idx, :3]
    delta = left_pos - right_pos
    frame = "world"
    if root_idx is not None:
        frame = "root_local"
        root_pos = arr[:, root_idx, :3]
        root_rot_inv = _quat_conjugate(arr[:, root_idx, 3:7])
        left_local = _quat_rotate(root_rot_inv, left_pos - root_pos)
        right_local = _quat_rotate(root_rot_inv, right_pos - root_pos)
        delta = left_local - right_local

    horizontal_width = np.linalg.norm(delta[:, :2], axis=1)
    return {
        "available": True,
        "frame": frame,
        "root_name": root_name,
        "left_name": resolved_left,
        "right_name": resolved_right,
        "horizontal_width_m": numeric_stats(horizontal_width),
        "lateral_x_abs_m": numeric_stats(np.abs(delta[:, 0])),
        "delta_local_m": component_stats(delta),
    }


def foot_target_stats(transforms: Any, names: list[str]) -> dict[str, Any]:
    arr = np.asarray(transforms, dtype=np.float64)
    out = {}
    for foot_name in _FOOT_JOINTS:
        idx, resolved = _resolve_index(names, (foot_name,))
        if idx is None or arr.ndim != 3 or arr.shape[-1] < 7:
            out[foot_name] = {"available": False, "reason": f"{foot_name} target is missing"}
            continue
        quats = arr[:, idx, 3:7]
        out[foot_name] = {
            "available": True,
            "position_m": component_stats(arr[:, idx, :3]),
            "rotation_xyzw": component_stats(quats, labels=("x", "y", "z", "w")),
            "rotation_angle_from_first_rad": numeric_stats(_angle_from_first(quats)),
        }
    return out


def foot_ik_residual_stats(
    scaled_targets: Any,
    scaled_names: list[str],
    robot_fk_transforms: Any,
    robot_names: list[str],
) -> dict[str, Any]:
    scaled = np.asarray(scaled_targets, dtype=np.float64)
    robot = np.asarray(robot_fk_transforms, dtype=np.float64)
    out = {}
    all_position_residuals = []
    all_rotation_residuals = []
    for foot_name in _FOOT_JOINTS:
        target_idx, _ = _resolve_index(scaled_names, (foot_name,))
        robot_idx, _ = _resolve_index(robot_names, (foot_name,))
        if (
            target_idx is None
            or robot_idx is None
            or scaled.ndim != 3
            or robot.ndim != 3
            or scaled.shape[-1] < 7
            or robot.shape[-1] < 7
        ):
            out[foot_name] = {"available": False, "reason": f"{foot_name} target or FK is missing"}
            continue

        n = min(len(scaled), len(robot))
        position_residual = np.linalg.norm(scaled[:n, target_idx, :3] - robot[:n, robot_idx, :3], axis=1)
        rotation_residual = _quat_angle_delta(scaled[:n, target_idx, 3:7], robot[:n, robot_idx, 3:7])
        all_position_residuals.append(position_residual)
        all_rotation_residuals.append(rotation_residual)
        out[foot_name] = {
            "available": True,
            "position_m": numeric_stats(position_residual),
            "rotation_rad": numeric_stats(rotation_residual),
        }

    position = np.concatenate(all_position_residuals) if all_position_residuals else np.asarray([])
    rotation = np.concatenate(all_rotation_residuals) if all_rotation_residuals else np.asarray([])
    out["summary"] = {
        "foot_position_m": numeric_stats(position),
        "foot_rotation_rad": numeric_stats(rotation),
    }
    return out


def joint_stats_from_raw_data(raw_data: Any, joint_name_to_q_index: dict[str, int]) -> dict[str, Any]:
    data = np.asarray(raw_data, dtype=np.float64)
    if data.ndim != 2:
        return {}
    out = {}
    for joint_name, q_index in sorted(joint_name_to_q_index.items()):
        lower = joint_name.lower()
        is_leg_yaw_or_roll = (
            ("hip" in lower or "thigh" in lower)
            and ("yaw" in lower or "roll" in lower)
        )
        if not is_leg_yaw_or_roll or q_index < 0 or q_index >= data.shape[1]:
            continue
        out[joint_name] = {
            "q_index": int(q_index),
            "rad": numeric_stats(data[:, q_index]),
        }
    return out


def ik_map_summary(ik_map: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for joint_name in ("LeftFoot", "RightFoot", "Hips", "Pelvis", "Chest"):
        entry = ik_map.get(joint_name)
        if isinstance(entry, dict):
            out[joint_name] = {
                "t_body": entry.get("t_body"),
                "r_body": entry.get("r_body"),
                "t_weight": entry.get("t_weight"),
                "r_weight": entry.get("r_weight"),
            }
        elif entry is not None:
            out[joint_name] = {
                "t_body": entry,
                "r_body": entry,
                "t_weight": None,
                "r_weight": None,
            }
        else:
            out[joint_name] = None
    return out


def _mean_width(width_stats: dict[str, Any]) -> float | None:
    if not isinstance(width_stats, dict) or not width_stats.get("available"):
        return None
    stats = width_stats.get("horizontal_width_m", {})
    return stats.get("mean")


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) < 1e-9:
        return None
    return _clean_float(float(numerator) / float(denominator))


def build_stance_width_report(
    *,
    source_soma_transforms: Any,
    source_soma_names: list[str],
    scaled_target_transforms: Any,
    scaled_target_names: list[str],
    robot_fk_transforms: Any | None = None,
    robot_fk_names: list[str] | None = None,
    ik_map: dict[str, Any] | None = None,
    debug_summary: dict[str, Any] | None = None,
    hip_thigh_joint_stats: dict[str, Any] | None = None,
    motion_name: str | None = None,
) -> dict[str, Any]:
    source_width = width_stats_from_named_transforms(source_soma_transforms, source_soma_names)
    scaled_width = width_stats_from_named_transforms(scaled_target_transforms, scaled_target_names)
    robot_width = (
        width_stats_from_named_transforms(robot_fk_transforms, robot_fk_names or [])
        if robot_fk_transforms is not None
        else {"available": False, "reason": "robot FK was not provided"}
    )

    source_mean = _mean_width(source_width)
    scaled_mean = _mean_width(scaled_width)
    robot_mean = _mean_width(robot_width)

    report = {
        "diagnostic": "stance_width",
        "motion_name": motion_name,
        "source_soma_left_right_foot_width_stats": source_width,
        "scaled_target_left_right_foot_width_stats": scaled_width,
        "robot_fk_left_right_foot_width_stats": robot_width,
        "robot_fk_stage": "after_ik",
        "ratio_scaled_to_source": _ratio(scaled_mean, source_mean),
        "ratio_robot_to_scaled": _ratio(robot_mean, scaled_mean),
        "foot_target_position_stats": {
            foot: payload.get("position_m") if payload.get("available") else payload
            for foot, payload in foot_target_stats(scaled_target_transforms, scaled_target_names).items()
        },
        "foot_target_rotation_stats": {
            foot: {
                "rotation_xyzw": payload.get("rotation_xyzw"),
                "rotation_angle_from_first_rad": payload.get("rotation_angle_from_first_rad"),
            }
            if payload.get("available")
            else payload
            for foot, payload in foot_target_stats(scaled_target_transforms, scaled_target_names).items()
        },
        "ik_residual": (
            foot_ik_residual_stats(
                scaled_target_transforms,
                scaled_target_names,
                robot_fk_transforms,
                robot_fk_names or [],
            )
            if robot_fk_transforms is not None
            else {"available": False, "reason": "robot FK was not provided"}
        ),
        "hip_thigh_yaw_roll_joint_stats": hip_thigh_joint_stats or {},
        "current_ik_map_bodies_and_weights": ik_map_summary(ik_map or {}),
        "debug_options": debug_summary or {"enabled": False},
    }
    return report
