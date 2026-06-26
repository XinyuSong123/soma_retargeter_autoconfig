"""Runtime adapter from SOMA semantic source frames to V3 robot targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from soma_retargeter.robotics.v3 import target_builder
from soma_retargeter.robotics.v3.rest_frames import RestCalibration
from soma_retargeter.robotics.v3.spatial import matrix_to_quat_xyzw

from .diagnostics import build_target_delta_diagnostics
from .source_frames import DEFAULT_SEMANTIC_NAMES, SourceSemanticFrameBatch, validate_se3_transform
from .source_frames import extract_source_semantic_frames, transform7_to_matrix


SEMANTIC_TO_CAPABILITY_TASK = {
    "Chest": "torso",
    "LeftHand": "left_hand",
    "RightHand": "right_hand",
    "LeftFoot": "left_foot",
    "RightFoot": "right_foot",
}


@dataclass(frozen=True)
class RuntimeSemanticTargetBatch:
    semantic_names: list[str]
    transforms: dict[str, np.ndarray]
    frame_count: int
    sample_rate: float
    mode: str
    target_source: str
    capability_status: dict[str, str]
    segment_length_errors: dict[str, list[float]]
    provenance: dict[str, str]


@dataclass(frozen=True)
class EffectorOrderedTargets:
    effector_names: list[str]
    semantic_by_effector: list[str | None]
    transforms: np.ndarray
    available_mask: np.ndarray
    output_format: str


class RuntimeV3TargetAdapter:
    """Thin object wrapper used by pipeline integration agents."""

    def __init__(
        self,
        profile,
        *,
        semantic_names: Iterable[str] = DEFAULT_SEMANTIC_NAMES,
        mode: str = "runtime",
        config=None,
    ):
        if config is not None:
            semantic_names = getattr(config, "semantic_tasks", semantic_names)
            mode = getattr(config, "mode", mode)
        self.profile = profile
        self.semantic_names = list(semantic_names)
        self.mode = mode
        self.calibration = rest_calibration_from_profile(profile)

    def build_targets(self, source_batch: SourceSemanticFrameBatch) -> RuntimeSemanticTargetBatch:
        return build_runtime_semantic_targets(
            source_batch,
            self.profile,
            semantic_names=self.semantic_names,
            mode=self.mode,
            calibration=self.calibration,
        )

    def to_effector_order(
        self,
        targets: RuntimeSemanticTargetBatch,
        effector_names: Iterable[str],
        *,
        fill_missing_with: np.ndarray | None = None,
        output_format: str = "matrix",
    ) -> EffectorOrderedTargets:
        return semantic_targets_to_effector_order(
            targets,
            effector_names,
            fill_missing_with=fill_missing_with,
            output_format=output_format,
        )

    def compute_targets(self, **kwargs) -> dict:
        """Compatibility adapter used by ``NewtonPipeline`` shadow integration."""
        buffer = kwargs["buffer"]
        mode = str(kwargs.get("mode", self.mode))
        semantic_tasks = list(kwargs.get("semantic_tasks", self.semantic_names))
        diagnostics_max_frames = kwargs.get("diagnostics_max_frames")
        max_frames = int(diagnostics_max_frames) if mode == "shadow" and diagnostics_max_frames else None
        offset_transform = _offset_to_matrix(kwargs.get("offset"))
        source_batch = extract_source_semantic_frames(
            buffer,
            semantic_names=semantic_tasks,
            max_frames=max_frames,
            offset_transform=offset_transform,
            source=f"clip_{kwargs.get('clip_index', 0)}",
        )
        targets = build_runtime_semantic_targets(
            source_batch,
            self.profile,
            semantic_names=semantic_tasks,
            mode=mode,
            calibration=self.calibration,
        )
        legacy_effectors = np.asarray(kwargs["legacy_buffer_effectors"], dtype=np.float32)
        legacy_names = list(kwargs["legacy_effector_names"])
        ordered = semantic_targets_to_effector_order(
            targets,
            legacy_names,
            fill_missing_with=legacy_effectors[: targets.frame_count],
            output_format="warp",
        )
        diagnostics = build_target_delta_diagnostics(
            robot_type=str(kwargs.get("robot_type", "")),
            mode=mode,
            clip_name=f"clip_{kwargs.get('clip_index', 0)}",
            frame_count=targets.frame_count,
            semantic_names=semantic_tasks,
            legacy_transforms=_legacy_semantic_matrices(
                legacy_effectors[: targets.frame_count],
                legacy_names,
                semantic_tasks,
            ),
            v3_transforms=targets.transforms,
            capability_status=targets.capability_status,
            root_policy={
                "horizontal_scale": self.calibration.root_horizontal_scale,
                "support_height_policy": "source_support_height_delta",
                "vertical_root_scale": self.calibration.vertical_root_scale,
            },
            target_source=targets.target_source,
        )
        target_effectors = ordered.transforms.astype(np.float32)
        if target_effectors.shape[0] != legacy_effectors.shape[0]:
            full = np.array(legacy_effectors, copy=True)
            full[: target_effectors.shape[0]] = target_effectors
            target_effectors = full
        return {
            "target_effectors": target_effectors,
            "diagnostics": diagnostics,
        }


def build_runtime_semantic_targets(
    source_batch: SourceSemanticFrameBatch,
    profile,
    *,
    semantic_names: Iterable[str] = DEFAULT_SEMANTIC_NAMES,
    mode: str = "runtime",
    calibration: RestCalibration | None = None,
) -> RuntimeSemanticTargetBatch:
    names = list(semantic_names)
    calibration = rest_calibration_from_profile(profile) if calibration is None else calibration
    missing_source = [name for name in names if name not in source_batch.transforms]
    if missing_source:
        raise ValueError("source semantic batch is missing required semantic(s): " + ", ".join(missing_source))
    missing_profile = [name for name in names if name not in calibration.robot_neutral_site_transforms]
    if missing_profile:
        raise ValueError("runtime profile is missing robot semantic target(s): " + ", ".join(missing_profile))

    capability_status = capability_status_by_semantic(profile, names)
    unsupported = [
        name
        for name, status in capability_status.items()
        if name != "Hips" and status in {"unsupported", "missing_capability_task"}
    ]
    if unsupported:
        raise ValueError("runtime profile lacks capability for semantic target(s): " + ", ".join(unsupported))

    frame_count = int(source_batch.frame_count)
    stacked = {name: np.empty((frame_count, 4, 4), dtype=np.float64) for name in names}
    segment_errors = {name: [] for name in calibration.segment_lengths}
    provenance: dict[str, str] = {}
    for frame_index in range(frame_count):
        source_pose = {name: np.asarray(source_batch.transforms[name][frame_index], dtype=np.float64) for name in names}
        for name, transform in source_pose.items():
            validate_se3_transform(transform, context=f"source[{name}][frame={frame_index}]")
        semantic_targets = target_builder.build_targets_from_source_semantic_frames(
            calibration,
            source_pose,
            mode=mode,
        )
        provenance.update(semantic_targets.provenance or {})
        for name in names:
            if name not in semantic_targets.transforms:
                raise ValueError(f"target_builder did not produce semantic target {name!r}")
            target = np.asarray(semantic_targets.transforms[name], dtype=np.float64)
            validate_se3_transform(target, context=f"target[{name}][frame={frame_index}]")
            stacked[name][frame_index] = target
        for edge_name, error in semantic_targets.segment_length_errors.items():
            segment_errors.setdefault(edge_name, []).append(float(error))
    return RuntimeSemanticTargetBatch(
        semantic_names=names,
        transforms=stacked,
        frame_count=frame_count,
        sample_rate=float(source_batch.sample_rate),
        mode=mode,
        target_source="target_builder.build_targets_from_source_semantic_frames",
        capability_status=capability_status,
        segment_length_errors=segment_errors,
        provenance=provenance,
    )


def semantic_targets_to_effector_order(
    targets: RuntimeSemanticTargetBatch,
    effector_names: Iterable[str],
    *,
    semantic_to_effector: dict[str, str] | None = None,
    fill_missing_with: np.ndarray | None = None,
    output_format: str = "matrix",
) -> EffectorOrderedTargets:
    names = list(effector_names)
    semantic_for_effector = _semantic_for_effector(names, semantic_to_effector)
    fallback = None if fill_missing_with is None else _coerce_effector_matrices(fill_missing_with, len(names), targets.frame_count)
    out = np.empty((targets.frame_count, len(names), 4, 4), dtype=np.float64)
    available = np.zeros(len(names), dtype=bool)
    for index, semantic in enumerate(semantic_for_effector):
        if semantic is not None and semantic in targets.transforms:
            out[:, index] = targets.transforms[semantic]
            available[index] = True
        elif fallback is not None:
            out[:, index] = fallback[:, index]
        else:
            raise ValueError(f"no V3 semantic target for effector {names[index]!r}")
    if output_format == "matrix":
        output = out
    elif output_format == "warp":
        output = _matrices_to_transform7(out)
    else:
        raise ValueError("output_format must be 'matrix' or 'warp'")
    return EffectorOrderedTargets(
        effector_names=names,
        semantic_by_effector=semantic_for_effector,
        transforms=output,
        available_mask=available,
        output_format=output_format,
    )


def rest_calibration_from_profile(profile) -> RestCalibration:
    payload = _profile_payload(profile)
    data = payload.get("rest_calibration")
    if isinstance(data, RestCalibration):
        return data
    if not isinstance(data, dict):
        raise ValueError("runtime profile payload is missing rest_calibration")
    return RestCalibration(
        source_rest_semantic_frames=_matrix_dict(data.get("source_rest_semantic_frames", {}), "source_rest_semantic_frames"),
        source_provenance=str(data.get("source_provenance", "")),
        robot_neutral_site_transforms=_matrix_dict(data.get("robot_neutral_site_transforms", {}), "robot_neutral_site_transforms"),
        edge_alignment_rotations=_rotation_dict(data.get("edge_alignment_rotations", {}), "edge_alignment_rotations"),
        edge_conditioning={str(k): float(v) for k, v in data.get("edge_conditioning", {}).items()},
        edge_frame_sources={str(k): str(v) for k, v in data.get("edge_frame_sources", {}).items()},
        segment_lengths={str(k): float(v) for k, v in data.get("segment_lengths", {}).items()},
        root_horizontal_scale=float(data.get("root_horizontal_scale", 1.0)),
        vertical_root_scale=float(data.get("vertical_root_scale", 1.0)),
        source_support_height=float(data.get("source_support_height", 0.0)),
        robot_support_height=float(data.get("robot_support_height", 0.0)),
        neutral_position_errors={str(k): float(v) for k, v in data.get("neutral_position_errors", {}).items()},
        neutral_orientation_errors={str(k): float(v) for k, v in data.get("neutral_orientation_errors", {}).items()},
        max_position_error=float(data.get("max_position_error", 0.0)),
        max_orientation_error=float(data.get("max_orientation_error", 0.0)),
        bilateral_symmetry={str(k): float(v) for k, v in data.get("bilateral_symmetry", {}).items()},
        confidence=float(data.get("confidence", 0.0)),
        fallbacks=[str(v) for v in data.get("fallbacks", [])],
    )


def capability_status_by_semantic(profile, semantic_names: Iterable[str]) -> dict[str, str]:
    payload = _profile_payload(profile)
    per_task = payload.get("task_certificate_summary", {}).get("per_task", {})
    status: dict[str, str] = {}
    for semantic in semantic_names:
        if semantic == "Hips":
            status[semantic] = "root_reference"
            continue
        task = SEMANTIC_TO_CAPABILITY_TASK.get(semantic)
        if task is None:
            status[semantic] = "unsupported"
            continue
        row = per_task.get(task)
        if not isinstance(row, dict):
            status[semantic] = "missing_capability_task"
            continue
        statuses = row.get("statuses")
        if isinstance(statuses, list) and statuses:
            status[semantic] = ",".join(sorted({str(value) for value in statuses}))
        else:
            status[semantic] = str(row.get("status", "available"))
    return status


def _profile_payload(profile) -> dict:
    if isinstance(profile, dict):
        return profile
    for attr in ("payload", "profile", "data"):
        value = getattr(profile, attr, None)
        if isinstance(value, dict):
            return value
    if hasattr(profile, "to_json"):
        value = profile.to_json()
        if isinstance(value, dict):
            return value
    if hasattr(profile, "rest_calibration"):
        return {
            "rest_calibration": getattr(profile, "rest_calibration"),
            "task_certificate_summary": getattr(profile, "task_certificate_summary", {}),
            "status": getattr(profile, "status", ""),
        }
    raise ValueError(f"unsupported runtime profile object: {type(profile).__name__}")


def _matrix_dict(data: dict, name: str) -> dict[str, np.ndarray]:
    out = {}
    for key, value in data.items():
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"{name}.{key} must be shaped (4, 4), got {matrix.shape}")
        validate_se3_transform(matrix, context=f"{name}.{key}")
        out[str(key)] = matrix
    return out


def _rotation_dict(data: dict, name: str) -> dict[str, np.ndarray]:
    out = {}
    for key, value in data.items():
        rotation = np.asarray(value, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"{name}.{key} must be shaped (3, 3), got {rotation.shape}")
        if not np.all(np.isfinite(rotation)):
            raise ValueError(f"{name}.{key} contains non-finite values")
        out[str(key)] = rotation
    return out


def _semantic_for_effector(names: list[str], semantic_to_effector: dict[str, str] | None) -> list[str | None]:
    if semantic_to_effector is None:
        semantic_by_effector = {name: name for name in DEFAULT_SEMANTIC_NAMES}
    else:
        semantic_by_effector = {effector: semantic for semantic, effector in semantic_to_effector.items()}
    return [semantic_by_effector.get(name) for name in names]


def _coerce_effector_matrices(value: np.ndarray, effector_count: int, frame_count: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (frame_count, effector_count, 4, 4):
        out = arr.copy()
    elif arr.shape == (frame_count, effector_count, 7):
        out = np.empty((frame_count, effector_count, 4, 4), dtype=np.float64)
        from .source_frames import transform7_to_matrix

        for frame_index in range(frame_count):
            for effector_index in range(effector_count):
                out[frame_index, effector_index] = transform7_to_matrix(arr[frame_index, effector_index])
    else:
        raise ValueError(
            "fill_missing_with must be shaped "
            f"({frame_count}, {effector_count}, 4, 4) or ({frame_count}, {effector_count}, 7), got {arr.shape}"
        )
    for frame_index in range(frame_count):
        for effector_index in range(effector_count):
            validate_se3_transform(out[frame_index, effector_index], context=f"fill_missing_with[{frame_index},{effector_index}]")
    return out


def _matrices_to_transform7(matrices: np.ndarray) -> np.ndarray:
    out = np.empty(matrices.shape[:2] + (7,), dtype=np.float64)
    for frame_index in range(matrices.shape[0]):
        for effector_index in range(matrices.shape[1]):
            matrix = matrices[frame_index, effector_index]
            validate_se3_transform(matrix, context=f"output[{frame_index},{effector_index}]")
            out[frame_index, effector_index, :3] = matrix[:3, 3]
            out[frame_index, effector_index, 3:7] = matrix_to_quat_xyzw(matrix[:3, :3])
    return out


def _legacy_semantic_matrices(
    legacy_effectors: np.ndarray,
    effector_names: list[str],
    semantic_names: list[str],
) -> dict[str, np.ndarray]:
    effector_index = {name: index for index, name in enumerate(effector_names)}
    out: dict[str, np.ndarray] = {}
    for semantic_name in semantic_names:
        if semantic_name not in effector_index:
            continue
        idx = effector_index[semantic_name]
        frames = []
        for frame_index in range(legacy_effectors.shape[0]):
            frames.append(transform7_to_matrix(legacy_effectors[frame_index, idx]))
        out[semantic_name] = np.stack(frames).astype(np.float64)
    return out


def _offset_to_matrix(offset) -> np.ndarray | None:
    if offset is None:
        return None
    arr = np.asarray(offset, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.reshape(-1).shape[0] == 7:
        return transform7_to_matrix(arr)
    raise ValueError(f"offset must be shaped (4, 4) or transform7, got {arr.shape}")
