"""Step 3.1 target stream generation for runtime quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from soma_retargeter.animation.animation_buffer import AnimationBuffer

from .clip_inventory import ClipInventoryEntry, FrameBudget, deterministic_frame_budget, inspect_motion_clip
from .source_frames import DEFAULT_SEMANTIC_NAMES, extract_source_semantic_frames, validate_se3_transform
from .target_adapter import RuntimeSemanticTargetBatch, build_runtime_semantic_targets


REQUIRED_TARGET_SEMANTICS: tuple[str, ...] = tuple(DEFAULT_SEMANTIC_NAMES)
FULL_PROFILE_STATUSES = frozenset({"passed", "capability_limited_passed"})
PARTIAL_PROFILE_STATUSES = frozenset({"partial_passed", "partial_humanoid"})
FULL_CAPABILITY_STATUSES = frozenset({"full_humanoid_ready", "capability_limited_humanoid"})
PARTIAL_CAPABILITY_STATUSES = frozenset({"partial_humanoid"})


@dataclass(frozen=True)
class TargetStreamResult:
    model_id: str
    profile_kind: str
    clip_path: str
    status: str
    target_stream_status: str
    semantic_names: tuple[str, ...]
    supported_semantics: tuple[str, ...]
    missing_semantics: tuple[str, ...]
    frame_budget: FrameBudget | None
    frame_count: int | None
    sample_rate: float | None
    target_source: str | None
    finite: bool | None
    se3_valid: bool | None
    per_semantic: dict[str, dict[str, Any]]
    capability_status: dict[str, str]
    failure: dict[str, Any] | None = None
    target_batch: RuntimeSemanticTargetBatch | None = field(default=None, repr=False, compare=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "profile_kind": self.profile_kind,
            "clip_path": self.clip_path,
            "status": self.status,
            "target_stream_status": self.target_stream_status,
            "semantic_names": list(self.semantic_names),
            "supported_semantics": list(self.supported_semantics),
            "missing_semantics": list(self.missing_semantics),
            "frame_budget": None if self.frame_budget is None else self.frame_budget.to_json(),
            "frame_count": self.frame_count,
            "sample_rate": self.sample_rate,
            "target_source": self.target_source,
            "finite": self.finite,
            "se3_valid": self.se3_valid,
            "per_semantic": self.per_semantic,
            "capability_status": self.capability_status,
            "failure": self.failure,
        }


@dataclass(frozen=True)
class TargetStreamMatrix:
    schema_version: int
    status: str
    rows: tuple[TargetStreamResult, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "rows": [row.to_json() for row in self.rows],
        }


def generate_target_stream_for_clip(
    profile: Any,
    clip_path: str | Path,
    *,
    repo_root: str | Path = ".",
    mode: str = "target_stream_only",
    frame_budget: FrameBudget | None = None,
    partial_report_only: bool = True,
    fail_fast: bool = False,
) -> TargetStreamResult:
    """Generate or report the Step 3.1 target stream for one profile and clip."""

    root = Path(repo_root).resolve()
    payload = _profile_payload(profile)
    model_id = _profile_model_id(payload, profile)
    profile_kind = classify_profile_kind(profile)
    clip_info = inspect_motion_clip(clip_path, repo_root=root)
    budget = frame_budget or clip_info.frame_budget

    if profile_kind == "partial_humanoid" and partial_report_only:
        return _partial_supported_semantics_result(
            profile,
            clip_info=clip_info,
            repo_root=root,
            frame_budget=budget,
        )

    semantic_names = semantic_tasks_for_profile(profile, repo_root=root, profile_kind=profile_kind)
    if not semantic_names:
        return _failed_result(
            model_id=model_id,
            profile_kind=profile_kind,
            clip_info=clip_info,
            frame_budget=budget,
            code="no_supported_semantics",
            message="profile does not declare supported target semantics",
            semantic_names=(),
            supported_semantics=(),
        )

    if clip_info.format != "bvh":
        return _failed_result(
            model_id=model_id,
            profile_kind=profile_kind,
            clip_info=clip_info,
            frame_budget=budget,
            code="unsupported_clip_format",
            message="target streams require BVH SOMA source clips",
            semantic_names=tuple(semantic_names),
            supported_semantics=tuple(semantic_names),
        )
    if clip_info.load_status != "loaded":
        return _failed_result(
            model_id=model_id,
            profile_kind=profile_kind,
            clip_info=clip_info,
            frame_budget=budget,
            code="clip_unloadable",
            message=clip_info.error or f"clip load status is {clip_info.load_status}",
            semantic_names=tuple(semantic_names),
            supported_semantics=tuple(semantic_names),
        )

    try:
        animation = _load_bvh_animation(_resolve_clip_path(clip_path, root))
        actual_budget = frame_budget or deterministic_frame_budget(int(animation.num_frames))
        source_buffer = _slice_animation_buffer(animation, actual_budget)
        source_batch = extract_source_semantic_frames(
            source_buffer,
            semantic_names=semantic_names,
            source=clip_info.path,
        )
        targets = build_runtime_semantic_targets(
            source_batch,
            profile,
            semantic_names=semantic_names,
            mode=mode,
        )
        metrics = target_stream_metrics(targets)
    except Exception as exc:
        if fail_fast:
            raise
        return _failed_result(
            model_id=model_id,
            profile_kind=profile_kind,
            clip_info=clip_info,
            frame_budget=budget,
            code=_failure_code(exc),
            message=str(exc),
            semantic_names=tuple(semantic_names),
            supported_semantics=tuple(semantic_names),
        )

    return TargetStreamResult(
        model_id=model_id,
        profile_kind=profile_kind,
        clip_path=clip_info.path,
        status="passed",
        target_stream_status="generated",
        semantic_names=tuple(targets.semantic_names),
        supported_semantics=tuple(semantic_names),
        missing_semantics=tuple(name for name in REQUIRED_TARGET_SEMANTICS if name not in semantic_names),
        frame_budget=actual_budget,
        frame_count=targets.frame_count,
        sample_rate=targets.sample_rate,
        target_source=targets.target_source,
        finite=metrics["finite"],
        se3_valid=metrics["se3_valid"],
        per_semantic=metrics["per_semantic"],
        capability_status=dict(targets.capability_status),
        failure=None,
        target_batch=targets,
    )


def build_target_stream_matrix(
    profiles: Mapping[str, Any],
    clips: Iterable[str | Path | ClipInventoryEntry],
    *,
    repo_root: str | Path = ".",
    mode: str = "target_stream_only",
    partial_report_only: bool = True,
) -> TargetStreamMatrix:
    """Build a deterministic target-stream matrix for profiles and clips."""

    rows: list[TargetStreamResult] = []
    for model_id in sorted(profiles):
        profile = profiles[model_id]
        for clip in sorted(clips, key=_clip_sort_key):
            clip_path = clip.path if isinstance(clip, ClipInventoryEntry) else clip
            rows.append(
                generate_target_stream_for_clip(
                    profile,
                    clip_path,
                    repo_root=repo_root,
                    mode=mode,
                    partial_report_only=partial_report_only,
                )
            )
    status = "passed" if all(row.status == "passed" for row in rows) else "blocked"
    return TargetStreamMatrix(schema_version=1, status=status, rows=tuple(rows))


def classify_profile_kind(profile: Any) -> str:
    payload = _profile_payload(profile)
    status = str(payload.get("status", getattr(profile, "status", "")))
    capability_status = str(payload.get("capability_status", getattr(profile, "capability_status", "")))
    if status in PARTIAL_PROFILE_STATUSES or capability_status in PARTIAL_CAPABILITY_STATUSES:
        return "partial_humanoid"
    if status in FULL_PROFILE_STATUSES or capability_status in FULL_CAPABILITY_STATUSES:
        return "full_humanoid"
    return "unsupported_profile"


def semantic_tasks_for_profile(
    profile: Any,
    *,
    repo_root: str | Path = ".",
    profile_kind: str | None = None,
) -> tuple[str, ...]:
    kind = profile_kind or classify_profile_kind(profile)
    if kind == "partial_humanoid":
        return supported_semantics_for_profile(profile, repo_root=repo_root)
    return tuple(REQUIRED_TARGET_SEMANTICS)


def supported_semantics_for_profile(profile: Any, *, repo_root: str | Path = ".") -> tuple[str, ...]:
    """Return explicitly supported semantics without inventing missing ones."""

    payload = _profile_payload(profile)
    candidates: list[Iterable[Any] | None] = [
        payload.get("supported_semantics"),
        _nested(payload, "morphology_classification", "supported_semantics"),
        _nested(payload, "failure_taxonomy", "semantic", "structured_partial", "supported_semantics"),
    ]
    expectation = _load_semantic_expectation(payload, repo_root=Path(repo_root).resolve())
    if expectation:
        candidates.append(expectation.get("supported_semantics"))
    if classify_profile_kind(profile) == "full_humanoid":
        candidates.append(REQUIRED_TARGET_SEMANTICS)

    for candidate in candidates:
        values = _string_sequence(candidate)
        if values:
            return _canonical_semantic_order(values)
    return ()


def target_stream_metrics(targets: RuntimeSemanticTargetBatch) -> dict[str, Any]:
    per_semantic: dict[str, dict[str, Any]] = {}
    finite_all = True
    se3_all = True
    for semantic in targets.semantic_names:
        frames = np.asarray(targets.transforms[semantic], dtype=np.float64)
        finite = bool(np.isfinite(frames).all())
        se3_valid = True
        se3_error = None
        for frame_index, transform in enumerate(frames):
            try:
                validate_se3_transform(transform, context=f"{semantic}[{frame_index}]")
            except ValueError as exc:
                se3_valid = False
                se3_error = str(exc)
                break
        finite_all = finite_all and finite
        se3_all = se3_all and se3_valid
        translations = frames[:, :3, 3] if frames.size else np.empty((0, 3), dtype=np.float64)
        if len(translations) > 1:
            max_translation_step = float(np.linalg.norm(np.diff(translations, axis=0), axis=1).max())
        else:
            max_translation_step = 0.0
        per_semantic[semantic] = {
            "status": "generated",
            "frame_count": int(frames.shape[0]),
            "finite": finite,
            "se3_valid": se3_valid,
            "max_translation_step": max_translation_step,
            "shape": list(frames.shape),
            "error": se3_error,
        }
    return {"finite": finite_all, "se3_valid": se3_all, "per_semantic": per_semantic}


def _partial_supported_semantics_result(
    profile: Any,
    *,
    clip_info: ClipInventoryEntry,
    repo_root: Path,
    frame_budget: FrameBudget | None,
) -> TargetStreamResult:
    payload = _profile_payload(profile)
    model_id = _profile_model_id(payload, profile)
    supported = supported_semantics_for_profile(profile, repo_root=repo_root)
    missing = tuple(name for name in REQUIRED_TARGET_SEMANTICS if name not in supported)
    if not supported:
        return _failed_result(
            model_id=model_id,
            profile_kind="partial_humanoid",
            clip_info=clip_info,
            frame_budget=frame_budget,
            code="partial_supported_semantics_missing",
            message="partial profile does not declare supported semantics",
            semantic_names=(),
            supported_semantics=(),
        )
    per_semantic = {
        semantic: {
            "status": "reported_supported",
            "generated": False,
            "reason": "partial profile reports supported semantics only",
        }
        for semantic in supported
    }
    return TargetStreamResult(
        model_id=model_id,
        profile_kind="partial_humanoid",
        clip_path=clip_info.path,
        status="passed",
        target_stream_status="supported_semantics_reported",
        semantic_names=supported,
        supported_semantics=supported,
        missing_semantics=missing,
        frame_budget=frame_budget,
        frame_count=None if frame_budget is None else frame_budget.selected_frame_count,
        sample_rate=clip_info.sample_rate,
        target_source=None,
        finite=None,
        se3_valid=None,
        per_semantic=per_semantic,
        capability_status={semantic: "reported_supported" for semantic in supported},
        failure=None,
        target_batch=None,
    )


def _failed_result(
    *,
    model_id: str,
    profile_kind: str,
    clip_info: ClipInventoryEntry,
    frame_budget: FrameBudget | None,
    code: str,
    message: str,
    semantic_names: tuple[str, ...],
    supported_semantics: tuple[str, ...],
) -> TargetStreamResult:
    return TargetStreamResult(
        model_id=model_id,
        profile_kind=profile_kind,
        clip_path=clip_info.path,
        status="blocked",
        target_stream_status="failed",
        semantic_names=semantic_names,
        supported_semantics=supported_semantics,
        missing_semantics=tuple(name for name in REQUIRED_TARGET_SEMANTICS if name not in semantic_names),
        frame_budget=frame_budget,
        frame_count=None if frame_budget is None else frame_budget.selected_frame_count,
        sample_rate=clip_info.sample_rate,
        target_source=None,
        finite=False,
        se3_valid=False,
        per_semantic={},
        capability_status={},
        failure={"code": code, "message": message},
        target_batch=None,
    )


def _load_bvh_animation(path: Path):
    from soma_retargeter.assets.bvh import load_bvh

    _skeleton, animation = load_bvh(str(path))
    return animation


def _slice_animation_buffer(animation, frame_budget: FrameBudget) -> AnimationBuffer:
    if not frame_budget.frame_indices:
        raise ValueError("frame budget selected zero frames")
    indices = np.asarray(frame_budget.frame_indices, dtype=np.int64)
    local_transforms = np.asarray(animation.local_transforms)[indices]
    sample_rate = float(animation.sample_rate) / float(max(frame_budget.stride, 1))
    return AnimationBuffer(
        animation.skeleton,
        int(len(indices)),
        sample_rate,
        local_transforms=np.array(local_transforms, copy=True),
    )


def _load_semantic_expectation(payload: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any] | None:
    paths: list[Path] = []
    resolution = payload.get("semantic_map_resolution")
    if isinstance(resolution, Mapping) and resolution.get("path"):
        paths.append(Path(str(resolution["path"])))
    model_id = _profile_model_id(payload, None)
    if model_id != "unknown":
        paths.append(Path("assets/robot_zoo/semantic_expectations") / f"{model_id}.json")
    for path in paths:
        candidate = path if path.is_absolute() else repo_root / path
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return None


def _profile_payload(profile: Any) -> dict[str, Any]:
    if isinstance(profile, dict):
        return profile
    for attr in ("payload", "profile", "data", "raw"):
        value = getattr(profile, attr, None)
        if isinstance(value, dict):
            return value
    if hasattr(profile, "to_json"):
        value = profile.to_json()
        if isinstance(value, dict):
            return value
    return {
        "model": {"id": getattr(profile, "model_id", "unknown")},
        "status": getattr(profile, "status", ""),
        "capability_status": getattr(profile, "capability_status", ""),
    }


def _profile_model_id(payload: Mapping[str, Any], profile: Any | None) -> str:
    model = payload.get("model")
    if isinstance(model, Mapping) and model.get("id"):
        return str(model["id"])
    for key in ("model_id", "id"):
        if payload.get(key):
            return str(payload[key])
    if profile is not None and getattr(profile, "model_id", None):
        return str(getattr(profile, "model_id"))
    return "unknown"


def _resolve_clip_path(path: str | Path, repo_root: Path) -> Path:
    clip_path = Path(path)
    if clip_path.is_absolute():
        return clip_path
    return repo_root / clip_path


def _nested(data: Mapping[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _canonical_semantic_order(values: Iterable[str]) -> tuple[str, ...]:
    unique = {str(value) for value in values}
    ordered = [semantic for semantic in REQUIRED_TARGET_SEMANTICS if semantic in unique]
    ordered.extend(sorted(unique.difference(REQUIRED_TARGET_SEMANTICS)))
    return tuple(ordered)


def _failure_code(exc: Exception) -> str:
    message = str(exc)
    if "missing source semantic" in message or "source semantic batch is missing" in message:
        return "missing_source_semantics"
    if "missing robot semantic target" in message or "lacks capability" in message:
        return "unsupported_semantics"
    if "frame budget selected zero frames" in message:
        return "empty_frame_budget"
    return type(exc).__name__


def _clip_sort_key(clip: str | Path | ClipInventoryEntry) -> str:
    if isinstance(clip, ClipInventoryEntry):
        return clip.path
    return Path(clip).as_posix()
