"""Runtime loader and safety gates for Step 2 V3 profile artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from soma_retargeter.pipelines import utils as pipeline_utils
import soma_retargeter.utils.io_utils as io_utils


DEFAULT_PROFILE_ARTIFACT_ROOT = Path("artifacts/retargeting_v3_step2_capability")
REQUIRED_SEMANTIC_SITES = frozenset({"Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot"})
TERMINAL_PROFILE_STATUSES = frozenset({"passed", "capability_limited_passed"})
OVERRIDE_PROFILE_STATUSES = frozenset({"passed"})
_LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


class RuntimeV3ProfileError(RuntimeError):
    """Structured failure raised before runtime can use an unsafe profile."""

    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


class RuntimeProfileStatusError(RuntimeV3ProfileError):
    """Backward-compatible profile status exception name."""


@dataclass(frozen=True)
class RuntimeModelIdentity:
    path: Path
    fingerprint: str
    source_sha256: str
    model_format: str
    backend: str = "newton"

    def to_json(self) -> dict[str, Any]:
        return {
            "path": _display_path(self.path),
            "fingerprint": self.fingerprint,
            "source_sha256": self.source_sha256,
            "model_format": self.model_format,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class RuntimeV3Profile:
    model_id: str
    schema_version: int
    artifact_path: Path
    status: str
    capability_status: str
    profile_fingerprint: str
    source_sha256: str | None
    semantic_sites: dict[str, dict[str, Any]]
    rest_calibration: dict[str, Any]
    canonical_targets: dict[str, Any]
    canonical_projection_reports: dict[str, Any]
    task_certificate_summary: dict[str, Any]
    model: dict[str, Any]
    runtime_adapter: dict[str, Any]
    deterministic_hash: str | None
    raw: dict[str, Any] = field(repr=False)

    @property
    def payload(self) -> dict[str, Any]:
        return self.raw

    def capability_status_for_semantic(self, semantic_name: str) -> str:
        if semantic_name == "Hips":
            return "exact"
        task_name = {
            "Chest": "torso",
            "LeftHand": "left_hand",
            "RightHand": "right_hand",
            "LeftFoot": "left_foot",
            "RightFoot": "right_foot",
        }.get(semantic_name)
        if task_name is None:
            return "unsupported"
        per_task = self.task_certificate_summary.get("per_task", {})
        task_payload = per_task.get(task_name) if isinstance(per_task, dict) else None
        if not isinstance(task_payload, dict):
            return "unsupported"
        if int(task_payload.get("projection_failure_count", 0)) != 0:
            return "unsupported"
        statuses = {str(value) for value in task_payload.get("statuses", [])}
        if statuses and statuses <= {"converged"}:
            return "exact"
        return "capability_limited"


@dataclass(frozen=True)
class RuntimeProfileResolution:
    robot_type: str
    profile_model_id: str | None
    profile_status: str | None
    profile_artifact_path: str | None
    runtime_mjcf_path: str
    runtime_fingerprint: str | None
    profile_fingerprint: str | None
    runtime_source_sha256: str | None
    profile_source_sha256: str | None
    fingerprint_match: bool | None
    source_hash_match: bool | None
    strict_match_required: bool
    resolution_status: str
    warnings: tuple[dict[str, Any], ...] = ()
    errors: tuple[dict[str, Any], ...] = ()
    override_allowed: bool = False
    shadow_targets_allowed: bool = False
    runtime_local_allowed: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "robot_type": self.robot_type,
            "profile_model_id": self.profile_model_id,
            "profile_status": self.profile_status,
            "profile_artifact_path": self.profile_artifact_path,
            "runtime_mjcf_path": self.runtime_mjcf_path,
            "runtime_fingerprint": self.runtime_fingerprint,
            "profile_fingerprint": self.profile_fingerprint,
            "runtime_source_sha256": self.runtime_source_sha256,
            "profile_source_sha256": self.profile_source_sha256,
            "fingerprint_match": self.fingerprint_match,
            "source_hash_match": self.source_hash_match,
            "strict_match_required": self.strict_match_required,
            "resolution_status": self.resolution_status,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "override_allowed": self.override_allowed,
            "shadow_targets_allowed": self.shadow_targets_allowed,
            "runtime_local_allowed": self.runtime_local_allowed,
        }


@dataclass(frozen=True)
class _RuntimeProfileConfig:
    profile_artifact_root: Path = DEFAULT_PROFILE_ARTIFACT_ROOT
    profile_model_id: str | None = None
    mode: str = "disabled"
    target_policy: str = "shadow_only"
    fail_on_fingerprint_mismatch: bool = True
    allow_runtime_recompile_on_mismatch: bool = False
    allow_runtime_local_profile: bool = False


def load_runtime_v3_profile(
    profile_model_id: Any,
    runtime_mjcf_path: str | Path | None = None,
    config: Mapping[str, Any] | Any | None = None,
    *,
    profile_artifact_root: str | Path | None = None,
) -> RuntimeV3Profile | tuple[RuntimeV3Profile | None, RuntimeProfileResolution]:
    """Load a Step 2 per-robot profile, or resolve+load for a runtime target.

    Direct use:
        ``load_runtime_v3_profile("roboparty_rpo_local", profile_artifact_root=...)``

    Compatibility use for pipeline integration:
        ``load_runtime_v3_profile(target_type, runtime_mjcf_path, config)``
    """

    if runtime_mjcf_path is not None:
        resolution = resolve_runtime_v3_profile_id(profile_model_id, runtime_mjcf_path, config)
        if resolution.errors:
            cfg = _runtime_config(config)
            if cfg.mode == "shadow" and resolution.resolution_status == "fingerprint_mismatch":
                return None, resolution
            first = resolution.errors[0]
            raise RuntimeProfileStatusError(first["code"], first["message"], first.get("details", {}))
        if resolution.profile_model_id is None:
            raise RuntimeProfileStatusError("profile_not_resolved", "profile resolution did not produce a profile id")
        profile = _load_profile_by_id(resolution.profile_model_id, _runtime_config(config).profile_artifact_root)
        return profile, resolution

    root = Path(profile_artifact_root) if profile_artifact_root is not None else DEFAULT_PROFILE_ARTIFACT_ROOT
    return _load_profile_by_id(str(profile_model_id), root)


def resolve_runtime_v3_profile_id(
    target_type: Any,
    runtime_mjcf_path: str | Path,
    config: Mapping[str, Any] | Any | None = None,
) -> RuntimeProfileResolution:
    """Resolve and validate the Step 2 profile for a runtime MJCF model."""

    cfg = _runtime_config(config)
    robot_type = _target_to_str(target_type)
    runtime_path = Path(runtime_mjcf_path)
    profile_model_id = cfg.profile_model_id or pipeline_utils.get_default_runtime_v3_profile_id(robot_type)
    strict_match_required = robot_type == "roboparty_rpo"
    runtime_local_allowed = cfg.allow_runtime_recompile_on_mismatch or cfg.allow_runtime_local_profile

    if profile_model_id is None:
        error = _error_dict(
            "profile_mapping_missing",
            f"no default V3 runtime profile mapping for target {robot_type!r}",
            robot_type=robot_type,
        )
        raise RuntimeV3ProfileError(error["code"], error["message"], error["details"])

    profile = _load_profile_by_id(profile_model_id, cfg.profile_artifact_root)
    runtime_identity = _compute_runtime_model_identity(runtime_path)

    fingerprint_match = runtime_identity.fingerprint == profile.profile_fingerprint
    source_hash_match = (
        None if profile.source_sha256 is None else runtime_identity.source_sha256 == profile.source_sha256
    )
    identity_mismatch = fingerprint_match is False or source_hash_match is False
    profile_path = cfg.profile_artifact_root / "per_robot" / f"{profile_model_id}.json"

    if identity_mismatch:
        error = _identity_mismatch_error(
            robot_type=robot_type,
            profile=profile,
            runtime_identity=runtime_identity,
            fingerprint_match=fingerprint_match,
            source_hash_match=source_hash_match,
            strict_match_required=strict_match_required,
            runtime_local_allowed=runtime_local_allowed,
        )
        if strict_match_required:
            raise RuntimeV3ProfileError(error["code"], error["message"], error["details"])
        if robot_type == "unitree_g1" and runtime_local_allowed:
            warning = _error_dict(
                "runtime_local_profile_required",
                "runtime-local profile generation is required before this G1 runtime model can override targets",
                robot_type=robot_type,
                profile_model_id=profile.model_id,
                runtime_fingerprint=runtime_identity.fingerprint,
                profile_fingerprint=profile.profile_fingerprint,
            )
            return _resolution(
                robot_type=robot_type,
                profile=profile,
                profile_path=profile_path,
                runtime_identity=runtime_identity,
                strict_match_required=strict_match_required,
                resolution_status="runtime_local_profile_required",
                fingerprint_match=fingerprint_match,
                source_hash_match=source_hash_match,
                warnings=[warning],
                override_allowed=False,
                shadow_targets_allowed=False,
                runtime_local_allowed=True,
            )
        if robot_type == "unitree_g1" and cfg.mode == "shadow" and cfg.target_policy == "shadow_only":
            return _resolution(
                robot_type=robot_type,
                profile=profile,
                profile_path=profile_path,
                runtime_identity=runtime_identity,
                strict_match_required=strict_match_required,
                resolution_status="fingerprint_mismatch",
                fingerprint_match=fingerprint_match,
                source_hash_match=source_hash_match,
                errors=[error],
                override_allowed=False,
                shadow_targets_allowed=False,
            )
        raise RuntimeV3ProfileError(error["code"], error["message"], error["details"])

    override_allowed = profile.status in OVERRIDE_PROFILE_STATUSES
    return _resolution(
        robot_type=robot_type,
        profile=profile,
        profile_path=profile_path,
        runtime_identity=runtime_identity,
        strict_match_required=strict_match_required,
        resolution_status="matched",
        fingerprint_match=fingerprint_match,
        source_hash_match=source_hash_match,
        override_allowed=override_allowed,
        shadow_targets_allowed=True,
        runtime_local_allowed=False,
    )


def _load_profile_by_id(profile_model_id: str, profile_artifact_root: Path) -> RuntimeV3Profile:
    path = profile_artifact_root / "per_robot" / f"{profile_model_id}.json"
    payload = _read_json_object(path, role="profile artifact")
    _validate_profile_payload(payload, path)

    model = dict(payload["model"])
    return RuntimeV3Profile(
        model_id=str(model.get("id") or profile_model_id),
        schema_version=int(payload["schema_version"]),
        artifact_path=path,
        status=str(payload["status"]),
        capability_status=str(payload.get("capability_status", "")),
        profile_fingerprint=str(model["fingerprint"]),
        source_sha256=_optional_str(model.get("local_file_sha256")),
        semantic_sites={str(k): dict(v) for k, v in payload["semantic_sites"].items()},
        rest_calibration=dict(payload["rest_calibration"]),
        canonical_targets=dict(payload["canonical_targets"]),
        canonical_projection_reports=dict(payload.get("canonical_projection_reports", {})),
        task_certificate_summary=dict(payload["task_certificate_summary"]),
        model=model,
        runtime_adapter=dict(payload.get("runtime_adapter", {})),
        deterministic_hash=_optional_str(payload.get("deterministic_hash")),
        raw=payload,
    )


def _validate_profile_payload(payload: dict[str, Any], path: Path) -> None:
    schema_version = payload.get("schema_version")
    if schema_version != 3:
        raise RuntimeV3ProfileError(
            "invalid_profile_schema",
            "runtime profile schema_version must be 3",
            {"path": _display_path(path), "schema_version": schema_version},
        )

    status = payload.get("status")
    if status not in TERMINAL_PROFILE_STATUSES:
        raise RuntimeV3ProfileError(
            "invalid_profile_status",
            "runtime profile status is not terminal-pass",
            {
                "path": _display_path(path),
                "status": status,
                "allowed": sorted(TERMINAL_PROFILE_STATUSES),
            },
        )

    for key in ("model", "semantic_sites", "rest_calibration", "canonical_targets", "task_certificate_summary"):
        if not isinstance(payload.get(key), dict):
            raise RuntimeV3ProfileError(
                "invalid_profile_schema",
                f"profile missing required object: {key}",
                {"path": _display_path(path), "field": key},
            )

    model = payload["model"]
    if not isinstance(model.get("fingerprint"), str) or not model["fingerprint"]:
        raise RuntimeV3ProfileError(
            "invalid_profile_schema",
            "profile model fingerprint is missing",
            {"path": _display_path(path), "field": "model.fingerprint"},
        )
    if not model.get("id"):
        raise RuntimeV3ProfileError(
            "invalid_profile_schema",
            "profile model id is missing",
            {"path": _display_path(path), "field": "model.id"},
        )

    semantic_sites = payload["semantic_sites"]
    missing = sorted(REQUIRED_SEMANTIC_SITES - set(semantic_sites))
    if missing:
        raise RuntimeV3ProfileError(
            "invalid_profile_schema",
            "profile missing semantic sites",
            {"path": _display_path(path), "missing": missing},
        )
    for semantic_name in sorted(REQUIRED_SEMANTIC_SITES):
        site = semantic_sites.get(semantic_name)
        if not isinstance(site, dict):
            raise RuntimeV3ProfileError(
                "invalid_profile_schema",
                "profile semantic site is not an object",
                {"path": _display_path(path), "semantic_name": semantic_name},
            )
        for field_name in ("body_name", "local_position", "local_rotation_xyzw"):
            if field_name not in site:
                raise RuntimeV3ProfileError(
                    "invalid_profile_schema",
                    "profile semantic site missing required field",
                    {"path": _display_path(path), "semantic_name": semantic_name, "field": field_name},
                )

    rest = payload["rest_calibration"]
    for field_name in (
        "confidence",
        "root_horizontal_scale",
        "vertical_root_scale",
        "robot_support_height",
        "source_support_height",
    ):
        if field_name not in rest:
            raise RuntimeV3ProfileError(
                "invalid_profile_schema",
                "profile rest calibration missing required field",
                {"path": _display_path(path), "field": f"rest_calibration.{field_name}"},
            )

    canonical_targets = payload["canonical_targets"]
    if "neutral" not in canonical_targets:
        raise RuntimeV3ProfileError(
            "invalid_profile_schema",
            "profile canonical targets missing neutral motion",
            {"path": _display_path(path), "field": "canonical_targets.neutral"},
        )

    task_summary = payload["task_certificate_summary"]
    if task_summary.get("status") != "available":
        raise RuntimeV3ProfileError(
            "invalid_profile_schema",
            "profile capability certificate summary is not available",
            {"path": _display_path(path), "status": task_summary.get("status")},
        )
    if int(task_summary.get("task_count", 0)) <= 0 or int(task_summary.get("canonical_motion_count", 0)) <= 0:
        raise RuntimeV3ProfileError(
            "invalid_profile_schema",
            "profile capability certificate summary is empty",
            {
                "path": _display_path(path),
                "task_count": task_summary.get("task_count"),
                "canonical_motion_count": task_summary.get("canonical_motion_count"),
            },
        )
    if not isinstance(task_summary.get("per_task"), dict) or not task_summary["per_task"]:
        raise RuntimeV3ProfileError(
            "invalid_profile_schema",
            "profile capability certificate summary missing per_task evidence",
            {"path": _display_path(path), "field": "task_certificate_summary.per_task"},
        )


def _compute_runtime_model_identity(runtime_mjcf_path: str | Path) -> RuntimeModelIdentity:
    path = Path(runtime_mjcf_path)
    _require_materialized_file(path, role="runtime MJCF")

    from soma_retargeter.robotics.v3.model_adapter import NewtonRuntimeModelAdapter
    from soma_retargeter.robotics.v3.model_fingerprint import sha256_file

    model_format = _model_format_for_runtime_path(path)
    source_sha256 = sha256_file(path)
    adapter_path = _adapter_input_path(path)
    try:
        adapter = NewtonRuntimeModelAdapter(adapter_path, model_format=model_format)
    except Exception as exc:  # pragma: no cover - exact backend messages are environment-dependent.
        raise RuntimeV3ProfileError(
            "runtime_fingerprint_failed",
            "failed to compute runtime model fingerprint",
            {"path": _display_path(path), "error": f"{type(exc).__name__}: {exc}"},
        ) from exc
    try:
        return RuntimeModelIdentity(
            path=path,
            fingerprint=adapter.fingerprint,
            source_sha256=source_sha256,
            model_format=model_format,
            backend="newton",
        )
    finally:
        adapter.close()


def _runtime_config(config: Mapping[str, Any] | Any | None) -> _RuntimeProfileConfig:
    if config is None:
        payload: Mapping[str, Any] = {}
    elif isinstance(config, Mapping):
        nested = config.get("v3_runtime_profile")
        payload = nested if isinstance(nested, Mapping) else config
    else:
        payload = {
            "profile_artifact_root": getattr(config, "profile_artifact_root", DEFAULT_PROFILE_ARTIFACT_ROOT),
            "profile_model_id": getattr(config, "profile_model_id", None),
            "mode": getattr(config, "mode", "disabled"),
            "target_policy": getattr(config, "target_policy", "shadow_only"),
            "fail_on_fingerprint_mismatch": getattr(config, "fail_on_fingerprint_mismatch", True),
            "allow_runtime_recompile_on_mismatch": getattr(config, "allow_runtime_recompile_on_mismatch", False),
            "allow_runtime_local_profile": getattr(config, "allow_runtime_local_profile", False),
        }

    profile_model_id = payload.get("profile_model_id")
    return _RuntimeProfileConfig(
        profile_artifact_root=Path(payload.get("profile_artifact_root", DEFAULT_PROFILE_ARTIFACT_ROOT)),
        profile_model_id=str(profile_model_id) if profile_model_id else None,
        mode=str(payload.get("mode", "disabled")),
        target_policy=str(payload.get("target_policy", "shadow_only")),
        fail_on_fingerprint_mismatch=bool(payload.get("fail_on_fingerprint_mismatch", True)),
        allow_runtime_recompile_on_mismatch=bool(payload.get("allow_runtime_recompile_on_mismatch", False)),
        allow_runtime_local_profile=bool(payload.get("allow_runtime_local_profile", False)),
    )


def _read_json_object(path: Path, *, role: str) -> dict[str, Any]:
    _require_materialized_file(path, role=role)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeV3ProfileError(
            "invalid_json",
            f"{role} is not valid JSON",
            {"path": _display_path(path), "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeV3ProfileError(
            "invalid_json",
            f"{role} must be a JSON object",
            {"path": _display_path(path), "actual_type": type(payload).__name__},
        )
    return payload


def _require_materialized_file(path: Path, *, role: str) -> None:
    if not path.exists():
        raise RuntimeV3ProfileError(
            "file_missing",
            f"{role} does not exist",
            {"path": _display_path(path)},
        )
    if not path.is_file():
        raise RuntimeV3ProfileError(
            "file_missing",
            f"{role} is not a file",
            {"path": _display_path(path)},
        )
    if _is_lfs_pointer(path):
        raise RuntimeV3ProfileError(
            "lfs_pointer",
            f"{role} is a Git LFS pointer, not materialized content",
            {"path": _display_path(path)},
        )


def _is_lfs_pointer(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    try:
        with p.open("rb") as handle:
            prefix = handle.read(256)
    except OSError:
        return False
    return prefix.decode("utf-8", errors="ignore").startswith(_LFS_POINTER_PREFIX)


def _identity_mismatch_error(
    *,
    robot_type: str,
    profile: RuntimeV3Profile,
    runtime_identity: RuntimeModelIdentity,
    fingerprint_match: bool,
    source_hash_match: bool | None,
    strict_match_required: bool,
    runtime_local_allowed: bool,
) -> dict[str, Any]:
    code = "fingerprint_mismatch" if not fingerprint_match else "source_hash_mismatch"
    message = "runtime model identity does not match the Step 2 V3 profile"
    return _error_dict(
        code,
        message,
        robot_type=robot_type,
        profile_model_id=profile.model_id,
        runtime_fingerprint=runtime_identity.fingerprint,
        profile_fingerprint=profile.profile_fingerprint,
        fingerprint_match=fingerprint_match,
        runtime_source_sha256=runtime_identity.source_sha256,
        profile_source_sha256=profile.source_sha256,
        source_hash_match=source_hash_match,
        strict_match_required=strict_match_required,
        runtime_local_allowed=runtime_local_allowed,
    )


def _resolution(
    *,
    robot_type: str,
    profile: RuntimeV3Profile,
    profile_path: Path,
    runtime_identity: RuntimeModelIdentity,
    strict_match_required: bool,
    resolution_status: str,
    fingerprint_match: bool,
    source_hash_match: bool | None,
    warnings: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
    override_allowed: bool = False,
    shadow_targets_allowed: bool = False,
    runtime_local_allowed: bool = False,
) -> RuntimeProfileResolution:
    return RuntimeProfileResolution(
        robot_type=robot_type,
        profile_model_id=profile.model_id,
        profile_status=profile.status,
        profile_artifact_path=_display_path(profile_path),
        runtime_mjcf_path=_display_path(runtime_identity.path),
        runtime_fingerprint=runtime_identity.fingerprint,
        profile_fingerprint=profile.profile_fingerprint,
        runtime_source_sha256=runtime_identity.source_sha256,
        profile_source_sha256=profile.source_sha256,
        fingerprint_match=fingerprint_match,
        source_hash_match=source_hash_match,
        strict_match_required=strict_match_required,
        resolution_status=resolution_status,
        warnings=tuple(warnings or ()),
        errors=tuple(errors or ()),
        override_allowed=override_allowed,
        shadow_targets_allowed=shadow_targets_allowed,
        runtime_local_allowed=runtime_local_allowed,
    )


def _error_dict(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def _target_to_str(target_type: Any) -> str:
    if isinstance(target_type, str):
        return target_type
    return pipeline_utils.get_target_str_from_type(target_type)


def _model_format_for_runtime_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".urdf":
        return "urdf"
    return "mjcf"


def _adapter_input_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(io_utils.get_repo_root().resolve())
    except ValueError:
        return path


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _display_path(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(io_utils.get_repo_root().resolve()).as_posix()
    except ValueError:
        return f"external:{p.name}"
