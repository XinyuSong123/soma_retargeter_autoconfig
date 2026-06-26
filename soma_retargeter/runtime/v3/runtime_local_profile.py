"""Runtime-local profile closure for Step 3.1 full-fleet evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from soma_retargeter.robotics.v3.model_adapter import NewtonRuntimeModelAdapter
from soma_retargeter.robotics.v3.model_fingerprint import sha256_file
from soma_retargeter.robotics.v3.profile import compile_kinematic_profile_v3, write_profile

from .fleet_inventory import (
    FULL_HUMANOID_PROFILE,
    NEGATIVE_CONTROL,
    PARTIAL_HUMANOID_PROFILE,
    FleetRuntimeCase,
    display_path,
    stable_payload_hash,
    write_json,
)


@dataclass(frozen=True)
class RuntimeProfileClosure:
    model_id: str
    category: str
    profile_path: Path
    runtime_source_path: Path
    runtime_fingerprint: str | None
    profile_fingerprint: str | None
    runtime_source_sha256: str | None
    profile_source_sha256: str | None
    fingerprint_match: bool | None
    source_hash_match: bool | None
    resolution_status: str
    runtime_local_profile_path: Path | None
    profile_status: str
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "category": self.category,
            "profile_path": display_path(self.profile_path),
            "runtime_source_path": display_path(self.runtime_source_path),
            "runtime_fingerprint": self.runtime_fingerprint,
            "profile_fingerprint": self.profile_fingerprint,
            "runtime_source_sha256": self.runtime_source_sha256,
            "profile_source_sha256": self.profile_source_sha256,
            "fingerprint_match": self.fingerprint_match,
            "source_hash_match": self.source_hash_match,
            "resolution_status": self.resolution_status,
            "runtime_local_profile_path": display_path(self.runtime_local_profile_path),
            "profile_status": self.profile_status,
            "error": self.error,
        }


def close_runtime_profile(
    case: FleetRuntimeCase,
    *,
    artifact_root: str | Path,
    low_discrepancy_count: int = 8,
) -> RuntimeProfileClosure:
    """Resolve or generate the runtime-local profile needed by one fleet row."""

    artifact_root = Path(artifact_root)
    profile = case.profile
    model = profile.get("model", {}) if isinstance(profile.get("model"), dict) else {}
    profile_fingerprint = _optional_str(model.get("fingerprint"))
    profile_source_sha = _optional_str(model.get("local_file_sha256"))

    identity = _runtime_identity(case)
    runtime_fingerprint = identity.get("fingerprint")
    runtime_source_sha = identity.get("source_sha256")
    error = identity.get("error")
    if error:
        return RuntimeProfileClosure(
            model_id=case.model_id,
            category=case.category,
            profile_path=case.profile_path,
            runtime_source_path=case.runtime_source_path,
            runtime_fingerprint=runtime_fingerprint,
            profile_fingerprint=profile_fingerprint,
            runtime_source_sha256=runtime_source_sha,
            profile_source_sha256=profile_source_sha,
            fingerprint_match=None if profile_fingerprint is None else runtime_fingerprint == profile_fingerprint,
            source_hash_match=None if profile_source_sha is None else runtime_source_sha == profile_source_sha,
            resolution_status="runtime_model_load_failed",
            runtime_local_profile_path=None,
            profile_status=case.profile_status,
            error=error,
        )

    fingerprint_match = None if profile_fingerprint is None else runtime_fingerprint == profile_fingerprint
    source_hash_match = None if profile_source_sha is None else runtime_source_sha == profile_source_sha

    if case.category == NEGATIVE_CONTROL:
        return RuntimeProfileClosure(
            model_id=case.model_id,
            category=case.category,
            profile_path=case.profile_path,
            runtime_source_path=case.runtime_source_path,
            runtime_fingerprint=runtime_fingerprint,
            profile_fingerprint=profile_fingerprint,
            runtime_source_sha256=runtime_source_sha,
            profile_source_sha256=profile_source_sha,
            fingerprint_match=fingerprint_match,
            source_hash_match=source_hash_match,
            resolution_status="negative_control_rejected",
            runtime_local_profile_path=None,
            profile_status=case.profile_status,
        )

    if fingerprint_match is not False and source_hash_match is not False:
        return RuntimeProfileClosure(
            model_id=case.model_id,
            category=case.category,
            profile_path=case.profile_path,
            runtime_source_path=case.runtime_source_path,
            runtime_fingerprint=runtime_fingerprint,
            profile_fingerprint=profile_fingerprint,
            runtime_source_sha256=runtime_source_sha,
            profile_source_sha256=profile_source_sha,
            fingerprint_match=fingerprint_match,
            source_hash_match=source_hash_match,
            resolution_status="profile_match" if case.category == FULL_HUMANOID_PROFILE else "structured_partial_supported",
            runtime_local_profile_path=None,
            profile_status=case.profile_status,
        )

    local_path = artifact_root / "runtime_local_profiles" / f"{case.model_id}.json"
    if case.category == PARTIAL_HUMANOID_PROFILE:
        _write_structured_partial_runtime_local(case, local_path, identity)
        status = "structured_partial_supported"
    else:
        status = _compile_runtime_local_full_profile(
            case,
            local_path,
            identity=identity,
            low_discrepancy_count=low_discrepancy_count,
        )
    return RuntimeProfileClosure(
        model_id=case.model_id,
        category=case.category,
        profile_path=case.profile_path,
        runtime_source_path=case.runtime_source_path,
        runtime_fingerprint=runtime_fingerprint,
        profile_fingerprint=profile_fingerprint,
        runtime_source_sha256=runtime_source_sha,
        profile_source_sha256=profile_source_sha,
        fingerprint_match=fingerprint_match,
        source_hash_match=source_hash_match,
        resolution_status=status,
        runtime_local_profile_path=local_path,
        profile_status=case.profile_status,
    )


def write_profile_resolution_artifacts(
    *,
    artifact_root: str | Path,
    closures: list[RuntimeProfileClosure],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_root = Path(artifact_root)
    rows = [closure.to_json() for closure in closures]
    matrix = {"schema_version": 1, "rows": rows, "row_count": len(rows)}
    summary = _summary(rows)
    write_json(artifact_root / "profile_resolution_matrix.json", matrix)
    write_json(artifact_root / "runtime_local_profile_summary.json", summary)
    for closure in closures:
        per_model = artifact_root / "per_model" / closure.model_id / "profile_resolution.json"
        write_json(per_model, closure.to_json())
    return matrix, summary


def _runtime_identity(case: FleetRuntimeCase) -> dict[str, Any]:
    try:
        adapter = NewtonRuntimeModelAdapter(case.runtime_source_path, model_format=case.model_format)
    except Exception as exc:
        return {
            "source_sha256": _safe_sha256(case.runtime_source_path),
            "fingerprint": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        return {
            "source_sha256": sha256_file(case.runtime_source_path),
            "fingerprint": adapter.fingerprint,
            "nq": adapter.nq,
            "nv": adapter.nv,
            "body_count": len(adapter.body_names),
            "error": None,
        }
    finally:
        adapter.close()


def _compile_runtime_local_full_profile(
    case: FleetRuntimeCase,
    output_path: Path,
    *,
    identity: dict[str, Any],
    low_discrepancy_count: int,
) -> str:
    if case.semantic_map_path is None:
        _write_failed_runtime_local_profile(case, output_path, "semantic_map_unavailable", identity)
        return "runtime_local_profile_failed"
    try:
        semantic_map = json.loads(case.semantic_map_path.read_text(encoding="utf-8"))
        profile = compile_kinematic_profile_v3(
            case.runtime_source_path,
            semantic_map,
            model_id=case.model_id,
            model_format=case.model_format,
            backend="newton",
            low_discrepancy_count=low_discrepancy_count,
            reproduction_command=(
                "PYTHONPATH=. python -m soma_retargeter.tools.compile_runtime_local_profiles_v3 "
                f"--model-id {case.model_id}"
            ),
        )
        write_profile(profile, output_path)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload.setdefault("model", {})["local_file_sha256"] = identity.get("source_sha256")
        payload["runtime_local_profile"] = _metadata(case, identity, "runtime_local_profile_generated")
        payload["status"] = "passed" if payload.get("capability_status") == "full_humanoid_ready" and not payload.get("failures") else payload.get("status", "passed")
        payload["deterministic_hash"] = stable_payload_hash({k: v for k, v in payload.items() if k != "deterministic_hash"})
        write_json(output_path, payload)
    except Exception as exc:
        _write_failed_runtime_local_profile(case, output_path, f"{type(exc).__name__}: {exc}", identity)
        return "runtime_local_profile_failed"
    return "runtime_local_profile_generated"


def _write_structured_partial_runtime_local(
    case: FleetRuntimeCase,
    output_path: Path,
    identity: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "model_id": case.model_id,
        "status": "partial_passed",
        "capability_status": "partial_humanoid",
        "supported_semantics": list(case.supported_semantics),
        "missing_required_semantics": list(case.missing_required_semantics),
        "runtime_local_profile": _metadata(case, identity, "structured_partial_supported"),
        "humanoid_profile_generated": False,
    }
    payload["deterministic_hash"] = stable_payload_hash(payload)
    write_json(output_path, payload)


def _write_failed_runtime_local_profile(
    case: FleetRuntimeCase,
    output_path: Path,
    reason: str,
    identity: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "model_id": case.model_id,
        "status": "runtime_local_profile_failed",
        "reason": str(reason),
        "runtime_local_profile": _metadata(case, identity, "runtime_local_profile_failed"),
    }
    payload["deterministic_hash"] = stable_payload_hash(payload)
    write_json(output_path, payload)


def _metadata(case: FleetRuntimeCase, identity: dict[str, Any], status: str) -> dict[str, Any]:
    semantic_map_hash = _safe_sha256(case.semantic_map_path) if case.semantic_map_path else None
    semantic_expectation_hash = _safe_sha256(case.semantic_expectation_path) if case.semantic_expectation_path else None
    return {
        "schema_version": 1,
        "status": status,
        "source_profile_id": case.model_id,
        "source_profile_path": display_path(case.profile_path),
        "runtime_source_path": display_path(case.runtime_source_path),
        "runtime_source_sha256": identity.get("source_sha256"),
        "runtime_fingerprint": identity.get("fingerprint"),
        "semantic_map_path": display_path(case.semantic_map_path),
        "semantic_map_sha256": semantic_map_hash,
        "semantic_expectation_path": display_path(case.semantic_expectation_path),
        "semantic_expectation_sha256": semantic_expectation_hash,
        "model_format": case.model_format,
        "category": case.category,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["resolution_status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": 1,
        "row_count": len(rows),
        "status_counts": counts,
        "profile_match_count": counts.get("profile_match", 0),
        "runtime_local_profile_generated_count": counts.get("runtime_local_profile_generated", 0),
        "runtime_local_profile_failed_count": counts.get("runtime_local_profile_failed", 0),
        "structured_partial_supported_count": counts.get("structured_partial_supported", 0),
        "negative_control_rejected_count": counts.get("negative_control_rejected", 0),
        "runtime_model_load_failed_count": counts.get("runtime_model_load_failed", 0),
    }


def _safe_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    try:
        return sha256_file(path)
    except Exception:
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
