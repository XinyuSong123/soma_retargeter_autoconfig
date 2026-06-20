"""Cross-format model conversion and equivalence scaffolding for Step 2."""

from __future__ import annotations

from pathlib import Path
import json

import mujoco
import numpy as np

from .kinematic_paths import discover_paths
from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .model_fingerprint import sha256_file
from .numerical_jacobian import matrix_rank_and_singular_values, numerical_relative_jacobian
from .semantic_sites import build_semantic_sites
from .spatial import rotation_error
from .target_builder import CANONICAL_MOTION_NAMES


DEFAULT_CONVERSION_SETTINGS = {
    "converter": "mujoco.mj_saveLastXML",
    "canonical_format": "mjcf",
    "preserve_runtime_topology": True,
}

CANONICAL_PROJECTION_TARGET_SOURCE = "canonical_semantic_targets"
CANONICAL_PROJECTION_DESIRED_SOURCE = "canonical_targets.transforms"
REQUIRED_CANONICAL_PROJECTION_TASKS = (
    "torso",
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
)


def convert_urdf_to_canonical_mjcf(
    urdf_path: str | Path,
    output_path: str | Path,
    *,
    settings: dict | None = None,
) -> dict:
    """Convert a URDF through MuJoCo's compiled loader to canonical MJCF.

    This is a same-source conversion primitive: the generated MJCF is intended
    for strict equivalence checks against the original URDF loaded by the same
    runtime adapter, not for comparing unrelated vendor and Menagerie variants.
    """

    source = Path(urdf_path)
    output = Path(output_path)
    merged_settings = dict(DEFAULT_CONVERSION_SETTINGS)
    merged_settings.update(settings or {})
    adapter = MuJoCoRuntimeModelAdapter(source, model_format="urdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(output), adapter.model)
    report = {
        "schema_version": 1,
        "source": str(source),
        "output": str(output),
        "settings": merged_settings,
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(output),
        "source_fingerprint": adapter.fingerprint,
        "loader_provenance": adapter.loader_provenance,
        "runtime_signature": runtime_signature(adapter),
    }
    adapter.close()
    return report


def write_conversion_report(report: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def runtime_signature(adapter: MuJoCoRuntimeModelAdapter) -> dict:
    return {
        "backend": adapter.__class__.__name__,
        "format": adapter.model_format,
        "nq": adapter.nq,
        "nv": adapter.nv,
        "body_names": adapter.body_names,
        "coordinates": [coord.to_json() for coord in adapter.coordinate_info],
    }


def compare_runtime_models(
    left: MuJoCoRuntimeModelAdapter,
    right: MuJoCoRuntimeModelAdapter,
    *,
    semantic_map: dict[str, str | dict] | None = None,
    canonical_projection_reports: tuple[dict, dict] | None = None,
    position_atol: float = 1e-7,
    rotation_atol: float = 1e-7,
    rank_singular_value_atol: float = 1e-6,
    projection_atol: float = 1e-9,
) -> dict:
    """Compare same-source runtime kinematics with explicit tolerances."""

    failures: list[str] = []
    left_sig = runtime_signature(left)
    right_sig = runtime_signature(right)
    if left_sig["nq"] != right_sig["nq"] or left_sig["nv"] != right_sig["nv"]:
        failures.append("qpos_or_velocity_dimension_mismatch")
    if left_sig["body_names"] != right_sig["body_names"]:
        failures.append("body_name_order_mismatch")
    left_coords = _coordinate_signature(left)
    right_coords = _coordinate_signature(right)
    coordinate_comparison = _compare_coordinate_signatures(left_coords, right_coords)
    if coordinate_comparison["failures"]:
        failures.append("coordinate_signature_mismatch")

    left_sites: dict[str, SemanticSite] | None = None
    right_sites: dict[str, SemanticSite] | None = None
    semantic_map_evidence = _semantic_map_evidence(semantic_map)
    if semantic_map and semantic_map_evidence["verified"]:
        try:
            left_sites = build_semantic_sites(left, semantic_map)
            right_sites = build_semantic_sites(right, semantic_map)
            semantic_fk = _compare_semantic_fk(
                left,
                right,
                left_sites,
                right_sites,
                position_atol=position_atol,
                rotation_atol=rotation_atol,
            )
        except Exception as exc:
            semantic_fk = _failed_section("semantic_map_build_failed", f"{type(exc).__name__}: {exc}")
    else:
        reason = "semantic_map_not_provided" if not semantic_map else "verified_semantic_map_not_provided"
        semantic_fk = _unavailable_section(reason, semantic_map_evidence=semantic_map_evidence)
    failures.extend(semantic_fk["failures"])

    if left_sites is None or right_sites is None:
        active_chains = _unavailable_section("semantic_sites_unavailable")
        rank_summary = _unavailable_section("semantic_sites_unavailable")
    else:
        active_chains = _compare_active_chains(left, right, left_sites, right_sites)
        failures.extend(active_chains["failures"])
        if active_chains["status"] == "failed":
            rank_summary = _unavailable_section("active_chain_comparison_failed")
        else:
            rank_summary = _compare_rank_summary(
                left,
                right,
                left_sites,
                right_sites,
                singular_value_atol=rank_singular_value_atol,
            )
            failures.extend(rank_summary["failures"])

    canonical_projection = _compare_canonical_projection_reports(
        canonical_projection_reports,
        projection_atol=projection_atol,
    )
    failures.extend(canonical_projection["failures"])
    gate_sections = {
        "semantic_fk": semantic_fk,
        "active_chains": active_chains,
        "rank_summary": rank_summary,
        "canonical_projection": canonical_projection,
    }

    return {
        "schema_version": 2,
        "comparison_mode": "same_source_strict",
        "gate_a_status": _gate_a_status(failures, gate_sections),
        "gate_a_evidence_complete": _gate_a_evidence_complete(gate_sections),
        "gate_a_required_sections": list(gate_sections),
        "strict_equivalent": not failures,
        "failures": failures,
        "left_fingerprint": left.fingerprint,
        "right_fingerprint": right.fingerprint,
        "left_signature": left_sig,
        "right_signature": right_sig,
        "coordinate_comparison": coordinate_comparison,
        "semantic_map_evidence": semantic_map_evidence,
        "semantic_fk": semantic_fk,
        "active_chains": active_chains,
        "rank_summary": rank_summary,
        "canonical_projection": canonical_projection,
        "tolerances": {
            "position_atol": position_atol,
            "rotation_atol": rotation_atol,
            "rank_singular_value_atol": rank_singular_value_atol,
            "projection_atol": projection_atol,
        },
    }


def _coordinate_signature(adapter: MuJoCoRuntimeModelAdapter) -> list[dict]:
    return [
        {
            "label": coord.label,
            "joint_name": coord.joint_name,
            "joint_type": coord.joint_type,
            "limited": coord.limited,
            "lower": _finite_or_none(coord.lower),
            "upper": _finite_or_none(coord.upper),
        }
        for coord in adapter.coordinate_info
    ]


def _compare_coordinate_signatures(left: list[dict], right: list[dict], *, limit_atol: float = 1e-5) -> dict:
    failures: list[str] = []
    if len(left) != len(right):
        failures.append("coordinate_count_mismatch")
        return {"passed": False, "failures": failures, "limit_atol": limit_atol}
    for index, (l_coord, r_coord) in enumerate(zip(left, right)):
        for key in ("label", "joint_name", "joint_type", "limited"):
            if l_coord[key] != r_coord[key]:
                failures.append(f"{index}:{key}_mismatch")
        for key in ("lower", "upper"):
            l_value = l_coord[key]
            r_value = r_coord[key]
            if l_value is None or r_value is None:
                if l_value != r_value:
                    failures.append(f"{index}:{key}_finite_mismatch")
            elif abs(float(l_value) - float(r_value)) > limit_atol:
                failures.append(f"{index}:{key}_limit_mismatch")
    return {"passed": not failures, "failures": failures, "limit_atol": limit_atol}


def _compare_semantic_fk(
    left: MuJoCoRuntimeModelAdapter,
    right: MuJoCoRuntimeModelAdapter,
    left_sites: dict[str, SemanticSite],
    right_sites: dict[str, SemanticSite],
    *,
    position_atol: float,
    rotation_atol: float,
) -> dict:
    left_state = left.forward_kinematics(left.neutral_q())
    right_state = right.forward_kinematics(right.neutral_q())
    per_site = {}
    failures: list[str] = []
    for name in sorted(set(left_sites) & set(right_sites)):
        left_t = left.site_transform(left_state, left_sites[name])
        right_t = right.site_transform(right_state, right_sites[name])
        position_error = float(np.linalg.norm(left_t[:3, 3] - right_t[:3, 3]))
        rot_error = rotation_error(left_t[:3, :3], right_t[:3, :3])
        per_site[name] = {
            "position_error": position_error,
            "rotation_error": rot_error,
            "passed": position_error <= position_atol and rot_error <= rotation_atol,
        }
        if not per_site[name]["passed"]:
            failures.append(f"semantic_fk_mismatch:{name}")
    if not per_site:
        return _unavailable_section("no_common_semantic_sites", per_site=per_site)
    return {"status": "passed" if not failures else "failed", "passed": not failures, "per_site": per_site, "failures": failures}


def _compare_active_chains(
    left: MuJoCoRuntimeModelAdapter,
    right: MuJoCoRuntimeModelAdapter,
    left_sites: dict[str, SemanticSite],
    right_sites: dict[str, SemanticSite],
) -> dict:
    left_paths = discover_paths(left, left_sites)
    right_paths = discover_paths(right, right_sites)
    per_task = {}
    failures: list[str] = []
    for task in sorted(set(left_paths) | set(right_paths)):
        if task not in left_paths or task not in right_paths:
            failures.append(f"active_chain_missing:{task}")
            per_task[task] = {
                "passed": False,
                "left": left_paths.get(task).to_json() if task in left_paths else None,
                "right": right_paths.get(task).to_json() if task in right_paths else None,
                "failures": [f"missing_{'left' if task not in left_paths else 'right'}"],
            }
            continue
        left_payload = _chain_signature(left_paths[task].to_json())
        right_payload = _chain_signature(right_paths[task].to_json())
        task_failures = [
            key
            for key in sorted(set(left_payload) | set(right_payload))
            if left_payload.get(key) != right_payload.get(key)
        ]
        if task_failures:
            failures.extend(f"active_chain_mismatch:{task}:{key}" for key in task_failures)
        per_task[task] = {
            "passed": not task_failures,
            "left": left_payload,
            "right": right_payload,
            "failures": task_failures,
        }
    if not per_task:
        return _unavailable_section("no_comparable_semantic_paths", per_task=per_task)
    return {"status": "passed" if not failures else "failed", "passed": not failures, "per_task": per_task, "failures": failures}


def _chain_signature(path_payload: dict) -> dict:
    keys = (
        "reference",
        "target",
        "reference_body",
        "target_body",
        "lca_body",
        "body_path",
        "reference_branch_bodies",
        "target_branch_bodies",
        "active_velocity_coordinates",
        "coordinate_labels",
        "joint_types",
    )
    return {key: path_payload.get(key) for key in keys}


def _compare_rank_summary(
    left: MuJoCoRuntimeModelAdapter,
    right: MuJoCoRuntimeModelAdapter,
    left_sites: dict[str, SemanticSite],
    right_sites: dict[str, SemanticSite],
    *,
    singular_value_atol: float,
) -> dict:
    left_paths = discover_paths(left, left_sites)
    right_paths = discover_paths(right, right_sites)
    common_tasks = sorted(set(left_paths) & set(right_paths))
    if not common_tasks:
        return _unavailable_section("no_common_semantic_paths", per_task={})
    per_task = {}
    failures: list[str] = []
    for task in common_tasks:
        try:
            left_rank = _neutral_rank_summary(left, left_sites, left_paths[task])
            right_rank = _neutral_rank_summary(right, right_sites, right_paths[task])
        except Exception as exc:
            failures.append(f"rank_summary_failed:{task}:{type(exc).__name__}")
            per_task[task] = {
                "passed": False,
                "left": None,
                "right": None,
                "failures": [f"{type(exc).__name__}: {exc}"],
            }
            continue
        task_failures = _rank_summary_failures(task, left_rank, right_rank, singular_value_atol=singular_value_atol)
        failures.extend(task_failures)
        per_task[task] = {
            "passed": not task_failures,
            "left": left_rank,
            "right": right_rank,
            "failures": task_failures,
        }
    return {"status": "passed" if not failures else "failed", "passed": not failures, "per_task": per_task, "failures": failures}


def _neutral_rank_summary(adapter: MuJoCoRuntimeModelAdapter, sites: dict[str, SemanticSite], path) -> dict:
    jac = numerical_relative_jacobian(
        adapter,
        adapter.neutral_q(),
        sites[path.reference],
        sites[path.target],
        path.active_velocity_coordinates,
    )
    translation_rank, translation_singular_values = matrix_rank_and_singular_values(jac.translation)
    rotation_rank, rotation_singular_values = matrix_rank_and_singular_values(jac.rotation)
    return {
        "active_coordinate_count": len(path.active_velocity_coordinates),
        "translation_rank": translation_rank,
        "rotation_rank": rotation_rank,
        "translation_singular_values": translation_singular_values.tolist(),
        "rotation_singular_values": rotation_singular_values.tolist(),
        "stability_gate_passed": jac.stability_gate_passed,
        "unstable_columns": jac.unstable_columns,
    }


def _rank_summary_failures(task: str, left: dict, right: dict, *, singular_value_atol: float) -> list[str]:
    failures: list[str] = []
    for key in ("active_coordinate_count", "translation_rank", "rotation_rank", "stability_gate_passed", "unstable_columns"):
        if left[key] != right[key]:
            failures.append(f"rank_summary_mismatch:{task}:{key}")
    for key in ("translation_singular_values", "rotation_singular_values"):
        left_values = np.asarray(left[key], dtype=float)
        right_values = np.asarray(right[key], dtype=float)
        if left_values.shape != right_values.shape or np.max(np.abs(left_values - right_values), initial=0.0) > singular_value_atol:
            failures.append(f"rank_summary_mismatch:{task}:{key}")
    return failures


def _compare_canonical_projection_reports(
    reports: tuple[dict, dict] | None,
    *,
    projection_atol: float,
) -> dict:
    if reports is None:
        return _unavailable_section("canonical_projection_reports_not_provided")
    left_report, right_report = reports
    left_evidence = _canonical_projection_evidence(left_report)
    right_evidence = _canonical_projection_evidence(right_report)
    if left_evidence["status"] != "passed" or right_evidence["status"] != "passed":
        reasons = []
        reasons.extend(f"left:{reason}" for reason in left_evidence["reasons"])
        reasons.extend(f"right:{reason}" for reason in right_evidence["reasons"])
        return _unavailable_section(
            _canonical_projection_unavailable_reason(left_evidence, right_evidence),
            evidence={
                "left": left_evidence,
                "right": right_evidence,
            },
            incomplete_reasons=reasons,
        )
    left_summary = _canonical_projection_summary(left_report)
    right_summary = _canonical_projection_summary(right_report)
    if not left_summary.get("motions") and not right_summary.get("motions"):
        return _unavailable_section(
            "canonical_projection_reports_empty",
            left=left_summary,
            right=right_summary,
        )
    mismatch_paths: list[str] = []
    _compare_projection_values(left_summary, right_summary, path="", mismatch_paths=mismatch_paths, atol=projection_atol)
    failures = [f"canonical_projection_mismatch:{path}" for path in mismatch_paths]
    return {
        "status": "passed" if not failures else "failed",
        "passed": not failures,
        "evidence": {
            "left": left_evidence,
            "right": right_evidence,
        },
        "left": left_summary,
        "right": right_summary,
        "failures": failures,
    }


def _canonical_projection_summary(report: dict) -> dict:
    motions = report.get("motions", {}) if isinstance(report, dict) else {}
    return {
        "motion_order": report.get("motion_order", []) if isinstance(report, dict) else [],
        "target_source": report.get("target_source") if isinstance(report, dict) else None,
        "failures": report.get("failures", []) if isinstance(report, dict) else [],
        "unreachable_demands": report.get("unreachable_demands", []) if isinstance(report, dict) else [],
        "motions": {
            motion_name: {
                "tasks": {
                    task_name: {
                        key: task_payload.get(key)
                        for key in (
                            "status",
                            "converged",
                            "residual",
                            "normalized_residual",
                            "active_coordinates",
                            "desired_source",
                            "reference",
                            "target",
                        )
                        if key in task_payload
                    }
                    for task_name, task_payload in sorted(motion_payload.get("tasks", {}).items())
                }
            }
            for motion_name, motion_payload in sorted(motions.items())
        },
    }


def _compare_projection_values(first, second, *, path: str, mismatch_paths: list[str], atol: float) -> None:
    if isinstance(first, bool) or isinstance(second, bool):
        if first != second:
            mismatch_paths.append(path or "$")
        return
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        if abs(float(first) - float(second)) > atol:
            mismatch_paths.append(path or "$")
        return
    if isinstance(first, dict) and isinstance(second, dict):
        for key in sorted(set(first) | set(second)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in first or key not in second:
                mismatch_paths.append(child_path)
                continue
            _compare_projection_values(first[key], second[key], path=child_path, mismatch_paths=mismatch_paths, atol=atol)
        return
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            mismatch_paths.append(f"{path}.length" if path else "length")
            return
        for index, (left_item, right_item) in enumerate(zip(first, second)):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            _compare_projection_values(left_item, right_item, path=child_path, mismatch_paths=mismatch_paths, atol=atol)
        return
    if first != second:
        mismatch_paths.append(path or "$")


def _semantic_map_evidence(semantic_map: dict[str, str | dict] | None) -> dict:
    if not semantic_map:
        return {
            "verified": False,
            "semantic_count": 0,
            "verified_semantics": [],
            "unverified_semantics": [],
            "reason": "semantic_map_not_provided",
        }
    verified: list[str] = []
    unverified: list[str] = []
    for semantic, entry in sorted(semantic_map.items()):
        if isinstance(entry, dict) and str(entry.get("source", "")).startswith("verified"):
            verified.append(semantic)
        else:
            unverified.append(semantic)
    return {
        "verified": bool(verified) and not unverified,
        "semantic_count": len(semantic_map),
        "verified_semantics": verified,
        "unverified_semantics": unverified,
        "reason": "verified_semantic_map" if verified and not unverified else "unverified_semantic_entries",
    }


def _canonical_projection_evidence(report: dict) -> dict:
    reasons: list[str] = []
    if not isinstance(report, dict):
        return {
            "status": "failed",
            "reasons": ["canonical_projection_report_not_a_dict"],
            "motion_count": 0,
            "task_coverage": [],
            "target_source": None,
        }

    target_source = report.get("target_source")
    if target_source != CANONICAL_PROJECTION_TARGET_SOURCE:
        reasons.append(f"target_source_not_canonical:{target_source!r}")

    motion_order = [str(item) for item in report.get("motion_order", []) if isinstance(item, str)]
    motions = report.get("motions", {})
    if not isinstance(motions, dict):
        motions = {}
    motion_names = list(motion_order) or sorted(str(name) for name in motions)
    if set(motion_names) == {"neutral"}:
        reasons.append("canonical_projection_reports_neutral_only")
    missing_motions = [name for name in CANONICAL_MOTION_NAMES if name not in set(motion_names)]
    if missing_motions:
        reasons.append("canonical_motion_coverage_insufficient")

    task_coverage: set[str] = set()
    bad_desired_sources: list[str] = []
    missing_task_motions: list[str] = []
    for motion_name in motion_names:
        motion_payload = motions.get(motion_name, {})
        tasks = motion_payload.get("tasks", {}) if isinstance(motion_payload, dict) else {}
        if not isinstance(tasks, dict):
            tasks = {}
        motion_task_names = {str(task_name) for task_name in tasks}
        task_coverage.update(motion_task_names)
        missing_for_motion = [task for task in REQUIRED_CANONICAL_PROJECTION_TASKS if task not in motion_task_names]
        if missing_for_motion:
            missing_task_motions.append(motion_name)
        for task_name, task_payload in sorted(tasks.items()):
            if not isinstance(task_payload, dict):
                bad_desired_sources.append(f"{motion_name}:{task_name}:missing_payload")
                continue
            desired_source = task_payload.get("desired_source")
            if desired_source != CANONICAL_PROJECTION_DESIRED_SOURCE:
                bad_desired_sources.append(f"{motion_name}:{task_name}:{desired_source!r}")
    missing_tasks = [task for task in REQUIRED_CANONICAL_PROJECTION_TASKS if task not in task_coverage]
    if missing_tasks:
        reasons.append("canonical_task_coverage_insufficient")
    if missing_task_motions:
        reasons.append("canonical_task_coverage_incomplete_per_motion")
    if bad_desired_sources:
        reasons.append("canonical_desired_source_not_real")

    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "motion_count": len(set(motion_names)),
        "required_motion_count": len(CANONICAL_MOTION_NAMES),
        "missing_motions": missing_motions,
        "task_coverage": sorted(task_coverage),
        "required_tasks": list(REQUIRED_CANONICAL_PROJECTION_TASKS),
        "missing_tasks": missing_tasks,
        "motions_missing_required_tasks": missing_task_motions,
        "bad_desired_sources": bad_desired_sources,
        "target_source": target_source,
    }


def _canonical_projection_unavailable_reason(left_evidence: dict, right_evidence: dict) -> str:
    reasons = set(left_evidence.get("reasons", [])) | set(right_evidence.get("reasons", []))
    if "canonical_projection_reports_neutral_only" in reasons:
        return "canonical_projection_reports_neutral_only"
    if any(str(reason).startswith("target_source_not_canonical") for reason in reasons):
        return "canonical_projection_target_source_not_canonical"
    if "canonical_desired_source_not_real" in reasons:
        return "canonical_projection_desired_source_not_real"
    if "canonical_motion_coverage_insufficient" in reasons:
        return "canonical_projection_motion_coverage_insufficient"
    if "canonical_task_coverage_insufficient" in reasons or "canonical_task_coverage_incomplete_per_motion" in reasons:
        return "canonical_projection_task_coverage_insufficient"
    return "canonical_projection_evidence_incomplete"


def _gate_a_status(failures: list[str], sections: dict[str, dict]) -> str:
    if failures:
        return "failed"
    if _gate_a_evidence_complete(sections):
        return "complete_passed"
    return "incomplete"


def _gate_a_evidence_complete(sections: dict[str, dict]) -> bool:
    return all(section.get("status") == "passed" for section in sections.values())


def _unavailable_section(reason: str, **extra) -> dict:
    payload = {"status": "unavailable", "passed": False, "reason": reason, "failures": []}
    payload.update(extra)
    return payload


def _failed_section(reason: str, detail: str) -> dict:
    failure = f"{reason}:{detail}"
    return {"status": "failed", "passed": False, "reason": reason, "detail": detail, "failures": [failure]}


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
