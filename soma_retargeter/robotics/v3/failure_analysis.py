"""Step 2.3 failure ledger extraction for capability baseline reports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import re
from typing import Any


DEFAULT_FAILURE_REPORT_DIR = Path("artifacts/retargeting_v3_step2_assets44/failures")
DEFAULT_BASELINE_LEDGER_PATH = Path("artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json")

BASELINE_ROBOT_IDS = (
    "atlas_drc_urdf",
    "atlas_v4_urdf",
    "booster_t1_mjcf",
    "booster_t1_urdf",
    "fourier_n1_mjcf",
    "jaxon_urdf",
    "mujoco_humanoid_mjcf",
    "pal_talos_mjcf_direct",
    "robotis_op3_mjcf",
    "talos_urdf",
    "valkyrie_urdf",
)

PROJECTION_POSITION_METRIC = "projection_position_normalized_residual"
PROJECTION_ROTATION_METRIC = "projection_rotation_normalized_residual"
NUMERICAL_STABILITY_METRIC = "numerical_stability_gate"

_PROJECTION_FAILURE_RE = re.compile(
    r"^projection residual gate failed: "
    r"(?P<motion>[^.]+)\.(?P<task>\S+) "
    r"normalized_residual=(?P<normalized_residual>\S+) "
    r"threshold=(?P<threshold>\S+) "
    r"residual=(?P<residual>\S+)$"
)
_PROJECTION_NONFINITE_RE = re.compile(
    r"^projection residual gate failed: (?P<motion>[^.]+)\.(?P<task>\S+) normalized_residual nonfinite$"
)
_NUMERICAL_FAILURE_RE = re.compile(
    r"^numerical stability gate failed: (?P<task>\S+) has numerical_stability_gate_passed=false$"
)

_RANK_FIELDS = (
    "task_block",
    "numerical_stability_gate_passed",
    "stable_sample_fraction",
    "stable_sample_fraction_threshold",
    "regular_rank_fraction_threshold",
    "samples",
    "rank_method",
    "engine_rank_translation",
    "fd_rank_translation",
    "engine_rank_rotation",
    "fd_rank_rotation",
    "regular_rank_translation",
    "nominal_rank_translation",
    "regular_rank_rotation",
    "nominal_rank_rotation",
    "rank_agreement_rate_translation",
    "rank_agreement_rate_rotation",
    "relevant_rank_agreement_rate",
    "projector_distance_p95",
    "principal_angle_p95",
    "engine_fd_normalized_error_p95",
    "singularity_fraction_translation",
    "singularity_fraction_rotation",
    "epsilon_stability_gate_passed",
    "epsilon_unstable_fraction",
    "epsilon_unstable_sample_fraction",
    "epsilon_unstable_columns",
    "conditioning_percentiles_translation",
    "conditioning_percentiles_rotation",
    "selected_singular_values_translation",
    "selected_singular_values_rotation",
)

_NUMERICAL_STABILITY_THRESHOLDS = {
    "stable_sample_fraction_min": 0.95,
    "engine_available_rate_min": 1.0,
    "relevant_rank_agreement_rate_min": 0.95,
    "projector_distance_p95_max": 0.05,
    "engine_fd_normalized_error_p95_max": 0.02,
}


def build_baseline_failure_ledger(failure_dir: str | Path = DEFAULT_FAILURE_REPORT_DIR) -> dict[str, Any]:
    """Build the frozen Step 2.3 capability failure ledger from report JSON."""

    root = Path(failure_dir)
    reports = _load_frozen_reports(root)
    source_reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for robot_id in BASELINE_ROBOT_IDS:
        path, report = reports[robot_id]
        semantic_map = _load_semantic_map(report)
        report_sha = _sha256_file(path)
        failures = list(report.get("failures") or [])
        source_reports.append(
            {
                "robot_id": robot_id,
                "path": _display_path(path),
                "sha256": report_sha,
                "status": report.get("status"),
                "deterministic_hash": report.get("deterministic_hash"),
                "failures": failures,
            }
        )
        for failure_index, message in enumerate(failures):
            rows.append(
                _failure_row(
                    robot_id=robot_id,
                    failure_index=failure_index,
                    message=str(message),
                    report=report,
                    report_path=path,
                    report_sha256=report_sha,
                    semantic_map=semantic_map,
                )
            )

    robot_counts = Counter(row["robot_id"] for row in rows)
    metric_counts = Counter(row["metric_type"] for row in rows)
    return {
        "schema_version": 1,
        "ledger_type": "retargeting_v3_step2_3_capability_baseline_failure_ledger",
        "source_report_directory": _display_path(root),
        "frozen_robot_ids": list(BASELINE_ROBOT_IDS),
        "counts": {
            "robots": len(BASELINE_ROBOT_IDS),
            "failed_rows": len(rows),
            "failure_rows_by_robot": {robot_id: robot_counts[robot_id] for robot_id in BASELINE_ROBOT_IDS},
            "failure_rows_by_metric_type": {metric: metric_counts[metric] for metric in sorted(metric_counts)},
        },
        "thresholds": {
            "projection_quality": {
                "neutral_normalized_residual": 0.001,
                "hand_rho_p": 0.12,
                "foot_rho_p": 0.06,
                "torso_rho_r": 0.08,
                "default_normalized_residual": 0.05,
                "source": "soma_retargeter.robotics.v3.profile._projection_quality_threshold",
            },
            "numerical_stability": {
                "criteria": dict(_NUMERICAL_STABILITY_THRESHOLDS),
                "source": "soma_retargeter.robotics.v3.reachability.analyze_reachability",
            },
        },
        "source_reports": source_reports,
        "failures": rows,
    }


def write_baseline_failure_ledger(
    output_path: str | Path = DEFAULT_BASELINE_LEDGER_PATH,
    *,
    failure_dir: str | Path = DEFAULT_FAILURE_REPORT_DIR,
) -> dict[str, Any]:
    ledger = build_baseline_failure_ledger(failure_dir)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return ledger


def _load_frozen_reports(root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"failure report directory missing: {root}")
    paths = {path.stem: path for path in sorted(root.glob("*.json"))}
    found = tuple(sorted(paths))
    if found != tuple(sorted(BASELINE_ROBOT_IDS)):
        missing = sorted(set(BASELINE_ROBOT_IDS) - set(found))
        extra = sorted(set(found) - set(BASELINE_ROBOT_IDS))
        raise ValueError(f"failure report IDs must match frozen baseline; missing={missing} extra={extra}")
    reports: dict[str, tuple[Path, dict[str, Any]]] = {}
    for robot_id in BASELINE_ROBOT_IDS:
        path = paths[robot_id]
        reports[robot_id] = (path, json.loads(path.read_text()))
    return reports


def _failure_row(
    *,
    robot_id: str,
    failure_index: int,
    message: str,
    report: dict[str, Any],
    report_path: Path,
    report_sha256: str,
    semantic_map: dict[str, Any],
) -> dict[str, Any]:
    projection = _PROJECTION_FAILURE_RE.match(message) or _PROJECTION_NONFINITE_RE.match(message)
    if projection:
        return _projection_failure_row(
            robot_id=robot_id,
            failure_index=failure_index,
            message=message,
            report=report,
            report_path=report_path,
            report_sha256=report_sha256,
            semantic_map=semantic_map,
            motion=projection.group("motion"),
            task=projection.group("task"),
        )

    numerical = _NUMERICAL_FAILURE_RE.match(message)
    if numerical:
        task = numerical.group("task")
        return _numerical_failure_row(
            robot_id=robot_id,
            failure_index=failure_index,
            message=message,
            report=report,
            report_path=report_path,
            report_sha256=report_sha256,
            semantic_map=semantic_map,
            task=task,
        )

    raise ValueError(f"{robot_id}: unsupported failure row: {message}")


def _projection_failure_row(
    *,
    robot_id: str,
    failure_index: int,
    message: str,
    report: dict[str, Any],
    report_path: Path,
    report_sha256: str,
    semantic_map: dict[str, Any],
    motion: str,
    task: str,
) -> dict[str, Any]:
    task_report = ((report.get("projection_reports") or {}).get(motion) or {}).get(task)
    if not isinstance(task_report, dict):
        raise ValueError(f"{robot_id}: projection report missing for {motion}.{task}")
    metric_type = _projection_metric_type(task)
    active_coordinates = list(task_report.get("active_coordinates") or _active_coordinates(report, task))
    actual = {
        "normalized_residual": task_report.get("normalized_residual"),
        "residual": task_report.get("residual"),
        "normalization_scale": task_report.get("normalization_scale"),
        "normalization_reference": task_report.get("normalization_reference"),
        "prior_residual_norm": task_report.get("prior_residual_norm"),
    }
    target_distance, target_angle = _target_distance_angle(task_report, metric_type)
    threshold_value = _projection_quality_threshold(task, motion)
    return _base_row(
        robot_id=robot_id,
        failure_index=failure_index,
        message=message,
        report=report,
        report_path=report_path,
        report_sha256=report_sha256,
        semantic_map=semantic_map,
        motion=motion,
        task=task,
        metric_type=metric_type,
        threshold={
            "value": threshold_value,
            "unit": "normalized_residual",
            "source": "soma_retargeter.robotics.v3.profile._projection_quality_threshold",
            "rule": _projection_threshold_rule(task, motion),
        },
        actual=actual,
        active_coordinates=active_coordinates,
        solver={
            "status": task_report.get("status"),
            "message": task_report.get("solver_message"),
            "converged": task_report.get("converged"),
            "iterations": task_report.get("iterations"),
        },
        target_distance=target_distance,
        target_angle=target_angle,
        target_evidence=_target_evidence(task_report),
    )


def _numerical_failure_row(
    *,
    robot_id: str,
    failure_index: int,
    message: str,
    report: dict[str, Any],
    report_path: Path,
    report_sha256: str,
    semantic_map: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    rank = (report.get("rank_stability") or {}).get(task)
    if not isinstance(rank, dict):
        raise ValueError(f"{robot_id}: rank_stability report missing for {task}")
    active_coordinates = _active_coordinates(report, task)
    reference_projection = ((report.get("projection_reports") or {}).get("neutral") or {}).get(task, {})
    task_block = str(rank.get("task_block") or _task_block_for_task(task))
    target_metric = PROJECTION_ROTATION_METRIC if task_block == "rotation" else PROJECTION_POSITION_METRIC
    target_distance, target_angle = _target_distance_angle(reference_projection, target_metric)
    actual = {
        "numerical_stability_gate_passed": rank.get("numerical_stability_gate_passed"),
        "stable_sample_fraction": rank.get("stable_sample_fraction"),
        "engine_available_rate": _engine_available_rate(rank),
        "relevant_rank_agreement_rate": rank.get("relevant_rank_agreement_rate"),
        "projector_distance_p95": rank.get("projector_distance_p95"),
        "engine_fd_normalized_error_p95": rank.get("engine_fd_normalized_error_p95"),
    }
    actual["failed_criteria"] = _failed_numerical_criteria(actual)
    return _base_row(
        robot_id=robot_id,
        failure_index=failure_index,
        message=message,
        report=report,
        report_path=report_path,
        report_sha256=report_sha256,
        semantic_map=semantic_map,
        motion="rank_stability",
        task=task,
        metric_type=NUMERICAL_STABILITY_METRIC,
        threshold={
            "value": False,
            "unit": "boolean_gate",
            "source": "soma_retargeter.robotics.v3.reachability.analyze_reachability",
            "criteria": dict(_NUMERICAL_STABILITY_THRESHOLDS),
        },
        actual=actual,
        active_coordinates=active_coordinates,
        solver={
            "status": "not_applicable/numerical_stability_gate",
            "message": "rank_stability gate failure; no projection solver generated this failure row",
            "reference_projection_status": reference_projection.get("status"),
            "reference_projection_message": reference_projection.get("solver_message"),
        },
        target_distance=target_distance,
        target_angle=target_angle,
        target_evidence=_target_evidence(reference_projection),
    )


def _base_row(
    *,
    robot_id: str,
    failure_index: int,
    message: str,
    report: dict[str, Any],
    report_path: Path,
    report_sha256: str,
    semantic_map: dict[str, Any],
    motion: str,
    task: str,
    metric_type: str,
    threshold: dict[str, Any],
    actual: dict[str, Any],
    active_coordinates: list[int],
    solver: dict[str, Any],
    target_distance: float | None,
    target_angle: float | None,
    target_evidence: dict[str, Any],
) -> dict[str, Any]:
    rank = _rank_summary(report, task)
    rank.update(_taxonomy_rank_paths(report, task))
    return {
        "id": f"{robot_id}:{failure_index:03d}:{motion}.{task}:{metric_type}",
        "status": "failed",
        "robot_id": robot_id,
        "motion": motion,
        "task": task,
        "metric_type": metric_type,
        "original_failure_message": message,
        "failure_index": failure_index,
        "source_report": {
            "path": _display_path(report_path),
            "sha256": report_sha256,
            "deterministic_hash": report.get("deterministic_hash"),
            "schema_version": report.get("schema_version"),
            "status": report.get("status"),
            "status_reason": report.get("status_reason"),
        },
        "threshold": threshold,
        "actual": actual,
        "active_coordinates": active_coordinates,
        "active_joint_limits": _active_joint_limits(report, active_coordinates),
        "engine_jacobian_source": _engine_jacobian_source(report, task),
        "rank": rank,
        "solver": solver,
        "semantic_source": (report.get("semantic_map_resolution") or {}).get("source"),
        "semantic_hash": semantic_map.get("model_fingerprint"),
        "semantic": _semantic_evidence(report, semantic_map, task),
        "chain_length": _chain_length(report, task, target_evidence),
        "target_distance": target_distance,
        "target_angle": target_angle,
        "target": target_evidence,
    }


def _projection_quality_threshold(task_name: str, motion_name: str) -> float:
    if motion_name == "neutral":
        return 1e-3
    if "foot" in task_name:
        return 0.06
    if "hand" in task_name:
        return 0.12
    if "torso" in task_name:
        return 0.08
    return 0.05


def _projection_threshold_rule(task_name: str, motion_name: str) -> str:
    if motion_name == "neutral":
        return "neutral_position_abs_m"
    if "foot" in task_name:
        return "foot_rho_p"
    if "hand" in task_name:
        return "hand_rho_p"
    if "torso" in task_name:
        return "torso_rho_r"
    return "default_normalized_residual"


def _projection_metric_type(task_name: str) -> str:
    if "torso" in task_name:
        return PROJECTION_ROTATION_METRIC
    return PROJECTION_POSITION_METRIC


def _target_distance_angle(task_report: Any, metric_type: str) -> tuple[float | None, float | None]:
    if not isinstance(task_report, dict):
        return None, None
    norm = _vector_norm(task_report.get("desired"))
    if metric_type == PROJECTION_ROTATION_METRIC:
        return None, norm
    return norm, None


def _target_evidence(task_report: Any) -> dict[str, Any]:
    if not isinstance(task_report, dict):
        return {
            "desired": None,
            "projected": None,
            "desired_source": None,
            "reference_semantic": None,
            "target_semantic": None,
        }
    return {
        "desired": task_report.get("desired"),
        "projected": task_report.get("projected"),
        "desired_source": task_report.get("desired_source"),
        "canonical_desired_source": task_report.get("canonical_desired_source"),
        "reference_semantic": task_report.get("reference_semantic"),
        "target_semantic": task_report.get("target_semantic"),
        "residual": task_report.get("residual"),
        "normalized_residual": task_report.get("normalized_residual"),
    }


def _active_coordinates(report: dict[str, Any], task: str) -> list[int]:
    jac = (report.get("neutral_jacobians") or {}).get(task) or {}
    if isinstance(jac, dict) and jac.get("active_coordinates") is not None:
        return list(jac["active_coordinates"])
    chain = (report.get("chains") or {}).get(task) or {}
    if isinstance(chain, dict):
        return list(chain.get("active_velocity_coordinates") or [])
    return []


def _active_joint_limits(report: dict[str, Any], active_coordinates: list[int]) -> list[dict[str, Any]]:
    coordinates = {
        int(coordinate["index"]): coordinate
        for coordinate in (report.get("runtime_adapter") or {}).get("coordinates") or []
        if isinstance(coordinate, dict) and "index" in coordinate
    }
    limits: list[dict[str, Any]] = []
    for index in active_coordinates:
        source = coordinates.get(int(index))
        if not isinstance(source, dict):
            limits.append({"coordinate_index": int(index), "missing": True})
            continue
        limits.append(
            {
                "coordinate_index": source.get("index"),
                "label": source.get("label"),
                "joint_name": source.get("joint_name"),
                "joint_type": source.get("joint_type"),
                "limited": source.get("limited"),
                "lower": source.get("lower"),
                "upper": source.get("upper"),
                "qpos_adr": source.get("qpos_adr"),
                "dof_adr": source.get("dof_adr"),
            }
        )
    return limits


def _engine_jacobian_source(report: dict[str, Any], task: str) -> dict[str, Any]:
    jac = (report.get("neutral_jacobians") or {}).get(task) or {}
    engine = jac.get("engine_relative_jacobian") if isinstance(jac, dict) else {}
    crosscheck = jac.get("engine_translation_crosscheck") if isinstance(jac, dict) else {}
    engine = engine if isinstance(engine, dict) else {}
    crosscheck = crosscheck if isinstance(crosscheck, dict) else {}
    return {
        "primary": jac.get("primary_jacobian_source") if isinstance(jac, dict) else None,
        "source": engine.get("source") or crosscheck.get("source"),
        "backend": engine.get("backend") or crosscheck.get("backend"),
        "finite": engine.get("finite"),
        "scalar_dtype": engine.get("scalar_dtype") or crosscheck.get("scalar_dtype"),
        "convention": engine.get("convention") or crosscheck.get("convention"),
        "translation_crosscheck": {
            "source": crosscheck.get("source"),
            "backend": crosscheck.get("backend"),
            "finite": crosscheck.get("finite"),
            "max_abs_error": crosscheck.get("max_abs_error"),
            "frobenius_error": crosscheck.get("frobenius_error"),
        },
    }


def _rank_summary(report: dict[str, Any], task: str) -> dict[str, Any]:
    rank = (report.get("rank_stability") or {}).get(task)
    if not isinstance(rank, dict):
        return {}
    return {field: rank[field] for field in _RANK_FIELDS if field in rank}


def _taxonomy_rank_paths(report: dict[str, Any], task: str) -> dict[str, Any]:
    taxonomy_tasks = (
        ((report.get("failure_taxonomy") or {}).get("algorithm") or {})
        .get("numerical_stability", {})
        .get("tasks", [])
    )
    for item in taxonomy_tasks:
        if isinstance(item, dict) and item.get("task") == task:
            return {
                "false_gate_paths": item.get("false_gate_paths", []),
                "severe_classification_paths": item.get("severe_classification_paths", []),
            }
    return {"false_gate_paths": [], "severe_classification_paths": []}


def _engine_available_rate(rank: dict[str, Any]) -> float | None:
    samples = rank.get("sample_diagnostics")
    if not isinstance(samples, list) or not samples:
        return None
    available = 0
    total = 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        total += 1
        engine = sample.get("engine_jacobian") or {}
        if isinstance(engine, dict) and engine.get("available") is True:
            available += 1
    if total == 0:
        return None
    return float(available / total)


def _failed_numerical_criteria(actual: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if _below(actual.get("stable_sample_fraction"), _NUMERICAL_STABILITY_THRESHOLDS["stable_sample_fraction_min"]):
        failed.append("stable_sample_fraction")
    if _below(actual.get("engine_available_rate"), _NUMERICAL_STABILITY_THRESHOLDS["engine_available_rate_min"]):
        failed.append("engine_available_rate")
    if _below(
        actual.get("relevant_rank_agreement_rate"),
        _NUMERICAL_STABILITY_THRESHOLDS["relevant_rank_agreement_rate_min"],
    ):
        failed.append("relevant_rank_agreement_rate")
    if _above(actual.get("projector_distance_p95"), _NUMERICAL_STABILITY_THRESHOLDS["projector_distance_p95_max"]):
        failed.append("projector_distance_p95")
    if _above(
        actual.get("engine_fd_normalized_error_p95"),
        _NUMERICAL_STABILITY_THRESHOLDS["engine_fd_normalized_error_p95_max"],
    ):
        failed.append("engine_fd_normalized_error_p95")
    return failed


def _semantic_evidence(report: dict[str, Any], semantic_map: dict[str, Any], task: str) -> dict[str, Any]:
    chain = (report.get("chains") or {}).get(task) or {}
    reference_name = chain.get("reference")
    target_name = chain.get("target")
    sites = report.get("semantic_sites") or {}
    map_source = semantic_map.get("model_source") or {}
    return {
        "map_resolution": report.get("semantic_map_resolution"),
        "map_model_id": semantic_map.get("model_id"),
        "map_verification_status": semantic_map.get("verification_status"),
        "map_model_fingerprint": semantic_map.get("model_fingerprint"),
        "map_file_sha256": semantic_map.get("_file_sha256"),
        "model_source": map_source,
        "reference": _semantic_site_row(sites.get(reference_name), reference_name),
        "target": _semantic_site_row(sites.get(target_name), target_name),
    }


def _semantic_site_row(site: Any, semantic_name: str | None) -> dict[str, Any]:
    if not isinstance(site, dict):
        return {"semantic_name": semantic_name, "source": None, "body_name": None}
    return {
        "semantic_name": site.get("semantic_name") or semantic_name,
        "source": site.get("source"),
        "body_name": site.get("body_name"),
        "reason": site.get("reason"),
        "confidence": site.get("confidence"),
        "evidence": site.get("evidence", []),
        "local_position": site.get("local_position"),
        "local_rotation_xyzw": site.get("local_rotation_xyzw"),
    }


def _chain_length(report: dict[str, Any], task: str, target_evidence: dict[str, Any]) -> dict[str, Any]:
    chain = (report.get("chains") or {}).get(task) or {}
    body_path = list(chain.get("body_path") or []) if isinstance(chain, dict) else []
    active = list(chain.get("active_velocity_coordinates") or []) if isinstance(chain, dict) else []
    return {
        "body_path_edges": max(len(body_path) - 1, 0),
        "body_path_bodies": len(body_path),
        "active_coordinate_count": len(active),
        "normalization_scale": _normalization_scale_from_projection(report, task, target_evidence),
        "normalization_reference": _normalization_reference_from_projection(report, task, target_evidence),
    }


def _normalization_scale_from_projection(
    report: dict[str, Any],
    task: str,
    target_evidence: dict[str, Any],
) -> float | None:
    projection = _find_projection_report(report, task, target_evidence)
    return projection.get("normalization_scale") if projection else None


def _normalization_reference_from_projection(
    report: dict[str, Any],
    task: str,
    target_evidence: dict[str, Any],
) -> str | None:
    projection = _find_projection_report(report, task, target_evidence)
    return projection.get("normalization_reference") if projection else None


def _find_projection_report(report: dict[str, Any], task: str, target_evidence: dict[str, Any]) -> dict[str, Any] | None:
    desired = target_evidence.get("desired")
    projected = target_evidence.get("projected")
    for motion in (report.get("projection_reports") or {}).values():
        if not isinstance(motion, dict):
            continue
        candidate = motion.get(task)
        if not isinstance(candidate, dict):
            continue
        if candidate.get("desired") == desired and candidate.get("projected") == projected:
            return candidate
    return None


def _load_semantic_map(report: dict[str, Any]) -> dict[str, Any]:
    resolution = report.get("semantic_map_resolution") or {}
    path = Path(resolution.get("path") or "")
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    payload["_file_sha256"] = _sha256_file(path)
    return payload


def _vector_norm(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    try:
        return math.sqrt(sum(float(value) * float(value) for value in values))
    except (TypeError, ValueError):
        return None


def _task_block_for_task(task: str) -> str:
    if "torso" in task:
        return "rotation"
    if "hand" in task or "foot" in task:
        return "translation"
    return "translation+rotation"


def _below(value: Any, threshold: float) -> bool:
    return value is not None and float(value) < threshold


def _above(value: Any, threshold: float) -> bool:
    return value is not None and float(value) > threshold


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    return path.as_posix()
