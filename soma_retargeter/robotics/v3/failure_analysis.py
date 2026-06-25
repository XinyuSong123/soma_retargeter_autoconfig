"""Step 2.3 failure ledger extraction for capability baseline reports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import re
import subprocess
from typing import Any


DEFAULT_FAILURE_REPORT_DIR = Path("artifacts/retargeting_v3_step2_assets44/failures")
DEFAULT_BASELINE_COMMIT = "5ad5a001c445c525d4c8bbaf6339dec5c5c2c719"
DEFAULT_BASELINE_ARTIFACT_ROOT = Path("artifacts/retargeting_v3_step2_assets44")
DEFAULT_BASELINE_LEDGER_PATH = Path("artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json")
DEFAULT_BASELINE_SUMMARY_PATH = Path("artifacts/retargeting_v3_step2_capability/baseline_summary.json")
DEFAULT_CAPABILITY_SUMMARY_PATH = Path("artifacts/retargeting_v3_step2_capability/summary.json")
DEFAULT_BEFORE_AFTER_PATH = Path("artifacts/retargeting_v3_step2_capability/before_after.json")

EXPECTED_BASELINE_STATUS_COUNTS = {
    "algorithm_failed": 11,
    "negative_control_passed": 9,
    "partial_passed": 3,
    "passed": 21,
}
ALLOWED_BASELINE_TRANSITIONS = frozenset(
    {
        ("passed", "passed"),
        ("partial_passed", "partial_passed"),
        ("negative_control_passed", "negative_control_passed"),
        ("algorithm_failed", "passed"),
        ("algorithm_failed", "capability_limited_passed"),
    }
)

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


class BaselineGitObjectError(RuntimeError):
    """Raised when the frozen baseline commit or object path cannot be read."""


@dataclass(frozen=True)
class _FrozenReport:
    path: Path
    report: dict[str, Any]
    sha256: str


def build_true_baseline_summary(
    *,
    source_commit: str = DEFAULT_BASELINE_COMMIT,
    artifact_root: str | Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Build the immutable baseline status summary from the pinned git commit."""

    root = Path(artifact_root)
    summary_path = root / "summary.json"
    raw_summary = _git_show_bytes(source_commit, summary_path)
    source_summary = _loads_git_json(raw_summary, summary_path)
    reports = source_summary.get("reports")
    if not isinstance(reports, dict):
        raise ValueError(f"{summary_path}: baseline summary reports must be an object")

    robot_ids = tuple(sorted(reports))
    if len(robot_ids) != 44:
        raise ValueError(f"{summary_path}: baseline must contain exactly 44 model IDs, found {len(robot_ids)}")

    computed_counts = _status_counts_from_reports(reports)
    if computed_counts != EXPECTED_BASELINE_STATUS_COUNTS:
        raise ValueError(
            f"{summary_path}: baseline status counts must be {EXPECTED_BASELINE_STATUS_COUNTS}, "
            f"found {computed_counts}"
        )
    declared_counts = source_summary.get("status_counts")
    if declared_counts is not None and dict(declared_counts) != EXPECTED_BASELINE_STATUS_COUNTS:
        raise ValueError(
            f"{summary_path}: declared status_counts must be {EXPECTED_BASELINE_STATUS_COUNTS}, "
            f"found {declared_counts}"
        )

    baseline_reports = {
        robot_id: _baseline_summary_report(reports[robot_id])
        for robot_id in robot_ids
        if isinstance(reports[robot_id], dict)
    }
    if tuple(sorted(baseline_reports)) != robot_ids:
        raise ValueError(f"{summary_path}: every baseline report row must be an object")

    return {
        "schema_version": 1,
        "baseline_commit": source_commit,
        "source_access": "git_object",
        "source_artifact_root": _display_path(root),
        "source_summary_path": _display_path(summary_path),
        "source_summary_sha256": _sha256_bytes(raw_summary),
        "model_count": len(robot_ids),
        "status_counts": dict(EXPECTED_BASELINE_STATUS_COUNTS),
        "robot_ids": list(robot_ids),
        "passed_robot_ids": _robot_ids_with_status(baseline_reports, "passed"),
        "partial_robot_ids": _robot_ids_with_status(baseline_reports, "partial_passed"),
        "negative_control_robot_ids": _robot_ids_with_status(baseline_reports, "negative_control_passed"),
        "algorithm_failure_robot_ids": _robot_ids_with_status(baseline_reports, "algorithm_failed"),
        "reports": baseline_reports,
    }


def write_true_baseline_summary(
    output_path: str | Path = DEFAULT_BASELINE_SUMMARY_PATH,
    *,
    source_commit: str = DEFAULT_BASELINE_COMMIT,
    artifact_root: str | Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
) -> dict[str, Any]:
    summary = build_true_baseline_summary(source_commit=source_commit, artifact_root=artifact_root)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_capability_before_after(
    current_summary_path: str | Path = DEFAULT_CAPABILITY_SUMMARY_PATH,
    *,
    source_commit: str = DEFAULT_BASELINE_COMMIT,
    artifact_root: str | Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Build the true before/after matrix using the pinned baseline commit."""

    baseline = build_true_baseline_summary(source_commit=source_commit, artifact_root=artifact_root)
    current_path = Path(current_summary_path)
    current_summary = json.loads(current_path.read_text())
    current_reports = current_summary.get("reports")
    if not isinstance(current_reports, dict):
        raise ValueError(f"{current_path}: current summary reports must be an object")

    baseline_reports = baseline["reports"]
    baseline_ids = set(baseline_reports)
    current_ids = set(current_reports)
    if current_ids != baseline_ids:
        missing = sorted(baseline_ids - current_ids)
        extra = sorted(current_ids - baseline_ids)
        raise ValueError(f"{current_path}: current model IDs must match baseline; missing={missing} extra={extra}")

    transition_counts: Counter[str] = Counter()
    invalid_transition_counts: Counter[str] = Counter()
    rows: dict[str, Any] = {}
    old_algorithm_rows: dict[str, Any] = {}
    for robot_id in sorted(baseline_reports):
        before = baseline_reports[robot_id]
        after = current_reports[robot_id]
        if not isinstance(after, dict):
            raise ValueError(f"{current_path}: current report row must be an object for {robot_id}")
        before_status = before.get("status")
        after_status = after.get("status")
        transition = f"{before_status}->{after_status}"
        transition_allowed = (before_status, after_status) in ALLOWED_BASELINE_TRANSITIONS
        transition_counts[transition] += 1
        if not transition_allowed:
            invalid_transition_counts[transition] += 1
        row = {
            "baseline_commit": source_commit,
            "before_status": before_status,
            "before_status_reason": before.get("status_reason"),
            "after_status": after_status,
            "after_status_reason": after.get("status_reason"),
            "transition": transition,
            "transition_allowed": transition_allowed,
            "status_changed": before_status != after_status,
            "baseline_failure_count": len(before.get("failures") or []),
            "after_failure_count": len(after.get("failures") or []),
            "task_certificate_summary": after.get("task_certificate_summary"),
        }
        if not transition_allowed:
            row["blocked_reason"] = "illegal_baseline_transition"
        rows[robot_id] = row
        if before_status == "algorithm_failed":
            old_algorithm_rows[robot_id] = row

    invalid_transitions = [
        {"robot_id": robot_id, **row}
        for robot_id, row in rows.items()
        if row["transition_allowed"] is False
    ]
    old_algorithm_counts = Counter(row["after_status"] for row in old_algorithm_rows.values())
    return {
        "schema_version": 2,
        "basis": "true_baseline_git_object",
        "baseline_commit": source_commit,
        "baseline_artifact_root": _display_path(Path(artifact_root)),
        "current_summary_path": _display_path(current_path),
        "row_count": len(rows),
        "baseline_counts": dict(baseline["status_counts"]),
        "after_counts": _status_counts_from_reports(current_reports),
        "status_transition_counts": dict(sorted(transition_counts.items())),
        "allowed_transitions": [
            f"{before}->{after}" for before, after in sorted(ALLOWED_BASELINE_TRANSITIONS)
        ],
        "transition_validation": {
            "status": "passed" if not invalid_transitions else "failed",
            "invalid_count": len(invalid_transitions),
            "invalid_transition_counts": dict(sorted(invalid_transition_counts.items())),
            "invalid_transitions": invalid_transitions,
        },
        "final_count_validation": _final_count_validation(_status_counts_from_reports(current_reports), len(rows)),
        "old_algorithm_failure_transitions": {
            "row_count": len(old_algorithm_rows),
            "after_status_counts": dict(sorted(old_algorithm_counts.items())),
            "rows": old_algorithm_rows,
        },
        "rows": rows,
    }


def write_capability_before_after(
    output_path: str | Path = DEFAULT_BEFORE_AFTER_PATH,
    *,
    current_summary_path: str | Path = DEFAULT_CAPABILITY_SUMMARY_PATH,
    source_commit: str = DEFAULT_BASELINE_COMMIT,
    artifact_root: str | Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
) -> dict[str, Any]:
    before_after = build_capability_before_after(
        current_summary_path=current_summary_path,
        source_commit=source_commit,
        artifact_root=artifact_root,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(before_after, indent=2, sort_keys=True) + "\n")
    return before_after


def build_baseline_failure_ledger(
    failure_dir: str | Path = DEFAULT_FAILURE_REPORT_DIR,
    *,
    source_commit: str | None = DEFAULT_BASELINE_COMMIT,
    artifact_root: str | Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Build the frozen Step 2.3 capability failure ledger from report JSON."""

    if source_commit is None:
        root = Path(failure_dir)
        reports = _load_frozen_reports(root)
        source_access = "workspace"
        source_report_directory = _display_path(root)
        source_artifact_root = None
    else:
        root = Path(artifact_root)
        reports = _load_frozen_git_reports(source_commit, root)
        source_access = "git_object"
        source_report_directory = _display_path(root / "failures")
        source_artifact_root = _display_path(root)
    source_reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for robot_id in BASELINE_ROBOT_IDS:
        frozen = reports[robot_id]
        path = frozen.path
        report = frozen.report
        semantic_map = _load_semantic_map(
            report,
            source_commit=source_commit,
            robot_id=robot_id,
            artifact_root=root,
        )
        report_sha = frozen.sha256
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
        "source_access": source_access,
        "source_commit": source_commit,
        "source_artifact_root": source_artifact_root,
        "source_report_directory": source_report_directory,
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
    source_commit: str | None = DEFAULT_BASELINE_COMMIT,
    artifact_root: str | Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
) -> dict[str, Any]:
    ledger = build_baseline_failure_ledger(
        failure_dir,
        source_commit=source_commit,
        artifact_root=artifact_root,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return ledger


def _load_frozen_reports(root: Path) -> dict[str, _FrozenReport]:
    if not root.is_dir():
        raise FileNotFoundError(f"failure report directory missing: {root}")
    paths = {path.stem: path for path in sorted(root.glob("*.json"))}
    found = tuple(sorted(paths))
    if found != tuple(sorted(BASELINE_ROBOT_IDS)):
        missing = sorted(set(BASELINE_ROBOT_IDS) - set(found))
        extra = sorted(set(found) - set(BASELINE_ROBOT_IDS))
        raise ValueError(f"failure report IDs must match frozen baseline; missing={missing} extra={extra}")
    reports: dict[str, _FrozenReport] = {}
    for robot_id in BASELINE_ROBOT_IDS:
        path = paths[robot_id]
        reports[robot_id] = _FrozenReport(
            path=path,
            report=json.loads(path.read_text()),
            sha256=_sha256_file(path),
        )
    return reports


def _load_frozen_git_reports(source_commit: str, artifact_root: Path) -> dict[str, _FrozenReport]:
    _ensure_git_commit_available(source_commit)
    reports: dict[str, _FrozenReport] = {}
    for robot_id in BASELINE_ROBOT_IDS:
        path = artifact_root / "failures" / f"{robot_id}.json"
        raw = _git_show_bytes(source_commit, path)
        reports[robot_id] = _FrozenReport(
            path=path,
            report=_loads_git_json(raw, path),
            sha256=_sha256_bytes(raw),
        )
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


def _load_semantic_map(
    report: dict[str, Any],
    *,
    source_commit: str | None = None,
    robot_id: str | None = None,
    artifact_root: Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
) -> dict[str, Any]:
    resolution = report.get("semantic_map_resolution") or {}
    path_text = str(resolution.get("path") or "")
    path = Path(path_text)
    if source_commit is not None:
        candidates: list[Path] = []
        if path_text:
            candidates.append(path)
        if robot_id is not None:
            candidates.append(artifact_root / "semantic_maps" / f"{robot_id}.json")
        errors: list[str] = []
        for candidate in candidates:
            try:
                raw = _git_show_bytes(source_commit, candidate)
            except BaselineGitObjectError as exc:
                errors.append(str(exc))
                continue
            payload = _loads_git_json(raw, candidate)
            payload["_file_sha256"] = _sha256_bytes(raw)
            return payload
        raise BaselineGitObjectError(
            f"semantic map unavailable for {robot_id or '<unknown>'} at baseline commit {source_commit}: {errors}"
        )
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


def _baseline_summary_report(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": row.get("status"),
        "status_reason": row.get("status_reason"),
        "expected_capability": row.get("expected_capability"),
        "robot_class": row.get("robot_class"),
        "required": row.get("required"),
        "redistribution": row.get("redistribution"),
        "failures": list(row.get("failures") or []),
        "warnings": list(row.get("warnings") or []),
    }


def _robot_ids_with_status(reports: dict[str, dict[str, Any]], status: str) -> list[str]:
    return [robot_id for robot_id in sorted(reports) if reports[robot_id].get("status") == status]


def _status_counts_from_reports(reports: dict[str, Any]) -> dict[str, int]:
    counts = Counter()
    for row in reports.values():
        if not isinstance(row, dict):
            continue
        counts[str(row.get("status"))] += 1
    return dict(sorted(counts.items()))


def _final_count_validation(after_counts: dict[str, int], row_count: int) -> dict[str, Any]:
    failures: list[str] = []

    def count(status: str) -> int:
        return int(after_counts.get(status, 0))

    if count("passed") < 21:
        failures.append(f"passed must be >= 21, found {count('passed')}")
    if count("capability_limited_passed") > 11:
        failures.append(
            "capability_limited_passed must be <= 11, "
            f"found {count('capability_limited_passed')}"
        )
    if count("partial_passed") != 3:
        failures.append(f"partial_passed must be 3, found {count('partial_passed')}")
    if count("negative_control_passed") != 9:
        failures.append(f"negative_control_passed must be 9, found {count('negative_control_passed')}")
    for status in ("algorithm_failed", "source_unavailable", "model_load_failed", "semantic_failed"):
        if count(status) != 0:
            failures.append(f"{status} must be 0, found {count(status)}")
    if count("passed") + count("capability_limited_passed") != 32:
        failures.append(
            "passed + capability_limited_passed must be 32, "
            f"found {count('passed') + count('capability_limited_passed')}"
        )
    if row_count != 44:
        failures.append(f"terminal total must be 44, found {row_count}")

    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _display_path(path: Path) -> str:
    return path.as_posix()


def _ensure_git_commit_available(source_commit: str) -> None:
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise BaselineGitObjectError(f"baseline commit unavailable: {source_commit}: {stderr}")


def _git_show_bytes(source_commit: str, path: str | Path) -> bytes:
    _ensure_git_commit_available(source_commit)
    display_path = _display_path(Path(path))
    spec = f"{source_commit}:{display_path}"
    proc = subprocess.run(
        ["git", "show", spec],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise BaselineGitObjectError(f"baseline git object unavailable: {spec}: {stderr}")
    return proc.stdout


def _loads_git_json(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: baseline git object is not valid UTF-8 JSON") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: baseline git object must be a JSON object")
    return data
