"""Audit Step 2.1 numerical-core artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = (
    "environment.json",
    "commands.txt",
    "baseline.json",
    "summary.json",
    "before_after.json",
    "threshold_calibration.json",
)
REQUIRED_DIRS = ("per_robot", "per_chain", "failures", "test_results")
SUMMARY_FIELDS = (
    "manifest_total",
    "source_available",
    "load_success",
    "profile_eligible",
    "baseline_pass",
    "corrected_pass",
    "baseline_algorithm_failed",
    "corrected_algorithm_failed",
    "epsilon_only_failures_before",
    "epsilon_only_failures_after",
    "semantic_failed_unchanged",
    "model_load_failed_unchanged",
    "source_unavailable_unchanged",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/retargeting_v3_step2_numerical")
    args = parser.parse_args(argv)
    root = Path(args.artifact_dir)
    failures = audit(root)
    if failures:
        print("Step 2.1 numerical audit FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Step 2.1 numerical audit PASS")
    return 0


def audit(root: Path) -> list[str]:
    failures: list[str] = []
    if not root.exists():
        return [f"artifact directory missing: {root}"]
    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            failures.append(f"required file missing: {rel}")
    for rel in REQUIRED_DIRS:
        if not (root / rel).is_dir():
            failures.append(f"required directory missing: {rel}")
    if failures:
        return failures
    summary = _read_json(root / "summary.json")
    before_after = _read_json(root / "before_after.json")
    threshold = _read_json(root / "threshold_calibration.json")
    for field in SUMMARY_FIELDS:
        if field not in summary:
            failures.append(f"summary missing field: {field}")
    if summary.get("epsilon_only_failures_after") != 0:
        failures.append(f"epsilon_only_failures_after must be 0, got {summary.get('epsilon_only_failures_after')!r}")
    if summary.get("corrected_pass", 0) < summary.get("baseline_pass", 0):
        failures.append("corrected_pass regressed below baseline_pass")
    rows = before_after.get("models", {})
    if not isinstance(rows, dict) or not rows:
        failures.append("before_after.models missing or empty")
    if before_after.get("schema_version") != 2:
        failures.append("before_after.schema_version must be 2")
    for model_id, row in rows.items():
        if row.get("corrected_epsilon_only"):
            failures.append(f"{model_id}: corrected_epsilon_only is true")
        if row.get("baseline_status") in {"passed", "partial_passed"} and row.get("corrected_status") == "algorithm_failed":
            failures.append(
                f"{model_id}: baseline {row.get('baseline_status')} regressed to corrected algorithm_failed"
            )
        for field in (
            "relevant_task_metric",
            "zero_column_count",
            "real_unstable_count",
            "engine_fd_error",
            "rank_agreement",
            "subspace_distance",
            "canonical_residual_distribution",
            "runtime",
        ):
            if field not in row:
                failures.append(f"{model_id}: before_after missing {field}")
    fd = threshold.get("finite_difference", {})
    if fd.get("scales") != ["2h", "h", "h/2"]:
        failures.append("threshold_calibration finite_difference.scales must be ['2h', 'h', 'h/2']")
    if not fd.get("backend_aware_epsilon"):
        failures.append("threshold_calibration backend_aware_epsilon must be true")
    for model_id, report in _read_reports(root / "per_robot").items():
        failures.extend(_audit_report(model_id, report))
    return failures


def _audit_report(model_id: str, report: dict) -> list[str]:
    failures: list[str] = []
    for task, jac in (report.get("neutral_jacobians") or {}).items():
        primary = jac.get("primary_jacobian_source")
        if primary != "engine_relative_jacobian":
            failures.append(f"{model_id}.{task}: primary_jacobian_source is not engine_relative_jacobian")
        engine = jac.get("engine_relative_jacobian", {})
        if isinstance(engine, dict) and engine.get("available") is not False:
            for field in ("backend", "scalar_dtype", "source", "finite", "convention"):
                if field not in engine:
                    failures.append(f"{model_id}.{task}: engine_relative_jacobian missing {field}")
        crosscheck = jac.get("engine_translation_crosscheck", {})
        if crosscheck.get("available") is True:
            if crosscheck.get("source") in {None, "finite_difference_fallback"}:
                failures.append(f"{model_id}.{task}: engine source missing or fallback")
            if crosscheck.get("finite") is not True:
                failures.append(f"{model_id}.{task}: engine Jacobian is not finite")
            if "backend" not in crosscheck:
                failures.append(f"{model_id}.{task}: engine crosscheck backend missing")
    for task, reachability in (report.get("rank_stability") or {}).items():
        if "numerical_stability_gate_passed" not in reachability:
            failures.append(f"{model_id}.{task}: numerical_stability_gate_passed missing")
        if "task_block" not in reachability:
            failures.append(f"{model_id}.{task}: task_block missing")
        for field in (
            "engine_rank_translation",
            "engine_rank_rotation",
            "fd_rank_translation",
            "fd_rank_rotation",
            "relevant_rank_agreement_rate",
            "projector_distance_p95",
            "engine_fd_normalized_error_p95",
        ):
            if field not in reachability:
                failures.append(f"{model_id}.{task}: {field} missing")
    return failures


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_reports(directory: Path) -> dict[str, dict]:
    return {path.stem: _read_json(path) for path in sorted(directory.glob("*.json"))}


if __name__ == "__main__":
    raise SystemExit(main())
