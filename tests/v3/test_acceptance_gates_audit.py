from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_retargeting_v3_step2 import (
    GOAL_FALSE_POSITIVE_GATES,
    GOAL_FALSE_POSITIVE_TRACEABILITY,
    REQUIRED_ARTIFACT_FILES,
    REQUIRED_CANONICAL_MOTIONS,
    run_audit,
)


def test_retargeting_v3_step2_acceptance_gate_coverage_is_complete():
    result = run_audit(
        artifact_dir=Path("artifacts/retargeting_v3_step2"),
        source_root=Path("."),
    )

    missing_gate_coverage = [gate for gate in GOAL_FALSE_POSITIVE_GATES if gate not in result.gate_counts]
    assert missing_gate_coverage == []
    missing_traceability = [
        false_positive
        for false_positive, gates in GOAL_FALSE_POSITIVE_TRACEABILITY.items()
        if not all(gate in result.gate_counts for gate in gates)
    ]
    assert missing_traceability == []
    assert result.status in {"PASS", "BLOCKED"}
    assert result.blocking_count == len(result.blocking_findings)


def test_current_artifact_false_negatives_are_not_silent():
    artifact_dir = Path("artifacts/retargeting_v3_step2")
    result = run_audit(artifact_dir=artifact_dir, source_root=Path("."))

    if _artifact_tree_has_local_absolute_paths(artifact_dir):
        assert result.gate_counts["absolute_cache_paths"] > 0
    if any(not (artifact_dir / relative).exists() for relative in REQUIRED_ARTIFACT_FILES):
        assert result.gate_counts["missing_required_artifacts"] > 0
    if _cross_format_gate_is_not_passed(artifact_dir / "cross_format.json"):
        assert result.gate_counts["cross_format_gates_not_run"] > 0
    if _terminal_profile_has_incomplete_motion_suite(artifact_dir / "per_robot"):
        assert result.gate_counts["canonical_motion_suite_incomplete"] > 0
    if _terminal_profile_has_unstable_epsilon_gate(artifact_dir / "per_robot"):
        assert result.gate_counts["epsilon_stability_failed"] > 0


def test_audit_blocks_local_absolute_paths_anywhere_in_artifacts(tmp_path: Path):
    artifact_dir = _write_baseline_artifacts(tmp_path)
    failures = artifact_dir / "failures"
    failures.mkdir()
    (failures / "robot.json").write_text('{"error": "`/mnt/ssd1/song/.cache/model.obj`"}\n')

    result = run_audit(artifact_dir=artifact_dir, source_root=Path("."))
    payload = json.dumps(result.to_json())

    assert result.gate_counts["absolute_cache_paths"] == 1
    assert result.status == "BLOCKED"
    assert "/mnt/ssd1/song" not in payload
    assert str(tmp_path) not in payload
    assert "${LOCAL_SOURCE_PATH}/model.obj" in payload


def test_audit_blocks_missing_required_reproducibility_artifacts(tmp_path: Path):
    artifact_dir = _write_baseline_artifacts(tmp_path, include_required_artifacts=False)

    result = run_audit(artifact_dir=artifact_dir, source_root=Path("."))

    assert result.gate_counts["missing_required_artifacts"] == len(REQUIRED_ARTIFACT_FILES)
    assert result.status == "BLOCKED"


def test_audit_blocks_cross_format_gates_left_not_run(tmp_path: Path):
    artifact_dir = _write_baseline_artifacts(
        tmp_path,
        cross_format={
            "gates": {
                "same_source_strict": {"status": "not_run", "reason": "scaffold"},
                "variant_compatibility": {"status": "not_run", "reason": "scaffold"},
            }
        },
    )

    result = run_audit(artifact_dir=artifact_dir, source_root=Path("."))

    assert result.gate_counts["cross_format_gates_not_run"] == 2
    assert result.status == "BLOCKED"


def test_audit_blocks_passed_profile_with_incomplete_canonical_motion_suite(tmp_path: Path):
    report = _passing_profile_report()
    report["canonical_projection_reports"]["motion_order"] = ["neutral", "single_step_target"]
    artifact_dir = _write_baseline_artifacts(tmp_path, reports={"toy_humanoid": report})

    result = run_audit(artifact_dir=artifact_dir, source_root=Path("."))

    assert result.gate_counts["canonical_motion_suite_incomplete"] == 1
    finding = _first_finding(result, "canonical_motion_suite_incomplete")
    assert "single_step" in finding.evidence["missing"]
    assert result.status == "BLOCKED"


def test_audit_blocks_passed_profile_with_failed_epsilon_stability_gate(tmp_path: Path):
    report = _passing_profile_report()
    report["rank_stability"]["left_hand"]["epsilon_stability_gate_passed"] = False
    artifact_dir = _write_baseline_artifacts(tmp_path, reports={"toy_humanoid": report})

    result = run_audit(artifact_dir=artifact_dir, source_root=Path("."))

    assert result.gate_counts["epsilon_stability_failed"] == 1
    assert result.status == "BLOCKED"


def _write_baseline_artifacts(
    tmp_path: Path,
    *,
    include_required_artifacts: bool = True,
    cross_format: dict | None = None,
    reports: dict[str, dict] | None = None,
) -> Path:
    artifact_dir = tmp_path / "artifacts"
    (artifact_dir / "per_robot").mkdir(parents=True)
    (artifact_dir / "semantic_maps").mkdir()
    (artifact_dir / "test_results").mkdir()
    (artifact_dir / "summary.json").write_text("{}\n")
    (artifact_dir / "validation_checks.json").write_text(
        json.dumps({"g1_mjcf_urdf_equivalence": {"status": "passed"}}) + "\n"
    )
    (artifact_dir / "environment.json").write_text(json.dumps({"git_status_short": ""}) + "\n")
    (artifact_dir / "commands.txt").write_text("\n")
    (artifact_dir / "cross_format.json").write_text(
        json.dumps(
            cross_format
            or {
                "gates": {
                    "same_source_strict": {"status": "passed"},
                    "variant_compatibility": {"status": "passed"},
                }
            }
        )
        + "\n"
    )
    for robot_id, report in (reports or {}).items():
        (artifact_dir / "per_robot" / f"{robot_id}.json").write_text(json.dumps(report) + "\n")
    if include_required_artifacts:
        for relative in REQUIRED_ARTIFACT_FILES:
            (artifact_dir / relative).write_text("{}\n")
    return artifact_dir


def _passing_profile_report() -> dict:
    return {
        "status": "passed",
        "canonical_projection_reports": {"motion_order": list(REQUIRED_CANONICAL_MOTIONS)},
        "rank_stability": {"left_hand": {"epsilon_stability_gate_passed": True}},
    }


def _first_finding(result, gate: str):
    return next(finding for finding in result.findings if finding.gate == gate)


def _artifact_tree_has_local_absolute_paths(artifact_dir: Path) -> bool:
    return any(
        any(prefix in path.read_text(errors="ignore") for prefix in ("/mnt/", "/home/", "/Users/"))
        for path in artifact_dir.rglob("*")
        if path.is_file()
    )


def _cross_format_gate_is_not_passed(path: Path) -> bool:
    if not path.exists():
        return True
    cross_format = json.loads(path.read_text())
    gates = cross_format.get("gates", {})
    return any(gates.get(name, {}).get("status") != "passed" for name in ("same_source_strict", "variant_compatibility"))


def _terminal_profile_has_incomplete_motion_suite(per_robot_dir: Path) -> bool:
    required = set(REQUIRED_CANONICAL_MOTIONS)
    for path in per_robot_dir.glob("*.json"):
        report = json.loads(path.read_text())
        if report.get("status") not in {"passed", "partial_passed"}:
            continue
        motion_order = report.get("canonical_projection_reports", {}).get("motion_order", [])
        if set(motion_order) != required:
            return True
    return False


def _terminal_profile_has_unstable_epsilon_gate(per_robot_dir: Path) -> bool:
    for path in per_robot_dir.glob("*.json"):
        report = json.loads(path.read_text())
        if report.get("status") not in {"passed", "partial_passed"}:
            continue
        for chain in report.get("rank_stability", {}).values():
            if isinstance(chain, dict) and chain.get("epsilon_stability_gate_passed") is False:
                return True
    return False
