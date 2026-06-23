from __future__ import annotations

import json
from pathlib import Path

import soma_retargeter.robotics.v3.validation as validation_module


def _summary(status: str, *, robot_class: str = "humanoid", expected_capability: str = "positive") -> dict:
    return {
        "status": status,
        "status_reason": "test",
        "failures": [],
        "warnings": [],
        "robot_class": robot_class,
        "expected_capability": expected_capability,
        "required": True,
        "redistribution": "kinematic_snapshot",
    }


def _write_report(per_robot: Path, model_id: str, *, status: str, tasks: dict[str, dict] | None = None) -> None:
    payload = {
        "schema_version": 4,
        "status": status,
        "manifest_entry": {
            "id": model_id,
            "robot_class": "humanoid",
            "expected_capability": "positive",
            "required": True,
        },
        "runtime_adapter": {"nq": 8, "nv": 8},
        "task_certificate_summary": {
            "schema_version": 1,
            "task_count": len(tasks or {}),
            "per_task": tasks or {},
        },
    }
    per_robot.mkdir(parents=True, exist_ok=True)
    (per_robot / f"{model_id}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _task_certificates() -> dict[str, dict]:
    return {
        "torso": {
            "reference": "Hips",
            "target": "Chest",
            "active_coordinate_count": 1,
            "joint_type_counts": {"revolute": 1},
            "rank": {"translation": 0, "rotation": 1},
            "motion_count": 2,
            "max_normalized_residual": 0.0,
            "statuses": ["converged"],
        },
        "left_foot": {
            "reference": "Hips",
            "target": "LeftFoot",
            "active_coordinate_count": 3,
            "joint_type_counts": {"revolute": 3},
            "rank": {"translation": 3, "rotation": 2},
            "motion_count": 2,
            "max_normalized_residual": 0.02,
            "statuses": ["converged", "converged/with_residual"],
        },
    }


def test_cross_format_ignores_negative_controls_and_compares_shared_task_certificates(tmp_path: Path):
    per_robot = tmp_path / "per_robot"
    tasks = _task_certificates()
    for model_id, status in {
        "limited_urdf": "capability_limited_passed",
        "limited_mjcf": "passed",
        "skip_urdf": "passed",
        "skip_mjcf": "algorithm_failed",
    }.items():
        _write_report(per_robot, model_id, status=status, tasks=tasks)
    for model_id in ("go2_urdf", "go2_mjcf"):
        _write_report(per_robot, model_id, status="negative_control_passed")

    reports = {
        "limited_urdf": _summary("capability_limited_passed"),
        "limited_mjcf": _summary("passed"),
        "skip_urdf": _summary("passed"),
        "skip_mjcf": _summary("algorithm_failed"),
        "go2_urdf": _summary("negative_control_passed", robot_class="quadruped", expected_capability="negative_control"),
        "go2_mjcf": _summary("negative_control_passed", robot_class="quadruped", expected_capability="negative_control"),
    }

    cross_format = validation_module._cross_format_report(
        reports,
        per_robot,
        validation_checks={
            "g1_mjcf_urdf_equivalence": {
                "status": "passed",
                "strict_equivalent": True,
                "gate_a_status": "complete_passed",
                "gate_a_evidence_complete": True,
                "evidence_statuses": {
                    "semantic_fk": "passed",
                    "active_chains": "passed",
                    "rank_summary": "passed",
                    "canonical_projection": "passed",
                },
                "evidence_incomplete_reasons": {},
                "differences": {},
            }
        },
    )

    gate = cross_format["gates"]["variant_compatibility"]
    assert gate["status"] == "passed"
    assert gate["eligible_pair_count"] == 1
    assert gate["passed_pair_count"] == 1
    assert gate["not_eligible_pair_count"] == 2
    assert gate["pair_statuses"]["go2"]["status"] == "not_eligible"
    assert "negative_control" in gate["pair_statuses"]["go2"]["reason"]
    assert gate["pair_statuses"]["skip"]["status"] == "not_eligible"
    assert "failures" not in gate["pair_statuses"]["skip"]
    limited = gate["pair_statuses"]["limited"]
    assert limited["status"] == "passed"
    assert limited["evidence"]["shared_task_certificates"]["status"] == "passed"
    assert limited["evidence"]["shared_task_certificates"]["shared_tasks"] == ["left_foot", "torso"]
    assert limited["evidence"]["shared_task_certificates"]["per_task"]["left_foot"]["status"] == "passed"


def test_not_eligible_pairs_do_not_fail_variant_gate(tmp_path: Path):
    per_robot = tmp_path / "per_robot"
    reports = {
        "toy_urdf": _summary("passed"),
        "toy_mjcf": _summary("source_unavailable"),
        "go2_urdf": _summary("negative_control_passed", robot_class="quadruped", expected_capability="negative_control"),
        "go2_mjcf": _summary("negative_control_passed", robot_class="quadruped", expected_capability="negative_control"),
    }

    cross_format = validation_module._cross_format_report(reports, per_robot, validation_checks={})

    gate = cross_format["gates"]["variant_compatibility"]
    assert gate["status"] == "blocked"
    assert gate["eligible_pair_count"] == 0
    assert gate["not_eligible_pair_count"] == 2
    assert all(pair["status"] == "not_eligible" for pair in gate["pair_statuses"].values())


def test_capability_limited_variant_pair_allows_matching_over_threshold_residuals(tmp_path: Path):
    per_robot = tmp_path / "per_robot"
    tasks = _task_certificates()
    tasks["torso"]["max_normalized_residual"] = 0.12126213
    _write_report(per_robot, "limited_urdf", status="capability_limited_passed", tasks=tasks)
    tasks = _task_certificates()
    tasks["torso"]["max_normalized_residual"] = 0.12126214
    _write_report(per_robot, "limited_mjcf", status="capability_limited_passed", tasks=tasks)

    reports = {
        "limited_urdf": _summary("capability_limited_passed"),
        "limited_mjcf": _summary("capability_limited_passed"),
    }

    cross_format = validation_module._cross_format_report(reports, per_robot, validation_checks={})
    shared = cross_format["gates"]["variant_compatibility"]["pair_statuses"]["limited"]["evidence"][
        "shared_task_certificates"
    ]

    assert shared["status"] == "passed"
    assert shared["per_task"]["torso"]["status"] == "passed"
