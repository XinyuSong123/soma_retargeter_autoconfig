from __future__ import annotations

import json
from pathlib import Path

import soma_retargeter.robotics.v3.validation as validation_module
from soma_retargeter.robotics.v3.robot_zoo import allowed_status_values
from soma_retargeter.robotics.v3.target_builder import CANONICAL_MOTION_NAMES
from soma_retargeter.robotics.v3.validation import write_validation_artifacts


def _write_manifest(path: Path, models: list[dict]) -> Path:
    payload = {
        "schema_version": 1,
        "catalog_name": "capability-status-test-zoo",
        "aggregators": {},
        "policies": {},
        "models": models,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _entry(model_id: str, **overrides) -> dict:
    entry = {
        "id": model_id,
        "description_name": None,
        "format": "mjcf",
        "robot_class": "humanoid",
        "expected_capability": "positive",
        "license": "test",
        "redistribution": "kinematic_snapshot",
        "required": True,
        "source_family": "local",
        "notes": "",
    }
    entry.update(overrides)
    return entry


def _semantic_map(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "semantics": {
                    "Hips": {"body": "pelvis", "source": "verified_map", "confidence": 0.99},
                    "Chest": {"body": "torso", "source": "verified_map", "confidence": 0.99},
                    "LeftFoot": {"body": "left_foot", "source": "verified_map", "confidence": 0.99},
                    "RightFoot": {"body": "right_foot", "source": "verified_map", "confidence": 0.99},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


class _FakeProfile:
    def __init__(self, payload: dict):
        self._payload = payload

    def to_json(self) -> dict:
        return json.loads(json.dumps(self._payload))


def _partial_profile_payload(source: Path, *, model_id: str) -> dict:
    exact_tasks = {
        "torso": {
            "status": "converged",
            "converged": True,
            "normalized_residual": 0.0,
            "residual": 0.0,
            "iterations": 1,
            "capability_certificate": {
                "certificate_class": "exact_reachable",
                "gates": {
                    "exact_threshold_passed": True,
                    "projected_gradient_kkt": True,
                    "seed_consensus": True,
                },
            },
        },
        "left_foot": {
            "status": "converged",
            "converged": True,
            "normalized_residual": 0.0,
            "residual": 0.0,
            "iterations": 1,
            "capability_certificate": {
                "certificate_class": "exact_reachable",
                "gates": {
                    "exact_threshold_passed": True,
                    "projected_gradient_kkt": True,
                    "seed_consensus": True,
                },
            },
        },
    }
    motions = {
        motion: {"tasks": json.loads(json.dumps(exact_tasks))}
        for motion in CANONICAL_MOTION_NAMES
    }
    motions["single_step"]["tasks"]["left_foot"]["status"] = "converged/with_residual"
    motions["single_step"]["tasks"]["left_foot"]["normalized_residual"] = 0.02
    motions["single_step"]["tasks"]["left_foot"]["residual"] = 0.01
    return {
        "schema_version": 3,
        "model": {
            "id": model_id,
            "path": str(source),
            "format": "mjcf",
            "backend": "newton",
            "fingerprint": "capability-limited-test",
        },
        "runtime_adapter": {"backend": "newton", "nq": 8, "nv": 8},
        "deterministic_hash": "limited-stable-hash",
        "semantic_sites": {
            "Hips": {"semantic_name": "Hips", "body_name": "pelvis", "source": "verified_map"},
            "Chest": {"semantic_name": "Chest", "body_name": "torso", "source": "verified_map"},
            "LeftFoot": {"semantic_name": "LeftFoot", "body_name": "left_foot", "source": "verified_map"},
            "RightFoot": {"semantic_name": "RightFoot", "body_name": "right_foot", "source": "verified_map"},
        },
        "chains": {
            "torso": {
                "reference": "Hips",
                "target": "Chest",
                "active_velocity_coordinates": [0],
                "coordinate_labels": ["torso_yaw"],
                "joint_types": ["revolute"],
            },
            "left_foot": {
                "reference": "Hips",
                "target": "LeftFoot",
                "active_velocity_coordinates": [1, 2, 3],
                "coordinate_labels": ["l_hip", "l_knee", "l_ankle"],
                "joint_types": ["revolute", "revolute", "revolute"],
            },
        },
        "rank_stability": {
            "torso": {
                "regular_rank_translation": 0,
                "nominal_rank_translation": 0,
                "regular_rank_rotation": 1,
                "nominal_rank_rotation": 1,
                "epsilon_stability_gate_passed": True,
            },
            "left_foot": {
                "regular_rank_translation": 3,
                "nominal_rank_translation": 3,
                "regular_rank_rotation": 2,
                "nominal_rank_rotation": 2,
                "epsilon_stability_gate_passed": True,
            },
        },
        "canonical_projection_reports": {
            "motion_order": list(CANONICAL_MOTION_NAMES),
            "target_source": "canonical_semantic_targets",
            "failures": [],
            "unreachable_demands": [],
            "motions": motions,
        },
        "failures": [],
        "warnings": ["partial humanoid downgrade; unavailable semantics: LeftHand, RightHand"],
        "capability_status": "partial_humanoid",
        "timing": {"compile_seconds": 0.01},
        "reproduction_command": "",
    }


def _capability_limited_profile_payload(source: Path, *, model_id: str) -> dict:
    payload = _partial_profile_payload(source, model_id=model_id)
    payload["warnings"] = []
    payload["capability_status"] = "full_humanoid_ready"
    limited_task = payload["canonical_projection_reports"]["motions"]["single_step"]["tasks"]["left_foot"]
    limited_task["status"] = "converged/with_residual"
    limited_task["normalized_residual"] = 0.2
    limited_task["residual"] = 0.2
    limited_task["capability_certificate"] = {
        "certificate_class": "capability_limited_rank",
        "gates": {
            "exact_threshold_passed": False,
            "residual_explained": True,
            "projected_gradient_kkt": True,
            "seed_consensus": True,
            "continuation": True,
            "joint_limits": True,
            "numerical": True,
        },
        "decomposition": {
            "rank": 2,
            "residual_norm": 0.2,
            "demand_norm": 0.2,
            "rank_incompatible_fraction": 0.98,
            "active_limit_fraction": 0.0,
            "rank_incompatible_residual_norm": 0.196,
            "active_limit_residual_norm": 0.0,
            "component_tolerance": 1e-7,
        },
        "active_limits": {"lower": [], "upper": [], "count": 0},
        "seed_consensus": {"checked": True, "passed": True, "start_count": 4},
    }
    return payload


def test_partial_passed_remains_terminal_status_with_task_certificates(monkeypatch, tmp_path: Path):
    source = tmp_path / "partial.xml"
    source.write_text("<mujoco/>\n")
    semantics = _semantic_map(tmp_path / "semantics.json")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("partial_bot", source_path=str(source), semantic_map_path=str(semantics))],
    )

    def fake_compile(model, semantic_map_payload, **kwargs):
        return _FakeProfile(_partial_profile_payload(Path(model), model_id=kwargs["model_id"]))

    monkeypatch.setattr(validation_module, "compile_kinematic_profile_v3", fake_compile)

    summary = write_validation_artifacts(
        tmp_path / "artifacts",
        manifest_path=manifest,
        low_discrepancy_count=1,
        deterministic_rerun=True,
    )
    report = json.loads((tmp_path / "artifacts" / "per_robot" / "partial_bot.json").read_text())

    assert "capability_limited_passed" in allowed_status_values()
    assert "partial_passed" in allowed_status_values()
    assert summary["status_counts"] == {"partial_passed": 1}
    assert summary["algorithm_pass_count"] == 1
    assert summary["deterministic_rerun"]["status"] == "passed"
    assert summary["deterministic_rerun"]["totals"]["compared_count"] == 1
    assert report["status"] == "partial_passed"
    assert report["status_reason"] != report["status"]
    assert "partial-humanoid" in report["status_reason"]
    assert sorted(report["task_certificate_summary"]["per_task"]) == ["left_foot", "torso"]
    assert report["task_certificate_summary"]["per_task"]["left_foot"]["motion_count"] == len(CANONICAL_MOTION_NAMES)
    assert report["task_certificate_summary"]["per_task"]["left_foot"]["max_normalized_residual"] == 0.02


def test_capability_limited_passed_is_terminal_status_with_task_certificates(monkeypatch, tmp_path: Path):
    source = tmp_path / "limited.xml"
    source.write_text("<mujoco/>\n")
    semantics = _semantic_map(tmp_path / "semantics.json")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("limited_bot", source_path=str(source), semantic_map_path=str(semantics))],
    )

    def fake_compile(model, semantic_map_payload, **kwargs):
        return _FakeProfile(_capability_limited_profile_payload(Path(model), model_id=kwargs["model_id"]))

    monkeypatch.setattr(validation_module, "compile_kinematic_profile_v3", fake_compile)

    summary = write_validation_artifacts(
        tmp_path / "artifacts",
        manifest_path=manifest,
        low_discrepancy_count=1,
        deterministic_rerun=True,
    )
    report = json.loads((tmp_path / "artifacts" / "per_robot" / "limited_bot.json").read_text())

    assert summary["status_counts"] == {"capability_limited_passed": 1}
    assert summary["algorithm_pass_count"] == 1
    assert summary["deterministic_rerun"]["status"] == "passed"
    assert report["status"] == "capability_limited_passed"
    assert "available task certificate" in report["status_reason"]
