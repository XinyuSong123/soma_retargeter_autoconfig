from __future__ import annotations

import json
from pathlib import Path

import soma_retargeter.robotics.v3.validation as validation_module
from soma_retargeter.robotics.v3.validation import write_validation_artifacts


def _write_manifest(path: Path, models: list[dict]) -> Path:
    payload = {
        "schema_version": 1,
        "catalog_name": "deterministic-test-zoo",
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
                    "Chest": {"body": "chest", "source": "verified_map", "confidence": 0.99},
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


def _profile_payload(source: Path, *, model_id: str, deterministic_hash: str, residual: float) -> dict:
    return {
        "schema_version": 3,
        "model": {
            "id": model_id,
            "path": str(source),
            "format": "mjcf",
            "backend": "newton",
            "fingerprint": "test-fingerprint",
        },
        "runtime_adapter": {"backend": "newton"},
        "deterministic_hash": deterministic_hash,
        "semantic_sites": {
            "Hips": {
                "semantic_name": "Hips",
                "body_name": "pelvis",
                "local_position": [0.0, 0.0, 0.0],
                "local_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "source": "verified_map",
                "confidence": 0.99,
                "reason": "verified_semantic_map",
            },
            "Chest": {
                "semantic_name": "Chest",
                "body_name": "chest",
                "local_position": [0.0, 0.0, 0.5],
                "local_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "source": "verified_map",
                "confidence": 0.99,
                "reason": "verified_semantic_map",
            },
        },
        "rank_stability": {
            "torso": {
                "regular_rank_translation": 0,
                "nominal_rank_translation": 0,
                "regular_rank_rotation": 1,
                "nominal_rank_rotation": 1,
                "singularity_fraction_translation": 0.0,
                "singularity_fraction_rotation": 0.0,
                "epsilon_unstable_fraction": 0.0,
                "epsilon_unstable_sample_fraction": 0.0,
                "epsilon_stability_gate_passed": True,
                "epsilon_unstable_columns": [],
                "samples": 3,
                "rank_method": "per-sample-local-rank-regular-fraction",
                "regular_rank_fraction_threshold": 0.2,
            }
        },
        "canonical_projection_reports": {
            "motion_order": ["neutral"],
            "target_source": "canonical_semantic_targets",
            "failures": [],
            "unreachable_demands": [],
            "motions": {
                "neutral": {
                    "tasks": {
                        "torso": {
                            "status": "converged",
                            "converged": True,
                            "residual": residual,
                            "normalized_residual": residual,
                            "normalization_scale": 1.0,
                            "iterations": 1,
                            "active_coordinates": [0],
                            "desired_source": "canonical_targets.transforms",
                            "reference": "Hips",
                            "target": "Chest",
                        }
                    }
                }
            },
        },
        "failures": [],
        "warnings": [],
        "capability_status": "full_humanoid_ready",
        "timing": {"compile_seconds": 0.01},
        "reproduction_command": "",
    }


def test_deterministic_rerun_compares_passed_profile_fields(monkeypatch, tmp_path: Path):
    source = tmp_path / "robot.xml"
    source.write_text("<mujoco/>\n")
    semantics = _semantic_map(tmp_path / "semantics.json")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("deterministic_bot", source_path=str(source), semantic_map_path=str(semantics))],
    )
    calls = {"count": 0}

    def fake_compile(model, semantic_map_payload, **kwargs):
        calls["count"] += 1
        return _FakeProfile(
            _profile_payload(
                Path(model),
                model_id="deterministic_bot",
                deterministic_hash="stable-hash",
                residual=0.0,
            )
        )

    monkeypatch.setattr(validation_module, "compile_kinematic_profile_v3", fake_compile)

    summary = write_validation_artifacts(
        tmp_path / "artifacts",
        manifest_path=manifest,
        low_discrepancy_count=1,
        deterministic_rerun=True,
    )

    deterministic = summary["deterministic_rerun"]
    model = deterministic["models"]["deterministic_bot"]
    assert calls["count"] == 2
    assert deterministic["status"] == "passed"
    assert deterministic["totals"]["compared_count"] == 1
    assert deterministic["totals"]["matched_count"] == 1
    assert model["status"] == "matched"
    assert model["comparisons"]["deterministic_hash"]["matched"]
    assert model["comparisons"]["status"]["matched"]
    assert model["comparisons"]["rank_summary"]["first"]["torso"]["regular_rank_rotation"] == 1
    assert model["comparisons"]["canonical_projection_residuals"]["matched"]
    assert model["comparisons"]["semantic_site_evidence"]["matched"]


def test_deterministic_rerun_reports_hash_and_projection_mismatches(monkeypatch, tmp_path: Path):
    source = tmp_path / "robot.xml"
    source.write_text("<mujoco/>\n")
    semantics = _semantic_map(tmp_path / "semantics.json")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("nondeterministic_bot", source_path=str(source), semantic_map_path=str(semantics))],
    )
    payloads = [
        _profile_payload(source, model_id="nondeterministic_bot", deterministic_hash="first-hash", residual=0.0),
        _profile_payload(source, model_id="nondeterministic_bot", deterministic_hash="second-hash", residual=0.2),
    ]

    def fake_compile(model, semantic_map_payload, **kwargs):
        return _FakeProfile(payloads.pop(0))

    monkeypatch.setattr(validation_module, "compile_kinematic_profile_v3", fake_compile)

    summary = write_validation_artifacts(
        tmp_path / "artifacts",
        manifest_path=manifest,
        low_discrepancy_count=1,
        deterministic_rerun=True,
    )

    model = summary["deterministic_rerun"]["models"]["nondeterministic_bot"]
    assert summary["deterministic_rerun"]["status"] == "failed"
    assert model["status"] == "mismatched"
    assert not model["comparisons"]["deterministic_hash"]["matched"]
    assert not model["comparisons"]["canonical_projection_residuals"]["matched"]
    assert "motions.neutral.torso.residual" in model["comparisons"]["canonical_projection_residuals"]["mismatch_paths"]


def test_source_unavailable_is_not_counted_as_deterministic_pass(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("missing_bot", source_path=str(tmp_path / "missing.xml"))],
    )

    summary = write_validation_artifacts(
        tmp_path / "artifacts",
        manifest_path=manifest,
        low_discrepancy_count=1,
        deterministic_rerun=True,
    )

    deterministic = summary["deterministic_rerun"]
    model = deterministic["models"]["missing_bot"]
    assert deterministic["status"] == "no_comparable_models"
    assert deterministic["totals"]["compared_count"] == 0
    assert deterministic["totals"]["source_unavailable_count"] == 1
    assert model["status"] == "source_unavailable"
    assert not model["compared"]
    assert "not counted as deterministic pass" in model["reason"]
