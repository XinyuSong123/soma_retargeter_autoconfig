from __future__ import annotations

import json
from pathlib import Path

import soma_retargeter.robotics.v3.validation as validation_module
from soma_retargeter.robotics.v3.robot_zoo import allowed_status_values
from soma_retargeter.robotics.v3.robot_zoo import model_load_failure_diagnostic
from soma_retargeter.robotics.v3.target_builder import CANONICAL_MOTION_NAMES
from soma_retargeter.robotics.v3.validation import write_validation_artifacts


def _write_manifest(path: Path, models: list[dict]) -> Path:
    payload = {
        "schema_version": 1,
        "catalog_name": "test-zoo",
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


def test_manifest_entries_all_materialize_structured_source_unavailable(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry("missing_positive", source_path=str(tmp_path / "missing.xml")),
            _entry(
                "missing_negative",
                source_path=str(tmp_path / "missing-negative.xml"),
                robot_class="single_arm",
                expected_capability="negative_control",
            ),
        ],
    )

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)

    assert summary["manifest"]["model_count"] == 2
    assert set(summary["reports"]) == {"missing_positive", "missing_negative"}
    assert summary["status_counts"] == {"source_unavailable": 2}
    assert summary["source_unavailable_count"] == 2
    assert summary["license_blocked_count"] == 0
    assert summary["failure_artifact_status_counts"] == {}
    assert summary["failure_artifacts_count"] == 0
    assert [path.name for path in (tmp_path / "artifacts" / "failures").glob("*.json")] == ["_no_failures.json"]
    assert "compiled_count" not in summary
    assert "compiled" not in summary["manifest"]["allowed_statuses"]
    for model_id in summary["reports"]:
        report = json.loads((tmp_path / "artifacts" / "per_robot" / f"{model_id}.json").read_text())
        assert report["status"] == "source_unavailable"
        assert report["status"] in allowed_status_values()
        assert report["model"]["manifest"]["manifest_model_id"] == model_id
        assert report["model"]["local_file_sha256"]["status"] == "unavailable"
        assert report["reproduction_command"].startswith("python -m soma_retargeter.tools.compile_kinematic_profile_v3")


def test_status_enum_matches_goal_contract():
    assert set(allowed_status_values()) == {
        "passed",
        "capability_limited_passed",
        "partial_passed",
        "negative_control_passed",
        "algorithm_failed",
        "semantic_failed",
        "model_load_failed",
        "source_unavailable",
        "license_blocked",
    }
    assert "compiled" not in allowed_status_values()


def test_license_blocked_is_not_a_failure_artifact(tmp_path: Path):
    source = tmp_path / "blocked.xml"
    source.write_text("<mujoco/>\n")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("blocked_positive", source_path=str(source), license_blocked=True)],
    )

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)
    report = json.loads((tmp_path / "artifacts" / "per_robot" / "blocked_positive.json").read_text())

    assert summary["status_counts"] == {"license_blocked": 1}
    assert summary["license_blocked_count"] == 1
    assert summary["failure_artifact_status_counts"] == {}
    assert summary["failure_artifacts_count"] == 0
    assert [path.name for path in (tmp_path / "artifacts" / "failures").glob("*.json")] == ["_no_failures.json"]
    assert report["status"] == "license_blocked"
    assert report["model"]["source_resolution"]["status"] == "license_blocked"
    assert report["model"]["local_file_sha256"]["status"] == "unavailable"


def test_reproduction_commands_do_not_contain_local_absolute_paths(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("missing_positive", source_path=str(tmp_path / "missing.xml"))],
    )

    write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)

    commands = (tmp_path / "artifacts" / "commands.txt").read_text()
    assert str(tmp_path) not in commands
    assert "/mnt/ssd1/" not in commands
    assert "${RETARGETING_V3_ARTIFACTS}/per_robot/missing_positive.json" in commands
    assert "--robot-id missing_positive" in commands


def test_artifacts_do_not_leak_absolute_manifest_paths(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("missing_positive", source_path=str(tmp_path / "missing.xml"))],
    )

    write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)

    artifact_text = "\n".join(
        path.read_text()
        for path in sorted((tmp_path / "artifacts").rglob("*"))
        if path.is_file() and path.suffix in {".json", ".txt"}
    )
    assert str(tmp_path) not in artifact_text
    assert "/mnt/ssd1/" not in artifact_text
    assert "${LOCAL_SOURCE_PATH}/missing.xml" in artifact_text


def test_resolved_source_loader_exception_is_model_load_failed(monkeypatch, tmp_path: Path):
    semantic_map = _semantic_map(tmp_path / "verified_semantics.json")
    source = tmp_path / "broken.xml"
    source.write_text("<not-a-model/>\n")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("broken_positive", source_path=str(source), semantic_map_path=str(semantic_map))],
    )

    def fail_semantic_map(*args, **kwargs):
        raise RuntimeError("loader failed")

    monkeypatch.setattr(validation_module, "_semantic_map_for_entry", fail_semantic_map)

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)
    report = json.loads((tmp_path / "artifacts" / "per_robot" / "broken_positive.json").read_text())

    assert summary["status_counts"] == {"model_load_failed": 1}
    assert summary["model_load_failed_count"] == 1
    assert summary["failure_artifact_status_counts"] == {"model_load_failed": 1}
    assert summary["failure_artifacts_count"] == 1
    assert report["status"] == "model_load_failed"
    assert report["failures"] == ["RuntimeError: loader failed"]
    assert isinstance(report["model"]["local_file_sha256"], str)
    assert len(report["model"]["local_file_sha256"]) == 64
    assert report["model"]["local_file_sha256"] == report["model"]["source_resolution"]["local_file_sha256"]
    taxonomy = report["failure_taxonomy"]["model_load"]["runtime_loader_exception"]
    assert taxonomy["classification"] == "model_load_failed"
    assert taxonomy["local_fix_available"] is False
    assert taxonomy["reproduction_command"] == report["reproduction_command"]
    failure_report = json.loads((tmp_path / "artifacts" / "failures" / "broken_positive.json").read_text())
    assert failure_report["status"] == "model_load_failed"


def test_model_load_failure_taxonomy_sanitizes_missing_asset_paths(monkeypatch, tmp_path: Path):
    semantic_map = _semantic_map(tmp_path / "verified_semantics.json")
    source = tmp_path / "robot.xml"
    missing_mesh = tmp_path / "meshes" / "base_link_0.obj"
    source.write_text("<mujoco/>\n")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("asset_missing_positive", source_path=str(source), semantic_map_path=str(semantic_map))],
    )

    def fail_semantic_map(*args, **kwargs):
        raise ValueError(f"string is not a file: `{missing_mesh}`")

    monkeypatch.setattr(validation_module, "_semantic_map_for_entry", fail_semantic_map)
    monkeypatch.setattr(validation_module, "_validation_checks", lambda output_dir: {})

    write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)
    report = json.loads((tmp_path / "artifacts" / "per_robot" / "asset_missing_positive.json").read_text())
    artifact_text = "\n".join(
        path.read_text()
        for path in sorted((tmp_path / "artifacts").rglob("*"))
        if path.is_file() and path.suffix in {".json", ".txt", ".xml"}
    )

    assert report["status"] == "model_load_failed"
    assert str(tmp_path) not in json.dumps(report)
    assert str(tmp_path) not in artifact_text
    assert "${WORKSPACE}" not in json.dumps(report)
    assert report["failure_taxonomy"]["model_load"]["missing_referenced_asset"]["kind"] == "missing_referenced_asset"
    assert "${LOCAL_SOURCE_PATH}/base_link_0.obj" in artifact_text
    assert "${HOME}" not in json.dumps(report)


def test_dae_decoder_failures_are_classified_as_pycollada_blockers():
    diagnostic = model_load_failure_diagnostic("no decoder found for mesh file robot/meshes/hip.dae")

    assert diagnostic["kind"] == "missing_optional_dependency_pycollada"
    assert diagnostic["classification"] == "model_load_failed"
    assert diagnostic["local_fix_available"] is False


def test_missing_verified_semantic_map_is_semantic_failed_without_inference(monkeypatch, tmp_path: Path):
    source = tmp_path / "robot.xml"
    source.write_text("<mujoco/>\n")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("unmapped_positive", source_path=str(source))],
    )

    def fail_if_compiled(*args, **kwargs):
        raise AssertionError("missing verified semantic maps must not reach profile compilation")

    monkeypatch.setattr(validation_module, "compile_kinematic_profile_v3", fail_if_compiled)

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)
    report = json.loads((tmp_path / "artifacts" / "per_robot" / "unmapped_positive.json").read_text())

    assert summary["status_counts"] == {"semantic_failed": 1}
    assert summary["semantic_failed_count"] == 1
    assert summary["failure_artifact_status_counts"] == {"semantic_failed": 1}
    assert report["status"] == "semantic_failed"
    assert report["semantic_map_resolution"]["status"] == "missing"
    assert report["semantic_map_resolution"]["source"] == "verified_semantic_map"
    assert report["failure_taxonomy"]["semantic"]["verified_semantic_map"]["kind"] == "missing_verified_semantic_map"
    assert "inferred_from_newton_body_names" not in json.dumps(report)


def test_profile_failures_are_classified_and_keep_source_hash(monkeypatch, tmp_path: Path):
    semantic_map = _semantic_map(tmp_path / "verified_semantics.json")
    algorithm_source = tmp_path / "algorithm.xml"
    semantic_source = tmp_path / "semantic.xml"
    algorithm_source.write_text("<mujoco/>\n")
    semantic_source.write_text("<mujoco/>\n")
    sources = {
        "algorithm_bot": algorithm_source,
        "semantic_bot": semantic_source,
    }
    failures = {
        "algorithm_bot": ["projection residual gate failed"],
        "semantic_bot": ["missing required semantics: LeftHand, RightHand"],
    }
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry("algorithm_bot", source_path=str(algorithm_source), semantic_map_path=str(semantic_map)),
            _entry("semantic_bot", source_path=str(semantic_source), semantic_map_path=str(semantic_map)),
        ],
    )

    class FakeProfile:
        def __init__(self, model_id: str):
            self.model_id = model_id

        def to_json(self):
            return {
                "schema_version": 3,
                "model": {
                    "id": self.model_id,
                    "path": str(sources[self.model_id]),
                    "format": "mjcf",
                    "backend": "newton",
                },
                "runtime_adapter": {},
                "failures": failures[self.model_id],
                "warnings": [],
                "capability_status": "full_humanoid_ready",
            }

    def fake_compile(model, semantic_map_payload, **kwargs):
        return FakeProfile(kwargs["model_id"])

    monkeypatch.setattr(validation_module, "compile_kinematic_profile_v3", fake_compile)

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)

    assert summary["status_counts"] == {"algorithm_failed": 1, "semantic_failed": 1}
    assert summary["algorithm_failed_count"] == 1
    assert summary["semantic_failed_count"] == 1
    assert summary["failure_artifact_status_counts"] == {"algorithm_failed": 1, "semantic_failed": 1}
    assert summary["failure_artifacts_count"] == 2
    assert sorted(path.stem for path in (tmp_path / "artifacts" / "failures").glob("*.json")) == [
        "algorithm_bot",
        "semantic_bot",
    ]
    for model_id, status in {"algorithm_bot": "algorithm_failed", "semantic_bot": "semantic_failed"}.items():
        report = json.loads((tmp_path / "artifacts" / "per_robot" / f"{model_id}.json").read_text())
        failure_report = json.loads((tmp_path / "artifacts" / "failures" / f"{model_id}.json").read_text())
        assert report["status"] == status
        assert failure_report["status"] == status
        assert isinstance(report["model"]["local_file_sha256"], str)
        assert len(report["model"]["local_file_sha256"]) == 64
        assert report["model"]["local_file_sha256"] == report["model"]["source_resolution"]["local_file_sha256"]


def test_numerical_stability_gate_failure_is_algorithm_failed(monkeypatch, tmp_path: Path):
    semantic_map = _semantic_map(tmp_path / "verified_semantics.json")
    source = tmp_path / "unstable.xml"
    source.write_text("<mujoco/>\n")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("unstable_bot", source_path=str(source), semantic_map_path=str(semantic_map))],
    )

    class FakeProfile:
        def to_json(self):
            return _unstable_epsilon_profile_payload(source, model_id="unstable_bot")

    def fake_compile(model, semantic_map_payload, **kwargs):
        return FakeProfile()

    monkeypatch.setattr(validation_module, "compile_kinematic_profile_v3", fake_compile)

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)
    report = json.loads((tmp_path / "artifacts" / "per_robot" / "unstable_bot.json").read_text())
    failure_report = json.loads((tmp_path / "artifacts" / "failures" / "unstable_bot.json").read_text())

    assert summary["status_counts"] == {"algorithm_failed": 1}
    assert summary["algorithm_failed_count"] == 1
    assert summary["failure_artifact_status_counts"] == {"algorithm_failed": 1}
    assert report["status"] == "algorithm_failed"
    assert report["status_reason"] == "numerical stability gate failed for task(s): left_hand"
    assert "numerical stability gate failed: left_hand has numerical_stability_gate_passed=false" in report["failures"]
    assert failure_report["status"] == "algorithm_failed"
    taxonomy = report["failure_taxonomy"]["algorithm"]["numerical_stability"]
    assert taxonomy["classification"] == "algorithm_failed"
    assert taxonomy["tasks"][0]["task"] == "left_hand"
    assert taxonomy["tasks"][0]["false_gate_paths"] == [
        "rank_stability.left_hand.numerical_stability_gate_passed",
        "rank_stability.left_hand.sample_diagnostics[0].numerical_stability_gate_passed",
    ]


def test_partial_profile_with_failed_epsilon_stability_is_not_partial_passed(monkeypatch, tmp_path: Path):
    semantic_map = _semantic_map(tmp_path / "verified_semantics.json")
    source = tmp_path / "partial-unstable.xml"
    source.write_text("<mujoco/>\n")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("partial_unstable_bot", source_path=str(source), semantic_map_path=str(semantic_map))],
    )

    class FakeProfile:
        def to_json(self):
            payload = _unstable_epsilon_profile_payload(source, model_id="partial_unstable_bot")
            payload["capability_status"] = "partial_humanoid"
            return payload

    def fake_compile(model, semantic_map_payload, **kwargs):
        return FakeProfile()

    monkeypatch.setattr(validation_module, "compile_kinematic_profile_v3", fake_compile)

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)
    report = json.loads((tmp_path / "artifacts" / "per_robot" / "partial_unstable_bot.json").read_text())

    assert summary["status_counts"] == {"algorithm_failed": 1}
    assert "partial_passed" not in summary["status_counts"]
    assert report["status"] == "algorithm_failed"
    assert report["failure_taxonomy"]["algorithm"]["numerical_stability"]["status"] == "failed"


def test_verified_semantic_map_path_is_loaded_without_inference(monkeypatch, tmp_path: Path):
    source = tmp_path / "robot.xml"
    source.write_text("<mujoco/>\n")
    semantic_map = tmp_path / "verified_semantics.json"
    semantic_map.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "semantics": {
                    "Hips": {"body": "pelvis", "source": "verified_map", "confidence": 0.99},
                    "Chest": {"body": "chest", "source": "verified_map", "confidence": 0.99},
                },
            }
        )
        + "\n"
    )
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("verified_positive", source_path=str(source), semantic_map_path=str(semantic_map))],
    )
    captured = {}

    class FakeProfile:
        failures = []

        def to_json(self):
            exact_torso = {
                "status": "converged",
                "converged": True,
                "normalized_residual": 0.0,
                "residual": 0.0,
                "capability_certificate": {
                    "certificate_class": "exact_reachable",
                    "gates": {
                        "exact_threshold_passed": True,
                        "projected_gradient_kkt": True,
                        "seed_consensus": True,
                        "primal_feasible": True,
                        "continuation": True,
                        "joint_limits": True,
                        "numerical": True,
                        "residual_explained": True,
                    },
                },
            }
            return {
                "schema_version": 3,
                "model": {
                    "id": "verified_positive",
                    "path": str(source),
                    "format": "mjcf",
                    "backend": "newton",
                },
                "runtime_adapter": {},
                "chains": {"torso": {}},
                "canonical_projection_reports": {
                    "motion_order": list(CANONICAL_MOTION_NAMES),
                    "motions": {
                        motion: {"tasks": {"torso": exact_torso}}
                        for motion in CANONICAL_MOTION_NAMES
                    },
                },
                "failures": [],
                "warnings": [],
                "capability_status": "full_humanoid_ready",
            }

    def fake_compile(model, semantic_map_payload, **kwargs):
        captured["model"] = model
        captured["semantic_map"] = semantic_map_payload
        captured["kwargs"] = kwargs
        return FakeProfile()

    monkeypatch.setattr(validation_module, "compile_kinematic_profile_v3", fake_compile)

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)
    report = json.loads((tmp_path / "artifacts" / "per_robot" / "verified_positive.json").read_text())
    semantic_artifact = json.loads((tmp_path / "artifacts" / "semantic_maps" / "verified_positive.json").read_text())

    assert summary["status_counts"] == {"passed": 1}
    assert captured["semantic_map"]["Hips"]["body"] == "pelvis"
    assert report["semantic_map_resolution"]["source"] == "verified_semantic_map"
    assert semantic_artifact["resolution"]["source"] == "verified_semantic_map"
    assert str(tmp_path) not in json.dumps(report)


def test_cross_format_and_deterministic_scaffolds_are_written(monkeypatch, tmp_path: Path):
    validation_checks = {
        "g1_mjcf_urdf_equivalence": {
            "status": "source_unavailable",
            "gate_a_status": "blocked",
            "gate_a_evidence_complete": False,
            "reason": "test fixture has no fixed same-source G1 URDF",
            "evidence_incomplete_reasons": {"source": "missing"},
        }
    }
    monkeypatch.setattr(validation_module, "_validation_checks", lambda output_dir: validation_checks)
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry("toy_urdf", format="urdf", source_path=str(tmp_path / "toy.urdf")),
            _entry("toy_mjcf", format="mjcf", source_path=str(tmp_path / "toy.xml")),
        ],
    )

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)

    cross_format = json.loads((tmp_path / "artifacts" / "cross_format.json").read_text())
    deterministic = json.loads((tmp_path / "artifacts" / "deterministic_rerun.json").read_text())
    same_source = cross_format["gates"]["same_source_strict"]

    assert cross_format["schema_version"] == 2
    assert cross_format["pairs"]["toy"]["status"] == "not_run"
    assert same_source["status"] == "blocked"
    assert same_source["validation_check_status"] == "source_unavailable"
    assert same_source["status"] not in {"not_run", "passed"}
    assert cross_format["gates"]["variant_compatibility"]["status"] == "blocked"
    assert deterministic["status"] == "not_run"
    assert set(deterministic["models"]) == {"toy_urdf", "toy_mjcf"}
    assert summary["validation_checks"] == validation_checks
    assert summary["cross_format"] == cross_format


def test_cross_format_same_source_gate_records_incomplete_evidence_from_validation_check(monkeypatch, tmp_path: Path):
    validation_checks = {
        "g1_mjcf_urdf_equivalence": {
            "status": "incomplete",
            "mode": "same_source_urdf_to_canonical_mjcf",
            "strict_equivalent": True,
            "gate_a_status": "incomplete",
            "gate_a_evidence_complete": False,
            "gate_a_required_sections": [
                "semantic_fk",
                "active_chains",
                "rank_summary",
                "canonical_projection",
            ],
            "evidence_statuses": {
                "semantic_fk": "unavailable",
                "active_chains": "unavailable",
                "rank_summary": "unavailable",
                "canonical_projection": "unavailable",
            },
            "evidence_incomplete_reasons": {
                "semantic_fk": "semantic_map_not_provided",
                "active_chains": "semantic_sites_unavailable",
                "rank_summary": "semantic_sites_unavailable",
                "canonical_projection": "canonical_projection_reports_not_provided",
            },
            "differences": {},
        }
    }
    monkeypatch.setattr(validation_module, "_validation_checks", lambda output_dir: validation_checks)
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry("toy_urdf", format="urdf", source_path=str(tmp_path / "toy.urdf")),
            _entry("toy_mjcf", format="mjcf", source_path=str(tmp_path / "toy.xml")),
        ],
    )

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)

    cross_format = json.loads((tmp_path / "artifacts" / "cross_format.json").read_text())
    same_source = cross_format["gates"]["same_source_strict"]
    assert same_source["status"] == "incomplete"
    assert same_source["status"] not in {"not_run", "passed"}
    assert same_source["validation_check_status"] == "incomplete"
    assert same_source["strict_equivalent"]
    assert same_source["gate_a_status"] == "incomplete"
    assert not same_source["gate_a_evidence_complete"]
    assert same_source["evidence_statuses"]["semantic_fk"] == "unavailable"
    assert same_source["evidence_incomplete_reasons"]["canonical_projection"] == (
        "canonical_projection_reports_not_provided"
    )
    assert cross_format["gates"]["variant_compatibility"]["status"] == "blocked"
    assert summary["validation_checks"] == validation_checks


def test_cross_format_same_source_gate_passes_only_complete_evidence(tmp_path: Path):
    reports = {
        "toy_urdf": {
            "status": "passed",
            "robot_class": "humanoid",
            "expected_capability": "positive",
            "required": True,
            "redistribution": "kinematic_snapshot",
        },
        "toy_mjcf": {
            "status": "passed",
            "robot_class": "humanoid",
            "expected_capability": "positive",
            "required": True,
            "redistribution": "kinematic_snapshot",
        },
    }
    validation_checks = {
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
    }

    cross_format = validation_module._cross_format_report(
        reports,
        tmp_path / "per_robot",
        validation_checks=validation_checks,
    )

    same_source = cross_format["gates"]["same_source_strict"]
    assert same_source["status"] == "passed"
    assert same_source["validation_check_status"] == "passed"
    assert same_source["gate_a_status"] == "complete_passed"
    assert same_source["gate_a_evidence_complete"]
    assert cross_format["gates"]["variant_compatibility"]["status"] == "failed"


def _unstable_epsilon_profile_payload(source: Path, *, model_id: str) -> dict:
    return {
        "schema_version": 3,
        "model": {
            "id": model_id,
            "path": str(source),
            "format": "mjcf",
            "backend": "newton",
        },
        "runtime_adapter": {},
        "failures": [],
        "warnings": [],
        "capability_status": "full_humanoid_ready",
        "rank_stability": {
            "left_hand": {
                "regular_rank_translation": 2,
                "nominal_rank_translation": 2,
                "regular_rank_rotation": 1,
                "nominal_rank_rotation": 1,
                "epsilon_stability_gate_passed": False,
                "numerical_stability_gate_passed": False,
                "epsilon_unstable_columns": [3],
                "epsilon_unstable_fraction": 0.25,
                "epsilon_unstable_sample_fraction": 0.5,
                "sample_diagnostics": [
                    {
                        "sample_index": 0,
                        "epsilon_stability_gate_passed": False,
                        "numerical_stability_gate_passed": False,
                        "unstable_columns": [3],
                        "column_classifications": [{"class": "unstable_nonsmooth"}],
                    }
                ],
            }
        },
    }
