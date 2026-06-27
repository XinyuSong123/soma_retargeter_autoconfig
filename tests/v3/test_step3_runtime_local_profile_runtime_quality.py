from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from soma_retargeter.robotics.v3.model_fingerprint import sha256_file
from soma_retargeter.runtime.v3 import runtime_local_profile as runtime_local_profile_module
from soma_retargeter.runtime.v3.fleet_harness import FleetCaseResult
from soma_retargeter.runtime.v3.fleet_inventory import load_fleet_runtime_cases
from soma_retargeter.runtime.v3.fleet_inventory import FULL_HUMANOID_PROFILE, NEGATIVE_CONTROL, stable_payload_hash
from soma_retargeter.runtime.v3.runtime_local_profile import close_runtime_profile, write_profile_resolution_artifacts


class _FakeProfile:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_json(self) -> dict:
        return json.loads(json.dumps(self._payload))


def _first_case(category: str):
    return next(case for case in load_fleet_runtime_cases() if case.category == category)


def _forced_mismatch_identity(case) -> dict:
    model = case.profile.get("model", {}) if isinstance(case.profile.get("model"), dict) else {}
    profile_source_sha = str(model.get("local_file_sha256") or "")
    runtime_source_sha = "0" * 64 if profile_source_sha != "0" * 64 else "1" * 64
    runtime_fingerprint = f"forced-runtime-fingerprint:{case.model_id}"
    assert runtime_fingerprint != str(model.get("fingerprint") or "")
    assert runtime_source_sha != profile_source_sha
    return {
        "source_sha256": runtime_source_sha,
        "fingerprint": runtime_fingerprint,
        "nq": 7,
        "nv": 6,
        "body_count": 3,
        "error": None,
    }


def _install_fake_runtime_local_compiler(monkeypatch: pytest.MonkeyPatch, identity: dict) -> None:
    def fake_compile(model_path, semantic_map, *, model_id=None, model_format=None, **_kwargs):
        assert semantic_map
        return _FakeProfile(
            {
                "schema_version": 3,
                "status": "passed",
                "capability_status": "full_humanoid_ready",
                "model": {
                    "id": model_id,
                    "format": model_format,
                    "backend": "newton",
                    "path": str(model_path),
                    "fingerprint": identity["fingerprint"],
                },
                "runtime_adapter": {"backend": "newton", "nq": identity["nq"], "nv": identity["nv"]},
                "semantic_sites": {},
                "canonical_targets": {"neutral": {"transforms": {}}},
                "canonical_projection_reports": {"motion_order": [], "motions": {}},
                "failures": [],
                "warnings": [],
            }
        )

    def fake_write_profile(profile: _FakeProfile, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(runtime_local_profile_module, "compile_kinematic_profile_v3", fake_compile)
    monkeypatch.setattr(runtime_local_profile_module, "write_profile", fake_write_profile)


def test_step3_1_runtime_profile_closure_is_terminal_for_all_rows(tmp_path: Path) -> None:
    cases = load_fleet_runtime_cases()
    closures = [close_runtime_profile(case, artifact_root=tmp_path) for case in cases]

    assert len(closures) == 44
    statuses = {closure.resolution_status for closure in closures}
    assert "runtime_model_load_failed" not in statuses
    assert "negative_control_rejected" in statuses
    assert all(
        closure.resolution_status in {
            "profile_match",
            "runtime_local_profile_generated",
            "runtime_local_profile_failed",
            "structured_partial_supported",
            "negative_control_rejected",
        }
        for closure in closures
    )


def test_step3_1_runtime_profile_closure_writes_matrix_and_summary(tmp_path: Path) -> None:
    cases = load_fleet_runtime_cases()
    closures = [close_runtime_profile(case, artifact_root=tmp_path) for case in cases]
    matrix, summary = write_profile_resolution_artifacts(artifact_root=tmp_path, closures=closures)

    assert matrix["row_count"] == 44
    assert summary["row_count"] == 44
    assert summary["negative_control_rejected_count"] == 9
    assert summary["structured_partial_supported_count"] == 3
    assert (tmp_path / "profile_resolution_matrix.json").exists()
    assert (tmp_path / "runtime_local_profile_summary.json").exists()
    for case in cases:
        payload = json.loads((tmp_path / "per_model" / case.model_id / "profile_resolution.json").read_text())
        assert payload["model_id"] == case.model_id
        assert not Path(payload["runtime_source_path"]).is_absolute()


def test_forced_full_humanoid_fingerprint_mismatch_generates_runtime_local_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _first_case(FULL_HUMANOID_PROFILE)
    assert case.semantic_map_path is not None
    identity = _forced_mismatch_identity(case)
    before_profile = case.profile_path.read_bytes()
    before_summary = Path("artifacts/retargeting_v3_step2_capability/summary.json").read_bytes()
    monkeypatch.setattr(runtime_local_profile_module, "_runtime_identity", lambda _case: identity)
    _install_fake_runtime_local_compiler(monkeypatch, identity)

    closure = close_runtime_profile(case, artifact_root=tmp_path, low_discrepancy_count=1)

    assert closure.resolution_status == "runtime_local_profile_generated"
    assert closure.fingerprint_match is False
    assert closure.source_hash_match is False
    assert closure.runtime_local_profile_path is not None
    payload = json.loads(closure.runtime_local_profile_path.read_text(encoding="utf-8"))
    metadata = payload["runtime_local_profile"]
    assert metadata["source_profile_id"] == case.model_id
    assert metadata["runtime_source_sha256"] == identity["source_sha256"]
    assert metadata["runtime_fingerprint"] == identity["fingerprint"]
    assert metadata["semantic_map_sha256"] == sha256_file(case.semantic_map_path)
    assert payload["model"]["local_file_sha256"] == identity["source_sha256"]
    assert payload["deterministic_hash"] == stable_payload_hash(
        {key: value for key, value in payload.items() if key != "deterministic_hash"}
    )
    assert case.profile_path.read_bytes() == before_profile
    assert Path("artifacts/retargeting_v3_step2_capability/summary.json").read_bytes() == before_summary


def test_forced_negative_control_mismatch_does_not_generate_humanoid_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_case = _first_case(NEGATIVE_CONTROL)
    base_model = base_case.profile.get("model", {}) if isinstance(base_case.profile.get("model"), dict) else {}
    profile = {
        **base_case.profile,
        "model": {
            **base_model,
            "fingerprint": "step2-negative-control-fingerprint",
            "local_file_sha256": "a" * 64,
        },
    }
    case = replace(base_case, profile=profile)
    identity = _forced_mismatch_identity(case)
    monkeypatch.setattr(runtime_local_profile_module, "_runtime_identity", lambda _case: identity)
    monkeypatch.setattr(
        runtime_local_profile_module,
        "compile_kinematic_profile_v3",
        lambda *_args, **_kwargs: pytest.fail("negative controls must not compile runtime-local humanoid profiles"),
    )

    closure = close_runtime_profile(case, artifact_root=tmp_path, low_discrepancy_count=1)

    assert closure.resolution_status == "negative_control_rejected"
    assert closure.fingerprint_match is False
    assert closure.source_hash_match is False
    assert closure.runtime_local_profile_path is None
    assert not (tmp_path / "runtime_local_profiles" / f"{case.model_id}.json").exists()


def test_runtime_local_profile_failure_blocks_pass_status_in_model_matrix() -> None:
    case = _first_case(FULL_HUMANOID_PROFILE)
    result = FleetCaseResult(
        case=case,
        clip_results=[],
        target_stream_status="passed",
        generic_smoke_status="passed",
        negative_control_status="not_applicable",
        final_status="runtime_quality_passed",
        quality_metrics={},
        failures=[],
    )

    row = result.model_matrix_row(
        profile_resolution_status="runtime_local_profile_failed",
        pipeline_backed_status="not_pipeline_backed",
    )

    assert row["final_step3_1_status"] in {"blocked_source_or_profile", "runtime_quality_failed"}
    assert row["runtime_quality_status"] in {"blocked_source_or_profile", "runtime_quality_failed"}
    assert row["quality_classification"] in {"blocked_source_or_profile", "runtime_quality_failed"}
    assert row["final_step3_1_status"] != "runtime_quality_passed"
    assert row["profile_resolution_blocked"] is True
    assert row["quality_evaluated"] is False
    assert row["humanoid_profile_generated"] is False
