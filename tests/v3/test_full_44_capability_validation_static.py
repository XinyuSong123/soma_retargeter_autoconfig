from __future__ import annotations

import json
from pathlib import Path


ARTIFACT_DIR = Path("artifacts/retargeting_v3_step2_capability")


def _load(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text())


def test_capability_artifact_records_44_model_terminal_counts() -> None:
    summary = _load("summary.json")
    before_after = _load("before_after.json")
    deterministic = _load("deterministic_rerun.json")

    assert summary["manifest_total"] == 46
    assert summary["in_scope_total"] == 44
    assert summary["deferred_snapshot_count"] == 2
    assert summary["status_counts"] == {
        "capability_limited_passed": 23,
        "negative_control_passed": 9,
        "partial_passed": 3,
        "passed": 9,
    }
    assert summary["algorithm_pass_count"] == 44
    assert summary["deterministic_compared"] == 44
    assert summary["deterministic_matched"] == 44
    assert deterministic["status"] == "passed"
    assert deterministic["totals"]["compared_count"] == 44
    assert deterministic["totals"]["matched_count"] == 44
    assert before_after["status_transition_counts"]["partial_passed->capability_limited_passed"] == 23
    assert before_after["status_transition_counts"]["partial_passed->partial_passed"] == 3


def test_capability_artifact_cross_format_gate_uses_humanoid_capability_only() -> None:
    cross_format = _load("cross_format.json")
    variant = cross_format["gates"]["variant_compatibility"]

    assert cross_format["gates"]["same_source_strict"]["status"] == "passed"
    assert variant["status"] == "passed"
    assert variant["eligible_pair_count"] == 7
    assert variant["not_eligible_pair_count"] == 2
    assert variant["pair_statuses"]["cassie"]["status"] == "not_eligible"
    assert variant["pair_statuses"]["unitree_go2"]["status"] == "not_eligible"
    assert "negative_control" in variant["pair_statuses"]["cassie"]["reason"]
    assert "negative_control" in variant["pair_statuses"]["unitree_go2"]["reason"]
    assert "shared_task_certificates" in variant["pair_statuses"]["unitree_h1"]["evidence"]
    assert variant["pair_statuses"]["unitree_h1"]["evidence"]["shared_task_certificates"]["status"] == "passed"


def test_capability_limited_reports_have_specific_reasons_and_compact_certificates() -> None:
    summary = _load("summary.json")
    limited_ids = sorted(
        model_id
        for model_id, row in summary["reports"].items()
        if row["status"] == "capability_limited_passed"
    )
    assert limited_ids == [
        "apptronik_apollo_mjcf",
        "atlas_drc_urdf",
        "atlas_v4_urdf",
        "booster_t1_mjcf",
        "booster_t1_urdf",
        "draco3_urdf",
        "elf2_mjcf",
        "elf2_urdf",
        "ergocub_urdf",
        "fourier_n1_mjcf",
        "fourier_n1_urdf",
        "jaxon_urdf",
        "jvrc_mjcf",
        "jvrc_urdf",
        "pal_talos_mjcf_direct",
        "robonaut2_urdf",
        "roboparty_rpo_local",
        "robotis_op3_mjcf",
        "talos_urdf",
        "toddlerbot_2xc_mjcf",
        "toddlerbot_2xm_mjcf",
        "toddlerbot_urdf",
        "valkyrie_urdf",
    ]
    for model_id in limited_ids:
        report = json.loads((ARTIFACT_DIR / "per_robot" / f"{model_id}.json").read_text())
        assert report["status_reason"] != report["status"]
        assert "available task certificate" in report["status_reason"]
        assert report["task_certificate_summary"]["status"] in {"available", "unavailable"}
        assert "per_task" in report["task_certificate_summary"]


def test_structured_partial_reports_remain_terminal_but_not_capability_limited() -> None:
    summary = _load("summary.json")
    partial_ids = sorted(
        model_id for model_id, row in summary["reports"].items() if row["status"] == "partial_passed"
    )
    assert partial_ids == [
        "berkeley_humanoid_mjcf_direct",
        "sigmaban_urdf",
        "simple_humanoid_urdf",
    ]
    for model_id in partial_ids:
        report = json.loads((ARTIFACT_DIR / "per_robot" / f"{model_id}.json").read_text())
        assert "structured partial" in report["status_reason"]
        assert report["task_certificate_summary"]["status"] == "unavailable"
