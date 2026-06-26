from __future__ import annotations

import json
from pathlib import Path


ARTIFACT_ROOT = Path("artifacts/retargeting_v3_step3_runtime_quality")


def test_step3_pipeline_backed_controls_retain_rpo_g1_regression_matrix() -> None:
    payload = json.loads((ARTIFACT_ROOT / "pipeline_backed_matrix.json").read_text())
    controls = payload["controls"]
    rows = payload["rows"]

    assert len(rows) == 12
    assert payload["status_counts"] == {"fail_closed": 2, "passed": 10}
    assert controls["rpo_present"] is True
    assert controls["g1_present"] is True
    assert controls["shadow_noop_verified"] is True
    assert controls["g1_fail_closed_recorded"] is True


def test_step3_pipeline_controls_are_linked_from_every_model_row() -> None:
    matrix = json.loads((ARTIFACT_ROOT / "model_matrix.json").read_text())
    for row in matrix["rows"]:
        assert row["pipeline_control_id"] == "pipeline_backed_matrix.json"
        assert {"disabled", "shadow"} <= set(row["control_modes"])
        assert row["legacy_default_unchanged"] is True
        assert row["shadow_noop_verified"] is True
        assert row["override_explicit_only"] is True
        assert row["fingerprint_gate_enforced"] is True
