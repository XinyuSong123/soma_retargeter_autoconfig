from __future__ import annotations

import json
from pathlib import Path


ARTIFACT_ROOT = Path("artifacts/retargeting_v3_step3_runtime_quality")


def test_step3_negative_controls_are_rejected_not_promoted() -> None:
    matrix = json.loads((ARTIFACT_ROOT / "model_matrix.json").read_text())
    rows = [row for row in matrix["rows"] if row["category"] == "negative_control"]

    assert len(rows) == 9
    for row in rows:
        assert row["expected_capability"] == "negative_control"
        assert row["negative_control_status"] == "negative_control_rejected"
        assert row["final_step3_1_status"] == "negative_control_runtime_passed"
        assert row["humanoid_profile_generated"] is False
        assert row["promoted_to_runtime_quality"] is False
        assert row["override_allowed"] is False


def test_step3_negative_control_rejection_is_deterministic() -> None:
    deterministic = json.loads((ARTIFACT_ROOT / "deterministic_rerun.json").read_text())
    assert deterministic["deterministic"] is True
    assert deterministic["deterministic_compared_count"] == 44
    assert deterministic["deterministic_matched_count"] == 44
