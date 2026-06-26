from __future__ import annotations

import json
from pathlib import Path


ARTIFACT_ROOT = Path("artifacts/retargeting_v3_step3_runtime_quality")


def test_step3_generic_harness_matrix_covers_all_required_runtime_modes() -> None:
    matrix = json.loads((ARTIFACT_ROOT / "generic_smoke_matrix.json").read_text())
    rows = matrix["rows"]

    full = [row for row in rows if row["category"] == "full_humanoid_profile"]
    partial = [row for row in rows if row["category"] == "partial_humanoid_profile"]
    negative = [row for row in rows if row["category"] == "negative_control"]

    assert len(full) == 64
    assert len(partial) == 12
    assert len(negative) == 9
    assert all(row["status"] == "passed" for row in full)
    assert all(row["status"] == "passed" for row in partial)
    assert all(row["status"] == "negative_control_rejected" for row in negative)


def test_step3_generic_harness_writes_per_model_quality_metrics() -> None:
    model_matrix = json.loads((ARTIFACT_ROOT / "model_matrix.json").read_text())
    for row in model_matrix["rows"]:
        path = ARTIFACT_ROOT / "per_model" / row["model_id"] / "quality_metrics.json"
        assert path.exists(), row["model_id"]
        metrics = json.loads(path.read_text())
        assert metrics.get("frame_count", metrics.get("output_frame_count", 0)) > 0
        assert metrics.get("output_nan_count", metrics.get("nan_count", 0)) == 0
        assert metrics.get("output_inf_count", metrics.get("inf_count", 0)) == 0
