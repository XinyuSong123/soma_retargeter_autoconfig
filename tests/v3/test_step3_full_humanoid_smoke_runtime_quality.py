from __future__ import annotations

import json
from pathlib import Path


ARTIFACT_ROOT = Path("artifacts/retargeting_v3_step3_runtime_quality")


def test_step3_full_humanoids_enter_target_stream_and_generic_smoke() -> None:
    model_matrix = json.loads((ARTIFACT_ROOT / "model_matrix.json").read_text())
    target_matrix = json.loads((ARTIFACT_ROOT / "target_stream_matrix.json").read_text())
    smoke_matrix = json.loads((ARTIFACT_ROOT / "generic_smoke_matrix.json").read_text())

    full_ids = {row["model_id"] for row in model_matrix["rows"] if row["category"] == "full_humanoid_profile"}
    target_ids = {row["model_id"] for row in target_matrix["rows"] if row["category"] == "full_humanoid_profile"}
    smoke_ids = {row["model_id"] for row in smoke_matrix["rows"] if row["category"] == "full_humanoid_profile"}

    assert len(full_ids) == 32
    assert full_ids <= target_ids
    assert full_ids <= smoke_ids
    assert all(row["final_step3_1_status"] == "runtime_quality_passed" for row in model_matrix["rows"] if row["model_id"] in full_ids)
