from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_retargeting_v3_numerical_core import audit


def test_numerical_core_audit_blocks_per_model_baseline_pass_regression(tmp_path: Path):
    root = tmp_path
    for name in ("per_robot", "per_chain", "failures", "test_results"):
        (root / name).mkdir()
    for name in ("environment.json", "commands.txt", "baseline.json", "threshold_calibration.json"):
        (root / name).write_text("{}\n")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "manifest_total": 1,
                "source_available": 1,
                "load_success": 1,
                "profile_eligible": 1,
                "baseline_pass": 1,
                "corrected_pass": 1,
                "baseline_algorithm_failed": 0,
                "corrected_algorithm_failed": 1,
                "epsilon_only_failures_before": 0,
                "epsilon_only_failures_after": 0,
                "semantic_failed_unchanged": 0,
                "model_load_failed_unchanged": 0,
                "source_unavailable_unchanged": 0,
            }
        )
        + "\n"
    )
    (root / "before_after.json").write_text(
        json.dumps(
            {
                "models": {
                    "regressed": {
                        "baseline_status": "passed",
                        "corrected_status": "algorithm_failed",
                        "corrected_epsilon_only": False,
                    }
                }
            }
        )
        + "\n"
    )
    (root / "threshold_calibration.json").write_text(
        json.dumps({"finite_difference": {"scales": ["2h", "h", "h/2"], "backend_aware_epsilon": True}}) + "\n"
    )

    failures = audit(root)

    assert any("regressed to corrected algorithm_failed" in failure for failure in failures)
