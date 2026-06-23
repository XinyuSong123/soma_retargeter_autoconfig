from __future__ import annotations

import json
from pathlib import Path

from scripts.extract_capability_failure_ledger import main
from soma_retargeter.robotics.v3.failure_analysis import build_baseline_failure_ledger


FAILURE_DIR = Path("artifacts/retargeting_v3_step2_assets44/failures")
LEDGER_PATH = Path("artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json")


def test_committed_capability_baseline_ledger_matches_source_reports() -> None:
    expected = build_baseline_failure_ledger(FAILURE_DIR)
    committed = json.loads(LEDGER_PATH.read_text())

    assert committed == expected
    assert committed["counts"]["failed_rows"] == 50
    assert committed["counts"]["failure_rows_by_metric_type"]["projection_position_normalized_residual"] == 43
    assert committed["counts"]["failure_rows_by_metric_type"]["projection_rotation_normalized_residual"] == 2
    assert committed["counts"]["failure_rows_by_metric_type"]["numerical_stability_gate"] == 5


def test_capability_failure_ledger_cli_check_mode() -> None:
    assert main(["--failure-dir", str(FAILURE_DIR), "--output", str(LEDGER_PATH), "--check"]) == 0
