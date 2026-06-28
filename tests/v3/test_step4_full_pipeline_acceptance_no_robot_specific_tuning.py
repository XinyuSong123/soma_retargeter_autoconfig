from __future__ import annotations

import ast
from pathlib import Path


def test_step4_source_has_no_model_keyed_tuning_tables() -> None:
    paths = [
        Path("soma_retargeter/runtime/v3/generic_smoke.py"),
        Path("soma_retargeter/tools/run_v3_full_pipeline_acceptance.py"),
        Path("scripts/audit_retargeting_v3_step4_full_pipeline_acceptance.py"),
    ]
    suspicious_names = {"per_robot_thresholds", "by_robot_weights", "model_id_thresholds", "accepted_models"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                assert names.isdisjoint(suspicious_names)
