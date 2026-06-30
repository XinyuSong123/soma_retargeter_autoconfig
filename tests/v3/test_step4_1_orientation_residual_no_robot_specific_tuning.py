from __future__ import annotations

import ast
from pathlib import Path


def test_step4_1_source_has_no_robot_specific_threshold_tables() -> None:
    suspicious_names = {"per_robot_thresholds", "by_robot_weights", "model_id_thresholds", "accepted_models"}
    for path in (
        Path("soma_retargeter/tools/step4_1_orientation_residual.py"),
        Path("scripts/audit_retargeting_v3_step4_1_orientation_residual_breakthrough.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                assert names.isdisjoint(suspicious_names)

