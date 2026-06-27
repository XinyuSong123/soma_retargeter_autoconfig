from __future__ import annotations

import ast
from pathlib import Path


def test_step3_3_runtime_code_has_no_model_keyed_solver_tuning_tables() -> None:
    suspicious: list[tuple[str, str]] = []
    for path in _implementation_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [_target_name(target) for target in targets]
            table_name = next((name for name in names if name), "")
            if _suspicious_table_name(table_name) and isinstance(node.value, ast.Dict):
                suspicious.append((str(path), table_name))

    assert suspicious == []


def _implementation_files() -> list[Path]:
    return [
        Path("soma_retargeter/runtime/v3/generic_smoke.py"),
        Path("soma_retargeter/runtime/v3/runtime_quality_gates.py"),
        Path("soma_retargeter/runtime/v3/fleet_harness.py"),
        Path("soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py"),
        Path("scripts/audit_retargeting_v3_step3_3_global_solver_quality.py"),
    ]


def _target_name(target: ast.expr) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _suspicious_table_name(name: str) -> bool:
    lowered = name.lower()
    has_gate_term = any(term in lowered for term in ("threshold", "whitelist", "allowlist", "gate", "weight", "damping"))
    has_model_term = any(term in lowered for term in ("by_model", "by_robot", "model_id", "robot_type", "per_model", "per_robot"))
    return has_gate_term and has_model_term
