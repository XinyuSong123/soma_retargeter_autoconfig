from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step2 import (
    GOAL_FALSE_POSITIVE_GATES,
    GOAL_FALSE_POSITIVE_TRACEABILITY,
    run_audit,
)


def test_retargeting_v3_step2_acceptance_gates_are_clean():
    result = run_audit(
        artifact_dir=Path("artifacts/retargeting_v3_step2"),
        source_root=Path("."),
    )

    missing_gate_coverage = [gate for gate in GOAL_FALSE_POSITIVE_GATES if gate not in result.gate_counts]
    assert missing_gate_coverage == []
    missing_traceability = [
        false_positive
        for false_positive, gates in GOAL_FALSE_POSITIVE_TRACEABILITY.items()
        if not all(gate in result.gate_counts for gate in gates)
    ]
    assert missing_traceability == []
    assert result.blocking_count == 0, _format_findings(result.blocking_findings)


def _format_findings(findings) -> str:
    lines = [f"{len(findings)} Step-2 acceptance blocker(s):"]
    lines.extend(f"- {finding.gate}: {finding.subject}: {finding.message}" for finding in findings[:40])
    if len(findings) > 40:
        lines.append(f"- ... {len(findings) - 40} more")
    return "\n".join(lines)
