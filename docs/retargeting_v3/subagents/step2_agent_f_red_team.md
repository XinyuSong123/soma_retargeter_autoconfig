# Step 2 Agent F Red-Team Handoff

Agent: **F - Reproducibility, CI & Independent Red Team**  
Reasoning strength: **xhigh**  
Current status: **SUPERSEDED HISTORICAL HANDOFF; STEP 2 ACCEPTANCE BLOCKED**
Branch recorded by historical handoff: `agent-b-xhigh-verified-semantics`

## Current Status Overlay

This file previously recorded a false-positive audit `PASS`. That record is
now superseded. The current Step 2 status is:

- Formal pytest artifact:
  `artifacts/retargeting_v3_step2/test_results/pytest.txt`.
- Pytest result: `209 passed, 10 skipped`.
- Acceptance artifact:
  `artifacts/retargeting_v3_step2/acceptance_ledger.json`.
- Acceptance result: `BLOCKED`, `blocking_count=3`.
- Remaining hard blockers:
  `arbitrary_g1_equivalence`,
  `cross_format.gates.same_source_strict`,
  and `cross_format.gates.variant_compatibility`.

Agent F must not be cited as a final full-Step-2 `PASS`. A final red-team pass
is still required after the current acceptance blockers are closed.

All current, continued, or newly launched Step 2 subagents must use `xhigh`
reasoning strength and record that fact in handoff artifacts.

## Current Blocking Findings

| Gate | Subject | Current status |
|---|---|---|
| `arbitrary_g1_equivalence` | `validation_checks.g1_mjcf_urdf_equivalence` | `incomplete`; not a strict G1 equivalence pass. |
| `cross_format_gates_not_run` | `cross_format.gates.same_source_strict` | `incomplete`; Gate A semantic/projection evidence is missing. |
| `cross_format_gates_not_run` | `cross_format.gates.variant_compatibility` | `blocked`; Gate B variant compatibility evidence is missing. |

## Current Numeric Results

```text
pytest: 209 passed, 10 skipped
acceptance_ledger.status: BLOCKED
acceptance_ledger.blocking_count: 3
robot_zoo_status_counts:
  passed=7
  negative_control_passed=4
  algorithm_failed=14
  semantic_failed=3
  model_load_failed=2
  source_unavailable=16
deterministic_rerun.status: passed
```

## Superseded Historical Record

The following details are retained for provenance only and are **not current
status**.

Historical status: **FALSE-POSITIVE AUDIT PASS; FULL STEP 2 NOT COMPLETE**

Historical input reviewed:

- `goal.md`, including the xhigh subagent requirement and false-positive list.
- Step 1 mathematical specification, complete-model cross-check, roadmap, and
  code audit.
- Prior Agent F audit work and the integrated A-E xhigh changes.

Historical files:

- `.github/workflows/retargeting_v3_step2.yml`
- `scripts/audit_retargeting_v3_step2.py`
- `tests/v3/test_acceptance_gates_audit.py`
- `docs/retargeting_v3/STEP2_ACCEPTANCE_LEDGER.md`
- `docs/retargeting_v3/STEP2_IMPLEMENTATION_REPORT.md`
- `docs/retargeting_v3/subagents/step2_agent_f_red_team.md`
- `artifacts/retargeting_v3_step2/test_results/acceptance_audit.json`
- `artifacts/retargeting_v3_step2/test_results/acceptance_audit.junit.xml`

Historical command result:

```bash
python scripts/audit_retargeting_v3_step2.py \
  --artifact-dir artifacts/retargeting_v3_step2 \
  --source-root . \
  --output-json artifacts/retargeting_v3_step2/test_results/acceptance_audit.json \
  --junit-xml artifacts/retargeting_v3_step2/test_results/acceptance_audit.junit.xml
```

Historical result: **PASS**, exit code 0, 0 blocking findings.

```bash
python -m pytest -q
```

Historical result: **124 passed, 10 skipped**.

The historical red-team conclusion said the false-positive gate was clean but
full Step 2 remained incomplete because most Robot Zoo sources were unavailable
and deterministic rerun was not executed. That conclusion is superseded by the
current artifacts above: pytest is now green at 209/10, deterministic rerun is
present, and acceptance remains blocked on G1/cross-format evidence.
