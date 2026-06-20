# Step 2 Acceptance Ledger

Status: **FALSE-POSITIVE AUDIT PASS; FULL STEP 2 NOT COMPLETE**  
Owner: Agent F - Reproducibility, CI & Independent Red Team  
Reasoning strength: **xhigh**  
Audit artifact: `artifacts/retargeting_v3_step2/test_results/acceptance_audit.json`

## Gate Results

| Gate | Blocking findings | Current result |
|---|---:|---|
| hardcoded zero calibration | 0 | PASS |
| neutral-to-neutral fake projection | 0 | PASS |
| canonical targets without projection | 0 | PASS |
| zero-offset RPO hand/sole | 0 | PASS |
| body-name depth heuristic | 0 | PASS |
| TALOS proximal foot mapping | 0 | PASS |
| Booster Hips=Chest hiding waist | 0 | PASS |
| arbitrary G1 equivalence | 0 | PASS |
| dirty artifact metadata | 0 | PASS |
| absolute cache paths | 0 | PASS |
| inferred semantics confidence=1 | 0 | PASS |
| rank0 false pass | 0 | PASS |
| robot-name special cases | 0 | PASS |
| legacy offsets | 0 | PASS |
| missing formal red-team artifacts | 0 | PASS |

Total: **0 blocking findings**.

## Current Evidence

- `scripts/audit_retargeting_v3_step2.py` returns `PASS`.
- `tests/v3/test_acceptance_gates_audit.py` passes.
- G1 same-source URDF to canonical MJCF strict equivalence is recorded as `passed` in `validation_checks.json`.
- `roboparty_rpo_local` includes per-motion canonical projection reports and passes the no-fetch validation run.
- Full Step 2 is still not complete: the current no-fetch Robot Zoo run reports `passed=1` and `source_unavailable=45`, and deterministic rerun remains `not_run`.

## Commands

```bash
python scripts/audit_retargeting_v3_step2.py \
  --artifact-dir artifacts/retargeting_v3_step2 \
  --source-root . \
  --output-json artifacts/retargeting_v3_step2/test_results/acceptance_audit.json \
  --junit-xml artifacts/retargeting_v3_step2/test_results/acceptance_audit.junit.xml
```

Result: **PASS**, exit code 0, 0 blockers.

## Acceptance Rule

The false-positive audit gate is clean. The overall Step 2 goal may not be
reported complete until all required Robot Zoo sources are resolved or
legitimately classified, required positive humanoids run algorithm gates, and
the deterministic rerun artifact is produced.
