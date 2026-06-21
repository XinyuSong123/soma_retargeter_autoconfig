# Step 2 Acceptance Ledger

Status: **BLOCKED**  
Owner: Agent F - Reproducibility, CI & Independent Red Team  
Audit artifact: `artifacts/retargeting_v3_step2/test_results/acceptance_audit.json`

## Gate Results

| Gate | Blocking findings | Current result |
|---|---:|---|
| hardcoded zero calibration | 11 | BLOCKED |
| neutral-to-neutral fake projection | 8 | BLOCKED |
| zero-offset RPO hand/sole | 5 | BLOCKED |
| TALOS proximal foot mapping | 2 | BLOCKED |
| Booster Hips=Chest hiding waist | 1 | BLOCKED |
| arbitrary G1 equivalence | 2 | BLOCKED |
| dirty artifact metadata | 2 | BLOCKED |
| absolute cache paths | 24 | BLOCKED |
| inferred semantics confidence=1 | 46 | BLOCKED |
| rank0 false pass | 3 | BLOCKED |
| robot-name special cases | 1 | BLOCKED |
| legacy offsets | 0 | PASS |

Total: **105 blocking findings**.

## Evidence Summary

- `rest_frames.py` still directly initializes `pos_errors` and `rot_errors` to zero, and all nine per-robot reports record zero calibration errors without independent measurement evidence.
- Eight reports have top-level `projection_reports` where every task has `desired == projected` and `residual == 0` without per-canonical-motion projection reports.
- RPO hand and foot endpoint sites remain body-origin local positions, and `validation_checks.json` treats the zero-offset hand endpoint as `passed_with_documented_scope`.
- TALOS feet map to `leg_left_1_link` and `leg_right_1_link`.
- Booster T1 maps both `Hips` and `Chest` to `Trunk`.
- G1 URDF/MJCF equivalence remains `documented_limitation` while `summary.json` reports nine compiled robots and zero failure artifacts.
- `environment.json` records a dirty worktree and a git head different from the audited checkout.
- `commands.txt`, per-robot `reproduction_command`, and `model.path` fields contain local absolute cache paths.
- Inferred semantic maps are emitted as `explicit_semantic_override` with `confidence: 1.0`.
- Rank-zero projection entries report zero residual without unreachable-demand evidence.
- V3 core source still contains a default RPO semantic mapping helper.

## Commands

```bash
python scripts/audit_retargeting_v3_step2.py \
  --artifact-dir artifacts/retargeting_v3_step2 \
  --source-root . \
  --output-json artifacts/retargeting_v3_step2/test_results/acceptance_audit.json \
  --junit-xml artifacts/retargeting_v3_step2/test_results/acceptance_audit.junit.xml
```

Result: **BLOCKED**, exit code 1, 105 blockers.

```bash
python -m pytest tests/v3/test_acceptance_gates_*.py \
  --junitxml=artifacts/retargeting_v3_step2/test_results/pytest_acceptance_gates.junit.xml
```

Result: **FAILED**, 1 failed, 0 passed. The failure is expected until the Step-2 false positives are closed.

## Acceptance Rule

Step 2 may not be reported as complete until `scripts/audit_retargeting_v3_step2.py` returns `PASS` with zero blocking findings and `tests/v3/test_acceptance_gates_*.py` passes without xfail, skip, or lowered thresholds.
