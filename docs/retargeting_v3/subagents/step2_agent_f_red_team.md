# Step 2 Agent F Red-Team Handoff

Agent: **F - Reproducibility, CI & Independent Red Team**  
Status: **BLOCKED**  
Worktree: `/mnt/ssd1/song/Desktop/soma_retargeter_autoconfig_agent_f`  
Branch: `agent-f-repro-ci-red-team`

## Commit

Pending at handoff time. Intended commit message:

```text
test: add retargeting v3 step2 acceptance audit
```

## Files

- `.github/workflows/retargeting_v3_step2.yml`
- `scripts/audit_retargeting_v3_step2.py`
- `tests/v3/test_acceptance_gates_audit.py`
- `docs/retargeting_v3/STEP2_ACCEPTANCE_LEDGER.md`
- `STEP2_IMPLEMENTATION_REPORT.md`
- `docs/retargeting_v3/subagents/step2_agent_f_red_team.md`
- `artifacts/retargeting_v3_step2/test_results/acceptance_audit.json`
- `artifacts/retargeting_v3_step2/test_results/acceptance_audit.junit.xml`
- `artifacts/retargeting_v3_step2/test_results/pytest_acceptance_gates.junit.xml`

## Commands and Results

```bash
python scripts/audit_retargeting_v3_step2.py --artifact-dir artifacts/retargeting_v3_step2 --source-root . --output-json artifacts/retargeting_v3_step2/test_results/acceptance_audit.json --junit-xml artifacts/retargeting_v3_step2/test_results/acceptance_audit.junit.xml
```

Result: **BLOCKED**, exit code 1, 105 blocking findings.

```bash
python -m pytest tests/v3/test_acceptance_gates_*.py
```

Result: **FAILED**, exit code 1, 1 failed.

```bash
python -m pytest tests/v3/test_acceptance_gates_*.py --junitxml=artifacts/retargeting_v3_step2/test_results/pytest_acceptance_gates.junit.xml
```

Result: **FAILED**, exit code 1, 1 failed, JUnit written.

## Numeric Results

| Gate | Count |
|---|---:|
| hardcoded zero calibration | 11 |
| neutral-to-neutral fake projection | 8 |
| zero-offset RPO hand/sole | 5 |
| TALOS proximal foot mapping | 2 |
| Booster Hips=Chest hiding waist | 1 |
| arbitrary G1 equivalence | 2 |
| dirty artifact metadata | 2 |
| absolute cache paths | 24 |
| inferred semantics confidence=1 | 46 |
| rank0 false pass | 3 |
| robot-name special cases | 1 |
| legacy offsets | 0 |

Total: **105** blocking findings.

## Assumptions

- Agent F does not implement A-E core feature design.
- Current artifacts are allowed to fail; the purpose is to prevent false-positive Step-2 acceptance.
- The clean acceptance target is zero blocking audit findings plus passing pytest acceptance gates.

## Failures

- Current artifacts are generated from dirty metadata and an older git head.
- Current projection reports do not provide per-canonical-motion projection evidence.
- Current semantic evidence is insufficient for inferred maps and distal endpoints.
- Current G1 equivalence remains a documented limitation while the summary is green.

## Risks

- The CI workflow is intentionally red until A-E close the blockers.
- Future artifact schema changes should preserve the same evidence requirements rather than bypassing checks by renaming fields.

## Integration Order

1. Merge Agent F audit gate after reviewing that it touches only owned files.
2. Integrate A-E fixes and regenerate artifacts from a clean commit.
3. Rerun the audit and pytest gate.
4. Update `STEP2_ACCEPTANCE_LEDGER.md`, `STEP2_IMPLEMENTATION_REPORT.md`, and test-result artifacts only after the gate returns PASS.
