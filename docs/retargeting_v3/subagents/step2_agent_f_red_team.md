# Step 2 Agent F Red-Team Handoff

Agent: **F - Reproducibility, CI & Independent Red Team**  
Reasoning strength: **xhigh**  
Status: **BLOCKED**  
Worktree: `/mnt/ssd1/song/Desktop/soma_retargeter_autoconfig`  
Branch: `agent-b-xhigh-verified-semantics`

## Input Reviewed

- `goal.md`, including the xhigh subagent requirement and the ten listed false positives.
- `docs/retargeting_v3/STEP1_MATHEMATICAL_SPEC.md`
- `docs/retargeting_v3/STEP1_COMPLETE_MODEL_CROSSCHECK.md`
- `docs/retargeting_v3/THREE_STEP_ROADMAP.md`
- `docs/retargeting_v3/CODE_AUDIT.md`
- Prior Agent F worktree `/mnt/ssd1/song/Desktop/soma_retargeter_autoconfig_agent_f`
- Prior Agent F commits:
  - `67f726b test: add retargeting v3 step2 acceptance audit`
  - `84f5439 docs: record agent f audit commit`

## Files

- `.github/workflows/retargeting_v3_step2.yml`
- `scripts/audit_retargeting_v3_step2.py`
- `tests/v3/test_acceptance_gates_audit.py`
- `docs/retargeting_v3/STEP2_ACCEPTANCE_LEDGER.md`
- `docs/retargeting_v3/STEP2_IMPLEMENTATION_REPORT.md`
- `docs/retargeting_v3/subagents/step2_agent_f_red_team.md`
- `artifacts/retargeting_v3_step2/test_results/acceptance_audit.json`
- `artifacts/retargeting_v3_step2/test_results/acceptance_audit.junit.xml`
- `artifacts/retargeting_v3_step2/test_results/pytest_acceptance_gates.junit.xml`

## Commands and Results

```bash
python scripts/audit_retargeting_v3_step2.py \
  --artifact-dir artifacts/retargeting_v3_step2 \
  --source-root . \
  --output-json artifacts/retargeting_v3_step2/test_results/acceptance_audit.json \
  --junit-xml artifacts/retargeting_v3_step2/test_results/acceptance_audit.junit.xml
```

Result: **BLOCKED**, exit code 1, 7 blocking findings.

```bash
python -m pytest tests/v3/test_acceptance_gates_*.py \
  --junitxml=artifacts/retargeting_v3_step2/test_results/pytest_acceptance_gates.junit.xml
```

Result: **FAILED**, exit code 1, 1 failed, 0 passed. JUnit written.

## Numeric Results

| Gate | Count |
|---|---:|
| hardcoded zero calibration | 0 |
| neutral-to-neutral fake projection | 0 |
| canonical targets without projection | 1 |
| zero-offset RPO hand/sole | 0 |
| body-name depth heuristic | 0 |
| TALOS proximal foot mapping | 0 |
| Booster Hips=Chest hiding waist | 0 |
| arbitrary G1 equivalence | 1 |
| dirty artifact metadata | 2 |
| absolute cache paths | 0 |
| inferred semantics confidence=1 | 0 |
| rank0 false pass | 0 |
| robot-name special cases | 2 |
| legacy offsets | 0 |
| missing formal red-team artifacts | 1 |

Total: **7** blocking findings.

## Assumptions

- Agent F remains independent and does not implement Agent A-E core compiler, semantic, calibration, projection, or validation design.
- The current `artifacts/retargeting_v3_step2` working tree at `6d4488c` is the acceptance target for this red-team pass.
- It is valid for corrected gates to have zero current findings when coverage remains explicit; `body_name_depth_heuristic` is covered and currently clean in source.
- The missing Agent B handoff remains a blocker; Agent F reports it but does not fabricate it.

## Failures

- `roboparty_rpo_local` has canonical targets without per-motion projection reports.
- G1 equivalence is missing or non-strict in `validation_checks.json`.
- Artifact metadata is dirty and records an older generation commit.
- `semantic_sites.py` still contains RPO-specific default semantic-map logic.
- Agent B handoff is missing.

## Risks

- The CI workflow is intentionally red until A-E close the blockers.
- Future artifact schema changes should preserve the same evidence requirements rather than bypassing checks by renaming fields.
- The audit is read-only and prevents false-positive acceptance; it does not prove a corrected implementation.

## Integration Order

1. Merge Agent F audit gate after confirming it touches only Agent F-owned files.
2. Integrate A-E fixes and complete the missing Agent B handoff.
3. Regenerate artifacts from a clean final commit.
4. Rerun the audit and pytest gate.
5. Update ledger, implementation report, handoff, and test-result artifacts only after the gate returns PASS.
