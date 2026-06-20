# Step 2 Acceptance Ledger

Status: **BLOCKED**  
Owner: Agent F - Reproducibility, CI & Independent Red Team  
Reasoning strength: **xhigh**  
Audit artifact: `artifacts/retargeting_v3_step2/test_results/acceptance_audit.json`

## Gate Results

| Gate | Blocking findings | Current result |
|---|---:|---|
| hardcoded zero calibration | 0 | PASS |
| neutral-to-neutral fake projection | 0 | PASS |
| canonical targets without projection | 1 | BLOCKED |
| zero-offset RPO hand/sole | 0 | PASS |
| body-name depth heuristic | 0 | PASS |
| TALOS proximal foot mapping | 0 | PASS |
| Booster Hips=Chest hiding waist | 0 | PASS |
| arbitrary G1 equivalence | 1 | BLOCKED |
| dirty artifact metadata | 2 | BLOCKED |
| absolute cache paths | 0 | PASS |
| inferred semantics confidence=1 | 0 | PASS |
| rank0 false pass | 0 | PASS |
| robot-name special cases | 2 | BLOCKED |
| legacy offsets | 0 | PASS |
| missing formal red-team artifacts | 1 | BLOCKED |

Total: **7 blocking findings**.

## Traceability To `goal.md` False Positives

| `goal.md` false positive | Audit coverage |
|---|---|
| neutral FK projected back to itself | `neutral_to_neutral_fake_projection` |
| hardcoded zero rest calibration | `hardcoded_zero_calibration` |
| RPO Hand/Foot body-origin sites | `zero_offset_rpo_hand_sole` |
| canonical targets not fed into real projection | `canonical_targets_without_projection` |
| body-name string length used as depth | `body_name_depth_heuristic` |
| inferred semantics marked explicit/confidence=1 | `inferred_semantics_confidence_one` |
| TALOS foot and Booster trunk semantic errors | `talos_proximal_foot_mapping`, `booster_hips_chest_alias` |
| G1 URDF/MJCF difference accepted as limitation | `arbitrary_g1_equivalence` |
| dirty/old-HEAD artifacts and absolute paths | `dirty_artifact_metadata`, `absolute_cache_paths` |
| missing pytest/JUnit/report/handoff/red-team evidence | `missing_formal_red_team_artifacts` |

## Current Blocking Evidence

- `roboparty_rpo_local` has `canonical_targets` without per-motion torso/hand/foot projection reports.
- `validation_checks.g1_mjcf_urdf_equivalence` is missing or non-strict, so G1 equivalence is not a pass.
- `environment.json` records artifacts generated from a dirty worktree and a git head different from the audited checkout.
- `semantic_sites.py` still contains `default_rpo_semantic_map()` and the RPO-specific verified-map path.
- Agent B handoff is missing.

## Commands

```bash
python scripts/audit_retargeting_v3_step2.py \
  --artifact-dir artifacts/retargeting_v3_step2 \
  --source-root . \
  --output-json artifacts/retargeting_v3_step2/test_results/acceptance_audit.json \
  --junit-xml artifacts/retargeting_v3_step2/test_results/acceptance_audit.junit.xml
```

Result: **BLOCKED**, exit code 1, 7 blockers.

```bash
python -m pytest tests/v3/test_acceptance_gates_*.py \
  --junitxml=artifacts/retargeting_v3_step2/test_results/pytest_acceptance_gates.junit.xml
```

Result: **FAILED**, 1 failed, 0 passed. The failure is expected until the Step-2 false positives are closed.

## Acceptance Rule

Step 2 may not be reported as complete until `scripts/audit_retargeting_v3_step2.py`
returns `PASS` with zero blocking findings and `tests/v3/test_acceptance_gates_*.py`
passes without xfail, skip, or lowered thresholds.
