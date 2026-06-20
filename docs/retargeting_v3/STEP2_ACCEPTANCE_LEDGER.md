# Step 2 Acceptance Ledger

Status: **BLOCKED**  
Owner: Agent F - Reproducibility, CI & Independent Red Team  
Reasoning strength: **xhigh**  
Audit artifact: `artifacts/retargeting_v3_step2/test_results/acceptance_audit.json`

## Gate Results

| Gate | Blocking findings | Current result |
|---|---:|---|
| hardcoded zero calibration | 9 | BLOCKED |
| neutral-to-neutral fake projection | 8 | BLOCKED |
| canonical targets without projection | 9 | BLOCKED |
| zero-offset RPO hand/sole | 5 | BLOCKED |
| body-name depth heuristic | 0 | PASS |
| TALOS proximal foot mapping | 2 | BLOCKED |
| Booster Hips=Chest hiding waist | 1 | BLOCKED |
| arbitrary G1 equivalence | 2 | BLOCKED |
| dirty artifact metadata | 2 | BLOCKED |
| absolute cache paths | 24 | BLOCKED |
| inferred semantics confidence=1 | 46 | BLOCKED |
| rank0 false pass | 3 | BLOCKED |
| robot-name special cases | 2 | BLOCKED |
| legacy offsets | 0 | PASS |
| missing formal red-team artifacts | 3 | BLOCKED |

Total: **116 blocking findings**.

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

## Evidence Summary

- All nine per-robot reports record zero calibration errors without independent measurement evidence.
- Eight reports have top-level `projection_reports` where every task has `desired == projected` and `residual == 0`.
- Nine reports include canonical target or motion data without per-motion torso/hand/foot projection reports.
- RPO hand and foot endpoint sites remain body-origin local positions, and `validation_checks.json` treats the zero-offset hand endpoint as `passed_with_documented_scope`.
- TALOS feet map to `leg_left_1_link` and `leg_right_1_link`.
- Booster T1 maps both `Hips` and `Chest` to `Trunk`.
- G1 URDF/MJCF equivalence remains non-strict while `summary.json` reports nine compiled robots and zero failure artifacts.
- `environment.json` records dirty/HEAD-mismatched metadata.
- `commands.txt`, per-robot `reproduction_command`, and `model.path` fields contain local absolute paths.
- Inferred semantic maps are emitted as `explicit_semantic_override` with `confidence: 1.0` in checked-in artifacts.
- Rank-zero projection entries report zero residual without unreachable-demand evidence.
- V3 core source still contains robot-name/default semantic special cases.
- Agent B, C, and D handoff documents are missing.

## Commands

```bash
python scripts/audit_retargeting_v3_step2.py \
  --artifact-dir artifacts/retargeting_v3_step2 \
  --source-root . \
  --output-json artifacts/retargeting_v3_step2/test_results/acceptance_audit.json \
  --junit-xml artifacts/retargeting_v3_step2/test_results/acceptance_audit.junit.xml
```

Result: **BLOCKED**, exit code 1, 116 blockers.

```bash
python -m pytest tests/v3/test_acceptance_gates_*.py \
  --junitxml=artifacts/retargeting_v3_step2/test_results/pytest_acceptance_gates.junit.xml
```

Result: **FAILED**, 1 failed, 0 passed. The failure is expected until the Step-2 false positives are closed.

## Acceptance Rule

Step 2 may not be reported as complete until `scripts/audit_retargeting_v3_step2.py`
returns `PASS` with zero blocking findings and `tests/v3/test_acceptance_gates_*.py`
passes without xfail, skip, or lowered thresholds.
