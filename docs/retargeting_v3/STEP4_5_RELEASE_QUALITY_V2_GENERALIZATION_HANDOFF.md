# Step 4.5 Release-Quality V2 Generalization Handoff

## 1. Branch and provenance
- Branch: `retargeting-v3-step4-5-release-quality-v2-generalization`
- Baseline Step 4.4 final head: `eebbfc138a7a668ca15219cb243ccaa04b2b2992`
- Baseline Step 4.4 artifact dir: `artifacts/retargeting_v3_step4_4_release_quality_v2_validation/`
- Step 4.5 artifact dir: `artifacts/retargeting_v3_step4_5_release_quality_v2_generalization/`
- Expanded clip manifest hash: `f8ed0032a25a466bf75dd17a6892f7e6995fad9903ccb0b39beabd4abd676971`
- Workflow run id: pending final-head CI
- Workflow conclusion: pending final-head CI
- LFS fsck result: `Git LFS fsck OK`

## 2. Final verdict
- `release_candidate_status`: `BLOCKED_CLIP_GENERALIZATION`
- Primary residual blocker: `BLOCKED_HIPS_ROTATION_RESIDUAL`
- Reason: the nine Step 4.4 candidate-warned rows do not survive the expanded frozen 11-clip suite under the audited worst-clip/p95 guard.
- `promotion_readiness_decision`: `keep_diagnostic_only`

## 3. Expanded frozen clip suite
- Frozen clips: `11` loadable BVH clips under `assets/motions/bvh`.
- Validation rows: `32` full humanoid models x `11` clips = `352` expanded release-quality-v2 validation rows.
- No hard clip removal, no robot-specific clip selection, no threshold weakening.

## 4. Step 4.4 baseline truth preserved
- Legacy runtime counts: `passed = 0`, `warned = 32`, `failed = 0`
- Step 4.4 candidate v2 counts: `passed = 0`, `warned = 9`, `blocked = 23`
- Step 4.4 policy remains diagnostic-only: `global_clip_mean_with_worst_clip_guard_v1_diagnostic`

## 5. Step 4.5 expanded-suite result
- Strict audited policy: `expanded_worst_clip_p95_guard_v1_audited`
- Expanded strict counts: `rows_below_candidate_warn_gate = 0`, `rows_below_candidate_pass_gate = 0`, `release_quality_candidate_blocked_count = 32`
- Delta vs Step 4.4: `rows_below_candidate_warn_gate = -9`, `rows_below_candidate_pass_gate = 0`, `release_quality_candidate_blocked_count = +9`
- Mean aggregation remains diagnostic-only and is not promoted.

## 6. Candidate-warned stability
- Original Step 4.3 candidate-warned rows stable on expanded suite: `0 / 6`
- Newly Step 4.4 candidate-warned rows stable on expanded suite: `0 / 3`
- Dominant blocking clip for all nine audited rows includes `Neutral_throw_ball_001__A057`.
- The PASS_RC condition is not met.

## 7. Task-level residual contract
- Contract artifact: `task_level_residual_contract.json`
- Contract status: `defined_audited_diagnostic_only`
- Task-class weighting counted: `false`
- The contract defines global equal task/component treatment for `torso`, `left_hand`, `right_hand`, `left_foot`, and `right_foot`, while requiring raw residual retention, component decomposition, worst-clip guard, and p95 guard before any future weighting can count.

## 8. Hips/root rotation localization
- Artifact: `hips_root_rotation_residual_decomposition.json`
- Hips dominant model count: `32 / 32`
- Hips rotation dominant model count: `32 / 32`
- Raw residuals remain retained; task-class weighting is not counted.

## 9. Solver convergence weak investigation
- Artifact: `solver_convergence_weak_global_diagnostics.json`
- Weak row count: `1`
- Affected model: `draco3_urdf`
- Pattern: `solver_success_fraction = 0.8` across all `11` clips with the same global solver settings; failed task is `torso` (`Chest` relative to `Hips`) with `unreachable_rank_zero`.
- No robot-specific solver tuning or clip-specific tuning is introduced.

## 10. Audit and tests
- Local Step 4.5 audit: `BLOCKED_CLIP_GENERALIZATION`, `blocking_count = 0`, `finding_count = 0`
- Focused tests: `8 passed`
- Test files added: `test_step4_5_release_quality_v2_generalization_audit.py`, `test_step4_5_release_quality_v2_generalization_blockers.py`, `test_step4_5_release_quality_v2_generalization_contract.py`
- Workflow added: `.github/workflows/retargeting_v3_step4_5_release_quality_v2_generalization.yml`
- Strict final-head CI audit remains pending until workflow evidence is written.

## 11. Prohibitions verified
- No candidate-to-legacy leakage: true
- No threshold weakening: true
- No robot-specific tuning: true
- No hard robot or hard clip removal: true
- No raw residual hiding: true
- No production default mutation: true
- No task-class weighting counted: true

## 12. Next direction
- Attack Hips/root rotation residual directly before revisiting promotion.
- Keep release-quality-v2 mean aggregation diagnostic-only.
- Do not count task-class weighting until the task-level residual contract is promoted by evidence rather than used to hide residuals.
