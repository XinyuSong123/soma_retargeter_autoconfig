# Step 4.4 Release-Quality V2 Validation Handoff

## 1. Branch and provenance
- Branch: `retargeting-v3-step4-4-release-quality-v2-validation`
- Final HEAD: `eebbfc138a7a668ca15219cb243ccaa04b2b2992`
- Baseline Step 4.3 branch: `retargeting-v3-step4-3-normalized-residual-gate-reconciliation`
- Baseline Step 4.3 artifact dir: `artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation/`
- Step 4.4 artifact dir: `artifacts/retargeting_v3_step4_4_release_quality_v2_validation/`
- Workflow run id: `28740291675`
- Workflow conclusion: `success`
- Job conclusions: `step4-4-static-and-unit=success`, `step4-4-artifact-audit=success`, `step4-4-release-quality-v2-validation-smoke=success`, `step4-4-candidate-stability-smoke=success`, `step4-4-export-and-temporal-smoke=success`, `step4-4-lfs-and-snapshot-smoke=success`, `step4-4-pipeline-controls-reference=success`, `step4-4-final-head-ci-evidence=success`
- LFS fsck result: `Git LFS fsck OK`

## 2. Final verdict
- `release_candidate_status`: `PASS_RC`
- `verdict`: `PASS`
- `reason`: candidate-warned coverage expanded globally from 6 to 9 and candidate-blocked count decreased from 26 to 23, while hard invariants remained intact.
- `promotion_readiness_decision`: `keep_diagnostic_only`

## 3. Baseline Step 4.3 truth
- Legacy runtime counts: `passed = 0`, `warned = 32`, `failed = 0`, `high_residual_warning_count = 32`
- Candidate v2 counts: `passed = 0`, `warned = 6`, `blocked = 26`
- Candidate-warned rows: `fourier_n1_mjcf`, `fourier_n1_urdf`, `robotis_op3_mjcf`, `toddlerbot_2xc_mjcf`, `toddlerbot_2xm_mjcf`, `toddlerbot_urdf`
- Preserved invariants: `44` rows, `32/3/9` partition, `solver_backed_count = 32`, `residual_only_count = 0`, `deterministic = 44/44`

## 4. Step 4.4 counts
- Legacy runtime counts: `passed = 0`, `warned = 32`, `failed = 0`
- Candidate v2 counts: `passed = 0`, `warned = 9`, `blocked = 23`
- Candidate-passed rows: none
- Candidate-warned rows: `elf2_mjcf`, `elf2_urdf`, `fourier_n1_mjcf`, `fourier_n1_urdf`, `roboparty_rpo_local`, `robotis_op3_mjcf`, `toddlerbot_2xc_mjcf`, `toddlerbot_2xm_mjcf`, `toddlerbot_urdf`
- Candidate-blocked rows: remaining 23 full humanoid rows under the selected diagnostic aggregation policy
- Deltas vs Step 4.3: `rows_below_candidate_warn_gate +3`, `rows_below_candidate_pass_gate +0`, `release_quality_candidate_blocked_count -3`, `runtime_quality_failed_count +0`

## 5. Global changes attempted
- `global_clip_aggregation_policy`: global, accepted for diagnostic expansion, changed candidate-warned rows from 6 to 9.
- `global_worst_clip_guard`: global, accepted as a guard for the diagnostic aggregation policy.
- `global_task_class_residual_weighting`: global, rejected before counting because no audited task-level release contract exists.
- `global_solver_retry_line_search_refinement`: global, existing Step 4.3 evidence reused; no new solver rerun counted.
- `global_temporal_consistency_residual_penalty`: global, diagnostic no-op because temporal diagnostics are finite and no row is promoted by it.

## 6. Candidate stability
- Candidate-warned stability: the original six Step 4.3 candidate-warned rows are stable across all four clips.
- Candidate-passed stability: no candidate-passed rows exist.
- Worst-clip guard: worst-clip-only status remains at the Step 4.3 six-row warned baseline.
- Clip generalization: the selected global aggregation improves candidate warned coverage only with the worst-clip guard active; it is kept diagnostic-only.

## 7. Blocker taxonomy
- Primary blocker categories: `orientation_integrated_fixed_global_scale_residual_p95_above_candidate_warn_gate = 26`, `task_class_dominance = 26`, `clip_instability = 6`, `normalization_ambiguity = 8`, `solver_convergence_weak = 1`
- Affected rows: the 26 Step 4.3 blocked full humanoid rows are covered by `release_quality_v2_blocker_taxonomy.json`.
- Recommended next global fix: validate a larger clip suite and task-level residual contract before counting task-class weighting or promoting the diagnostic aggregation policy.

## 8. Stress tests
- Deterministic rerun: `44/44` matched.
- Clip leave-one-out: represented in per-clip stability and mean/p95/max sensitivity diagnostics.
- Worst-clip: worst-clip-only policy keeps `rows_below_candidate_warn_gate = 6`.
- Normalization integrity: fixed-global denominator, no denominator inflation, no raw residual hiding.
- Temporal/support/contact/collision: `128` finite temporal diagnostics, `128` support/contact diagnostics, `128` collision proxy diagnostics.

## 9. Audit and tests
- Strict audit result: `PASS_RC`, `blocking_count = 0`; local non-strict audit is also `PASS_RC` with `blocking_count = 0`.
- Focused tests result: `63 passed` for Step 4.4/4.3/4.2 focused tests; Step 4.4-only collection has 22 tests.
- Test files added: `test_step4_4_release_quality_v2_validation_audit.py`, `test_step4_4_release_quality_v2_validation_artifact_schema.py`, `test_step4_4_release_quality_v2_validation_candidate_counts.py`, `test_step4_4_release_quality_v2_validation_stress.py`, `test_step4_4_release_quality_v2_validation_promotion_readiness.py`, `test_step4_4_release_quality_v2_validation_no_robot_specific_tuning.py`, `test_step4_4_release_quality_v2_validation_status_semantics.py`, `test_step4_4_release_quality_v2_validation_determinism.py`
- Workflow jobs: `step4-4-static-and-unit`, `step4-4-artifact-audit`, `step4-4-release-quality-v2-validation-smoke`, `step4-4-candidate-stability-smoke`, `step4-4-export-and-temporal-smoke`, `step4-4-lfs-and-snapshot-smoke`, `step4-4-pipeline-controls-reference`, `step4-4-final-head-ci-evidence`

## 10. Prohibitions verified
- Legacy gates unchanged: true
- No candidate-to-legacy leakage: true
- No robot-specific tuning: true
- No threshold lowering: true
- No whitelist/blacklist: true
- No hard row/clip removal: true
- No denominator inflation: true
- No production default mutation: true
- No visual/deployment claim: true

## 11. Known limitations
- Legacy runtime quality still has zero full-humanoid passes; all 32 full humanoid rows remain legacy warned.
- The release-quality v2 policy remains diagnostic-only and is not a visual readiness, deployment readiness, or production default change.

## 12. Next direction
- Recommended Step 4.5 or next blocked investigation: validate the diagnostic aggregation policy on a larger clip suite and define an audited task-level residual contract before promotion.
- Do not start automatically.
