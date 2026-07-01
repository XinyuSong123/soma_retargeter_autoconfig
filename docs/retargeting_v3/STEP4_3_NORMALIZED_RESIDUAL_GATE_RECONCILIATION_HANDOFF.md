# Step 4.3 Normalized Residual Gate Reconciliation Handoff

Branch:

```text
retargeting-v3-step4-3-normalized-residual-gate-reconciliation
```

Artifact directory:

```text
artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation/
```

Step 4.3 clean source commit used for artifact generation:

```text
0a364a83b40b2a31881d132f49955d946a58ce08
```

Baseline Step 4.2 artifact/source commit:

```text
eac94dc821ead1e66ad7cbbadbcb8859c05be763
```

## Result

```text
release_candidate_status = PASS_RC
primary_quality_breakthrough = true
normalization_policy_selected = candidate_1_fixed_global_body_scale_normalization
gate_policy_selected = candidate_6_release_quality_gate_v2_candidate
legacy_gates_unchanged = true
candidate_release_gates_defined = true
candidate_release_gates_active = true
production_default_changed = false
runtime_override_default_enabled = false
```

Step 4.3 does not rename legacy warned rows into legacy passed rows:

```text
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
high_residual_warning_count = 32
```

The PASS_RC basis is the audited release-quality v2 candidate gate:

```text
rows_below_legacy_warn_gate = 0
rows_below_candidate_warn_gate = 6
rows_below_candidate_pass_gate = 0
release_quality_candidate_passed_count = 0
release_quality_candidate_warned_count = 6
```

Rows below the release-quality v2 candidate warn gate:

```text
fourier_n1_mjcf
fourier_n1_urdf
robotis_op3_mjcf
toddlerbot_2xc_mjcf
toddlerbot_2xm_mjcf
toddlerbot_urdf
```

## Normalization Finding

The legacy normalized residual remains near 1.0 because it is row-local:

```text
legacy_normalization_formula = residual / max(1.0, current_row_raw_residual_max)
denominator_scope = row_local_legacy_metric
normalized_max_all_rows_equal_one = true
rows_improve_raw_but_remain_legacy_normalized_blocked_count = 32
```

Step 4.2 strongly reduced active raw/orientation-integrated residuals, but the row-local denominator follows the same row max. The metric therefore mostly measures p95/max shape within each row, not fixed physical residual quality.

Selected fixed global scale:

```text
selected_fixed_global_scale = 4.268140459384
selected_fixed_global_scale_source = closed_step4_1_raw_task_residual_p95_p95_distribution
```

Normalization integrity:

```text
raw_residual_regression_count = 0
denominator_inflation_detected = false
normalization_hides_raw_residual_regression = false
```

Dominant task-class blocker:

```text
task_class_dominance = Hips: 32
dominant_component = rotation: 32
```

## Gate Reconciliation

Legacy blocker taxonomy is unchanged:

```text
high_task_residual = 32
normalized_task_residual_p95_above_pass_gate = 32
normalized_task_residual_p95_above_warn_gate = 32
solver_convergence_weak = 1
solver_success_fraction_below_one = 1
```

Candidate v2 blocker taxonomy:

```text
orientation_integrated_fixed_global_scale_residual_p95_above_candidate_warn_gate = 26
```

The release-quality v2 candidate is separate from `runtime_quality_passed`. Legacy gates remain unchanged and are not silently weakened.

## Preserved Invariants

```text
in_scope_total = 44
full_humanoid_total = 32
partial_total = 3
negative_total = 9
solver_backed_smoke_attempted_count = 32
solver_backed_completed_count = 32
solver_backed_count = 32
residual_only_count = 0
partial_runtime_passed_count = 3
negative_control_runtime_passed_count = 9
trajectory_exports_count = 128
temporal_continuity_finite_count = 128
support_contact_diagnostic_count = 128
collision_proxy_diagnostic_count = 128
deterministic_compared_count = 44
deterministic_matched_count = 44
```

## Verification

Focused tests:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step4_3_normalized_residual_gate_reconciliation_*.py \
  tests/v3/test_step4_2_orientation_policy_runtime_scoring_*.py \
  tests/v3/test_step4_1_orientation_residual_*.py \
  --junitxml=artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation/test_results/junit.xml
```

Result:

```text
63 passed
```

Local Step 4.3 audit:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation.py \
  --artifact-dir artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation \
  --baseline-step4-2-artifact-dir artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring \
  --source-root .
```

Result:

```text
status = PASS_RC
blocking_count = 0
finding_count = 0
matrix_row_count = 44
```

Git LFS:

```text
git lfs fsck = OK
```

## Final-Head CI

The workflow added for this step is:

```text
.github/workflows/retargeting_v3_step4_3_normalized_residual_gate_reconciliation.yml
```

Required final-head CI jobs:

```text
step4-3-static-and-unit
step4-3-artifact-audit
step4-3-gate-semantics-smoke
step4-3-normalization-integrity-smoke
step4-3-export-and-temporal-smoke
step4-3-lfs-and-snapshot-smoke
step4-3-pipeline-controls-reference
step4-3-final-head-ci-evidence
```

Final-head CI evidence is validated after the artifact/handoff commit is pushed:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation.py \
  --artifact-dir artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation \
  --baseline-step4-2-artifact-dir artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring \
  --source-root . \
  --require-final-head-ci
```

The strict audit must report concrete `workflow_run_id`, current `head_sha`, `conclusion = success`, and all required job conclusions as `success`.

## Known Limits

This is a normalized residual and release-quality gate semantics reconciliation, not a visual/deployment readiness claim.

Legacy runtime quality gates still warn all 32 full humanoids. Step 4.3 adds an audited, globally selected, separately named release-quality v2 candidate gate that identifies 6 rows below the candidate warn gate without changing production defaults, default-enabling runtime overrides, using robot-specific thresholds, weakening legacy gates, hiding raw residual regression, or using denominator inflation.

Next work, if a later stage is explicitly started, should decide whether release-quality v2 should remain diagnostic release evidence or become a future production-quality gate after additional validation.
