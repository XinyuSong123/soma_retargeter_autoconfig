# Step 4.2 Orientation Policy Runtime Scoring Handoff

Branch:

```text
retargeting-v3-step4-2-orientation-policy-runtime-scoring
```

Artifact directory:

```text
artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/
```

Baseline Step 4.1 artifact/source head:

```text
97537f015012b3207a25756633729ee37f190a16
```

Step 4.2 artifact source commit:

```text
eac94dc821ead1e66ad7cbbadbcb8859c05be763
```

## Result

```text
release_candidate_status = PASS_RC
primary_quality_breakthrough = true
selected_orientation_policy = parent_relative_runtime_inv_target
orientation_policy_active_for_scoring = true
orientation_policy_production_default_changed = false
runtime_override_default_enabled = false
```

Step 4.2 promotes the Step 4.1 parent-relative orientation policy into active runtime scoring evidence only behind the explicit flag:

```text
--enable-parent-relative-orientation-runtime-scoring
```

Production/default runtime scoring remains the legacy world-relative policy unless this Step 4.2 flag is explicitly enabled.

## Runtime Counts

Baseline Step 4.1:

```text
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
high_residual_warning_count = 32
```

Step 4.2:

```text
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
high_residual_warning_count = 32
```

No warned rows were renamed into passed rows.

## Active Scoring Impact

The PASS_RC breakthrough is active scoring residual distribution improvement:

```text
p95_orientation_integrated_residual = 3.309012625858
p95_orientation_integrated_residual_delta_vs_step4_1 = -0.959127833526
p95_normalized_task_residual_p95_delta_vs_step4_1 = -0.006465520975
raw_residual_regression_count = 0
normalization_hides_raw_residual_regression = false
denominator_inflation_detected = false
```

Gate reconciliation remains honest:

```text
rows_newly_passing = 0
rows_still_warned = 32
pass_gate_thresholds_unchanged = true
warn_gate_thresholds_unchanged = true
gates_weakened = false
```

Remaining blocker taxonomy:

```text
high_task_residual = 32
normalized_task_residual_p95_above_pass_gate = 32
normalized_task_residual_p95_above_warn_gate = 32
solver_convergence_weak = 1
solver_success_fraction_below_one = 1
```

## Verification

Focused tests:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step4_2_orientation_policy_runtime_scoring_*.py \
  tests/v3/test_step4_1_orientation_residual_*.py \
  tests/v3/test_step4_full_pipeline_acceptance_*.py \
  --junitxml=artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/test_results/junit.xml
```

Result:

```text
74 passed
```

Local strict artifact audit:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring.py \
  --artifact-dir artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring \
  --baseline-step4-1-artifact-dir artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough \
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

Final HEAD CI evidence is validated after the artifact/handoff commit is pushed:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring.py \
  --artifact-dir artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring \
  --baseline-step4-1-artifact-dir artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough \
  --source-root . \
  --require-final-head-ci
```

The workflow must report:

```text
workflow_run_id
head_sha
conclusion
job_conclusions
```

## Known Limits

This is a runtime scoring evidence breakthrough, not a visual/deployment readiness claim. All 32 full-humanoid rows still warn under the unchanged gates, and high residual warnings remain 32. The active parent-relative orientation policy improves the scoring residual distribution substantially without raw residual regression, but it does not yet produce gate-valid passed rows.

Next technical direction, if a later stage is started, is to address the remaining normalized residual gate blockers globally without robot-specific thresholds, whitelists, or gate weakening.
