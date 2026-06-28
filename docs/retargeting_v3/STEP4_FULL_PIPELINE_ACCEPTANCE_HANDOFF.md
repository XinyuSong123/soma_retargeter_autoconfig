# Step 4.0 Full Pipeline Acceptance Handoff

Generated on the `retargeting-v3-step4-full-pipeline-acceptance` branch.

## Result

Step 4.0 is complete as an honest blocked acceptance artifact:

```text
release_candidate_status = BLOCKED_RESIDUAL_QUALITY
verdict = BLOCKED
primary_quality_breakthrough = false
blocking structural audit findings = 0
```

The run preserved Step 3.4 solver-backed coverage and produced the requested full-pipeline diagnostics, trajectory exports, temporal diagnostics, support/contact diagnostics, collision proxy diagnostics, normalization audit, orientation residual taxonomy, quality delta, audit script, focused tests, workflow, and final artifact tree.

`PASS_RC` was not claimed because no accepted primary quality breakthrough was found under global methods:

```text
runtime_quality_passed_count = 0
high_residual_warning_count = 32
primary_quality_breakthrough = false
regressions = []
improvements = ["raw_task_residual_p95_distribution_improved"]
```

## Provenance

Baseline Step 3.4 final HEAD:

```text
77e7c02393a6678ccab40cdb847021d7d94392c9
```

Step 4 artifact source commit:

```text
a21d48b897bd08c6a88fc7504d3bb80abf8361af
```

Clean provenance fields in `artifacts/retargeting_v3_step4_full_pipeline_acceptance/environment.json`:

```text
source_worktree_clean_before_run = true
source_worktree_clean_after_run = true
core_diff_after_source_commit = []
```

## Artifact Tree

Primary artifact directory:

```text
artifacts/retargeting_v3_step4_full_pipeline_acceptance/
```

Key artifacts:

```text
quality_summary.json
acceptance_ledger.json
full_pipeline_matrix.json
clip_matrix.json
solver_smoke_matrix.json
solver_diagnostics_matrix.json
quality_delta_vs_step3_4.json
orientation_residual_taxonomy.json
normalization_audit.json
trajectory_export_manifest.json
temporal_continuity_matrix.json
support_contact_diagnostics.json
collision_proxy_diagnostics.json
pipeline_config.json
pipeline_controls_reference.json
red_team_report.json
test_results/pytest.txt
test_results/pytest_summary.json
test_results/junit.xml
```

## Final Metrics

```text
in_scope_total = 44
full_humanoid_total = 32
partial_total = 3
negative_total = 9

solver_backed_smoke_attempted_count = 32
solver_backed_completed_count = 32
solver_backed_count = 32
residual_only_count = 0

runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
partial_runtime_passed_count = 3
negative_control_runtime_passed_count = 9
high_residual_warning_count = 32

deterministic_compared_count = 44
deterministic_matched_count = 44

clip_matrix_rows = 140
solver_smoke_rows = 128
trajectory_exports_count = 128
temporal_continuity_finite_count = 128
support_contact_diagnostic_count = 128
collision_proxy_diagnostic_count = 128
```

Residual distributions:

```text
median_raw_task_residual_p95 = 3.5578656079654998
p95_raw_task_residual_p95 = 4.26814045938395
max_raw_task_residual_p95 = 4.302029690497

median_normalized_task_residual_p95 = 0.997502471969
p95_normalized_task_residual_p95 = 0.9998567673000001
max_normalized_task_residual_p95 = 0.999930378142

orientation_rotation_residual_p95_distribution = {
  "median": 2.50134081541,
  "p95": 2.90919255987585,
  "max": 2.921195688897
}
rotation_residual_dominates = 27
translation_residual_dominates = 5
```

Normalization audit:

```text
denominator_inflation_detected = false
normalization_hides_raw_residual_regression = false
normalization_reconstruction_mismatch_count = 22
suspicious_rows = []
```

## Verification

Focused tests:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step4_full_pipeline_acceptance_*.py \
  tests/v3/test_step3_4_global_residual_quality_*.py \
  --junitxml=artifacts/retargeting_v3_step4_full_pipeline_acceptance/test_results/junit.xml
```

Result:

```text
53 passed
```

Local strict artifact audit:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step4_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --baseline-step3-4-artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --source-root .
```

Result:

```text
status = BLOCKED_RESIDUAL_QUALITY
blocking_count = 0
finding_count = 0
matrix_row_count = 44
```

LFS:

```text
git lfs fsck = OK
```

## Regeneration Command

```bash
PYTHONPATH=. python soma_retargeter/tools/run_v3_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --baseline-step3-4-artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --enable-global-orientation-residual-hardening \
  --enable-full-pipeline-exports \
  --deterministic-rerun
```

Expected generated status:

```text
BLOCKED_RESIDUAL_QUALITY
```

## CI Follow-Up

After the artifact/docs commit is pushed, wait for the `Retargeting V3 Step 4 Full Pipeline Acceptance` workflow on the final branch HEAD, then run:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step4_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --baseline-step3-4-artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --source-root . \
  --require-final-head-ci
```

The expected strict final-head audit status is still `BLOCKED_RESIDUAL_QUALITY`, with `blocking_count = 0`, because the blocked residual-quality outcome is the honest release status and not a pipeline/provenance failure.

## Next Technical Focus

The remaining work is not artifact plumbing. It is the residual-quality problem:

```text
runtime_quality_passed_count remains 0
high_residual_warning_count remains 32
rotation residuals dominate 27 of 32 full-humanoid rows
```

Future work should target global orientation residual formulation, task weighting, and target-frame semantics without robot-specific tuning, gate weakening, or hiding raw residuals behind normalization.
