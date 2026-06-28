# Step 3.4 Global Residual Quality Handoff

## Branch

```text
retargeting-v3-step3-4-global-residual-quality
```

## Baseline

Closed Step 3.3 final HEAD:

```text
1a8ae37670a3f3f8c49cefd78c6bf7292aa15489
```

Step 3.4 preserves the closed Step 3.3 artifact tree under:

```text
artifacts/retargeting_v3_step3_3_global_solver_quality/
```

Step 3.4 writes new evidence under:

```text
artifacts/retargeting_v3_step3_4_global_residual_quality/
```

## Implementation

Source commit used to generate the Step 3.4 artifacts:

```text
2f7c7346df615bc2653af8596d893e7be5d37036
```

The artifact/handoff commit is the branch HEAD that adds
`artifacts/retargeting_v3_step3_4_global_residual_quality/` and this document.
Final-head CI is validated by live strict audit after the final push.

## Global Residual Quality Config

From `solver_config.json`:

```text
solver_config_hash = fe15673fe87a1b924830ff7821ba757430e112fefc354de07af31239d9f0128a
global_config = true
robot_specific_tuning = false
task_order = [torso, left_hand, right_hand, left_foot, right_foot]
enable_global_residual_quality_hardening = true
anchor_reliability_policy_version = global_semantic_anchor_reliability_v1
residual_normalization_policy = legacy_row_max_recorded_with_raw_residual_guard
semantic_task_weights = uniform_global
```

The Step 3.4 solver attempts the global semantic task order, records task coverage
and anchor reliability, and applies a global all-semantic residual candidate guard.
Runtime pass gates are unchanged.

## Artifact Counts

Step 3.3 baseline:

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
high_residual_warning_count = 32
deterministic_matched_count = 44
```

Step 3.4:

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
joint_limit_warning_count = 1
deterministic_compared_count = 44
deterministic_matched_count = 44
```

## Residual Delta Vs Step 3.3

`quality_delta_vs_step3_3.json` verdict:

```text
PASS
```

High residual warning count remains 32, but raw residual distribution improves:

```text
raw_task_residual_p95_delta.median = -0.031452299139999695
raw_task_residual_p95_delta.p95 = -0.05660261550525014
raw_task_residual_p95_delta.max = -0.10800909694899996
```

Normalized p95 distribution has mixed movement because the legacy row-local
normalization remains saturated, but p95 improves:

```text
normalized_task_residual_p95_delta.median = +0.0014194740480000245
normalized_task_residual_p95_delta.p95 = -0.000940503765850087
normalized_task_residual_p95_delta.max = +0.0008017714960000255
```

Normalization integrity:

```text
raw_residual_regression_count = 0
normalization_hides_raw_regression = false
```

## Task Coverage And Anchor Reliability

Task coverage:

```text
task_coverage_mean = 1.0
task_coverage_min = 1.0
task_coverage_mean_delta = +0.8
task_coverage_min_delta = +0.8
```

Anchor reliability:

```text
anchor_reliability_mean = 0.994791666667
anchor_reliability_min = 0.833333333333
```

## Residual Taxonomy

Aggregate blocker buckets from `residual_taxonomy.json`:

```text
residual_normalization_scale_issue = 32
rotation_residual_dominates = 31
translation_residual_dominates = 1
anchor_unavailable_or_unreliable = 1
solver_converged_but_task_underconstrained = 1
```

Interpretation:

```text
Step 3.4 improves raw residual severity and task coverage globally.
It does not reduce high_residual_warning_count below 32.
It does not produce runtime_quality_passed full-humanoid rows.
The remaining blocker is still residual quality, mostly rotation-dominant, under unchanged gates.
```

## Commands And Evidence

Artifact generation command is recorded in:

```text
artifacts/retargeting_v3_step3_4_global_residual_quality/commands.txt
```

Focused pytest evidence:

```text
artifacts/retargeting_v3_step3_4_global_residual_quality/test_results/pytest.txt
artifacts/retargeting_v3_step3_4_global_residual_quality/test_results/pytest_summary.json
artifacts/retargeting_v3_step3_4_global_residual_quality/test_results/junit.xml
```

Recorded pytest result:

```text
21 passed
```

Local audit command:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_4_global_residual_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --source-root .
```

Local audit result:

```text
status = PASS
blocking_count = 0
finding_count = 0
```

LFS evidence:

```text
git lfs fsck = OK
```

## GitHub Actions

Workflow:

```text
Retargeting V3 Step 3.4 Global Residual Quality
.github/workflows/retargeting_v3_step3_4_global_residual_quality.yml
```

Required jobs:

```text
step3-4-static-and-unit
step3-4-artifact-audit
step3-4-lfs-and-snapshot-smoke
step3-4-pipeline-controls-reference
step3-4-final-head-ci-evidence
```

Strict final-head audit must be run after pushing the artifact/handoff commit:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_4_global_residual_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --source-root . \
  --require-final-head-ci
```

## Known Limitations

```text
runtime_quality_passed_count remains 0.
32 full-humanoid rows remain runtime_quality_warned due to high residuals.
High-residual warning count is unchanged at 32.
Visual motion quality, production retargeting, contact, collision, and temporal smoothing are not solved.
No robot-specific tuning, thresholds, whitelists, or per-robot IK weights were used.
```

## Next Direction

Do not start Step 3.5 automatically. A future step should target the remaining
rotation-dominant residual class with global orientation-task coverage and stronger
normalization semantics, while preserving the same fail-closed audit semantics.
