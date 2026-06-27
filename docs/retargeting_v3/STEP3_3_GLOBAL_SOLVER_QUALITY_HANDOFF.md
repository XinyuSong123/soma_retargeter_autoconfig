# Step 3.3 Global Solver Quality Handoff

## Branch

```text
retargeting-v3-step3-3-global-solver-quality-hardening
```

## Baseline

Closed Step 3.2 final HEAD:

```text
6ae0bfbc1153e3aba1291f38f0c82dfac6c2fa57
```

Step 3.3 preserves the closed Step 3.2 artifact tree under
`artifacts/retargeting_v3_step3_2_solver_backed_smoke/` and writes new evidence under:

```text
artifacts/retargeting_v3_step3_3_global_solver_quality/
```

## Implementation

Source commit used to generate the Step 3.3 artifacts:

```text
e5239dc8a340be0f1fb6f41d5fa56abba91284d2
```

The artifact/handoff commit is the branch HEAD that adds
`artifacts/retargeting_v3_step3_3_global_solver_quality/` and this document.
Final-head CI is validated by live strict audit rather than stale committed metadata.

## Global Solver Config

From `solver_config.json`:

```text
solver_config_hash = fefdee2e9c2da9c361162591277b729112a2920ef5acd8e99c691bd36d3b96ca
global_config = true
robot_specific_tuning = false
seed_policy = neutral_with_previous_frame_reuse
reuse_previous_frame_seed = true
residual_nonincrease_guard = true
line_search_alphas = [1.0, 0.5, 0.25, 0.1, 0.0]
max_update_norm = 1.0
project_joint_limits = true
task_order = [torso]
```

The quality-metric path now evaluates joint limits against `qpos_adr` when available,
so generic solver output is checked against the actual qpos coordinate addressed by
each runtime coordinate rather than a velocity-coordinate ordinal.

## Artifact Counts

Step 3.2 baseline:

```text
in_scope_total = 44
full_humanoid_total = 32
partial_total = 3
negative_total = 9
solver_backed_count = 32
residual_only_count = 0
runtime_quality_passed_count = 0
runtime_quality_warned_count = 23
runtime_quality_failed_count = 9
joint_limit_warning_count = 12
joint_limit_smoke_warning_count = 10
high_residual_warning_count = 32
deterministic_matched_count = 44
```

Step 3.3:

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
joint_limit_warning_count = 1
joint_limit_smoke_warning_count = 0
high_residual_warning_count = 32
deterministic_compared_count = 44
deterministic_matched_count = 44
```

Delta vs Step 3.2:

```text
runtime_quality_failed_count_delta = -9
runtime_quality_warned_count_delta = +9
runtime_quality_passed_count_delta = 0
joint_limit_warning_count_delta = -11
joint_limit_smoke_warning_count_delta = -10
high_residual_warning_count_delta = 0
solver_backed_count_delta = 0
residual_only_count_delta = 0
deterministic_matched_count_delta = 0
```

## Failure Taxonomy

Resolved hard failures:

```text
apptronik_apollo_mjcf
elf2_mjcf
jaxon_urdf
jvrc_mjcf
pal_talos_mjcf_direct
robonaut2_urdf
roboparty_rpo_local
unitree_h1_2_mjcf
unitree_h1_mjcf
```

These rows moved from `runtime_quality_failed` to `runtime_quality_warned`.
No full-humanoid row remains `runtime_quality_failed`.

Remaining warnings:

```text
32 full-humanoid rows remain runtime_quality_warned.
All 32 retain high_task_residual / normalized_task_residual gate warnings.
One full-humanoid row retains solver_convergence_weak and solver_success_fraction_below_one.
No full-humanoid row retains joint_limit_violation.
```

Interpretation:

```text
Step 3.3 removes the hard joint-limit failure class while preserving honest high-residual warnings.
It does not claim any runtime_quality_passed full-humanoid rows.
```

## Commands And Evidence

Artifact generation command is recorded in:

```text
artifacts/retargeting_v3_step3_3_global_solver_quality/commands.txt
```

Focused pytest evidence:

```text
artifacts/retargeting_v3_step3_3_global_solver_quality/test_results/pytest.txt
artifacts/retargeting_v3_step3_3_global_solver_quality/test_results/pytest_summary.json
artifacts/retargeting_v3_step3_3_global_solver_quality/test_results/junit.xml
```

Recorded pytest result:

```text
33 passed
```

Local audit command:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_3_global_solver_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_2_solver_backed_smoke \
  --source-root .
```

Local audit result:

```text
status = PASS
blocking_count = 0
finding_count = 0
```

Strict final-head audit command:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_3_global_solver_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_2_solver_backed_smoke \
  --source-root . \
  --require-final-head-ci
```

Strict audit uses live GitHub check-runs for the current branch HEAD and reports the
concrete `workflow_run_id`, `head_sha`, and job conclusions at validation time.

LFS evidence:

```text
git lfs fsck = OK
```

## GitHub Actions

Workflow:

```text
Retargeting V3 Step 3.3 Global Solver Quality
.github/workflows/retargeting_v3_step3_3_global_solver_quality.yml
```

Required jobs:

```text
step3-3-static-and-unit
step3-3-artifact-audit
step3-3-lfs-and-snapshot-smoke
step3-3-pipeline-controls-reference
step3-3-final-head-ci-evidence
```

The strict Step 3.3 audit requires the core Step 3.3 jobs to be successful for the
current HEAD.

## Known Limitations

```text
runtime_quality_passed_count remains 0.
32 full-humanoid rows remain runtime_quality_warned due to high residuals.
High-residual warning count is unchanged at 32.
This step does not solve visual motion quality, production retargeting, contact,
collision, temporal smoothing, or Step 3.4.
No robot-specific tuning, thresholds, whitelists, or per-robot IK weights were used.
```

## Next Direction

Do not start Step 3.4 automatically. A future step can address global residual
quality through broader generic task coverage, better anchor reliability, and
global residual normalization while preserving the same fail-closed audit semantics.
