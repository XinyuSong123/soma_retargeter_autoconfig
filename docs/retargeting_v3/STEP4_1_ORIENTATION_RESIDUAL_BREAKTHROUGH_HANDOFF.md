# Step 4.1 Orientation Residual Breakthrough Handoff

## Result

Branch:

```text
retargeting-v3-step4-1-orientation-residual-breakthrough
```

Artifact source commit:

```text
7c610bafe55dfc2170e4aa1b570459eaa8c252c1
```

Artifact directory:

```text
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/
```

Release candidate status:

```text
PASS_RC
primary_quality_breakthrough = true
```

Step 4.1 did not rename warned rows into passed rows. Runtime quality gates are unchanged:

```text
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
high_residual_warning_count = 32
```

The accepted breakthrough is the audited global orientation residual semantics delta:

```text
selected_policy = parent_relative_runtime_inv_target
p95_rotation_residual_p95 = 2.0492009223205
p95_rotation_residual_p95_delta_vs_step4_0 = -0.859991637555
raw_residual_regression_count = 0
normalization_hides_raw_residual_regression = false
```

## Baseline

Step 4.0 baseline artifact tree:

```text
artifacts/retargeting_v3_step4_full_pipeline_acceptance/
```

Baseline counts:

```text
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
high_residual_warning_count = 32
rotation_dominant_residual_count = 27
p95_rotation_residual_p95 = 2.90919255987585
```

Step 4.0 final branch head recorded for Step 4.1:

```text
e1ff15737bbbd2f70ac1d59394b232f99e88b2d9
```

## Orientation Semantics

Step 4.1 evaluates global candidate policies without per-robot tuning:

```text
candidate_0_current_step4_policy
candidate_1_quaternion_sign_only
candidate_2_world_delta_order_target_inv_runtime
candidate_3_world_delta_order_runtime_inv_target
candidate_4_parent_relative_delta
candidate_5_rest_pose_body_offset
candidate_6_profile_rest_offset
candidate_7_source_to_runtime_rest_delta
candidate_8_semantic_orientation_weight_balance
candidate_9_combined_global_best_policy
```

The selected policy is global parent-relative orientation residuals for non-root anchors. It uses the same semantic parent map for every model and clip:

```text
Hips -> Chest / feet
Chest -> hands
Hips root remains world-relative
```

Quaternion/log-map policy:

```text
quaternion order = xyzw
shortest-arc sign canonicalization = true
q_delta = inverse(runtime) * target in the selected comparison frame
residual = SO3 log-map vector
angle units = radians
```

## Artifacts

New Step 4.1 artifacts include:

```text
orientation_frame_semantics_matrix.json
orientation_residual_math_audit.json
orientation_offset_candidate_matrix.json
orientation_policy_selection.json
orientation_clip_consistency_matrix.json
quality_delta_vs_step4_0.json
orientation_delta_vs_step4_0.json
release_candidate_impact_report.json
```

Full-pipeline evidence is preserved:

```text
in_scope_total = 44
full_humanoid_total = 32
partial_total = 3
negative_total = 9
solver_backed_count = 32
residual_only_count = 0
trajectory_exports_count = 128
temporal_continuity_finite_count = 128
support_contact_diagnostic_count = 128
collision_proxy_diagnostic_count = 128
deterministic_compared_count = 44
deterministic_matched_count = 44
```

## Verification

Focused tests:

```text
PYTHONPATH=. python -m pytest -q tests/v3/test_step4_1_orientation_residual_*.py tests/v3/test_step4_full_pipeline_acceptance_*.py
54 passed
```

Local Step 4.1 audit:

```text
PYTHONPATH=. python scripts/audit_retargeting_v3_step4_1_orientation_residual_breakthrough.py \
  --artifact-dir artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough \
  --baseline-step4-artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --source-root .

status = PASS_RC
blocking_count = 0
finding_count = 0
matrix_row_count = 44
```

Git LFS:

```text
git lfs fsck = OK
```

Final-head CI evidence is intentionally validated by the strict audit after the artifact/handoff commit is pushed:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step4_1_orientation_residual_breakthrough.py \
  --artifact-dir artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough \
  --baseline-step4-artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --source-root . \
  --require-final-head-ci
```

## Known Limits

This is an orientation residual semantics breakthrough, not a visual/deployment readiness claim. Runtime gates are unchanged, 32 full-humanoid rows remain warned, and no production/default runtime override is enabled.

Next technical direction, if a later stage is started, is to decide whether the global parent-relative orientation residual policy should become runtime scoring logic or remain audited diagnostic evidence.

