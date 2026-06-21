# Step 2 Agent C-xhigh Handoff - Rest Calibration and Offset-Free Target Geometry

## Scope

Agent C-xhigh owned only:

- `soma_retargeter/robotics/v3/source_rest.py`
- `soma_retargeter/robotics/v3/rest_frames.py`
- `soma_retargeter/robotics/v3/target_builder.py`
- `soma_retargeter/robotics/v3/calibration_validation.py`
- `tests/v3/test_rest_calibration_*.py`
- `tests/v3/test_target_builder_*.py`
- `docs/retargeting_v3/subagents/step2_agent_c_handoff.md`

No Step 3 runtime pipeline, contact runtime, production NewtonPipeline, or whole-body IK tuning was started.

## Reasoning Effort

This Agent C run was executed as Agent C-xhigh per the updated `goal.md` subagent requirement. Prior Agent C worktree commits reviewed as input:

- `b2db0655432506e21541a235ed03cf831bcf6d09` - `feat: implement independent rest calibration targets`
- `107ae72ee023dc5b4a65e96c3ec40ee20990a697` - `docs: add agent c step2 handoff`

This run repaired the prior target-builder source-zero fallback bug before committing.

## Commits

- `e153693b323cc97f815094f7067d88e5b780b09c` - `feat: repair rest calibration target geometry`

## Files Changed

- `soma_retargeter/robotics/v3/rest_frames.py`
  - Removed hardcoded zero calibration errors.
  - Records neutral position/orientation errors from independently reconstructed neutral targets.
  - Adds conditioned edge-frame alignment from edge direction plus bilateral/torso semantic vectors.
  - Records `edge_conditioning`, `edge_frame_sources`, fallbacks, confidence, root horizontal scale, vertical scale, and support heights.
- `soma_retargeter/robotics/v3/target_builder.py`
  - Uses calibrated root translation/orientation decomposition instead of copying a source global transform onto robot targets.
  - Maps horizontal root motion with leg-length ratio.
  - Maps vertical root motion from pelvis-to-support-foot height delta.
  - Applies rotation deltas with `Delta_r = A * Delta_h * A.T` and emits relative rotation as `R_robot_rest_rel * Delta_r`.
  - Uses robot rest-relative fallback for missing/degenerate edge calibration.
- `soma_retargeter/robotics/v3/calibration_validation.py`
  - Adds independent neutral reconstruction validation by regenerating neutral targets and comparing against runtime neutral FK snapshots.
- `tests/v3/test_rest_calibration_validation.py`
  - Covers independent neutral errors, alignment diagnostics, root scale/support heights, and degeneracy fallback reporting.
- `tests/v3/test_target_builder_calibrated_geometry.py`
  - Covers rotation conjugation, root leg-ratio scaling, source global-yaw local-articulation invariance, and missing-alignment fallback.

## Commands and Tests

Pass:

```bash
PYTHONPATH=. pytest -q tests/v3/test_rest_calibration_validation.py tests/v3/test_target_builder_calibrated_geometry.py
```

Result: `8 passed in 0.67s`.

Pass:

```bash
python -m compileall -q soma_retargeter/robotics/v3/source_rest.py soma_retargeter/robotics/v3/rest_frames.py soma_retargeter/robotics/v3/target_builder.py soma_retargeter/robotics/v3/calibration_validation.py
```

Result: clean compile.

Pass:

```bash
rg -n "pos_errors = \{name: 0\.0|rot_errors = \{name: 0\.0|global_delta|_apply_global_delta|joint_scales|joint_offsets|human_to_robot_scaler|pose_pair|legacy" soma_retargeter/robotics/v3/source_rest.py soma_retargeter/robotics/v3/rest_frames.py soma_retargeter/robotics/v3/target_builder.py soma_retargeter/robotics/v3/calibration_validation.py tests/v3/test_rest_calibration_validation.py tests/v3/test_target_builder_calibrated_geometry.py -S
```

Result: no matches.

Pass:

```bash
git diff --check -- soma_retargeter/robotics/v3/source_rest.py soma_retargeter/robotics/v3/rest_frames.py soma_retargeter/robotics/v3/target_builder.py soma_retargeter/robotics/v3/calibration_validation.py tests/v3/test_rest_calibration_validation.py tests/v3/test_target_builder_calibrated_geometry.py
```

Result: clean.

Pass:

```bash
PYTHONPATH=. pytest -q tests/test_kinematic_profile_v3.py::test_prismatic_torso_profile_compiles_with_translation_rank_one
```

Result: `1 passed in 0.63s`.

Known non-owned failure:

```bash
PYTHONPATH=. pytest -q tests/test_kinematic_profile_v3_calibration_targets.py
```

Result: `4 passed, 1 failed`.

Failure: `test_source_reference_target_reconstruction_is_global_transform_equivariant` expects absolute copied source translation `[0.3, -0.2, 0.4]`. Agent C-xhigh now follows Step 1 root policy: horizontal root displacement is scaled by leg-length ratio to `[0.15, -0.1]`, and vertical common global source translation is ignored unless pelvis-to-support-foot height changes.

Known non-owned/artifact failure:

```bash
PYTHONPATH=. pytest -q tests/v3
```

Result: `42 passed, 1 failed`.

Failure: `test_retargeting_v3_step2_acceptance_gates_are_clean` reports 8 Step-2 blockers:

- `canonical_targets_without_projection`: `roboparty_rpo_local`
- `arbitrary_g1_equivalence`: `validation_checks.g1_mjcf_urdf_equivalence`
- `dirty_artifact_metadata`: dirty worktree and artifact `git_head` mismatch
- `robot_name_special_cases`: `soma_retargeter/robotics/v3/semantic_sites.py:38` and `:39`
- `missing_formal_red_team_artifacts`: missing `step2_agent_b_handoff.md` and `step2_agent_d_handoff.md`

## Numeric Results

Simple rotated source/robot calibration fixture:

- `root_horizontal_scale = 0.5`
- `vertical_root_scale = 0.5`
- `source_support_height = 2.0 m`
- `robot_support_height = 1.0 m`
- `neutral_max_position_error = 2.48253415325e-16 m`
- `neutral_max_orientation_error = 0.0 rad`
- source pelvis local displacement `[0.4, 0.0, 0.0]` maps to robot root local displacement `[0.2, 0.0, 0.0]`
- torso conjugation rotation error `0.0 rad`
- torso edge conditioning `1.0`

## Assumptions

- Agent A/B provide trustworthy runtime FK and verified semantic/distal site transforms.
- `Hips -> Chest` alignment is the default root frame alignment for Agent C root displacement and orientation mapping.
- Common global source Z translation is not a crouch signal; vertical root target follows pelvis-to-support-foot geometry.
- Welded or zero-length robot/source edges may reconstruct neutral through robot rest-relative fallback while recording fallback provenance and reduced confidence.

## Failures and Risks

- Full Robot Zoo validation was not run by Agent C-xhigh.
- Complete-model projection artifacts still need regeneration/integration after Agent D; the acceptance audit still blocks on missing RPO per-motion projection reports.
- Non-owned `tests/test_kinematic_profile_v3_calibration_targets.py::test_source_reference_target_reconstruction_is_global_transform_equivariant` encodes the pre-Step-1 copied-global-translation behavior and needs integrator/Agent F review.
- Source rest frames still depend on available SOMA semantic transforms and verified downstream site maps.
- Edge-frame conditioning uses semantic vectors available at rest; pathological collinear semantics fall back and lower confidence.

## Integration Notes

- Integrate after runtime FK/site truth and verified distal sites, because calibration correctness depends on those transforms.
- Agent D should consume Agent C `canonical_motion_targets` as desired morphology targets before chain projection; projection reports must not be neutral-to-neutral residual-zero artifacts.
- Per-robot reports should include `edge_conditioning`, `edge_frame_sources`, root scales, support heights, neutral reconstruction errors, and fallback provenance.
- Existing artifacts must be regenerated after all agents land; otherwise the acceptance audit will keep reporting dirty metadata and missing projection blockers.
