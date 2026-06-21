# Step 2 Agent C Handoff - Rest Calibration and Offset-Free Target Geometry

## Scope

Agent C owned only:

- `soma_retargeter/robotics/v3/source_rest.py`
- `soma_retargeter/robotics/v3/rest_frames.py`
- `soma_retargeter/robotics/v3/target_builder.py`
- `soma_retargeter/robotics/v3/calibration_validation.py`
- `tests/v3/test_rest_calibration_*.py`
- `tests/v3/test_target_builder_*.py`
- `docs/retargeting_v3/subagents/step2_agent_c_handoff.md`

No Step 3 runtime pipeline or whole-body IK work was started.

## Reasoning-Effort Note

Integrator update received after Agent C was already spawned: `goal.md` now requires all subagents to use xhigh reasoning. This running Agent C session was spawned earlier with high effort, and tool/session parameters cannot be changed in place. Agent C continued with highest available diligence and records this limitation for Integrator/Agent F review. Any replacement or extra spawned agent should use xhigh.

## Commits

- `b2db0655432506e21541a235ed03cf831bcf6d09` - `feat: implement independent rest calibration targets`

## Files Changed

- `soma_retargeter/robotics/v3/rest_frames.py`
  - Removed hard-coded `pos_errors={...: 0.0}` and `rot_errors={...: 0.0}`.
  - Added independent neutral reconstruction before recording calibration errors.
  - Added conditioned semantic edge-frame alignment from edge direction plus bilateral/torso vectors.
  - Preserves the primary parent-to-child edge direction exactly and records a Wahba residual diagnostic for the available multi-vector set.
  - Added frame conditioning/source diagnostics, leg-ratio root scale, and pelvis-to-support-foot heights.
  - Degenerate zero-length edges now record fallbacks and reduce confidence; neutral exactness uses an explicit rest-relative fallback for welded/zero-length robot edges.
- `soma_retargeter/robotics/v3/target_builder.py`
  - Replaced global-copy target construction with calibrated root translation/orientation decomposition.
  - Root horizontal displacement uses the calibrated leg-length ratio.
  - Root vertical displacement uses pelvis-to-support-foot height delta.
  - Relative edge rotations now use `Delta_r = A * Delta_h * A.T`.
  - Target relative rotation is rebuilt as `R_robot_rest_rel * Delta_r`.
  - Target builder does not read v1 scaler offsets, joint offsets, or pose-pair data.
- `soma_retargeter/robotics/v3/calibration_validation.py`
  - New independent neutral target validation helper that regenerates neutral targets and compares them to runtime neutral FK snapshots.
- `tests/v3/test_rest_calibration_validation.py`
  - New tests for independent neutral validation, frame diagnostics, root scale/support height values, degeneracy fallback recording, and rotated robot neutral geometry.
- `tests/v3/test_target_builder_calibrated_geometry.py`
  - New tests for conjugated relative rotation, root horizontal leg-ratio scaling, and source global yaw local-articulation invariance.

## Commands and Results

Pass:

```bash
PYTHONPATH=. pytest -q tests/v3/test_rest_calibration_validation.py tests/v3/test_target_builder_calibrated_geometry.py
```

Result: `7 passed in 0.67s`.

Pass:

```bash
PYTHONPATH=. pytest -q tests/test_kinematic_profile_v3.py::test_prismatic_torso_profile_compiles_with_translation_rank_one
```

Result: `1 passed in 0.71s`.

Pass:

```bash
PYTHONPATH=. pytest -q tests/v3/test_rest_calibration_validation.py tests/v3/test_target_builder_calibrated_geometry.py tests/test_kinematic_profile_v3.py::test_prismatic_torso_profile_compiles_with_translation_rank_one
```

Result: `8 passed in 0.61s`.

Pass:

```bash
python -m compileall -q soma_retargeter/robotics/v3/source_rest.py soma_retargeter/robotics/v3/rest_frames.py soma_retargeter/robotics/v3/target_builder.py soma_retargeter/robotics/v3/calibration_validation.py
```

Result: clean compile.

Pass:

```bash
rg -n "joint_scales|joint_offsets|human_to_robot_scaler|pose_pair|legacy|offset" soma_retargeter/robotics/v3/source_rest.py soma_retargeter/robotics/v3/rest_frames.py soma_retargeter/robotics/v3/target_builder.py soma_retargeter/robotics/v3/calibration_validation.py tests/v3
```

Result: only `root_offset` local variable matches; no legacy scaler/pose-pair/joint-offset data path found in Agent C owned files.

Known failing existing test:

```bash
PYTHONPATH=. pytest -q tests/test_kinematic_profile_v3_calibration_targets.py
```

Result: `4 passed, 1 failed`.

Failure: `test_source_reference_target_reconstruction_is_global_transform_equivariant` expects absolute source global translation to be copied exactly into robot target translation. Agent C changed this to Step-1/goal behavior: horizontal root displacement is scaled by leg-length ratio and vertical root motion is derived from pelvis-to-support-foot height, so the old expected `[0.3, -0.2, 0.4]` becomes scaled horizontal `[0.15, -0.1]` with zero support-height vertical delta in the simple 2:1 source-to-robot fixture.

## Numeric Results

- Simple source/robot calibration fixture:
  - `root_horizontal_scale = 0.5`
  - `vertical_root_scale = 0.5`
  - `source_support_height = 2.0 m`
  - `robot_support_height = 1.0 m`
  - independent neutral validation max position error `< 1e-12 m`
  - independent neutral validation max orientation error `< 1e-12 rad`
- Root translation test:
  - source pelvis local displacement `[0.4, 0.0, 0.0]`
  - robot pelvis local displacement `[0.2, 0.0, 0.0]`
- Prismatic torso compile smoke:
  - profile failures: `[]`
  - previous zero-length torso neutral exactness failure fixed by explicit rest-relative fallback

## Assumptions

- Agent A/B provide trustworthy runtime FK and verified semantic site transforms.
- The `Hips -> Chest` alignment is the default root frame alignment for Agent C root displacement and orientation mapping.
- The root vertical policy in this commit intentionally ignores common global source Z translation and follows pelvis-to-support-foot height change.
- Degenerate or welded robot edges can reconstruct neutral through the robot rest-relative transform while still recording fallback provenance and reduced confidence.

## Failures and Risks

- Full Robot Zoo validation was not run by Agent C.
- Complete-model RPO/G1/H1/OP3/Booster/TALOS/Berkeley calibration reports were not generated by Agent C.
- Existing non-owned test `test_source_reference_target_reconstruction_is_global_transform_equivariant` needs integration review because it encodes the old unscaled global-translation behavior.
- Source rest frames still depend on available SOMA semantic transforms; missing semantics are handled structurally but not exhaustively validated across all robot zoo entries in this commit.
- The conditioned edge-frame alignment records a Wahba residual diagnostic but intentionally does not use the unconstrained Wahba rotation when it would compromise exact edge-length neutral reconstruction.

## Integration Notes

- Integrate after Agent A runtime FK/site truth and Agent B verified distal sites, because calibration correctness depends on those transforms.
- Agent D should consume Agent C `canonical_motion_targets` as desired morphology targets before chain projection; rank-zero residuals must remain nonzero when source demand is unreachable.
- Agent E should include the new `edge_conditioning`, `edge_frame_sources`, `root_horizontal_scale`, `source_support_height`, and `robot_support_height` fields in per-robot reports.
- Agent F should update/replace the old unscaled global-translation equivariance assertion with two checks: local-articulation invariance under common source yaw and root horizontal displacement scaled by leg-length ratio.
