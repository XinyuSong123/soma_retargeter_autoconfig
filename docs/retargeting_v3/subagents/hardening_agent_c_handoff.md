# Hardening Agent C Handoff

Scope: motion-class and profile status policy for canonical projection reports.

## Owned Changes

- Added `soma_retargeter/robotics/v3/capability_status.py`.
  - Fixed motion classes for invariance, ordinary, extended, and stress motions.
  - Added partition validation for canonical motion order.
  - Added matrix-level profile evaluation returning `passed`, `capability_limited_passed`, or `algorithm_failed`.
  - Enforced invariance exact-only behavior.
  - Ignored stress motion diagnostics for profile status.
  - Added class-specific checks for exact, rank-limited, joint-limit-limited, mixed, unsupported rank-zero, failed, and missing certificate cases.
  - Kept the module independent of Robot Zoo IDs and manifests.
- Updated `soma_retargeter/robotics/v3/profile.py`.
  - Full-humanoid profiles now consume the new evaluator after canonical projection.
  - Partial and semantic-incomplete profiles keep their semantic status and do not enter capability-limited evaluation.
  - Removed the old over-threshold any-certificate bypass from profile-owned logic.
  - Preserved `_projection_quality_threshold()` as a compatibility wrapper around the policy threshold helper.
- Added focused tests:
  - `tests/v3/test_motion_class_partition_policy.py`
  - `tests/v3/test_capability_profile_status_policy.py`
  - `tests/v3/test_stress_motion_status_isolation_policy.py`
  - `tests/v3/test_baseline_exact_no_regression_motion_status.py`

No edits were made to `robot_zoo.py` or `status_schema.py`; they were not necessary for this scoped change.

## Verification

- Initial failing test run before implementation:
  - `PYTHONPATH=. pytest -q tests/v3/test_motion_class_partition_policy.py tests/v3/test_capability_profile_status_policy.py tests/v3/test_stress_motion_status_isolation_policy.py tests/v3/test_baseline_exact_no_regression_motion_status.py`
  - Result: failed during collection because `soma_retargeter.robotics.v3.capability_status` did not exist.
- Focused Agent C policy tests:
  - `PYTHONPATH=. pytest -q tests/v3/test_motion_class_partition_policy.py tests/v3/test_capability_profile_status_policy.py tests/v3/test_stress_motion_status_isolation_policy.py tests/v3/test_baseline_exact_no_regression_motion_status.py`
  - Result: `11 passed`.
- Compile check:
  - `PYTHONPATH=. python -m compileall -q soma_retargeter/robotics/v3/capability_status.py soma_retargeter/robotics/v3/profile.py`
  - Result: passed.
- Existing nearby profile tests:
  - `PYTHONPATH=. pytest -q tests/v3/test_profile_canonical_projection_contract.py tests/v3/test_rank_zero_projection_evidence.py`
  - Earlier result before concurrent projection-certificate edits: `5 passed`.
  - Current rerun result: `2 failed, 3 passed`.
  - Current failures are `NameError: name '_optional_bound_array' is not defined` in `soma_retargeter/robotics/v3/projection_certificate.py::_active_limits`, which is outside Agent C ownership and already dirty from concurrent work.
- Profile compile smoke:
  - A direct `compile_kinematic_profile_v3(...)` smoke check is currently blocked by the same `projection_certificate.py` `_optional_bound_array` `NameError`.
- Broader nearby run:
  - `PYTHONPATH=. pytest -q tests/v3/test_profile_canonical_projection_contract.py tests/v3/test_capability_status_terminal.py tests/v3/test_rank_zero_projection_evidence.py`
  - Earlier result before the current projection-certificate blocker: `1 failed, 6 passed`.
  - The non-blocker failure was in `tests/v3/test_capability_status_terminal.py::test_partial_passed_remains_terminal_status_with_task_certificates`: the fixture currently builds all 15 canonical motions but still asserts `motion_count == 2`. That file is outside Agent C ownership and was not edited.

## Integration Notes

- `profile.py` no longer uses any single limited certificate to decide profile status. It evaluates the required non-stress motion/task matrix.
- `capability_status.py` treats absent optional producer gates (`continuation`, `joint_limits`, `numerical`, `primal_feasible`) as compatible unless explicitly false, because the current certificate producer does not always emit all of those fields yet. Core limited gates (`projected_gradient_kkt`, `seed_consensus`, `residual_explained`) are required.
- `soma_retargeter/robotics/v3/validation.py` still contains legacy validator-side `_has_capability_limited_certificate()` any-certificate semantics. That file is outside Agent C ownership in this handoff. Integrator should reconcile it with `capability_status.evaluate_profile_status()` or trust `profile.capability_status` to avoid stress-only downgrades in externally supplied validation payloads.
- The worktree contains concurrent dirty files outside Agent C ownership. Agent C did not revert or edit them.
