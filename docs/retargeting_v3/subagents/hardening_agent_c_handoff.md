# Hardening Agent C Handoff

Agent: C - Motion-Class and Profile Status Policy

Status: FINAL PASS

## Scope

Agent C owned motion-class partitioning and profile terminal-status policy.

## Files

- `soma_retargeter/robotics/v3/capability_status.py`
- `soma_retargeter/robotics/v3/profile.py`
- `tests/v3/test_motion_class_partition_policy.py`
- `tests/v3/test_capability_profile_status_policy.py`
- `tests/v3/test_stress_motion_status_isolation_policy.py`
- `tests/v3/test_baseline_exact_no_regression_motion_status.py`

## Final Evidence

- Fixed motion classes cover every canonical motion exactly once: invariance, ordinary, extended, and stress.
- Invariance motions are exact-only.
- Stress diagnostics do not affect profile status.
- Full humanoid status is computed from the required non-stress motion/task matrix rather than any single limited certificate.
- Partial and negative-control paths do not enter capability status evaluation.
- Final status counts are `passed=32`, `partial_passed=3`, `negative_control_passed=9`, `algorithm_failed=0`.
- Baseline `passed` models remain `passed`: transition count `passed->passed=21`.

## Verification

- Clean full pytest artifact: `365 passed, 10 skipped, 148 warnings in 5354.42s (1:29:14)`.
- Final capability audit: `PASS`.

Remaining blockers: none.
