# Hardening Agent E Handoff

## Scope

Owned validation-side files changed:

- `soma_retargeter/robotics/v3/validation.py`
- `tests/v3/test_full_44_hardened_validation_baseline.py`
- `tests/v3/test_hardened_deterministic_rerun_profile_fields.py`
- `tests/v3/test_hardened_cross_format_negative_control.py`
- `docs/retargeting_v3/STEP2_3_1_VALIDATION_REPORT.md`
- `docs/retargeting_v3/subagents/hardening_agent_e_handoff.md`

No final capability artifacts were regenerated.

## Behavior

- Full 44 model sets now load the fixed Assets44 baseline from commit
  `5ad5a001c445c525d4c8bbaf6339dec5c5c2c719`.
- Non-baseline synthetic manifests produce explicit `baseline_unavailable` before/after rows rather than fabricated baseline rows.
- Full 44 before/after validation blocks illegal transitions and final count violations.
- Deterministic profile comparisons now include q vectors, desired/projected task vectors, and certificate class/digest.
- Negative-control cross-format pairs remain not eligible for humanoid variant compatibility.

## Commands

```bash
PYTHONPATH=. pytest -q tests/v3/test_full_44_hardened_validation_baseline.py tests/v3/test_hardened_deterministic_rerun_profile_fields.py tests/v3/test_hardened_cross_format_negative_control.py
PYTHONPATH=. pytest -q tests/v3/test_full_44_hardened_validation_baseline.py tests/v3/test_hardened_deterministic_rerun_profile_fields.py tests/v3/test_hardened_cross_format_negative_control.py tests/v3/test_deterministic_rerun_validation.py
PYTHONPATH=. python -m py_compile soma_retargeter/robotics/v3/validation.py soma_retargeter/tools/validate_kinematic_profile_v3.py tests/v3/test_full_44_hardened_validation_baseline.py tests/v3/test_hardened_deterministic_rerun_profile_fields.py tests/v3/test_hardened_cross_format_negative_control.py
```

Result: targeted pytest set passed, 16 tests total. Syntax check passed.

## Remaining Integration Risk

The current committed capability artifacts still reflect the pre-hardening generated state. A coordinated integration run must
regenerate artifacts after all upstream Agent A/B/C/D status and certificate changes land. With this validation wiring, a run
that still downgrades baseline exact passes to `capability_limited_passed` will fail.
