# Capability Agent E Handoff

Scope: Status integration, 44-model validation, and cross-format capability behavior.

## Changed Files

- `soma_retargeter/robotics/v3/profile.py`
- `soma_retargeter/robotics/v3/robot_zoo.py`
- `soma_retargeter/robotics/v3/validation.py`
- `artifacts/retargeting_v3_step2_capability/`
- `tests/v3/test_capability_status_terminal.py`
- `tests/v3/test_cross_format_capability_pairs.py`
- `tests/v3/test_full_44_capability_validation_static.py`
- `docs/retargeting_v3/STEP2_3_CAPABILITY_REPORT.md`
- `docs/retargeting_v3/subagents/capability_agent_e_handoff.md`

## Commands And Tests

Failing tests written first:

```bash
python -m pytest tests/v3/test_capability_status_terminal.py tests/v3/test_cross_format_capability_pairs.py tests/v3/test_full_44_capability_validation_static.py -q
```

Initial result: 6 failed.

Passing verification:

```bash
python -m pytest tests/v3/test_capability_status_terminal.py tests/v3/test_cross_format_capability_pairs.py tests/v3/test_full_44_capability_validation_static.py -q
python -m pytest tests/v3/test_deterministic_rerun_validation.py -q
python -m pytest tests/v3/test_cross_format_adapter_scaffolding.py -q
python -m pytest tests/v3/test_full_robot_zoo_validation.py -q
```

Results:

- New capability tests: 6 passed.
- Existing deterministic rerun tests: 3 passed.
- Existing cross-format adapter scaffold tests: 9 passed.
- Existing full Robot Zoo validation tests: 15 passed, 1 failed because the unowned test still expects legacy `partial_passed` instead of required `capability_limited_passed`.

Capability artifact generation attempted:

```bash
python -m soma_retargeter.tools.validate_kinematic_profile_v3 --manifest assets/robot_zoo/robot_zoo_manifest.json --lock assets/robot_zoo/robot_zoo_lock.json --output-dir artifacts/retargeting_v3_step2_capability --low-discrepancy-count 1 --deterministic-rerun
```

Fresh compile was blocked by concurrent unowned projection import errors. The committed capability artifact was therefore derived from the stable `retargeting_v3_step2_assets44` evidence with status and cross-format integration applied.

## Summary Counts

From `artifacts/retargeting_v3_step2_capability/summary.json`:

- `passed`: 21
- `capability_limited_passed`: 3
- `negative_control_passed`: 9
- `algorithm_failed`: 11
- terminal pass total: 33
- deterministic compared/matched: 33/33

Cross-format:

- same-source strict G1: `passed`
- variant compatibility: `passed`
- eligible positive humanoid pairs: 5
- not-eligible pairs: 4
- negative controls are not eligible for humanoid equivalence

## Risks

- Concurrent unowned changes are present in projection/certificate files and currently block fresh full recompilation. Do not interpret the failed fresh compile attempt as a status regression in this integration layer.
- An existing unowned test still hard-codes `partial_passed`; it should be updated by its owner to the new terminal status contract.

## Unfinished Items

- Re-run fresh 44-model validation through `validate_kinematic_profile_v3` after the unowned projection import issue is resolved.
- Update downstream audit scripts/tests outside Agent E ownership that still expect `partial_passed`.

