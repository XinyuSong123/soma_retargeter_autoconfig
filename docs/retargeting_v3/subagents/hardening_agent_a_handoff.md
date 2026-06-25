# Hardening Agent A Handoff

Agent: A - Baseline Truth and Real Before/After

Status: implemented true baseline extraction; current after artifacts remain blocked.

## Files Changed

- `soma_retargeter/robotics/v3/failure_analysis.py`
- `scripts/extract_capability_failure_ledger.py`
- `scripts/build_capability_before_after.py`
- `artifacts/retargeting_v3_step2_capability/baseline_summary.json`
- `artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json`
- `artifacts/retargeting_v3_step2_capability/before_after.json`
- `tests/v3/test_capability_true_baseline_git.py`
- `docs/retargeting_v3/STEP2_3_1_TRUE_BASELINE.md`
- `docs/retargeting_v3/subagents/hardening_agent_a_handoff.md`

## Implementation

- Added git-object baseline loading from
  `5ad5a001c445c525d4c8bbaf6339dec5c5c2c719`.
- Added fail-closed `BaselineGitObjectError` for missing commits or missing
  baseline object paths.
- Added `baseline_summary.json` with immutable 44-model baseline counts:
  21 `passed`, 3 `partial_passed`, 9 `negative_control_passed`, and
  11 `algorithm_failed`.
- Rebuilt `baseline_failure_ledger.json` from pinned commit objects, including
  semantic-map evidence from the same commit.
- Added `scripts/build_capability_before_after.py` and regenerated
  `before_after.json` with true baseline statuses.
- Kept thresholds and robot-specific logic unchanged.

## Current Truth

The true transition matrix reports:

```text
algorithm_failed->capability_limited_passed  10
algorithm_failed->passed                      1
negative_control_passed->negative_control_passed  9
partial_passed->partial_passed                3
passed->capability_limited_passed            13
passed->passed                                8
```

`transition_validation.status=failed` because 13 baseline `passed` models are
currently marked `capability_limited_passed`.

`final_count_validation.status=failed` because current after counts are:

```text
passed                       9
capability_limited_passed   23
partial_passed               3
negative_control_passed      9
```

The 11 old algorithm failures are now explicitly reported:

```text
algorithm_failed->capability_limited_passed  10
algorithm_failed->passed                      1
```

## Commands Run

Initial failing test:

```bash
PYTHONPATH=. pytest tests/v3/test_capability_true_baseline_git.py -q
```

Result: failed during collection because the true-baseline API did not exist.

Artifact regeneration:

```bash
PYTHONPATH=. python scripts/extract_capability_failure_ledger.py \
  --output artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json
```

Result: PASS, wrote 50 failures across 11 robots from `git_object`.

```bash
PYTHONPATH=. python scripts/build_capability_before_after.py \
  --baseline-output artifacts/retargeting_v3_step2_capability/baseline_summary.json \
  --output artifacts/retargeting_v3_step2_capability/before_after.json
```

Result: PASS, wrote artifacts with
`transition_validation=failed` and `final_count_validation=failed`.

Targeted tests:

```bash
PYTHONPATH=. pytest tests/v3/test_capability_true_baseline_git.py -q
```

Result: 4 passed.

```bash
PYTHONPATH=. pytest \
  tests/v3/test_capability_failure_ledger_extraction.py \
  tests/v3/test_capability_failure_ledger_artifact.py -q
```

Result: 6 passed.

Check modes:

```bash
PYTHONPATH=. python scripts/build_capability_before_after.py --check
```

Result: PASS, committed artifacts match generated truth; transition and final
count validations are failed.

```bash
PYTHONPATH=. python scripts/build_capability_before_after.py --check --strict-transitions
```

Result: exit code 2. Committed artifacts match generated truth, but strict mode
correctly fails because `transition_validation=failed`.

```bash
PYTHONPATH=. python scripts/extract_capability_failure_ledger.py --check
```

Result: PASS, 50 failures across 11 robots from `git_object`.

## Blockers and Integration Notes

- Agent A did not modify unowned `soma_retargeter/robotics/v3/validation.py`.
  Its embedded `_before_after_matrix()` still needs integration by the owning
  status/artifact path, or future summary generation may recreate the legacy
  fabricated `partial_passed` before statuses.
- `artifacts/retargeting_v3_step2_capability/summary.json` is outside Agent A
  ownership and still embeds the old fabricated `before_after` payload.
- Current after artifacts are not acceptable under the true baseline: the 13
  `passed->capability_limited_passed` transitions must be resolved by the
  profile status owner, then Agent A artifacts should be regenerated.
- `--strict-transitions` is available on
  `scripts/build_capability_before_after.py` for final acceptance checks; it
  returns nonzero on the current artifacts.
