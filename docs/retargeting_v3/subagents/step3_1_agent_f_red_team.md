# Step 3.1 Agent F Red Team Handoff

Date: 2026-06-27

verdict = PASS

## Scope

Agent F owns the independent acceptance audit for Step 3.1 full-fleet runtime
quality evidence under `artifacts/retargeting_v3_step3_runtime_quality`.

## Audit Contract

The audit must fail closed unless the artifact tree proves all of the following:

- Full-fleet matrix has exactly `44` unique model rows.
- Status partition is exactly `passed=32`, `partial_passed=3`, and `negative_control_passed=9`.
- Evidence is not RPO/G1-only; at least `40` rows must be outside the Step 3.0 RPO/G1 smoke pair.
- Required artifact files are present: environment, commands, matrix, summary, pipeline controls, acceptance ledger, and pytest result files.
- Negative controls remain `negative_control_not_promoted` and are not accepted as runtime quality passes.
- Runtime quality numeric fields are finite, non-negative, ordered where applicable, and record zero NaN/Inf output samples.
- Artifact and handoff text do not leak local absolute paths.
- Pipeline controls record disabled/shadow no-op coverage, explicit override gating, fingerprint gating, negative-control exclusion, and sanitized artifacts.
- Full repo pytest evidence includes an explicit caveat and classification.

## Current Evidence

The live Step 3.1 artifact directory is present at
`artifacts/retargeting_v3_step3_runtime_quality` and passes the full-fleet
acceptance audit with `blocking_count=0`.

## Full Repo Pytest

- full repo pytest classification: `full_repo_failed`
- full repo pytest caveat: Full repo pytest was interrupted after 31m15s with 41 passed and no visible test failures before interruption; it is not reported as passed.

## Commands

```bash
PYTHONPATH=. python -m pytest -q tests/v3/test_step3_full_fleet_acceptance_audit.py
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_full_fleet.py \
  --artifact-dir artifacts/retargeting_v3_step3_runtime_quality \
  --source-root . \
  --output-json artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json
```

## Remaining Blockers

None for Step 3.1 acceptance audit.
