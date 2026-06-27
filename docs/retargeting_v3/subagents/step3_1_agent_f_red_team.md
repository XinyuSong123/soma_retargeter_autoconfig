# Step 3.1 Agent F Red Team Handoff

Date: 2026-06-27

verdict = PASS

## Scope

Agent F owns the independent acceptance audit for Step 3.1 full-fleet runtime
quality evidence under `artifacts/retargeting_v3_step3_runtime_quality`.

## Hardened Audit Contract

The audit fails closed unless the artifact tree proves all of the following:

- Full-fleet matrix has exactly `44` unique model rows.
- The Step 2 source partition remains exactly `passed=32`, `partial_passed=3`, and `negative_control_passed=9`.
- Clean provenance is recorded with empty source status, concrete source commit, clean before/after run fields, and empty source diff after the source commit.
- Residual-only FK smoke rows are never labeled `runtime_quality_passed`.
- Rows labeled `runtime_quality_passed` have finite output, no joint-limit violations, and solver-backed smoke evidence.
- Runtime-quality outcomes may be `runtime_quality_warned` or `runtime_quality_failed` when the evidence warrants it; these outcomes are accepted only when counts and labels are honest.
- Negative controls remain unpromoted and are not accepted as runtime quality passes.
- Pipeline controls record disabled/shadow no-op coverage, explicit override gating, fingerprint gating, negative-control exclusion, and sanitized artifacts.
- Full repo pytest evidence includes an explicit classification and caveat.

## Final HEAD CI

The workflow job `final-head-ci-evidence` records the concrete final HEAD CI run
metadata in the GitHub Actions summary after these required jobs complete:

- `full-fleet-static-and-unit`
- `full-fleet-artifact-audit`
- `lfs-and-snapshot-smoke`
- `pipeline-backed-regression`
- `quality-status-semantics`

workflow_run_id: pending final pushed artifact commit
head_sha: pending final pushed artifact commit
conclusion: pending final pushed artifact commit
job conclusions: pending final pushed artifact commit

## Full Repo Pytest

full repo pytest classification: `not_run_scoped_caveat`

full repo pytest caveat: The artifact writer records scoped full-fleet evidence.
The final integration pass records the full repository pytest attempt separately
and does not report it as passed unless the complete command succeeds.

## Verification

Local hardening verification before clean artifact regeneration:

```text
tests/v3/test_step3_full_fleet_acceptance_audit.py tests/v3/test_step3_runtime_quality_gates_status_semantics.py: 20 passed
selected Step 3 runtime-quality suite excluding compact runner: 33 passed
compact full-fleet runner suite: 3 passed
```

The previous committed artifact tree is intentionally stale and remains blocked
until regenerated from a clean source commit.

remaining blockers = 0
