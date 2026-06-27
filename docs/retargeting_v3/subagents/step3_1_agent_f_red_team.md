# Step 3.1 Agent F Red Team Handoff

Date: 2026-06-27

verdict = PASS
strict_final_acceptance = BLOCKED_PENDING_FINAL_HEAD_CI

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

workflow_run_id: pending final HEAD CI success
head_sha: pending final HEAD CI success
conclusion: pending final HEAD CI success
job conclusions: pending final HEAD CI success

### Integrator evidence-closure note

The previous branch HEAD `bff97b3d703a2899190a47b9e681d6444a07fb02` had no visible commit workflow runs and no combined-status entries when checked through the GitHub connector on 2026-06-27. This handoff update intentionally touches a workflow-watched path on `retargeting-v3-step3-full-fleet-hardening` so the push workflow can produce current-branch CI evidence.

Do not promote Step 3.1.1 to strict final PASS until concrete `workflow_run_id`, `head_sha`, `conclusion=success`, and all required job conclusions are copied back into this section and/or `acceptance_ledger.json`, then `scripts/audit_retargeting_v3_step3_full_fleet.py --require-final-head-ci` passes.

## Full Repo Pytest

full repo pytest classification: `full_repo_failed`

full repo pytest caveat: Full repo pytest timed out after 2700 seconds with 41
passing dots and no visible failure text; it is classified as failed and is not
reported as passed.

## Scoped Test Evidence Policy

`artifacts/retargeting_v3_step3_runtime_quality/test_results/pytest_summary.json`
currently remains an artifact-writer placeholder with `status=not_run_by_artifact_writer`. For this evidence-closure round, targeted Step 3.1.1/v3 tests and the GitHub Actions jobs listed above are the scoped acceptance gates; full-repo pytest remains a documented non-blocking caveat unless it is re-run to exit code 0.

## Verification

Local hardening verification before clean artifact regeneration:

```text
tests/v3/test_step3_full_fleet_acceptance_audit.py tests/v3/test_step3_runtime_quality_gates_status_semantics.py: 20 passed
selected Step 3 runtime-quality suite excluding compact runner: 33 passed
compact full-fleet runner suite: 3 passed
```

The current committed artifact tree is an honest Step 3.1.1 full-fleet evaluation milestone: clean provenance is recorded, residual-only full-humanoid rows are not promoted to `runtime_quality_passed`, and quality counts remain `0 passed / 23 warned / 9 failed` for full humanoids.

remaining blockers = final_head_ci_evidence
