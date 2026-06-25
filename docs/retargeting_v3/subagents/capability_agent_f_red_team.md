# Capability Agent F Red Team Handoff

Date: 2026-06-25

verdict = BLOCKED

## Scope

Owned files only:

- `scripts/audit_retargeting_v3_capability.py`
- `.github/workflows/retargeting_v3_capability.yml`
- `tests/v3/test_capability_acceptance_*.py`
- `docs/retargeting_v3/STEP2_3_1_CAPABILITY_ACCEPTANCE.md`
- `docs/retargeting_v3/subagents/capability_agent_f_red_team.md`
- `artifacts/retargeting_v3_step2_capability/test_results/`
- `artifacts/retargeting_v3_step2_capability/capability_audit.json`

No branch was created and no push was performed.

## Current Head

- final_head: `b84f12000599e41af34d3e2a9ace19410cc870be`
- source_code_commit: not final integrated evidence
- artifact_commit: not final integrated evidence
- workflow_run_id: not available in this local red-team pass

## Work Performed

- Added red-team fixture coverage for fabricated baseline status, unresolvable commits, wrong workflow branch/root, missing pytest/JUnit/LFS state, synthetic KKT fields, missing raw evidence, seed endpoint divergence, continuation alpha, stress-only status changes, baseline downgrades, partial deterministic comparison, LFS pointer-only validation, stale handoff text, and stale Step 2.2 ledgers.
- Hardened `scripts/audit_retargeting_v3_capability.py` to require final artifact evidence, frozen baseline provenance, independent KKT recomputation from raw numeric evidence, task-space seed consensus, continuation completion, deterministic comparison surfaces, LFS materialization, workflow config, and current handoff state.
- Updated `.github/workflows/retargeting_v3_capability.yml` to use branch `retargeting-v3-step2-capability-acceptance-hardening`, the capability artifact root, LFS pull/fsck, and required hosted jobs.

## Local Verification

```bash
PYTHONPATH=. python -m pytest -q tests/v3/test_capability_acceptance_red_team.py
PYTHONPATH=. python -m pytest -q tests/v3/test_capability_acceptance_*.py
git lfs fsck
PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py \
  --artifact-dir artifacts/retargeting_v3_step2_capability \
  --numerical-artifact-dir artifacts/retargeting_v3_step2_numerical \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --write-report artifacts/retargeting_v3_step2_capability/capability_audit.json
```

Results:

- Red-team fixture file: `27 passed`.
- Owned capability acceptance glob: `29 passed`.
- Git LFS fsck: `Git LFS fsck OK`.
- Live capability audit: `failed`, `failure_count=82`.

## Live Audit State

The live audit is still expected to fail before integration because the committed capability artifact tree is incomplete for the hardened acceptance contract. Known blockers include:

- Missing `test_results/pytest.txt`, `test_results/junit.xml`, and `test_results/pytest_summary.json`.
- Missing `certificate_thresholds.json`, `capability_matrix.json`, `per_task/`, and non-empty `failures/`.
- Baseline-passed models are currently downgraded to `capability_limited_passed`.
- Several `capability_limited_passed` models have no ordinary limited evidence or are stress-only status changes.
- Current KKT certificates lack the required raw `J/e/q/bounds` evidence for independent recomputation.
- Negative-control deterministic rows lack the required comparison surfaces.

remaining blockers > 0
