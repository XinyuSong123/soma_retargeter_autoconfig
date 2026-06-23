# Capability Agent F Red Team Handoff

Date: 2026-06-23

## Scope

Owned files only:

- `scripts/audit_retargeting_v3_capability.py`
- `.github/workflows/retargeting_v3_capability.yml`
- `tests/v3/test_capability_acceptance_*.py`
- `docs/retargeting_v3/STEP2_3_CAPABILITY_ACCEPTANCE.md`
- `docs/retargeting_v3/subagents/capability_agent_f_red_team.md`
- `artifacts/retargeting_v3_step2_capability/test_results/`

No branch was created, no push was performed, and no other agents' files were reverted.

## Work Performed

- Added failing-first red-team fixture tests, then implemented the audit.
- Added a read-only live audit over Assets44, numerical threshold calibration, core source special-case scans, LFS policy, and clean provenance.
- Added CI workflow `Retargeting V3 Capability Acceptance`.
- Wrote local test and audit artifacts under the owned capability test-results directory.

## Verification

```bash
PYTHONPATH=. python -m pytest -q tests/v3/test_capability_acceptance_red_team.py
PYTHONPATH=. python -m pytest -q tests/v3/test_capability_acceptance_*.py
git lfs fsck
PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py \
  --write-report artifacts/retargeting_v3_step2_capability/test_results/capability_audit.json
```

Results:

- Fixture tests: `13 passed`.
- Git LFS fsck: `Git LFS fsck OK`.
- Live capability audit: **FAIL**, with 49 findings.

## Verdict

**Do not accept Step 2.3 capability yet.**

The current artifact set fails two hard red-team requirements:

- Over-threshold ordinary residuals lack KKT certificates. The live audit found 45 task reports above threshold without certificate evidence.
- Deterministic rerun does not compare all 44 in-scope models. It reports `compared_count=33`, `matched_count=33`, and `skipped_non_pass_count=11`.

## Residual Risk

- The audit defines the KKT certificate schema, but core projection code does not emit that certificate yet.
- Deterministic comparison currently skips non-pass models, so it cannot detect unstable failure evidence for those 11 models.
- CI will intentionally fail until those evidence gaps are closed.

## Unfinished Items

- Add KKT certificate serialization for over-threshold residual reports.
- Compare deterministic evidence for all 44 in-scope models, including algorithm-failed models.
- Rerun the live capability audit and update `capability_audit.json` after the above fixes.
