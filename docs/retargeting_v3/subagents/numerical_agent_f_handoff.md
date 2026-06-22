# Numerical Agent F Handoff

Reasoning strength: xhigh

Scope: red-team audit, CI workflow, acceptance tests, and final verdict.

Final files:

- `scripts/audit_retargeting_v3_numerical_core.py`
- `.github/workflows/retargeting_v3_numerical_core.yml`
- `tests/v3/test_numerical_core_acceptance_audit.py`
- `docs/retargeting_v3/STEP2_NUMERICAL_ACCEPTANCE.md`

Validation:

- `PYTHONPATH=. pytest -q tests/v3/test_numerical_core_acceptance_audit.py`
- `PYTHONPATH=. python scripts/audit_retargeting_v3_numerical_core.py --artifact-dir artifacts/retargeting_v3_step2_numerical`

Result: PASS. The strict audit rejects epsilon-only corrected failures, per-model baseline pass regressions, missing engine-primary metadata, missing rank/subspace evidence, and incomplete before/after schema.

Related detailed note: `numerical_agent_f_red_team.md`.
