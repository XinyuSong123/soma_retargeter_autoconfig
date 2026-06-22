# Numerical Agent E Handoff

Reasoning strength: xhigh

Scope: available-model before/after matrix and failure taxonomy.

Final files:

- `scripts/write_retargeting_v3_numerical_artifacts.py`
- `artifacts/retargeting_v3_step2_numerical/`
- `docs/retargeting_v3/STEP2_NUMERICAL_ACCEPTANCE.md`

Validation:

- `PYTHONPATH=. python scripts/write_retargeting_v3_numerical_artifacts.py --deterministic-rerun`
- `PYTHONPATH=. python scripts/audit_retargeting_v3_numerical_core.py --artifact-dir artifacts/retargeting_v3_step2_numerical`

Result: PASS. The artifact covers all 46 manifest entries, keeps semantic/load/source failures unchanged, reports `epsilon_only_failures_after=0`, improves corrected pass count from 7 baseline pass to 16, and leaves 5 corrected algorithm failures with non-epsilon evidence.

Related detailed note: `numerical_agent_e_available_models.md`.
