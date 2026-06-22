# Numerical Agent F Handoff: Red Team

Scope: audit, CI, docs, artifacts, and process gates.

Findings incorporated:

- Old Step 2 audit passes but does not validate Step 2.1 numerical-core requirements.
- New artifacts, before/after matrix, threshold calibration, CI workflow, and six numerical handoffs were missing.
- The audit must reject corrected epsilon-only failures and missing engine-source metadata.

Integrated result:

- Added `scripts/audit_retargeting_v3_numerical_core.py`.
- Added `.github/workflows/retargeting_v3_numerical_core.yml`.
- Added `docs/retargeting_v3/STEP2_NUMERICAL_ACCEPTANCE.md`.
- Added six numerical handoff documents in this directory.
- `python scripts/audit_retargeting_v3_numerical_core.py --artifact-dir artifacts/retargeting_v3_step2_numerical` returns PASS.

Residual risk:

- The committed artifact was generated with `--low-discrepancy-count 1` for turnaround during this repair. The script supports default sampling for the slower full run.
