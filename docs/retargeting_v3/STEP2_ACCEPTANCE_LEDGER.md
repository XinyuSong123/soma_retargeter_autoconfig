# Step 2 Acceptance Ledger

Status: **PASS**
Owner: Integrator
Reasoning strength: **xhigh**
Formal pytest artifact: `artifacts/retargeting_v3_step2/test_results/pytest.txt`
Acceptance audit artifact: `artifacts/retargeting_v3_step2/acceptance_ledger.json`

## Current Status

The current formal pytest artifact is green:

```text
215 passed, 10 skipped, 146 warnings in 3046.97s (0:50:46)
```

The current acceptance ledger is `PASS` with `blocking_count=0`.

| Gate | Subject | Current evidence |
|---|---|---|
| `arbitrary_g1_equivalence` | `validation_checks.g1_mjcf_urdf_equivalence` | `status="passed"`, `gate_a_status="complete_passed"`, semantic FK, active chains, rank summary, and canonical projection all passed. |
| `cross_format_gates_not_run` | `cross_format.gates.same_source_strict` | `status="passed"` with complete Gate A evidence. |
| `cross_format_gates_not_run` | `cross_format.gates.variant_compatibility` | `status="passed"` for the independently passing G1 URDF/MJCF pair; non-passing pairs are recorded as not eligible. |

`validation_checks.json` and `cross_format.json` now record the accepted G1
same-source strict pass and the G1 variant compatibility pass. Vendor URDF vs
Menagerie MJCF is still recorded as variant compatibility, not strict
same-source equivalence.

## Current Artifact Summary

- Robot Zoo status counts: `passed=7`, `negative_control_passed=4`,
  `algorithm_failed=14`, `semantic_failed=3`, `model_load_failed=2`,
  `source_unavailable=16`.
- `algorithm_pass_count=11`.
- `deterministic_rerun.status="passed"` with 11 matched models, 16
  source-unavailable models, and 19 skipped non-pass statuses.
- `cross_format.gates.same_source_strict.status="passed"`.
- `cross_format.gates.variant_compatibility.status="passed"`.

## Superseded Historical Record

The following Agent F record is retained for provenance only. It is
**superseded/stale** and must not be cited as current acceptance status.

Historical status: **FALSE-POSITIVE AUDIT PASS; FULL STEP 2 NOT COMPLETE**
Historical owner: Agent F - Reproducibility, CI & Independent Red Team
Historical reasoning strength: **xhigh**
Historical audit artifact:
`artifacts/retargeting_v3_step2/test_results/acceptance_audit.json`

The old record stated that `scripts/audit_retargeting_v3_step2.py` returned
`PASS`, `tests/v3/test_acceptance_gates_audit.py` passed, the G1 same-source
URDF-to-canonical-MJCF equivalence was recorded as `passed`, and the full
no-fetch Robot Zoo run had `passed=1` and `source_unavailable=45`.

Those statements describe an earlier false-positive-audit snapshot. They are
not current: the formal pytest artifact now reports `215 passed, 10 skipped`,
and the current acceptance audit is `PASS` with zero blocking findings.

## Acceptance Rule

Step 2 may be reported complete only together with the current PASS ledger,
formal pytest/JUnit/coverage artifacts, and clean-head generation metadata.
