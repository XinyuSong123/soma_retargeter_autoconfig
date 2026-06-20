# Step 2 Acceptance Ledger

Status: **BLOCKED**
Owner: Step2 docs/status worker
Reasoning strength: **xhigh**
Formal pytest artifact: `artifacts/retargeting_v3_step2/test_results/pytest.txt`
Acceptance audit artifact: `artifacts/retargeting_v3_step2/acceptance_ledger.json`

## Current Status

The current formal pytest artifact is green:

```text
209 passed, 10 skipped, 146 warnings in 2630.08s (0:43:50)
```

This is not a Step 2 acceptance pass. The current acceptance ledger is
`BLOCKED` with `blocking_count=3`.

| Gate | Subject | Current evidence |
|---|---|---|
| `arbitrary_g1_equivalence` | `validation_checks.g1_mjcf_urdf_equivalence` | `status="incomplete"`; the same-source generated MJCF coordinate comparison exists, but strict Step 2 Gate A evidence is not complete. |
| `cross_format_gates_not_run` | `cross_format.gates.same_source_strict` | `status="incomplete"`; Gate A is missing semantic FK, active-chain, rank-summary, and canonical-projection evidence. |
| `cross_format_gates_not_run` | `cross_format.gates.variant_compatibility` | `status="blocked"`; Gate B still requires semantic FK, common-chain, rank, DoF-difference, and projection evidence for independently passing variants. |

`validation_checks.json` and `cross_format.json` must not be read as a G1
strict equivalence pass. Current Step 2 remains blocked until Gate A, Gate B,
and the arbitrary-G1-equivalence audit finding are closed.

## Current Artifact Summary

- Robot Zoo status counts: `passed=7`, `negative_control_passed=4`,
  `algorithm_failed=14`, `semantic_failed=3`, `model_load_failed=2`,
  `source_unavailable=16`.
- `algorithm_pass_count=11`.
- `deterministic_rerun.status="passed"` with 11 matched models, 16
  source-unavailable models, and 19 skipped non-pass statuses.
- `cross_format.gates.same_source_strict.status="incomplete"`.
- `cross_format.gates.variant_compatibility.status="blocked"`.

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
not current: the formal pytest artifact now reports `209 passed, 10 skipped`,
and the current acceptance audit is `BLOCKED` on the G1/cross-format findings
listed above.

## Acceptance Rule

Step 2 may not be reported complete or `PASS` while any acceptance audit
finding remains. A green pytest run is necessary evidence, but it is not
sufficient acceptance evidence.
