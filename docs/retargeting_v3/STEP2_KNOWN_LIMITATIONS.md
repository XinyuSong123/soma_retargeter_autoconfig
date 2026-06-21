# Step 2 Known Limitations

Status: **BLOCKED**

This file records current Step 2 limitations without changing artifacts or
lowering gates.

## Current Blockers

- The hardened Agent F audit now blocks on checked-in artifacts that still
  contain local absolute paths, missing reproducibility files, cross-format
  gates left `not_run`, incomplete canonical motion suites, and failed epsilon
  stability gates in reports marked `passed` or `partial_passed`.
- Required artifact files are not present at the final Step 2 paths:
  `acceptance_ledger.json`, `test_results/pytest.txt`,
  `test_results/junit.xml`, and `test_results/coverage.json`.
- `cross_format.json` is still a scaffold: `same_source_strict` and
  `variant_compatibility` are not completed passes.
- Current terminal humanoid reports use a 12-motion canonical sequence and do
  not include all 15 required motions from `goal.md`.
- Several passed/partial reports still record
  `epsilon_stability_gate_passed=false`.
- Required positive and negative Robot Zoo entries still include
  `source_unavailable` and `model_load_failed` statuses.
- The current local branch is not the required pushed target branch, and there
  is no current-commit CI status evidence.

## Scope Boundaries

- No production `NewtonPipeline` integration has been started here.
- No private or company-internal robot asset is intentionally used by the
  Step 2 manifest.
- This pass does not regenerate artifacts; it only hardens the audit and CI
  gates so existing blockers are reported honestly.
