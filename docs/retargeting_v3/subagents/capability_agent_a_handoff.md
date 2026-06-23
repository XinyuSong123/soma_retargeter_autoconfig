# Capability Agent A Handoff

Reasoning strength: xhigh

Scope: Step 2.3 compact failure oracle and baseline ledger for the 11 existing
algorithm-failed reports under `artifacts/retargeting_v3_step2_assets44/failures`.

## Changed Files

- `soma_retargeter/robotics/v3/failure_analysis.py`
- `scripts/extract_capability_failure_ledger.py`
- `tests/v3/test_capability_failure_ledger_extraction.py`
- `tests/v3/test_capability_failure_ledger_artifact.py`
- `artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json`
- `docs/retargeting_v3/STEP2_3_FAILURE_BASELINE.md`
- `docs/retargeting_v3/subagents/capability_agent_a_handoff.md`

## Commands Run

- `git status --short`
- `rg --files artifacts/retargeting_v3_step2_assets44/failures soma_retargeter/robotics/v3 scripts tests/v3 docs/retargeting_v3 | sort`
- `find artifacts/retargeting_v3_step2_assets44/failures -maxdepth 1 -type f -name '*.json' -printf '%f\n' | sort`
- Read-only Python JSON probes over the 11 failure reports and semantic maps.
- `PYTHONPATH=. pytest -q tests/v3/test_capability_failure_ledger_*.py`
  - Initial expected result: failed on missing module/script before implementation.
- `PYTHONPATH=. pytest -q tests/v3/test_capability_failure_ledger_extraction.py`
  - Result after implementation: `4 passed`.
- `PYTHONPATH=. pytest -q tests/v3/test_capability_failure_ledger_artifact.py`
  - Expected pre-generation result: failed on missing baseline ledger artifact.
- `PYTHONPATH=. python scripts/extract_capability_failure_ledger.py --failure-dir artifacts/retargeting_v3_step2_assets44/failures --output artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json`
  - Result: wrote 50 failures across 11 robots.
- `PYTHONPATH=. pytest -q tests/v3/test_capability_failure_ledger_*.py`
  - Result: `6 passed`.

## Numeric Counts

- Frozen robots: 11.
- Baseline failure rows: 50.
- `projection_position_normalized_residual`: 43.
- `projection_rotation_normalized_residual`: 2.
- `numerical_stability_gate`: 5.

Robot counts:

- `atlas_drc_urdf`: 2
- `atlas_v4_urdf`: 1
- `booster_t1_mjcf`: 2
- `booster_t1_urdf`: 2
- `fourier_n1_mjcf`: 3
- `jaxon_urdf`: 26
- `mujoco_humanoid_mjcf`: 7
- `pal_talos_mjcf_direct`: 2
- `robotis_op3_mjcf`: 2
- `talos_urdf`: 1
- `valkyrie_urdf`: 2

## Ledger Contract

The ledger preserves every source failure as `status: failed`. It records robot ID, motion, task, metric type, exact thresholds, actual normalized/absolute residuals or numerical gate values, active coordinates, active joint limits, rank fields, engine Jacobian source, solver status/message, semantic source/hash, chain length evidence, and target distance/angle.

## Risks

- The source reports contain non-standard JSON numeric values for unbounded joint limits (`Infinity` and `-Infinity`). The new ledger preserves those values because they are the exact runtime joint-limit evidence used by the reports.
- `artifacts/retargeting_v3_step2_capability/` contains other pre-existing or concurrently generated untracked artifacts outside this handoff scope. This handoff only owns `baseline_failure_ledger.json`.
- The ledger is a baseline oracle over the existing Step 2.2 failure reports. If those reports are regenerated, rerun the extractor and check mode rather than manually editing counts.

## Unfinished Items

None for Agent A scope.
