# Step 2.3.1 Capability Acceptance

Status: PASS

Step 2.3.1 capability acceptance is complete for branch
`retargeting-v3-step2-capability-acceptance-hardening`.

## Final Evidence

- Final pushed head before this documentation update: `22e9c4206c2de37d39ecafcd0224fdf89c4e7da2`
- Clean source commit: `2651709798c0f8f18b3a691b132c1d95d0bb08f7`
- Published artifact commit: `f10a7af99971eb85b465d21a6b7e3250b6de207c`
- Capability CI workflow: `Retargeting V3 Capability Acceptance`
- Previous successful workflow run: `28215929924`
- Final artifacts: `artifacts/retargeting_v3_step2_capability/`

## Required Evidence

The final artifact tree contains:

- `environment.json`
- `lfs_state.json`
- `commands.txt`
- `acceptance_ledger.json`
- `baseline_summary.json`
- `baseline_failure_ledger.json`
- `before_after.json`
- `certificate_thresholds.json`
- `summary.json`
- `capability_matrix.json`
- `target_geometry_matrix.json`
- `deterministic_rerun.json`
- `cross_format.json`
- `deferred_snapshots.json`
- `source_inventory.json`
- `load_matrix.json`
- `semantic_matrix.json`
- `validation_checks.json`
- `capability_audit.json`
- `per_robot/*.json`
- `per_task/*.json`
- `failures/*.json`
- `test_results/pytest.txt`
- `test_results/junit.xml`
- `test_results/pytest_summary.json`

## Acceptance Results

- Baseline commit: `5ad5a001c445c525d4c8bbaf6339dec5c5c2c719`
- Baseline counts: `passed=21`, `partial_passed=3`, `negative_control_passed=9`, `algorithm_failed=11`
- Final counts: `passed=32`, `partial_passed=3`, `negative_control_passed=9`, `algorithm_failed=0`
- Transition counts: `algorithm_failed->passed=11`, `passed->passed=21`, `partial_passed->partial_passed=3`, `negative_control_passed->negative_control_passed=9`
- Deterministic rerun: `44 compared`, `44 matched`, `0 mismatches`
- LFS state: `Git LFS fsck OK`, pointer files `0`, missing LFS objects `0`, materialized snapshots `38`
- Full pytest artifact: `365 passed, 10 skipped, 148 warnings in 5354.42s (1:29:14)`
- Live capability audit: `status=passed`, `failure_count=0`

## Audit Command

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py \
  --artifact-dir artifacts/retargeting_v3_step2_capability \
  --numerical-artifact-dir artifacts/retargeting_v3_step2_numerical \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --write-report artifacts/retargeting_v3_step2_capability/capability_audit.json
```

## Agent F

`docs/retargeting_v3/subagents/capability_agent_f_red_team.md` records
`verdict = PASS`, concrete source/artifact/workflow metadata, live audit PASS,
LFS fsck PASS, and `remaining blockers = 0`.
