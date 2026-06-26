# Capability Agent F Red Team Handoff

Date: 2026-06-26

verdict = PASS

## Scope

Owned evidence reviewed:

- `scripts/audit_retargeting_v3_capability.py`
- `.github/workflows/retargeting_v3_capability.yml`
- `tests/v3/test_capability_acceptance_*.py`
- `docs/retargeting_v3/STEP2_3_1_CAPABILITY_ACCEPTANCE.md`
- `docs/retargeting_v3/subagents/capability_agent_f_red_team.md`
- `artifacts/retargeting_v3_step2_capability/`

## Final Evidence

- final_head: `22e9c4206c2de37d39ecafcd0224fdf89c4e7da2`
- source_code_commit: `2651709798c0f8f18b3a691b132c1d95d0bb08f7`
- artifact_commit: `f10a7af99971eb85b465d21a6b7e3250b6de207c`
- workflow_run_id: `28215929924`
- workflow name: `Retargeting V3 Capability Acceptance`
- workflow head SHA: `22e9c4206c2de37d39ecafcd0224fdf89c4e7da2`
- workflow conclusion: `success`
- CI job conclusions: `capability-artifact-live-audit=success`, `lfs-snapshot-smoke=success`, `capability-synthetic-tests=success`

The `final_head` above is the CI-audited branch head before this handoff document update.

## Verification

- pytest summary: `365 passed, 10 skipped, 148 warnings in 5354.42s (1:29:14)`
- live audit command: `PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py --artifact-dir artifacts/retargeting_v3_step2_capability --numerical-artifact-dir artifacts/retargeting_v3_step2_numerical --manifest assets/robot_zoo/robot_zoo_manifest.json --lock assets/robot_zoo/robot_zoo_lock.json --write-report artifacts/retargeting_v3_step2_capability/capability_audit.json`
- live audit PASS: `artifacts/retargeting_v3_step2_capability/capability_audit.json` has `status=passed`, and CI job `capability-artifact-live-audit` completed successfully.
- LFS fsck PASS: `Git LFS fsck OK`
- deterministic rerun: `44 compared`, `44 matched`, `0 mismatches`
- before/after validation: `algorithm_failed->passed=11`, `passed->passed=21`, `partial_passed->partial_passed=3`, `negative_control_passed->negative_control_passed=9`

remaining blockers = 0
