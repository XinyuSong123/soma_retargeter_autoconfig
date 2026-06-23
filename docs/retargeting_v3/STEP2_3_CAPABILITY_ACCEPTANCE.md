# Step 2.3 Capability Acceptance

Date: 2026-06-23

Scope: independent red-team audit over the freshly regenerated Step 2.3 capability artifacts, Step 2.1 numerical threshold artifact, Git LFS payload state, and provenance metadata. This step does not relax capability criteria.

## Gates

- Projection thresholds are locked to exact global values: neutral `1e-3`, foot `0.06`, hand `0.12`, torso `0.08`, default `0.05`.
- Threshold calibration must remain global. Per-robot overrides, robot-ID keys, or special-case threshold exceptions are audit failures.
- Core capability math files must not contain Robot Zoo model IDs.
- Ordinary projection residuals above threshold require a serialized `kkt_certificate` with stationarity, complementarity, feasibility, task-gradient, prior-cancellation, and seed-consistency evidence.
- Rank-zero projections must preserve demand through `demand_residual`, `unreachable_demand`, and `rank_zero_reason`.
- Compatible torso-axis conditioning must remain present in rest-frame code and represented in per-robot arm calibration evidence.
- Canonical target geometry failures cannot be packaged as capability pass statuses.
- `extreme_but_valid_joint_limit_stress` is diagnostic only and cannot mask ordinary motion failures.
- Negative controls must remain separate from humanoid profile passes.
- Deterministic rerun must cover and compare all 44 in-scope models.
- Git LFS must pass `git lfs fsck`; validation must not run against pointer-only files.
- Robot Zoo snapshots must not leak meshes, texture payloads, upstream VCS directories, or fetch-only assets through LFS.
- Accepted artifacts must record clean provenance.

## Current Verdict

Status: **PASS**

The fixture tests for the red-team audit pass, and the live audit passes against the accepted artifact set.

Accepted artifact gates:

- All 44 locked-scope models are terminal pass statuses: 9 `passed`, 23 `capability_limited_passed`, 3 `partial_passed`, and 9 `negative_control_passed`.
- Deterministic rerun compared and matched 44/44 models.
- Red-team capability audit passed.

Checks covered by the live audit:

- Exact threshold artifact values and profile source literals remained unchanged.
- No Robot Zoo model IDs were found in core capability math files.
- Residuals over threshold include `kkt_certificate` evidence.
- Rank-zero projection evidence fields are present for rank-zero reports.
- Compatible torso-axis conditioning remains present.
- Negative controls are not counted as humanoid profile passes.
- Git LFS fsck passed and no pointer-only, mesh, upstream, or fetch-only snapshot leakage was found.
- Step 2.3 provenance records clean artifact generation state.

## Commands

```bash
PYTHONPATH=. python -m pytest -q tests/v3/test_capability_acceptance_*.py
git lfs fsck
PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py \
  --artifact-dir artifacts/retargeting_v3_step2_capability \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --write-report artifacts/retargeting_v3_step2_capability/capability_audit.json
```

Artifacts:

- `artifacts/retargeting_v3_step2_capability/test_results/pytest.txt`
- `artifacts/retargeting_v3_step2_capability/test_results/junit.xml`
- `artifacts/retargeting_v3_step2_capability/capability_audit.json`
