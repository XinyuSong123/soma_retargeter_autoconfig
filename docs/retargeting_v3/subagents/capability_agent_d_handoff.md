# Capability Agent D Handoff

Area: target geometry and variant differential audit.
Backend: `newton`.
Artifact: `artifacts/retargeting_v3_step2_capability/target_geometry_matrix.json`.

## Changed Files

- `soma_retargeter/robotics/v3/target_geometry_audit.py`
- `tests/v3/test_failure_robot_target_geometry_oracles.py`
- `tests/v3/test_failure_robot_target_geometry_matrix.py`
- `artifacts/retargeting_v3_step2_capability/target_geometry_matrix.json`
- `docs/retargeting_v3/STEP2_3_TARGET_GEOMETRY_AUDIT.md`
- `docs/retargeting_v3/subagents/capability_agent_d_handoff.md`

No semantic maps or semantic expectations were changed.

## Commands

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_failure_robot_target_geometry_oracles.py \
  tests/v3/test_failure_robot_target_geometry_matrix.py
```

Result: `6 passed in 10.67s`.

```bash
PYTHONPATH=. python -m soma_retargeter.robotics.v3.target_geometry_audit \
  --backend newton \
  --output artifacts/retargeting_v3_step2_capability/target_geometry_matrix.json
```

Result: completed.

```bash
python -m json.tool \
  artifacts/retargeting_v3_step2_capability/target_geometry_matrix.json \
  > /tmp/target_geometry_matrix.validated.json
```

Result: valid JSON.

## Matrix Counts

- `requested=17`
- `runtime_loaded=15`
- `runtime_unavailable_artifact_fallback=2`
- `source_unavailable=2`
- `runtime_failed_artifact_fallback=0`
- `runtime_failed_no_artifact=0`
- `fingerprint_matches_runtime=4`
- `map_change_recommended=0`

Fallback rows: `talos_urdf`, `jaxon_urdf`.

## Evidence For Map Changes

No evidence-backed map corrections were found.

Runtime-loadable maps resolved their configured bodies and nonzero distal
offsets. The generated matrix reports `map_change_recommended=0`. The observed
differences are recorded as variant geometry, source unavailability, or
fingerprint/source identity drift.

## Differential Notes

- Booster T1: URDF chest uses compiled `world`; MJCF chest uses `Trunk`.
  Hands/feet remain equivalent target bodies with near-identical ratios.
- Fourier N1: URDF hands target `*_hand_yaw_link` with about `0.13201 m`
  local distal offsets; MJCF hands target `*_end_effector_link` with about
  `0.03 m` offsets. Arm chain length differs substantially.
- TALOS: URDF live source unavailable; artifact fallback shows URDF gripper
  base/sole targets while MJCF uses fingertip/foot model-site targets.
- Atlas: DRC palms vs V4 hand links differ as variant targets with matching
  active coordinate counts.
- H1/G1 controls: symmetry checks stay small; no corrective evidence.
- RPO: `yaw_only_torso_control` with one revolute torso coordinate.
- OP3: fixed torso with zero active coordinates and zero desired torso
  distance.
- JAXON: source-unavailable artifact fallback.
- Valkyrie and MuJoCo Humanoid: runtime-loaded standalone probes with symmetric
  left/right geometry.

## Risks And Unfinished Items

- Re-audit `talos_urdf` and `jaxon_urdf` after their declared fixed sources are
  present locally.
- Fingerprint mismatches remain for many rows because current runtime source
  identity differs from the verified map artifact identity. This is recorded in
  the matrix and was not converted into a semantic-map edit.
- This handoff does not claim projection-gate fixes; it isolates target
  geometry evidence.
