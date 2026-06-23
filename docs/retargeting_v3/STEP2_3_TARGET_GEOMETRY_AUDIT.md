# Step 2.3 Target Geometry Audit

Status: complete for Agent D scope.
Backend: `newton`.
Matrix artifact: `artifacts/retargeting_v3_step2_capability/target_geometry_matrix.json`.

## Scope

This audit records runtime target-geometry evidence for the requested Step 2
differential groups:

- Booster T1 URDF vs MJCF.
- Fourier N1 URDF vs MJCF.
- TALOS URDF vs MJCF.
- Atlas DRC vs V4.
- Unitree H1/G1 passing controls.
- Roboparty RPO yaw-only control.
- ROBOTIS OP3 fixed-torso/rank-zero control.
- JAXON, Valkyrie, and MuJoCo Humanoid standalone probes.

For each runtime-loadable model the matrix records semantic site body/local
offset, neutral world transform, target frame convention, left/right symmetry,
chain topology, active coordinates, limits, source-to-robot segment ratio, and
desired-distance-to-chain-length ratio.

## Matrix Counts

From `target_geometry_matrix.json`:

- `requested=17`
- `runtime_loaded=15`
- `runtime_unavailable_artifact_fallback=2`
- `source_unavailable=2`
- `runtime_failed_artifact_fallback=0`
- `runtime_failed_no_artifact=0`
- `fingerprint_matches_runtime=4`
- `map_change_recommended=0`

The two artifact fallbacks are `talos_urdf` and `jaxon_urdf`. Their declared
fixed `robot_descriptions` cache files are not currently available in this
workspace, so the matrix uses prior Step-2 per-robot artifacts for topology
context and does not recommend map edits.

## Differential Findings

Booster T1:

- Both variants runtime-load.
- URDF maps `Chest` to compiled `world`; MJCF maps it to `Trunk`.
- Hand and foot target bodies remain the same across variants.
- Torso is a single revolute path in both variants.
- Arm desired/chain ratios are nearly identical: about `0.8630`.
- No evidence-backed semantic-map correction was found.

Fourier N1:

- Both variants runtime-load.
- URDF maps `Hips` to `world`, `Chest` to `waist_yaw_link` with a body-local
  `z=0.24115` offset, and hands to `*_hand_yaw_link` with about `0.13201 m`
  distal offset.
- MJCF maps `Hips` to `base_link`, `Chest` to `torso_link`, and hands to
  `*_end_effector_link` with about `0.03 m` distal offset.
- Arm active coordinate counts match (`5` each), but left/right arm chain
  lengths differ substantially across formats: URDF about `1.1191`, MJCF about
  `0.62065`.
- The matrix records this as real variant target-geometry drift, not a map
  typo.

TALOS:

- `pal_talos_mjcf_direct` runtime-loads.
- `talos_urdf` is source-unavailable and uses artifact fallback.
- Artifact comparison shows URDF hands on gripper base links and soles on
  `*_sole_link`; MJCF uses fingertip/foot model-site targets.
- Because live URDF runtime evidence is unavailable, no semantic-map correction
  was made.

Atlas:

- DRC and V4 runtime-load.
- Both share pelvis/utorso and leg topology.
- Arm target bodies differ by variant (`left_palm`/`right_palm` vs
  `l_hand`/`r_hand`), with equal active coordinate counts.
- The arm distance deltas are variant geometry, not a proven map error.

H1/G1 controls:

- All four requested Unitree control rows runtime-load.
- H1 URDF/MJCF have no material differential in the matrix.
- G1 URDF/MJCF differ in root convention (`world` vs `pelvis`) and torso path
  body path, while distal symmetry remains within test thresholds.
- No map correction was indicated.

Standalone controls:

- RPO has a single-revolute torso path and the explicit
  `yaw_only_torso_control` flag.
- OP3 has a fixed torso: `body_link -> body_link`, `active_coordinate_count=0`,
  and `desired_distance=0.0`.
- Valkyrie and MuJoCo Humanoid runtime-load with symmetric left/right geometry.
- JAXON is source-unavailable and uses artifact fallback only.

## Semantic Map Changes

No semantic maps or semantic expectations were changed.

Reason: all runtime-loadable rows resolve their verified semantic maps without
body-resolution or nonzero-distal-offset failures, and the generated matrix has
`map_change_recommended=0`. The observed differences are variant conventions,
source availability limits, or fingerprint/source identity drift, not proven
semantic target errors.

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

Result: completed and wrote the matrix artifact.

```bash
python -m json.tool \
  artifacts/retargeting_v3_step2_capability/target_geometry_matrix.json \
  > /tmp/target_geometry_matrix.validated.json
```

Result: valid JSON.

## Risks And Unfinished Items

- `talos_urdf` and `jaxon_urdf` need live fixed-source availability before
  their target geometry can be fully re-audited.
- Only 4 of 17 rows match the runtime fingerprint generated in this workspace.
  The mismatches are recorded as source/fingerprint drift and were not treated
  as semantic errors.
- This audit explains target geometry inputs. It does not rerun or relax the
  projection acceptance gates.
