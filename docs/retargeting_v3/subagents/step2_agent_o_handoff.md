# Step 2 Agent O Handoff - Unitree H1 Remaining Semantic Coverage

## Scope

Agent: O-xhigh

Owned files touched:

- `assets/robot_zoo/semantic_maps/unitree_h1_2_urdf.json`
- `assets/robot_zoo/semantic_maps/unitree_h1_2_mjcf.json`
- `assets/robot_zoo/semantic_expectations/unitree_h1_urdf.json`
- `assets/robot_zoo/semantic_expectations/unitree_h1_2_urdf.json`
- `assets/robot_zoo/semantic_expectations/unitree_h1_2_mjcf.json`
- `tests/v3/test_semantic_maps_unitree_h1_remaining.py`
- `docs/retargeting_v3/subagents/step2_agent_o_handoff.md`

No `assets/robot_zoo/semantic_maps/unitree_h1_urdf.json` file was created. The locally resolved URDF cannot currently provide compiled geometry truth for its DAE meshes through MuJoCo.

## Resolved Sources

All sources were resolved with `allow_fetch=False`.

| Model | Resolved local source | Source SHA256 | Compile status | Runtime fingerprint |
|---|---|---:|---|---:|
| `unitree_h1_urdf` | `${ROBOT_DESCRIPTIONS_CACHE}/unitree_ros/robots/h1_description/urdf/h1.urdf` | `1b692e435b78fc93030bb893be39c0cbcacd1571f37d0e7eb30412bade8cdf97` | blocked | none |
| `unitree_h1_2_urdf` | `${ROBOT_DESCRIPTIONS_CACHE}/unitree_ros/robots/h1_2_description/h1_2.urdf` | `dcf7f22984a77adddb58a9f6a3baa2692c665c3700431f7bd96c6b887f9b164e` | compiled | `7e05906dad7beecdbd967bd0f9f74bfa2575f16aa3aa8f8b867537723ce6847f` |
| `unitree_h1_2_mjcf` | `${ROBOT_DESCRIPTIONS_CACHE}/unitree_ros/robots/h1_2_description/h1_2.xml` | `3ef5bb241e4d78ca06391a14125c58ccf91613538badb5153f9338593c5ff5f1` | compiled | `fccd3e346b037a4938b73f1fadd72d9f70000ca67926fe11b549fa77c80af130` |

## Verified H1-2 Evidence

- `unitree_h1_2_urdf`: source URDF root link is `pelvis`, compiled into MuJoCo `world`. `Hips -> Chest` path is `world -> torso_link` with active `torso_joint`.
- `unitree_h1_2_mjcf`: Hips uses floating root body `pelvis`. `Hips -> Chest` path is `pelvis -> torso_link` with active `torso_joint`.
- Both H1-2 variants use full compiled arm chains through shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw, and full leg chains through hip yaw/pitch/roll, knee, ankle pitch/roll.
- Distal hands use positive-X compiled bounds on `left_wrist_yaw_link` and `right_wrist_yaw_link`.
- Feet, toes, and heels use compiled bounds on `left_ankle_roll_link` and `right_ankle_roll_link`.

## H1 URDF Blocker

`unitree_h1_urdf` remains blocked:

- The local source resolves and the DAE mesh files exist.
- `MuJoCoRuntimeModelAdapter` raises the original package mesh-opening error for `package://h1_description/meshes/left_hip_roll_link.dae`.
- Manually applying the adapter's URDF mesh absolutization creates a patched URDF with local absolute mesh paths, but MuJoCo then reports `no decoder found for mesh file` on a local `.dae` mesh.
- Because compiled body and geometry truth is unavailable, no verified semantic map was emitted for this ID.

## Verification Commands

Required focused command:

```bash
PYTHONPATH=. pytest -q tests/v3/test_semantic_maps_unitree_verified.py tests/v3/test_semantic_maps_unitree_h1_remaining.py tests/v3/test_semantic_sites_verified.py
```

Result: `26 passed in 4.33s`.

Coverage gate command:

```bash
PYTHONPATH=. pytest -q tests/v3/test_semantic_map_coverage_manifest.py
```

Result: `1 failed, 4 passed`.

The remaining available-source gate failure is `unitree_h1_urdf`. H1-2 URDF and H1-2 MJCF are now verified maps in the report. There are no remaining non-Unitree available-source blockers in `available_missing_verified_map_ids`; the report still lists these non-Unitree source-unavailable IDs separately: `draco3_urdf`, `sigmaban_urdf`, `simple_humanoid_urdf`, `atlas_drc_urdf`, `atlas_v4_urdf`, `romeo_urdf`, `mujoco_humanoid_mjcf`, `fourier_gr1_urdf`, `jaxon_urdf`, `talos_urdf`, `valkyrie_urdf`, `robonaut2_urdf`.

The coverage helper currently recognizes verified maps and evidence-backed structural partial blockers. It does not classify runtime-loader blockers like the H1 URDF DAE decoder failure as a pass condition, and this handoff intentionally does not mark that model as structurally incomplete.
