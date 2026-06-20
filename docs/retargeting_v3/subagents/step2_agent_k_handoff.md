# Step 2 Agent K Handoff - Unitree G1/H1 Semantic Maps Batch 1

## Scope

Agent: K-xhigh  
Reasoning mode: xhigh requested and used for this audit. This file records verifiable inputs, outputs, commands, and blockers.

Owned files touched:

- `assets/robot_zoo/semantic_maps/unitree_g1_urdf.json`
- `assets/robot_zoo/semantic_maps/unitree_g1_mjcf.json`
- `assets/robot_zoo/semantic_maps/unitree_h1_mjcf.json`
- `assets/robot_zoo/semantic_expectations/unitree_h1_urdf.json`
- `tests/v3/test_semantic_maps_unitree_verified.py`
- `docs/retargeting_v3/subagents/step2_agent_k_handoff.md`

No `assets/robot_zoo/semantic_maps/unitree_h1_urdf.json` file was created because the local resolved URDF could not be compiled to inspect body/site/geometry truth.

## Inputs Read

- `goal.md`
- `docs/retargeting_v3/STEP1_MATHEMATICAL_SPEC.md`
- `docs/retargeting_v3/STEP1_COMPLETE_MODEL_CROSSCHECK.md`
- `docs/retargeting_v3/THREE_STEP_ROADMAP.md`
- `docs/retargeting_v3/subagents/step2_agent_h_handoff.md`
- `soma_retargeter/robotics/v3/robot_zoo.py`
- Existing semantic-site schema/tests and `roboparty_rpo_local` semantic map/expectation.

## Resolved Sources

All sources below were resolved through `robot_zoo.py` with `allow_fetch=False`.

| Model | Resolved local source | Source SHA256 | Compile status | Runtime fingerprint |
|---|---|---:|---|---:|
| `unitree_g1_urdf` | `${ROBOT_DESCRIPTIONS_CACHE}/unitree_ros/robots/g1_description/g1_29dof.urdf` | `e1dc89366bf96aa33ba4ef211884f6d81320ddd2ac6f8b1ad82986e64008be31` | compiled | `67485e6802e62960922d241535c2bf8bd7314755bd8856079327a337c2d446b1` |
| `unitree_g1_mjcf` | `${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/unitree_g1/g1.xml` | `04b0a6d0a4359bb954174457fd5cb680f187f7a1648609a014cf9e906ffd98ff` | compiled | `d11a0034b3d341fdb5b58c977678aab429fcb76e6eb29a9c04816fe9edae7821` |
| `unitree_h1_urdf` | `${ROBOT_DESCRIPTIONS_CACHE}/unitree_ros/robots/h1_description/urdf/h1.urdf` | `1b692e435b78fc93030bb893be39c0cbcacd1571f37d0e7eb30412bade8cdf97` | blocked | none |
| `unitree_h1_mjcf` | `${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/unitree_h1/h1.xml` | `bdc9924af79cec7a4fe70a864abec2f97b418d4e8a573683b8a3949f00382455` | compiled | `ddcbcd290c29fd6833d0a1393cbc8120776a4f41f8212dd6762d4767b28bc969` |

## Verified Maps Added

### `unitree_g1_urdf`

- `Hips`: compiled body `world`. Evidence: source URDF root link is `pelvis`; MuJoCo collapses that root link into compiled `world`, and compiled `world` carries pelvis mesh geometry.
- `Chest`: `torso_link`, reached by `waist_yaw_joint`, `waist_roll_joint`, `waist_pitch_joint`.
- `LeftHand` / `RightHand`: distal positive-X compiled bounds on `left_wrist_yaw_link` / `right_wrist_yaw_link`.
- `LeftFoot` / `RightFoot`: sole min-Z compiled bounds on `left_ankle_roll_link` / `right_ankle_roll_link`; `LeftToe`, `RightToe`, `LeftHeel`, `RightHeel` added as auxiliary anchors.

Distal offsets:

- G1 hands: left `[0.1733293506327025, -0.0074530312018906576, 0.010210499184342359]`, right `[0.17332810173168528, 0.00745303203059176, 0.010210486561053458]`
- G1 soles: left `[0.03826198904017965, 4.9556029198885576e-05, -0.035409143860934696]`, right `[0.03826198952738826, -4.955513612300497e-05, -0.03540914398770632]`

### `unitree_g1_mjcf`

- `Hips`: compiled floating root body `pelvis`.
- `Chest`: `torso_link`, reached by the same 3-DoF waist chain as the URDF.
- Hands and feet use the same compiled geometry offsets as the URDF variant.
- The compiled MJCF `left_foot` / `right_foot` model sites are at ankle-roll body origin, so they were not used as sole evidence.

### `unitree_h1_mjcf`

- `Hips`: compiled floating root body `pelvis`.
- `Chest`: `torso_link`, reached by single yaw joint `torso`.
- `LeftHand` / `RightHand`: distal positive-X compiled bounds on `left_elbow_link` / `right_elbow_link`. H1 has no wrist/hand body in this resolved MJCF, so the map records distal forearm geometry for position semantics only; orientation support remains a chain-rank decision.
- `LeftFoot` / `RightFoot`: sole min-Z compiled bounds on `left_ankle_link` / `right_ankle_link`; toe/heel auxiliary anchors added.

Distal offsets:

- H1 hands: left `[0.3159998939037015, 0.0004998894982974275, -0.007366467977919858]`, right `[0.3159999009957901, -0.0004998892342714875, -0.007366469196975804]`
- H1 soles: left `[0.05500112732531626, 0.0, -0.07247375535160748]`, right `[0.05500112651828221, 0.0, -0.07247375535160748]`

## Topology Evidence

Checked via compiled MuJoCo body paths and active velocity coordinates.

- G1 `Hips -> Chest`: `world/pelvis -> waist_yaw_link -> waist_roll_link -> torso_link`; active waist yaw/roll/pitch.
- G1 `Chest -> LeftHand`: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw.
- G1 `Hips -> LeftFoot`: hip pitch/roll/yaw, knee, ankle pitch/roll.
- H1 `Hips -> Chest`: pelvis -> torso_link; active `torso`.
- H1 `Chest -> LeftHand`: shoulder pitch/roll/yaw and elbow.
- H1 `Hips -> LeftFoot`: hip yaw/roll/pitch, knee, ankle.

## Blocker

`unitree_h1_urdf` is locally available but could not be verified:

- The adapter first reports MuJoCo cannot open `package://h1_description/meshes/left_hip_yaw_link.dae`.
- Applying the adapter's URDF mesh absolutization manually removes `package://` references, but MuJoCo then reports: `no decoder found for mesh file .../h1_description/meshes/right_knee_link.dae`.
- Without compiled geometry truth, distal hand/sole/toe/heel offsets cannot be verified.

Recorded this in `assets/robot_zoo/semantic_expectations/unitree_h1_urdf.json` instead of fabricating a semantic map.

## Commands and Results

```bash
python -m pytest -q tests/v3/test_semantic_maps_unitree_verified.py
```

Result: `10 passed in 2.26s`.

```bash
python -m pytest -q tests/v3/test_semantic_sites_verified.py tests/v3/test_semantic_maps_unitree_verified.py
```

Result: `19 passed in 2.82s`.

```bash
python -m pytest -q tests/v3/test_semantic_map_coverage_manifest.py
```

Result: `1 failed, 3 passed in 0.69s`.

Expected remaining failure: the full positive-humanoid semantic coverage gate still reports 11 locally available missing verified maps. Unitree status after this batch is:

- verified: `unitree_g1_urdf`, `unitree_g1_mjcf`, `unitree_h1_mjcf`
- blocked/missing by evidence: `unitree_h1_urdf`
- out of scope for Agent K: `unitree_h1_2_urdf`, `unitree_h1_2_mjcf`

## Risks and Follow-Up

1. `unitree_h1_urdf` needs a loader path that can compile its DAE meshes, or an alternate trusted compiled source, before a verified map can be committed.
2. G1 URDF uses compiled `world` as `Hips` because MuJoCo collapses the URDF root `pelvis` link into that body. This is explicitly evidenced in the map and tested through the full waist/limb topology.
3. The repository had unrelated modified/untracked files from other agents during this pass; I left them untouched.

No commit was created by Agent K in this workspace.
