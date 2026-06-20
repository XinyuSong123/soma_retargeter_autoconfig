# Step 2 Agent L Handoff - Verified Semantic Maps Batch 2

## Scope

Agent: L-xhigh  
Reasoning mode: xhigh requested for this batch.

Owned files added:

- `assets/robot_zoo/semantic_maps/robotis_op3_mjcf.json`
- `assets/robot_zoo/semantic_maps/booster_t1_urdf.json`
- `assets/robot_zoo/semantic_maps/booster_t1_mjcf.json`
- `assets/robot_zoo/semantic_maps/toddlerbot_urdf.json`
- `assets/robot_zoo/semantic_maps/toddlerbot_2xc_mjcf.json`
- `assets/robot_zoo/semantic_maps/toddlerbot_2xm_mjcf.json`
- `tests/v3/test_semantic_maps_compact_batch2.py`
- `docs/retargeting_v3/subagents/step2_agent_l_handoff.md`

No optional `semantic_expectations` files were added in this batch.

## Inputs Read

- `goal.md`
- `docs/retargeting_v3/STEP1_MATHEMATICAL_SPEC.md`
- `docs/retargeting_v3/STEP1_COMPLETE_MODEL_CROSSCHECK.md`
- `docs/retargeting_v3/THREE_STEP_ROADMAP.md`
- `docs/retargeting_v3/ASSET_POLICY.md`
- `docs/retargeting_v3/subagents/step2_agent_h_handoff.md`
- existing RPO semantic map and semantic validation/site tests

## Local Source Evidence

All assigned sources resolved from the local `robot_descriptions` cache and were loaded with `MuJoCoRuntimeModelAdapter`.

| Model id | Local source | Fingerprint | Verified torso semantics |
|---|---|---|---|
| `robotis_op3_mjcf` | `${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/robotis_op3/op3.xml` | `a65607203f5f2d28d1d47e7ed5ef3f63246469dbc6258ff7a81c395f4ebf3603` | `Hips=body_link`, `Chest=body_link`, active torso coords `[]` |
| `booster_t1_urdf` | `${ROBOT_DESCRIPTIONS_CACHE}/booster_gym/resources/T1/T1_serial.urdf` | `1e7af6d1390160bc205e05ba06d41c1a1d9bcbc9452deb1d575485e3a3d3684e` | `Hips=Waist`, `Chest=world`, active torso coord `Waist`; MuJoCo collapses the root trunk link into compiled `world` |
| `booster_t1_mjcf` | `${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/booster_t1/t1.xml` | `5d767b53b12e780810ae9d4b13aecdd5c0818e44f10992a28139f38ca472e289` | `Hips=Waist`, `Chest=Trunk`, active torso coord `Waist` |
| `toddlerbot_urdf` | `${ROBOT_DESCRIPTIONS_CACHE}/toddlerbot/toddlerbot/descriptions/toddlerbot_2xc/toddlerbot_2xc.urdf` | `98facdad650ae0f86753c972c7e21aae4fdf515c355fe4b1926cae973fb2982e` | `Hips=waist_roll_body`, `Chest=torso_freejoint_body`, active torso coords `waist_yaw, waist_roll` |
| `toddlerbot_2xc_mjcf` | `${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/toddlerbot_2xc/toddlerbot_2xc.xml` | `144673ad6ceb4cbf5f9832e2ee3b8980d5c0c7e3c5601efc2244fce17bf144df` | `Hips=pelvis_link`, `Chest=torso`, active torso coords `waist_yaw, waist_roll` |
| `toddlerbot_2xm_mjcf` | `${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/toddlerbot_2xm/toddlerbot_2xm.xml` | `64072667c74d54e8bad8a8e65c7ca7821dddbead4682c54494638bceffac9da4` | `Hips=pelvis_link`, `Chest=torso`, active torso coords `waist_yaw, waist_roll` |

## Distal Site Evidence

- OP3 hands use compiled mesh bounds on `l_el_link` and `r_el_link`, with distal local `+Y` and `-Y` respectively. Feet use `l_ank_roll_link` and `r_ank_roll_link` sole `min Z`, with toe/heel at `max X`/`min X`.
- Booster URDF and MJCF hands use compiled hand-link bounds, with distal local `+Y` for left and `-Y` for right. Feet use foot-link sole `min Z`, with toe/heel at `max X`/`min X`.
- Toddlerbot URDF hands use compiled wrist-roll body bounds at hand-surface `min Z`; feet use ankle-roll body sole `min Z`, toe `max X`, heel `min X`.
- Toddlerbot MJCF hand/foot core sites use explicit compiled model sites: `left_hand_center`, `right_hand_center`, `left_foot_center`, `right_foot_center`. Toe/heel auxiliary anchors still come from compiled foot mesh bounds.

All distal hand, foot, toe, and heel local positions are nonzero. Hips/Chest zero local positions are not distal endpoints and are backed by topology evidence.

## Tests

Command:

```bash
python -m pytest -q tests/v3/test_semantic_maps_compact_batch2.py
```

Result: **5 passed**.

Command:

```bash
python -m pytest -q tests/v3/test_semantic_sites_verified.py
```

Result: **9 passed**.

Command:

```bash
python -m pytest -q tests/v3/test_semantic_map_coverage_manifest.py
```

Result: **1 failed, 3 passed**.

Expected out-of-scope failure: `test_available_positive_humanoids_have_verified_semantic_maps` still reports 11 locally available positive humanoids without verified maps:

- `unitree_h1_urdf`
- `unitree_h1_2_urdf`
- `unitree_h1_2_mjcf`
- `berkeley_humanoid_urdf`
- `jvrc_urdf`
- `jvrc_mjcf`
- `elf2_urdf`
- `elf2_mjcf`
- `ergocub_urdf`
- `pal_talos_mjcf_direct`
- `berkeley_humanoid_mjcf_direct`

The six Agent L ids are reported as `source_status=available` and `semantic_map_status=verified` by the same coverage helper.

## Risks and Notes

1. Booster URDF has no separate compiled `Trunk` body because MuJoCo collapses the fixed URDF root into `world`. The map keeps the waist/chest split as `Waist -> world` with active torso coord `Waist`; downstream cross-format checks should compare semantic topology/capability, not body-name equality.
2. Map fingerprints are tied to the current local cached source files and MuJoCo loader fingerprint payload. If upstream cache contents, loader version, or loader patching changes, regenerate these maps rather than reusing stale fingerprints.
3. Full Step 2 semantic coverage remains blocked by missing maps outside this batch; no placeholder maps were added for those ids.

