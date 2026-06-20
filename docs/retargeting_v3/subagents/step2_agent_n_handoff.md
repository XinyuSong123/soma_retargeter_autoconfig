# Step 2 Agent N Handoff - Verified Semantic Maps Batch 4

## Scope

Agent: N-xhigh

Reasoning mode: xhigh requested by the user for this batch.

Owned files changed:

- `assets/robot_zoo/semantic_maps/jvrc_urdf.json`
- `assets/robot_zoo/semantic_maps/jvrc_mjcf.json`
- `assets/robot_zoo/semantic_maps/elf2_urdf.json`
- `assets/robot_zoo/semantic_maps/elf2_mjcf.json`
- `assets/robot_zoo/semantic_maps/ergocub_urdf.json`
- `assets/robot_zoo/semantic_maps/pal_talos_mjcf_direct.json`
- `assets/robot_zoo/semantic_expectations/jvrc_urdf.json`
- `assets/robot_zoo/semantic_expectations/jvrc_mjcf.json`
- `assets/robot_zoo/semantic_expectations/elf2_urdf.json`
- `assets/robot_zoo/semantic_expectations/elf2_mjcf.json`
- `assets/robot_zoo/semantic_expectations/ergocub_urdf.json`
- `assets/robot_zoo/semantic_expectations/pal_talos_mjcf_direct.json`
- `tests/v3/test_semantic_maps_humanoid_batch4_verified.py`
- `docs/retargeting_v3/subagents/step2_agent_n_handoff.md`

Commit: not committed in this pass. Current HEAD while working was `f7cf036245a3e1fe82f44c9b73990f6a75da0ece`.

I did not modify pre-existing out-of-scope changes in `semantic_validation.py`, Agent H/J docs, or other agents' untracked semantic maps/tests.

## Inputs Read

- `goal.md`
- `docs/retargeting_v3/STEP1_MATHEMATICAL_SPEC.md`
- `docs/retargeting_v3/STEP1_COMPLETE_MODEL_CROSSCHECK.md`
- `docs/retargeting_v3/THREE_STEP_ROADMAP.md`
- `docs/retargeting_v3/subagents/step2_agent_h_handoff.md`
- `assets/robot_zoo/robot_zoo_manifest.json`
- existing `semantic_sites.py`, `semantic_validation.py`, `site_geometry.py`, and semantic-site tests

## Local Sources Inspected

- `${ROBOT_DESCRIPTIONS_CACHE}/jvrc_description/urdf/jvrc1.urdf`
- `${ROBOT_DESCRIPTIONS_CACHE}/jvrc_mj_description/xml/jvrc1.xml`
- `${ROBOT_DESCRIPTIONS_CACHE}/bxi_robot_models/elf2_dof25/urdf/elf2_dof25.urdf`
- `${ROBOT_DESCRIPTIONS_CACHE}/bxi_robot_models/elf2_dof25/xml/scene.xml`
- `${ROBOT_DESCRIPTIONS_CACHE}/ergocub-software/urdf/ergoCub/robots/ergoCubSN002/model.urdf`
- `${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/pal_talos/talos.xml`

## Map Decisions

JVRC:

- Hips: `PELVIS_S`.
- Chest: `WAIST_R_S`, preserving the three waist joints.
- Hands: `L_WRIST_Y_S` / `R_WRIST_Y_S` with nonzero distal local positions from wrist mesh bounds.
- Feet: `L_ANKLE_P_S` / `R_ANKLE_P_S` with sole, toe, and heel offsets from foot mesh bounds.
- JVRC URDF could not be runtime-compiled in this environment because DAE loading needs `pycollada`; its map records that blocker and uses URDF topology plus local JVRC MJCF compiled mesh cross-checks for the same link names.

ELF2:

- Hips: `base_link`.
- Chest: `waist_y_link`.
- Hands: terminal reduced-arm links `l_elb_z_link` / `r_elb_z_link`; no wrist or palm body is fabricated.
- Feet: `l_ankle_x_link` / `r_ankle_x_link` with sole/toe/heel offsets from compiled mesh bounds.

ergoCub:

- Hips: `root_link`.
- Chest: `chest`.
- Hands: `l_hand_palm` / `r_hand_palm` with distal palm mesh offsets.
- Feet: `l_ankle_2` / `r_ankle_2` with sole/toe/heel offsets from compiled foot geometry, cross-checked against fixed URDF front/rear/sole frames.

PAL TALOS direct:

- Hips: `base_link`.
- Chest: `torso_2_link`.
- Hands: compiled model palm sites copied into distal fingertip bodies.
- Feet: `leg_left_6_link` / `leg_right_6_link` with compiled `left_foot` / `right_foot` site offsets.
- The TALOS map explicitly does not use proximal `leg_left_1_link` / `leg_right_1_link` for feet.

## Tests

Passed:

```bash
python -m pytest -q tests/v3/test_semantic_maps_humanoid_batch4_verified.py
```

Result: `6 passed in 1.31s`.

Passed:

```bash
python -m pytest -q tests/v3/test_semantic_sites_verified.py tests/v3/test_semantic_maps_humanoid_batch4_verified.py
```

Result: `15 passed in 1.76s`.

Coverage gate run:

```bash
python -m pytest -q tests/v3/test_semantic_map_coverage_manifest.py
```

Result: `1 failed, 3 passed`.

The remaining failure is outside this batch's ownership:

- `unitree_h1_urdf`
- `unitree_h1_2_urdf`
- `unitree_h1_2_mjcf`
- `berkeley_humanoid_urdf`
- `berkeley_humanoid_mjcf_direct`

The same report also keeps 12 source-unavailable positive humanoids separate from the local available-source gate.

## Risks And Blockers

1. `jvrc_urdf` is verified from source XML topology and same-family MJCF geometry cross-checks, not from a compiled JVRC URDF runtime model. The concrete blocker is missing `pycollada` for DAE mesh loading.
2. Full semantic-map coverage is still blocked by five locally available positive humanoids outside Agent N's scope.
3. This pass only verifies semantic maps and focused semantic tests. It does not claim Step 2 completion, full Robot Zoo pass, rest calibration pass, canonical projection pass, or Agent F red-team pass.
4. The worktree contains unrelated pre-existing modified/untracked files from other agents; they were left untouched.

## Integration Notes

- These six maps should be integrated after Agent H's coverage helper, because that helper now recognizes them as verified payloads.
- TALOS downstream tests should assert foot bodies remain `leg_left_6_link` and `leg_right_6_link`; mapping TALOS feet to `leg_left_1_link` or `leg_right_1_link` would reintroduce the known proximal-foot false positive.
- ELF2 downstream capability code should treat hand tracking as reduced distal-arm position support, not wrist-rich hand orientation support.
