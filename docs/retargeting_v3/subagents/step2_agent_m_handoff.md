# Step 2 Agent M Handoff - Verified Semantic Maps Batch 3

## Scope

Agent: M-xhigh
Reasoning mode: xhigh requested for this batch.

Owned files touched:

- `assets/robot_zoo/semantic_maps/adam_lite_mjcf.json`
- `assets/robot_zoo/semantic_maps/apptronik_apollo_mjcf.json`
- `assets/robot_zoo/semantic_maps/fourier_n1_urdf.json`
- `assets/robot_zoo/semantic_maps/fourier_n1_mjcf.json`
- `assets/robot_zoo/semantic_expectations/adam_lite_mjcf.json`
- `assets/robot_zoo/semantic_expectations/apptronik_apollo_mjcf.json`
- `assets/robot_zoo/semantic_expectations/berkeley_humanoid_urdf.json`
- `assets/robot_zoo/semantic_expectations/berkeley_humanoid_mjcf_direct.json`
- `assets/robot_zoo/semantic_expectations/fourier_n1_urdf.json`
- `assets/robot_zoo/semantic_expectations/fourier_n1_mjcf.json`
- `tests/v3/test_semantic_maps_humanoid_batch3_verified.py`
- `docs/retargeting_v3/subagents/step2_agent_m_handoff.md`

No Berkeley semantic map JSON was emitted. Both Berkeley cached models lack hand/arm bodies and distal hand geometry, so a full verified map would be fabricated.

## Inputs Read

- `goal.md`
- `docs/retargeting_v3/STEP1_MATHEMATICAL_SPEC.md`
- `docs/retargeting_v3/STEP1_COMPLETE_MODEL_CROSSCHECK.md`
- `docs/retargeting_v3/THREE_STEP_ROADMAP.md`
- `docs/retargeting_v3/ASSET_POLICY.md`
- `docs/retargeting_v3/subagents/step2_agent_h_handoff.md`
- `assets/robot_zoo/robot_zoo_manifest.json`
- existing semantic map/site validation helpers and tests

## Local Source Evidence

All inspections used local cached sources resolved through the Robot Zoo manifest and `MuJoCoRuntimeModelAdapter`.

| Model ID | Source | Source SHA256 | Compiled fingerprint | Outcome |
|---|---|---:|---:|---|
| `adam_lite_mjcf` | `${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/pndbotics_adam_lite/adam_lite.xml` | `a6e8319e51dd33c1d6fb13907673f8c3bceebaacb7b3c7cbc60416f15fa26468` | `c2c8c2fe9a5626ba8612c629d1cb7c0c26ef70ad412221ddd2ced5917fb28d37` | full verified map |
| `apptronik_apollo_mjcf` | `${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/apptronik_apollo/apptronik_apollo.xml` | `a4bab6e94f0a179afa02db8db21cee74515fa320d3e2b0aebd65e671e43bf549` | `4e146c2ac3c40f01c7044571b4b3eff8ea6bd64dbf62c1c575b995a27d55bc5b` | full verified map |
| `fourier_n1_urdf` | `${ROBOT_DESCRIPTIONS_CACHE}/Wiki-GRx-Models-FourierN1/N1/urdf/N1_raw.urdf` | `a1afda3ad63e27938fe9adc41dd3247aed17d6970b6a0b30da6d3dc564b88867` | `6a7794523aef0927b662f2204cccceae10935812ae9fbd8c64ad6d9ee63837ee` | full verified map using compiled merged bodies |
| `fourier_n1_mjcf` | `${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/fourier_n1/n1.xml` | `dfd1c66dfb245ec0a59e8cf18afeef080845156e577293dbada496978d0f997a` | `0ca53e1f71cb6a1d75cf0b2473489d7acba92fb3999373e9083bab5607a8073c` | full verified map |
| `berkeley_humanoid_urdf` | `${ROBOT_DESCRIPTIONS_CACHE}/berkeley_humanoid_description/urdf/robot.urdf` | `d92c2fbaa9829f3c122cc2defb59e69bd5b0958209403020f55580031eb27931` | `131555b02c811abee04a30a89e0904202f72f2ec2ce1364bdb0fe144f404a008` | partial/incomplete; no map |
| `berkeley_humanoid_mjcf_direct` | `${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/berkeley_humanoid/berkeley_humanoid.xml` | `fdf0f53d0abd602662f58a52152b113c1f03d79e5e9c099a75ffabf819805047` | `72d06aa6fe776821421b6c96b235526011efd93b162f77a7df9bf04637aedd2e` | partial humanoid; no map |

## Map Decisions

Full maps were added only where all required core semantics had compiled topology and distal geometry evidence:

- Adam Lite: `pelvis`, `torso`, `wristYawLeft/Right`, `toeLeft/Right`.
- Apollo: `base_link`, `torso_link`, `l/r_wrist_pitch_link`, `l/r_foot_link`.
- Fourier N1 MJCF: `base_link`, `torso_link`, `left/right_end_effector_link`, `left/right_foot_pitch_link`.
- Fourier N1 URDF: MuJoCo collapses fixed `base_link`, `torso_link`, and end-effector links; the map uses compiled `world`, `waist_yaw_link`, `left/right_hand_yaw_link`, and `left/right_foot_pitch_link` with evidence documenting the fixed-body merge.

Each full map includes:

- `verification_status: verified`;
- compiled model fingerprint;
- local source path and SHA256;
- evidence strings for name, topology, side, chain, and compiled geometry;
- nonzero distal local offsets for hands, soles, toes, and heels;
- toe/heel entries marked as `auxiliary_semantics`.

Berkeley was not mapped:

- URDF compiled body tree: `world` plus `ll_/lr_` leg bodies only; no arm/hand bodies and no separate pelvis/chest body remain after compilation.
- MJCF direct compiled body tree: `torso` plus `ll_/lr_` leg bodies; no shoulder, arm, elbow, wrist, hand, palm, or end-effector bodies.
- Expectations record partial status and blockers. Hand semantics remain unavailable.

## Tests

Commands run:

```bash
python -m pytest -q tests/v3/test_semantic_maps_humanoid_batch3_verified.py
python -m json.tool assets/robot_zoo/semantic_maps/adam_lite_mjcf.json >/dev/null
python -m json.tool assets/robot_zoo/semantic_maps/apptronik_apollo_mjcf.json >/dev/null
python -m json.tool assets/robot_zoo/semantic_maps/fourier_n1_urdf.json >/dev/null
python -m json.tool assets/robot_zoo/semantic_maps/fourier_n1_mjcf.json >/dev/null
python -m pytest -q tests/v3/test_semantic_sites_verified.py
python -m pytest -q tests/v3/test_semantic_map_coverage_manifest.py
```

Results:

- Batch 3 focused tests: **6 passed**.
- JSON syntax checks: **passed** for new full maps and Berkeley expectation files.
- Existing semantic-site verifier: **9 passed**.
- Manifest coverage test: **1 failed, 3 passed**.

The remaining manifest coverage failure is expected at this integration point. It now reports 11 locally available positive humanoids missing maps:

```text
unitree_h1_urdf
unitree_h1_2_urdf
unitree_h1_2_mjcf
berkeley_humanoid_urdf
jvrc_urdf
jvrc_mjcf
elf2_urdf
elf2_mjcf
ergocub_urdf
pal_talos_mjcf_direct
berkeley_humanoid_mjcf_direct
```

Berkeley remains on that list by design until the broader coverage gate can represent partial humanoid expectations or the manifest is changed; this batch must not fake hands to satisfy a full-positive gate.

## Risks And Follow-Up

1. Current `compute_verified_semantic_map_coverage()` treats every positive humanoid as requiring complete hand semantics. Berkeley needs a partial-aware coverage path or manifest/expectation integration.
2. Fourier N1 URDF and MJCF are not body-name identical because the URDF compiler collapses fixed links. The maps bind to actual compiled bodies and tests verify the active chains, but cross-format equivalence should compare semantics rather than raw body names.
3. Adam Lite and Apollo distal hands are verified from compiled geometry bounds, not explicit palm sites. That is acceptable for Step 2 position sites, but orientation support must still be decided from the full chain rank.
4. I did not modify preexisting dirty/untracked files from other agents, including `semantic_validation.py` and other semantic maps/tests outside Batch 3 ownership.
