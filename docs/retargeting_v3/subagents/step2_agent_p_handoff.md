# Step 2 Agent P Handoff - Berkeley Partial Coverage Gate

## Scope

Agent: P-xhigh

Allowed files touched:

- `assets/robot_zoo/semantic_expectations/berkeley_humanoid_urdf.json`
- `assets/robot_zoo/semantic_expectations/berkeley_humanoid_mjcf_direct.json`
- `soma_retargeter/robotics/v3/semantic_validation.py`
- `tests/v3/test_semantic_map_coverage_manifest.py`
- `tests/v3/test_semantic_maps_berkeley_partial.py`
- `docs/retargeting_v3/subagents/step2_agent_p_handoff.md`

No Berkeley semantic map was created.

## Independent Evidence

Local Berkeley sources resolved from the Robot Zoo manifest and were compiled with MuJoCo for body/site inspection.

| Model ID | Local Source SHA256 | Compiled Fingerprint | Compiled Bodies | Compiled Sites | Result |
|---|---:|---:|---|---|---|
| `berkeley_humanoid_urdf` | `d92c2fbaa9829f3c122cc2defb59e69bd5b0958209403020f55580031eb27931` | `131555b02c811abee04a30a89e0904202f72f2ec2ce1364bdb0fe144f404a008` | `world`, `ll_*`, `lr_*` | none | structurally incomplete |
| `berkeley_humanoid_mjcf_direct` | `fdf0f53d0abd602662f58a52152b113c1f03d79e5e9c099a75ffabf819805047` | `72d06aa6fe776821421b6c96b235526011efd93b162f77a7df9bf04637aedd2e` | `world`, `torso`, `ll_*`, `lr_*` | `imu`, `LL_FOOT`, `LR_FOOT` | structurally incomplete |

Both compiled models have no shoulder, arm, elbow, wrist, hand, palm, or end-effector body/site names. They cannot support verified full humanoid maps for `LeftHand` and `RightHand`; the URDF also lacks separate compiled pelvis/chest bodies.

## Changes

- Berkeley expectations now declare `verification_status: partial_blocked` and `coverage_status: structurally_incomplete`, with structured blocker evidence, observed compiled bodies/sites, source SHA256, and compiled fingerprints.
- `compute_verified_semantic_map_coverage()` now reports `coverage_status` separately from legacy `semantic_map_status`.
- Coverage statuses now distinguish:
  - `verified_map`
  - `structurally_incomplete` / `partial_blocked`
  - `blocked`
  - `missing`
  - `invalid`
- `available_missing_verified_map_ids` now means local positive humanoids with no verified map and no explicit evidence-backed blocker.
- Added Berkeley-specific tests proving the cached models lack upper-limb/hand bodies and that Berkeley is excluded from the missing-map gate.
- Added a temp-manifest coverage test proving a locally available positive humanoid with no map and no blocker still enters the missing gate.

## Verification

Commands run:

```bash
python -m json.tool assets/robot_zoo/semantic_expectations/berkeley_humanoid_urdf.json >/dev/null
python -m json.tool assets/robot_zoo/semantic_expectations/berkeley_humanoid_mjcf_direct.json >/dev/null
PYTHONPATH=. python -m py_compile soma_retargeter/robotics/v3/semantic_validation.py tests/v3/test_semantic_map_coverage_manifest.py tests/v3/test_semantic_maps_berkeley_partial.py tests/v3/test_semantic_sites_verified.py
PYTHONPATH=. pytest -q tests/v3/test_semantic_map_coverage_manifest.py tests/v3/test_semantic_maps_berkeley_partial.py tests/v3/test_semantic_sites_verified.py
```

Result:

```text
18 passed in 1.34s
```

Coverage snapshot after this change:

```text
verified_map: 22
blocked: 1
structurally_incomplete: 2
missing: 12
available_missing_verified_map_ids: ()
available_blocked_ids: unitree_h1_urdf
available_structurally_incomplete_ids: berkeley_humanoid_urdf, berkeley_humanoid_mjcf_direct
```

Current remaining coverage gaps are source-unavailable missing maps:

```text
draco3_urdf
sigmaban_urdf
simple_humanoid_urdf
atlas_drc_urdf
atlas_v4_urdf
romeo_urdf
mujoco_humanoid_mjcf
fourier_gr1_urdf
jaxon_urdf
talos_urdf
valkyrie_urdf
robonaut2_urdf
```
