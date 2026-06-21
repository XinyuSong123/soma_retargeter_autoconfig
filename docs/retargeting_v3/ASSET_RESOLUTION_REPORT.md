# Robot Zoo Asset Resolution Report

Status: **upstream source ambiguity resolved on the web**  
Branch: `retargeting-v3-step2-jacobian-assets`  
Lock file: `assets/robot_zoo/source_lock.json`

## 1. What was actually wrong

The `source_unavailable=16` result did not mean that the robot models were
closed-source or had disappeared. The validation run was executed without a
complete local `robot_descriptions` cache. All 16 model files were located in
public repositories at fixed commits, and their exact repository-relative paths
and Git blob SHAs are now recorded in `source_lock.json`.

Two additional failures were not source failures:

- `jvrc_urdf`: public source resolved, but COLLADA decoding required
  `pycollada`;
- `unitree_go2_urdf`: same dependency issue.

Three additional failures were not asset failures either:

- `berkeley_humanoid_urdf`;
- `berkeley_humanoid_mjcf_direct`;
- `unitree_h1_urdf`.

Their sources were available; they require fingerprint-bound verified semantic
maps or an explicit partial-morphology classification.

## 2. Pinned dependency set

The project now exposes a `robot-zoo` optional dependency group:

```bash
python -m pip install -e ".[robot-zoo,test]"
```

Pinned asset-side dependencies:

```text
robot_descriptions == 2.0.0
mujoco            == 3.8.1
pycollada          == 0.9.3
```

`pycollada 0.9.3` is BSD-licensed and is required to read the public DAE meshes
referenced by JVRC and Go2 URDFs. It must not be worked around by deleting mesh
or site evidence.

## 3. Resolved `source_unavailable` entries

| Model ID | Repository | Exact commit | Model path | Blob SHA | Policy |
|---|---|---|---|---|---|
| `atlas_drc_urdf` | `RobotLocomotion/drake` | `7abea0556ede980a5077fe1a8cfbae59b57c7c27` | `examples/atlas/urdf/atlas_convex_hull.urdf` | `2eb99cdc270300b3eef67c59ad6b742bfbfddf77` | required snapshot |
| `atlas_v4_urdf` | `openai/roboschool` | `c8ee2812207ef620af618b065eca3bde10f5823c` | `roboschool/models_robot/atlas_description/urdf/atlas_v4_with_multisense.urdf` | `7dd163ffddd0365288c05c6c9b3b44e84fdf9df4` | required snapshot |
| `bolt_urdf` | `Gepetto/example-robot-data` | `d0d9098d752014aec3725b07766962acf06c5418` | `robots/bolt_description/robots/bolt.urdf` | `88b4238af073dfb356e89ba5f103009737591f60` | required negative control |
| `cassie_urdf` | `robot-descriptions/cassie_description` | `81a2d8bbd77201cc974afb127adda4e2857a6dbf` | `urdf/cassie_v4.urdf` | `e855b8c29038a325c203fa488edbc47a1d551d7b` | required negative control |
| `draco3_urdf` | `shbang91/draco3_description` | `5afd19733d7b3e9f1135ba93e0aad90ed1a24cc7` | `urdf/draco3.urdf` | `3f1e396d1d29f51cb374956190aefdebb312b262` | required snapshot |
| `fourier_gr1_urdf` | `FFTAI/Wiki-GRx-Models` | `351245ac8fa4bf6f4b0c41556e1e6976a438bcef` | `GRX/GR1/GR1T1/urdf/GR1T1_nohand.urdf` | `3377229d1762ce4e7d7d4531f0ee88a8461bf73a` | GPL fetch-only |
| `jaxon_urdf` | `robot-descriptions/jaxon_description` | `4a0cb7a4a737864312f8d6e3f89823a741539bfc` | `urdf/jaxon_jvrc.urdf` | `05b0b41ad3a334e187cf3394276b871cfae124b7` | CC-BY-SA fetch-only |
| `mujoco_humanoid_mjcf` | `google-deepmind/mujoco` | `ad0dc0de5e10a075a2c65be629e9a8d557d383a6` | `model/humanoid/humanoid.xml` | `a2c4a16fb59c725253859a9093493524db888675` | required snapshot |
| `rhea_urdf` | `G-Levine/rhea_description` | `1dc0f1abcf51b5d8a8f7ff8a548399ff0df1414f` | `urdf/rhea.urdf` | `1abd17470bf46ffa4eabfde26a19405b73de877f` | required negative control |
| `robonaut2_urdf` | `gkjohnson/nasa-urdf-robots` | `54cdeb1dbfb529b79ae3185a53e24fce26e1b74b` | `r2_description/robots/r2c_sim_full_control.urdf.urdf` | `990420d8d049c587e5f34eacb31f5f7cea49cb00` | NASA fetch-only |
| `romeo_urdf` | `ros-aldebaran/romeo_robot` | `1f5b28807bed4d7b13a6781801f1d0c8bc8b09fb` | `romeo_description/urdf/romeo_generated_urdf/romeo.urdf` | `9164cc7a5893bdfb59852053864146d042dd9b25` | required snapshot |
| `sigmaban_urdf` | `Rhoban/sigmaban_urdf` | `d5d023fd35800d00d7647000bce8602617a4960d` | `robot.urdf` | `2992e9802b7166cda64d5a630d994709ec4c421d` | required snapshot |
| `simple_humanoid_urdf` | `laas/simple_humanoid_description` | `4e859aed7df3c29954c9cca2a1ecb94069f7cfce` | `urdf/simple_humanoid.urdf` | `57ccdead8da83300d1fd13537bfab6b5c8a538ec` | required snapshot |
| `talos_urdf` | `stack-of-tasks/talos-data` | `77169405d6a48a5d3f3f75eb014209f375ff23b6` | `urdf/talos_full.urdf` | `1e359a0352ed28b2fda87ff6d5a62b115a51eeaf` | LGPL fetch-only |
| `upkie_urdf` | `upkie/upkie_description` | `19a91ce69cab6742c613cab104986e3f8a18d6a5` | `urdf/upkie.urdf` | `42324e8259dfac44f2de27d8f7ae878675b00a4c` | required negative control |
| `valkyrie_urdf` | `gkjohnson/nasa-urdf-robots` | `54cdeb1dbfb529b79ae3185a53e24fce26e1b74b` | `val_description/model/robots/valkyrie_sim.urdf` | `1e6a6c0af94073a64e94f3b68e50db22ec05d996` | NASA fetch-only |

All paths above were opened successfully at their pinned commits before the lock
was written. Tags used by `robot_descriptions` were resolved to immutable commit
SHAs (`roboschool 1.0.49` and `romeo_robot 0.1.5`).

## 4. Required acquisition behavior

The next implementation must read `source_lock.json`, not rediscover these
repositories from floating branches.

For each locked model:

1. require `robot_descriptions==2.0.0`;
2. reject a user-supplied `ROBOT_DESCRIPTION_COMMIT` override;
3. fetch into `ROBOT_DESCRIPTIONS_CACHE` or `ROBOT_ZOO_CACHE`;
4. check out the exact locked commit in detached mode;
5. verify the repository origin;
6. verify the repository-relative model path;
7. verify `git hash-object <model>` equals the locked blob SHA;
8. preserve every referenced mesh/include/package directory;
9. write a machine-readable acquisition report;
10. never commit GPL, CC-BY-SA, LGPL, or NASA fetch-only trees into this Apache repository.

The source lock resolves *where and what to fetch*. Codex still has to implement
and execute the deterministic fetch/verify orchestration in the next round,
because a web session cannot populate the user's local cache.

## 5. Acceptance after local synchronization

After synchronization, the old `source_unavailable=16` category should become:

- `available` and then an algorithm/semantic result; or
- `license_blocked` only where policy explicitly prohibits local use; or
- a concrete network/integrity failure that includes the locked repository,
  commit, model path, expected blob SHA, and reproduction command.

A required permissive model may not remain `source_unavailable` merely because
validation was run in no-fetch mode.

## 6. What this does not fix

Asset synchronization will not fix the 14 numerical `algorithm_failed` reports.
Those require the Jacobian/reachability changes specified in the new `/goal.md`:
engine Jacobians as primary truth, backend-aware finite differences,
near-zero-column classification, task-specific rank/subspace stability, and
chain-length residual normalization.
