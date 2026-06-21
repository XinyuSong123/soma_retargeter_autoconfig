# Robot Zoo Asset Resolution — 2026-06-21

## Outcome

The previous `source_unavailable=16` result was not evidence that the models
were unavailable upstream. The validator ran without source fetching, while the
manifest recorded only `robot_descriptions` module names. The local cache did
not contain those repositories.

This branch adds:

- `assets/robot_zoo/source_lock.json`: fixed repository/ref/model-path records;
- `assets/robot_zoo/manifest_overlay_assets.json`: official real G1 23DoF URDF
  and MJCF additions;
- `soma_retargeter.tools.fetch_robot_zoo_sources`: data-only fixed-ref fetch,
  Git blob verification, inventory generation, and resolved-manifest output;
- the `robot-zoo` optional dependency group, including `pycollada`.

Fetched repositories stay outside Git. No upstream Python or shell code is
executed by the fetcher.

## Recovered source set

The following 16 entries previously reported `source_unavailable` and now have
verified public source records:

| ID | Fixed upstream | Model path |
|---|---|---|
| `cassie_urdf` | `robot-descriptions/cassie_description@81a2d8b` | `urdf/cassie_v4.urdf` |
| `draco3_urdf` | `shbang91/draco3_description@5afd197` | `urdf/draco3.urdf` |
| `sigmaban_urdf` | `Rhoban/sigmaban_urdf@d5d023f` | `robot.urdf` |
| `simple_humanoid_urdf` | `laas/simple_humanoid_description@4e859ae` | `urdf/simple_humanoid.urdf` |
| `atlas_drc_urdf` | `RobotLocomotion/drake@7abea05` | `examples/atlas/urdf/atlas_convex_hull.urdf` |
| `atlas_v4_urdf` | `openai/roboschool@1.0.49` | `roboschool/models_robot/atlas_description/urdf/atlas_v4_with_multisense.urdf` |
| `romeo_urdf` | `ros-aldebaran/romeo_robot@0.1.5` | `romeo_description/urdf/romeo_generated_urdf/romeo.urdf` |
| `mujoco_humanoid_mjcf` | `google-deepmind/mujoco@ad0dc0d` | `model/humanoid/humanoid.xml` |
| `upkie_urdf` | `upkie/upkie_description@19a91ce` | `urdf/upkie.urdf` |
| `bolt_urdf` | `Gepetto/example-robot-data@d0d9098` | `robots/bolt_description/robots/bolt.urdf` |
| `rhea_urdf` | `G-Levine/rhea_description@1dc0f1a` | `urdf/rhea.urdf` |
| `fourier_gr1_urdf` | `FFTAI/Wiki-GRx-Models@351245a` | `GRX/GR1/GR1T1/urdf/GR1T1_nohand.urdf` |
| `jaxon_urdf` | `robot-descriptions/jaxon_description@4a0cb7a` | `urdf/jaxon_jvrc.urdf` |
| `talos_urdf` | `stack-of-tasks/talos-data@7716940` | `urdf/talos_full.urdf` |
| `valkyrie_urdf` | `gkjohnson/nasa-urdf-robots@54cdeb1` | `val_description/model/robots/valkyrie_sim.urdf` |
| `robonaut2_urdf` | `gkjohnson/nasa-urdf-robots@54cdeb1` | `r2_description/robots/r2c_sim_full_control.urdf.urdf` |

Each model path was checked through the GitHub contents API. Its Git blob SHA is
stored in the source lock and is verified again after local checkout.

## Real Unitree G1 23DoF

Two official fixed sources are added through the manifest overlay:

- URDF: `unitreerobotics/unitree_ros@d6f13aad...`,
  `robots/g1_description/g1_23dof.urdf`;
- MJCF: `unitreerobotics/unitree_mujoco@ae6a8403...`,
  `unitree_robots/g1/g1_23dof.xml`.

Both upstream repositories use BSD-3-Clause. These are real public 23DoF
models, not a 29DoF model with coordinates locked by the benchmark.

## Loader dependency failures

`jvrc_urdf` and `unitree_go2_urdf` resolved correctly but failed because their
DAE geometry requires `pycollada`. The branch pins `pycollada==0.9.3` in the
`robot-zoo` extra. Do not delete geometry or weaken model-load gates to avoid
this dependency.

## Bootstrap commands

```bash
conda activate soma-retargeter-v2
python -m pip install -e '.[robot-zoo,test]'

export ROBOT_ZOO_CACHE="${ROBOT_ZOO_CACHE:-$HOME/.cache/soma_retargeter/robot_zoo}"

python -m soma_retargeter.tools.fetch_robot_zoo_sources \
  --fetch \
  --strict-required \
  --cache-root "$ROBOT_ZOO_CACHE" \
  --output-inventory "$ROBOT_ZOO_CACHE/source_inventory.json" \
  --output-manifest "$ROBOT_ZOO_CACHE/resolved_manifest.json"
```

To include GPL, CC-BY-SA, LGPL and NASA fetch-only entries without committing
them:

```bash
python -m soma_retargeter.tools.fetch_robot_zoo_sources \
  --fetch \
  --include-optional \
  --cache-root "$ROBOT_ZOO_CACHE"
```

Then use the generated manifest:

```bash
python -m soma_retargeter.tools.validate_kinematic_profile_v3 \
  --manifest "$ROBOT_ZOO_CACHE/resolved_manifest.json" \
  --output-dir artifacts/retargeting_v3_step2_next \
  --low-discrepancy-count 32 \
  --deterministic-rerun
```

## Scope remaining for Codex

Asset resolution does not turn an `algorithm_failed` model into a pass. The
next round must address numerical Jacobian stability, task-specific
reachability gates, rotation-frame consistency, semantic maps for available
partial models, and capability-aware projection quality. Those requirements
are defined in `/goal.md` on this branch.
