# Robot Zoo Public Asset Resolution

Status: **source research and reproducible fetch plan complete**  
Branch: `retargeting-v3-step2-assets-ready`

## What is now fixed in the repository

1. `assets/robot_zoo/robot_zoo_manifest.json` is schema v2 and pins:
   - `robot_descriptions==2.0.0` / tag `v2.0.0` / commit `2431c405312001e10cb0a8d5315dbb2e90f0c732`;
   - MuJoCo Menagerie commit `feadf76d42f8a2162426f7d226a3b539556b3bf5`;
   - official Unitree ROS commit `267182b8521c8d6a631bab1fe63836873237a525`.
2. A machine-readable provider lock is stored in `assets/robot_zoo/source_lock.json`.
3. The official real G1 23DoF models are included as required entries:
   - `robots/g1_description/g1_23dof.urdf`;
   - `robots/g1_description/g1_23dof.xml`.
4. `scripts/bootstrap_robot_zoo_assets.py` fetches only manifest-listed public sources, checks pinned git revisions, computes model hashes, and fails when a required source is unavailable.
5. The missing COLLADA decoder dependency is explicit through the `robot-zoo` optional dependency and `assets/robot_zoo/requirements-assets.txt`.
6. Fetched repositories and model caches are ignored by Git. Fetch-only/GPL/CC/NASA assets are never copied into the Apache repository.
7. Private or unlisted assets are forbidden. No `cxxx_190` or company-internal model is present.

## Why the previous result had 16 `source_unavailable`

The prior full-zoo artifact was generated from a partially populated local cache. The resolver was usually run without source fetching. The missing set was dominated by public `robot_descriptions` entries such as Atlas, Cassie URDF, Draco, Romeo, SigmaBan, Simple Humanoid, Upkie, Rhea and optional fetch-only humanoids. This was an acquisition/cache problem, not a kinematics-formula result.

`robot_descriptions` documents that importing or pulling a description automatically downloads it into a cache. Version 2.0.0 also contains exact upstream repository revisions, so the bootstrap uses that immutable index rather than following upstream `main`.

## Why two models had `model_load_failed`

`jvrc_urdf` and `unitree_go2_urdf` reference COLLADA `.dae` geometry. Their failures were classified as a missing public `pycollada` dependency. The asset environment now declares `pycollada>=0.8,<1`; do not remove their geometry to make loading appear green.

## Bootstrap commands

Use the intended conda environment:

```bash
conda activate soma-retargeter-v2
python -m pip install -e '.[robot-zoo]'
```

Fetch and verify all 48 manifest entries, including fetch-only entries in the local cache:

```bash
python scripts/bootstrap_robot_zoo_assets.py
```

Verify an already-populated cache without network mutation:

```bash
python scripts/bootstrap_robot_zoo_assets.py --verify-only
```

Fetch/verify only required entries:

```bash
python scripts/bootstrap_robot_zoo_assets.py --required-only
```

The generated machine-local inventory is:

```text
assets/robot_zoo/resolved_source_inventory.json
```

It is intentionally ignored by Git because local paths are environment-specific. It must report:

```text
required_failure_count == 0
```

before full Robot Zoo validation begins.

## Cache layout

By default:

```text
assets/robot_zoo/cache/
  robot_descriptions/
  mujoco_menagerie/
  unitree_ros/
```

A different cache can be supplied with `--cache-root`. Reproduction artifacts should display it as `${ROBOT_ZOO_CACHE}` rather than a machine-specific absolute path.

## License handling

- Apache/MIT/BSD sources may be fetched and may later produce reviewed mesh-free kinematic snapshots.
- GPL, LGPL, CC-BY-SA and NASA sources remain fetch-only.
- Fetch-only models may be tested from the cache but must not be committed to this repository.
- Visual meshes are not committed by default.
- Every source remains tied to its manifest license and upstream revision.

## Remaining implementation boundary

The web-side source research, version selection, license classification and deterministic bootstrap inputs are complete. The next Codex pass still needs to:

1. integrate `pinned_git` source resolution into the normal validator for the two G1 23DoF entries;
2. run the bootstrap on the development machine and prove all required sources resolve;
3. add/fix verified semantic maps where current reports are `semantic_failed`;
4. repair Jacobian/reachability mathematics for current `algorithm_failed` models;
5. regenerate clean full-zoo artifacts.

These remaining tasks no longer require searching the web for robot asset locations or guessing upstream revisions.
