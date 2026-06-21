# Robot Zoo Asset Bootstrap Report

Status: **asset-source preparation complete**  
Branch: `retargeting-v3-assets-ready`

## What was fixed

The previous Step-2 run reported 16 `source_unavailable` entries. They were not
missing from the public internet; the validation run had only inspected the
local cache and had not materialized their fixed upstream repositories.

A committed source specification now exists at:

```text
assets/robot_zoo/source_lock.json
```

It records the upstream repository, fixed ref, cache path and model-relative
path for all 16 entries:

### Required sources

- `atlas_drc_urdf`
- `atlas_v4_urdf`
- `bolt_urdf`
- `cassie_urdf`
- `draco3_urdf`
- `mujoco_humanoid_mjcf`
- `rhea_urdf`
- `romeo_urdf`
- `sigmaban_urdf`
- `simple_humanoid_urdf`
- `upkie_urdf`

### Optional fetch-only sources

- `fourier_gr1_urdf`
- `jaxon_urdf`
- `talos_urdf`
- `valkyrie_urdf`
- `robonaut2_urdf`

The lock is derived from `robot_descriptions==2.0.0` and its pinned repository
registry. MuJoCo Menagerie remains fixed at
`feadf76d42f8a2162426f7d226a3b539556b3bf5`.

## Fetch implementation

`python -m soma_retargeter.tools.sync_robot_zoo_v3` is no longer only a plan
printer. It now:

1. requires `robot_descriptions==2.0.0`;
2. compares its repository registry against the committed source lock;
3. fetches required public descriptions into `ROBOT_DESCRIPTIONS_CACHE`;
4. materializes the fixed MuJoCo Menagerie checkout for direct entries;
5. resolves every selected manifest model after fetching;
6. writes a machine-readable inventory;
7. exits nonzero if any required source remains unavailable;
8. keeps GPL/CC/NASA sources external and optional unless explicitly requested.

The fetched repositories are external cache data. They are not copied into this
Apache-licensed repository.

## Loader dependency fix

JVRC URDF and Unitree Go2 URDF previously failed because their Collada meshes
required `pycollada`. The `robot-zoo` project extra now pins:

```text
mujoco==3.8.1
robot_descriptions==2.0.0
pycollada==0.9.3
```

## Bootstrap command

From the repository root in the intended Conda environment:

```bash
python -m pip install -e '.[robot-zoo]'
bash scripts/bootstrap_robot_zoo_assets.sh
```

To also fetch the optional fetch-only descriptions into the external cache:

```bash
bash scripts/bootstrap_robot_zoo_assets.sh --include-fetch-only
```

Expected output:

```text
artifacts/retargeting_v3_assets/source_sync.json
artifacts/retargeting_v3_assets/source_sync.verify.json
```

Both files must report `status="passed"` and an empty
`required_failures` list before the next algorithm validation run.

## What is intentionally not claimed

This asset preparation does not make the current retargeting algorithm pass all
robots. It removes source-cache and Collada dependency blockers so the next run
can expose real semantic, kinematic and numerical failures.

Still assigned to the next development goal:

- verified semantic maps for newly available positive humanoids;
- backend-aware engine Jacobians and finite-difference validation;
- near-zero Jacobian-column classification;
- task-specific rank/subspace stability gates;
- full rerun of every required model;
- real G1 23DoF source onboarding.

## Private-asset boundary

The lock and bootstrap only reference public sources already represented by the
manifest. No company-internal model or `cxxx_190` identifier is present or
accepted.
