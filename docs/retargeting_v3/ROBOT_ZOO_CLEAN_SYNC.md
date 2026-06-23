# Robot Zoo Clean Fetch and Vendor Workflow

## Run

```bash
git fetch origin
git checkout retargeting-v3-step2-assets-vendored
git pull --ff-only
git lfs install
git lfs pull
git lfs fsck
```

The committed Assets44 branch contains the frozen 44/2 scope and the mesh-free kinematic snapshots used by later validation.

## Cache and Git boundaries

Full upstream repositories are stored outside Git under `${ROBOT_ZOO_CACHE}` or the default sibling cache directory. They may contain meshes.

Git receives only:

```text
assets/robot_zoo/source_inventory.json
assets/robot_zoo/robot_zoo_lock.json
assets/robot_zoo/snapshots/**
```

A permissive snapshot contains a mesh-free URDF/MJCF, upstream license files, and `SOURCE.json` with repository, commit and hashes.

Entries marked `fetch_only` are downloaded to the external cache for validation but never copied into the Apache repository.

## Git LFS

Git LFS is accepted for formats already selected by the repository's `.gitattributes`, including MJCF/XML snapshots.

Every environment that loads these assets must run:

```bash
git lfs install
git lfs pull
git lfs fsck
```

GitHub Actions must use:

```yaml
- uses: actions/checkout@v4
  with:
    lfs: true
```

The following remain forbidden even through LFS:

- full upstream repositories;
- visual/collision meshes;
- fetch-only or non-redistributable models;
- unpinned or oversized assets;
- pointer-only validation presented as a successful model load.

## Frozen Assets44 scope

```text
46 public sources resolved
44 in scope
38 committed snapshots
5 fetch-only cached models
1 project-local RPO
2 deferred snapshots: berkeley_humanoid_urdf, romeo_urdf
```

The two deferred entries remain recorded in `robot_zoo_lock.json` but are outside future algorithm gates.

## Security and license rules

- Only manifest-declared public sources are processed.
- Unknown upstream scripts are not executed.
- MuJoCo Menagerie is checked out at the exact manifest SHA.
- No visual/collision meshes are committed.
- GPL/LGPL/CC-SA/NASA entries remain fetch-only.
- Local absolute paths and private denylist tokens fail generation.
- Franka Panda in the manifest is the public Franka Emika Panda model.
