# Robot Zoo Clean Fetch and Vendor Workflow

## Run

```bash
git fetch origin
git checkout -b retargeting-v3-step2-assets-clean-sync \
  origin/retargeting-v3-step2-assets-clean-sync

bash scripts/fetch_and_vendor_robot_zoo_assets.sh
```

The script creates a separate clean worktree and pushes:

```text
retargeting-v3-step2-assets-vendored
```

The current checkout may contain old generated artifacts; it is not used as the generation worktree.

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

## Strict and diagnostic modes

Normal strict run:

```bash
ALLOW_PARTIAL_ASSETS=0 bash scripts/fetch_and_vendor_robot_zoo_assets.sh
```

Diagnostic run that records unresolved sources without claiming completion:

```bash
ALLOW_PARTIAL_ASSETS=1 PUSH_ASSET_BRANCH=0 KEEP_WORKTREE=1 \
  bash scripts/fetch_and_vendor_robot_zoo_assets.sh
```

A diagnostic run must not be reported as Step 2.2 PASS.

## Existing output branch

The script refuses to overwrite an existing remote output branch. Review or delete the old branch, or choose a new name:

```bash
OUTPUT_BRANCH=retargeting-v3-step2-assets-vendored-2 \
  bash scripts/fetch_and_vendor_robot_zoo_assets.sh
```

## Security and license rules

- Only manifest-declared public sources are processed.
- Unknown upstream scripts are not executed.
- MuJoCo Menagerie is checked out at the exact manifest SHA.
- No visual/collision meshes are committed.
- No Git LFS is used.
- GPL/LGPL/CC-SA/NASA entries remain fetch-only.
- Local absolute paths and private denylist tokens fail generation.
- Franka Panda in the manifest is the public Franka Emika Panda model.
