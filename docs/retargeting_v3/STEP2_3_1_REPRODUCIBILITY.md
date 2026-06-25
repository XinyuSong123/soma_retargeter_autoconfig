# Step 2.3.1 Capability Reproducibility

This file documents the Agent D provenance protocol for hardened capability artifacts.

## Source Commit

- Source branch: `retargeting-v3-step2-capability-acceptance-hardening`
- Current recorded source commit: `b84f12000599e41af34d3e2a9ace19410cc870be`
- Remote resolvability check:
  - `git ls-remote --heads origin retargeting-v3-step2-capability-acceptance-hardening`
  - observed `b84f12000599e41af34d3e2a9ace19410cc870be`

Final artifact publication must use a new frozen code commit `C` after all code-side blockers are resolved and pushed.

## Clean Generation

Run from the main worktree after code freeze:

```bash
C=<pushed source commit>
scripts/generate_capability_artifacts_clean.sh "$C"
```

The generator:

1. Fetches the expected remote branch and verifies `C` is reachable from it.
2. Creates a detached clean source worktree at `C`.
3. Requires `git status --short` to be empty before and after the run.
4. Runs `git lfs pull` and `git lfs fsck`.
5. Writes pytest text, JUnit XML, and pytest summary under external `$OUT/artifacts/test_results`.
6. Runs the full capability artifact generation command into external `$OUT/artifacts`.
7. Runs `scripts/audit_retargeting_v3_capability.py` against `$OUT/artifacts`.
8. Writes `environment.json`, `lfs_state.json`, `commands.txt`, and `acceptance_ledger.json` with `scripts/write_capability_provenance.py`.

The generator does not push and does not create a git branch.

## Artifact Commit Check

After copying `$OUT/artifacts` to `artifacts/retargeting_v3_step2_capability` and committing artifact commit `A`, verify:

```bash
git merge-base --is-ancestor "$C" "$A"
git diff --name-only "$C..$A" -- soma_retargeter tests scripts .github
PYTHONPATH=. python scripts/write_capability_provenance.py \
  --artifact-dir artifacts/retargeting_v3_step2_capability \
  --check
```

The diff command must print nothing. Any source/core drift after `C` invalidates the artifact commit.

## Current Metadata State

- `environment.json` now records a remote-resolvable source commit and clean source-worktree fields.
- `lfs_state.json` records `git lfs fsck` return code `0`, zero pointer files, zero missing LFS objects, and `materialized_snapshot_count=38`.
- `acceptance_ledger.json` now records a real Step 2.3 capability audit command and return code.
- The current live audit still fails; this is not final acceptance evidence.

Known current blockers include missing final test artifacts, missing final capability artifact files, baseline/certificate/status integration blockers, CI workflow blockers, and stale Agent F handoff state. Those are outside Agent D ownership and must be resolved before final clean artifact publication.
