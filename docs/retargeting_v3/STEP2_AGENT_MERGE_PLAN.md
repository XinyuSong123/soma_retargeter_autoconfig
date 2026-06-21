# Step 2 Agent Integration Plan

This file accompanies `scripts/apply_step2_agent_merges.sh`.

## Purpose

The six Codex subagents pushed independent branches. The source Step-2 branch
still contains only the goal plus the pre-agent prototype. The script rebuilds
a local integration branch in the dependency order required by `goal.md`:

```text
A runtime truth
→ B verified semantics/sites
→ C calibration/targets
→ D canonical projection
→ E manifest-driven validation
→ F acceptance audit/red team
```

It does not modify `main`, `dev`, or the source branch, and does not push.

## Pinned branches and commits

| Agent | Remote branch | Integrated commits |
|---|---|---|
| A | `agent-a-runtime-model-truth` | `ca4c19c`, `dc560a6` |
| B | `agent-b-verified-semantics` | `b449eb2`, `c2551e2` |
| C | `agent-c-rest-calibration` | `b2db065`, `107ae72` |
| D | `agent-d-reachability` | `8dee96e` |
| E | `agent-e-full-robot-zoo-validation` | `b25b5f1`, `4ff8c41` |
| F | `agent-f-repro-ci-red-team` | `67f726b`, `84f5439` |

The script uses full commit SHAs internally.

## Selective integration decisions

### Agent E

The code commit is integrated. The later artifact commit `9549b9e` is skipped
because its run was explicitly a no-fetch scaffold:

- only 1 of 46 models ran;
- 45 were `source_unavailable`;
- cross-format checks were `not_run`;
- deterministic rerun was `not_run`;
- artifacts were generated from a dirty worktree.

Those artifacts are useful historical evidence but are not final Step-2
validation results.

### Agent F

The audit and CI implementation are integrated. Stale generated audit output
from the pre-integration prototype is removed, because it reports 105 blockers
against an older tree and contains a local absolute path. The audit must be
rerun after integration and after final Robot Zoo artifact generation.

### Agent D

The branch did not include the required handoff file. The script adds an
explicit integration-generated reconstruction, marked as incomplete evidence.
This must not be confused with an Agent D self-attestation.

## How to run

```bash
git checkout retargeting-v3-step2-kinematic-core
git pull --ff-only
chmod +x scripts/apply_step2_agent_merges.sh
./scripts/apply_step2_agent_merges.sh
```

To use another local integration branch name:

```bash
./scripts/apply_step2_agent_merges.sh my-step2-integration
```

To rebuild an existing local integration branch:

```bash
FORCE_RESET=1 ./scripts/apply_step2_agent_merges.sh
```

To merge without running the selected pytest suite:

```bash
RUN_TESTS=0 ./scripts/apply_step2_agent_merges.sh
```

## Expected outcome

The script creates `retargeting-v3-step2-integrated-local`, cherry-picks the
pinned commits, removes stale generated audit results, adds the missing Agent D
record, runs compile checks, and by default runs a combined pytest subset.

A successful merge is **not** equivalent to Step 2 passing. Before entering
Step 3, the integrated branch still needs:

1. resolution of combined-test failures;
2. verified semantic maps for all required positive humanoids;
3. wiring `project_canonical_targets()` into profile/full-zoo compilation;
4. full-source Robot Zoo validation, not no-fetch scaffolding;
5. same-source G1 URDF→MJCF strict equivalence;
6. deterministic two-run validation;
7. clean final artifacts from the final commit;
8. Agent F audit and red-team result `PASS`;
9. CI status checks passing.

## Existing remote review branch

A conservative remote review branch, `retargeting-v3-step2-integrated-review`,
was also created during inspection. It contains selectively integrated agent
code for review, but it has not been used to claim Step-2 completion. Running
the script locally remains the recommended reproducible path because it shows
each pinned cherry-pick and exposes any local conflict or test failure directly.
