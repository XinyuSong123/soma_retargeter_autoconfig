# Handover — Retargeting V3 / Step 3.1.1 Full-Fleet Runtime Hardening

This file is the starting point for a new chat window. Read it before reading the long historical conversation.

## 0. Current repository and branch

Repository:

```text
XinyuSong123/soma_retargeter_autoconfig
```

Current working branch for the next step:

```text
retargeting-v3-step3-full-fleet-hardening
```

Current remote HEAD at the time this handover was written:

```text
bff97b3d703a2899190a47b9e681d6444a07fb02
```

Latest commit message:

```text
Refresh Step 3.1 artifacts after CI workflow fix
```

This branch is based on:

```text
retargeting-v3-step3-runtime-quality-eval
fc0d21a7c00efbdc0c9d64177d4638354066d77e
```

The current `goal.md` on this branch is Step 3.1.1:

```text
Goal — Step 3.1.1：Full-Fleet Runtime Hardening
```

The new window should continue from this branch, not from any older Step 2 branch.

---

## 1. Immediate checkout commands

Use the project environment, not `env_isaaclab`:

```bash
conda activate soma-retargeter-v2
cd ~/Desktop/soma_retargeter_autoconfig

git fetch origin --prune
git switch retargeting-v3-step3-full-fleet-hardening || \
  git switch --track -c retargeting-v3-step3-full-fleet-hardening origin/retargeting-v3-step3-full-fleet-hardening

git pull --ff-only

git branch --show-current
git rev-parse HEAD
git status --short

git lfs install
git lfs pull
git lfs fsck
```

Expected branch:

```text
retargeting-v3-step3-full-fleet-hardening
```

Expected HEAD should be at or after:

```text
bff97b3d703a2899190a47b9e681d6444a07fb02
```

If local `git status --short` is dirty before starting, stash or move the local files. Do not generate artifacts from a dirty worktree.

---

## 2. Important warning about old local logs

The user’s local terminal history contains old output from:

```text
retargeting-v3-step2-integrated-local
```

That branch is an obsolete Step 2 integration branch. It has old `104 blockers`, outdated artifacts, and older tests. Do not interpret those old failures as current Step 3.1.1 status.

There is also an older `env_isaaclab` run that failed because it did not have `mujoco`. Ignore that environment for this project. The correct environment is:

```text
soma-retargeter-v2
```

Known good package environment previously used:

```text
Python 3.12.x
mujoco 3.8.1
newton 1.0.0
warp 1.12.x / 1.14.x depending on local environment
numpy 2.x
scipy 1.17.1
```

---

## 3. What has been completed so far

### Step 2.3.1 offline profile/capability acceptance

Accepted baseline:

```text
44 in-scope Robot Zoo/local assets
32 full humanoid profiles passed offline
3 partial humanoids passed offline
9 negative controls rejected correctly
algorithm_failed = 0
deterministic rerun = 44 / 44 matched
Git LFS fsck OK
```

This means the offline V3 profile/capability compiler is currently accepted as a trusted input.

### Step 3.0 runtime shadow integration

Branch completed earlier:

```text
retargeting-v3-step3-runtime-profile-shadow
1b0d223841909bec3fb9b850dee5c4fad1956be1
```

Step 3.0 established:

```text
default / disabled runtime path unchanged
shadow mode computes V3 target diagnostics but does not mutate IK inputs
RPO override experimental smoke works
G1 override remains fail-closed on fingerprint mismatch
CI passed for Step 3.0
```

Step 3.0 was intentionally small: RPO/G1 only.

### Step 3.1 full-fleet runtime quality eval

Branch before hardening:

```text
retargeting-v3-step3-runtime-quality-eval
fc0d21a7c00efbdc0c9d64177d4638354066d77e
```

It expanded the matrix to all 44 in-scope models, but had two major issues:

1. Artifacts were generated from a dirty worktree.
2. Generic smoke was only neutral-FK residual evaluation, yet high-residual rows were marked `runtime_quality_passed`.

### Step 3.1.1 current branch

Current branch fixed the above status honesty problem. It now explicitly reports that no full humanoid has solver-backed runtime-quality pass yet.

Current artifact summary:

```text
in_scope_total = 44
full_humanoid_total = 32
partial_total = 3
negative_total = 9

runtime_quality_passed_count = 0
runtime_quality_warned_count = 23
runtime_quality_failed_count = 9
partial_runtime_passed_count = 3
negative_control_runtime_passed_count = 9

solver_backed_count = 0
residual_only_count = 32
deterministic_compared_count = 44
deterministic_matched_count = 44
```

Interpretation:

```text
Step 3.1.1 is an honest full-fleet evaluation milestone, not a runtime-quality success milestone.
```

The full humanoids are still residual-only evaluation rows. Those rows are now correctly warned/failed instead of being overclaimed as quality passes.

---

## 4. Current true status

Current state should be treated as:

```text
Step 3.1.1 Full-Fleet Runtime Hardening: BLOCKED / nearly ready
```

Why not PASS yet?

1. Current remote HEAD has no visible GitHub Actions run/status at the time of review.
2. Agent F handoff still contains pending final CI placeholders.
3. Test artifacts are incomplete / placeholder-like:

```text
artifacts/retargeting_v3_step3_runtime_quality/test_results/pytest_summary.json
status = not_run_by_artifact_writer
```

4. The audit supports `--require-final-head-ci`, but the default audit run does not require this self-referential final CI gate.

What is already good?

1. Clean provenance appears fixed in `environment.json`.
2. Residual-only rows are no longer labeled runtime quality pass.
3. 44-row full-fleet matrix is present.
4. Quality statuses are now honest:

```text
0 full runtime quality passed
23 runtime quality warned
9 runtime quality failed
```

5. Deterministic artifact hash reports 44/44 matched.

---

## 5. Files to inspect first in the new window

Read these files in this order:

```text
goal.md
artifacts/retargeting_v3_step3_runtime_quality/environment.json
artifacts/retargeting_v3_step3_runtime_quality/quality_summary.json
artifacts/retargeting_v3_step3_runtime_quality/model_matrix.json
artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json
docs/retargeting_v3/subagents/step3_1_agent_f_red_team.md
.github/workflows/retargeting_v3_step3_full_fleet.yml
scripts/audit_retargeting_v3_step3_full_fleet.py
```

Also inspect representative full-humanoid rows in:

```text
artifacts/retargeting_v3_step3_runtime_quality/model_matrix.json
```

Specifically confirm the current status pattern:

```text
runtime_quality_warned rows have residual_only=true and quality_pass_allowed=false
runtime_quality_failed rows expose joint-limit / residual reasons
negative controls are not promoted
partial models stay partial
```

---

## 6. Immediate next task

Do **not** broaden scope. Do **not** start tuning or optimization.

Next task is final evidence closure for Step 3.1.1:

```text
1. Wait for / trigger current branch CI.
2. Verify CI run is on current HEAD or newer.
3. Ensure required jobs pass.
4. Write workflow_run_id, head_sha, conclusion and job conclusions into Agent F handoff and/or acceptance ledger.
5. Run audit with --require-final-head-ci.
6. Replace placeholder pytest artifacts with real targeted/v3 test output, or clearly document why only scoped CI tests are accepted.
7. Re-run final artifact audit.
8. Push final evidence commit.
```

If final-head CI is still absent, final status remains BLOCKED.

---

## 7. Recommended exact commands for final evidence closure

After checkout and LFS:

```bash
# Check whether current HEAD has a workflow run in GitHub UI/API.
# If not, push an empty/doc-only commit or rerun workflow_dispatch if appropriate.

git rev-parse HEAD
```

Run local targeted tests:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step3_full_fleet_acceptance_audit.py \
  tests/v3/test_step3_runtime_quality_gates_status_semantics.py
```

Run artifact audit in strict final-CI mode after CI evidence is recorded:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_full_fleet.py \
  --artifact-dir artifacts/retargeting_v3_step3_runtime_quality \
  --source-root . \
  --require-final-head-ci \
  --write-report artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json
```

If the audit fails, fix only the evidence/CI/provenance issue. Do not weaken runtime quality status semantics.

---

## 8. What PASS means for Step 3.1.1

A Step 3.1.1 PASS should mean:

```text
44-row matrix complete
clean provenance confirmed
runtime status labels honest
residual-only full humanoid rows are warned/failed, not passed
RPO/G1 pipeline controls retained
negative controls not promoted
final HEAD CI green
Agent F final handoff contains real workflow metadata
audit with --require-final-head-ci passes
```

PASS does **not** mean:

```text
all robot motions look good
all full humanoids pass runtime quality
solver-backed runtime quality exists for all robots
production retargeting is solved
```

The honest current quality conclusion is:

```text
0 / 32 full humanoids have runtime_quality_passed
23 / 32 full humanoids are runtime_quality_warned
9 / 32 full humanoids are runtime_quality_failed
```

This is useful: it identifies the next real optimization/solver work.

---

## 9. Likely next phase after Step 3.1.1

Only after Step 3.1.1 final evidence closure, the next phase should be:

```text
Step 3.2: solver-backed generic runtime smoke / runtime quality improvement
```

Do not start it automatically.

Expected Step 3.2 direction:

1. Replace residual-only FK evaluation with chain-projection frame smoke or generic Newton IK smoke.
2. Keep global gates, no robot-specific tuning.
3. Prioritize reducing the 9 failed rows and 23 warned rows.
4. Retain RPO/G1 pipeline-backed controls.
5. Continue to avoid contact/collision/temporal smoothing until basic solver-backed smoke is credible.

---

## 10. Hard prohibitions for the new window

Do not:

```text
- go back to retargeting-v3-step2-integrated-local;
- use env_isaaclab for V3 tests;
- treat old Step 2 blockers as current;
- modify Step 2 artifacts;
- modify offline thresholds;
- rename warned/failed rows to pass;
- weaken the audit to hide final-head CI absence;
- claim full repo pytest passed unless it actually exits 0;
- claim runtime visual quality is solved;
- add robot-specific math or thresholds;
- tune IK weights in this final evidence closure round.
```

---

## 11. Short status for the new assistant

If you need a single paragraph:

```text
We are on retargeting-v3-step3-full-fleet-hardening. Step 2.3.1 offline profile acceptance is trusted. Step 3.0 RPO/G1 runtime shadow passed. Step 3.1 full-fleet matrix now covers all 44 in-scope assets. The latest hardening branch fixed dirty provenance and fixed the overclaim that residual-only full humanoid FK evaluations were runtime_quality_passed. Current truthful quality result is: 0 full humanoid runtime_quality_passed, 23 warned, 9 failed, 3 partial passed, 9 negative passed, deterministic 44/44. Remaining blocker is final evidence closure: current HEAD needs CI evidence, Agent F handoff must be updated with real workflow metadata, audit should pass with --require-final-head-ci, and test artifacts should be real or explicitly scoped. Do not start optimization until this is closed.
```
