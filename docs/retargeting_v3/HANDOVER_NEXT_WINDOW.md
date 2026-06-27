# Handover — Retargeting V3 / Step 3.2 Solver-Backed Generic Runtime Smoke

This file is the starting point for a new Codex/chat window. Read it first, then read `goal.md`.

---

## 0. Current repository and branch

Repository:

```text
XinyuSong123/soma_retargeter_autoconfig
```

Current working branch for the next development step:

```text
retargeting-v3-step3-2-solver-backed-generic-smoke
```

This branch was created from the closed Step 3.1.1 final HEAD:

```text
26817de67bdda0cb315a1237b53c30e4d8199c78
```

Base branch:

```text
retargeting-v3-step3-full-fleet-hardening
```

The current `goal.md` on this branch is:

```text
Goal — Step 3.2：Solver-Backed Generic Runtime Smoke
```

Do not continue development on `retargeting-v3-step3-full-fleet-hardening`; that branch is the closed Step 3.1.1 evidence baseline.

---

## 1. Immediate checkout commands

Use the project environment, not `env_isaaclab`:

```bash
conda activate soma-retargeter-v2
cd ~/Desktop/soma_retargeter_autoconfig

git fetch origin --prune
git switch retargeting-v3-step3-2-solver-backed-generic-smoke || \
  git switch --track -c retargeting-v3-step3-2-solver-backed-generic-smoke origin/retargeting-v3-step3-2-solver-backed-generic-smoke

git pull --ff-only

git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor 26817de67bdda0cb315a1237b53c30e4d8199c78 HEAD
git status --short

git lfs install
git lfs pull
git lfs fsck
```

Expected branch:

```text
retargeting-v3-step3-2-solver-backed-generic-smoke
```

Expected HEAD should be at or after the bootstrap goal commit on this branch:

```text
e78c7032cf947c46d5eb9dc4b1662831bbb8770b
```

If local `git status --short` is dirty before starting, stash or move local files. Do not generate artifacts from a dirty worktree.

---

## 2. Closed baseline from Step 3.1.1

Step 3.1.1 is closed as an honest full-fleet runtime evidence milestone.

Final evidence from the previous step:

```text
final_head = 26817de67bdda0cb315a1237b53c30e4d8199c78
GitHub Actions run = 28286161586
strict audit = PASS, blocking_count = 0
focused tests = 20 passed
git lfs fsck = OK
```

Artifact truth that must remain preserved:

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
deterministic_matched_count = 44
```

Interpretation:

```text
Step 3.1.1 proved clean provenance, final HEAD CI, and honest status labels.
It did not solve runtime quality or provide solver-backed generic smoke.
```

---

## 3. Step 3.2 purpose

Step 3.2 starts the next real technical phase:

```text
Replace or augment residual-only FK evaluation with generic solver-backed runtime smoke evidence.
```

This means:

1. Try solver-backed generic smoke for all 32 full humanoid profiles.
2. Keep 44-row fleet accounting and 32/3/9 partition.
3. Use global gates only.
4. Do not use robot-specific thresholds, whitelists, or IK weight tuning.
5. Keep residual-only fallback honest: residual-only rows cannot be `runtime_quality_passed`.
6. Generate a new Step 3.2 artifact tree; do not overwrite closed Step 3.1.1 artifacts.

---

## 4. Files to inspect first

Read these in order:

```text
goal.md
docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md
docs/retargeting_v3/STEP3_1_FULL_FLEET_RUNTIME_QUALITY_ACCEPTANCE.md
docs/retargeting_v3/subagents/step3_1_agent_f_red_team.md
artifacts/retargeting_v3_step3_runtime_quality/environment.json
artifacts/retargeting_v3_step3_runtime_quality/quality_summary.json
artifacts/retargeting_v3_step3_runtime_quality/model_matrix.json
artifacts/retargeting_v3_step3_runtime_quality/generic_smoke_matrix.json
artifacts/retargeting_v3_step3_runtime_quality/pipeline_backed_matrix.json
artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json
soma_retargeter/runtime/v3/generic_smoke.py
soma_retargeter/runtime/v3/fleet_harness.py
soma_retargeter/runtime/v3/quality_metrics.py
soma_retargeter/runtime/v3/runtime_quality_gates.py
soma_retargeter/runtime/v3/runtime_status.py
soma_retargeter/runtime/v3/runtime_local_profile.py
soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py
scripts/audit_retargeting_v3_step3_full_fleet.py
.github/workflows/retargeting_v3_step3_full_fleet.yml
tests/v3/test_step3_full_fleet_acceptance_audit.py
tests/v3/test_step3_runtime_quality_gates_status_semantics.py
```

Do not infer Step 3.2 requirements from old Step 2 terminal logs.

---

## 5. Immediate next task

Implement according to `goal.md`, not by improvising scope.

The first development actions should be:

```text
1. Create Step 3.2 artifact schema and audit skeleton.
2. Add focused tests that fail if residual-only rows are promoted to runtime_quality_passed.
3. Add generic solver-backed smoke design/implementation.
4. Wire it into the full-fleet runner behind explicit Step 3.2 flags.
5. Generate artifacts under artifacts/retargeting_v3_step3_2_solver_backed_smoke/.
6. Run deterministic rerun, focused tests, audit, LFS fsck, and final HEAD CI.
```

If solver-backed completed count remains zero, final status must be BLOCKED with diagnostics, not PASS.

---

## 6. Hard prohibitions

Do not:

```text
- go back to retargeting-v3-step2-integrated-local;
- continue development on the closed Step 3.1.1 hardening branch;
- use env_isaaclab for V3 tests;
- modify Step 2 artifacts or thresholds;
- overwrite Step 3.1.1 closed artifacts to hide Step 3.2 failures;
- rewrite production NewtonPipeline default behavior;
- default-enable runtime override;
- add robot-specific math, thresholds, or whitelist exceptions;
- tune IK/objective weights per robot;
- add contact/collision/temporal smoothing in this step;
- claim visual motion quality is solved;
- label residual-only smoke as runtime_quality_passed;
- promote negative controls into humanoid quality counts;
- start Step 3.3 automatically.
```

---

## 7. Short status for the next assistant / Codex

```text
We are on retargeting-v3-step3-2-solver-backed-generic-smoke, created from closed Step 3.1.1 final HEAD 26817de67bdda0cb315a1237b53c30e4d8199c78. Step 3.1.1 proved clean full-fleet evidence but had solver_backed_count=0 and residual_only_count=32. Step 3.2 must implement generic solver-backed runtime smoke evidence for full humanoids, preserve 44-row accounting, keep global gates and honest statuses, generate a new artifact tree, and pass a new strict audit/CI. Do not tune IK, do not use robot-specific thresholds, and do not overwrite closed Step 3.1.1 artifacts.
```
