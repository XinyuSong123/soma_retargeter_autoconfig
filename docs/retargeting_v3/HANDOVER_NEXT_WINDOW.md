# Handover — Retargeting V3 / Step 3.4 Global Residual Quality

This file is the starting point for a new Codex/chat window. Read it first, then read `goal.md` fully.

---

## 0. Current repository and branch

Repository:

```text
XinyuSong123/soma_retargeter_autoconfig
```

Current working branch for the next development step:

```text
retargeting-v3-step3-4-global-residual-quality
```

This branch was created from the closed Step 3.3 final HEAD:

```text
1a8ae37670a3f3f8c49cefd78c6bf7292aa15489
```

Base branch:

```text
retargeting-v3-step3-3-global-solver-quality-hardening
```

The current `goal.md` on this branch is:

```text
Goal — Step 3.4：Global Residual Quality / Task Coverage Hardening
```

Do not continue development on the closed Step 3.3 branch except as historical baseline. Step 3.4 must write new artifacts and leave closed Step 3.1.1 / Step 3.2 / Step 3.3 artifacts intact.

---

## 1. Immediate checkout commands

Use the project environment, not `env_isaaclab`:

```bash
conda activate soma-retargeter-v2
cd ~/Desktop/soma_retargeter_autoconfig

git fetch origin --prune
git switch retargeting-v3-step3-4-global-residual-quality || \
  git switch --track -c retargeting-v3-step3-4-global-residual-quality origin/retargeting-v3-step3-4-global-residual-quality

git pull --ff-only

git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor 1a8ae37670a3f3f8c49cefd78c6bf7292aa15489 HEAD
git status --short

git lfs install
git lfs pull
git lfs fsck
```

Expected branch:

```text
retargeting-v3-step3-4-global-residual-quality
```

Expected HEAD should be at or after the bootstrap goal commit on this branch:

```text
0b4f831d282e0603eb23bd8e1f9ed17665b9538a
```

If local `git status --short` is dirty before starting, stash or move local files. Do not generate artifacts from a dirty worktree.

---

## 2. Closed baseline from Step 3.3

Step 3.3 is closed as a global solver quality hardening milestone.

Final evidence from the previous step:

```text
final_head = 1a8ae37670a3f3f8c49cefd78c6bf7292aa15489
GitHub Actions run = 28295149155
strict audit = PASS, blocking_count = 0
focused tests = 33 passed
git lfs fsck = OK
```

Artifact truth that must remain preserved:

```text
in_scope_total = 44
full_humanoid_total = 32
partial_total = 3
negative_total = 9

solver_backed_smoke_attempted_count = 32
solver_backed_completed_count = 32
solver_backed_count = 32
residual_only_count = 0

deterministic_compared_count = 44
deterministic_matched_count = 44

runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
partial_runtime_passed_count = 3
negative_control_runtime_passed_count = 9

high_residual_warning_count = 32
joint_limit_warning_count = 1
joint_limit_smoke_warning_count = 0
```

Interpretation:

```text
Step 3.3 removed hard joint-limit failures.
It did not improve high residuals or produce runtime_quality_passed full humanoids.
```

---

## 3. Step 3.4 purpose

Step 3.4 starts the next technical phase:

```text
Global residual quality / task coverage / anchor reliability hardening.
```

This means:

1. Keep all Step 3.3 invariants: 44 rows, 32/3/9, 32 solver-backed completions, residual_only_count=0, failed_count=0.
2. Attack the remaining high residual warning problem globally.
3. Improve task coverage, anchor reliability, and residual normalization in a way that is model-agnostic and auditable.
4. Produce new Step 3.4 artifacts under `artifacts/retargeting_v3_step3_4_global_residual_quality/`.
5. Do not overwrite closed Step 3.1.1 / Step 3.2 / Step 3.3 artifacts.
6. Do not use robot-specific thresholds, whitelists, math, or per-robot IK weights.

---

## 4. Files to inspect first

Read these in order:

```text
goal.md
docs/retargeting_v3/STEP3_3_GLOBAL_SOLVER_QUALITY_HANDOFF.md
docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md
artifacts/retargeting_v3_step3_3_global_solver_quality/quality_summary.json
artifacts/retargeting_v3_step3_3_global_solver_quality/quality_delta_vs_step3_2.json
artifacts/retargeting_v3_step3_3_global_solver_quality/model_matrix.json
artifacts/retargeting_v3_step3_3_global_solver_quality/solver_smoke_matrix.json
artifacts/retargeting_v3_step3_3_global_solver_quality/generic_smoke_matrix.json
artifacts/retargeting_v3_step3_3_global_solver_quality/solver_config.json
artifacts/retargeting_v3_step3_3_global_solver_quality/solver_diagnostics_matrix.json
artifacts/retargeting_v3_step3_3_global_solver_quality/acceptance_ledger.json
soma_retargeter/runtime/v3/generic_smoke.py
soma_retargeter/runtime/v3/fleet_harness.py
soma_retargeter/runtime/v3/quality_metrics.py
soma_retargeter/runtime/v3/runtime_quality_gates.py
soma_retargeter/runtime/v3/runtime_status.py
soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py
scripts/audit_retargeting_v3_step3_3_global_solver_quality.py
.github/workflows/retargeting_v3_step3_3_global_solver_quality.yml
tests/v3/test_step3_3_global_solver_quality_*.py
```

Do not infer Step 3.4 requirements from old Step 2 terminal logs.

---

## 5. Immediate next task

Implement according to `goal.md`, not by improvising scope.

The first development actions should be:

```text
1. Build residual taxonomy from Step 3.3 artifacts.
2. Add Step 3.4 audit skeleton and focused failing tests.
3. Add global task coverage / anchor reliability / residual normalization evidence.
4. Wire Step 3.4 artifact generation to a new artifact directory.
5. Generate quality_delta_vs_step3_3.json, residual_taxonomy.json, task_coverage_matrix.json, anchor_reliability_matrix.json.
6. Iterate until high residual count or residual distribution improves, or produce honest BLOCKED diagnostics.
7. Run focused tests, audit, deterministic rerun, LFS fsck, final HEAD CI.
```

If `high_residual_warning_count` remains 32 and residual distribution metrics do not improve, final status must be BLOCKED with diagnostics, not PASS.

---

## 6. Hard prohibitions

Do not:

```text
- go back to Step 2 branches;
- continue development on closed Step 3.3 branch;
- use env_isaaclab for V3 tests;
- modify Step 2 artifacts or thresholds;
- overwrite Step 3.1.1 / Step 3.2 / Step 3.3 closed artifacts;
- rewrite production NewtonPipeline default behavior;
- default-enable runtime override;
- add robot-specific math, thresholds, weights, or whitelist exceptions;
- add contact/collision/temporal smoothing in this step;
- claim visual motion quality is solved;
- label warned rows as runtime_quality_passed unless all pass gates are truly satisfied;
- promote negative controls into humanoid quality counts;
- start Step 3.5 automatically.
```

---

## 7. Short status for the next assistant / Codex

```text
We are on retargeting-v3-step3-4-global-residual-quality, created from closed Step 3.3 final HEAD 1a8ae37670a3f3f8c49cefd78c6bf7292aa15489. Step 3.3 removed hard failures: runtime_quality_failed_count is now 0, solver_backed_count remains 32, residual_only_count remains 0, and deterministic is 44/44. The remaining problem is residual quality: runtime_quality_passed_count is 0, runtime_quality_warned_count is 32, and high_residual_warning_count is 32. Step 3.4 must improve high residuals globally through task coverage, anchor reliability, and residual normalization evidence. Do not tune per robot, do not weaken gates, and do not overwrite closed artifacts.
```
