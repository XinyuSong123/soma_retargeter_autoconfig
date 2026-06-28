# Handover — Retargeting V3 / Step 4.0 Full Pipeline Acceptance

This file is the starting point for a new Codex/chat window. Read it first, then read `goal.md` fully.

---

## 0. Current repository and branch

Repository:

```text
XinyuSong123/soma_retargeter_autoconfig
```

Current working branch for the next long-run development step:

```text
retargeting-v3-step4-full-pipeline-acceptance
```

This branch was created from the closed Step 3.4 final HEAD:

```text
77e7c02393a6678ccab40cdb847021d7d94392c9
```

Base branch:

```text
retargeting-v3-step3-4-global-residual-quality
```

The current `goal.md` on this branch is:

```text
Goal — Step 4.0：Full Pipeline Acceptance / Release-Candidate Closure
```

Do not continue development on the closed Step 3.4 branch except as historical baseline. Step 4.0 must write new artifacts and leave closed Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 artifacts intact.

---

## 1. Immediate checkout commands

Use the project environment, not `env_isaaclab`:

```bash
conda activate soma-retargeter-v2
cd ~/Desktop/soma_retargeter_autoconfig

git fetch origin --prune
git switch retargeting-v3-step4-full-pipeline-acceptance || \
  git switch --track -c retargeting-v3-step4-full-pipeline-acceptance origin/retargeting-v3-step4-full-pipeline-acceptance

git pull --ff-only

git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor 77e7c02393a6678ccab40cdb847021d7d94392c9 HEAD
git status --short

git lfs install
git lfs pull
git lfs fsck
```

Expected branch:

```text
retargeting-v3-step4-full-pipeline-acceptance
```

Expected HEAD should be at or after the bootstrap goal commit on this branch:

```text
e9a4cf292e530a8b061cfcf33be069e5fb212dc6
```

If local `git status --short` is dirty before starting, stash or move local files. Do not generate artifacts from a dirty worktree.

---

## 2. Closed baseline from Step 3.4

Step 3.4 is closed as a global residual quality evidence milestone.

Final evidence from the previous step:

```text
final_head = 77e7c02393a6678ccab40cdb847021d7d94392c9
GitHub Actions run = 28327096821
strict audit = PASS, blocking_count = 0, finding_count = 0
focused tests = 21 passed
Step 3.3 + Step 3.4 focused tests = 39 passed
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
task_coverage_mean = 1.0
task_coverage_min = 1.0
anchor_reliability_mean = 0.994791666667
anchor_reliability_min = 0.833333333333
```

Interpretation:

```text
Step 3.4 improved raw residual severity and task coverage globally.
It did not reduce high_residual_warning_count below 32.
It did not produce runtime_quality_passed full-humanoid rows.
The remaining blocker is mostly rotation-dominant residual quality under unchanged gates.
```

---

## 3. Step 4.0 purpose

Step 4.0 is a full-pipeline long run, not another small incremental patch.

It must close the full V3 flow from:

```text
asset/profile/target-stream inputs
→ runtime local profile
→ multi-clip solver-backed smoke
→ global residual/orientation quality
→ trajectory exports
→ temporal/contact/collision diagnostics
→ artifact/audit/CI/handoff evidence
```

Primary quality breakthrough required for `PASS_RC`:

```text
runtime_quality_passed_count > 0
OR high_residual_warning_count < 32
OR audited residual/orientation distribution breakthrough relative to Step 3.4
```

If no breakthrough is possible under global methods, final status must be `BLOCKED_RESIDUAL_QUALITY` with complete diagnostics, not a fake PASS.

---

## 4. Files to inspect first

Read these in order:

```text
goal.md
docs/retargeting_v3/STEP3_4_GLOBAL_RESIDUAL_QUALITY_HANDOFF.md
docs/retargeting_v3/STEP3_3_GLOBAL_SOLVER_QUALITY_HANDOFF.md
docs/retargeting_v3/STEP3_2_SOLVER_BACKED_GENERIC_SMOKE_HANDOFF.md
docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md
artifacts/retargeting_v3_step3_4_global_residual_quality/quality_summary.json
artifacts/retargeting_v3_step3_4_global_residual_quality/quality_delta_vs_step3_3.json
artifacts/retargeting_v3_step3_4_global_residual_quality/residual_taxonomy.json
artifacts/retargeting_v3_step3_4_global_residual_quality/task_coverage_matrix.json
artifacts/retargeting_v3_step3_4_global_residual_quality/anchor_reliability_matrix.json
artifacts/retargeting_v3_step3_4_global_residual_quality/model_matrix.json
artifacts/retargeting_v3_step3_4_global_residual_quality/solver_smoke_matrix.json
artifacts/retargeting_v3_step3_4_global_residual_quality/generic_smoke_matrix.json
artifacts/retargeting_v3_step3_4_global_residual_quality/solver_config.json
artifacts/retargeting_v3_step3_4_global_residual_quality/solver_diagnostics_matrix.json
artifacts/retargeting_v3_step3_4_global_residual_quality/acceptance_ledger.json
soma_retargeter/runtime/v3/generic_smoke.py
soma_retargeter/runtime/v3/fleet_harness.py
soma_retargeter/runtime/v3/quality_metrics.py
soma_retargeter/runtime/v3/runtime_quality_gates.py
soma_retargeter/runtime/v3/runtime_status.py
soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py
scripts/audit_retargeting_v3_step3_4_global_residual_quality.py
.github/workflows/retargeting_v3_step3_4_global_residual_quality.yml
tests/v3/test_step3_4_global_residual_quality_*.py
```

Do not infer Step 4.0 requirements from old Step 2 logs or older branches.

---

## 5. Immediate next task

Implement according to `goal.md`, not by improvising scope.

The first development actions should be:

```text
1. Build full baseline forensics from Step 3.4 artifacts.
2. Add Step 4 audit skeleton and focused failing tests.
3. Add global orientation residual / normalization / full pipeline evidence design.
4. Add trajectory export and temporal diagnostic schemas.
5. Add support/contact/collision diagnostic schemas as diagnostic-only unless strongly validated.
6. Wire new Step 4 artifact generation under artifacts/retargeting_v3_step4_full_pipeline_acceptance/.
7. Iterate until a quality breakthrough or honest BLOCKED_RESIDUAL_QUALITY.
8. Run focused tests, audit, deterministic rerun, LFS fsck, final HEAD CI, strict final-head audit.
```

---

## 6. Hard prohibitions

Do not:

```text
- go back to Step 2 branches;
- continue development on closed Step 3.4 branch;
- use env_isaaclab for V3 tests;
- modify Step 2 artifacts or thresholds;
- overwrite Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 closed artifacts;
- rewrite production/default runtime behavior;
- default-enable runtime override;
- add robot-specific math, thresholds, weights, or whitelist exceptions;
- remove robots or clips to make metrics look better;
- hide raw residual regression via normalization;
- claim visual/deployment readiness without evidence;
- label warned rows as runtime_quality_passed unless all pass gates are truly satisfied;
- promote negative controls into humanoid quality counts;
- start the next stage automatically.
```

---

## 7. Short status for the next assistant / Codex

```text
We are on retargeting-v3-step4-full-pipeline-acceptance, created from closed Step 3.4 final HEAD 77e7c02393a6678ccab40cdb847021d7d94392c9. Step 3.4 preserved 44 rows, 32/3/9, solver_backed_count=32, residual_only_count=0, deterministic 44/44, and runtime_quality_failed_count=0. The remaining blocker is residual quality: runtime_quality_passed_count=0, runtime_quality_warned_count=32, high_residual_warning_count=32, mostly rotation-dominant. Step 4.0 must run the full V3 pipeline acceptance flow with orientation/residual quality, trajectory exports, temporal/contact/collision diagnostics, strict audit, CI, and honest PASS_RC/BLOCKED status. Do not tune per robot, do not weaken gates, and do not overwrite closed artifacts.
```
