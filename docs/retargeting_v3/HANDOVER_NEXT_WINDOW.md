# Handover — Retargeting V3 / Step 4.2 Orientation Policy Runtime Scoring

This file is the starting point for a new Codex/chat window. Read it first, then read `goal.md` fully.

---

## 0. Current repository and branch

Repository:

```text
XinyuSong123/soma_retargeter_autoconfig
```

Current working branch for the next development step:

```text
retargeting-v3-step4-2-orientation-policy-runtime-scoring
```

This branch was created from the Step 4.1 orientation residual breakthrough branch:

```text
retargeting-v3-step4-1-orientation-residual-breakthrough
```

The current `goal.md` on this branch is:

```text
Goal — Step 4.2：Orientation Policy Runtime Scoring Integration
```

Do not continue development on the Step 4.1 branch except as historical baseline. Step 4.2 must write new artifacts and leave closed Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 / Step 4.1 artifacts intact.

---

## 1. Immediate checkout commands

Use the project environment, not `env_isaaclab`:

```bash
conda activate soma-retargeter-v2
cd ~/Desktop/soma_retargeter_autoconfig

git fetch origin --prune
git switch retargeting-v3-step4-2-orientation-policy-runtime-scoring || \
  git switch --track -c retargeting-v3-step4-2-orientation-policy-runtime-scoring origin/retargeting-v3-step4-2-orientation-policy-runtime-scoring

git pull --ff-only

git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor origin/retargeting-v3-step4-1-orientation-residual-breakthrough HEAD
git status --short

git lfs install
git lfs pull
git lfs fsck
```

Expected branch:

```text
retargeting-v3-step4-2-orientation-policy-runtime-scoring
```

Expected HEAD should be at or after the bootstrap goal commit on this branch:

```text
ccfc12b21e8ddfad689125d8a159d71d0c475e7d
```

If local `git status --short` is dirty before starting, stash or move local files. Do not generate artifacts from a dirty worktree.

---

## 2. Step 4.1 baseline status

Step 4.1 proved an audited global orientation residual semantics breakthrough:

```text
release_candidate_status = PASS_RC
primary_quality_breakthrough = true
selected_policy = parent_relative_runtime_inv_target
p95_rotation_residual_p95_delta_vs_step4_0 = -0.859991637555
raw_residual_regression_count = 0
normalization_hides_raw_residual_regression = false
```

But Step 4.1 did not change runtime quality counts:

```text
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
high_residual_warning_count = 32
```

It also did not change production/default runtime behavior:

```text
production_default_changed = false
runtime_override_default_enabled = false
runtime_quality_gates_changed = false
runtime_quality_passed_count_changed_by_policy = false
```

Interpretation:

```text
Step 4.1 proved the orientation comparison policy is better as audited diagnostic evidence.
Step 4.2 must determine whether that policy can become active runtime scoring evidence without weakening gates or changing production defaults silently.
```

---

## 3. Step 4.2 purpose

Step 4.2 must promote the selected global orientation policy into Step 4.2 runtime scoring / quality metrics evidence behind an explicit flag/config.

Primary breakthrough required for `PASS_RC`:

```text
runtime_quality_passed_count > 0
OR high_residual_warning_count < 32
OR active-scoring residual distribution improves enough to satisfy audit
```

If no active-scoring breakthrough is possible, final status must be honest:

```text
BLOCKED_SCORING_INTEGRATION
BLOCKED_GATE_RECONCILIATION
BLOCKED_NORMALIZATION_INTEGRITY
```

Do not fake PASS_RC.

---

## 4. Files to inspect first

Read these in order:

```text
goal.md
docs/retargeting_v3/STEP4_1_ORIENTATION_RESIDUAL_BREAKTHROUGH_HANDOFF.md
docs/retargeting_v3/STEP4_FULL_PIPELINE_ACCEPTANCE_HANDOFF.md
docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/quality_summary.json
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/acceptance_ledger.json
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/orientation_delta_vs_step4_0.json
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/orientation_policy_selection.json
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/release_candidate_impact_report.json
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/full_pipeline_matrix.json
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/clip_matrix.json
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/solver_smoke_matrix.json
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/generic_smoke_matrix.json
soma_retargeter/runtime/v3/generic_smoke.py
soma_retargeter/runtime/v3/fleet_harness.py
soma_retargeter/runtime/v3/quality_metrics.py
soma_retargeter/runtime/v3/runtime_quality_gates.py
soma_retargeter/tools/run_v3_full_pipeline_acceptance.py
scripts/audit_retargeting_v3_step4_1_orientation_residual_breakthrough.py
.github/workflows/retargeting_v3_step4_1_orientation_residual_breakthrough.yml
tests/v3/test_step4_1_orientation_residual_*.py
```

---

## 5. Immediate next task

Implement according to `goal.md`, not by improvising scope.

The first development actions should be:

```text
1. Trace why Step 4.1 orientation breakthrough did not affect runtime_quality_status.
2. Add Step 4.2 audit skeleton and focused failing tests.
3. Add active-vs-diagnostic orientation policy evidence.
4. Integrate parent_relative_runtime_inv_target into Step 4.2 scoring path behind explicit flag/config.
5. Add gate_reconciliation_report and runtime_scoring_delta_vs_step4_1.
6. Regenerate full-pipeline artifacts under artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/.
7. Run focused tests, audit, deterministic rerun, LFS fsck, final HEAD CI.
8. Write final handoff with PASS_RC or honest blocked status.
```

---

## 6. Hard prohibitions

Do not:

```text
- go back to Step 2 branches;
- continue development on Step 4.1 branch;
- use env_isaaclab for V3 tests;
- modify Step 2 artifacts or thresholds;
- overwrite Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 / Step 4.1 artifact trees;
- rewrite production/default runtime behavior;
- default-enable runtime override;
- promote diagnostic-only learned rest offsets into active policy;
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
We are on retargeting-v3-step4-2-orientation-policy-runtime-scoring, created from Step 4.1. Step 4.1 selected a global parent_relative_runtime_inv_target orientation policy and achieved p95 rotation residual delta -0.859991637555 with no raw residual regression. However runtime quality counts remain unchanged: 0 passed, 32 warned, 0 failed, high_residual_warning_count 32. Step 4.2 must integrate the selected global policy into active Step 4.2 runtime scoring evidence, reconcile gates, and prove whether this can reduce high residual warnings or create gate-valid pass rows. Do not tune per robot, do not weaken gates, do not change production defaults silently, and do not overwrite closed artifacts.
```
