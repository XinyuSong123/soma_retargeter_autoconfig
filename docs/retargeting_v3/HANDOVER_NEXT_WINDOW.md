# Handover — Retargeting V3 / Step 4.3 Normalized Residual Gate Reconciliation

This file is the starting point for a new Codex/chat window. Read it first, then read `goal.md` fully.

---

## 0. Current repository and branch

Repository:

```text
XinyuSong123/soma_retargeter_autoconfig
```

Current working branch for the next development step:

```text
retargeting-v3-step4-3-normalized-residual-gate-reconciliation
```

This branch was created from the Step 4.2 orientation policy runtime scoring branch:

```text
retargeting-v3-step4-2-orientation-policy-runtime-scoring
```

The current `goal.md` on this branch is:

```text
Goal — Step 4.3：Normalized Residual Gate Reconciliation
```

Do not continue development on the Step 4.2 branch except as historical baseline. Step 4.3 must write new artifacts and leave closed Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 / Step 4.1 / Step 4.2 artifacts intact.

---

## 1. Immediate checkout commands

Use the project environment, not `env_isaaclab`:

```bash
conda activate soma-retargeter-v2
cd ~/Desktop/soma_retargeter_autoconfig

git fetch origin --prune
git switch retargeting-v3-step4-3-normalized-residual-gate-reconciliation || \
  git switch --track -c retargeting-v3-step4-3-normalized-residual-gate-reconciliation origin/retargeting-v3-step4-3-normalized-residual-gate-reconciliation

git pull --ff-only

git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor origin/retargeting-v3-step4-2-orientation-policy-runtime-scoring HEAD
git status --short

git lfs install
git lfs pull
git lfs fsck
```

Expected branch:

```text
retargeting-v3-step4-3-normalized-residual-gate-reconciliation
```

Expected HEAD should be at or after the bootstrap goal commit on this branch:

```text
34723fab28d56ac3298ee8683e7ed3ddee76200c
```

If local `git status --short` is dirty before starting, stash or move local files. Do not generate artifacts from a dirty worktree.

---

## 2. Step 4.2 baseline status

Step 4.2 proved the parent-relative orientation policy can be active in runtime scoring evidence:

```text
release_candidate_status = PASS_RC
primary_quality_breakthrough = true
orientation_policy_active_for_scoring = true
active_runtime_scoring_orientation_policy = parent_relative_runtime_inv_target
orientation_policy_production_default_changed = false
runtime_override_default_enabled = false
```

Step 4.2 preserved full-pipeline invariants:

```text
in_scope_total = 44
full_humanoid_total = 32
partial_total = 3
negative_total = 9
solver_backed_smoke_attempted_count = 32
solver_backed_completed_count = 32
solver_backed_count = 32
residual_only_count = 0
runtime_quality_failed_count = 0
trajectory_exports_count = 128
temporal_continuity_finite_count = 128
support_contact_diagnostic_count = 128
collision_proxy_diagnostic_count = 128
deterministic_compared_count = 44
deterministic_matched_count = 44
```

Step 4.2 active scoring improved residual distributions:

```text
p95_orientation_integrated_residual_delta_vs_step4_1 = -0.959127833526
median_orientation_integrated_residual_delta_vs_step4_1 = -0.683986984223
max_orientation_integrated_residual_delta_vs_step4_1 = -0.681007367545
p95_normalized_task_residual_p95_delta_vs_step4_1 = -0.006465520975
median_normalized_task_residual_p95_delta_vs_step4_1 = -0.038211678732
raw_residual_regression_count = 0
normalization_hides_raw_residual_regression = false
denominator_inflation_detected = false
```

But Step 4.2 still has no gate-count breakthrough:

```text
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
high_residual_warning_count = 32
rows_newly_passing = 0
rows_still_warned = 32
```

Gate blockers:

```text
high_task_residual = 32
normalized_task_residual_p95_above_pass_gate = 32
normalized_task_residual_p95_above_warn_gate = 32
solver_convergence_weak = 1
solver_success_fraction_below_one = 1
```

Interpretation:

```text
Step 4.2 made active scoring better, but normalized residual gates still block every full humanoid. Step 4.3 must reconcile normalized residual scale and gate semantics globally without weakening safety or hiding raw residuals.
```

---

## 3. Step 4.3 purpose

Step 4.3 must answer:

```text
Why do active-scoring raw/orientation residuals improve strongly while normalized_task_residual_p95 remains above the warn/pass gates for all 32 full humanoids?
```

Primary breakthrough required for `PASS_RC`:

```text
runtime_quality_passed_count > 0
OR high_residual_warning_count < 32
OR rows_below_candidate_warn_gate > 0 with audited release-quality v2 semantics
OR p95_normalized_task_residual_p95_delta <= -0.10 with no raw residual regression and no denominator inflation
```

If no safe normalization/gate policy can break through, final status must be honest:

```text
BLOCKED_GATE_RECONCILIATION
BLOCKED_NORMALIZATION_INTEGRITY
BLOCKED_SCORING_SCALE_AMBIGUITY
```

Do not fake PASS_RC.

---

## 4. Files to inspect first

Read these in order:

```text
goal.md
docs/retargeting_v3/STEP4_2_ORIENTATION_POLICY_RUNTIME_SCORING_HANDOFF.md
docs/retargeting_v3/STEP4_1_ORIENTATION_RESIDUAL_BREAKTHROUGH_HANDOFF.md
docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md
artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/quality_summary.json
artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/acceptance_ledger.json
artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/gate_reconciliation_report.json
artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/runtime_scoring_delta_vs_step4_1.json
artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/orientation_policy_runtime_impact_report.json
artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/scoring_normalization_audit.json
artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/orientation_integrated_residual_matrix.json
soma_retargeter/runtime/v3/quality_metrics.py
soma_retargeter/runtime/v3/generic_smoke.py
soma_retargeter/runtime/v3/fleet_harness.py
soma_retargeter/runtime/v3/runtime_quality_gates.py
soma_retargeter/tools/run_v3_full_pipeline_acceptance.py
scripts/audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring.py
.github/workflows/retargeting_v3_step4_2_orientation_policy_runtime_scoring.yml
tests/v3/test_step4_2_orientation_policy_runtime_scoring_*.py
```

---

## 5. Immediate next task

Implement according to `goal.md`, not by improvising scope.

The first development actions should be:

```text
1. Build normalized residual scale audit from Step 4.2 artifacts.
2. Build gate semantics audit explaining legacy pass/warn thresholds.
3. Add Step 4.3 audit skeleton and focused failing tests.
4. Evaluate global normalization and gate candidate policies.
5. Add gate_reconciliation_v2_report and normalization_policy_selection.
6. Regenerate full-pipeline artifacts under artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation/.
7. Run focused tests, audit, deterministic rerun, LFS fsck, final HEAD CI.
8. Write final handoff with PASS_RC or honest blocked gate/normalization status.
```

---

## 6. Hard prohibitions

Do not:

```text
- go back to Step 2 branches;
- continue development on Step 4.2 branch;
- use env_isaaclab for V3 tests;
- modify Step 2 artifacts or thresholds;
- overwrite Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 / Step 4.1 / Step 4.2 artifact trees;
- rewrite production/default runtime behavior;
- default-enable runtime override;
- add robot-specific math, thresholds, weights, or whitelist exceptions;
- remove robots or clips to make metrics look better;
- silently relax legacy pass/warn gates;
- call diagnostic-only candidate gates old runtime_quality_passed;
- hide raw residual regression via normalization;
- use denominator inflation;
- claim visual/deployment readiness without evidence;
- start the next stage automatically.
```

---

## 7. Short status for the next assistant / Codex

```text
We are on retargeting-v3-step4-3-normalized-residual-gate-reconciliation, created from Step 4.2. Step 4.2 integrated the parent-relative orientation policy into active runtime scoring evidence and achieved a strong residual distribution improvement: p95 orientation-integrated residual delta -0.959, raw residual regression 0, no denominator inflation. However runtime quality counts remain unchanged: 0 passed, 32 warned, 0 failed, high_residual_warning_count 32. All rows are still blocked by normalized_task_residual_p95_above_pass_gate and normalized_task_residual_p95_above_warn_gate. Step 4.3 must reconcile normalized residual scale and gate semantics globally without robot-specific tuning, gate weakening, or hiding raw residuals.
```
