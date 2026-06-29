# Handover — Retargeting V3 / Step 4.1 Orientation Residual Breakthrough

This file is the starting point for a new Codex/chat window. Read it first, then read `goal.md` fully.

---

## 0. Current repository and branch

Repository:

```text
XinyuSong123/soma_retargeter_autoconfig
```

Current working branch for the next development step:

```text
retargeting-v3-step4-1-orientation-residual-breakthrough
```

This branch was created from the current Step 4.0 full-pipeline acceptance branch:

```text
retargeting-v3-step4-full-pipeline-acceptance
```

The current `goal.md` on this branch is:

```text
Goal — Step 4.1：Global Orientation Residual Breakthrough
```

Do not continue development on the Step 4.0 branch except as historical baseline. Step 4.1 must write new artifacts and leave closed Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 artifacts intact.

---

## 1. Immediate checkout commands

Use the project environment, not `env_isaaclab`:

```bash
conda activate soma-retargeter-v2
cd ~/Desktop/soma_retargeter_autoconfig

git fetch origin --prune
git switch retargeting-v3-step4-1-orientation-residual-breakthrough || \
  git switch --track -c retargeting-v3-step4-1-orientation-residual-breakthrough origin/retargeting-v3-step4-1-orientation-residual-breakthrough

git pull --ff-only

git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor origin/retargeting-v3-step4-full-pipeline-acceptance HEAD
git status --short

git lfs install
git lfs pull
git lfs fsck
```

Expected branch:

```text
retargeting-v3-step4-1-orientation-residual-breakthrough
```

Expected HEAD should be at or after the bootstrap goal commit on this branch:

```text
d900e4adb8b7163ad6daebd806ae187d409add56
```

If local `git status --short` is dirty before starting, stash or move local files. Do not generate artifacts from a dirty worktree.

---

## 2. Step 4.0 baseline status

Step 4.0 produced the full-pipeline artifact tree but remained blocked by residual quality:

```text
release_candidate_status = BLOCKED_RESIDUAL_QUALITY
verdict = BLOCKED
primary_quality_breakthrough = false
blocking structural audit findings = 0
```

The artifact tree exists under:

```text
artifacts/retargeting_v3_step4_full_pipeline_acceptance/
```

Preserved Step 4.0 invariants:

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

clip_suite_count = 4
trajectory_exports_count = 128
temporal_continuity_finite_count = 128
support_contact_diagnostic_count = 128
collision_proxy_diagnostic_count = 128
```

Remaining blocker:

```text
high_residual_warning_count = 32
rotation_dominant_residual_count = 27
translation_dominant_residual_count = 5
runtime_quality_passed_count = 0
```

Interpretation:

```text
The pipeline plumbing is now mostly done. The next hard problem is global orientation residual semantics and target-frame consistency.
```

---

## 3. Step 4.1 purpose

Step 4.1 must target the remaining residual blocker directly:

```text
Global orientation residual breakthrough / target-frame semantics / quaternion residual math.
```

Primary breakthrough required for `PASS_RC`:

```text
runtime_quality_passed_count > 0
OR high_residual_warning_count < 32
OR rotation_dominant_residual_count < 27
OR audited p95_rotation_residual_p95 improvement without raw residual regression
```

If no global policy can break through, final status must be an honest blocked status such as:

```text
BLOCKED_ORIENTATION_SEMANTICS
BLOCKED_TARGET_FRAME_AMBIGUITY
BLOCKED_NORMALIZATION_INTEGRITY
```

Do not fake PASS_RC.

---

## 4. Files to inspect first

Read these in order:

```text
goal.md
docs/retargeting_v3/STEP4_FULL_PIPELINE_ACCEPTANCE_HANDOFF.md
docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md
artifacts/retargeting_v3_step4_full_pipeline_acceptance/quality_summary.json
artifacts/retargeting_v3_step4_full_pipeline_acceptance/acceptance_ledger.json
artifacts/retargeting_v3_step4_full_pipeline_acceptance/quality_delta_vs_step3_4.json
artifacts/retargeting_v3_step4_full_pipeline_acceptance/orientation_residual_taxonomy.json
artifacts/retargeting_v3_step4_full_pipeline_acceptance/normalization_audit.json
artifacts/retargeting_v3_step4_full_pipeline_acceptance/full_pipeline_matrix.json
artifacts/retargeting_v3_step4_full_pipeline_acceptance/clip_matrix.json
artifacts/retargeting_v3_step4_full_pipeline_acceptance/solver_smoke_matrix.json
artifacts/retargeting_v3_step4_full_pipeline_acceptance/generic_smoke_matrix.json
artifacts/retargeting_v3_step4_full_pipeline_acceptance/trajectory_export_manifest.json
soma_retargeter/runtime/v3/generic_smoke.py
soma_retargeter/runtime/v3/fleet_harness.py
soma_retargeter/runtime/v3/quality_metrics.py
soma_retargeter/runtime/v3/runtime_quality_gates.py
soma_retargeter/tools/run_v3_full_pipeline_acceptance.py
scripts/audit_retargeting_v3_step4_full_pipeline_acceptance.py
.github/workflows/retargeting_v3_step4_full_pipeline_acceptance.yml
tests/v3/test_step4_full_pipeline_acceptance_*.py
```

---

## 5. Immediate next task

Implement according to `goal.md`, not by improvising scope.

The first development actions should be:

```text
1. Build orientation residual forensics from Step 4.0 artifacts.
2. Add Step 4.1 audit skeleton and focused failing tests.
3. Add orientation frame semantics matrix and quaternion/log-map residual math audit.
4. Add global offset candidate sweep and orientation policy selection.
5. Regenerate full-pipeline artifacts under artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/.
6. Run focused tests, audit, deterministic rerun, LFS fsck, final HEAD CI.
7. Write final handoff with PASS_RC or honest blocked status.
```

---

## 6. Hard prohibitions

Do not:

```text
- go back to Step 2 branches;
- continue development on Step 4.0 branch;
- use env_isaaclab for V3 tests;
- modify Step 2 artifacts or thresholds;
- overwrite Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 artifact trees;
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
We are on retargeting-v3-step4-1-orientation-residual-breakthrough, created from the current Step 4.0 full-pipeline branch. Step 4.0 completed full pipeline plumbing and diagnostics but ended as BLOCKED_RESIDUAL_QUALITY: runtime_quality_passed_count=0, runtime_quality_warned_count=32, high_residual_warning_count=32, rotation_dominant_residual_count=27. Step 4.1 must attack global orientation residual semantics: target/runtime/source frame conventions, quaternion/log-map residual math, rest-offset policies, orientation normalization, and global policy selection. Do not tune per robot, do not weaken gates, do not overwrite closed artifacts, and do not fake PASS_RC.
```
