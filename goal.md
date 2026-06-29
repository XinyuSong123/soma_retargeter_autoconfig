# Goal — Step 4.1：Global Orientation Residual Breakthrough（全局方向残差语义突破）

> **Codex 强制执行契约——不得自由发挥，不得拆成小步只做局部**
>
> 当前分支：`retargeting-v3-step4-1-orientation-residual-breakthrough`  
> 基线分支：`retargeting-v3-step4-full-pipeline-acceptance`  
> 基线提交：当前 `retargeting-v3-step4-full-pipeline-acceptance` HEAD  
> 当前阶段：**Step 4.1 Global Orientation Residual Breakthrough / Target-Frame Semantics**
>
> Step 4.0 已经完成 full-pipeline evidence artifact，但诚实状态是 `BLOCKED_RESIDUAL_QUALITY`，不是 `PASS_RC`。它保留了 44-row / 32-3-9 / solver-backed coverage / deterministic / trajectory exports / temporal/support/contact/collision diagnostics，但没有取得 primary quality breakthrough：`runtime_quality_passed_count=0`，`high_residual_warning_count=32`，`rotation_dominant_residual_count=27`。
>
> **本轮唯一核心目标：用全局 orientation residual / target-frame semantics / normalization math 改善 remaining residual-quality blocker，争取把 Step 4.0 的 `BLOCKED_RESIDUAL_QUALITY` 推进到 `PASS_RC` 或至少形成不可再简化的数学诊断。**
>
> **禁止修改 Step 2 artifacts / thresholds；禁止覆盖 Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 closed artifacts；禁止 robot-specific 数学、阈值、白名单或 per-robot IK 权重调参；禁止默认启用 production/runtime override；禁止把 warned 改名成 passed；禁止通过放宽 gates 制造 pass；禁止隐藏 raw residual regression；禁止声称视觉质量或真实部署完全解决，除非相应证据真实存在并由 audit 验证。**

---

## 0. 开始前必须执行

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

which python
python --version
python - <<'PY'
import mujoco, newton, warp, numpy, scipy
print('mujoco', mujoco.__version__)
print('newton', getattr(newton, '__version__', 'unknown'))
print('warp', getattr(warp, '__version__', 'unknown'))
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
PY
```

必须确认：

```text
branch = retargeting-v3-step4-1-orientation-residual-breakthrough
base Step 4.0 branch is ancestor
worktree clean
Git LFS fsck OK
snapshot XML/URDF materialized, not pointer-only
```

环境不满足时，只修环境或 checkout，不得改代码绕过。

---

## 1. 必须先阅读的文件

按顺序完整阅读，不得只读 summary：

1. `goal.md`（本文件）
2. `docs/retargeting_v3/STEP4_FULL_PIPELINE_ACCEPTANCE_HANDOFF.md`
3. `docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md`
4. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/quality_summary.json`
5. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/acceptance_ledger.json`
6. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/quality_delta_vs_step3_4.json`
7. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/orientation_residual_taxonomy.json`
8. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/normalization_audit.json`
9. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/full_pipeline_matrix.json`
10. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/clip_matrix.json`
11. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/solver_smoke_matrix.json`
12. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/generic_smoke_matrix.json`
13. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/trajectory_export_manifest.json`
14. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/temporal_continuity_matrix.json`
15. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/support_contact_diagnostics.json`
16. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/collision_proxy_diagnostics.json`
17. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/pipeline_config.json`
18. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/solver_config.json`
19. `artifacts/retargeting_v3_step4_full_pipeline_acceptance/red_team_report.json`
20. `soma_retargeter/runtime/v3/generic_smoke.py`
21. `soma_retargeter/runtime/v3/fleet_harness.py`
22. `soma_retargeter/runtime/v3/quality_metrics.py`
23. `soma_retargeter/runtime/v3/runtime_quality_gates.py`
24. `soma_retargeter/runtime/v3/runtime_status.py`
25. `soma_retargeter/runtime/v3/runtime_local_profile.py`
26. `soma_retargeter/tools/run_v3_full_pipeline_acceptance.py`
27. `soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py`
28. `scripts/audit_retargeting_v3_step4_full_pipeline_acceptance.py`
29. `.github/workflows/retargeting_v3_step4_full_pipeline_acceptance.yml`
30. `tests/v3/test_step4_full_pipeline_acceptance_*.py`
31. Step 3.4 artifacts under `artifacts/retargeting_v3_step3_4_global_residual_quality/` as immutable baseline

不得从旧 Step 2 logs 或旧 Step 3 branches 推断当前状态。不得改 closed artifact tree 来“修历史”。

---

## 2. 当前真实基线

Step 4.0 full-pipeline artifact status：

```text
release_candidate_status = BLOCKED_RESIDUAL_QUALITY
verdict = BLOCKED
primary_quality_breakthrough = false
blocking structural audit findings = 0
```

Step 4.0 preserved invariants：

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
clip_matrix_rows = 140
solver_smoke_rows = 128
trajectory_exports_count = 128
temporal_continuity_finite_count = 128
support_contact_diagnostic_count = 128
collision_proxy_diagnostic_count = 128
```

Step 4.0 residual blocker truth：

```text
high_residual_warning_count = 32
rotation_dominant_residual_count = 27
translation_dominant_residual_count = 5

median_raw_task_residual_p95 = 3.5578656079654998
p95_raw_task_residual_p95 = 4.26814045938395
max_raw_task_residual_p95 = 4.302029690497

median_normalized_task_residual_p95 = 0.997502471969
p95_normalized_task_residual_p95 = 0.9998567673000001
max_normalized_task_residual_p95 = 0.999930378142

median_rotation_residual_p95 = 2.50134081541
p95_rotation_residual_p95 = 2.90919255987585
max_rotation_residual_p95 = 2.921195688897
```

Step 4.0 normalization truth：

```text
denominator_inflation_detected = false
normalization_hides_raw_residual_regression = false
normalization_reconstruction_mismatch_count = 22
suspicious_rows = []
```

Interpretation：

```text
Full pipeline plumbing is now mostly done.
The remaining hard problem is orientation residual semantics and target-frame consistency.
```

---

## 3. 本轮唯一总目标

把 Step 4.0 的 residual-quality blocker 压缩到一个真实突破：

```text
Either achieve PASS_RC by producing real gate-valid quality improvement, or produce a mathematically precise BLOCKED_ORIENTATION_SEMANTICS diagnosis that identifies the exact global orientation frame/normalization blocker.
```

Primary quality target：

```text
runtime_quality_passed_count > 0
OR high_residual_warning_count < 32
OR rotation_dominant_residual_count < 27
OR p95_rotation_residual_p95 improves by an audited, nontrivial delta without raw residual regression
```

Preferred target：

```text
runtime_quality_passed_count >= 4
OR high_residual_warning_count <= 28
OR rotation_dominant_residual_count <= 20
OR p95_rotation_residual_p95_delta <= -0.25
```

Stretch target：

```text
runtime_quality_passed_count >= 8
OR high_residual_warning_count <= 24
OR p95_rotation_residual_p95_delta <= -0.50
```

Gold target：

```text
runtime_quality_passed_count >= 16
runtime_quality_failed_count = 0
high_residual_warning_count <= 16
```

If primary target cannot be reached under global methods, final status must be one of：

```text
BLOCKED_ORIENTATION_SEMANTICS
BLOCKED_TARGET_FRAME_AMBIGUITY
BLOCKED_NORMALIZATION_INTEGRITY
BLOCKED_CI_OR_PROVENANCE
```

Do not claim `PASS_RC` without a true primary quality target.

---

## 4. What exactly must be solved

Step 4.1 must stop treating orientation residual as a scalar afterthought. It must explicitly audit and, if possible, correct global frame semantics.

### 4.1 Global orientation target-frame semantics

For every orientation task, identify and record：

```text
target orientation frame convention
runtime model body frame convention
source motion frame convention
root frame convention
world/local frame choice
quaternion component order
quaternion sign canonicalization
left/right handedness assumption
axis-up convention
body rest-orientation offset
parent-relative vs world-relative orientation
```

Produce：

```text
orientation_frame_semantics_matrix.json
```

### 4.2 Quaternion/log-map residual formulation

Implement or audit global residual math：

```text
q_target normalized
q_runtime normalized
q_delta = q_target * inverse(q_runtime) or inverse(q_runtime) * q_target, chosen globally and justified
shortest-arc sign canonicalization
log-map residual vector
angle residual in radians
axis residual diagnostics
rotation residual p50/p95/max
raw residual preserved
normalized residual derived transparently
```

Produce：

```text
orientation_residual_math_audit.json
```

### 4.3 Body rest offset / canonicalization

A likely blocker is that target orientations and runtime body orientations are compared without a rest-frame offset. Step 4.1 must test global candidates：

```text
no_offset
rest_pose_body_offset
profile_rest_orientation_offset
source_to_runtime_rest_delta
parent_relative_rest_delta
```

Rules：

- Candidate selection must be global, not per robot.
- If per-row best candidate is only diagnostic, it cannot become runtime pass unless the final policy is global.
- All candidates must be recorded with residual distributions.

Produce：

```text
orientation_offset_candidate_matrix.json
orientation_policy_selection.json
```

### 4.4 Global semantic orientation task weighting

If orientation tasks are overweighted or underweighted, Step 4.1 may introduce global semantic weights only：

```text
torso_orientation_weight
head_orientation_weight
hand_orientation_weight
foot_orientation_weight
pelvis/root_orientation_weight
```

Rules：

- Must be one global config.
- No per robot or per clip weight tables.
- Raw residual cannot be hidden.
- Pass gates cannot be weakened.

### 4.5 Target-frame consistency across clips

For each full humanoid and clip, record consistency：

```text
clip_id
orientation residual distribution
dominant semantic anchor
dominant axis
frame convention candidate scores
clip-level pass/warn/fail status
```

Produce：

```text
orientation_clip_consistency_matrix.json
```

### 4.6 Release-candidate impact

After global orientation policy is selected, regenerate full-pipeline evidence and compare against Step 4.0.

Must produce：

```text
quality_delta_vs_step4_0.json
orientation_delta_vs_step4_0.json
release_candidate_impact_report.json
```

---

## 5. Strict invariants

All Step 4.1 work must preserve：

```text
44-row matrix
32/3/9 partition
solver_backed_smoke_attempted_count >= 32
solver_backed_completed_count >= 32
solver_backed_count >= 32
residual_only_count = 0
runtime_quality_failed_count = 0
partial_runtime_passed_count = 3
negative_control_runtime_passed_count = 9
negative controls not promoted
partial rows not counted as full-humanoid passes
trajectory exports present and finite
temporal diagnostics finite
support/contact diagnostics present
collision proxy diagnostics present
deterministic 44/44 matched
clean provenance
final HEAD CI green
```

Any regression in these invariants must result in：

```text
BLOCKED_PIPELINE_REGRESSION
```

---

## 6. Strict prohibitions

Do not：

```text
- modify Step 2 artifacts / thresholds；
- overwrite Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 closed artifact trees；
- modify historical Step 4.0 numbers to make Step 4.1 look better；
- rewrite production/default runtime behavior；
- default-enable runtime override；
- add robot-id special cases；
- add per-robot IK weights；
- add robot-specific threshold tables；
- whitelist / blacklist robots or clips；
- remove hard robots or clips from the suite；
- call diagnostic-only candidate search a production pass；
- relax pass gates；
- change warned labels to passed without metrics satisfying gates；
- hide raw residual regression with normalization；
- claim visual/deployment readiness without evidence；
- report full repo pytest passed unless it actually exits 0；
- start Step 4.2 automatically。
```

---

## 7. Artifact strategy

Create a new artifact tree only：

```text
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/
```

Required artifacts：

```text
environment.json
commands.txt
quality_summary.json
acceptance_ledger.json
full_pipeline_matrix.json
clip_matrix.json
solver_smoke_matrix.json
generic_smoke_matrix.json
deterministic_rerun.json
quality_delta_vs_step4_0.json
orientation_delta_vs_step4_0.json
release_candidate_impact_report.json
orientation_frame_semantics_matrix.json
orientation_residual_math_audit.json
orientation_offset_candidate_matrix.json
orientation_policy_selection.json
orientation_clip_consistency_matrix.json
normalization_audit.json
pipeline_config.json
solver_config.json
trajectory_export_manifest.json
temporal_continuity_matrix.json
support_contact_diagnostics.json
collision_proxy_diagnostics.json
pipeline_controls_reference.json 或 pipeline_backed_matrix.json
red_team_report.json
test_results/pytest.txt
test_results/pytest_summary.json
test_results/junit.xml
```

`quality_summary.json` must include：

```text
schema_version
base_step4_0_final_head
release_candidate_status
primary_quality_breakthrough
in_scope_total
full_humanoid_total
partial_total
negative_total
clip_suite_count
solver_backed_smoke_attempted_count
solver_backed_completed_count
solver_backed_count
residual_only_count
runtime_quality_passed_count
runtime_quality_warned_count
runtime_quality_failed_count
partial_runtime_passed_count
negative_control_runtime_passed_count
high_residual_warning_count
rotation_dominant_residual_count
translation_dominant_residual_count
median_rotation_residual_p95
p95_rotation_residual_p95
max_rotation_residual_p95
median_raw_task_residual_p95
p95_raw_task_residual_p95
max_raw_task_residual_p95
median_normalized_task_residual_p95
p95_normalized_task_residual_p95
max_normalized_task_residual_p95
normalization_reconstruction_mismatch_count
denominator_inflation_detected
normalization_hides_raw_residual_regression
trajectory_exports_count
temporal_continuity_finite_count
support_contact_diagnostic_count
collision_proxy_diagnostic_count
deterministic_compared_count
deterministic_matched_count
```

`orientation_policy_selection.json` must include：

```text
candidate_policies
global_selected_policy
selection_reason
why_policy_is_global
raw_residual_delta
rotation_residual_delta
pass_gate_impact
rejected_candidates
risk_notes
```

`orientation_frame_semantics_matrix.json` must include per task/row/clip：

```text
model_id
clip_id
semantic_anchor
runtime_body_name
target_frame
runtime_frame
source_frame
quaternion_order
sign_canonicalized
rest_offset_policy
parent_relative
world_relative
axis_convention_notes
validity_status
warning_reasons
```

---

## 8. Audit requirements

Add a new audit：

```text
scripts/audit_retargeting_v3_step4_1_orientation_residual_breakthrough.py
```

Audit must fail closed for structural issues and must distinguish honest blocked quality from pipeline failure.

Required gates：

```text
required artifacts present
44 rows present
32/3/9 partition preserved
base_step4_0_final_head recorded and matches current baseline
clean provenance
solver_backed_smoke_attempted_count >= 32
solver_backed_completed_count >= 32
solver_backed_count >= 32
residual_only_count == 0
runtime_quality_failed_count == 0
negative controls not promoted
partial rows not counted as full humanoid passes
trajectory_export_manifest exists and matches rows/clips
temporal_continuity_matrix exists and finite
support/contact diagnostics exists
collision proxy diagnostics exists
orientation_frame_semantics_matrix exists
orientation_residual_math_audit exists
orientation_offset_candidate_matrix exists
orientation_policy_selection exists
orientation_clip_consistency_matrix exists
normalization_audit proves raw residual not hidden
quality_delta_vs_step4_0 exists and is internally consistent
orientation_delta_vs_step4_0 exists and is internally consistent
release_candidate_impact_report exists
runtime_quality_passed rows require solver_backed=true and unchanged gates valid
warned rows cannot be hidden by status rename
no robot-specific threshold/whitelist/model-id special case introduced
numeric metrics finite and ordered
NaN/Inf counts zero for completed rows and exports
deterministic rerun 44/44 matched
pipeline controls retained/referenced
final HEAD CI evidence present in strict mode
```

Quality status rules：

```text
PASS_RC requires primary_quality_breakthrough=true and at least one real quality target.
BLOCKED_ORIENTATION_SEMANTICS is acceptable only with complete diagnostics and blocking_count=0.
BLOCKED_TARGET_FRAME_AMBIGUITY is acceptable only if candidate policies are audited and inconclusive.
BLOCKED_NORMALIZATION_INTEGRITY is acceptable only if normalization audit identifies unresolved mismatch/regression.
BLOCKED_PIPELINE_REGRESSION if any baseline invariant regresses.
BLOCKED_CI_OR_PROVENANCE if CI/provenance/LFS/strict audit incomplete.
```

Strict PASS_RC requires at least one：

```text
runtime_quality_passed_count > 0
high_residual_warning_count < 32
rotation_dominant_residual_count < 27
p95_rotation_residual_p95_delta <= -0.25
p95_raw_task_residual_p95_delta <= -0.10 with no normalized/raw integrity issue
```

If none holds, audit must return a blocked quality status, not PASS_RC.

---

## 9. Tests required

Add focused tests：

```text
tests/v3/test_step4_1_orientation_residual_audit.py
tests/v3/test_step4_1_orientation_residual_artifact_schema.py
tests/v3/test_step4_1_orientation_residual_frame_semantics.py
tests/v3/test_step4_1_orientation_residual_math.py
tests/v3/test_step4_1_orientation_residual_policy_selection.py
tests/v3/test_step4_1_orientation_residual_delta.py
tests/v3/test_step4_1_orientation_residual_status_semantics.py
tests/v3/test_step4_1_orientation_residual_no_robot_specific_tuning.py
tests/v3/test_step4_1_orientation_residual_exports_temporal.py
tests/v3/test_step4_1_orientation_residual_determinism.py
```

At minimum test：

```text
missing orientation_frame_semantics_matrix fails audit
missing orientation_residual_math_audit fails audit
missing orientation_policy_selection fails audit
missing orientation_delta_vs_step4_0 fails audit
PASS_RC rejected without primary quality breakthrough
blocked quality status accepted only with complete diagnostics
pipeline regression rejected
runtime_quality_failed_count > 0 rejected
residual_only_count > 0 rejected
solver coverage regression rejected
pass rows without solver-backed evidence rejected
pass rows without unchanged gates rejected
negative/partial promotion rejected
robot-specific threshold/whitelist/model-id special cases rejected
normalization hiding raw residual regression rejected
quaternion sign canonicalization recorded
log-map residual metrics finite
offset candidates recorded and globally selected/rejected
trajectory exports remain finite
temporal diagnostics remain finite
determinism 44/44 required
closed artifact trees not modified
```

Recommended focused test command：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step4_1_orientation_residual_audit.py \
  tests/v3/test_step4_1_orientation_residual_artifact_schema.py \
  tests/v3/test_step4_1_orientation_residual_frame_semantics.py \
  tests/v3/test_step4_1_orientation_residual_math.py \
  tests/v3/test_step4_1_orientation_residual_policy_selection.py \
  tests/v3/test_step4_1_orientation_residual_delta.py \
  tests/v3/test_step4_1_orientation_residual_status_semantics.py \
  tests/v3/test_step4_1_orientation_residual_no_robot_specific_tuning.py \
  tests/v3/test_step4_1_orientation_residual_exports_temporal.py \
  tests/v3/test_step4_1_orientation_residual_determinism.py
```

---

## 10. Long-run execution plan

### Phase A — Orientation forensics

1. Parse Step 4.0 full-pipeline artifacts.
2. Rank 32 full humanoids by rotation residual p95/max.
3. Rank clips by orientation residual contribution.
4. Identify dominant semantic anchors.
5. Identify whether residual is world/local, parent-relative, rest-offset, quaternion-sign, or axis convention driven.
6. Produce initial orientation frame semantics matrix.
7. Do not change status labels.

### Phase B — Tests and audit first

1. Add Step 4.1 audit skeleton.
2. Add missing-artifact tests.
3. Add quaternion/log-map math tests.
4. Add frame-semantics schema tests.
5. Add no robot-specific tuning tests.
6. Add policy selection tests.
7. Ensure missing artifacts fail before implementation.

### Phase C — Candidate policy sweep

Evaluate global candidates：

```text
candidate_0_current_step4_policy
candidate_1_quaternion_sign_only
candidate_2_world_delta_order_target_inv_runtime
candidate_3_world_delta_order_runtime_inv_target
candidate_4_parent_relative_delta
candidate_5_rest_pose_body_offset
candidate_6_profile_rest_offset
candidate_7_source_to_runtime_rest_delta
candidate_8_semantic_orientation_weight_balance
candidate_9_combined_global_best_policy
```

For each candidate record：

```text
median/p95/max rotation residual
raw task residual
normalized residual
pass/warn/fail counts
raw regression count
normalization integrity
risk notes
```

### Phase D — Select a global policy

Selection rules：

```text
Policy must be global.
Policy must not use model_id.
Policy must not relax gates.
Policy must not hide raw residual regression.
Policy must preserve solver coverage and failed_count=0.
Policy must improve at least one primary quality metric or honestly block.
```

### Phase E — Regenerate full pipeline artifacts

Generate under：

```text
artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/
```

Run command may be：

```bash
PYTHONPATH=. python soma_retargeter/tools/run_v3_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough \
  --baseline-step4-artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --enable-global-orientation-residual-hardening \
  --enable-orientation-frame-semantics-audit \
  --enable-full-pipeline-exports \
  --deterministic-rerun
```

Codex may adjust flags if it updates code/tests/docs consistently.

### Phase F — Evidence closure

1. Run focused tests.
2. Run local Step 4.1 audit.
3. Run deterministic rerun.
4. Run Git LFS fsck.
5. Verify clean worktree before/after artifact generation.
6. Commit artifacts and handoff.
7. Push.
8. Wait for final HEAD CI.
9. Run strict audit with `--require-final-head-ci`.
10. Confirm branch HEAD matches origin.
11. Write final handoff.
12. Do not start next phase.

---

## 11. Workflow requirements

Add：

```text
.github/workflows/retargeting_v3_step4_1_orientation_residual_breakthrough.yml
```

Required jobs：

```text
step4-1-static-and-unit
step4-1-artifact-audit
step4-1-orientation-math-smoke
step4-1-export-and-temporal-smoke
step4-1-lfs-and-snapshot-smoke
step4-1-pipeline-controls-reference
step4-1-final-head-ci-evidence
```

The final-head evidence job must print：

```text
workflow_run_id
head_sha
conclusion
job_conclusions
```

---

## 12. Recommended commands

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step4_1_orientation_residual_*.py \
  tests/v3/test_step4_full_pipeline_acceptance_*.py

PYTHONPATH=. python soma_retargeter/tools/run_v3_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough \
  --baseline-step4-artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --enable-global-orientation-residual-hardening \
  --enable-orientation-frame-semantics-audit \
  --enable-full-pipeline-exports \
  --deterministic-rerun

PYTHONPATH=. python scripts/audit_retargeting_v3_step4_1_orientation_residual_breakthrough.py \
  --artifact-dir artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough \
  --baseline-step4-artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --source-root .

PYTHONPATH=. python scripts/audit_retargeting_v3_step4_1_orientation_residual_breakthrough.py \
  --artifact-dir artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough \
  --baseline-step4-artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --source-root . \
  --require-final-head-ci

git lfs fsck
```

---

## 13. PASS / BLOCKED definitions

### `PASS_RC`

Requires：

```text
Step 4.0 baseline preserved
new Step 4.1 artifact tree generated from clean source
44-row full-fleet matrix preserved
32/3/9 partition preserved
solver_backed_count >= 32
residual_only_count = 0
runtime_quality_failed_count = 0
primary_quality_breakthrough = true
runtime_quality_passed_count > 0 OR high_residual_warning_count < 32 OR orientation residual distribution breakthrough
orientation frame semantics evidence complete
orientation math audit complete
orientation policy selection global and justified
normalization audit passes
trajectory exports finite
temporal diagnostics finite
deterministic 44/44 matched
no robot-specific tuning/threshold/whitelist introduced
focused tests passed
Git LFS fsck OK
final HEAD CI green
strict Step 4.1 audit passes
```

### `BLOCKED_ORIENTATION_SEMANTICS`

Allowed only if：

```text
all structural artifacts complete
baseline invariants preserved
orientation candidate sweep complete
no global policy improves quality enough
frame/offset/normalization blocker is documented
blocking_count = 0 for structural audit
```

### `BLOCKED_TARGET_FRAME_AMBIGUITY`

Allowed only if target-frame conventions cannot be disambiguated from existing artifacts without new source annotations.

### `BLOCKED_NORMALIZATION_INTEGRITY`

Allowed only if safe normalization cannot be selected without hiding raw residual regression.

### `BLOCKED_PIPELINE_REGRESSION`

Any baseline invariant regression.

### `BLOCKED_CI_OR_PROVENANCE`

Final CI/provenance/LFS/audit incomplete.

---

## 14. Final handoff

Create：

```text
docs/retargeting_v3/STEP4_1_ORIENTATION_RESIDUAL_BREAKTHROUGH_HANDOFF.md
```

Must record：

```text
branch
final_head
workflow_run_id
job conclusions
strict audit result
focused tests result
LFS fsck result
artifact dir
baseline Step 4.0 counts
Step 4.1 counts
deltas vs Step 4.0
release_candidate_status
primary_quality_breakthrough
orientation frame semantics summary
orientation residual math audit summary
orientation policy selection
candidate policy sweep summary
normalization audit result
trajectory export summary
temporal continuity summary
remaining warned rows
remaining blockers
known limitations
next direction, if any
```

Do not start the next phase automatically.

---

## 15. Final reminder to Codex

```text
This is the orientation residual breakthrough run.
Do not stop after docs or audit shell.
Do real orientation math, frame semantics audit, candidate policy sweep, artifacts, tests, CI, and handoff.
Do not tune per robot.
Do not weaken gates.
Do not overwrite closed artifacts.
Do not hide raw residual regression.
If no global policy breaks through, deliver honest BLOCKED_ORIENTATION_SEMANTICS with complete diagnostics.
```
