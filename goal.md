# Goal — Step 4.3：Normalized Residual Gate Reconciliation（全局归一化残差 / Gate 一致性收敛）

> **Codex 强制执行契约——不得自由发挥，不得拆成小步只做局部**
>
> 当前分支：`retargeting-v3-step4-3-normalized-residual-gate-reconciliation`  
> 基线分支：`retargeting-v3-step4-2-orientation-policy-runtime-scoring`  
> 当前阶段：**Step 4.3 Normalized Residual Gate Reconciliation / Active Scoring Calibration**
>
> Step 4.2 已经把 Step 4.1 选中的全局 `parent_relative_runtime_inv_target` orientation policy 接入 active runtime scoring evidence。结果很好但仍未彻底过 gate：
>
> ```text
> release_candidate_status = PASS_RC
> primary_quality_breakthrough = true
> orientation_policy_active_for_scoring = true
> orientation_policy_production_default_changed = false
> runtime_override_default_enabled = false
> p95_orientation_integrated_residual_delta_vs_step4_1 = -0.959127833526
> p95_normalized_task_residual_p95_delta_vs_step4_1 = -0.006465520975
> raw_residual_regression_count = 0
> normalization_hides_raw_residual_regression = false
> denominator_inflation_detected = false
> ```
>
> 但是 runtime quality counts 仍然不变：
>
> ```text
> runtime_quality_passed_count = 0
> runtime_quality_warned_count = 32
> runtime_quality_failed_count = 0
> high_residual_warning_count = 32
> rows_newly_passing = 0
> rows_still_warned = 32
> ```
>
> Step 4.2 的 gate reconciliation 显示所有 full humanoid 仍被同一组 gate 卡住：
>
> ```text
> high_task_residual = 32
> normalized_task_residual_p95_above_pass_gate = 32
> normalized_task_residual_p95_above_warn_gate = 32
> solver_convergence_weak = 1
> solver_success_fraction_below_one = 1
> ```
>
> **本轮唯一核心目标：解释并修复 active scoring residual 明显下降但 normalized gate 仍全部卡住的矛盾。Step 4.3 必须建立可审计、全局、非 robot-specific 的 normalized residual / gate reconciliation 方案。若它能合法降低 warnings 或产生 pass rows，则给出 PASS_RC；若不能，必须给出严格的 BLOCKED_GATE_RECONCILIATION / BLOCKED_NORMALIZATION_INTEGRITY 诊断。**
>
> **禁止修改 Step 2 artifacts / thresholds；禁止覆盖 Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 / Step 4.1 / Step 4.2 closed artifacts；禁止 robot-specific 数学、阈值、白名单或 per-robot IK 权重调参；禁止默认启用 production/runtime override；禁止把 warned 改名成 passed；禁止通过放宽 gates 制造 pass；禁止隐藏 raw residual regression；禁止 denominator inflation；禁止 claim visual/deployment readiness。**

---

## 0. 开始前必须执行

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
branch = retargeting-v3-step4-3-normalized-residual-gate-reconciliation
base Step 4.2 branch is ancestor
worktree clean
Git LFS fsck OK
snapshot XML/URDF materialized, not pointer-only
```

环境不满足时，只修环境或 checkout，不得改代码绕过。

---

## 1. 必须先阅读的文件

按顺序完整阅读，不得只读 summary：

1. `goal.md`（本文件）
2. `docs/retargeting_v3/STEP4_2_ORIENTATION_POLICY_RUNTIME_SCORING_HANDOFF.md`
3. `docs/retargeting_v3/STEP4_1_ORIENTATION_RESIDUAL_BREAKTHROUGH_HANDOFF.md`
4. `docs/retargeting_v3/STEP4_FULL_PIPELINE_ACCEPTANCE_HANDOFF.md`
5. `docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md`
6. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/quality_summary.json`
7. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/acceptance_ledger.json`
8. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/gate_reconciliation_report.json`
9. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/runtime_scoring_delta_vs_step4_1.json`
10. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/orientation_policy_runtime_impact_report.json`
11. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/scoring_normalization_audit.json`
12. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/orientation_integrated_residual_matrix.json`
13. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/full_pipeline_matrix.json`
14. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/clip_matrix.json`
15. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/solver_smoke_matrix.json`
16. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/generic_smoke_matrix.json`
17. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/pipeline_config.json`
18. `artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/solver_config.json`
19. `soma_retargeter/runtime/v3/quality_metrics.py`
20. `soma_retargeter/runtime/v3/generic_smoke.py`
21. `soma_retargeter/runtime/v3/fleet_harness.py`
22. `soma_retargeter/runtime/v3/runtime_quality_gates.py`
23. `soma_retargeter/runtime/v3/runtime_status.py`
24. `soma_retargeter/tools/run_v3_full_pipeline_acceptance.py`
25. `scripts/audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring.py`
26. `.github/workflows/retargeting_v3_step4_2_orientation_policy_runtime_scoring.yml`
27. `tests/v3/test_step4_2_orientation_policy_runtime_scoring_*.py`
28. Step 4.1 artifacts under `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/` as immutable baseline
29. Step 4.0 artifacts under `artifacts/retargeting_v3_step4_full_pipeline_acceptance/` as historical baseline

不得从旧 Step 2 logs 或旧 Step 3 branches 推断当前状态。不得改 closed artifact tree 来“修历史”。

---

## 2. 当前真实基线

Step 4.2 summary truth：

```text
release_candidate_status = PASS_RC
primary_quality_breakthrough = true
active_runtime_scoring_orientation_policy = parent_relative_runtime_inv_target
diagnostic_orientation_policy = parent_relative_runtime_inv_target
production_default_orientation_policy = world_runtime_inv_target
orientation_policy_active_for_scoring = true
orientation_policy_production_default_changed = false
runtime_override_default_enabled = false
```

Preserved full-pipeline invariants：

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
partial_runtime_passed_count = 3
negative_control_runtime_passed_count = 9
trajectory_exports_count = 128
temporal_continuity_finite_count = 128
support_contact_diagnostic_count = 128
collision_proxy_diagnostic_count = 128
deterministic_compared_count = 44
deterministic_matched_count = 44
```

Step 4.2 active scoring metrics：

```text
p95_orientation_integrated_residual = 3.309012625858
median_orientation_integrated_residual = 2.873878623742
max_orientation_integrated_residual = 3.621022322952
p95_raw_task_residual_p95 = 3.3090126258582
median_raw_task_residual_p95 = 2.873878623742
max_raw_task_residual_p95 = 3.621022322952
p95_normalized_task_residual_p95 = 0.9933912463251
median_normalized_task_residual_p95 = 0.9592907932370001
max_normalized_task_residual_p95 = 0.99661104095
```

Delta vs Step 4.1：

```text
p95_orientation_integrated_residual_delta = -0.959127833526
median_orientation_integrated_residual_delta = -0.683986984223
max_orientation_integrated_residual_delta = -0.681007367545
p95_raw_task_residual_delta = -0.959127833526
median_raw_task_residual_delta = -0.683986984223
max_raw_task_residual_delta = -0.681007367545
p95_normalized_task_residual_p95_delta = -0.006465520975
median_normalized_task_residual_p95_delta = -0.038211678732
max_normalized_task_residual_p95_delta = -0.003319337192
raw_residual_regression_count = 0
normalization_hides_raw_residual_regression = false
```

Runtime quality counts still blocked：

```text
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
high_residual_warning_count = 32
rows_newly_passing = 0
rows_still_warned = 32
```

Current gate thresholds recorded by Step 4.2：

```text
normalized_task_residual_p95_pass = 0.15
normalized_task_residual_p95_warn = 0.60
normalized_task_residual_max_warn = 1.20
nan_count = 0
inf_count = 0
joint_limit_violation_severe_threshold = 1e-5
```

Interpretation：

```text
Step 4.2 made active scoring much better in raw/orientation-integrated terms, but the normalized residual p95 remains ~0.90-0.99 for every full humanoid, far above the unchanged warn gate 0.60 and pass gate 0.15. The remaining problem is no longer orientation frame semantics; it is normalized residual scale/gate reconciliation.
```

---

## 3. 本轮唯一总目标

Step 4.3 must determine whether the normalized residual gate is correctly calibrated for the active scoring residual, and if not, fix it globally without weakening the safety semantics.

Primary target：

```text
runtime_quality_passed_count > 0
OR high_residual_warning_count < 32
OR normalized_task_residual_p95 distribution crosses below warn gate for at least one full humanoid under an audited global normalization/gate scheme
OR produce BLOCKED_GATE_RECONCILIATION with exact mathematical proof why unchanged gates still block all rows despite active-scoring residual improvement
```

Preferred target：

```text
runtime_quality_passed_count >= 4
OR high_residual_warning_count <= 28
OR rows_below_warn_gate_count >= 4
OR p95_normalized_task_residual_p95_delta <= -0.10 without raw residual regression
```

Stretch target：

```text
runtime_quality_passed_count >= 8
OR high_residual_warning_count <= 24
OR rows_below_warn_gate_count >= 8
```

If Step 4.3 cannot improve runtime quality counts or normalized-gate status, final status must be one of：

```text
BLOCKED_GATE_RECONCILIATION
BLOCKED_NORMALIZATION_INTEGRITY
BLOCKED_SCORING_SCALE_AMBIGUITY
BLOCKED_PIPELINE_REGRESSION
BLOCKED_CI_OR_PROVENANCE
```

Do not claim `PASS_RC` unless a real gate/scoring breakthrough is present.

---

## 4. What must be implemented

### 4.1 Normalized residual scale audit

Step 4.3 must explicitly audit why normalized residual p95 remains near 1.0 after raw residual p95 improved by about 0.96.

Produce：

```text
normalized_residual_scale_audit.json
```

Must answer：

```text
What denominator is used for normalized_task_residual_p95?
Is it row-local, clip-local, task-local, semantic-class global, or fixed global?
Does normalized residual saturate because denominator follows max residual?
Does normalized p95 preserve ordering of raw p95?
Does normalized p95 preserve improvements over Step 4.2?
Which rows improve raw residual but remain normalized-blocked?
Which task classes dominate normalized score?
```

### 4.2 Gate semantics audit

Produce：

```text
gate_semantics_audit.json
```

Must explain：

```text
What does normalized_task_residual_p95 <= 0.15 mean physically?
What does normalized_task_residual_p95 <= 0.60 mean physically?
Were these gates designed for legacy world residual, orientation-integrated residual, or solver-output residual?
Do active parent-relative residuals need a new metric name to avoid reusing legacy thresholds incorrectly?
Are pass/warn gates safety gates or quality gates?
Which gates are hard safety gates and which are release quality gates?
```

### 4.3 Candidate normalization/gate schemes

Evaluate global candidates without weakening safety semantics：

```text
candidate_0_current_legacy_normalized_gate
candidate_1_fixed_global_body_scale_normalization
candidate_2_semantic_task_scale_normalization
candidate_3_orientation_integrated_scale_normalization
candidate_4_raw_residual_percentile_gate_diagnostic
candidate_5_two_metric_gate_raw_plus_normalized
candidate_6_release_quality_gate_v2_candidate
```

Rules：

- Candidate selection must be global.
- Candidate cannot use model_id, robot_type, clip_id, or per-row thresholds.
- Candidate cannot hide raw residual regression.
- Candidate cannot relabel warned rows as passed unless all active gates are satisfied.
- Candidate may define a new metric name, e.g. `orientation_integrated_normalized_task_residual_p95`, if legacy normalized residual semantics are incompatible.
- Candidate may propose `release_quality_v2` gates only as explicit Step 4.3 artifact semantics, not as silent replacement of Step 2/Step 3 gates.

Produce：

```text
normalization_candidate_matrix.json
gate_candidate_matrix.json
normalization_policy_selection.json
```

### 4.4 Active gate reconciliation v2

Produce：

```text
gate_reconciliation_v2_report.json
```

Must include：

```text
active_scoring_metrics
diagnostic_only_metrics
legacy_gate_inputs
candidate_gate_inputs
hard_safety_gate_inputs
release_quality_gate_inputs
pass_gate_thresholds_unchanged_for_legacy
candidate_release_gate_thresholds_if_any
rows_below_legacy_warn_gate
rows_below_candidate_warn_gate
rows_below_candidate_pass_gate
rows_newly_passing_under_candidate
rows_still_warned_under_candidate
why_no_pass_if_zero
```

### 4.5 Runtime status semantics

Step 4.3 may introduce additional explicit statuses only if audit/tests enforce them：

```text
runtime_quality_passed
runtime_quality_warned
runtime_quality_failed
runtime_quality_warned_legacy_gate
runtime_quality_warned_release_gate_v2
release_quality_candidate_passed
release_quality_candidate_warned
```

Rules：

- Existing `runtime_quality_passed` cannot be granted unless unchanged hard gates are satisfied or the audit explicitly defines and approves a new release-quality semantics.
- New statuses must not be counted as old `runtime_quality_passed_count` unless gates truly match.
- If release-quality v2 is diagnostic-only, it must not be mixed with old runtime pass counts.

### 4.6 Full pipeline regeneration

Generate a new artifact tree and preserve all previous artifacts. It must include full-pipeline exports/temporal/support/collision diagnostics, same as Step 4.2.

---

## 5. Strict invariants

All Step 4.3 work must preserve：

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
production default unchanged unless explicitly allowed by future goal update
runtime override default disabled
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
- overwrite Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 / Step 4.1 / Step 4.2 closed artifact trees；
- modify historical Step 4.2 numbers to make Step 4.3 look better；
- rewrite production/default runtime behavior；
- default-enable runtime override；
- add robot-id special cases；
- add per-robot IK weights；
- add robot-specific threshold tables；
- whitelist / blacklist robots or clips；
- remove hard robots or clips from the suite；
- relax legacy pass gates silently；
- call diagnostic-only candidate gates old runtime_quality_passed；
- hide raw residual regression with normalization；
- use denominator inflation；
- claim visual/deployment readiness without evidence；
- report full repo pytest passed unless it actually exits 0；
- start Step 4.4 automatically。
```

---

## 7. Artifact strategy

Create a new artifact tree only：

```text
artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation/
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
quality_delta_vs_step4_2.json
runtime_scoring_delta_vs_step4_2.json
normalized_residual_scale_audit.json
gate_semantics_audit.json
normalization_candidate_matrix.json
gate_candidate_matrix.json
normalization_policy_selection.json
gate_reconciliation_v2_report.json
scoring_normalization_audit.json
orientation_policy_runtime_impact_report.json
orientation_integrated_residual_matrix.json
active_vs_diagnostic_policy_matrix.json
trajectory_export_manifest.json
temporal_continuity_matrix.json
support_contact_diagnostics.json
collision_proxy_diagnostics.json
pipeline_config.json
solver_config.json
pipeline_controls_reference.json 或 pipeline_backed_matrix.json
red_team_report.json
test_results/pytest.txt
test_results/pytest_summary.json
test_results/junit.xml
```

`quality_summary.json` must include：

```text
schema_version
base_step4_2_final_head
release_candidate_status
primary_quality_breakthrough
normalization_policy_selected
gate_policy_selected
legacy_gates_unchanged
candidate_release_gates_defined
candidate_release_gates_active
production_default_changed
runtime_override_default_enabled
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
release_quality_candidate_passed_count
release_quality_candidate_warned_count
rows_below_legacy_warn_gate
rows_below_candidate_warn_gate
rows_below_candidate_pass_gate
high_residual_warning_count
p95_normalized_task_residual_p95
median_normalized_task_residual_p95
max_normalized_task_residual_p95
p95_orientation_integrated_residual
median_orientation_integrated_residual
max_orientation_integrated_residual
p95_raw_task_residual_p95
median_raw_task_residual_p95
max_raw_task_residual_p95
raw_residual_regression_count
denominator_inflation_detected
normalization_hides_raw_residual_regression
trajectory_exports_count
temporal_continuity_finite_count
support_contact_diagnostic_count
collision_proxy_diagnostic_count
deterministic_compared_count
deterministic_matched_count
```

---

## 8. Audit requirements

Add a new audit：

```text
scripts/audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation.py
```

Audit must fail closed for structural issues and distinguish honest blocked gate status from pipeline failure.

Required gates：

```text
required artifacts present
44 rows present
32/3/9 partition preserved
base_step4_2_final_head recorded
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
normalized_residual_scale_audit exists
gate_semantics_audit exists
normalization_candidate_matrix exists
gate_candidate_matrix exists
normalization_policy_selection exists
gate_reconciliation_v2_report exists
runtime_scoring_delta_vs_step4_2 exists and is internally consistent
normalization audit proves raw residual not hidden
no denominator inflation
runtime_quality_passed rows require solver_backed=true and unchanged gates valid, unless a new audited release-quality status is used separately
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
PASS_RC requires primary_quality_breakthrough=true and a real gate/scoring reconciliation impact.
PASS_DIAGNOSTIC_ONLY is not allowed unless explicitly justified as a separate release-quality candidate and not mixed with runtime_quality_passed_count.
BLOCKED_GATE_RECONCILIATION is acceptable only with complete gate math and blocker taxonomy.
BLOCKED_NORMALIZATION_INTEGRITY is acceptable only if safe normalization cannot be selected without hiding raw residual regression.
BLOCKED_SCORING_SCALE_AMBIGUITY is acceptable only if raw/orientation residuals improve but no globally valid normalization/gate interpretation can be selected.
BLOCKED_PIPELINE_REGRESSION if any baseline invariant regresses.
BLOCKED_CI_OR_PROVENANCE if CI/provenance/LFS/strict audit incomplete.
```

Strict PASS_RC requires at least one：

```text
runtime_quality_passed_count > 0 under unchanged legacy hard gates
high_residual_warning_count < 32 under unchanged legacy gates
rows_below_candidate_warn_gate > 0 with audited release-quality v2 semantics
rows_below_candidate_pass_gate > 0 with audited release-quality v2 semantics
p95_normalized_task_residual_p95_delta <= -0.10 with no raw residual regression and no denominator inflation
```

If none holds, audit must return a blocked gate/normalization status, not PASS_RC.

---

## 9. Tests required

Add focused tests：

```text
tests/v3/test_step4_3_normalized_residual_gate_reconciliation_audit.py
tests/v3/test_step4_3_normalized_residual_gate_reconciliation_artifact_schema.py
tests/v3/test_step4_3_normalized_residual_gate_reconciliation_scale_audit.py
tests/v3/test_step4_3_normalized_residual_gate_reconciliation_gate_semantics.py
tests/v3/test_step4_3_normalized_residual_gate_reconciliation_candidate_policies.py
tests/v3/test_step4_3_normalized_residual_gate_reconciliation_status_semantics.py
tests/v3/test_step4_3_normalized_residual_gate_reconciliation_no_robot_specific_tuning.py
tests/v3/test_step4_3_normalized_residual_gate_reconciliation_normalization_integrity.py
tests/v3/test_step4_3_normalized_residual_gate_reconciliation_exports_temporal.py
tests/v3/test_step4_3_normalized_residual_gate_reconciliation_determinism.py
```

At minimum test：

```text
missing normalized_residual_scale_audit fails audit
missing gate_semantics_audit fails audit
missing normalization_candidate_matrix fails audit
missing gate_candidate_matrix fails audit
missing gate_reconciliation_v2_report fails audit
PASS_RC rejected without gate/scoring reconciliation impact
blocked gate status accepted only with complete diagnostics
pipeline regression rejected
runtime_quality_failed_count > 0 rejected
residual_only_count > 0 rejected
solver coverage regression rejected
pass rows without solver-backed evidence rejected
legacy runtime_quality_passed cannot use diagnostic-only release gates
negative/partial promotion rejected
robot-specific threshold/whitelist/model-id special cases rejected
normalization hiding raw residual regression rejected
denominator inflation rejected
production_default_changed=true rejected unless explicitly allowed by future goal update
runtime_override_default_enabled=true rejected
trajectory exports remain finite
temporal diagnostics remain finite
determinism 44/44 required
closed artifact trees not modified
```

Recommended focused test command：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step4_3_normalized_residual_gate_reconciliation_*.py \
  tests/v3/test_step4_2_orientation_policy_runtime_scoring_*.py \
  tests/v3/test_step4_1_orientation_residual_*.py
```

---

## 10. Long-run execution plan

### Phase A — Gate blocker forensics

1. Parse Step 4.2 artifacts.
2. Rank all 32 full humanoids by normalized p95 and raw/orientation p95.
3. Explain why each remains above warn gate 0.60.
4. Identify rows closest to warn gate and pass gate.
5. Build normalized residual scale audit.
6. Build gate semantics audit.
7. Do not change status labels yet.

### Phase B — Tests and audit first

1. Add Step 4.3 audit skeleton.
2. Add missing-artifact tests.
3. Add normalization integrity tests.
4. Add gate semantics tests.
5. Add candidate policy tests.
6. Add no robot-specific tuning tests.
7. Ensure missing artifacts fail before implementation.

### Phase C — Candidate normalization/gate sweep

Evaluate all global candidates and record results.

Selection rules：

```text
Candidate must be global.
Candidate must not use model_id or clip_id.
Candidate must not hide raw residual regression.
Candidate must not silently replace legacy runtime gates.
Candidate must either improve legacy normalized residual meaningfully or define a separate release-quality v2 status with clear audit semantics.
```

### Phase D — Regenerate full pipeline artifacts

Generate under：

```text
artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation/
```

### Phase E — Evidence closure

1. Run focused tests.
2. Run local Step 4.3 audit.
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
.github/workflows/retargeting_v3_step4_3_normalized_residual_gate_reconciliation.yml
```

Required jobs：

```text
step4-3-static-and-unit
step4-3-artifact-audit
step4-3-gate-semantics-smoke
step4-3-normalization-integrity-smoke
step4-3-export-and-temporal-smoke
step4-3-lfs-and-snapshot-smoke
step4-3-pipeline-controls-reference
step4-3-final-head-ci-evidence
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
  tests/v3/test_step4_3_normalized_residual_gate_reconciliation_*.py \
  tests/v3/test_step4_2_orientation_policy_runtime_scoring_*.py \
  tests/v3/test_step4_1_orientation_residual_*.py

PYTHONPATH=. python soma_retargeter/tools/run_v3_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation \
  --baseline-step4-2-artifact-dir artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --enable-global-orientation-residual-hardening \
  --enable-parent-relative-orientation-runtime-scoring \
  --enable-normalized-residual-gate-reconciliation \
  --enable-full-pipeline-exports \
  --deterministic-rerun

PYTHONPATH=. python scripts/audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation.py \
  --artifact-dir artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation \
  --baseline-step4-2-artifact-dir artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring \
  --source-root .

PYTHONPATH=. python scripts/audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation.py \
  --artifact-dir artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation \
  --baseline-step4-2-artifact-dir artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring \
  --source-root . \
  --require-final-head-ci

git lfs fsck
```

---

## 13. PASS / BLOCKED definitions

### `PASS_RC`

Requires：

```text
Step 4.2 baseline preserved
new Step 4.3 artifact tree generated from clean source
44-row full-fleet matrix preserved
32/3/9 partition preserved
solver_backed_count >= 32
residual_only_count = 0
runtime_quality_failed_count = 0
normalization/gate reconciliation evidence complete
primary_quality_breakthrough = true
runtime_quality_passed_count > 0 OR high_residual_warning_count < 32 OR accepted release-quality v2 gate breakthrough
legacy gates not silently weakened
candidate release-quality status kept separate unless explicitly accepted by audit
normalization audit passes
trajectory exports finite
temporal diagnostics finite
deterministic 44/44 matched
no robot-specific tuning/threshold/whitelist introduced
focused tests passed
Git LFS fsck OK
final HEAD CI green
strict Step 4.3 audit passes
```

### `BLOCKED_GATE_RECONCILIATION`

Allowed only if：

```text
active scoring residuals improved but unchanged gates still block all rows
complete gate semantics audit exists
complete per-gate blocker taxonomy exists
no safe candidate gate policy is accepted
blocking_count = 0 for structural audit
```

### `BLOCKED_NORMALIZATION_INTEGRITY`

Allowed only if no safe normalization can be selected without raw residual regression or denominator inflation.

### `BLOCKED_SCORING_SCALE_AMBIGUITY`

Allowed only if raw/orientation residuals are improved but no globally valid normalized scale interpretation can be selected.

### `BLOCKED_PIPELINE_REGRESSION`

Any baseline invariant regression.

### `BLOCKED_CI_OR_PROVENANCE`

Final CI/provenance/LFS/audit incomplete.

---

## 14. Final handoff

Create：

```text
docs/retargeting_v3/STEP4_3_NORMALIZED_RESIDUAL_GATE_RECONCILIATION_HANDOFF.md
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
baseline Step 4.2 counts
Step 4.3 counts
deltas vs Step 4.2
release_candidate_status
primary_quality_breakthrough
normalization policy selected
gate policy selected
legacy gate impact
candidate release-quality impact
per-gate blocker taxonomy
normalization integrity summary
rows below warn/pass gates
remaining warned rows
remaining blockers
known limitations
next direction, if any
```

Do not start the next phase automatically.

---

## 15. Final reminder to Codex

```text
This is the normalized residual/gate reconciliation run.
Step 4.2 already made active scoring residuals much better, but all rows remain warned because normalized gates still block them.
Do not stop after docs or audit shell.
Do not tune per robot.
Do not weaken gates silently.
Do not overwrite closed artifacts.
Do not hide raw residual regression.
If gate reconciliation cannot break through, deliver honest BLOCKED_GATE_RECONCILIATION or BLOCKED_NORMALIZATION_INTEGRITY with complete diagnostics.
```
