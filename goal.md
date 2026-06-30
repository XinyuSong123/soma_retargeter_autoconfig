# Goal — Step 4.2：Orientation Policy Runtime Scoring Integration（全局方向策略进入运行时评分）

> **Codex 强制执行契约——不得自由发挥，不得拆成小步只做局部**
>
> 当前分支：`retargeting-v3-step4-2-orientation-policy-runtime-scoring`  
> 基线分支：`retargeting-v3-step4-1-orientation-residual-breakthrough`  
> 当前阶段：**Step 4.2 Orientation Policy Runtime Scoring Integration / Gate Reconciliation**
>
> Step 4.1 已经证明一个全局 orientation semantics breakthrough：选中策略 `parent_relative_runtime_inv_target`，`p95_rotation_residual_p95` 从 Step 4.0 的 `2.90919255987585` 降到 `2.0492009223205`，delta `-0.859991637555`，且 `raw_residual_regression_count=0`、`normalization_hides_raw_residual_regression=false`。Step 4.1 因此记录 `release_candidate_status=PASS_RC`、`primary_quality_breakthrough=true`。
>
> 但 Step 4.1 仍然没有改变 runtime quality 计数：`runtime_quality_passed_count=0`、`runtime_quality_warned_count=32`、`runtime_quality_failed_count=0`、`high_residual_warning_count=32`。Step 4.1 handoff 明确指出：下一步要决定 parent-relative orientation residual policy 应该成为 runtime scoring logic，还是只保留为 audited diagnostic evidence。
>
> **本轮唯一核心目标：把 Step 4.1 已审计的全局 parent-relative orientation policy 接入 Step 4.2 runtime scoring / quality metrics / gate reconciliation，验证它能否真实降低 high residual warnings 或产生 gate-valid runtime_quality_passed rows。**
>
> **禁止修改 Step 2 artifacts / thresholds；禁止覆盖 Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 / Step 4.1 closed artifacts；禁止 robot-specific 数学、阈值、白名单或 per-robot IK 权重调参；禁止默认启用 production/runtime override；禁止把 warned 改名成 passed；禁止通过放宽 gates 制造 pass；禁止隐藏 raw residual regression；禁止把 diagnostic-only policy 当 production default；禁止声称视觉质量或真实部署完全解决，除非相应证据真实存在并由 audit 验证。**

---

## 0. 开始前必须执行

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
branch = retargeting-v3-step4-2-orientation-policy-runtime-scoring
base Step 4.1 branch is ancestor
worktree clean
Git LFS fsck OK
snapshot XML/URDF materialized, not pointer-only
```

环境不满足时，只修环境或 checkout，不得改代码绕过。

---

## 1. 必须先阅读的文件

按顺序完整阅读，不得只读 summary：

1. `goal.md`（本文件）
2. `docs/retargeting_v3/STEP4_1_ORIENTATION_RESIDUAL_BREAKTHROUGH_HANDOFF.md`
3. `docs/retargeting_v3/STEP4_FULL_PIPELINE_ACCEPTANCE_HANDOFF.md`
4. `docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md`
5. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/quality_summary.json`
6. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/acceptance_ledger.json`
7. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/orientation_delta_vs_step4_0.json`
8. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/orientation_policy_selection.json`
9. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/orientation_residual_math_audit.json`
10. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/orientation_frame_semantics_matrix.json`
11. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/orientation_offset_candidate_matrix.json`
12. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/orientation_clip_consistency_matrix.json`
13. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/release_candidate_impact_report.json`
14. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/full_pipeline_matrix.json`
15. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/clip_matrix.json`
16. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/solver_smoke_matrix.json`
17. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/generic_smoke_matrix.json`
18. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/pipeline_config.json`
19. `artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough/solver_config.json`
20. `soma_retargeter/runtime/v3/generic_smoke.py`
21. `soma_retargeter/runtime/v3/fleet_harness.py`
22. `soma_retargeter/runtime/v3/quality_metrics.py`
23. `soma_retargeter/runtime/v3/runtime_quality_gates.py`
24. `soma_retargeter/runtime/v3/runtime_status.py`
25. `soma_retargeter/tools/run_v3_full_pipeline_acceptance.py`
26. `scripts/audit_retargeting_v3_step4_1_orientation_residual_breakthrough.py`
27. `.github/workflows/retargeting_v3_step4_1_orientation_residual_breakthrough.yml`
28. `tests/v3/test_step4_1_orientation_residual_*.py`
29. Step 4.0 artifacts under `artifacts/retargeting_v3_step4_full_pipeline_acceptance/` as immutable baseline

不得从旧 Step 2 logs 或旧 Step 3 branches 推断当前状态。不得改 closed artifact tree 来“修历史”。

---

## 2. 当前真实基线

Step 4.1 artifact truth：

```text
release_candidate_status = PASS_RC
primary_quality_breakthrough = true
orientation_selected_policy = parent_relative_runtime_inv_target
```

Preserved fleet/pipeline invariants：

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
trajectory_exports_count = 128
temporal_continuity_finite_count = 128
support_contact_diagnostic_count = 128
collision_proxy_diagnostic_count = 128
```

Runtime quality counts unchanged：

```text
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
high_residual_warning_count = 32
```

Orientation breakthrough metrics：

```text
selected_policy = parent_relative_runtime_inv_target
baseline Step 4.0 p95_rotation_residual_p95 = 2.90919255987585
Step 4.1 p95_rotation_residual_p95 = 2.0492009223205
p95_rotation_residual_p95_delta = -0.859991637555
median_rotation_residual_p95_delta = -0.484373865345
max_rotation_residual_p95_delta = -0.871994720115
raw_residual_regression_count = 0
normalization_hides_raw_residual_regression = false
runtime_quality_gates_changed = false
runtime_quality_passed_count_changed_by_policy = false
production_default_changed = false
runtime_override_default_enabled = false
robot_specific_tuning_used = false
```

Interpretation：

```text
Step 4.1 proved the orientation residual comparison semantics were wrong enough to block quality interpretation.
Step 4.2 must test whether this policy can be promoted from audited diagnostic evidence into actual runtime scoring without weakening gates or production defaults.
```

---

## 3. 本轮唯一总目标

Promote the selected global orientation policy into Step 4.2 scoring path, then regenerate full-pipeline evidence and measure real runtime-quality impact。

Primary target：

```text
high_residual_warning_count < 32
OR runtime_quality_passed_count > 0
OR normalized_task_residual_p95 / orientation-integrated residual distribution improves under active scoring without raw residual regression
```

Preferred target：

```text
runtime_quality_passed_count >= 4
OR high_residual_warning_count <= 28
OR p95_orientation_integrated_residual_p95_delta <= -0.25
```

Stretch target：

```text
runtime_quality_passed_count >= 8
OR high_residual_warning_count <= 24
OR p95_orientation_integrated_residual_p95_delta <= -0.50
```

If Step 4.2 cannot improve runtime quality counts or active-scoring residual distributions, final status must be an honest blocked status with complete diagnostics：

```text
BLOCKED_SCORING_INTEGRATION
BLOCKED_GATE_RECONCILIATION
BLOCKED_NORMALIZATION_INTEGRITY
BLOCKED_PIPELINE_REGRESSION
BLOCKED_CI_OR_PROVENANCE
```

Do not claim `PASS_RC` unless a real scoring/gate breakthrough is present.

---

## 4. What must be implemented

### 4.1 Active vs diagnostic policy separation

Step 4.1 selected policy was accepted as audited orientation semantics. Step 4.2 must explicitly separate：

```text
diagnostic_orientation_policy
active_runtime_scoring_policy
production_default_policy
```

Rules：

- The selected policy may become active for Step 4.2 scoring artifacts.
- It must not silently change production/default runtime behavior.
- It must be behind an explicit Step 4.2 flag or config.
- Artifacts must record whether policy was diagnostic-only, scoring-active, or production-default.

Produce：

```text
orientation_policy_runtime_scoring_matrix.json
active_vs_diagnostic_policy_matrix.json
```

### 4.2 Runtime scoring integration

Integrate `parent_relative_runtime_inv_target` into runtime scoring / quality metrics path used by Step 4.2 artifacts：

```text
non-root orientation anchors compare in semantic parent frame
Hips root remains world-relative
q_delta = inverse(runtime) * target in selected comparison frame
shortest-arc sign canonicalization
SO3 log-map residual vector
angle residual in radians
raw orientation residual retained
normalized orientation residual derived transparently
```

Code paths to inspect/update as needed：

```text
soma_retargeter/runtime/v3/quality_metrics.py
soma_retargeter/runtime/v3/generic_smoke.py
soma_retargeter/runtime/v3/fleet_harness.py
soma_retargeter/tools/run_v3_full_pipeline_acceptance.py
```

### 4.3 Gate reconciliation

Step 4.2 must reconcile active orientation scoring with existing runtime quality gates.

Produce：

```text
gate_reconciliation_report.json
```

Must answer：

```text
Which metrics feed runtime_quality_status?
Which metrics remain diagnostic-only?
Did runtime_quality_passed_count change?
If no pass rows appear, which unchanged gate still blocks them?
Did high_residual_warning_count change?
Did normalized_task_residual_p95 change?
Did orientation-integrated residual improve?
Were any gates weakened? Must be false.
```

### 4.4 Normalization v2 candidate for active scoring

If legacy normalized residual is saturated at ~1.0, Step 4.2 may introduce an active-scoring normalization candidate only if：

```text
raw residual is preserved
normalization is global
normalization is semantic-class based, not robot-specific
normalization does not hide raw residual regression
denominator inflation is audited and rejected
```

Produce：

```text
scoring_normalization_audit.json
```

### 4.5 Runtime-quality impact measurement

Regenerate full pipeline and compare：

```text
quality_delta_vs_step4_1.json
runtime_scoring_delta_vs_step4_1.json
orientation_policy_runtime_impact_report.json
```

Must include：

```text
runtime_quality_passed_count_delta
runtime_quality_warned_count_delta
high_residual_warning_count_delta
p95_normalized_task_residual_p95_delta
p95_orientation_integrated_residual_delta
raw_residual_regression_count
gate_blocker_taxonomy
```

---

## 5. Strict invariants

All Step 4.2 work must preserve：

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
- overwrite Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 / Step 4.0 / Step 4.1 closed artifact trees；
- modify historical Step 4.1 numbers to make Step 4.2 look better；
- rewrite production/default runtime behavior；
- default-enable runtime override；
- promote diagnostic-only learned rest offsets into active policy；
- add robot-id special cases；
- add per-robot IK weights；
- add robot-specific threshold tables；
- whitelist / blacklist robots or clips；
- remove hard robots or clips from the suite；
- relax pass gates；
- change warned labels to passed without metrics satisfying unchanged gates；
- hide raw residual regression with normalization；
- claim visual/deployment readiness without evidence；
- report full repo pytest passed unless it actually exits 0；
- start Step 4.3 automatically。
```

---

## 7. Artifact strategy

Create a new artifact tree only：

```text
artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring/
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
quality_delta_vs_step4_1.json
runtime_scoring_delta_vs_step4_1.json
orientation_policy_runtime_impact_report.json
orientation_policy_runtime_scoring_matrix.json
active_vs_diagnostic_policy_matrix.json
gate_reconciliation_report.json
scoring_normalization_audit.json
orientation_integrated_residual_matrix.json
orientation_integrated_clip_consistency_matrix.json
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
base_step4_1_final_head
release_candidate_status
primary_quality_breakthrough
orientation_policy_active_for_scoring
orientation_policy_production_default_changed
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
partial_runtime_passed_count
negative_control_runtime_passed_count
high_residual_warning_count
p95_orientation_integrated_residual
median_orientation_integrated_residual
max_orientation_integrated_residual
median_normalized_task_residual_p95
p95_normalized_task_residual_p95
max_normalized_task_residual_p95
median_raw_task_residual_p95
p95_raw_task_residual_p95
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

`gate_reconciliation_report.json` must include：

```text
active_scoring_metrics
diagnostic_only_metrics
runtime_quality_gate_inputs
unchanged_gate_policy
pass_gate_thresholds_unchanged
warn_gate_thresholds_unchanged
per_gate_blocker_counts
rows_newly_passing
rows_still_warned
why_no_pass_if_zero
```

---

## 8. Audit requirements

Add a new audit：

```text
scripts/audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring.py
```

Audit must fail closed for structural issues and must distinguish honest blocked scoring from pipeline failure.

Required gates：

```text
required artifacts present
44 rows present
32/3/9 partition preserved
base_step4_1_final_head recorded
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
orientation_policy_runtime_scoring_matrix exists
active_vs_diagnostic_policy_matrix exists
gate_reconciliation_report exists
scoring_normalization_audit exists
runtime_scoring_delta_vs_step4_1 exists and is internally consistent
orientation_policy_runtime_impact_report exists
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
PASS_RC requires primary_quality_breakthrough=true and active runtime-scoring impact.
PASS_DIAGNOSTIC_ONLY is not allowed for Step 4.2; Step 4.1 already did diagnostic breakthrough.
BLOCKED_SCORING_INTEGRATION is acceptable only with complete diagnostics and blocking_count=0.
BLOCKED_GATE_RECONCILIATION is acceptable only if unchanged gates still block all rows and the blockers are identified.
BLOCKED_NORMALIZATION_INTEGRITY is acceptable only if safe normalization cannot be selected without hiding raw residual regression.
BLOCKED_PIPELINE_REGRESSION if any baseline invariant regresses.
BLOCKED_CI_OR_PROVENANCE if CI/provenance/LFS/strict audit incomplete.
```

Strict PASS_RC requires at least one：

```text
runtime_quality_passed_count > 0
high_residual_warning_count < 32
p95_orientation_integrated_residual_delta <= -0.25 AND gate reconciliation shows active scoring impact
p95_normalized_task_residual_p95_delta <= -0.05 with no raw residual regression
```

If none holds, audit must return a blocked scoring/gate status, not PASS_RC.

---

## 9. Tests required

Add focused tests：

```text
tests/v3/test_step4_2_orientation_policy_runtime_scoring_audit.py
tests/v3/test_step4_2_orientation_policy_runtime_scoring_artifact_schema.py
tests/v3/test_step4_2_orientation_policy_runtime_scoring_active_vs_diagnostic.py
tests/v3/test_step4_2_orientation_policy_runtime_scoring_gate_reconciliation.py
tests/v3/test_step4_2_orientation_policy_runtime_scoring_delta.py
tests/v3/test_step4_2_orientation_policy_runtime_scoring_status_semantics.py
tests/v3/test_step4_2_orientation_policy_runtime_scoring_no_robot_specific_tuning.py
tests/v3/test_step4_2_orientation_policy_runtime_scoring_normalization.py
tests/v3/test_step4_2_orientation_policy_runtime_scoring_exports_temporal.py
tests/v3/test_step4_2_orientation_policy_runtime_scoring_determinism.py
```

At minimum test：

```text
missing orientation_policy_runtime_scoring_matrix fails audit
missing active_vs_diagnostic_policy_matrix fails audit
missing gate_reconciliation_report fails audit
missing runtime_scoring_delta_vs_step4_1 fails audit
PASS_RC rejected without active scoring impact
PASS_DIAGNOSTIC_ONLY rejected for Step 4.2
blocked scoring status accepted only with complete diagnostics
pipeline regression rejected
runtime_quality_failed_count > 0 rejected
residual_only_count > 0 rejected
solver coverage regression rejected
pass rows without solver-backed evidence rejected
pass rows without unchanged gates rejected
negative/partial promotion rejected
robot-specific threshold/whitelist/model-id special cases rejected
normalization hiding raw residual regression rejected
production_default_changed=true rejected unless explicitly allowed by goal update
runtime_override_default_enabled=true rejected
trajectory exports remain finite
temporal diagnostics remain finite
determinism 44/44 required
closed artifact trees not modified
```

Recommended focused test command：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step4_2_orientation_policy_runtime_scoring_*.py \
  tests/v3/test_step4_1_orientation_residual_*.py \
  tests/v3/test_step4_full_pipeline_acceptance_*.py
```

---

## 10. Long-run execution plan

### Phase A — Step 4.1 policy forensics

1. Parse Step 4.1 artifacts.
2. Verify selected policy is global and non-production-default.
3. Trace where current runtime_quality_status is computed.
4. Identify why orientation residual delta did not alter runtime quality counts.
5. Build gate blocker taxonomy for all 32 warned full humanoids.
6. Do not change status labels.

### Phase B — Tests and audit first

1. Add Step 4.2 audit skeleton.
2. Add missing artifact tests.
3. Add active-vs-diagnostic policy tests.
4. Add gate reconciliation tests.
5. Add no robot-specific tuning tests.
6. Add normalization integrity tests.
7. Ensure missing artifacts fail before implementation.

### Phase C — Active scoring integration

Implement parent-relative orientation policy in the Step 4.2 scoring path under an explicit flag/config：

```text
--enable-parent-relative-orientation-runtime-scoring
```

or equivalent config recorded in `pipeline_config.json`.

The policy must be active for Step 4.2 artifacts but must not become production/default runtime behavior.

### Phase D — Gate reconciliation and regeneration

1. Generate Step 4.2 artifacts.
2. Generate gate reconciliation report.
3. Generate active-vs-diagnostic matrix.
4. Generate runtime scoring delta vs Step 4.1.
5. Generate full pipeline matrix / clip matrix / trajectory exports / temporal diagnostics.
6. Generate deterministic rerun.
7. Generate acceptance ledger.

### Phase E — Iterate until scoring breakthrough or honest BLOCKED

If active scoring does not reduce warnings or create pass rows, do not fake PASS。
Deliver：

```text
BLOCKED_SCORING_INTEGRATION
or
BLOCKED_GATE_RECONCILIATION
```

with per-gate blocker taxonomy.

### Phase F — Evidence closure

1. Run focused tests.
2. Run local Step 4.2 audit.
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
.github/workflows/retargeting_v3_step4_2_orientation_policy_runtime_scoring.yml
```

Required jobs：

```text
step4-2-static-and-unit
step4-2-artifact-audit
step4-2-active-scoring-smoke
step4-2-gate-reconciliation-smoke
step4-2-export-and-temporal-smoke
step4-2-lfs-and-snapshot-smoke
step4-2-pipeline-controls-reference
step4-2-final-head-ci-evidence
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
  tests/v3/test_step4_2_orientation_policy_runtime_scoring_*.py \
  tests/v3/test_step4_1_orientation_residual_*.py \
  tests/v3/test_step4_full_pipeline_acceptance_*.py

PYTHONPATH=. python soma_retargeter/tools/run_v3_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring \
  --baseline-step4-1-artifact-dir artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --enable-global-orientation-residual-hardening \
  --enable-orientation-frame-semantics-audit \
  --enable-parent-relative-orientation-runtime-scoring \
  --enable-full-pipeline-exports \
  --deterministic-rerun

PYTHONPATH=. python scripts/audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring.py \
  --artifact-dir artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring \
  --baseline-step4-1-artifact-dir artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough \
  --source-root .

PYTHONPATH=. python scripts/audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring.py \
  --artifact-dir artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring \
  --baseline-step4-1-artifact-dir artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough \
  --source-root . \
  --require-final-head-ci

git lfs fsck
```

---

## 13. PASS / BLOCKED definitions

### `PASS_RC`

Requires：

```text
Step 4.1 baseline preserved
new Step 4.2 artifact tree generated from clean source
44-row full-fleet matrix preserved
32/3/9 partition preserved
solver_backed_count >= 32
residual_only_count = 0
runtime_quality_failed_count = 0
orientation policy active for scoring
production default unchanged
runtime override default disabled
primary_quality_breakthrough = true
runtime_quality_passed_count > 0 OR high_residual_warning_count < 32 OR active-scoring residual distribution breakthrough
active-vs-diagnostic policy evidence complete
gate reconciliation report complete
normalization audit passes
trajectory exports finite
temporal diagnostics finite
deterministic 44/44 matched
no robot-specific tuning/threshold/whitelist introduced
focused tests passed
Git LFS fsck OK
final HEAD CI green
strict Step 4.2 audit passes
```

### `BLOCKED_SCORING_INTEGRATION`

Allowed only if：

```text
policy was integrated into Step 4.2 scoring path
structural artifacts complete
baseline invariants preserved
active scoring did not improve quality sufficiently
per-gate blockers documented
blocking_count = 0 for structural audit
```

### `BLOCKED_GATE_RECONCILIATION`

Allowed only if unchanged gates still block all rows and blocker reasons are explicit.

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
docs/retargeting_v3/STEP4_2_ORIENTATION_POLICY_RUNTIME_SCORING_HANDOFF.md
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
baseline Step 4.1 counts
Step 4.2 counts
deltas vs Step 4.1
release_candidate_status
primary_quality_breakthrough
selected orientation policy
active-vs-diagnostic policy summary
gate reconciliation summary
runtime scoring impact summary
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
This is the runtime scoring integration run.
Step 4.1 already proved a diagnostic orientation breakthrough.
Now integrate the selected global policy into active Step 4.2 scoring evidence.
Do not stop after docs or audit shell.
Do not tune per robot.
Do not weaken gates.
Do not overwrite closed artifacts.
Do not change production defaults silently.
If active scoring cannot break through, deliver honest BLOCKED_SCORING_INTEGRATION or BLOCKED_GATE_RECONCILIATION with complete diagnostics.
```
