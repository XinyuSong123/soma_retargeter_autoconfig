# Goal — Step 4.0：Full Pipeline Acceptance / Release-Candidate Closure（全流程长跑总目标）

> **Codex 强制执行契约——不得自由发挥，不得拆成小步只做局部**
>
> 当前分支：`retargeting-v3-step4-full-pipeline-acceptance`  
> 基线分支：`retargeting-v3-step3-4-global-residual-quality`  
> 基线提交：`77e7c02393a6678ccab40cdb847021d7d94392c9`  
> 当前阶段：**Step 4.0 Full Pipeline Acceptance / Release-Candidate Evidence Closure**
>
> Step 3.4 已闭环：final CI run `28327096821` success，strict audit PASS，focused tests 21 passed，Step 3.3 + Step 3.4 focused tests 39 passed，Git LFS fsck OK。Step 4.0 必须从该闭环基线继续。
>
> **这是一次全流程长跑总目标，不是再做一个小补丁。** 本轮要把 V3 retargeting 从“逐项 evidence milestone”推进到“端到端 release-candidate evidence closure”：从资产/clip/profile/target stream，到 runtime local profile、solver-backed full-fleet smoke、global residual/orientation quality、trajectory export、playback/continuity diagnostics、artifact/audit/CI/handoff 全部闭环。
>
> 当前真实 blocker 不是 joint-limit failure：Step 3.3 已经把 `runtime_quality_failed_count` 降为 0；Step 3.4 保持 `runtime_quality_failed_count=0`，但 `runtime_quality_passed_count=0`、`runtime_quality_warned_count=32`、`high_residual_warning_count=32`。Step 4.0 必须直面这个事实，优先解决 rotation-dominant residual / normalization saturation / task coverage semantics，而不是继续重命名 status。
>
> **禁止修改 Step 2 artifacts / thresholds；禁止覆盖 Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 closed artifacts；禁止 robot-specific 数学、阈值、白名单或 per-robot IK 权重调参；禁止默认启用 production/runtime override；禁止把 warned 改名成 passed；禁止通过放宽 gates 制造 pass；禁止声称视觉质量或真实部署完全解决，除非相应证据真实存在并由 audit 验证。**

---

## 0. 开始前必须执行

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
branch = retargeting-v3-step4-full-pipeline-acceptance
base Step 3.4 final commit is ancestor
worktree clean
Git LFS fsck OK
snapshot XML/URDF materialized, not pointer-only
```

环境不满足时，只修环境或 checkout，不得改代码绕过。

---

## 1. 必须先阅读的文件

按顺序完整阅读，不得只读 summary：

1. `goal.md`（本文件）
2. `docs/retargeting_v3/STEP3_4_GLOBAL_RESIDUAL_QUALITY_HANDOFF.md`
3. `docs/retargeting_v3/STEP3_3_GLOBAL_SOLVER_QUALITY_HANDOFF.md`
4. `docs/retargeting_v3/STEP3_2_SOLVER_BACKED_GENERIC_SMOKE_HANDOFF.md`
5. `docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md`
6. `artifacts/retargeting_v3_step3_4_global_residual_quality/quality_summary.json`
7. `artifacts/retargeting_v3_step3_4_global_residual_quality/quality_delta_vs_step3_3.json`
8. `artifacts/retargeting_v3_step3_4_global_residual_quality/residual_taxonomy.json`
9. `artifacts/retargeting_v3_step3_4_global_residual_quality/task_coverage_matrix.json`
10. `artifacts/retargeting_v3_step3_4_global_residual_quality/anchor_reliability_matrix.json`
11. `artifacts/retargeting_v3_step3_4_global_residual_quality/model_matrix.json`
12. `artifacts/retargeting_v3_step3_4_global_residual_quality/solver_smoke_matrix.json`
13. `artifacts/retargeting_v3_step3_4_global_residual_quality/generic_smoke_matrix.json`
14. `artifacts/retargeting_v3_step3_4_global_residual_quality/solver_config.json`
15. `artifacts/retargeting_v3_step3_4_global_residual_quality/solver_diagnostics_matrix.json`
16. `artifacts/retargeting_v3_step3_4_global_residual_quality/acceptance_ledger.json`
17. `soma_retargeter/runtime/v3/generic_smoke.py`
18. `soma_retargeter/runtime/v3/fleet_harness.py`
19. `soma_retargeter/runtime/v3/quality_metrics.py`
20. `soma_retargeter/runtime/v3/runtime_quality_gates.py`
21. `soma_retargeter/runtime/v3/runtime_status.py`
22. `soma_retargeter/runtime/v3/runtime_local_profile.py`
23. `soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py`
24. `scripts/audit_retargeting_v3_step3_4_global_residual_quality.py`
25. `.github/workflows/retargeting_v3_step3_4_global_residual_quality.yml`
26. `tests/v3/test_step3_4_global_residual_quality_*.py`
27. Step 2.3.1 artifacts under `artifacts/retargeting_v3_step2_capability/` for historical partition only
28. Step 3.1.1 / Step 3.2 / Step 3.3 artifact trees for immutable baselines only

不得从旧 Step 2 terminal logs 推断当前状态。不得改 closed artifact tree 来“修历史”。

---

## 2. 当前真实基线

Step 3.4 closed baseline：

```text
final_head = 77e7c02393a6678ccab40cdb847021d7d94392c9
GitHub Actions run = 28327096821
strict audit = PASS, blocking_count = 0, finding_count = 0
focused tests = 21 passed
Step 3.3 + Step 3.4 focused tests = 39 passed
git lfs fsck = OK
```

Step 3.4 artifact truth：

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
task_coverage_mean = 1.0
task_coverage_min = 1.0
anchor_reliability_mean = 0.994791666667
anchor_reliability_min = 0.833333333333
raw_residual_regression_count = 0
normalization_integrity_status = legacy_row_max_recorded_raw_guarded
```

Step 3.4 handoff interpretation：

```text
Step 3.4 improves raw residual severity and task coverage globally.
It does not reduce high_residual_warning_count below 32.
It does not produce runtime_quality_passed full-humanoid rows.
The remaining blocker is residual quality, mostly rotation-dominant, under unchanged gates.
```

Residual taxonomy from Step 3.4：

```text
residual_normalization_scale_issue = 32
rotation_residual_dominates = 31
translation_residual_dominates = 1
anchor_unavailable_or_unreliable = 1
solver_converged_but_task_underconstrained = 1
```

Interpretation：

```text
Step 4.0 must stop treating residual quality as a side metric. It is now the central release-candidate blocker.
```

---

## 3. 本轮唯一总目标

本轮不是 Step 3.5 小步。本轮是：

```text
Full V3 pipeline acceptance: produce a complete end-to-end release-candidate evidence tree that either earns real runtime_quality_passed rows under unchanged gates or honestly proves the remaining blockers with full diagnostics.
```

Step 4.0 必须同时覆盖：

1. Full-fleet artifact accounting；
2. Offline profile/capability baseline integrity；
3. Runtime local profile resolution；
4. Multi-clip target stream generation；
5. Solver-backed runtime smoke；
6. Global orientation/residual quality hardening；
7. Task coverage and anchor reliability；
8. Temporal continuity diagnostics；
9. Contact/collision/support diagnostics as diagnostic evidence only, unless validated strongly enough to gate；
10. Trajectory/export artifact generation；
11. Deterministic rerun；
12. Strict audit；
13. Git LFS and artifact sanitization；
14. Final HEAD CI evidence；
15. Final handoff with PASS/BLOCKED/RC status.

Hard non-negotiable invariants：

```text
44-row matrix preserved
32/3/9 partition preserved
32 full humanoids attempted solver-backed pipeline
32 full humanoids completed solver-backed pipeline, unless honest BLOCKED with diagnostics
solver_backed_count >= 32, no regression allowed
residual_only_count = 0
runtime_quality_failed_count = 0, no regression allowed
negative controls not promoted
partial models not counted as full-humanoid passes
deterministic 44/44 matched
clean provenance
final HEAD CI green
```

Primary quality target：

```text
runtime_quality_passed_count > 0
OR high_residual_warning_count < 32
OR normalized/raw rotation residual distribution improves enough to be accepted by the Step 4 audit.
```

Preferred target：

```text
runtime_quality_passed_count >= 4
high_residual_warning_count <= 28
rotation_dominant_residual_count < 31
```

Stretch target：

```text
runtime_quality_passed_count >= 8
high_residual_warning_count <= 24
all residual deltas non-regressive
```

Gold target（如果全局方法能做到）：

```text
runtime_quality_passed_count >= 16
runtime_quality_failed_count = 0
high_residual_warning_count <= 16
```

如果做不到 primary target，最终必须 `BLOCKED`，但不能空手失败：必须产出 complete artifact tree、taxonomy、attempted global methods、why not enough、next direction。

---

## 4. Release-candidate 状态语义

Step 4.0 可以输出以下顶层状态：

```text
PASS_RC
PASS_DIAGNOSTIC_ONLY
BLOCKED_RESIDUAL_QUALITY
BLOCKED_PIPELINE_REGRESSION
BLOCKED_CI_OR_PROVENANCE
```

### 4.1 `PASS_RC`

只有在下列条件同时满足时允许：

```text
runtime_quality_passed_count > 0
runtime_quality_failed_count = 0
high_residual_warning_count < 32 OR residual distribution accepted by audit
solver_backed_count >= 32
residual_only_count = 0
no baseline invariant regression
trajectory/export evidence generated
continuity diagnostics finite
strict final-head CI audit PASS
```

### 4.2 `PASS_DIAGNOSTIC_ONLY`

只有在用户目标允许，并且 primary quality target 有真实 metric 改善但仍没有 pass row 时允许。默认本 goal 不鼓励这个状态；如果使用，必须在 handoff 中解释 why not `PASS_RC`。

### 4.3 `BLOCKED_RESIDUAL_QUALITY`

如果 residual quality 仍无法突破，必须输出此状态，且必须保留完整 diagnostics。

### 4.4 `BLOCKED_PIPELINE_REGRESSION`

如果 solver coverage、determinism、negative/partial semantics、runtime failed count、LFS/provenance 任一 regression，必须输出此状态。

### 4.5 `BLOCKED_CI_OR_PROVENANCE`

如果 final HEAD CI / strict audit / clean provenance 未闭环，必须输出此状态。

---

## 5. 全流程范围

### 5.1 必须包含

#### A. Baseline preservation

- Step 2.3.1 44-row capability partition 不变；
- Step 3.1.1 clean evidence milestone 不变；
- Step 3.2 solver-backed coverage 不变；
- Step 3.3 failure elimination 不变；
- Step 3.4 residual/task coverage diagnostics 不变。

#### B. Full pipeline runner

新增或扩展 runner，使 Step 4.0 有独立入口，例如：

```bash
PYTHONPATH=. python soma_retargeter/tools/run_v3_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --baseline-step3-4-artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --enable-global-orientation-residual-hardening \
  --enable-full-pipeline-exports \
  --deterministic-rerun
```

如果不新增 runner，而复用 `run_v3_full_fleet_runtime_quality.py`，必须保证 flags/docs/tests/CI 清楚，不得混淆 Step 3 artifact dirs。

#### C. Multi-clip acceptance

必须从现有 target stream / per_clip artifacts 中构建 Step 4 clip suite。至少覆盖：

```text
Neutral_walk_forward_002__A057
wave_R_001__A428
body_stretch_1_004__A069
item_pick_up_standing_R_001__A410
```

每个 full humanoid 至少要记录：

```text
clip_count_attempted
clip_count_completed
per_clip_runtime_quality_status
per_clip_residual metrics
per_clip_orientation residual metrics
per_clip_joint-limit metrics
per_clip_temporal metrics
per_clip_export path
```

如果某些 clip 当前只具备 placeholder / target delta evidence，必须 fail-closed 或 mark diagnostic-only，不得把 placeholder 当 runtime quality pass。

#### D. Orientation-dominant residual hardening

Step 3.4 taxonomy 指出 `rotation_residual_dominates = 31`。Step 4.0 必须对此专门建模：

```text
global orientation task anchors
torso/chest/pelvis/head orientation tasks when available
left/right hand orientation tasks when semantically reliable
left/right foot orientation tasks when semantically reliable
global quaternion/log-map residual handling
global rotation residual normalization
global orientation/translation balance
raw orientation residual guard
normalized orientation residual audit
```

禁止：

```text
per robot orientation weights
robot name special cases
threshold exceptions
hiding raw rotation residual by denominator inflation
```

#### E. Residual normalization v2

Step 3.4 的 normalization 仍是：

```text
legacy_row_max_recorded_raw_guarded
```

Step 4.0 必须尝试引入可审计的 global normalization v2：

```text
semantic task class scales
raw residual always retained
normalized residual monotonic with raw residual within class
global scale constants recorded
normalization hash recorded
audit detects denominator inflation
```

如果 normalization v2 不能 safely improve metrics，保留 legacy 但必须记录 blocked reason。

#### F. Trajectory/export evidence

必须新增 trajectory/export artifacts，不只是 matrix summary：

```text
exports/trajectory_manifest.json
exports/per_model/<model_id>/<clip_id>/qpos.npz 或 qpos.json
exports/per_model/<model_id>/<clip_id>/target_trace.json
exports/per_model/<model_id>/<clip_id>/diagnostics.json
exports/per_model/<model_id>/<clip_id>/playback_summary.json
```

若二进制 NPZ 不合适，可用 JSON/JSONL，但必须 deterministic、sanitized、可 audit。

每个 export 必须记录：

```text
model_id
clip_id
frame_count
qpos_shape
finite_qpos
nan_count
inf_count
source_profile_hash
target_stream_hash
solver_config_hash
export_hash
```

#### G. Temporal continuity diagnostics

必须新增：

```text
temporal_continuity_matrix.json
```

记录：

```text
joint_velocity_mean / p95 / max
joint_acceleration_mean / p95 / max
root_translation_velocity_mean / p95 / max
root_rotation_velocity_mean / p95 / max
finite_velocity
finite_acceleration
temporal_jump_count
```

Temporal 不得成为伪 pass；如果 metrics 未验证，只能作为 diagnostic/warn gate。

#### H. Support/contact/collision diagnostics（diagnostic-only unless proven）

必须至少尝试构建 diagnostic placeholders with real finite data where available：

```text
support_contact_diagnostics.json
collision_proxy_diagnostics.json
```

允许实现：

```text
foot height / support height statistics
root height stability
simple body/geom AABB or sphere proxy overlap if available
self-collision proxy count if robustly computable
```

禁止：

```text
claim contact/collision solved without validated model-wide evidence
use visual inspection as pass
make contact/collision a pass gate unless audit proves semantics are reliable
```

#### I. Audit and red-team evidence

新增：

```text
scripts/audit_retargeting_v3_step4_full_pipeline_acceptance.py
```

该 audit 必须是 Step 4 专用，不能只是调用 Step 3 audit 并换名字。

#### J. Final HEAD CI

新增 workflow：

```text
.github/workflows/retargeting_v3_step4_full_pipeline_acceptance.yml
```

必须含：

```text
step4-static-and-unit
step4-artifact-audit
step4-lfs-and-snapshot-smoke
step4-export-and-trajectory-smoke
step4-pipeline-controls-reference
step4-final-head-ci-evidence
```

---

## 6. 严格禁止

不得：

```text
- 修改 Step 2 artifacts / thresholds；
- 覆盖 Step 3.1.1 / Step 3.2 / Step 3.3 / Step 3.4 closed artifact trees；
- 修改历史 Step 3.4 numbers 来制造 delta；
- production/runtime default behavior rewrite；
- default-enable runtime override；
- robot-id special cases；
- per-robot IK weights；
- robot-specific threshold tables；
- whitelist / blacklist robots or clips to manufacture pass；
- remove hard clips/models to improve metrics；
- status rename，把 warned 变 passed；
- denominator inflation hiding raw residual；
- NaN/Inf replacement without evidence；
- visual inspection as pass；
- claim all deployment problems solved；
- full repo pytest failed/timed out but report passed；
- start Step 4.1 or Step 5 automatically。
```

---

## 7. Artifact 策略

本轮必须创建新的 artifact tree：

```text
artifacts/retargeting_v3_step4_full_pipeline_acceptance/
```

至少包含：

```text
environment.json
commands.txt
model_matrix.json
clip_matrix.json
solver_smoke_matrix.json
generic_smoke_matrix.json
full_pipeline_matrix.json
quality_summary.json
acceptance_ledger.json
deterministic_rerun.json
quality_delta_vs_step3_4.json
residual_taxonomy.json
orientation_residual_taxonomy.json
task_coverage_matrix.json
anchor_reliability_matrix.json
normalization_audit.json
solver_config.json
pipeline_config.json
solver_diagnostics_matrix.json
temporal_continuity_matrix.json
support_contact_diagnostics.json
collision_proxy_diagnostics.json
trajectory_export_manifest.json
pipeline_controls_reference.json 或 pipeline_backed_matrix.json
red_team_report.json
test_results/pytest.txt
test_results/pytest_summary.json
test_results/junit.xml
```

`quality_summary.json` 至少记录：

```text
schema_version
base_step3_4_final_head
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
joint_limit_warning_count
deterministic_compared_count
deterministic_matched_count
task_coverage_mean
task_coverage_min
anchor_reliability_mean
anchor_reliability_min
median_normalized_task_residual_p95
p95_normalized_task_residual_p95
max_normalized_task_residual_p95
median_raw_task_residual_p95
p95_raw_task_residual_p95
max_raw_task_residual_p95
median_rotation_residual_p95
p95_rotation_residual_p95
max_rotation_residual_p95
trajectory_exports_count
temporal_continuity_finite_count
support_contact_diagnostic_count
collision_proxy_diagnostic_count
release_candidate_status
```

`quality_delta_vs_step3_4.json` 至少记录：

```text
baseline_artifact_dir
current_artifact_dir
baseline_final_head
current_source_commit
baseline_counts
current_counts
count_deltas
metric_distribution_deltas
orientation_residual_deltas
normalization_deltas
task_coverage_deltas
anchor_reliability_deltas
temporal_diagnostic_deltas
regressions
improvements
verdict
```

`full_pipeline_matrix.json` 每个 full humanoid row 至少记录：

```text
model_id
category
source_status
runtime_quality_status
release_candidate_row_status
solver_backed_smoke_attempted
solver_backed_smoke_completed
solver_backed
residual_only
clip_count_attempted
clip_count_completed
trajectory_export_count
task_anchor_count
task_anchor_semantic_counts
task_coverage_ratio
anchor_reliability_score
normalized_task_residual_mean / p95 / max
raw_task_residual_mean / p95 / max
translation_residual_mean / p95 / max
rotation_residual_mean / p95 / max
joint_limit_violation_count
max_joint_limit_violation
output_nan_count
output_inf_count
temporal_jump_count
velocity_p95
acceleration_p95
support_height_p95
collision_proxy_count
warning_reasons
failure_reasons
solver_config_hash
pipeline_config_hash
export_hashes
deterministic_hash_inputs
```

Artifact provenance 必须记录：

```text
git_status_short = ""
source_worktree_clean_before_run = true
source_worktree_clean_after_run = true
source_code_commit_remote_resolvable = true
source_code_commit_is_artifact_commit_ancestor = true
core_diff_after_source_commit = []
```

---

## 8. Audit 要求

新增：

```text
scripts/audit_retargeting_v3_step4_full_pipeline_acceptance.py
```

Audit 必须 fail closed：

```text
required artifacts present
44 rows present
32/3/9 partition preserved
base_step3_4_final_head == 77e7c02393a6678ccab40cdb847021d7d94392c9
clean provenance
solver_backed_smoke_attempted_count >= 32
solver_backed_completed_count >= 32
solver_backed_count >= 32
residual_only_count == 0
runtime_quality_failed_count == 0
negative controls not promoted
partial rows not counted as full humanoid passes
trajectory_export_manifest exists and matches matrix rows/clips
temporal_continuity_matrix exists and finite
support/contact diagnostics exists
collision proxy diagnostics exists
runtime_quality_passed rows require solver_backed=true and all unchanged gates valid
warned rows cannot be hidden by status rename
quality_delta_vs_step3_4 exists and is internally consistent
orientation_residual_taxonomy exists
normalization_audit exists and does not hide raw residual regression
high_residual_warning_count < 32 OR runtime_quality_passed_count > 0 OR audited residual distribution improves beyond Step 3.4
no robot-specific threshold/whitelist/model-id special case introduced
numeric metrics finite and ordered
NaN/Inf counts zero for completed rows and exports
deterministic rerun 44/44 matched
pipeline controls retained/referenced
final HEAD CI evidence present in strict mode
```

Strict PASS requires at least one primary quality breakthrough:

```text
runtime_quality_passed_count > 0
OR high_residual_warning_count < 32
OR p95_rotation_residual_p95 improves by audited threshold and raw residual does not regress
OR p95_raw_task_residual_p95 improves by audited threshold and normalized residual is explainable
```

If none of these hold，audit must return BLOCKED, not PASS.

---

## 9. 测试要求

新增 focused tests：

```text
tests/v3/test_step4_full_pipeline_acceptance_audit.py
tests/v3/test_step4_full_pipeline_acceptance_artifact_schema.py
tests/v3/test_step4_full_pipeline_acceptance_delta.py
tests/v3/test_step4_full_pipeline_acceptance_status_semantics.py
tests/v3/test_step4_full_pipeline_acceptance_no_robot_specific_tuning.py
tests/v3/test_step4_full_pipeline_acceptance_orientation_residual.py
tests/v3/test_step4_full_pipeline_acceptance_normalization.py
tests/v3/test_step4_full_pipeline_acceptance_exports.py
tests/v3/test_step4_full_pipeline_acceptance_temporal.py
tests/v3/test_step4_full_pipeline_acceptance_contact_collision_diagnostics.py
tests/v3/test_step4_full_pipeline_acceptance_determinism.py
```

至少测试：

- audit rejects missing full_pipeline_matrix；
- audit rejects missing trajectory_export_manifest；
- audit rejects missing orientation_residual_taxonomy；
- audit rejects missing normalization_audit；
- audit rejects no quality breakthrough relative to Step 3.4；
- audit rejects `runtime_quality_failed_count > 0` regression；
- audit rejects solver_backed_count regression；
- audit rejects residual_only_count > 0；
- audit rejects residual-only pass；
- audit rejects pass row without solver-backed evidence；
- audit rejects partial/negative promotion；
- audit rejects robot-specific threshold tables / whitelist strings / model-id special cases；
- audit rejects suspicious normalization hiding raw residual regression；
- audit rejects missing task coverage or anchor reliability evidence；
- audit rejects missing export finite checks；
- audit rejects missing temporal continuity finite checks；
- audit rejects dirty provenance；
- audit rejects deterministic mismatch；
- release_candidate_status semantics are enforced；
- old closed artifact trees are not modified by Step 4 generation。

Recommended focused test command：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step4_full_pipeline_acceptance_audit.py \
  tests/v3/test_step4_full_pipeline_acceptance_artifact_schema.py \
  tests/v3/test_step4_full_pipeline_acceptance_delta.py \
  tests/v3/test_step4_full_pipeline_acceptance_status_semantics.py \
  tests/v3/test_step4_full_pipeline_acceptance_no_robot_specific_tuning.py \
  tests/v3/test_step4_full_pipeline_acceptance_orientation_residual.py \
  tests/v3/test_step4_full_pipeline_acceptance_normalization.py \
  tests/v3/test_step4_full_pipeline_acceptance_exports.py \
  tests/v3/test_step4_full_pipeline_acceptance_temporal.py \
  tests/v3/test_step4_full_pipeline_acceptance_contact_collision_diagnostics.py \
  tests/v3/test_step4_full_pipeline_acceptance_determinism.py
```

---

## 10. 全流程长跑执行计划

### Phase A — Full baseline forensics

1. Parse Step 3.4 artifacts；
2. Verify Step 4 branch starts from final Step 3.4 HEAD；
3. Extract 32 full-humanoid warned rows；
4. Build orientation-dominant residual ranking；
5. Compare raw vs normalized residual；
6. Identify semantic task coverage gaps；
7. Identify anchor reliability gaps；
8. Identify whether residual is root/torso/head/hand/foot dominated；
9. Write initial `orientation_residual_taxonomy.json`；
10. Do not change status labels yet。

### Phase B — Tests and audit first

1. Add Step 4 audit skeleton；
2. Add missing-artifact tests；
3. Add no-quality-breakthrough failure test；
4. Add pass-row semantics tests；
5. Add no robot-specific tuning tests；
6. Add normalization integrity tests；
7. Add trajectory export schema tests；
8. Add temporal diagnostics tests；
9. Confirm tests fail before implementation if artifacts missing。

### Phase C — Global orientation residual hardening

Implement global orientation task/residual improvements：

```text
global semantic orientation anchors
quaternion sign canonicalization
global log-map orientation residual
global orientation residual scaling
global orientation/translation balance
global anchor reliability filter
raw orientation residual guard
normalized residual audit
```

All config must be global and recorded in `pipeline_config.json` / `solver_config.json`。

### Phase D — Full pipeline artifact generation

1. Add Step 4 artifact dir support；
2. Preserve all closed artifact dirs；
3. Generate full pipeline matrix；
4. Generate clip matrix；
5. Generate solver smoke and generic smoke matrices；
6. Generate orientation residual taxonomy；
7. Generate normalization audit；
8. Generate trajectory exports；
9. Generate temporal continuity diagnostics；
10. Generate support/contact diagnostics；
11. Generate collision proxy diagnostics；
12. Generate quality delta vs Step 3.4；
13. Generate deterministic rerun；
14. Generate acceptance ledger。

### Phase E — Iterate until release-candidate breakthrough or honest BLOCKED

Run repeated cycles, but do not violate global/no-tuning constraints。

Suggested runner command：

```bash
PYTHONPATH=. python soma_retargeter/tools/run_v3_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --baseline-step3-4-artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --enable-global-orientation-residual-hardening \
  --enable-full-pipeline-exports \
  --deterministic-rerun
```

If using existing runner：

```bash
PYTHONPATH=. python soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py \
  --artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --enable-global-orientation-residual-hardening \
  --enable-full-pipeline-exports \
  --deterministic-rerun
```

If the primary target cannot be met：

```text
Do not fake PASS.
Deliver BLOCKED_RESIDUAL_QUALITY with complete artifacts and next-step recommendations.
```

### Phase F — Final evidence closure

1. Run focused tests；
2. Run Step 4 audit local；
3. Run deterministic rerun；
4. Run Git LFS fsck；
5. Check clean worktree before artifact generation and after；
6. Commit artifacts/handoff；
7. Push；
8. Wait for final HEAD CI；
9. Run strict audit with `--require-final-head-ci`；
10. Confirm branch HEAD matches origin；
11. Write final handoff；
12. Do not start next phase。

---

## 11. 推荐命令模板

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step4_full_pipeline_acceptance_audit.py \
  tests/v3/test_step4_full_pipeline_acceptance_artifact_schema.py \
  tests/v3/test_step4_full_pipeline_acceptance_delta.py \
  tests/v3/test_step4_full_pipeline_acceptance_status_semantics.py \
  tests/v3/test_step4_full_pipeline_acceptance_no_robot_specific_tuning.py \
  tests/v3/test_step4_full_pipeline_acceptance_orientation_residual.py \
  tests/v3/test_step4_full_pipeline_acceptance_normalization.py \
  tests/v3/test_step4_full_pipeline_acceptance_exports.py \
  tests/v3/test_step4_full_pipeline_acceptance_temporal.py \
  tests/v3/test_step4_full_pipeline_acceptance_contact_collision_diagnostics.py \
  tests/v3/test_step4_full_pipeline_acceptance_determinism.py

PYTHONPATH=. python soma_retargeter/tools/run_v3_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --baseline-step3-4-artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --enable-global-orientation-residual-hardening \
  --enable-full-pipeline-exports \
  --deterministic-rerun

PYTHONPATH=. python scripts/audit_retargeting_v3_step4_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --baseline-step3-4-artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --source-root .

PYTHONPATH=. python scripts/audit_retargeting_v3_step4_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance \
  --baseline-step3-4-artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --source-root . \
  --require-final-head-ci

git lfs fsck
```

---

## 12. PASS / BLOCKED 定义

### Step 4.0 `PASS_RC` 必须表示

```text
Step 3.4 baseline preserved
new Step 4 artifact tree generated from clean source
44-row full-fleet matrix preserved
32/3/9 partition preserved
32 full humanoid solver-backed attempts/completions preserved
solver_backed_count >= 32
residual_only_count = 0
runtime_quality_failed_count = 0
runtime_quality_passed_count > 0 OR high_residual_warning_count < 32 OR audited residual/orientation distribution breakthrough
trajectory exports generated and finite
full pipeline matrix complete
temporal diagnostics finite
support/contact diagnostics present
collision proxy diagnostics present
quality delta vs Step 3.4 internally consistent
task coverage / anchor reliability evidence exists
normalization audit proves raw residual not hidden
negative controls not promoted
partial rows not counted as full humanoid passes
deterministic rerun matched 44/44
no robot-specific tuning/threshold/whitelist introduced
focused tests passed
Git LFS fsck OK
final HEAD CI green
strict Step 4 audit passes
```

`PASS_RC` 不代表：

```text
all humanoids pass runtime quality
visual motion quality is solved
all contact/collision questions are solved
deployment on hardware is safe
robot-specific tuning has been performed
```

### Step 4.0 `BLOCKED_RESIDUAL_QUALITY` 必须表示

如果 global orientation/residual work still cannot produce a breakthrough：

```text
runtime_quality_passed_count remains 0
high_residual_warning_count remains 32
no audited residual distribution breakthrough
```

Then deliver all artifacts and diagnostics honestly.

### Step 4.0 `BLOCKED_PIPELINE_REGRESSION` 必须表示

If any baseline invariant regresses：

```text
solver coverage regression
residual_only_count > 0
runtime_quality_failed_count > 0
determinism mismatch
NaN/Inf exports
negative/partial promotion
closed artifacts modified
```

### Step 4.0 `BLOCKED_CI_OR_PROVENANCE` 必须表示

If final CI / provenance / LFS / strict audit is incomplete.

---

## 13. 建议 subagents

这是总收敛任务，使用 8 个专业 subagents；主 Codex 是 Integrator。

```text
Agent A — Baseline forensics / residual taxonomy
Agent B — Global orientation residual design
Agent C — Normalization integrity / raw residual guard
Agent D — Task coverage / anchor reliability
Agent E — Trajectory export / temporal diagnostics
Agent F — Support/contact/collision diagnostics
Agent G — Artifact schema / deterministic regeneration / CI
Agent H — Red-team audit / no robot-specific tuning enforcement
```

每个 agent 都必须遵守：不 robot-specific tuning、不改 Step 2 thresholds、不弱化 gates、不把 warned 改名成 pass、不覆盖 closed artifacts。

---

## 14. 最终 handoff 必须写清

最终交付时新增：

```text
docs/retargeting_v3/STEP4_FULL_PIPELINE_ACCEPTANCE_HANDOFF.md
```

必须记录：

```text
branch
final_head
workflow_run_id
job conclusions
strict audit result
focused tests result
LFS fsck result
artifact dir
baseline Step 3.4 counts
Step 4 counts
deltas vs Step 3.4
release_candidate_status
solver_config_hash
pipeline_config_hash
orientation residual taxonomy
normalization audit result
task coverage summary
anchor reliability summary
trajectory export summary
temporal continuity summary
support/contact diagnostic summary
collision proxy diagnostic summary
remaining warned rows
remaining blockers
known limitations
next direction, if any
```

不要自动开始下一阶段。

---

## 15. 给 Codex 的最后提醒

```text
This is the full-pipeline long run. Do not stop after docs or a small audit wrapper.
Do real implementation, tests, artifacts, trajectory exports, diagnostics, audit, CI, and final evidence closure.
Do not tune per robot.
Do not weaken gates.
Do not overwrite closed artifacts.
Do not claim visual/deployment readiness without evidence.
The central blocker is residual quality, especially rotation-dominant residual.
If breakthrough is impossible under global methods, deliver honest BLOCKED artifacts and diagnostics.
```
