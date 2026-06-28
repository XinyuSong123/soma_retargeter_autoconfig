# Goal — Step 3.4：Global Residual Quality / Task Coverage Hardening（10h+ 长跑任务）

> **Codex 强制执行契约——不得自由发挥**
>
> 当前分支：`retargeting-v3-step3-4-global-residual-quality`  
> 基线分支：`retargeting-v3-step3-3-global-solver-quality-hardening`  
> 基线提交：`1a8ae37670a3f3f8c49cefd78c6bf7292aa15489`  
> 当前阶段：**Step 3.4 Global Residual Quality / Task Coverage Hardening**
>
> Step 3.3 已闭环：final CI run `28295149155` success，strict audit PASS，`runtime_quality_failed_count=0`，`solver_backed_count=32`，`residual_only_count=0`，44-row / 32-3-9 fleet accounting preserved。Step 3.4 必须从该闭环基线继续。
>
> 本轮是一个 **10 小时起步的长跑开发任务**。Step 3.3 已经移除了 hard joint-limit failure class；Step 3.4 的唯一核心问题是：**32/32 full humanoids 仍然因为 high residual 而 `runtime_quality_warned`，且 `runtime_quality_passed_count=0`**。本轮必须用全局、通用、可审计的方法改进 residual quality / task coverage / anchor reliability，不能通过 status rename、robot-specific tuning 或 gate weakening 来制造通过。
>
> **禁止修改 Step 2 artifacts / thresholds；禁止覆盖 Step 3.1.1 / Step 3.2 / Step 3.3 closed artifacts；禁止 robot-specific 数学、阈值、白名单或 per-robot IK 权重调参；禁止默认启用 production runtime override；禁止 contact / collision / temporal smoothing；禁止把 warned/failed 重命名成 passed；禁止宣称 visual quality、production retargeting 或 all humanoids solved。**

---

## 0. 开始前必须执行

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
branch = retargeting-v3-step3-4-global-residual-quality
base Step 3.3 final commit is ancestor
worktree clean
Git LFS fsck OK
snapshot XML/URDF materialized, not pointer-only
```

环境不满足时，只修环境或 checkout，不得改代码绕过。

---

## 1. 必须先阅读的文件

按顺序完整阅读：

1. `goal.md`（本文件）
2. `docs/retargeting_v3/STEP3_3_GLOBAL_SOLVER_QUALITY_HANDOFF.md`
3. `docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md`
4. `artifacts/retargeting_v3_step3_3_global_solver_quality/quality_summary.json`
5. `artifacts/retargeting_v3_step3_3_global_solver_quality/quality_delta_vs_step3_2.json`
6. `artifacts/retargeting_v3_step3_3_global_solver_quality/model_matrix.json`
7. `artifacts/retargeting_v3_step3_3_global_solver_quality/solver_smoke_matrix.json`
8. `artifacts/retargeting_v3_step3_3_global_solver_quality/generic_smoke_matrix.json`
9. `artifacts/retargeting_v3_step3_3_global_solver_quality/solver_config.json`
10. `artifacts/retargeting_v3_step3_3_global_solver_quality/solver_diagnostics_matrix.json`
11. `artifacts/retargeting_v3_step3_3_global_solver_quality/acceptance_ledger.json`
12. `soma_retargeter/runtime/v3/generic_smoke.py`
13. `soma_retargeter/runtime/v3/fleet_harness.py`
14. `soma_retargeter/runtime/v3/quality_metrics.py`
15. `soma_retargeter/runtime/v3/runtime_quality_gates.py`
16. `soma_retargeter/runtime/v3/runtime_status.py`
17. `soma_retargeter/runtime/v3/runtime_local_profile.py`
18. `soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py`
19. `scripts/audit_retargeting_v3_step3_3_global_solver_quality.py`
20. `.github/workflows/retargeting_v3_step3_3_global_solver_quality.yml`
21. `tests/v3/test_step3_3_global_solver_quality_audit.py`
22. `tests/v3/test_step3_3_global_solver_quality_artifact_schema.py`
23. `tests/v3/test_step3_3_global_solver_quality_delta.py`
24. `tests/v3/test_step3_3_global_solver_quality_status_semantics.py`
25. `tests/v3/test_step3_3_global_solver_quality_no_robot_specific_tuning.py`
26. `tests/v3/test_step3_3_global_solver_quality_determinism.py`

不要只读 summary。必须理解 Step 3.3 已经解决 hard failures，但 residual quality 没有改善。

---

## 2. 当前真实基线

Step 3.3 closed baseline：

```text
final_head = 1a8ae37670a3f3f8c49cefd78c6bf7292aa15489
GitHub Actions run = 28295149155
strict audit = PASS, blocking_count = 0
focused tests = 33 passed
git lfs fsck = OK
```

Step 3.3 artifact truth：

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

Step 3.3 delta truth：

```text
runtime_quality_failed_count_delta = -9
joint_limit_warning_count_delta = -11
joint_limit_smoke_warning_count_delta = -10
high_residual_warning_count_delta = 0
normalized_task_residual_p95_delta = 0
runtime_quality_passed_count_delta = 0
```

Interpretation：

```text
Step 3.3 removed hard joint-limit failures.
Step 3.4 must attack the remaining high residual problem globally.
```

---

## 3. 本轮唯一目标

用全局、通用、可审计的方法降低 residual warnings 或 residual severity：

```text
Reduce high_residual_warning_count below 32 OR produce a meaningful, audited improvement in residual distribution metrics, while preserving solver-backed coverage and honest status semantics.
```

Preferred stretch target：

```text
runtime_quality_passed_count > 0
```

但只有 gate 真实满足时才允许出现 pass；不能放松 pass gates，不能把 warned 改名为 passed。

完成后必须能回答：

1. Step 3.4 是否仍覆盖 44 rows 和 32/3/9 partition？
2. 32 个 full humanoids 是否仍全部 solver-backed attempted/completed？
3. `solver_backed_count=32` 和 `residual_only_count=0` 是否保持？
4. `runtime_quality_failed_count=0` 是否保持？
5. high residual warning count 是否下降？如果不下降，residual p50/p95/max distribution 是否有明确量化改善？
6. 是否出现真实 gate-valid `runtime_quality_passed` rows？
7. task coverage / anchor reliability / residual normalization 到底改善了什么？
8. 是否没有 robot-specific tuning、阈值、白名单、per-robot IK weights？
9. 是否保持 deterministic rerun、clean provenance、LFS fsck、final HEAD CI、strict audit？

---

## 4. 成功优先级

### Priority 0 — Preserve Step 3.3 invariants

必须保持：

```text
44-row matrix
32/3/9 source partition
32 solver-backed attempts
32 solver-backed completions
solver_backed_count = 32
residual_only_count = 0
runtime_quality_failed_count = 0
negative controls not promoted
partial models not counted as full humanoid passes
RPO/G1 pipeline controls retained or referenced
clean provenance
deterministic 44/44
```

### Priority 1 — Improve high residual outcome

Step 3.3 baseline：

```text
high_residual_warning_count = 32
normalized_task_residual_p95 median = unchanged from Step 3.2
normalized_task_residual_p95 p95 = unchanged from Step 3.2
runtime_quality_passed_count = 0
```

Step 3.4 的首要目标至少满足一个：

```text
high_residual_warning_count < 32
OR median_normalized_task_residual_p95_delta < 0
OR p95_normalized_task_residual_p95_delta < 0
OR max_normalized_task_residual_p95_delta < 0
```

如果这些都做不到，最终状态必须是 BLOCKED，并给出 failure taxonomy 和下一步建议。不能用 status rename 逃避。

### Priority 2 — Earn real pass rows if globally justified

允许 `runtime_quality_passed_count > 0`，但 pass row 必须满足：

```text
solver_backed=true
solver_backed_smoke_attempted=true
solver_backed_smoke_completed=true
finite metrics
output_nan_count=0
output_inf_count=0
joint_limit_violation_count within global tolerance
normalized_task_residual_p95 <= global pass gate
deterministic matched
```

### Priority 3 — Improve diagnostics if quality target is blocked

如果 residual quality 无法改善，也不能空手失败。必须产出：

```text
residual_taxonomy.json
task_coverage_matrix.json
anchor_reliability_matrix.json
per-model residual blocker reasons
attempted global improvements
why robot-specific tuning was not used
recommended next direction
```

---

## 5. 严格范围边界

### 5.1 本轮包含

- generic task coverage expansion；
- generic anchor reliability scoring / filtering；
- global residual normalization audit and correction；
- semantic task weighting only if global and not robot-specific；
- task coverage diagnostics；
- anchor availability / degeneracy diagnostics；
- residual distribution delta vs Step 3.3；
- new Step 3.4 artifact tree；
- Step 3.4 strict audit；
- focused tests and final HEAD CI closure。

### 5.2 本轮禁止

- 修改 Step 2 artifacts / thresholds；
- 覆盖 Step 3.1.1 / Step 3.2 / Step 3.3 closed artifact trees；
- 修改历史 Step 3.3 numbers 来制造 delta；
- production `NewtonPipeline` 默认行为重写；
- 默认启用 runtime override；
- per-robot IK weights；
- robot-id special cases；
- robot-specific threshold tables；
- whitelist / blacklist 某些机器人来制造 pass；
- contact / collision / temporal smoothing；
- video/visual eyeballing 作为 pass evidence；
- status rename 把 warned 改 passed；
- full repo pytest 失败时写成 passed；
- 自动开始 Step 3.5。

---

## 6. 建议技术路线

Codex 可调整实现细节，但必须保持全局通用、可审计、fail-closed。

### 6.1 Residual taxonomy first

先对 Step 3.3 的 32 high residual rows 建 taxonomy：

```text
task coverage too narrow
anchor unavailable
anchor degenerate
body name mismatch / alias issue
torso-only task insufficient
residual normalization scale issue
root pose normalization issue
rotation residual dominates
translation residual dominates
solver converged but task underconstrained
```

产物：

```text
residual_taxonomy.json
per-model residual blocker reasons
aggregate residual buckets
```

### 6.2 Generic task coverage expansion

允许从 profile/model introspection 中全局扩展 task anchors：

```text
torso / pelvis / chest
head if available
left/right hand if available
left/right foot if available
upper/lower limb anchors if reliably mapped
```

要求：

- task selection 不得按 model_id；
- body alias / semantic matching 必须是全局规则；
- unavailable anchors must be recorded, not silently ignored；
- duplicate/degenerate anchors must be filtered globally；
- task count and coverage ratio must be recorded per full humanoid。

### 6.3 Anchor reliability scoring

实现全局 reliability policy：

```text
body exists in runtime model
body has finite FK
target stream contains finite target
anchor is not duplicate
anchor displacement scale is reasonable
anchor contributes nonzero residual information
```

产物：

```text
anchor_reliability_matrix.json
anchor rejection reasons
coverage summary
```

### 6.4 Residual normalization audit and correction

检查现有 normalized residual 是否被固定为接近 1 的饱和值。允许做全局 normalization correction：

```text
normalization denominator sanity
per-task scale derived from global semantic class, not robot id
translation/rotation residual balancing
residual clipping diagnostics, not hidden clipping
```

要求：

- 不允许通过扩大 denominator 来刷低 residual；
- normalization config must be recorded in solver_config.json；
- before/after raw residual and normalized residual must both be recorded；
- audit must reject suspicious normalization that hides raw residual regressions。

### 6.5 Solver/task integration improvements

允许全局增强 solver smoke：

```text
multi-anchor task construction
global semantic task weights
root/torso canonicalization
residual non-increase guard retained
joint-limit projection retained
finite-metric rollback retained
```

要求：

- weights must be global by semantic task class；
- no robot id branch；
- no special thresholds per model；
- all config recorded and hashed。

### 6.6 Metrics delta and regression guard

必须计算 Step 3.3 → Step 3.4 deltas：

```text
high_residual_warning_count_delta
runtime_quality_passed_count_delta
runtime_quality_warned_count_delta
runtime_quality_failed_count_delta
median_normalized_task_residual_p95_delta
p95_normalized_task_residual_p95_delta
max_normalized_task_residual_p95_delta
raw_task_residual_p95_delta
task_coverage_mean_delta
task_coverage_min_delta
anchor_rejection_rate_delta
solver_backed_count_delta
residual_only_count_delta
```

如果核心 invariant regress，audit fail。

---

## 7. Artifact 策略

本轮必须创建新的 Step 3.4 artifact tree，不覆盖旧 artifacts：

```text
artifacts/retargeting_v3_step3_4_global_residual_quality/
```

至少包含：

```text
environment.json
commands.txt
model_matrix.json
solver_smoke_matrix.json
generic_smoke_matrix.json
quality_summary.json
acceptance_ledger.json
deterministic_rerun.json
quality_delta_vs_step3_3.json
residual_taxonomy.json
task_coverage_matrix.json
anchor_reliability_matrix.json
solver_config.json
solver_diagnostics_matrix.json
pipeline_controls_reference.json 或 pipeline_backed_matrix.json
test_results/pytest.txt
test_results/pytest_summary.json
test_results/junit.xml
```

`quality_summary.json` 至少记录：

```text
schema_version
base_step3_3_final_head
in_scope_total
full_humanoid_total
partial_total
negative_total
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
joint_limit_warning_count
deterministic_compared_count
deterministic_matched_count
task_coverage_mean
task_coverage_min
anchor_reliability_mean
median_normalized_task_residual_p95
p95_normalized_task_residual_p95
max_normalized_task_residual_p95
```

`quality_delta_vs_step3_3.json` 至少记录：

```text
baseline_artifact_dir
current_artifact_dir
baseline_final_head
current_source_commit
baseline_counts
current_counts
count_deltas
metric_distribution_deltas
task_coverage_deltas
anchor_reliability_deltas
raw_vs_normalized_residual_deltas
regressions
improvements
verdict
```

每个 full humanoid row 至少记录：

```text
model_id
category
source_status
runtime_quality_status
solver_backed_smoke_attempted
solver_backed_smoke_completed
solver_backed
solver_mode
solver_config_hash
task_anchor_count
task_anchor_semantic_counts
task_coverage_ratio
anchor_reliability_score
anchor_rejection_reasons
normalized_task_residual_mean / p95 / max
raw_task_residual_mean / p95 / max
target_translation_error_mean / p95 / max
target_rotation_error_mean / p95 / max
joint_limit_violation_count
max_joint_limit_violation
output_nan_count
output_inf_count
runtime_seconds
warning_reasons
failure_reasons
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

新增 Step 3.4 专用 audit，推荐：

```text
scripts/audit_retargeting_v3_step3_4_global_residual_quality.py
```

Audit 必须 fail closed：

```text
required artifacts present
44 rows present
32/3/9 partition preserved
base_step3_3_final_head == 1a8ae37670a3f3f8c49cefd78c6bf7292aa15489
clean provenance
solver_backed_smoke_attempted_count == 32
solver_backed_completed_count == 32
solver_backed_count == 32
residual_only_count == 0
runtime_quality_failed_count == 0
negative controls not promoted
partial rows not counted as full humanoid passes
runtime_quality_passed rows require solver_backed=true and all gates valid
warned rows cannot be hidden by status rename
quality_delta_vs_step3_3 exists and is internally consistent
residual_taxonomy exists
task_coverage_matrix exists
anchor_reliability_matrix exists
high_residual_warning_count < 32 OR residual distribution metric improves
normalization does not hide raw residual regression
no robot-specific threshold/whitelist/model-id special case introduced
numeric metrics finite and ordered
NaN/Inf counts zero for completed rows
deterministic rerun 44/44 matched
pipeline controls retained/referenced
final HEAD CI evidence present in strict mode
```

如果 `high_residual_warning_count == 32` 且 residual distribution 没有改善，strict PASS 不允许。默认必须 BLOCKED，并记录 why。

---

## 9. 测试要求

新增 focused tests，推荐：

```text
tests/v3/test_step3_4_global_residual_quality_audit.py
tests/v3/test_step3_4_global_residual_quality_artifact_schema.py
tests/v3/test_step3_4_global_residual_quality_delta.py
tests/v3/test_step3_4_global_residual_quality_status_semantics.py
tests/v3/test_step3_4_global_residual_quality_no_robot_specific_tuning.py
tests/v3/test_step3_4_global_residual_quality_task_coverage.py
tests/v3/test_step3_4_global_residual_quality_normalization.py
tests/v3/test_step3_4_global_residual_quality_determinism.py
```

至少测试：

- Step 3.4 audit rejects missing delta file；
- audit rejects no residual improvement with 32 high-residual warnings；
- audit rejects `runtime_quality_failed_count > 0` regression；
- audit rejects solver_backed_count regression；
- audit rejects residual_only_count > 0；
- audit rejects residual-only pass；
- audit rejects partial/negative promotion；
- audit rejects robot-specific threshold tables / whitelist strings / model-id special cases；
- audit rejects suspicious normalization hiding raw residual regression；
- audit rejects missing task coverage or anchor reliability evidence；
- audit rejects dirty provenance；
- audit rejects deterministic mismatch；
- status semantics tests prove pass rows require solver-backed + finite metrics + gates；
- delta tests prove current counts match summary and baseline counts match Step 3.3；
- solver/task config hash is stable/deterministic。

Recommended focused test command:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step3_4_global_residual_quality_audit.py \
  tests/v3/test_step3_4_global_residual_quality_artifact_schema.py \
  tests/v3/test_step3_4_global_residual_quality_delta.py \
  tests/v3/test_step3_4_global_residual_quality_status_semantics.py \
  tests/v3/test_step3_4_global_residual_quality_no_robot_specific_tuning.py \
  tests/v3/test_step3_4_global_residual_quality_task_coverage.py \
  tests/v3/test_step3_4_global_residual_quality_normalization.py \
  tests/v3/test_step3_4_global_residual_quality_determinism.py
```

---

## 10. 10h+ 执行计划

这是一份长跑计划。Codex 应持续推进到完整闭环或明确 BLOCKED，不要中途只写文档。

### Phase A — Residual forensics and taxonomy

1. Parse Step 3.3 artifacts；
2. Extract 32 warned full humanoid residual metrics；
3. Compare raw vs normalized residual；
4. Identify task coverage / anchor coverage limits；
5. Rank rows by normalized p95/max and raw residual；
6. Write residual taxonomy；
7. Do not change status labels yet。

### Phase B — Audit and tests first

1. Add Step 3.4 audit skeleton；
2. Add artifact schema tests；
3. Add residual delta/regression tests；
4. Add normalization safety tests；
5. Add no robot-specific tuning tests；
6. Ensure tests fail before implementation if artifacts missing。

### Phase C — Implement global task/anchor improvements

Choose global improvements based on taxonomy：

```text
semantic anchor expansion
anchor reliability filtering
duplicate/degenerate anchor rejection
root/torso/head/hand/foot availability handling
semantic task weighting
raw/normalized residual accounting
```

All config must be global and recorded in `solver_config.json`。

### Phase D — Wire runner and artifact generation

1. Add Step 3.4 artifact dir support；
2. Preserve Step 3.3 artifact dir；
3. Generate Step 3.4 matrices；
4. Generate quality delta vs Step 3.3；
5. Generate residual taxonomy；
6. Generate task coverage and anchor reliability matrices；
7. Generate deterministic rerun hash；
8. Generate test result artifacts。

### Phase E — Iterate until residual target or honest BLOCKED

Run multiple cycles if necessary:

```bash
PYTHONPATH=. python soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --deterministic-rerun
```

If counts regress, revert or fix.
If residual target cannot be met with global methods, stop and produce BLOCKED diagnostics.

### Phase F — CI and final evidence closure

1. Add/update workflow `.github/workflows/retargeting_v3_step3_4_global_residual_quality.yml`；
2. Run focused tests；
3. Run audit local；
4. Run LFS fsck；
5. Push；
6. Wait for final HEAD CI；
7. Run strict audit with final-head CI；
8. Write final handoff。

---

## 11. 推荐命令模板

Codex 可调整 flags，但必须同步更新 tests/audit/docs/CI。

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step3_4_global_residual_quality_audit.py \
  tests/v3/test_step3_4_global_residual_quality_artifact_schema.py \
  tests/v3/test_step3_4_global_residual_quality_delta.py \
  tests/v3/test_step3_4_global_residual_quality_status_semantics.py \
  tests/v3/test_step3_4_global_residual_quality_no_robot_specific_tuning.py \
  tests/v3/test_step3_4_global_residual_quality_task_coverage.py \
  tests/v3/test_step3_4_global_residual_quality_normalization.py \
  tests/v3/test_step3_4_global_residual_quality_determinism.py

PYTHONPATH=. python soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --deterministic-rerun

PYTHONPATH=. python scripts/audit_retargeting_v3_step3_4_global_residual_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --source-root .

PYTHONPATH=. python scripts/audit_retargeting_v3_step3_4_global_residual_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_4_global_residual_quality \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --source-root . \
  --require-final-head-ci

git lfs fsck
```

---

## 12. PASS / BLOCKED 定义

### Step 3.4 PASS 必须表示

```text
Step 3.3 baseline preserved
new Step 3.4 artifact tree generated from clean source
44-row full-fleet matrix preserved
32/3/9 partition preserved
32 full humanoid solver-backed attempts/completions preserved
solver_backed_count = 32
residual_only_count = 0
runtime_quality_failed_count = 0
high_residual_warning_count < 32 OR residual distribution metrics improved
quality delta vs Step 3.3 is recorded and internally consistent
task coverage / anchor reliability evidence exists
runtime_quality_passed rows, if any, are solver-backed and gate-valid
negative controls not promoted
partial rows not counted as full humanoid passes
deterministic rerun matched 44/44
no robot-specific tuning/threshold/whitelist introduced
normalization does not hide raw residual regression
focused tests passed
Git LFS fsck OK
final HEAD CI green
strict Step 3.4 audit passes
```

PASS 不代表：

```text
all humanoids pass runtime quality
visual motion quality is solved
production deployment is ready
contact/collision/smoothing is solved
robot-specific tuning has been performed
```

### Step 3.4 BLOCKED 必须表示

如果无法用全局方法降低 high residual warnings 或 residual distribution，或 solver coverage / determinism / CI / audit 失败，最终必须 BLOCKED，并包含：

```text
blocker taxonomy
per-model residual reasons
task coverage limits
anchor reliability limits
attempted global improvements
why robot-specific tuning was not used
recommended next direction
```

BLOCKED 也必须保留 artifacts 和 diagnostics。不要空手失败。

---

## 13. 6 个建议 subagents

继续使用专业分工；主 Codex 是 Integrator。

```text
Agent A — Residual taxonomy / baseline forensics
Agent B — Generic task coverage / anchor reliability
Agent C — Residual normalization / metric integrity
Agent D — Artifact schema / deterministic regeneration
Agent E — Tests / CI / LFS evidence
Agent F — Red-team audit / no robot-specific tuning enforcement
```

每个 agent 都必须遵守：不 robot-specific tuning、不改 Step 2 thresholds、不弱化 audit、不把 warned 改名成 pass。

---

## 14. 最终 handoff 必须写清

最终交付时新增：

```text
docs/retargeting_v3/STEP3_4_GLOBAL_RESIDUAL_QUALITY_HANDOFF.md
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
baseline Step 3.3 counts
Step 3.4 counts
deltas vs Step 3.3
solver/task config hash
residual taxonomy
task coverage summary
anchor reliability summary
remaining warned rows
known limitations
next direction
```

不要自动开始 Step 3.5。

---

## 15. 给 Codex 的最后提醒

```text
This is a 10h+ engineering run.
Do real implementation, tests, artifacts, audit, CI, and evidence closure.
Do not stop after writing docs.
Do not tune per robot.
Do not weaken gates.
Do not overwrite closed artifacts.
Attack the remaining high residual problem globally.
If residual quality cannot improve, deliver honest BLOCKED diagnostics with artifacts.
```
