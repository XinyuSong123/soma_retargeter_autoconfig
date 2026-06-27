# Goal — Step 3.3：Global Solver Quality Hardening（10h+ 长跑任务）

> **Codex 强制执行契约——不得自由发挥**
>
> 当前分支：`retargeting-v3-step3-3-global-solver-quality-hardening`  
> 基线分支：`retargeting-v3-step3-2-solver-backed-generic-smoke`  
> 基线提交：`6ae0bfbc1153e3aba1291f38f0c82dfac6c2fa57`  
> 当前阶段：**Step 3.3 Global Solver Quality Hardening / Residual and Joint-Limit Reduction**
>
> Step 3.2 已闭环：44-row full fleet preserved，32/32 full humanoids 有 solver-backed smoke attempts/completions，`solver_backed_count=32`，`residual_only_count=0`，final CI run `28288601910` success，strict audit PASS。Step 3.3 必须从这个闭环基线继续。
>
> 本轮是一个 **10 小时起步的长跑开发任务**。目标不是“改个 label 通过”，而是系统性提升 generic solver-backed smoke 的全局质量：降低 residual / joint-limit / convergence 问题，保留所有 evidence honesty、audit、CI、determinism、fail-closed 语义。
>
> **禁止修改 Step 2 artifacts / thresholds；禁止覆盖 Step 3.1.1 或 Step 3.2 closed artifacts；禁止 robot-specific 数学、阈值、白名单或 per-robot IK 权重调参；禁止默认启用 production runtime override；禁止 contact / collision / temporal smoothing；禁止把失败/警告重命名成通过；禁止宣称视觉质量、production retargeting 或所有 humanoid quality 已解决。**

---

## 0. 开始前必须执行

```bash
conda activate soma-retargeter-v2
cd ~/Desktop/soma_retargeter_autoconfig

git fetch origin --prune
git switch retargeting-v3-step3-3-global-solver-quality-hardening || \
  git switch --track -c retargeting-v3-step3-3-global-solver-quality-hardening origin/retargeting-v3-step3-3-global-solver-quality-hardening

git pull --ff-only

git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor 6ae0bfbc1153e3aba1291f38f0c82dfac6c2fa57 HEAD
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
branch = retargeting-v3-step3-3-global-solver-quality-hardening
base Step 3.2 final commit is ancestor
worktree clean
Git LFS fsck OK
snapshot XML/URDF materialized, not pointer-only
```

环境不满足时，只修环境或 checkout，不得改代码绕过。

---

## 1. 必须先阅读的文件

按顺序完整阅读：

1. `goal.md`（本文件）
2. `docs/retargeting_v3/STEP3_2_SOLVER_BACKED_GENERIC_SMOKE_HANDOFF.md`
3. `docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md`
4. `artifacts/retargeting_v3_step3_2_solver_backed_smoke/quality_summary.json`
5. `artifacts/retargeting_v3_step3_2_solver_backed_smoke/model_matrix.json`
6. `artifacts/retargeting_v3_step3_2_solver_backed_smoke/solver_smoke_matrix.json`
7. `artifacts/retargeting_v3_step3_2_solver_backed_smoke/generic_smoke_matrix.json`
8. `artifacts/retargeting_v3_step3_2_solver_backed_smoke/acceptance_ledger.json`
9. `artifacts/retargeting_v3_step3_2_solver_backed_smoke/deterministic_rerun.json`
10. `soma_retargeter/runtime/v3/generic_smoke.py`
11. `soma_retargeter/runtime/v3/fleet_harness.py`
12. `soma_retargeter/runtime/v3/quality_metrics.py`
13. `soma_retargeter/runtime/v3/runtime_quality_gates.py`
14. `soma_retargeter/runtime/v3/runtime_status.py`
15. `soma_retargeter/runtime/v3/runtime_local_profile.py`
16. `soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py`
17. `scripts/audit_retargeting_v3_step3_2_solver_backed_smoke.py`
18. `.github/workflows/retargeting_v3_step3_2_solver_backed_smoke.yml`
19. `tests/v3/test_step3_2_solver_backed_smoke_status_semantics.py`
20. `tests/v3/test_step3_2_solver_backed_smoke_artifact_schema.py`
21. `tests/v3/test_step3_2_solver_backed_smoke_determinism.py`
22. `tests/v3/test_step3_2_solver_backed_smoke_audit.py`

不要只读 summary。必须理解 Step 3.2 的 solver-backed smoke 是“有证据但质量仍差”，不是“所有动作质量已好”。

---

## 2. 当前真实基线

Step 3.2 closed baseline：

```text
final_head = 6ae0bfbc1153e3aba1291f38f0c82dfac6c2fa57
GitHub Actions run = 28288601910
strict audit = PASS, blocking_count = 0
focused tests = 25 passed, 8 warnings
git lfs fsck = OK
```

Step 3.2 artifact truth：

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
runtime_quality_warned_count = 23
runtime_quality_failed_count = 9
partial_runtime_passed_count = 3
negative_control_runtime_passed_count = 9
```

Interpretation：

```text
Step 3.2 proved generic solver-backed smoke coverage.
Step 3.3 must improve global solver quality while preserving honest failure/warning labels.
```

---

## 3. 本轮唯一目标

用全局、通用、可审计的方法提升 solver-backed smoke quality：

```text
Reduce full-humanoid runtime_quality_failed_count and/or warning severity without robot-specific tuning, while keeping solver_backed_count=32 and residual_only_count=0.
```

完成后必须能回答：

1. Step 3.3 是否仍覆盖全部 44 rows 和 32/3/9 partition？
2. 32 个 full humanoid 是否仍全部 attempted/completed solver-backed smoke？
3. `solver_backed_count` 是否仍为 32，`residual_only_count` 是否仍为 0？
4. 哪些全局 solver improvements 降低了 residual、joint-limit、convergence 或 finite-metric问题？
5. `runtime_quality_failed_count` 是否比 Step 3.2 的 9 更低？如果没有，是否有清楚 blockers？
6. `runtime_quality_warned_count` 是否下降，或 warning severity 是否有量化改善？
7. 是否没有 robot-specific threshold / whitelist / math？
8. 是否没有修改 closed Step 3.1.1 / Step 3.2 artifacts 来刷结果？
9. 是否保持 deterministic rerun、final HEAD CI、strict audit？

---

## 4. 成功优先级

按优先级推进，不得跳级宣称胜利。

### Priority 0 — Preserve baseline invariants

必须保持：

```text
44-row matrix
32/3/9 source partition
32 solver-backed attempts
32 solver-backed completions
solver_backed_count = 32
residual_only_count = 0
negative controls not promoted
partial models not counted as full humanoid passes
RPO/G1 pipeline controls retained or referenced
clean provenance
deterministic 44/44
```

### Priority 1 — Reduce hard failures

Step 3.2 baseline：

```text
runtime_quality_failed_count = 9
```

Step 3.3 的首要质量目标：

```text
runtime_quality_failed_count < 9
```

如果做不到，最终状态必须诚实写 BLOCKED 或 PASS_WITH_NO_FAILURE_REDUCTION_NOT_ALLOWED；不要用 status rename 逃避。

### Priority 2 — Reduce warning severity

Step 3.2 baseline：

```text
runtime_quality_warned_count = 23
high_residual_warning_count = 32
joint_limit_warning_count = 12
```

目标：

```text
reduce high_residual_warning_count and/or joint_limit_warning_count
reduce normalized_task_residual_p95 aggregate statistics
reduce max_joint_limit_violation aggregate statistics
```

如果 count 不下降，必须证明 severity distribution 下降，并在 artifact 中记录 before/after deltas。

### Priority 3 — Earn real runtime_quality_passed rows if globally justified

允许出现 `runtime_quality_passed_count > 0`，但只能在所有 gate 成立时出现：

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

不能为了获得 passed row 放松 gates。

---

## 5. 严格范围边界

### 5.1 本轮包含

- global solver initialization improvements；
- global damping / trust-region / line-search / clipping policy，只能全局；
- generic joint-limit projection / repair，只能全局；
- generic task scaling / residual normalization improvements，只能全局；
- solver convergence diagnostics；
- before/after Step 3.2 vs Step 3.3 metrics delta；
- new Step 3.3 artifact tree；
- Step 3.3 strict audit；
- long-running deterministic reruns and quality sweeps；
- focused tests and final HEAD CI。

### 5.2 本轮禁止

- 修改 Step 2 artifacts / thresholds；
- 覆盖 Step 3.1.1 或 Step 3.2 closed artifact tree；
- 修改 Step 3.2 numbers 作为“历史修补”；
- production `NewtonPipeline` 默认行为重写；
- 默认启用 runtime override；
- per-robot IK weights；
- robot-id special cases；
- robot-specific threshold tables；
- whitelist / blacklist 某些机器人来制造 pass；
- contact / collision / temporal smoothing；
- video/visual eyeballing 作为 pass evidence；
- status rename 把 failed/warned 变 passed；
- full repo pytest 失败时写成 passed；
- 自动开始 Step 3.4。

---

## 6. 建议技术路线

Codex 可调整实现细节，但必须保持全局通用、可审计、fail-closed。

### 6.1 Global solver initialization

实现或改进：

```text
neutral/rest pose seeding
previous deterministic frame seed reuse within sampled sequence
profile-provided nominal pose when available
joint mid-range fallback
root/body pose canonicalization
```

要求：

- 不按 model_id 分支；
- seed policy 作为全局算法记录在 artifacts；
- seed failure 进入 diagnostics。

### 6.2 Global trust-region / damping / line-search

实现或改进 solver update policy：

```text
global damping schedule
global max update norm
global line-search acceptance rule
global finite-metric rollback
global residual non-increase guard
```

要求：

- 参数必须是全局常量或 artifact-recorded config；
- 不允许按机器人调 damping；
- 每个 full humanoid row 记录 solver iterations、rollback count、line-search count、convergence status。

### 6.3 Generic joint-limit projection / repair

实现或改进：

```text
joint range parsing
joint mid-point prior
post-step joint clipping/projection
severe violation fail-closed
violation diagnostics by count/max/p95
```

要求：

- 不允许机器人特例；
- 不允许把 clipping 后 residual 变差隐藏；
- artifacts 必须记录 pre/post joint-limit metrics。

### 6.4 Global task coverage / anchor reliability

改进：

```text
filter invalid anchors
rank anchors by model/profile availability
avoid duplicate or degenerate task anchors
record task count and task coverage
```

要求：

- 不按 robot id；
- anchor selection policy 可审计；
- 如果 task coverage 太低，fail/warn with reason。

### 6.5 Metrics delta and regression guard

必须计算 Step 3.2 → Step 3.3 deltas：

```text
runtime_quality_failed_count_delta
runtime_quality_warned_count_delta
high_residual_warning_count_delta
joint_limit_warning_count_delta
median_normalized_task_residual_p95_delta
p95_normalized_task_residual_p95_delta
max_joint_limit_violation_delta
solver_backed_count_delta
residual_only_count_delta
```

如果核心 invariant regress，audit fail。

---

## 7. Artifact 策略

本轮必须创建新的 Step 3.3 artifact tree，不覆盖旧 artifacts：

```text
artifacts/retargeting_v3_step3_3_global_solver_quality/
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
quality_delta_vs_step3_2.json
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
base_step3_2_final_head
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
```

`quality_delta_vs_step3_2.json` 至少记录：

```text
baseline_artifact_dir
current_artifact_dir
baseline_final_head
current_source_commit
baseline_counts
current_counts
count_deltas
metric_distribution_deltas
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
solver_iteration_count_mean / p95 / max
solver_converged_frame_count
solver_failed_frame_count
line_search_count
rollback_count
frame_count
sampled_frame_indices
normalized_task_residual_mean / p95 / max
target_translation_error_mean / p95 / max
target_rotation_error_mean / p95 / max
joint_limit_violation_count
max_joint_limit_violation
pre_projection_joint_limit_violation_count
post_projection_joint_limit_violation_count
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

新增 Step 3.3 专用 audit，推荐：

```text
scripts/audit_retargeting_v3_step3_3_global_solver_quality.py
```

Audit 必须 fail closed：

```text
required artifacts present
44 rows present
32/3/9 partition preserved
base_step3_2_final_head == 6ae0bfbc1153e3aba1291f38f0c82dfac6c2fa57
clean provenance
solver_backed_smoke_attempted_count == 32
solver_backed_completed_count == 32
solver_backed_count == 32
residual_only_count == 0
negative controls not promoted
partial rows not counted as full humanoid passes
runtime_quality_passed rows require solver_backed=true and all gates valid
failed/warned rows cannot be hidden by status rename
quality_delta_vs_step3_2 exists and is internally consistent
runtime_quality_failed_count < 9 OR final status BLOCKED
no robot-specific threshold/whitelist table introduced
numeric metrics finite and ordered
NaN/Inf counts zero for completed rows
deterministic rerun 44/44 matched
pipeline controls retained/referenced
final HEAD CI evidence present in strict mode
```

如果 `runtime_quality_failed_count >= 9`，strict PASS 不允许，除非 goal.md 被用户明确更新。默认必须 BLOCKED，并记录 why。

---

## 9. 测试要求

新增 focused tests，推荐：

```text
tests/v3/test_step3_3_global_solver_quality_audit.py
tests/v3/test_step3_3_global_solver_quality_artifact_schema.py
tests/v3/test_step3_3_global_solver_quality_delta.py
tests/v3/test_step3_3_global_solver_quality_status_semantics.py
tests/v3/test_step3_3_global_solver_quality_no_robot_specific_tuning.py
tests/v3/test_step3_3_global_solver_quality_determinism.py
```

至少测试：

- Step 3.3 audit rejects missing delta file；
- audit rejects `runtime_quality_failed_count >= 9` for PASS；
- audit rejects solver_backed_count regression；
- audit rejects residual_only_count > 0；
- audit rejects residual-only pass；
- audit rejects partial/negative promotion；
- audit rejects robot-specific threshold tables / whitelist strings / model-id special cases；
- audit rejects dirty provenance；
- audit rejects deterministic mismatch；
- status semantics tests prove pass rows require solver-backed + finite metrics + gates；
- delta tests prove current counts match summary and baseline counts match Step 3.2；
- solver config hash is stable/deterministic。

Recommended focused test command:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step3_3_global_solver_quality_audit.py \
  tests/v3/test_step3_3_global_solver_quality_artifact_schema.py \
  tests/v3/test_step3_3_global_solver_quality_delta.py \
  tests/v3/test_step3_3_global_solver_quality_status_semantics.py \
  tests/v3/test_step3_3_global_solver_quality_no_robot_specific_tuning.py \
  tests/v3/test_step3_3_global_solver_quality_determinism.py
```

---

## 10. 10h+ 执行计划

这是一份长跑计划。Codex 应该持续推进到完整闭环或明确 BLOCKED，不要中途因为 scope 大而只写文档。

### Phase A — Baseline forensics and failure taxonomy

1. Parse Step 3.2 artifacts；
2. Rank 9 failed rows by failure reasons；
3. Rank 23 warned rows by residual / joint-limit severity；
4. Build failure taxonomy：

```text
source/profile issue
target stream issue
solver convergence issue
joint-limit issue
residual scale issue
anchor/task coverage issue
finite metric issue
determinism issue
```

5. Write temporary notes into implementation docs or artifact diagnostics；
6. Do not change status labels yet。

### Phase B — Audit and tests first

1. Add Step 3.3 audit skeleton；
2. Add artifact schema tests；
3. Add delta/regression tests；
4. Add no robot-specific tuning tests；
5. Ensure tests fail before implementation if artifacts missing。

### Phase C — Implement global solver improvements

Choose global improvements based on taxonomy：

```text
initialization
trust-region / damping / line-search
joint-limit projection
anchor filtering / coverage
residual normalization consistency
failure diagnostics
```

All config must be global and recorded in `solver_config.json`。

### Phase D — Wire runner and artifact generation

1. Add Step 3.3 artifact dir support；
2. Preserve Step 3.2 artifact dir；
3. Generate Step 3.3 matrices；
4. Generate quality delta vs Step 3.2；
5. Generate solver diagnostics matrix；
6. Generate deterministic rerun hash；
7. Generate test result artifacts。

### Phase E — Iterate until quality target or honest BLOCKED

Run multiple cycles if necessary:

```bash
PYTHONPATH=. python soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_2_solver_backed_smoke \
  --deterministic-rerun
```

If counts regress, revert or fix.
If target cannot be met with global methods, stop and produce BLOCKED diagnostics.

### Phase F — CI and final evidence closure

1. Add/update workflow `.github/workflows/retargeting_v3_step3_3_global_solver_quality.yml`；
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
  tests/v3/test_step3_3_global_solver_quality_audit.py \
  tests/v3/test_step3_3_global_solver_quality_artifact_schema.py \
  tests/v3/test_step3_3_global_solver_quality_delta.py \
  tests/v3/test_step3_3_global_solver_quality_status_semantics.py \
  tests/v3/test_step3_3_global_solver_quality_no_robot_specific_tuning.py \
  tests/v3/test_step3_3_global_solver_quality_determinism.py

PYTHONPATH=. python soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_2_solver_backed_smoke \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --deterministic-rerun

PYTHONPATH=. python scripts/audit_retargeting_v3_step3_3_global_solver_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_2_solver_backed_smoke \
  --source-root .

PYTHONPATH=. python scripts/audit_retargeting_v3_step3_3_global_solver_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_3_global_solver_quality \
  --baseline-artifact-dir artifacts/retargeting_v3_step3_2_solver_backed_smoke \
  --source-root . \
  --require-final-head-ci

git lfs fsck
```

---

## 12. PASS / BLOCKED 定义

### Step 3.3 PASS 必须表示

```text
Step 3.2 baseline preserved
new Step 3.3 artifact tree generated from clean source
44-row full-fleet matrix preserved
32/3/9 partition preserved
32 full humanoid solver-backed attempts/completions preserved
solver_backed_count = 32
residual_only_count = 0
runtime_quality_failed_count < 9
quality delta vs Step 3.2 is recorded and internally consistent
runtime_quality_passed rows, if any, are solver-backed and gate-valid
negative controls not promoted
partial rows not counted as full humanoid passes
deterministic rerun matched 44/44
no robot-specific tuning/threshold/whitelist introduced
focused tests passed
Git LFS fsck OK
final HEAD CI green
strict Step 3.3 audit passes
```

PASS 不代表：

```text
all humanoids pass runtime quality
visual motion quality is solved
production deployment is ready
contact/collision/smoothing is solved
robot-specific tuning has been performed
```

### Step 3.3 BLOCKED 必须表示

如果无法用全局方法把 `runtime_quality_failed_count` 降到 9 以下，或 solver coverage / determinism / CI / audit 失败，最终必须 BLOCKED，并包含：

```text
blocker taxonomy
per-model failure reasons
attempted global improvements
why robot-specific tuning was not used
recommended next direction
```

BLOCKED 也必须保留 artifacts 和 diagnostics。不要空手失败。

---

## 13. 6 个建议 subagents

继续使用专业分工；主 Codex 是 Integrator。

```text
Agent A — Failure taxonomy / baseline forensics
Agent B — Global solver initialization and trust-region design
Agent C — Joint-limit projection / metric delta design
Agent D — Artifact schema / deterministic regeneration
Agent E — Tests / CI / LFS evidence
Agent F — Red-team audit / no robot-specific tuning enforcement
```

每个 agent 都必须遵守：不 robot-specific tuning、不改 Step 2 thresholds、不弱化 audit、不把 failed/warned 改名成 pass。

---

## 14. 最终 handoff 必须写清

最终交付时新增：

```text
docs/retargeting_v3/STEP3_3_GLOBAL_SOLVER_QUALITY_HANDOFF.md
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
baseline Step 3.2 counts
Step 3.3 counts
deltas vs Step 3.2
solver config hash
failure taxonomy
remaining failed/warned rows
known limitations
next direction
```

不要自动开始 Step 3.4。

---

## 15. 给 Codex 的最后提醒

```text
This is a 10h+ engineering run.
Do real implementation, tests, artifacts, audit, CI, and evidence closure.
Do not stop after writing docs.
Do not tune per robot.
Do not weaken gates.
Do not overwrite closed artifacts.
If global quality target cannot be met, deliver honest BLOCKED diagnostics with artifacts.
```
