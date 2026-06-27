# Goal — Step 3.2：Solver-Backed Generic Runtime Smoke（通用 solver-backed 运行时 smoke / 质量改进）

> **Codex 强制执行契约——不得自由发挥**
>
> 当前分支：`retargeting-v3-step3-2-solver-backed-generic-smoke`  
> 基线分支：`retargeting-v3-step3-full-fleet-hardening`  
> 基线提交：`26817de67bdda0cb315a1237b53c30e4d8199c78`  
> 当前阶段：**Step 3.2 Solver-Backed Generic Runtime Smoke / Runtime Quality Improvement**
>
> Step 3.1.1 已作为 strict final evidence closure 闭环：final HEAD CI green、strict audit PASS、44-row artifacts clean-provenance、status semantics honest。Step 3.2 必须从该闭环基线继续，不能回退到 Step 2 或 Step 3.1 旧分支。
>
> 本轮目标是把 Step 3.1.1 的 **residual-only FK evaluation** 推进为 **generic solver-backed runtime smoke evidence**。这不是动作美化、不是视觉验收、不是 robot-specific tuning，也不是生产 `NewtonPipeline` 重写。
>
> **禁止修改 Step 2 artifacts / thresholds；禁止修改 Step 3.1.1 frozen artifacts 作为“补丁”；禁止按 robot id 写数学特例或阈值；禁止 IK 权重调参式刷分；禁止 contact / collision / temporal smoothing；禁止把 residual-only 行标成 `runtime_quality_passed`；禁止宣称 production retargeting 或视觉动作质量已经解决。**

---

## 0. 开始前必须执行

```bash
conda activate soma-retargeter-v2
cd ~/Desktop/soma_retargeter_autoconfig

git fetch origin --prune
git switch retargeting-v3-step3-2-solver-backed-generic-smoke || \
  git switch --track -c retargeting-v3-step3-2-solver-backed-generic-smoke origin/retargeting-v3-step3-2-solver-backed-generic-smoke

git pull --ff-only

git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor 26817de67bdda0cb315a1237b53c30e4d8199c78 HEAD
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
branch = retargeting-v3-step3-2-solver-backed-generic-smoke
base Step 3.1.1 final commit is ancestor
worktree clean
Git LFS fsck OK
snapshot XML/URDF materialized, not pointer-only
```

环境不满足时，只修环境或 checkout，不得改代码绕过。

---

## 1. 必须先阅读的文件

按顺序完整阅读：

1. `goal.md`（本文件）
2. `docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md`
3. `docs/retargeting_v3/STEP3_1_FULL_FLEET_RUNTIME_QUALITY_ACCEPTANCE.md`
4. `docs/retargeting_v3/subagents/step3_1_agent_f_red_team.md`
5. `artifacts/retargeting_v3_step3_runtime_quality/environment.json`
6. `artifacts/retargeting_v3_step3_runtime_quality/quality_summary.json`
7. `artifacts/retargeting_v3_step3_runtime_quality/model_matrix.json`
8. `artifacts/retargeting_v3_step3_runtime_quality/generic_smoke_matrix.json`
9. `artifacts/retargeting_v3_step3_runtime_quality/pipeline_backed_matrix.json`
10. `artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json`
11. `soma_retargeter/runtime/v3/generic_smoke.py`
12. `soma_retargeter/runtime/v3/fleet_harness.py`
13. `soma_retargeter/runtime/v3/quality_metrics.py`
14. `soma_retargeter/runtime/v3/runtime_quality_gates.py`
15. `soma_retargeter/runtime/v3/runtime_status.py`
16. `soma_retargeter/runtime/v3/runtime_local_profile.py`
17. `soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py`
18. `scripts/audit_retargeting_v3_step3_full_fleet.py`
19. `.github/workflows/retargeting_v3_step3_full_fleet.yml`
20. `tests/v3/test_step3_full_fleet_acceptance_audit.py`
21. `tests/v3/test_step3_runtime_quality_gates_status_semantics.py`

不要只读 summary。必须理解 Step 3.1.1 为什么是 honest evidence milestone，而不是 runtime quality success milestone。

---

## 2. 当前真实基线

Step 3.1.1 closed baseline：

```text
final_head = 26817de67bdda0cb315a1237b53c30e4d8199c78
GitHub Actions run = 28286161586
strict audit = PASS, blocking_count = 0
focused tests = 20 passed
git lfs fsck = OK
```

Step 3.1.1 artifact truth：

```text
in_scope_total = 44
full_humanoid_total = 32
partial_total = 3
negative_total = 9

runtime_quality_passed_count = 0
runtime_quality_warned_count = 23
runtime_quality_failed_count = 9
partial_runtime_passed_count = 3
negative_control_runtime_passed_count = 9

solver_backed_count = 0
residual_only_count = 32
deterministic_matched_count = 44
```

Interpretation：

```text
Step 3.1.1 proved evidence honesty.
Step 3.2 must improve runtime smoke depth by adding generic solver-backed evidence.
```

---

## 3. 本轮唯一目标

实现并验收一个 **generic solver-backed runtime smoke**，用于 full humanoid profiles 的 Step 3 runtime quality evidence。

完成后必须能回答：

1. 32 个 full humanoid 是否都被 solver-backed smoke 尝试过？
2. 哪些模型 solver-backed smoke completed？哪些 fail-closed？为什么？
3. `solver_backed_count` 是否从 `0` 变为真实的正数？
4. residual-only fallback 是否仍被禁止计入 `runtime_quality_passed`？
5. 哪些 warned/failed 是由 residual、joint-limit、NaN/Inf、target stream、solver convergence 或 determinism 引起？
6. 是否保持全局 gates，不按 robot id 调阈值？
7. RPO/G1 pipeline controls、negative controls、partial profiles 是否仍保持 Step 3.1.1 语义？
8. 新 artifacts 是否来自 clean source commit，并由 final HEAD CI 验证？

---

## 4. 严格范围边界

### 4.1 本轮包含

- generic solver-backed smoke 的设计与实现；
- full humanoid runtime smoke 从 residual-only FK evaluation 向 solver-backed evidence 迁移；
- 全局 solver-smoke gates / status semantics；
- fail-closed solver diagnostics；
- deterministic frame sampling / deterministic artifact hashing；
- 新 Step 3.2 artifact tree；
- Step 3.2 audit；
- focused tests 与 final HEAD CI evidence。

### 4.2 本轮禁止

- 修改 Step 2 artifacts、Step 2 thresholds 或 Step 2 acceptance 语义；
- 修改 Step 3.1.1 closed artifact tree 作为“刷结果”手段；
- 重写 production `NewtonPipeline` 默认行为；
- 默认启用 runtime override；
- 按 robot id 写特殊数学、特殊阈值、特殊白名单；
- 调 IK/objective weights 来追单个机器人；
- contact、collision、temporal smoothing、视觉动作质量调优；
- 把 residual-only fallback 叫做 `runtime_quality_passed`；
- 将 negative controls 提升为 humanoid runtime quality passes；
- 声称 full-fleet production retargeting 已解决。

---

## 5. 允许的 solver-backed smoke 方向

Codex 可以选择下面任一方案，但必须全局、通用、可审计、fail-closed。

### 5.1 方案 A：generic chain-projection frame smoke（推荐优先）

对每个 full humanoid，从 V3 profile / target stream 中提取可用 body/task anchors，执行 deterministic frame-level projection：

```text
source target stream frame
→ runtime model kinematic state candidate
→ generic chain/body projection update
→ FK verify target residuals + joint limits
```

要求：

- 不依赖 robot id；
- anchor/task 来自 profile capability 和 runtime model introspection；
- 使用固定全局 gates；
- 如果 anchor 不足、模型不可解、projection 不收敛，则 fail-closed 并记录原因；
- completed smoke 必须产生 solver-backed metrics。

### 5.2 方案 B：generic Newton IK smoke

使用 repo 中已有 Newton/MuJoCo 能力，在 isolated smoke path 内对抽样 frames 执行 generic IK/projection。

要求：

- 不重写 production `NewtonPipeline`；
- 不默认打开 experimental runtime override；
- 不调 robot-specific IK weights；
- 全局 iteration / tolerance / sampling config；
- convergence failure 必须进入 `runtime_quality_warned` 或 `runtime_quality_failed`，不能被吞掉。

---

## 6. Status 语义必须保持

### 6.1 `runtime_quality_passed`

只允许 full humanoid 使用，且必须满足：

```text
runtime profile resolved
runtime source materialized
V3 target stream finite
solver_backed_smoke_attempted = true
solver_backed_smoke_completed = true
solver_backed = true
normalized_task_residual_p95 <= global pass gate
joint_limit_violation_count == 0 或 <= global tolerance
output_nan_count = 0
output_inf_count = 0
deterministic matched
```

### 6.2 `runtime_quality_warned`

表示 infrastructure 和 solver/evaluation 完成，但 global warning gates 暴露问题，例如：

```text
high residual
joint-limit warning
solver convergence weak
target stream jump
support-height warning
velocity / smoothness warning
```

Warned 不能计入 `runtime_quality_passed_count`。

### 6.3 `runtime_quality_failed`

表示：

```text
solver failed closed
source/profile unexpectedly unavailable
target stream invalid
NaN/Inf
severe joint-limit violation
required metric missing
determinism mismatch
```

### 6.4 `runtime_evaluation_completed`

只用于 residual-only fallback。没有 solver-backed smoke 时，不能写 `runtime_quality_passed`。

### 6.5 Partial / negative

Partial 保持：

```text
partial_runtime_passed
```

Negative controls 保持：

```text
negative_control_runtime_passed
```

它们不进入 full humanoid runtime quality passed count。

---

## 7. Artifact 策略

本轮必须创建新的 Step 3.2 artifact tree，不要覆盖 Step 3.1.1 closed artifacts：

```text
artifacts/retargeting_v3_step3_2_solver_backed_smoke/
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
pipeline_backed_matrix.json 或 pipeline_controls_reference.json
test_results/pytest.txt
test_results/pytest_summary.json
```

`quality_summary.json` 至少记录：

```text
schema_version
base_step3_1_1_final_head
in_scope_total
full_humanoid_total
partial_total
negative_total
solver_backed_attempted_count
solver_backed_completed_count
solver_backed_failed_count
solver_backed_count
residual_only_count
runtime_quality_passed_count
runtime_quality_warned_count
runtime_quality_failed_count
partial_runtime_passed_count
negative_control_runtime_passed_count
deterministic_compared_count
deterministic_matched_count
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
solver_failure_reason / warning_reasons
frame_count
sampled_frame_indices
normalized_task_residual_mean / p95 / max
target_translation_error_mean / p95 / max
target_rotation_error_mean / p95 / max
joint_limit_violation_count
max_joint_limit_violation
output_nan_count
output_inf_count
runtime_seconds
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

## 8. Audit 与测试要求

必须新增或扩展 Step 3.2 专用 audit。推荐新增：

```text
scripts/audit_retargeting_v3_step3_2_solver_backed_smoke.py
```

Audit 必须 fail closed：

```text
44 rows present
32/3/9 partition preserved
Step 3.1.1 final head recorded as baseline
clean provenance
full humanoid solver_backed_smoke_attempted_count == 32
solver_backed_completed_count > 0
solver_backed_count > 0
residual_only rows are never runtime_quality_passed
runtime_quality_passed rows require solver_backed=true
negative controls not promoted
partial rows not counted as full humanoid passes
pipeline controls retained/referenced
numeric metrics finite and ordered
NaN/Inf counts zero for completed rows
deterministic rerun matched
final HEAD CI evidence present in strict mode
```

如果 solver-backed completed count 仍为 0，最终状态必须是 BLOCKED，并记录 blockers；不得把 Step 3.2 写成 PASS。

必须新增 focused tests。推荐覆盖：

```text
tests/v3/test_step3_2_solver_backed_smoke_status_semantics.py
tests/v3/test_step3_2_solver_backed_smoke_artifact_schema.py
tests/v3/test_step3_2_solver_backed_smoke_audit.py
tests/v3/test_step3_2_solver_backed_smoke_determinism.py
```

至少测试：

- solver-backed false 的 full humanoid 不能 `runtime_quality_passed`；
- residual-only fallback 不能 pass；
- `runtime_quality_passed` 必须有 `solver_backed=true`、finite metrics、joint-limit gate；
- negative controls 不被 promoted；
- missing solver evidence 触发 audit failure；
- deterministic hash 对稳定 artifacts 匹配；
- no robot-specific threshold table / whitelist introduced。

---

## 9. 建议执行顺序

1. 建立 Step 3.2 artifact schema 和 audit skeleton；
2. 写 failing tests，锁定 status semantics；
3. 实现 generic solver-backed smoke path；
4. 接入 fleet harness / full-fleet runner；
5. 生成新 Step 3.2 artifacts；
6. 做 deterministic rerun；
7. 更新 acceptance ledger / handoff；
8. 增加或更新 GitHub Actions workflow；
9. 跑 focused tests、audit、LFS fsck；
10. push 后记录 final HEAD CI evidence；
11. strict audit with final-head CI；
12. 写最终诚实结论。

---

## 10. 推荐命令模板

Codex 可以调整文件名，但如果调整，必须同步更新 goal / docs / tests / CI。

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step3_2_solver_backed_smoke_status_semantics.py \
  tests/v3/test_step3_2_solver_backed_smoke_artifact_schema.py \
  tests/v3/test_step3_2_solver_backed_smoke_audit.py \
  tests/v3/test_step3_2_solver_backed_smoke_determinism.py

PYTHONPATH=. python soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py \
  --artifact-dir artifacts/retargeting_v3_step3_2_solver_backed_smoke \
  --enable-solver-backed-generic-smoke \
  --deterministic-rerun

PYTHONPATH=. python scripts/audit_retargeting_v3_step3_2_solver_backed_smoke.py \
  --artifact-dir artifacts/retargeting_v3_step3_2_solver_backed_smoke \
  --source-root .

PYTHONPATH=. python scripts/audit_retargeting_v3_step3_2_solver_backed_smoke.py \
  --artifact-dir artifacts/retargeting_v3_step3_2_solver_backed_smoke \
  --source-root . \
  --require-final-head-ci
```

---

## 11. PASS / BLOCKED 定义

### Step 3.2 PASS 必须表示

```text
Step 3.1.1 baseline preserved
new Step 3.2 artifact tree generated from clean source
44-row full-fleet matrix preserved
32 full humanoids attempted solver-backed smoke
solver_backed_completed_count > 0
solver_backed_count > 0
residual_only_count reduced below Step 3.1.1 baseline OR every residual-only fallback is explicitly non-pass
no residual-only full humanoid row is runtime_quality_passed
runtime_quality_passed rows, if any, are solver-backed and gate-valid
negative controls not promoted
partial profiles not counted as full humanoid passes
deterministic rerun matched
focused tests passed
Git LFS fsck OK
final HEAD CI green
strict Step 3.2 audit passes
```

PASS 不代表：

```text
all humanoids pass runtime quality
visual motion quality is solved
production deployment is ready
contact/collision/smoothing is solved
robot-specific tuning has been performed
```

### Step 3.2 BLOCKED 必须表示

如果 solver-backed completed count 仍为 0、final HEAD CI 缺失、strict audit 有 blocker、或 status semantics 不诚实，必须写 BLOCKED。BLOCKED 也必须产出有用 diagnostics，不得隐藏失败。

---

## 12. 6 个建议 subagents

继续使用专业分工；主 Codex 是 Integrator。

```text
Agent A — Solver-smoke design / generic kinematic projection
Agent B — Runtime integration / fleet harness wiring
Agent C — Artifact schema / deterministic regeneration
Agent D — Tests / status semantics
Agent E — CI / workflow / LFS evidence
Agent F — Red-team audit / final strict acceptance
```

每个 agent 都必须遵守：不 robot-specific tuning、不修改 Step 2 thresholds、不弱化 audit、不把 residual-only 改名成 pass。

---

## 13. 最终 handoff 必须写清

最终交付时写入 `docs/retargeting_v3/HANDOVER_NEXT_WINDOW.md` 或 Step 3.2 handoff：

```text
branch
final_head
workflow_run_id
job conclusions
strict audit result
focused tests result
LFS fsck result
artifact counts
solver_backed_attempted_count
solver_backed_completed_count
solver_backed_count
runtime_quality_passed/warned/failed counts
known blockers or next Step 3.3 direction
```

不要自动开始 Step 3.3。
