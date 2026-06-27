# Goal — Step 3.1.1：Full-Fleet Runtime Hardening（全量运行时证据加固与真实 smoke）

> **Codex 强制执行契约——不得自由发挥**
>
> 当前分支：`retargeting-v3-step3-full-fleet-hardening`  
> 基线分支：`retargeting-v3-step3-runtime-quality-eval`  
> 基线提交：`fc0d21a7c00efbdc0c9d64177d4638354066d77e`  
> 当前阶段：**Step 3.1.1 Full-Fleet Runtime Evidence Hardening**
>
> 本轮不是继续扩大范围，也不是调动作质量。范围已经固定为 **44 个 in-scope Robot Zoo / local assets**。本轮只修三个问题：
>
> 1. 当前 Step 3.1 artifacts 不是 clean-provenance 生成；
> 2. 当前 generic smoke 太弱，只是 neutral FK residual 评估，却把高 residual 行标成 `runtime_quality_passed`；
> 3. 当前 CI / red-team evidence 对最终 HEAD 不够硬。
>
> 必须继续使用 **6 个 xhigh 专业 subagents**。主 Codex 是 Integrator，只负责接口冻结、合并、全量重跑和最终诚实结论。
>
> **禁止重写生产 `NewtonPipeline`，禁止调 IK/objective 权重，禁止 contact/collision/temporal smoothing，禁止修改 Step 2.1/2.3 thresholds，禁止 robot-ID 数学特例，禁止用 status rename 掩盖质量失败，禁止宣称视觉动作质量已经最终解决。**

---

## 0. 开始前必须执行

```bash
conda activate soma-retargeter-v2

git fetch origin --prune
git checkout retargeting-v3-step3-full-fleet-hardening
git pull --ff-only

git status --short
git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor fc0d21a7c00efbdc0c9d64177d4638354066d77e HEAD

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
branch = retargeting-v3-step3-full-fleet-hardening
worktree clean
base commit is an ancestor
Git LFS fsck OK
snapshot XML materialized, not pointer-only
```

环境不满足时，只修环境或 checkout，不得改代码绕过。

---

## 1. 必须先阅读的文件

按顺序完整阅读：

1. `/goal.md`（本文件）
2. `docs/retargeting_v3/STEP3_1_FULL_FLEET_RUNTIME_QUALITY_ACCEPTANCE.md`
3. `docs/retargeting_v3/subagents/step3_1_agent_f_red_team.md`
4. `artifacts/retargeting_v3_step3_runtime_quality/environment.json`
5. `artifacts/retargeting_v3_step3_runtime_quality/model_matrix.json`
6. `artifacts/retargeting_v3_step3_runtime_quality/quality_summary.json`
7. `artifacts/retargeting_v3_step3_runtime_quality/target_stream_matrix.json`
8. `artifacts/retargeting_v3_step3_runtime_quality/generic_smoke_matrix.json`
9. `artifacts/retargeting_v3_step3_runtime_quality/pipeline_backed_matrix.json`
10. `artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json`
11. `soma_retargeter/runtime/v3/generic_smoke.py`
12. `soma_retargeter/runtime/v3/fleet_harness.py`
13. `soma_retargeter/runtime/v3/quality_metrics.py`
14. `soma_retargeter/runtime/v3/runtime_local_profile.py`
15. `soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py`
16. `scripts/audit_retargeting_v3_step3_full_fleet.py`
17. `.github/workflows/retargeting_v3_step3_full_fleet.yml`
18. Step 3.0 runtime shadow artifacts under `artifacts/retargeting_v3_step3_runtime_shadow/`
19. Step 2.3.1 profile artifacts under `artifacts/retargeting_v3_step2_capability/`

不得只读 acceptance 文档就把当前状态视为通过。

---

## 2. 当前真实状态与硬 blocker

### 2.1 Step 3.1 当前声明

当前分支声称：

```text
in_scope_total = 44
full_humanoid_profile = 32
partial_humanoid_profile = 3
negative_control = 9
runtime_quality_passed = 32
partial_runtime_passed = 3
negative_control_runtime_passed = 9
quality_failed_count = 0
deterministic = 44/44
```

这些数字只能称为 **claimed artifact result**，不能直接接受为 Step 3.1 PASS。

### 2.2 Hard blocker A — dirty provenance

当前：

```text
artifacts/retargeting_v3_step3_runtime_quality/environment.json
```

记录了非空 `git.status_short`，包含大量 `M` 和 `??` 文件。也就是说当前 artifacts 是在脏 worktree / 未提交代码状态下生成的。

本轮必须修复为：

```text
git_status_short = ""
source_code_commit_remote_resolvable = true
source_code_commit_is_artifact_commit_ancestor = true
source_worktree_clean_before_run = true
source_worktree_clean_after_run = true
core_diff_after_source_commit = []
```

### 2.3 Hard blocker B — generic smoke 太弱

当前 `generic_smoke.py` 的 full humanoid smoke 是：

```text
runtime model neutral FK
vs
V3 target stream residual
```

它没有真正做 runtime per-frame projection / IK / chain solve。很多行 residual 很大，但只要 finite 就被标成 `runtime_quality_passed`。

示例模式：

```text
normalized_task_residual_mean ≈ 0.75
task_residual_p95 ≈ 4.0
status = passed
```

本轮必须二选一，不能模糊：

1. **实现更强 generic runtime solver smoke**，使 `runtime_quality_passed` 真的表示 residual / limit / determinism gates 通过；或
2. **收紧 status 语义**，把当前 neutral-FK residual evaluation 改成 `runtime_evaluation_completed` / `runtime_quality_warned` / `runtime_quality_failed`，不得把高 residual 行称为 quality passed。

推荐优先选择方案 1；若计算量或工程风险过大，可以选择方案 2，但必须在 summary 中诚实显示质量失败/警告数量。

### 2.4 Hard blocker C — current HEAD 没有 CI 证据

当前 HEAD 上没有可见 workflow run/status。最终必须有：

```text
workflow_run_id
head_sha
conclusion=success
job conclusions
```

并写入 Agent F final handoff。

### 2.5 Hard blocker D — full repo pytest 仍未完成

当前 acceptance ledger 已诚实记录 full repo pytest 中断，这是好的，但最终 PASS 需要：

- 要么 full repo pytest 完成；
- 要么明确将 full repo pytest 作为 non-blocking caveat，并确保 targeted / v3 / acceptance tests 全部覆盖本轮改动。

不得把 full repo pytest 写成 passed，除非真实 exit code 0。

---

## 3. 本轮唯一目标

将当前：

```text
44-row full-fleet artifact scaffold
```

硬化为：

```text
clean-provenance + CI-verified + status-honest + runtime-solver-backed full-fleet evaluation
```

完成后必须能回答：

1. 44 行矩阵是否来自 clean source commit？
2. 每行是否经过真实 runtime source/profile resolution？
3. Full humanoid 的 generic smoke 到底是 solver-backed 通过，还是只是 residual evaluation 警告/失败？
4. 哪些机器人 residual 高、joint limit 违规、target stream jump 或 solver失败？
5. 哪些结果只是 evaluation completed，而不是 quality passed？
6. RPO/G1 pipeline-backed controls 是否仍保持 Step 3.0 no-op / override / fail-closed 语义？
7. CI 是否在最终 HEAD 上通过？

---

## 4. 严格范围边界

### 4.1 本轮包含

- clean artifact generation protocol；
- audit 对 dirty provenance 的硬检查；
- generic smoke status 语义收紧；
- solver-backed full humanoid smoke 或 honest residual-only classification；
- quality thresholds / warning gates，仅用于 runtime quality classification，不修改 Step 2 thresholds；
- runtime-local profile forced-mismatch tests；
- full 44 matrix deterministic rerun；
- CI / red-team final evidence。

### 4.2 本轮禁止

- 修改 production `NewtonPipeline` 默认行为；
- 修改 Step 2 artifacts；
- 修改 Step 2 exact residual / numerical thresholds；
- 调 IK weights；
- 添加 robot-specific quality thresholds；
- 把 residual 大的 neutral-FK evaluation 叫 `runtime_quality_passed`；
- 只改 audit 来接受坏 artifacts；
- 只改文档；
- 删除 high-residual metrics；
- 用截图/视频目测作为 pass；
- 开始优化/tuning；
- 将 deferred `berkeley_humanoid_urdf` / `romeo_urdf` 重新加入 scope。

---

## 5. 新 status 语义

必须统一使用以下 final statuses：

```text
runtime_quality_passed
runtime_quality_warned
runtime_quality_failed
runtime_evaluation_completed
partial_runtime_passed
negative_control_runtime_passed
blocked_source_or_profile
```

### 5.1 `runtime_quality_passed`

只允许 full humanoid 使用，且必须满足：

```text
runtime profile resolved
source/load finite
target stream finite
solver-backed smoke executed
normalized_task_residual_p95 <= global runtime gate
joint_limit_violation_count == 0 或 violation <= global tolerance
nan_count = 0
inf_count = 0
deterministic matched
```

### 5.2 `runtime_quality_warned`

允许 full humanoid 使用，表示：

```text
all infrastructure complete
all metrics finite
solver/evaluation completed
but residual, joint-limit, smoothness, or support-height metrics exceed warning gates
```

Warned 不是 failure，但不能计入 `runtime_quality_passed_count`。

### 5.3 `runtime_quality_failed`

表示：

```text
runtime solver failed
NaN/Inf
source/profile unexpectedly unavailable
target stream invalid
joint limits severely violated
metric missing
```

### 5.4 `runtime_evaluation_completed`

仅用于 residual-only evaluation。若没有真正 solver-backed smoke，只能写这个或 `runtime_quality_warned/failed`，不能写 `runtime_quality_passed`。

### 5.5 Partial / negative

Partial：

```text
partial_runtime_passed
```

只表示 supported semantics smoke 完成，不表示 full humanoid pass。

Negative：

```text
negative_control_runtime_passed
```

只表示 load/rejection成功，不能进入 humanoid quality counts。

---

## 6. Runtime quality gates

新增文件：

```text
soma_retargeter/runtime/v3/runtime_quality_gates.py
```

全局 gates 必须固定，不得按机器人改：

```text
nan_count = 0
inf_count = 0
joint_limit_violation_severe_threshold = 1e-5
normalized_task_residual_p95_pass = 0.15
normalized_task_residual_p95_warn = 0.60
normalized_task_residual_max_warn = 1.20
joint_velocity_p95_warn = global numeric gate, derived from data but fixed in artifact
joint_acceleration_p95_warn = global numeric gate, derived from data but fixed in artifact
se3_orthogonality_error_max <= 1e-8
```

If Codex thinks these gates are too strict/loose, it must:

1. add synthetic/calibration evidence;
2. use one global change;
3. write ADR in report;
4. never set per-robot gates.

The gates classify runtime quality only. They must not modify Step 2 offline pass/fail.

---

## 7. Stronger generic smoke requirement

Current `run_full_humanoid_fk_smoke()` must not remain the only full-humanoid smoke path if status is `runtime_quality_passed`.

### 7.1 Acceptable smoke implementations

Choose one and document it:

#### Option A — chain-projection frame smoke, preferred

For each full humanoid / required clip / sampled frame:

```text
V3 target stream target transform
+ runtime profile chain/path/sites
+ Step 2 projection solver
→ projected q for each semantic task
→ residual / limit / certificate metrics
```

This is still not whole-body production IK, but it is solver-backed and uses runtime model truth.

#### Option B — generic Newton IK smoke

Generate generic semantic body objectives and run Newton IK with fixed global weights.

Allowed only if:

- no per-robot weights;
- objective mapping derives from verified semantics;
- no production config mutation;
- deterministic;
- failed robots report numeric failures.

#### Option C — residual-only evaluation

Allowed only as fallback. It must be classified as:

```text
runtime_evaluation_completed
runtime_quality_warned
runtime_quality_failed
```

Not `runtime_quality_passed`.

### 7.2 Solver-backed metrics

For solver-backed rows, save:

```text
solver_type
sampled_frame_count
semantic_task_count
solver_success_fraction
projection_status_counts
normalized_task_residual_mean/p95/max
joint_limit_violation_count
max_joint_limit_violation
certificate_class_counts
runtime_seconds
```

### 7.3 Residual-only metrics

For residual-only rows, save:

```text
solver_type = runtime_model_fk_residual_evaluation_only
quality_pass_allowed = false
classification_reason
```

Audit must fail if residual-only row is labeled `runtime_quality_passed`.

---

## 8. Runtime-local profile mismatch coverage

Current full-fleet artifacts show zero runtime-local profiles generated. That may be legitimate if all committed snapshot runtime sources match, but the code path must be tested.

Must add tests:

1. forced fingerprint mismatch full humanoid -> runtime-local profile generation;
2. forced mismatch negative control -> no humanoid profile generation;
3. runtime-local profile fails -> row is `blocked_source_or_profile` or `runtime_quality_failed`, not pass;
4. runtime-local profile artifacts include source profile ID, runtime source hash, semantic map hash, deterministic hash;
5. committed Step 2 artifacts are unchanged.

---

## 9. Artifact schema changes

Output root remains:

```text
artifacts/retargeting_v3_step3_runtime_quality/
```

Required files remain, but update contents.

### 9.1 `environment.json`

Must include:

```text
schema_version
source_code_commit
source_code_commit_remote_resolvable=true
source_code_commit_is_artifact_commit_ancestor=true
source_worktree_clean_before_run=true
source_worktree_clean_after_run=true
git_status_short=""
core_diff_after_source_commit=[]
artifact_commit
python/mujoco/newton/warp/numpy/scipy versions
```

`git.status_short` containing `M` or `??` is a hard failure.

### 9.2 `quality_summary.json`

Must include:

```text
in_scope_total = 44
full_humanoid_total = 32
partial_total = 3
negative_total = 9
runtime_quality_passed_count
runtime_quality_warned_count
runtime_quality_failed_count
runtime_evaluation_completed_count
partial_runtime_passed_count
negative_control_runtime_passed_count
solver_backed_row_count
residual_only_row_count
quality_failed_count
high_residual_row_count
joint_limit_warning_row_count
deterministic_compared_count
deterministic_matched_count
```

### 9.3 `model_matrix.json`

Each full humanoid row must include:

```text
model_id
category
final_step3_1_status
runtime_quality_classification
smoke_type
solver_backed
residual_only
normalized_task_residual_p95
normalized_task_residual_max
joint_limit_violation_count
max_joint_limit_violation
quality_gate_results
failure_or_warning_reasons
```

### 9.4 `generic_smoke_matrix.json`

Each row must include:

```text
model_id
clip_id
mode
solver_type
solver_backed
quality_pass_allowed
status
quality_classification
metrics
failure_or_warning_reasons
```

### 9.5 `acceptance_ledger.json`

Must be generated by current audit only:

```text
scripts/audit_retargeting_v3_step3_full_fleet.py
```

Must include:

```text
blocking_count
finding_count
quality_warning_count
runtime_quality_passed_count
runtime_quality_warned_count
runtime_quality_failed_count
ci_run_id if available
full_repo_pytest classification
```

---

## 10. Clean generation protocol

Must implement script:

```text
scripts/generate_step3_full_fleet_hardened_artifacts.sh
```

Protocol:

1. Ensure current worktree clean.
2. Commit all code/tests/docs first -> code commit `C`.
3. Push `C`.
4. Create detached clean worktree from `C`.
5. Run `git lfs pull && git lfs fsck`.
6. Run targeted tests.
7. Run `tests/v3`.
8. Attempt full repo pytest and record honest classification.
9. Run full-fleet quality command into external temp dir, not tracked tree.
10. Run audit on temp artifacts.
11. Ensure clean source worktree remains clean.
12. Copy artifacts into main worktree.
13. Commit artifact commit `A`.
14. Verify `C` is ancestor of `A`.
15. Verify `git diff --name-only C..A -- soma_retargeter tests scripts .github` is empty.
16. Push `A`.
17. Wait for CI, record run ID.

No artifact is accepted without this protocol.

---

## 11. CI requirements

Workflow:

```text
.github/workflows/retargeting_v3_step3_full_fleet.yml
```

Must run on:

```yaml
push:
  branches:
    - retargeting-v3-step3-full-fleet-hardening
```

Required jobs:

```text
full-fleet-static-and-unit
full-fleet-artifact-audit
lfs-and-snapshot-smoke
pipeline-backed-regression
quality-status-semantics
```

CI must fail if:

- model matrix rows != 44;
- dirty provenance;
- residual-only row labeled `runtime_quality_passed`;
- high residual row labeled pass without solver-backed evidence;
- RPO/G1 pipeline controls missing;
- CI is RPO/G1-only;
- acceptance ledger missing current audit;
- Agent F handoff stale.

---

## 12. Six xhigh subagents

Every agent must:

- use independent worktree/branch;
- record reasoning strength=`xhigh`;
- write failing tests first;
- only modify owned files;
- make small commits;
- write handoff with commit/files/commands/tests/numeric results/blockers/risks;
- never claim unrun tests.

### Agent A — Clean Provenance and Artifact Protocol

**Goal**: make dirty artifact generation impossible.

Owned files:

```text
scripts/generate_step3_full_fleet_hardened_artifacts.sh
scripts/write_step3_full_fleet_environment.py
soma_retargeter/runtime/v3/provenance.py
artifacts/retargeting_v3_step3_runtime_quality/environment.json
artifacts/retargeting_v3_step3_runtime_quality/commands.txt
tests/v3/test_step3_full_fleet_provenance_*.py
docs/retargeting_v3/subagents/step3_1_1_agent_a_handoff.md
```

Must implement:

1. source/artifact two-commit protocol;
2. clean detached worktree;
3. LFS fsck state;
4. remote-resolvable source commit;
5. source clean before/after;
6. core diff after source commit empty;
7. audit fails old dirty `environment.json` fixture;
8. no local absolute paths.

### Agent B — Quality Status Semantics and Gates

**Goal**: stop labeling residual-only/high-residual rows as quality pass.

Owned files:

```text
soma_retargeter/runtime/v3/runtime_quality_gates.py
soma_retargeter/runtime/v3/runtime_status.py
soma_retargeter/runtime/v3/quality_metrics.py
tests/v3/test_step3_quality_status_semantics_*.py
tests/v3/test_step3_runtime_quality_gates_*.py
docs/retargeting_v3/subagents/step3_1_1_agent_b_handoff.md
```

Must implement:

1. final statuses from section 5;
2. global quality gates;
3. pass/warn/fail classification;
4. residual-only cannot pass;
5. high residual -> warn/fail unless solver-backed gates pass;
6. negative/partial not counted as full quality pass;
7. no per-robot thresholds;
8. tests for current bad pattern.

### Agent C — Solver-Backed Generic Runtime Smoke

**Goal**: replace or complement neutral-FK residual smoke with actual solver-backed runtime smoke.

Owned files:

```text
soma_retargeter/runtime/v3/generic_smoke.py
soma_retargeter/runtime/v3/fleet_harness.py
soma_retargeter/runtime/v3/generic_projection_smoke.py      # may add
soma_retargeter/runtime/v3/generic_ik.py                    # only if needed
tests/v3/test_step3_solver_backed_generic_smoke_*.py
tests/v3/test_step3_residual_only_not_quality_pass_*.py
docs/retargeting_v3/subagents/step3_1_1_agent_c_handoff.md
```

Must implement:

1. chain projection or generic IK smoke;
2. runtime model FK/solver actually used;
3. no neutral-only pass;
4. full humanoid 2 core clips minimum solver-backed smoke;
5. deterministic frame sampling;
6. solver success / residual / limits metrics;
7. fallback residual-only classified as evaluation/warn/fail;
8. no per-robot tuning.

### Agent D — Runtime-Local Profile Mismatch Tests and Closure

**Goal**: prove mismatch path works even if real 44 snapshots mostly match.

Owned files:

```text
soma_retargeter/runtime/v3/runtime_local_profile.py
soma_retargeter/runtime/v3/profile_loader.py
soma_retargeter/tools/compile_runtime_local_profiles_v3.py
tests/v3/test_step3_forced_runtime_profile_mismatch_*.py
tests/v3/test_step3_negative_mismatch_no_profile_*.py
artifacts/retargeting_v3_step3_runtime_quality/runtime_local_profile_summary.json
artifacts/retargeting_v3_step3_runtime_quality/runtime_local_profiles/
docs/retargeting_v3/subagents/step3_1_1_agent_d_handoff.md
```

Must implement:

1. forced mismatch fixture;
2. runtime-local profile generation for full/partial;
3. fail closed for negative;
4. deterministic hash;
5. no Step 2 artifact mutation;
6. source/profile/map hash saved;
7. mismatch never silently ignored.

### Agent E — Full-Fleet Orchestration and Artifacts

**Goal**: regenerate full 44 artifacts with hardened status and solver evidence.

Owned files:

```text
soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py
scripts/run_step3_full_fleet_runtime_quality.sh
artifacts/retargeting_v3_step3_runtime_quality/
tests/v3/test_step3_full_fleet_artifact_schema_runtime_quality.py
tests/v3/test_step3_full_fleet_determinism_runtime_quality.py
docs/retargeting_v3/STEP3_1_1_FULL_FLEET_HARDENING_REPORT.md
docs/retargeting_v3/subagents/step3_1_1_agent_e_handoff.md
```

Must implement:

1. 44 rows still present;
2. new status counts;
3. quality warning/failure rows preserved;
4. deterministic rerun compares status, metrics, residuals and hashes;
5. full repo pytest classification saved;
6. no dirty artifacts;
7. commands saved;
8. high residual not hidden.

### Agent F — Independent Red Team and CI

**Goal**: attack final claims and decide PASS/BLOCKED.

Owned files:

```text
scripts/audit_retargeting_v3_step3_full_fleet.py
.github/workflows/retargeting_v3_step3_full_fleet.yml
tests/v3/test_step3_full_fleet_acceptance_*.py
docs/retargeting_v3/STEP3_1_1_FULL_FLEET_HARDENING_ACCEPTANCE.md
docs/retargeting_v3/subagents/step3_1_1_agent_f_red_team.md
artifacts/retargeting_v3_step3_runtime_quality/test_results/
artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json
```

Must fail on:

1. dirty `environment.json`;
2. source commit not remote-resolvable;
3. residual-only row marked `runtime_quality_passed`;
4. normalized residual p95 above pass gate but pass status without solver-backed evidence;
5. 44-row matrix missing;
6. negative control promoted;
7. RPO/G1 pipeline controls missing;
8. full repo pytest caveat absent or falsely passed;
9. current HEAD CI absent;
10. stale handoff PASS;
11. Step 2 artifacts modified;
12. local absolute path leakage;
13. LFS pointer-only files.

Final handoff must include:

```text
verdict = PASS or BLOCKED
final_head
source_code_commit
artifact_commit
workflow_run_id
44-row matrix summary
runtime_quality_passed/warned/failed counts
residual-only row count
solver-backed row count
RPO/G1 pipeline control summary
pytest summary
full-repo pytest classification
remaining blockers
```

---

## 13. Integrator fixed order

Do not change order.

### Wave 0 — Red-team tests first

1. Start six xhigh agents.
2. Agent F writes failing tests for dirty provenance and residual-only pass.
3. Agent B freezes status semantics.
4. Integrator freezes artifact schema.

### Wave 1 — Provenance and status

Merge:

```text
Agent A -> Agent B
```

Run targeted tests.

### Wave 2 — Smoke solver

Merge Agent C. Verify residual-only rows cannot pass.

### Wave 3 — Runtime-local mismatch

Merge Agent D. Verify forced mismatch tests.

### Wave 4 — Full orchestration

Merge Agent E. Generate development artifacts. Run Agent F live audit. Return blockers.

### Wave 5 — Clean final artifacts

1. code freeze commit `C`;
2. push `C`;
3. clean detached worktree;
4. LFS pull/fsck;
5. targeted tests;
6. tests/v3;
7. full repo pytest attempt/classification;
8. full-fleet command;
9. audit;
10. artifact commit `A`;
11. push `A`;
12. CI;
13. Agent F final PASS/BLOCKED.

---

## 14. Required commands

### 14.1 Targeted tests

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step3_full_fleet_provenance_*.py \
  tests/v3/test_step3_quality_status_semantics_*.py \
  tests/v3/test_step3_runtime_quality_gates_*.py \
  tests/v3/test_step3_solver_backed_generic_smoke_*.py \
  tests/v3/test_step3_residual_only_not_quality_pass_*.py \
  tests/v3/test_step3_forced_runtime_profile_mismatch_*.py \
  tests/v3/test_step3_negative_mismatch_no_profile_*.py \
  tests/v3/test_step3_full_fleet_artifact_schema_runtime_quality.py \
  tests/v3/test_step3_full_fleet_determinism_runtime_quality.py \
  tests/v3/test_step3_full_fleet_acceptance_*.py
```

### 14.2 V3 suite

```bash
PYTHONPATH=. python -m pytest -q tests/v3
```

### 14.3 Full repo suite

```bash
PYTHONPATH=. python -m pytest -q
```

If interrupted/too slow, save exact classification. Do not report pass unless exit code 0.

### 14.4 Full fleet command

```bash
PYTHONPATH=. python -m soma_retargeter.tools.run_v3_full_fleet_runtime_quality \
  --artifact-root artifacts/retargeting_v3_step3_runtime_quality \
  --step2-profile-root artifacts/retargeting_v3_step2_capability \
  --step3-shadow-root artifacts/retargeting_v3_step3_runtime_shadow \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --clip-root assets/motions \
  --required-core-clips \
    assets/motions/bvh/Neutral_walk_forward_002__A057.bvh \
    assets/motions/bvh/wave_R_001__A428.bvh \
    assets/motions/bvh/body_stretch_1_004__A069.bvh \
    assets/motions/bvh/item_pick_up_standing_R_001__A410.bvh \
  --short-max-frames 120 \
  --mid-max-frames 300 \
  --deterministic-rerun
```

### 14.5 Audit

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_full_fleet.py \
  --artifact-dir artifacts/retargeting_v3_step3_runtime_quality \
  --source-root . \
  --write-report artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json
```

---

## 15. Final acceptance checklist

### Scope

- [ ] 44 model rows;
- [ ] 32 full / 3 partial / 9 negative;
- [ ] non-RPO/G1 rows >= 41;
- [ ] no deferred assets reintroduced;
- [ ] no model omitted.

### Provenance

- [ ] clean source commit;
- [ ] artifact commit;
- [ ] source remote-resolvable;
- [ ] source clean before/after;
- [ ] core diff after source commit empty;
- [ ] environment git status empty;
- [ ] LFS OK;
- [ ] no absolute paths.

### Runtime status honesty

- [ ] residual-only rows not labeled runtime_quality_passed;
- [ ] high residual rows warned/failed unless solver-backed gates pass;
- [ ] quality warning/failure counts explicit;
- [ ] no hidden NaN/Inf;
- [ ] joint limit violations reported.

### Generic smoke

- [ ] solver-backed rows exist for full humanoids, or residual-only status is honest;
- [ ] solver type recorded;
- [ ] solver success metrics recorded;
- [ ] partial supported smoke works;
- [ ] negatives rejected.

### Pipeline controls

- [ ] RPO/G1 Step 3.0 controls retained;
- [ ] shadow no-op retained;
- [ ] G1 fail-closed or runtime-local policy recorded;
- [ ] production default unchanged.

### Tests/CI

- [ ] targeted tests pass;
- [ ] tests/v3 pass;
- [ ] full repo pytest pass or caveat recorded honestly;
- [ ] audit PASS;
- [ ] CI current HEAD success;
- [ ] Agent F final PASS/BLOCKED.

---

## 16. Completion definition

Final report must be exactly one of:

```text
Step 3.1.1 Full-Fleet Runtime Hardening: PASS
```

or:

```text
Step 3.1.1 Full-Fleet Runtime Hardening: BLOCKED
```

PASS means:

```text
44-row matrix complete
clean provenance fixed
CI current HEAD green
runtime quality status honest
residual-only smoke not overclaimed
solver-backed or warned/failed rows classified correctly
RPO/G1 pipeline controls intact
all artifacts deterministic and auditable
```

PASS does **not** mean every robot motion looks good. If runtime quality is bad, the correct result is `runtime_quality_warned` or `runtime_quality_failed` with numeric evidence.

If any hard gate fails, write BLOCKED and list blockers. Stop after this round. Do not start tuning/optimization automatically.
