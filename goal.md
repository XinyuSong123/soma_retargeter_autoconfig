# Goal — Step 2.3.1：Capability Acceptance Hardening（严格验收加固）

> **Codex 强制执行契约——不得自由扩展范围**
>
> 当前分支：`retargeting-v3-step2-capability-acceptance-hardening`  
> 基线分支：`retargeting-v3-step2-capability-projection-fix`  
> 基线提交：`95a631173feadd20985a112959d0896203640d8f`  
> 真实 Assets44 对照提交：`5ad5a001c445c525d4c8bbaf6339dec5c5c2c719`  
> 当前阶段：**Step 2.3.1 Acceptance Evidence Hardening**
>
> 本轮只做验收、证据、状态策略、可复现性和 CI 加固。现有 continuation / KKT / capability solver 不得被重新设计。
>
> 本轮必须一次创建并持续高强度使用 **6 个 xhigh 专业 subagents**。主 Codex 仅作为 Integrator，必须按本文给出的文件 ownership、合并顺序和验收命令执行。
>
> **禁止进入 Step 3，禁止修改生产 `NewtonPipeline`，禁止放宽任何 numerical 或 projection threshold，禁止按机器人名称加特例，禁止重新处理两个 deferred assets。**

---

## 0. Codex 启动时必须逐项确认

```bash
conda activate soma-retargeter-v2

git fetch origin --prune
git checkout retargeting-v3-step2-capability-acceptance-hardening
git pull --ff-only

git status --short
git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor 95a631173feadd20985a112959d0896203640d8f HEAD

git lfs install
git lfs pull
git lfs fsck

which python
python --version
python - <<'PY'
import mujoco, newton, warp, numpy, scipy
print("mujoco", mujoco.__version__)
print("newton", getattr(newton, "__version__", "unknown"))
print("warp", getattr(warp, "__version__", "unknown"))
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
PY
```

开始开发前必须满足：

```text
branch = retargeting-v3-step2-capability-acceptance-hardening
worktree clean
base commit is an ancestor
Git LFS fsck OK
所有 snapshot XML 已 materialize，不是 LFS pointer
```

若环境不满足，只修环境或 checkout，不得修改算法来绕过依赖问题。

---

## 1. 必须先阅读的文件

按顺序完整阅读：

1. `/goal.md`（本文件）
2. `docs/retargeting_v3/STEP2_3_CAPABILITY_REPORT.md`
3. `docs/retargeting_v3/STEP2_3_CAPABILITY_ACCEPTANCE.md`
4. `docs/retargeting_v3/subagents/capability_agent_f_red_team.md`
5. `artifacts/retargeting_v3_step2_capability/summary.json`
6. `artifacts/retargeting_v3_step2_capability/before_after.json`
7. `artifacts/retargeting_v3_step2_capability/environment.json`
8. `artifacts/retargeting_v3_step2_capability/capability_audit.json`
9. `artifacts/retargeting_v3_step2_capability/deterministic_rerun.json`
10. `artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json`
11. `soma_retargeter/robotics/v3/projection_solver.py`
12. `soma_retargeter/robotics/v3/projection_certificate.py`
13. `soma_retargeter/robotics/v3/capability_projection.py`
14. `soma_retargeter/robotics/v3/chain_projection.py`
15. `soma_retargeter/robotics/v3/canonical_projection.py`
16. `soma_retargeter/robotics/v3/profile.py`
17. `soma_retargeter/robotics/v3/validation.py`
18. `scripts/audit_retargeting_v3_capability.py`
19. `.github/workflows/retargeting_v3_capability.yml`

不得根据文档中写的 `PASS` 假定当前已经通过。

---

## 2. 当前真实状态和已确认缺陷

### 2.1 当前 artifacts 的声明

当前提交声明：

```text
passed                       9
capability_limited_passed   23
partial_passed               3
negative_control_passed      9
algorithm_failed             0
deterministic matched       44/44
```

这些数字目前只能称为 **claimed result**，不能称为 accepted result。

### 2.2 当前必须修复的 10 个硬缺陷

#### 缺陷 A：CI 没有在实际分支触发

`.github/workflows/retargeting_v3_capability.yml` 当前 push branch 写成：

```text
retargeting-v3-step2-capability
```

实际分支是：

```text
retargeting-v3-step2-capability-acceptance-hardening
```

此外 workflow 当前将 `--artifact-dir` 指向旧的 Assets44 artifact root，而不是新的 capability artifact root。

#### 缺陷 B：Agent F handoff 仍然是 FAIL

`docs/retargeting_v3/subagents/capability_agent_f_red_team.md` 当前仍写：

```text
Live capability audit: FAIL
Do not accept Step 2.3 capability yet
```

它必须在最终代码和最终 artifacts 上重新运行后更新，不能只修改文字。

#### 缺陷 C：缺少声明过的测试 artifacts

当前远端缺少：

```text
artifacts/retargeting_v3_step2_capability/test_results/pytest.txt
artifacts/retargeting_v3_step2_capability/test_results/junit.xml
artifacts/retargeting_v3_step2_capability/lfs_state.json
```

#### 缺陷 D：acceptance ledger 是旧 Step 2.2 ledger

`artifacts/retargeting_v3_step2_capability/acceptance_ledger.json` 当前运行的是：

```text
audit_retargeting_v3_assets44.py
```

必须替换为本轮 capability audit 的真实命令、return code、stdout、source commit 和时间。

#### 缺陷 E：artifact source commit 在远端不可解析

当前 `environment.json` 记录：

```text
8451289ea6ac658b615b9bedcc93b3cf2b31f378
```

该 commit 当前无法在远端解析。最终 `source_code_commit` 必须：

- 是本仓库真实 commit；
- 已 push 到远端；
- 是 artifact commit 的祖先；
- 可以通过 `git cat-file -e <sha>^{commit}`；
- artifacts 生成后不得再修改 core code。

#### 缺陷 F：before/after 是程序伪造的

当前 `validation.py::_before_after_matrix()` 使用：

```python
before = "partial_passed" if status == capability_limited_passed else status
```

这不是读取真实 baseline，而是在根据 after status 编造 before status。必须删除。

真实 baseline 必须从固定提交：

```text
5ad5a001c445c525d4c8bbaf6339dec5c5c2c719
```

读取。

#### 缺陷 G：capability pass 只检查三个 gate

当前 `profile.py::_projection_certificate_passes()` 只要求：

```text
projected_gradient_kkt
seed_consensus
residual_explained
```

这不足以证明 capability-limited pass。

#### 缺陷 H：`_red_team_kkt_certificate()` 生成了不可独立验证的默认值

当前 `canonical_projection.py::_red_team_kkt_certificate()` 存在以下危险行为：

- `primal_feasible=True` 直接写死；
- `prior_gradient_inf_norm=0.0` 直接写死；
- `prior_cancellation_ratio=0.0` 直接写死；
- `residual_explained` 缺失时默认 True；
- task gradient 为零时人为改成 `1e-15`；
- complementarity 不是从原始 q/bounds/gradient 独立计算。

该 helper 必须删除。不能用另一个同名 helper 继续合成“审计友好字段”。

#### 缺陷 I：seed consensus 只比较 residual norm

当前 seed consensus 没有可靠比较多个竞争 seed 的最终 task-space 投影点和 certificate class。

#### 缺陷 J：stress motion 会污染 profile status

当前 `_has_capability_limited_certificate()` 只要发现任何 limited certificate 就将整机标记为 capability-limited，包括：

```text
extreme_but_valid_joint_limit_stress
```

Stress motion 只能作为诊断，不能把一个普通动作全部 exact 的机器人降级为 capability-limited。

---

## 3. 本轮唯一目标

本轮必须把：

```text
claimed 44/44 terminal pass
```

变成：

```text
independently recomputed + reproducible + CI-verified 44/44 terminal pass
```

必须满足：

1. 真正读取固定 baseline；
2. 真实 before/after；
3. 21 个 baseline exact pass 不得降级；
4. 11 个 baseline algorithm failures 才是本轮允许转成 capability-limited 的主要集合；
5. 每个超阈值普通/扩展任务的 certificate 必须由 raw solver evidence 独立重算；
6. 不得通过合成字段或默认 True 通过 audit；
7. stress motion 不参与 profile status；
8. artifacts 对应远端可解析 clean source commit；
9. deterministic 真实覆盖 44/44；
10. CI 在实际分支上真实运行并通过；
11. Agent F 最终 handoff 必须在最终 HEAD 上给出 PASS。

---

## 4. 绝对禁止事项

本轮禁止：

- 重新设计或重写 continuation solver；
- 替换 SciPy solver；
- 修改 task weights；
- 修改 neutral/hand/foot/torso exact thresholds；
- 修改 Step 2.1 epsilon、rank、subspace thresholds；
- 添加任何 per-robot threshold；
- 添加 robot ID 数学分支；
- 修改 semantic maps，除非独立审计发现明确 hash/fingerprint 错误；
- 修改 Robot Zoo manifest；
- 重新纳入 `berkeley_humanoid_urdf` 或 `romeo_urdf`；
- 修改生产 `NewtonPipeline`；
- 开始 Step 3；
- 仅改文档中的 FAIL 为 PASS；
- 仅改 audit 来接受当前 artifacts；
- 仅改 status 名称；
- 编造 baseline；
- 编造 source commit；
- 手写 KKT 通过字段；
- 将 stress motion 用作 capability-limited status 依据；
- 删除失败测试；
- 用旧 artifacts 冒充新结果。

若真实重算发现 solver 数学错误：

1. 必须先新增最小 failing synthetic test；
2. Agent B 写出数学推导；
3. Agent F 确认不是 threshold/证据问题；
4. Integrator 才允许最小修复；
5. 修复后全量重跑。

---

## 5. 不可修改的 baseline truth

### 5.1 Baseline 来源

唯一合法 baseline：

```text
commit = 5ad5a001c445c525d4c8bbaf6339dec5c5c2c719
artifact root = artifacts/retargeting_v3_step2_assets44
```

必须通过 Git 读取，不得读取当前工作树后反推：

```bash
git show 5ad5a001c445c525d4c8bbaf6339dec5c5c2c719:artifacts/retargeting_v3_step2_assets44/summary.json
```

每台机器人的 baseline status 必须通过：

```bash
git show 5ad5a001c445c525d4c8bbaf6339dec5c5c2c719:artifacts/retargeting_v3_step2_assets44/per_robot/<id>.json
```

或读取该 commit 中 `summary.json.reports`。

### 5.2 Baseline 固定计数

```text
passed                    21
partial_passed             3
negative_control_passed    9
algorithm_failed          11
in_scope total            44
```

### 5.3 合法状态转换

只允许：

```text
passed                  -> passed
partial_passed          -> partial_passed
negative_control_passed -> negative_control_passed
algorithm_failed        -> passed
algorithm_failed        -> capability_limited_passed
```

不允许：

```text
passed -> capability_limited_passed
passed -> partial_passed
partial_passed -> capability_limited_passed
negative_control_passed -> 任何 humanoid status
```

若真实新 solver 结果使 baseline `passed` 无法维持 exact pass，则必须：

```text
Step 2.3.1: BLOCKED
```

不得修改 baseline 或伪造 transition。

### 5.4 最终计数约束

最终必须满足：

```text
passed >= 21
capability_limited_passed <= 11
partial_passed = 3
negative_control_passed = 9
algorithm_failed = 0
source_unavailable = 0
model_load_failed = 0
semantic_failed = 0
terminal total = 44
```

其中：

```text
passed + capability_limited_passed = 32
```

即 32 个完整 humanoid profile，外加 3 partial 和 9 negative controls。

---

## 6. Canonical motion 状态策略——必须精确实现

### 6.1 Motion classes

在代码中定义固定、无机器人特例的 motion class：

```python
INVARIANCE_MOTIONS = {
    "neutral",
    "root_translation",
    "global_root_yaw",
}

ORDINARY_MOTIONS = {
    "torso_pitch",
    "torso_roll",
    "torso_yaw",
    "arms_forward",
    "elbow_bend",
    "squat",
    "single_step",
}

EXTENDED_MOTIONS = {
    "mixed_torso_rotation",
    "overhead_reach",
    "asymmetric_arm_reach",
    "crossed_body_reach",
}

STRESS_MOTIONS = {
    "extreme_but_valid_joint_limit_stress",
}
```

必须测试 motion 集合：

- 无遗漏；
- 无重复；
- canonical motion order 中每项恰好属于一个 class。

### 6.2 Invariance motion

Invariance motion 必须 exact：

```text
normalized_residual <= 现有 exact threshold
```

不允许 capability-limited certificate 将 calibration、root policy 或 frame bug 包装为 pass。

### 6.3 Ordinary / Extended motion

每个 task 只能得到：

```text
exact_reachable
capability_limited_rank
capability_limited_joint_limits
capability_limited_mixed
unsupported_rank_zero
solver_failed
numerical_invalid
invalid_target_geometry
```

Profile status：

- 所有 ordinary + extended required tasks exact：`passed`；
- 至少一个 ordinary/extended task 是有效 limited certificate，其他 required tasks 均 exact/valid limited：`capability_limited_passed`；
- 任意 required task 是 failed/invalid：`algorithm_failed`。

### 6.4 Stress motion

Stress motion 必须：

- finite；
- within limits；
- deterministic；
- 有 exact/limited/failed 诊断；

但：

```text
Stress motion 不得改变 robot profile status。
```

### 6.5 Rank-zero

Rank-zero 非零 demand 必须保留：

```text
demand_residual > 0
unreachable_demand = true
rank_zero_reason = no_active_coordinates_nonzero_demand
```

并且：

- 不向其他 chain 泄漏；
- projected q 保持合法；
- certificate class=`unsupported_rank_zero`；
- profile 可以 capability-limited pass。

---

## 7. Certificate schema v2——必须按此字段实现

所有普通/扩展 motion task 都必须输出：

```json
{
  "capability_certificate": {
    "schema_version": 2,
    "certificate_class": "...",
    "passed": true,
    "motion_class": "ordinary|extended|invariance|stress",
    "task_block": "translation|rotation",
    "exact_threshold": 0.0,
    "exact_threshold_passed": false,
    "gates": {},
    "decomposition": {},
    "kkt": {},
    "seed_consensus": {},
    "continuation": {},
    "joint_limits": {},
    "numerical": {},
    "audit_evidence": {},
    "deterministic_digest": "..."
  }
}
```

### 7.1 `audit_evidence` 必须包含原始数据

```text
desired_vector
projected_vector
seed_vector
normalized_residual_vector
demand_vector
relevant_task_jacobian
active_coordinates
q_active
lower_bounds
upper_bounds
task_gradient
prior_gradient
seed_results
continuation_history
scalar_dtype
normalization_scale
```

`audit_evidence` 不能只保存已经计算后的 bool。

### 7.2 禁止重复的伪 certificate

删除：

```python
canonical_projection.py::_red_team_kkt_certificate
```

新的 audit 直接读取：

```text
capability_certificate
active_limit_kkt
seed_results
continuation_history
```

若为了兼容保留 `kkt_certificate` 字段，它只能是：

```text
capability_certificate.kkt 的无修改引用/复制
```

不得补默认值，不得改零值，不得独立制造 `certified=true`。

---

## 8. 独立数学重算规则

Audit 必须从 `audit_evidence` 独立重算，不能相信 artifact 里的 `passed` 或 gate bool。

### 8.1 Task residual

Position：

```text
e = (p_projected - p_desired) / L_chain
```

Orientation：

```text
e = Log(R_projected^T R_desired) / pi
```

artifact 中的 residual vector 与重算结果必须一致。

### 8.2 Task gradient

```text
g_task = J_task^T e
```

Audit 重算的 `g_task` 必须与序列化值一致。

### 8.3 Bound-constrained projected gradient

```text
G(q) = q - clip(q - g_task, lower, upper)
```

```text
projected_gradient_inf_norm = ||G(q)||_inf
```

### 8.4 Primal feasibility

```text
primal_violation = max(lower - q, q - upper, 0)
```

`primal_feasible` 必须由该值重算，不能写死 True。

### 8.5 KKT sign / dual feasibility

对最小化任务：

```text
free coordinate: |g_i| <= tol
at lower bound:   g_i >= -tol
at upper bound:   g_i <=  tol
fixed coordinate: 记录为 fixed，不伪造 free
```

Audit 必须重算 complementarity/sign violation。

### 8.6 Prior cancellation

必须序列化：

```text
g_prior = J_prior^T r_prior
```

定义：

```text
if ||g_task||_inf <= stationarity_tol and ||g_prior||_inf <= stationarity_tol:
    prior_cancellation_ratio = 0
else:
    prior_cancellation_ratio = ||g_prior||_inf / max(||g_task||_inf, stationarity_tol)
```

不得要求 `task_gradient_inf_norm > 0`。对于 rank-incompatible residual，正确的 task gradient 可以精确接近零。

### 8.7 Reachable decomposition

使用 uncertainty-aware retained rank：

```text
J = U S V^T
P = U_r U_r^T
e_parallel = P e
e_orthogonal = (I-P)e
```

重算并保存：

```text
reachable_residual_norm
orthogonal_residual_norm
reachable_residual_fraction
orthogonal_residual_fraction
rank
rank_threshold
singular_values
```

### 8.8 Seed consensus

每个 active chain 至少尝试：

```text
neutral seed
selected continuation seed
limit-aware midpoint/axis seed
```

每个 seed result 必须保存：

```text
accepted
final_task_vector
normalized_residual
certificate_class
active_limits
final_q_active
```

定义 competitive seeds：

```text
normalized_residual <= best + global_seed_residual_tolerance
```

对 competitive seeds 比较：

```text
max pairwise ||final_task_vector_i - final_task_vector_j|| / task_scale
```

必须小于全局 task-space tolerance。

只比较 residual norm 不算 seed consensus。

Rank-zero 可使用 trivial deterministic consensus，但必须明确：

```text
source = trivial_rank_zero
```

### 8.9 Continuation

Audit 必须检查：

- alpha 从 0 开始；
- accepted alpha 严格递增；
- 最终 alpha=1；
- 所有 step finite；
- 所有 q within bounds；
- final task vector 与最终 artifact 一致；
- 未达到 alpha=1 不能 certificate pass。

### 8.10 Numerical gate

必须记录：

```text
jacobian source
scalar dtype
rank stability gate
engine/FD validation status
nonfinite count
```

存在 `engine_fd_mismatch`、`nonfinite` 或 `unstable_nonsmooth` 时不能 capability pass。

---

## 9. Class-specific certificate gates

不得只检查统一的三个 bool。

### 9.1 `exact_reachable`

必须：

```text
exact_threshold_passed
primal_feasible
projected_gradient_kkt
seed_consensus
continuation
joint_limits
numerical
```

### 9.2 `capability_limited_rank`

必须：

```text
exact_threshold_passed = false
rank < task dimension 或存在明确 rank-incompatible component
orthogonal_residual_fraction >= 0.95（使用全局固定 gate）
reachable_residual_fraction <= 0.05
projected_gradient_kkt
seed_consensus
continuation
joint_limits
numerical
residual_explained
```

### 9.3 `capability_limited_joint_limits`

必须：

```text
exact_threshold_passed = false
active bound count > 0
primal_feasible
KKT sign/complementarity pass
projected_gradient_kkt
seed_consensus
continuation
joint_limits
numerical
residual_explained
```

### 9.4 `capability_limited_mixed`

必须同时具有：

```text
rank-incompatible evidence
active-limit evidence
全部 rank 和 joint-limit gates
```

### 9.5 `unsupported_rank_zero`

必须：

```text
rank = 0
active coordinate count = 0
nonzero demand preserved
projected q unchanged/valid
no chain leakage
rank-zero KKT vacuously satisfied
numerical finite
deterministic
```

### 9.6 Failed classes

以下任何一个出现，robot 必须 `algorithm_failed`：

```text
solver_failed
numerical_invalid
invalid_target_geometry
certificate passed=false
missing raw audit evidence
independent audit mismatch
```

---

## 10. Profile status 必须由完整任务矩阵计算

删除当前“发现任意 limited certificate 就 capability-limited”的策略。

实现一个单独模块：

```text
soma_retargeter/robotics/v3/capability_status.py
```

该模块不得 import Robot Zoo ID，不得读取 manifest，不得按机器人特例。

输入：

```text
canonical motion task reports
motion class
required tasks
```

输出：

```text
passed
capability_limited_passed
algorithm_failed
```

### 10.1 `passed`

- invariance 全 exact；
- ordinary 全 exact；
- extended 全 exact；
- stress 不影响 status；
- 无 failed certificate。

### 10.2 `capability_limited_passed`

- invariance 全 exact；
- ordinary/extended 至少一个 valid limited；
- ordinary/extended 其余全部 exact 或 valid limited；
- stress 不影响 status；
- 无 failed/invalid certificate。

### 10.3 `algorithm_failed`

- invariance 任一不 exact；
- required ordinary/extended 任一 failed/invalid；
- certificate evidence 缺失；
- independent audit不一致。

Structured partial 和 negative controls 继续由原独立路径处理，不能进入该模块。

---

## 11. 六个 xhigh Subagents——严格 ownership

每个 agent 必须：

- 独立 worktree/branch；
- reasoning strength=`xhigh`；
- 先写 failing tests；
- 只修改 owned files；
- 不得修改别人的 owned files；
- 每个逻辑变更一个小 commit；
- handoff 写明 commit、files、commands、results、risks；
- 不得声明未运行的测试通过。

### Agent A — Baseline Truth and Real Before/After

**唯一目标**：删除伪 baseline，建立固定 commit 的真实对照。

**Owned files**

```text
soma_retargeter/robotics/v3/failure_analysis.py
scripts/extract_capability_failure_ledger.py
scripts/build_capability_before_after.py              # 新增
artifacts/retargeting_v3_step2_capability/baseline_summary.json
artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json
artifacts/retargeting_v3_step2_capability/before_after.json
tests/v3/test_capability_true_baseline_*.py
docs/retargeting_v3/STEP2_3_1_TRUE_BASELINE.md
docs/retargeting_v3/subagents/hardening_agent_a_handoff.md
```

**必须实现**

1. 使用 `git show 5ad5...:<path>` 读取 baseline；
2. commit 不可解析时 fail closed；
3. baseline model IDs 必须正好44；
4. baseline counts必须21/3/9/11；
5. transition matrix必须使用真实 baseline status；
6. 删除或禁止 `before = partial_passed if after capability_limited`；
7. 测试伪造 baseline 会失败；
8. 输出 11 个旧 algorithm failures 的真实 transition。

### Agent B — Raw Certificate Evidence and Math Recompute Inputs

**唯一目标**：让 certificate 由真实 solver evidence支持，不允许合成字段。

**Owned files**

```text
soma_retargeter/robotics/v3/projection_solver.py
soma_retargeter/robotics/v3/projection_certificate.py
soma_retargeter/robotics/v3/capability_projection.py
soma_retargeter/robotics/v3/chain_projection.py
soma_retargeter/robotics/v3/canonical_projection.py
tests/v3/test_certificate_raw_evidence_*.py
tests/v3/test_certificate_independent_recompute_*.py
tests/v3/test_seed_task_space_consensus_*.py
tests/v3/test_continuation_evidence_*.py
docs/retargeting_v3/subagents/hardening_agent_b_handoff.md
```

**必须实现**

1. 删除 `_red_team_kkt_certificate()`；
2. certificate schema v2；
3. 序列化 q/bounds/J/e/task gradient/prior gradient；
4. 序列化每个 seed final task vector；
5. task-space seed spread；
6. continuation final alpha=1 evidence；
7. primal/dual/complementarity真实计算；
8. 不把零 task gradient改成正数；
9. 不默认 residual_explained=True；
10. 不更改 solver thresholds；
11. 若只增加 evidence，不改变求解结果，必须证明 q/residual 与基线一致。

### Agent C — Motion-Class and Profile Status Policy

**唯一目标**：正确决定 exact/capability/failed status，stress不污染状态。

**Owned files**

```text
soma_retargeter/robotics/v3/capability_status.py       # 新增
soma_retargeter/robotics/v3/profile.py
soma_retargeter/robotics/v3/robot_zoo.py               # 仅 status enum/terminal set
soma_retargeter/robotics/v3/status_schema.py           # 可新增
tests/v3/test_motion_class_partition_*.py
tests/v3/test_capability_profile_status_*.py
tests/v3/test_stress_motion_status_isolation_*.py
tests/v3/test_baseline_exact_no_regression_*.py
docs/retargeting_v3/subagents/hardening_agent_c_handoff.md
```

**必须实现**

1. 固定 motion classes；
2. invariance exact-only；
3. stress status isolation；
4. class-specific certificate gates；
5. `passed` baseline不降级；
6. partial/negative路径不进入capability状态计算；
7. 删除 `_has_capability_limited_certificate()` 的 any-certificate语义；
8. profile failure必须列具体 motion/task/gate。

### Agent D — Clean Provenance, LFS and Artifact Protocol

**唯一目标**：保证 artifacts 来自远端可解析 clean source commit。

**Owned files**

```text
scripts/generate_capability_artifacts_clean.sh         # 新增
scripts/write_capability_provenance.py                 # 新增
artifacts/retargeting_v3_step2_capability/environment.json
artifacts/retargeting_v3_step2_capability/lfs_state.json
artifacts/retargeting_v3_step2_capability/commands.txt
artifacts/retargeting_v3_step2_capability/acceptance_ledger.json
tests/v3/test_capability_provenance_*.py
tests/v3/test_capability_lfs_state_*.py
docs/retargeting_v3/STEP2_3_1_REPRODUCIBILITY.md
docs/retargeting_v3/subagents/hardening_agent_d_handoff.md
```

**必须实现**

1. 两阶段 code commit / artifact commit；
2. clean detached worktree运行；
3. output先写外部临时目录；
4. source worktree运行前后clean；
5. source commit已push且可解析；
6. source commit是artifact commit祖先；
7. source commit之后无core code diff；
8. `git lfs pull` 和 `git lfs fsck`；
9. pointer count=0；
10. 保存LFS版本/object状态；
11. no absolute path；
12. acceptance ledger运行新的capability audit。

### Agent E — Full 44 Validation, Determinism and CI Integration

**唯一目标**：接入真实 status/certificate，完成44模型重算。

**Owned files**

```text
soma_retargeter/robotics/v3/validation.py
soma_retargeter/tools/validate_kinematic_profile_v3.py
artifacts/retargeting_v3_step2_capability/summary.json
artifacts/retargeting_v3_step2_capability/deterministic_rerun.json
artifacts/retargeting_v3_step2_capability/cross_format.json
artifacts/retargeting_v3_step2_capability/per_robot/
artifacts/retargeting_v3_step2_capability/per_task/
artifacts/retargeting_v3_step2_capability/failures/
tests/v3/test_full_44_hardened_validation_*.py
tests/v3/test_hardened_deterministic_rerun_*.py
tests/v3/test_hardened_cross_format_*.py
docs/retargeting_v3/STEP2_3_1_VALIDATION_REPORT.md
docs/retargeting_v3/subagents/hardening_agent_e_handoff.md
```

**必须实现**

1. 读取 Agent A真实 baseline；
2. before/after不再内部编造；
3. 44/44真实rerun；
4. deterministic比较status、q、task vector、certificate class/digest；
5. baseline 21 passed无降级；
6. baseline 3 partial无变化；
7. baseline 9 negative无变化；
8. 11旧algorithm failures精确转移；
9. G1 strict gate保持；
10. negative-control pair继续not_eligible；
11. summary计数满足本文件约束；
12. 若任何 hard gate失败，保留algorithm_failed，不得包装。

### Agent F — Independent Red Team and Final CI

**唯一目标**：独立攻击最终实现，不参与core修复。

**Owned files**

```text
scripts/audit_retargeting_v3_capability.py
.github/workflows/retargeting_v3_capability.yml
tests/v3/test_capability_acceptance_*.py
docs/retargeting_v3/STEP2_3_1_CAPABILITY_ACCEPTANCE.md
docs/retargeting_v3/subagents/capability_agent_f_red_team.md
artifacts/retargeting_v3_step2_capability/test_results/
artifacts/retargeting_v3_step2_capability/capability_audit.json
```

**必须先写失败测试覆盖**

1. fabricated before status；
2. unresolvable source commit；
3. wrong workflow branch；
4. wrong workflow artifact root；
5. missing pytest/junit/lfs_state；
6. hardcoded primal feasible；
7. hardcoded prior gradient zero；
8. zero task gradient被改成`1e-15`；
9. missing raw J/e/q/bounds；
10. KKT bool与独立重算不一致；
11. seed residual相同但task-space endpoint不同；
12. continuation未达到alpha=1；
13. stress-only limited错误改变status；
14. baseline passed降级；
15. deterministic只比较部分模型；
16. LFS pointer-only；
17. stale Agent F handoff；
18. stale Step2.2 acceptance ledger。

**最终 handoff 必须包含**

```text
final_head
source_code_commit
artifact_commit
workflow_run_id
pytest summary
live audit command
live audit PASS
LFS fsck PASS
remaining blockers = 0
verdict = PASS
```

如果最终仍有 blocker，handoff 必须写 BLOCKED，不能只改 acceptance 文档为 PASS。

---

## 12. Integrator 的固定执行顺序

不得自行改变顺序。

### Wave 0 — 冻结基线

1. 启动6个xhigh agents；
2. Agent A生成真实 baseline；
3. Agent F提交 failing red-team tests；
4. Integrator冻结 certificate schema v2 和 status policy；
5. 此时不得生成最终 artifacts。

### Wave 1 — Core evidence

并行：

```text
Agent B raw certificate evidence
Agent C motion/status policy
Agent D clean provenance scripts
```

### Wave 2 — 合并顺序

严格按：

```text
A → B → C → D
```

每次合并后运行对应 tests。

### Wave 3 — Full integration

1. Agent E rebase到 A/B/C/D；
2. 接入 validation；
3. 运行44模型开发态验证；
4. 不提交最终 artifacts；
5. Agent F第一次live audit；
6. blockers返给owner。

### Wave 4 — Code freeze

1. 所有core tests通过；
2. 创建 code commit `C`；
3. push `C`；
4. 记录 `C` SHA；
5. 从此禁止修改 core source。

### Wave 5 — Clean artifact generation

1. Agent D脚本从 `C`创建clean detached worktree；
2. LFS materialize/fsck；
3. full pytest；
4. full44 deterministic rerun；
5. artifacts输出到外部临时目录；
6. live audit临时目录；
7. copy到主worktree；
8. commit artifact commit `A`；
9. 验证 `C`是`A`祖先；
10. 验证`C..A`无core code变化。

### Wave 6 — Final red team / CI

1. Agent F在 `A`运行live audit；
2. 更新Agent F handoff；
3. push `A`；
4. GitHub Actions触发；
5. 记录workflow run ID；
6. 若CI失败，修复后重新code freeze和全量重跑；
7. CI green后才允许最终PASS。

---

## 13. Clean artifact generation 固定协议

### 13.1 Code freeze commit

假设：

```text
C=<code source commit>
```

必须先：

```bash
git push origin C:retargeting-v3-step2-capability-acceptance-hardening
git cat-file -e ${C}^{commit}
```

### 13.2 Detached worktree

```bash
WORKTREE=../.soma-retargeter-worktrees/capability-hardening-source
rm -rf "$WORKTREE"
git worktree add --detach "$WORKTREE" "$C"
cd "$WORKTREE"
git status --porcelain   # 必须为空
git lfs pull
git lfs fsck
```

### 13.3 外部输出目录

不得直接在clean source worktree下生成 tracked artifacts：

```bash
OUT=$(mktemp -d)
```

运行：

```bash
PYTHONPATH=. python -m pytest -q \
  --junitxml "$OUT/junit.xml" \
  | tee "$OUT/pytest.txt"

PYTHONPATH=. python -m soma_retargeter.tools.validate_kinematic_profile_v3 \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --output-dir "$OUT/artifacts" \
  --low-discrepancy-count 32 \
  --deterministic-rerun

PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py \
  --artifact-dir "$OUT/artifacts" \
  --numerical-artifact-dir artifacts/retargeting_v3_step2_numerical \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --write-report "$OUT/artifacts/capability_audit.json"
```

Source worktree运行后：

```bash
git status --porcelain   # 仍必须为空
```

### 13.4 Artifact commit

将 `$OUT/artifacts` 复制到主开发worktree后：

```bash
git add artifacts/retargeting_v3_step2_capability
git commit -m "artifacts: publish hardened capability acceptance evidence"
A=$(git rev-parse HEAD)

git merge-base --is-ancestor "$C" "$A"
```

检查 core drift：

```bash
git diff --name-only "$C..$A" -- \
  soma_retargeter \
  tests \
  scripts \
  .github
```

结果必须为空。若不为空，artifact无效，必须重新freeze和重跑。

允许 artifact commit修改：

```text
artifacts/retargeting_v3_step2_capability/**
docs/retargeting_v3/**/final-report-only files
```

---

## 14. 最终 artifacts 必须存在

```text
artifacts/retargeting_v3_step2_capability/
  environment.json
  lfs_state.json
  commands.txt
  acceptance_ledger.json
  baseline_summary.json
  baseline_failure_ledger.json
  before_after.json
  certificate_thresholds.json
  summary.json
  capability_matrix.json
  target_geometry_matrix.json
  deterministic_rerun.json
  cross_format.json
  deferred_snapshots.json
  source_inventory.json
  load_matrix.json
  semantic_matrix.json
  validation_checks.json
  capability_audit.json
  per_robot/*.json
  per_task/*.json
  failures/*.json
  test_results/
    pytest.txt
    junit.xml
    pytest_summary.json
```

### 14.1 `environment.json`

必须包含真实值：

```text
source_code_commit
source_code_commit_remote_resolvable=true
source_code_commit_is_artifact_commit_ancestor=true
source_worktree_clean_before_run=true
source_worktree_clean_after_run=true
core_diff_after_source_commit=[]
git_status_short=""
package versions
seed
```

### 14.2 `lfs_state.json`

必须包含：

```text
git_lfs_version
fsck_returncode=0
pointer_files_detected=[]
missing_lfs_objects=[]
materialized_snapshot_count=38
tracked_paths
```

### 14.3 `acceptance_ledger.json`

必须运行：

```text
scripts/audit_retargeting_v3_capability.py
```

不得引用旧 Step 2.2 audit。

### 14.4 `before_after.json`

必须记录：

```text
baseline_commit=5ad5...
baseline_artifact_root
baseline status
current status
transition
real source command
```

不得使用after status推断before status。

---

## 15. CI 文件必须精确修改

文件：

```text
.github/workflows/retargeting_v3_capability.yml
```

### 15.1 Push trigger

必须包含实际分支：

```yaml
push:
  branches:
    - retargeting-v3-step2-capability-acceptance-hardening
```

### 15.2 LFS

```yaml
- uses: actions/checkout@v4
  with:
    lfs: true

- name: Verify Git LFS
  run: |
    git lfs pull
    git lfs fsck
```

### 15.3 Artifact root

必须：

```yaml
RETARGETING_V3_CAPABILITY_ARTIFACTS: artifacts/retargeting_v3_step2_capability
```

Audit命令必须使用 capability root：

```bash
python scripts/audit_retargeting_v3_capability.py \
  --artifact-dir "$RETARGETING_V3_CAPABILITY_ARTIFACTS" \
  --numerical-artifact-dir artifacts/retargeting_v3_step2_numerical \
  --lock assets/robot_zoo/robot_zoo_lock.json
```

禁止继续 audit：

```text
artifacts/retargeting_v3_step2_assets44
```

### 15.4 Hosted jobs

至少包括：

```text
capability-synthetic-tests
capability-artifact-live-audit
lfs-snapshot-smoke
```

### 15.5 Full 44 job

5个fetch-only模型需要外部cache，因此 full44 可以是：

```text
self-hosted 或 workflow_dispatch
```

但最终 committed artifacts必须来自本文件规定的clean local full44 run。

### 15.6 Workflow run evidence

最终 Agent F handoff必须记录：

```text
workflow name
run ID
head SHA
conclusion=success
job conclusions
```

---

## 16. 必须新增的测试

### Baseline truth

- fixed commit读取成功；
- fixed commit不存在时失败；
- baseline counts不匹配时失败；
- fabricated before status失败；
- baseline passed降级失败。

### Motion/status

- stress-only limited仍是`passed`；
- ordinary limited是`capability_limited_passed`；
- invariance limited是`algorithm_failed`；
- missing certificate是`algorithm_failed`；
- failed certificate是`algorithm_failed`；
- partial不进入capability状态；
- negative不进入capability状态。

### Certificate recompute

- q越界但`primal_feasible=True`时audit失败；
- KKT sign错误时失败；
- raw J/e与gradient不一致时失败；
- prior gradient被写零时失败；
- rank residual decomposition不一致时失败；
- task gradient为真实零时允许；
- task gradient被人为改`1e-15`时不应成为通过依据；
- seed residual相同但endpoint不同，consensus失败；
- continuation final alpha<1失败；
- pointer-only LFS失败。

### Provenance

- source commit不可解析失败；
- source不是artifact ancestor失败；
- source后core drift失败；
- dirty source worktree失败；
- missing pytest/junit/lfs_state失败；
- stale acceptance ledger失败。

### Full regression

- baseline 21 passed保持；
- 3 partial保持；
- 9 negative保持；
- 11 failure transitions合法；
- deterministic 44/44；
- G1 strict equivalence保持；
- RPO保持terminal pass；
- no epsilon-only failure。

---

## 17. 测试命令——必须全部执行

### 17.1 Targeted hardening tests

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_capability_true_baseline_*.py \
  tests/v3/test_certificate_raw_evidence_*.py \
  tests/v3/test_certificate_independent_recompute_*.py \
  tests/v3/test_seed_task_space_consensus_*.py \
  tests/v3/test_continuation_evidence_*.py \
  tests/v3/test_motion_class_partition_*.py \
  tests/v3/test_capability_profile_status_*.py \
  tests/v3/test_stress_motion_status_isolation_*.py \
  tests/v3/test_capability_provenance_*.py \
  tests/v3/test_capability_lfs_state_*.py \
  tests/v3/test_capability_acceptance_*.py
```

### 17.2 V3 suite

```bash
PYTHONPATH=. python -m pytest -q tests/v3
```

### 17.3 Full repository suite

```bash
PYTHONPATH=. python -m pytest -q
```

必须保存：

```text
pytest.txt
junit.xml
pytest_summary.json
```

### 17.4 LFS

```bash
git lfs fsck
```

### 17.5 Live audit

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py \
  --artifact-dir artifacts/retargeting_v3_step2_capability \
  --numerical-artifact-dir artifacts/retargeting_v3_step2_numerical \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --write-report artifacts/retargeting_v3_step2_capability/capability_audit.json
```

最终：

```text
status=passed
failure_count=0
```

---

## 18. Red-team audit 必须独立检查

`scripts/audit_retargeting_v3_capability.py` 必须：

1. 读取真实 baseline commit；
2. 验证真实 transitions；
3. 独立重算所有超阈值 task residual；
4. 独立重算 Jᵀe；
5. 独立重算 projected gradient；
6. 独立重算 primal/dual/complementarity；
7. 独立重算 prior cancellation；
8. 独立重算 residual decomposition；
9. 独立重算 task-space seed spread；
10. 验证 continuation final alpha=1；
11. 验证 class-specific gates；
12. 验证 stress不影响status；
13. 验证invariance exact；
14. 验证44/44 deterministic；
15. 验证source commit；
16. 验证source→artifact ancestry；
17. 验证no core drift；
18. 验证pytest/junit/lfs_state；
19. 验证LFS materialized；
20. 验证workflow branch/root；
21. 验证Agent F最终handoff不是stale FAIL；
22. 验证acceptance ledger是本轮audit。

Audit 不得：

- 仅验证字段存在；
- 相信artifact中的`certified=true`；
- 相信artifact中的gate bool；
- 使用默认True补字段；
- 为当前artifact修改期望值；
- 忽略baseline exact downgrade。

---

## 19. 推荐 commit 顺序

必须使用小 commits：

1. `test: expose fabricated capability baseline transitions`
2. `feat: load immutable Assets44 baseline from git object`
3. `test: expose synthetic KKT and seed evidence fields`
4. `feat: serialize raw capability certificate evidence v2`
5. `fix: remove synthetic red-team KKT certificate fields`
6. `test: isolate stress motions from profile status`
7. `feat: add deterministic motion-class status policy`
8. `test: independently recompute certificate gates`
9. `feat: harden capability red-team audit`
10. `feat: add clean capability artifact generation protocol`
11. `test: validate true 44-model before-after transitions`
12. `ci: run capability audit on actual branch and artifact root`
13. `docs: publish six hardening handoffs`
14. `artifacts: publish hardened capability evidence`

禁止将所有修改压成一个大 commit。

---

## 20. 最终验收表

### Baseline

- [ ] baseline来自`5ad5...`；
- [ ] baseline counts=21/3/9/11；
- [ ] 无fabricated before；
- [ ] transitions合法；
- [ ] 21 baseline passed无降级；
- [ ] 3 partial无变化；
- [ ] 9 negative无变化。

### Certificate

- [ ] schema v2；
- [ ] raw J/e/q/bounds；
- [ ]真实task/prior gradient；
- [ ] primal/dual/complementarity真实计算；
- [ ] task-space seed consensus；
- [ ] continuation alpha=1；
- [ ] class-specific gates；
- [ ] 无synthetic KKT helper；
- [ ] 无默认True；
- [ ] 无`1e-15`造假。

### Status

- [ ] invariance exact-only；
- [ ] stress不影响status；
- [ ] passed>=21；
- [ ] capability_limited<=11；
- [ ] partial=3；
- [ ] negative=9；
- [ ] algorithm_failed=0；
- [ ] total=44。

### Determinism

- [ ] compared=44；
- [ ] matched=44；
- [ ] compare status；
- [ ] compare q；
- [ ] compare task vector；
- [ ] compare certificate class；
- [ ] compare certificate digest。

### Provenance/LFS

- [ ] source commit远端可解析；
- [ ] source是artifact ancestor；
- [ ] source后无core drift；
- [ ] source worktree before/after clean；
- [ ] LFS fsck PASS；
- [ ] pointer count=0；
- [ ] no meshes/fetch-only leakage；
- [ ] no absolute paths。

### Tests/CI

- [ ] targeted tests通过；
- [ ] tests/v3通过；
- [ ] full pytest通过；
- [ ] pytest.txt存在；
- [ ] junit.xml存在；
- [ ] pytest_summary.json存在；
- [ ] capability audit PASS；
- [ ] workflow实际分支触发；
- [ ] workflow audit capability root；
- [ ] GitHub Actions run success；
- [ ] workflow run ID记录。

### Process

- [ ] 6个xhigh handoffs完整；
- [ ] Agent F最终handoff为PASS；
- [ ] acceptance ledger为Step2.3.1；
- [ ] 未修改thresholds；
- [ ] 未添加robot特例；
- [ ] 未进入Step3。

---

## 21. 完成定义

最终只允许写：

```text
Step 2.3.1 Capability Acceptance Hardening: PASS
```

或：

```text
Step 2.3.1 Capability Acceptance Hardening: BLOCKED
```

PASS 必须同时满足：

```text
真实baseline
真实transition
21 exact baseline passes无回归
11 old failures全部转exact或有效limited
algorithm_failed=0
44/44 deterministic
所有certificate可独立重算
source commit可解析
clean provenance
LFS fsck PASS
full tests PASS
live audit PASS
CI PASS
Agent F final PASS
```

任一项不满足必须写 BLOCKED。

**即使本轮 PASS，也必须在此停止。不得自动进入 Step 3，等待用户重新审查。**
