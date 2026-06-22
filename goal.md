# Goal — Step 2.1 数值运动学修复：Engine Jacobian、可达秩与目标坐标系

> **Codex 执行契约**
>
> 当前分支：`retargeting-v3-step2-numerical-core-fix`  
> 基线分支：`retargeting-v3-step2-finalize`  
> 基线提交：`4a93386e41c31fa0ab55533b7f6d891042187bc7`  
> 本轮阶段：**Step 2.1 Numerical Core Correction**  
>
> 本轮必须一次创建并持续高强度使用 **6 个 xhigh 专业 subagents**。主 Codex 是 Integrator，负责冻结接口、合并、全量可用模型重跑和最终诚实判定。
>
> **本轮暂不处理模型下载、source-unavailable、许可证、缺失资产、缺 verified map 或 pycollada 安装问题。不要定位或补齐失败资产。**
>
> **不得进入 Step 3；不得修改生产 `NewtonPipeline`；不得做 whole-body IK、contact、collision、temporal smoothing 或逐机器人权重调参。**

---

## 0. 本轮为什么存在

当前 artifacts 的总状态是：

```text
passed                   7
negative_control_passed  4
algorithm_failed        14
semantic_failed          3
model_load_failed        2
source_unavailable      16
```

真正进入正向人形算法判定的模型中，多个完整机器人并不是 target projection 失败，而是统一被如下原因卡住：

```text
epsilon stability gate failed for task(s): left_hand/right_hand
```

已知代表：

- RoboParty RPO；
- Booster T1；
- Unitree H1；
- ROBOTIS OP3；
- 其他当前 `algorithm_failed` 的可加载机器人。

RPO 的 canonical hand projection 已经能得到很小 residual，但因某些物理上接近零的 hand-position Jacobian 列在 float32 Newton FK 中出现 epsilon-halving 噪声，整个机器人仍被判为失败。

当前实现的问题包括：

1. finite difference 是主 Jacobian，engine Jacobian 只是诊断；
2. Newton/Warp float32 与 MuJoCo float64 使用相同数量级的差分步长；
3. 只比较 `h` 和 `h/2`，默认更小步长更好；
4. 理论零列没有独立分类；
5. hand-position task 同时要求 translation 和 rotation block 都稳定；
6. 只要任意 sample 任意 column 不稳定，整个 chain 失败：`unstable == 0`；
7. rank SVD threshold 没有包含数值误差；
8. 未以 rank、子空间和 engine/FD 一致性作为主要 gate；
9. canonical capability motions 被当作时间序列，跨 motion 使用 continuity prior；
10. position residual normalization 使用极限姿态最大位移，不是链长；
11. source→robot rotation delta 混用了 parent-frame alignment 与 child/body-frame delta。

本轮只修这些数学和数值问题。

---

## 1. 本轮唯一目标

建立下列可信数据流：

```text
runtime model
→ engine site/body Jacobian
→ general relative Jacobian
→ backend-aware multi-scale finite-difference validator
→ zero-column / nonzero-column / mismatch classification
→ uncertainty-aware rank and reachable subspace
→ task-specific statistical stability gate
→ corrected parent-frame rotation transfer
→ independent canonical capability projection
→ available-model before/after validation
```

完成后必须能够回答：

1. 某一 Jacobian 列是真正的零列，还是数值不稳定？
2. engine Jacobian 与 finite difference 是否在正确坐标系内一致？
3. rank 和 reachable subspace 是否在多姿态下稳定？
4. RPO 单 yaw 腰和无腕 hand position 是否因真实能力而启用/禁用，而不是因 epsilon 噪声失败？
5. 仍然失败的机器人，到底是 projection residual、semantic geometry、模型加载还是 source 问题？

---

## 2. 硬性范围边界

### 2.1 本轮包含

- Engine Jacobian 统一接口；
- 通用 relative site Jacobian；
- Newton/MuJoCo backend Jacobian；
- dtype-aware finite difference；
- Richardson / multi-scale stability；
- physical/numerical zero-column classification；
- task-specific stability；
- uncertainty-aware SVD rank；
- principal-angle/projector subspace metric；
- statistical sample gates；
- parent-frame rotation transfer correction；
- canonical capability motions 独立求解；
- 单独 temporal sequence benchmark；
- chain-length residual normalization；
- 当前 source-available 且能进入 profile 的全部模型重跑；
- before/after artifacts；
- tests、CI、red-team。

### 2.2 本轮明确不包含

- 下载或定位 16 个 `source_unavailable` 模型；
- 添加或修改 Robot Zoo source URLs；
- 许可证处理；
- 添加新的机器人资产；
- 为 Berkeley/H1 等缺失项补 verified semantic map；
- 安装或处理 `pycollada`；
- 修改 `assets/robot_zoo/robot_zoo_manifest.json`；
- 修改 `assets/robot_zoo/semantic_maps/`；
- 把 semantic_failed/model_load_failed/source_unavailable 强行改为 passed；
- 生产 pipeline；
- whole-body IK；
- contact/collision/temporal runtime；
- robot-specific mathematical branches；
- 调 hand/foot/chest 权重。

这些状态必须原样保留并在 summary 中与 numerical-core 结果分开。

---

## 3. 六个 xhigh Subagents

每个 agent 必须：

- 使用独立 worktree/branch；
- 只修改 owned files；
- 先写失败测试或数学 oracle；
- 提交小 commits；
- 保存 handoff；
- handoff 记录 reasoning strength=`xhigh`、commit、文件、命令、测试、数值结果、风险和未完成项；
- 不得降低 gate 来制造通过。

### Agent A — Engine Jacobian & Runtime Twist Conventions

**职责**

- MuJoCo engine site Jacobian；
- Newton engine body/site Jacobian；
- point-offset twist conversion；
- 通用 relative Jacobian；
- backend dtype/machine epsilon metadata；
- synthetic analytic cross-check。

**Owned files**

```text
soma_retargeter/robotics/v3/model_adapter.py
soma_retargeter/robotics/v3/engine_jacobian.py        # 新增
soma_retargeter/robotics/v3/spatial.py                # 仅 twist/adjoint helper
tests/v3/test_engine_jacobian_*.py
tests/v3/test_relative_jacobian_*.py
docs/retargeting_v3/subagents/numerical_agent_a_handoff.md
```

**必须实现的公式**

设 reference site 为 A，target site 为 B：

```text
p_AB = R_A^T (p_B - p_A)
```

若 engine Jacobians 都表达在 world frame：

```text
Jp_AB = R_A^T (Jp_B - Jp_A)
        + [p_AB]_x R_A^T Jw_A

Jw_AB = R_A^T (Jw_B - Jw_A)
```

注意：

- `Jp_A/Jp_B` 必须是 semantic site point，不只是 body origin；
- Newton body spatial Jacobian必须按 local site offset转换到点速度；
- cross-branch relative task必须支持；
- 不能只在 `reference == LCA` 时正确；
- 明确记录 angular/linear block convention；
- synthetic 解析 Jacobian必须对齐。

**输出接口**

```python
@dataclass
class EngineRelativeJacobian:
    translation: np.ndarray
    rotation: np.ndarray
    backend: str
    scalar_dtype: str
    source: str
    finite: bool
    convention: str
```

### Agent B — Multi-scale Finite Difference & Uncertainty

**职责**

- backend-aware epsilon；
- `2h, h, h/2` 多尺度差分；
- plateau selection；
- Richardson extrapolation；
- error estimate；
- zero-column classification；
- engine-vs-FD validation。

**Owned files**

```text
soma_retargeter/robotics/v3/numerical_jacobian.py
soma_retargeter/robotics/v3/fd_uncertainty.py          # 新增
tests/v3/test_fd_uncertainty_*.py
tests/v3/test_zero_jacobian_columns_*.py
docs/retargeting_v3/subagents/numerical_agent_b_handoff.md
```

**步长原则**

不得继续使用统一的：

```text
1e-4 * joint_span，然后直接取 h/2
```

使用：

```text
h0 = c * eps_backend^(1/3) * coordinate_scale
```

其中：

- `eps_backend` 来自实际 scalar dtype；
- revolute coordinate scale 使用安全关节范围和当前量级；
- prismatic scale 同时考虑关节范围与 chain characteristic length；
- 对 `2h0, h0, h0/2` 求中心差分；
- 选择 truncation/roundoff 平衡最好的平台；
- 不允许按机器人名字选择 epsilon。

**Richardson**

中心差分在稳定区可使用：

```text
J* = (4 J(h/2) - J(h)) / 3
E  ≈ ||J(h/2) - J(h)|| / 3
```

若 `h/2` 比 `h` 更差，应选择 `2h/h` 平台，不得强制最小步长。

**列分类**

```text
stable_nonzero
numerically_zero
engine_fd_mismatch
unstable_roundoff
unstable_nonsmooth
nonfinite
```

理论/数值零列不得标成 unstable。零列判断必须同时参考：

- engine norm；
- FD norm；
- estimated noise floor；
- chain characteristic length；
- backend dtype。

### Agent C — Reachability Rank & Subspace Gates

**职责**

- engine Jacobian 作为 rank 主结果；
- FD uncertainty 作为验证；
- task-specific blocks；
- uncertainty-aware SVD threshold；
- multi-pose rank distribution；
- reachable subspace/projector stability；
- statistical gate 替代 one-strike gate。

**Owned files**

```text
soma_retargeter/robotics/v3/reachability.py
soma_retargeter/robotics/v3/reachability_metrics.py     # 新增
tests/v3/test_rank_uncertainty_*.py
tests/v3/test_subspace_stability_*.py
docs/retargeting_v3/subagents/numerical_agent_c_handoff.md
```

**Task-specific block**

```text
hand position       → translation
foot position       → translation
torso orientation   → rotation
future orientation  → rotation
pose task            → translation + rotation separately
```

不允许 hand-position 因 rotation block 噪声失败。

**Rank threshold**

不得只用：

```text
max(1e-8, 1e-5*sigma1)
```

应使用：

```text
tau_rank = max(
    abs_floor_backend,
    rel_tol * sigma1,
    k_uncertainty * jacobian_noise_norm,
)
```

**Subspace metric**

对 retained left singular vectors U：

```text
d_projector = || U1 U1^T - U2 U2^T ||_2
```

同时记录 principal angles。

**统计 gate**

禁止：

```python
epsilon_stability_gate_passed = unstable == 0
```

新的默认 gate 必须通过 synthetic calibration 固定，不得逐机器人调整。最低要求：

```text
no NaN/Inf
relevant rank agreement rate >= 95%
relevant stable sample rate  >= 95%
projector distance p95       <= 0.05
nonzero engine/FD normalized error p95 <= 0.02
```

数值零列单独统计，不计作不稳定。

如果 synthetic/backend 证据表明某阈值应改变，必须全局改变并在 ADR 中说明，不能按机器人设置。

### Agent D — Rotation Transfer, Canonical Independence & Normalization

**职责**

- 修正 source→robot relative rotation frame convention；
- capability canonical motions 独立求解；
- 单独 temporal sequences；
- chain-length normalization；
- projection quality metrics。

**Owned files**

```text
soma_retargeter/robotics/v3/rest_frames.py
soma_retargeter/robotics/v3/target_builder.py
soma_retargeter/robotics/v3/canonical_projection.py
soma_retargeter/robotics/v3/chain_projection.py
tests/v3/test_rotation_transfer_frames_*.py
tests/v3/test_canonical_independence_*.py
tests/v3/test_chain_length_normalization_*.py
docs/retargeting_v3/subagents/numerical_agent_d_handoff.md
```

**统一使用 parent-frame rotation delta**

设：

```text
R_h0 = source neutral child relative to source parent
R_h  = source current child relative to source parent
R_r0 = robot neutral child relative to robot parent
A    = source-parent coordinates → robot-parent coordinates
```

定义：

```text
Delta_R_h_parent = R_h R_h0^T
Delta_R_r_parent = A Delta_R_h_parent A^T
R_r_target        = Delta_R_r_parent R_r0
```

不得继续混用：

```text
R_h0^T R_h
```

与 parent-frame alignment A。

必须新增非 identity rest-frame、oblique axis、rotated parent 测试；identity T-pose 测试不够。

**Canonical capability benchmark**

每个 canonical motion 独立：

```text
q_seed = neutral_q
previous_q = None
```

不能让 `arms_forward` 的解依赖它前面是否运行了 `torso_yaw`。

**Temporal benchmark**

另外定义真实序列：

```text
neutral → arms_forward → overhead → neutral
neutral → squat → stand
neutral → step → support
```

只有这些序列启用 continuity prior。

**Position normalization**

使用：

```text
normalized residual = ||p(q)-p*|| / max(L_chain, eps)
```

`L_chain` 是完整 neutral kinematic path 的 segment-length sum。

不得再用“所有关节同时到极限时 endpoint 最大位移”作为 normalization。

### Agent E — Available-model Before/After Matrix

**职责**

- 只重跑当前已有 source、能加载、已有 semantic map 且进入 profile 的模型；
- 建立 baseline vs corrected 对比；
- RPO/H1/Booster/OP3 专项；
- projection quality gates；
- failure taxonomy。

**Owned files**

```text
soma_retargeter/robotics/v3/profile.py
soma_retargeter/robotics/v3/validation.py
soma_retargeter/tools/validate_kinematic_profile_v3.py
artifacts/retargeting_v3_step2_numerical/
tests/v3/test_available_model_numerical_matrix_*.py
docs/retargeting_v3/STEP2_NUMERICAL_CORE_REPORT.md
docs/retargeting_v3/subagents/numerical_agent_e_handoff.md
```

**模型选择规则**

从基线 artifacts 自动选择：

```text
status in {passed, partial_passed, algorithm_failed}
```

排除：

```text
semantic_failed
model_load_failed
source_unavailable
license_blocked
negative controls（只做回归 smoke）
```

不得在代码中手写一份漂移的模型清单。

**Before/after**

每台模型保存：

- baseline status/reason；
- corrected status/reason；
- relevant task metric；
- zero-column count；
- real unstable count；
- engine/FD error；
- rank agreement；
- subspace distance；
- canonical residual distribution；
- runtime。

**关键 gate**

1. 不得有模型仅因 `near-zero FD column` 失败；
2. 不得有模型仅因 irrelevant rotation/translation block 失败；
3. 不得有 `unstable == 0` one-strike gate；
4. RPO 不得继续仅因 hand epsilon stability 失败；
5. H1/Booster/OP3 不得继续仅因 hand epsilon stability 失败；
6. 当前已 passed 的模型不得回归为 algorithm_failed；
7. 仍 algorithm_failed 的模型必须有非 epsilon-only 的可解释 gate，例如：
   - projection residual 超标；
   - engine/FD true mismatch；
   - nonfinite；
   - rank/subspace inconsistent；
   - semantic/site geometry problem。

不要求通过篡改 status 把所有 14 台变绿。

### Agent F — Independent Red Team, CI & Final Verdict

**职责**

- 先写失败测试；
- 查 tolerance cheating；
- 查 robot-name special case；
- 查 engine Jacobian 坐标错误；
- 查 zero-column 被隐藏；
- 查 rank gate 被过度放宽；
- 查 canonical order dependence；
- 查旧 artifacts 污染；
- CI；
- 最终 `Numerical Core PASS/BLOCKED`。

**Owned files**

```text
scripts/audit_retargeting_v3_numerical_core.py
.github/workflows/retargeting_v3_numerical_core.yml
tests/v3/test_numerical_core_acceptance_*.py
docs/retargeting_v3/STEP2_NUMERICAL_ACCEPTANCE.md
docs/retargeting_v3/subagents/numerical_agent_f_red_team.md
artifacts/retargeting_v3_step2_numerical/test_results/
```

**红队必须检查**

- 是否只是把 `stability_rtol` 调到很大；
- 是否直接忽略所有小列；
- 是否以机器人名字选择 epsilon/tolerance；
- 是否用 finite difference 假装 engine Jacobian；
- 是否 relative Jacobian 少了 reference twist 项；
- 是否 point offset 未进入 engine Jacobian；
- 是否 rank threshold 仍低于数值噪声；
- 是否 canonical results 依赖 motion order；
- 是否现有 passed 模型回归；
- 是否把 unavailable/semantic/load failure 算入 numerical pass；
- 是否修改旧 audit 以掩盖失败。

---

## 4. Integrator 的接口冻结

Wave 1 后必须冻结：

```python
EngineRelativeJacobian
FiniteDifferenceEstimate
JacobianColumnClassification
JacobianValidationReport
ReachabilityReportV2
ProjectionNormalization
NumericalModelComparison
```

推荐接口：

```python
@dataclass
class JacobianColumnClassification:
    dof: int
    relevant_block: str
    classification: Literal[
        "stable_nonzero",
        "numerically_zero",
        "engine_fd_mismatch",
        "unstable_roundoff",
        "unstable_nonsmooth",
        "nonfinite",
    ]
    engine_norm: float
    fd_norm: float
    error_estimate: float
    normalized_error: float | None
```

```python
@dataclass
class JacobianValidationReport:
    engine: EngineRelativeJacobian
    finite_difference: FiniteDifferenceEstimate
    columns: list[JacobianColumnClassification]
    relevant_rank_engine: int
    relevant_rank_fd: int
    rank_agreement: bool
    projector_distance: float | None
    passed: bool
```

旧 JSON 字段可兼容读取，但新 artifacts 必须使用明确 schema version。

---

## 5. Synthetic 数学测试

至少新增以下 fixtures：

1. revolute joint，site 不在 axis 上；
2. revolute joint，site 正好在 axis 上：position Jacobian 真零、rotation 非零；
3. prismatic joint；
4. oblique axis；
5. free joint；
6. ball joint；
7. reference 和 target 在 LCA 两侧；
8. reference site 有 local offset；
9. target site 有 local offset；
10. fixed/rank-zero torso；
11. yaw-only torso；
12. 5DoF no-wrist arm；
13. float32-noise simulated FK；
14. near-singular but rank-stable chain；
15. truly nonsmooth/discontinuous fake adapter。

每个 fixture 必须有 analytic or engine oracle。

---

## 6. Available-model 专项验收

### 6.1 RPO

必须验证：

- torso rotation rank=1；
- hand-position chain 包含完整 shoulder/elbow；
- elbow-yaw 对 hand position 的近零列被分类为 `numerically_zero`，而不是 unstable；
- hand rotation block 不影响 hand-position gate；
- engine/FD relevant-block 一致；
- 15 个 canonical motions 全部独立运行；
- `arms_forward`、`overhead_reach`、`elbow_bend` 有真实 residual；
- RPO 最终不能仅因 epsilon stability 失败；
- 如果仍失败，必须是新的非 epsilon-only 证据。

### 6.2 H1 / Booster / OP3

这些当前代表性模型不得继续仅因 hand epsilon stability 失败。

OP3 fixed/zero-DoF torso仍应：

- rank-zero；
- 保留 unreachable torso demand；
- 不向 legs 泄漏。

### 6.3 当前 passed 模型

所有基线 `passed/partial_passed` 模型：

- status 不得回归；
- rank 不得无解释改变；
- canonical residual 不得显著恶化；
- deterministic hash/schema 更新需有迁移说明。

---

## 7. Projection quality gate

Solver `converged=True` 不是自动 pass。

定义：

```text
rho_p = ||p_projected - p_desired|| / L_chain
rho_R = ||log(R_projected^T R_desired)||
```

全局初始 gate（可由 synthetic/backend calibration 统一调整，但不得逐机器人调整）：

```text
neutral position absolute error       < 1 mm
ordinary hand position rho_p          < 0.05
ordinary foot position rho_p          < 0.03
compatible torso residual/demand      < 0.01
```

对于：

```text
extreme_but_valid_joint_limit_stress
```

不强制可达，但必须：

- 保存 unreachable demand；
- 无 NaN/Inf；
- 不越 joint limits；
- status 不伪造为零 residual。

若这些初始 gate 被修改，必须在 ADR 中用跨模型 residual 分布说明，不能为了某台机器人单独改。

---

## 8. 新 artifacts

不要覆盖旧 Step 2 artifacts。新目录：

```text
artifacts/retargeting_v3_step2_numerical/
  environment.json
  commands.txt
  baseline.json
  summary.json
  before_after.json
  threshold_calibration.json
  per_robot/
  per_chain/
  failures/
  test_results/
```

`summary.json` 必须分别统计：

```text
manifest_total
source_available
load_success
profile_eligible
baseline_pass
corrected_pass
baseline_algorithm_failed
corrected_algorithm_failed
epsilon_only_failures_before
epsilon_only_failures_after
semantic_failed_unchanged
model_load_failed_unchanged
source_unavailable_unchanged
```

不得再只给一个 `11/46`。

---

## 9. Acceptance Gates

### Engine Jacobian

- [ ] MuJoCo engine relative Jacobian通过 synthetic oracle；
- [ ] Newton engine relative Jacobian通过 synthetic oracle；
- [ ] site point offsets正确；
- [ ] cross-branch relative formula正确；
- [ ] finite all。

### FD uncertainty

- [ ] dtype-aware h；
- [ ] 2h/h/h2；
- [ ] plateau selection；
- [ ] Richardson/error estimate；
- [ ] physical zero classification；
- [ ] 不按机器人调 epsilon。

### Reachability

- [ ] task-specific block；
- [ ] uncertainty-aware rank；
- [ ] rank agreement；
- [ ] subspace projector metric；
- [ ] statistical gate；
- [ ] 删除 `unstable == 0` 语义。

### Target/projection

- [ ] parent-frame rotation delta；
- [ ] non-identity rest-frame tests；
- [ ] canonical motions独立；
- [ ] temporal sequences分离；
- [ ] chain-length normalization；
- [ ] residual quality gate。

### Available models

- [ ] 所有 baseline algorithm_failed 可用模型都有 before/after；
- [ ] RPO 不再 epsilon-only fail；
- [ ] H1/Booster/OP3 不再 epsilon-only fail；
- [ ] epsilon-only failures after = 0；
- [ ] baseline passed模型无回归；
- [ ] remaining failures都有非 epsilon-only evidence。

### Process

- [ ] 6个xhigh handoffs；
- [ ] full tests通过；
- [ ] new numerical audit PASS；
- [ ] CI有结果；
- [ ] worktree clean；
- [ ] artifacts可复现；
- [ ] 未改资产/manifest/semantic maps；
- [ ] 未进入Step3。

---

## 10. 明确禁止的捷径

以下任一项都算失败：

- 仅将 `stability_rtol` 从0.05改成0.5；
- 只要列很小就无条件忽略；
- 只检查 RPO；
- 机器人名字分支；
- 将 `unstable == 0` 改成一个任意很大的比例阈值而无数学证据；
- 用 FD 结果冒充 engine Jacobian；
- relative Jacobian 不减 reference Jacobian；
- 忽略 semantic site local offset；
- hand position仍检查rotation block；
- rank threshold低于估计noise floor；
- canonical benchmark继续跨动作使用continuity prior；
- 以solver converged代替projection quality；
- 把semantic/load/source failure改成algorithm pass；
- 下载或添加资产；
- 修改manifest或semantic maps；
- 修改生产pipeline；
- 删除失败测试；
- 修改旧audit来掩盖结果。

---

## 11. 推荐提交顺序

1. `test: add analytic relative Jacobian fixtures`
2. `feat: add engine relative Jacobian adapters`
3. `test: expose float32 near-zero FD false failures`
4. `feat: add backend-aware multi-scale FD uncertainty`
5. `feat: classify numerical zero Jacobian columns`
6. `feat: add task-specific rank and subspace gates`
7. `fix: use parent-frame rotation transfer`
8. `fix: separate capability motions from temporal sequences`
9. `fix: normalize endpoint residuals by chain length`
10. `test: rerun RPO H1 Booster OP3 numerical gates`
11. `test: run all available profile models before-after`
12. `ci: add numerical core acceptance workflow`
13. `docs: add six handoffs and numerical report`
14. `artifacts: publish reproducible numerical-core evidence`

不要压成单个大提交。

---

## 12. Codex 启动步骤

1. `git status --short`
2. `git branch --show-current`
3. `git rev-parse HEAD`
4. 确认分支：`retargeting-v3-step2-numerical-core-fix`
5. 确认祖先包含：`4a93386e41c31fa0ab55533b7f6d891042187bc7`
6. 激活正确 conda 环境；
7. 记录实际 Python/MuJoCo/Newton/Warp/NumPy/SciPy 版本；
8. 一次创建6个xhigh subagents；
9. 保存 baseline artifacts；
10. Wave 1 数学审计与接口冻结；
11. A/B/C/D 并行实现；
12. E 构建 before/after；
13. F 红队；
14. blockers 返回对应 owner 修复；
15. full tests；
16. 所有 available profile models重跑；
17. clean artifacts；
18. push全部commits；
19. 停止，不处理资产，不进入Step3。

---

## 13. 完成定义

最终报告只能使用：

```text
Step 2.1 Numerical Core: PASS
```

或：

```text
Step 2.1 Numerical Core: BLOCKED
```

`PASS` 必须同时满足：

- [ ] engine relative Jacobian成为主能力依据；
- [ ] FD成为独立 uncertainty validator；
- [ ] float32/float64使用backend-aware epsilon；
- [ ] near-zero columns不再误判；
- [ ] task-specific stability生效；
- [ ] rank/subspace gate生效；
- [ ] parent-frame rotation transfer正确；
- [ ] canonical capability motions独立；
- [ ] chain-length normalization生效；
- [ ] epsilon-only failures after=0；
- [ ] RPO不再epsilon-only fail；
- [ ] H1/Booster/OP3不再epsilon-only fail；
- [ ] baseline passed模型无回归；
- [ ] remaining failures均有真实非epsilon证据；
- [ ] all tests通过；
- [ ] numerical audit PASS；
- [ ] 6个xhigh handoffs完整；
- [ ] CI有结果；
- [ ] clean artifacts；
- [ ] 未改资产、manifest、semantic maps；
- [ ] 未进入Step3。

即使 Step 2.1 PASS，原始 Full Step 2 仍可能因 source-unavailable、semantic map、load dependency 等资产问题保持 BLOCKED；不得在本轮将其包装成全量46台通过。
