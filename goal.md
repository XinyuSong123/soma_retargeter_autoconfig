# Goal — Step 2.3：Capability-Aware Closest-Reachable Projection

> **Codex 执行契约**
>
> 当前分支：`retargeting-v3-step2-capability-projection-fix`  
> 基线分支：`retargeting-v3-step2-assets-vendored`  
> 基线提交：`5ad5a001c445c525d4c8bbaf6339dec5c5c2c719`  
> 当前阶段：**Step 2.3 Capability Projection Closure**  
>
> 本轮必须一次创建并持续高强度使用 **6 个 xhigh 专业 subagents**。主 Codex 是 Integrator，负责接口冻结、冲突解决、全量验证、clean artifacts 和最终诚实判定。
>
> **Git LFS 已正式接受。必须执行 `git lfs pull` 和 `git lfs fsck`，不得在 pointer-only 文件上运行模型验证。LFS 不得用于提交 meshes、完整上游仓库或 fetch-only 模型。**
>
> **不得进入 Step 3；不得修改生产 `NewtonPipeline`；不得调低/放宽 Step 2.1 已冻结的 numerical epsilon、rank、subspace 或 projection residual thresholds；不得按机器人名称写数学特例。**

---

## 0. 开始前必须执行

```bash
git status --short
git branch --show-current
git rev-parse HEAD

git lfs install
git lfs pull
git lfs fsck
```

必须确认：

```text
branch = retargeting-v3-step2-capability-projection-fix
HEAD ancestor includes 5ad5a001c445c525d4c8bbaf6339dec5c5c2c719
worktree clean
LFS objects materialized
```

必须阅读：

1. `/goal.md`
2. `docs/retargeting_v3/ASSET_POLICY.md`
3. `docs/retargeting_v3/STEP2_NUMERICAL_ACCEPTANCE.md`
4. `docs/retargeting_v3/STEP2_ASSETS44_ACCEPTANCE.md`
5. `docs/retargeting_v3/STEP2_ASSETS44_VALIDATION_REPORT.md`
6. `docs/retargeting_v3/subagents/numerical_agent_f_red_team.md`
7. `docs/retargeting_v3/subagents/assets44_agent_f_red_team.md`
8. `artifacts/retargeting_v3_step2_assets44/summary.json`
9. `artifacts/retargeting_v3_step2_assets44/failures/*.json`
10. `assets/robot_zoo/robot_zoo_lock.json`

不得只读总结文档后开始修改 solver。

---

## 1. 当前真实基线

### 1.1 Asset / loader / semantics 已闭环

```text
manifest public sources       46
in-scope models               44
deferred snapshots             2
committed snapshots           38
fetch-only cached models       5
project-local RPO              1
source_unavailable             0
model_load_failed              0
semantic_failed                0
```

两个永久 deferred IDs：

```text
berkeley_humanoid_urdf
romeo_urdf
```

它们不进入本轮算法 gates，也不得重新加入。

### 1.2 当前 terminal results

```text
passed                         21
partial_passed                  3
negative_control_passed         9
algorithm_failed               11
terminal pass total            33
in-scope total                 44
```

当前 deterministic rerun：

```text
compared = 33
matched  = 33
```

### 1.3 当前 11 个真实 algorithm failures

```text
atlas_drc_urdf
atlas_v4_urdf
booster_t1_mjcf
booster_t1_urdf
fourier_n1_mjcf
jaxon_urdf
mujoco_humanoid_mjcf
pal_talos_mjcf_direct
robotis_op3_mjcf
talos_urdf
valkyrie_urdf
```

这些模型已经：

- source available；
- runtime load passed；
- verified semantics or valid full profile map；
- engine Jacobian / FD numerical core available；
- 不再有 epsilon-only failure。

因此本轮不允许继续把失败归因于资产、loader、semantic scaffolding 或 epsilon。

### 1.4 当前跨格式证据

- G1 same-source URDF → canonical MJCF strict equivalence 已通过；
- Assets44 CI 和 numerical-core CI 已通过；
- 当前 Assets44 CI 主要是 static/audit，不是 44-model 从零完整重算；
- Cassie 等 negative-control pair 不应被正向 humanoid cross-format gate 误判。

---

## 2. 本轮唯一目标

把当前单一的：

```text
residual exceeds exact threshold → algorithm_failed
```

升级为数学上可审计的：

```text
canonical target
→ deterministic continuation solve
→ engine-Jacobian bounded optimum
→ reachable tangent/subspace decomposition
→ joint-limit/KKT optimality certificate
→ exact_reachable | capability_limited | solver_failed | invalid_target
→ capability-aware profile status
```

本轮必须区分：

1. **Exact reachable**：目标在当前全局 residual gate 内；
2. **Capability limited**：目标超出形态、可达子空间或 joint limits，但 solver 找到了可证明的最近可达解；
3. **Solver failed**：残差仍含显著可达方向分量、KKT 不满足、seed 不一致或数值失败；
4. **Invalid target geometry**：semantic site、frame、segment length、target construction 本身错误；
5. **Unsupported task**：rank-zero 或缺失任务能力，但 demand 被诚实保留且没有泄漏到错误链。

最终目标不是通过改阈值把 11 台染绿，而是让每个失败都得到正确的数学结论。

---

## 3. 本轮成功后的状态语义

新增 terminal pass 状态：

```text
capability_limited_passed
```

状态定义：

### `passed`

- 所有 required ordinary tasks 达到现有 exact residual gates；
- numerical / rank / subspace gates 通过；
- 无 invalid target 或 solver failure。

### `capability_limited_passed`

- 至少一个 ordinary/extended task 超过 exact residual gate；
- 但每个超标 task 都有完整 closest-reachable certificate；
- compatible/reachable 分量已被求解；
- 剩余 residual 主要属于 tangent-orthogonal、rank-incompatible 或 active joint-limit demand；
- 无 NaN、无 limit violation、无错误链泄漏；
- deterministic seeds 与 rerun 一致。

### `partial_passed`

- 结构上缺失 required semantics，例如无 arms；
- 由 verified partial expectation 支持；
- 不与 capability-limited 混用。

### `algorithm_failed`

只允许：

- solver nonfinite；
- engine/FD mismatch；
- rank/subspace inconsistent；
- target geometry invalid；
- projected gradient/KKT 不满足；
- deterministic seed consensus 失败；
- residual 中仍有明显 reachable/tangent component；
- ordinary task 的最近可达解无法被认证。

不得再使用 generic：

```text
compiler recorded algorithm failures
```

---

## 4. 核心数学契约

### 4.1 Position task

定义链长归一化 residual：

```text
e_p(q) = (p(q) - p_desired) / L_chain
```

其中 `L_chain` 仍使用 Step 2.1 已冻结的 neutral full-chain length。

### 4.2 Orientation task

```text
e_R(q) = log(R(q)^T R_desired) / pi
```

不得改回 Euler angles。

### 4.3 Relevant engine Jacobian

```text
position task    → J = J_translation / L_chain
orientation task → J = J_log_rotation / pi
```

Orientation residual Jacobian必须正确处理 SO(3) log 的 Jacobian，不得简单假设大角度时 `de/dq = J_omega`。

必须用 FD 对 analytic residual Jacobian 进行独立 cross-check。

### 4.4 Reachable tangent projector

对 relevant engine Jacobian：

```text
J = U S V^T
P = U_r U_r^T
```

rank threshold 必须继续使用 Step 2.1 uncertainty-aware rank，不得重新改成固定 `1e-8`。

对最终 residual：

```text
e_parallel = P e
e_orthogonal = (I - P) e
```

记录：

```text
reachable_residual_norm
orthogonal_residual_norm
reachable_residual_fraction
orthogonal_residual_fraction
```

### 4.5 Bounded KKT / projected-gradient certificate

目标函数：

```text
f(q) = 0.5 * ||e_task(q)||^2 + explicit tiny priors
```

任务梯度：

```text
g_task = J^T e_task
```

对于 bound-constrained q：

```text
g_projected = q - clip(q - g_task, lower, upper)
```

必须分别记录：

- task-only projected gradient；
- prior contribution；
- active lower/upper joint limits；
- complementarity/sign consistency；
- normalized projected-gradient norm。

不能用 prior 抵消 task gradient 后声称最近可达。

### 4.6 Closest-reachable certificate

`capability_limited` 至少要求：

```text
solver finite
joint limits satisfied
engine Jacobian finite
numerical stability gate passed
rank/subspace gate passed
projected-gradient gate passed
reachable residual fraction gate passed
seed consensus gate passed
continuation monotonicity gate passed
no semantic/target geometry failure
```

推荐初始全局 gates：

```text
reachable_residual_fraction <= 0.05
projector distance          <= existing Step 2.1 gate
seed task-space spread      <= 1e-3 normalized
joint-limit violation       <= 1e-9
continuation residual increase only within numerical tolerance
```

Projected-gradient tolerance必须由 backend dtype + synthetic calibration 确定并写入 artifact，不能按机器人设置。

任何 gate 调整必须：

- 全局；
- synthetic fixture 有证据；
- 在 ADR 中说明；
- 不能修改现有 exact projection thresholds。

### 4.7 Rank-zero task

若 relevant rank = 0：

```text
P = 0
e_parallel = 0
e_orthogonal = e
```

若：

- q 保持 neutral/合法；
- demand 非零被完整保存；
- 不向 legs/arms 或错误链泄漏；
- deterministic；

则该 task 可得到：

```text
unsupported_rank_zero / capability_limited
```

而不是 algorithm failure。

### 4.8 Single-axis torso

若 torso rank = 1：

- compatible axis demand必须正确保留并求解；
- incompatible pitch/roll/yaw 分量必须落入 orthogonal demand；
- 不得用单 yaw 腰拟合 pitch/roll；
- 不得因 incompatible residual 自动判整机失败。

### 4.9 Deterministic continuation / homotopy

大动作不再只从 neutral 直接单次求解。

Position：

```text
p(lambda) = p0 + lambda * (p_desired - p0)
```

Orientation：

```text
R(lambda) = R0 * Exp(lambda * Log(R0^T R_desired))
```

要求：

- deterministic initial step schedule；
- adaptive subdivision only by deterministic rules；
- engine Jacobian callback；
- active bounds；
- continuation history；
- failure后可缩小 step；
- 不得按 robot ID 调 step count。

### 4.10 Seed consensus

至少使用：

- neutral seed；
- previous continuation step；
- deterministic limit-aware alternative seed；

记录多个 seed 的：

```text
final task transform
residual
active limits
certificate class
```

若多个 seed 到达不同局部最优且差异超过全局 gate，则不能 certification pass。

---

## 5. Canonical motion 分类

### Invariance motions

```text
neutral
root_translation
global_root_yaw
```

必须 exact；不允许 capability-limited 包装 calibration/target bug。

### Ordinary capability motions

```text
torso_pitch
torso_roll
torso_yaw
arms_forward
elbow_bend
squat
single_step
```

允许：

- exact_reachable；
- capability_limited with full certificate。

### Extended motions

```text
mixed_torso_rotation
overhead_reach
asymmetric_arm_reach
crossed_body_reach
```

同样需要 exact 或 certificate，但在 summary 中单独统计。

### Stress motion

```text
extreme_but_valid_joint_limit_stress
```

不要求 exact reachability，但必须：

- finite；
- within limits；
- deterministic；
- closest-reachable certificate or explicit solver failure；
- 不参与 ordinary exact pass count。

---

## 6. 六个 xhigh Subagents

每个 agent 必须：

- 独立 worktree/branch；
- reasoning strength 记录为 `xhigh`；
- 先写 failing tests/oracles；
- 只修改 owned files；
- 小 commits；
- handoff 包含 commit、files、commands、tests、数值结果、风险和未完成项；
- 不得降低 thresholds；
- 不得只修改 status 名称制造通过。

### Agent A — Compact Failure Oracle and Baseline Ledger

**目标**：把 11 个大 JSON 失败报告压缩成可审查、可复现的任务级 oracle。

**Owned files**

```text
soma_retargeter/robotics/v3/failure_analysis.py
scripts/extract_capability_failure_ledger.py
tests/v3/test_capability_failure_ledger_*.py
artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json
docs/retargeting_v3/STEP2_3_FAILURE_BASELINE.md
docs/retargeting_v3/subagents/capability_agent_a_handoff.md
```

**必须输出每个失败 task**

```text
robot_id
motion
task
metric type
exact threshold
actual residual
normalized residual
active coordinates
regular/nominal rank
engine Jacobian source
reachable/orthogonal residual（修复前可为空）
active joint limits
solver status/message
semantic site source/hash
chain length
desired target distance/angle
```

**必须冻结 11-model baseline**

```text
atlas_drc_urdf
atlas_v4_urdf
booster_t1_mjcf
booster_t1_urdf
fourier_n1_mjcf
jaxon_urdf
mujoco_humanoid_mjcf
pal_talos_mjcf_direct
robotis_op3_mjcf
talos_urdf
valkyrie_urdf
```

不得只列机器人，不列 motion/task。

### Agent B — Deterministic Bounded Continuation Solver

**目标**：实现真正稳定的 closest-point projection solver。

**Owned files**

```text
soma_retargeter/robotics/v3/chain_projection.py
soma_retargeter/robotics/v3/projection_solver.py
tests/v3/test_projection_continuation_*.py
tests/v3/test_projection_seed_consensus_*.py
tests/v3/test_projection_joint_limit_kkt_*.py
docs/retargeting_v3/subagents/capability_agent_b_handoff.md
```

**必须实现**

- engine residual Jacobian callback；
- SO(3) log Jacobian；
- deterministic continuation；
- adaptive deterministic subdivision；
- bound handling；
- deterministic multi-seed；
- continuation history；
- task-only gradient；
- active-limit/KKT evidence；
- no robot-specific tuning。

**Synthetic fixtures**

1. reachable revolute endpoint；
2. unreachable target beyond link length；
3. joint-limit constrained closest point；
4. yaw-only torso with pitch demand；
5. fixed torso rank-zero demand；
6. near singular arm；
7. two-local-minimum chain；
8. nonidentity orientation target；
9. float32 Newton backend；
10. deterministic rerun。

### Agent C — Capability Decomposition and Certificates

**目标**：实现 reachable tangent decomposition 和严谨 certification。

**Owned files**

```text
soma_retargeter/robotics/v3/capability_projection.py
soma_retargeter/robotics/v3/projection_certificate.py
soma_retargeter/robotics/v3/canonical_projection.py
soma_retargeter/robotics/v3/reachability_metrics.py
tests/v3/test_capability_projector_*.py
tests/v3/test_projection_certificate_*.py
tests/v3/test_rank_zero_capability_*.py
tests/v3/test_single_axis_torso_capability_*.py
docs/retargeting_v3/subagents/capability_agent_c_handoff.md
```

**Certificate classes**

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

**必须证明**

- compatible demand 被保留；
- residual 主要 orthogonal to reachable tangent；
- projected gradient/KKT 满足；
- seed consensus；
- no leakage；
- certificate deterministic。

### Agent D — Target Geometry and Variant Differential Audit

**目标**：排除 11 台中的错误 semantic site、frame、length 或 variant mapping。

**Owned files**

```text
soma_retargeter/robotics/v3/target_geometry_audit.py
assets/robot_zoo/semantic_maps/<only evidence-backed corrections>
assets/robot_zoo/semantic_expectations/<only evidence-backed corrections>
tests/v3/test_failure_robot_target_geometry_*.py
artifacts/retargeting_v3_step2_capability/target_geometry_matrix.json
docs/retargeting_v3/STEP2_3_TARGET_GEOMETRY_AUDIT.md
docs/retargeting_v3/subagents/capability_agent_d_handoff.md
```

**Differential groups**

```text
Booster T1 URDF vs MJCF
Fourier N1 URDF(pass) vs MJCF(fail)
TALOS URDF vs Menagerie MJCF
Atlas DRC vs Atlas V4
H1/G1 passing controls
RPO yaw-only passing control
OP3 rank-zero/fixed-torso control
JAXON / Valkyrie / MuJoCo Humanoid standalone
```

**每个 task 检查**

- reference/target site body；
- local offset；
- left/right symmetry；
- chain topology；
- joint limits；
- neutral relative transform；
- source/robot segment-length ratio；
- desired target distance / chain length；
- target frame convention；
- URDF/MJCF variant differences。

Map 只有在 runtime topology/FK/geometry 证据证明错误时才允许修改；所有修改必须更新 fingerprint/source hash 和 tests。

### Agent E — Status Integration, 44-Model Validation and Cross-Format

**目标**：把 B/C/D 的结果接入 compiler、status、full validation 和 artifacts。

**Owned files**

```text
soma_retargeter/robotics/v3/profile.py
soma_retargeter/robotics/v3/robot_zoo.py
soma_retargeter/robotics/v3/validation.py
soma_retargeter/tools/validate_kinematic_profile_v3.py
artifacts/retargeting_v3_step2_capability/
tests/v3/test_capability_status_*.py
tests/v3/test_full_44_capability_validation_*.py
tests/v3/test_cross_format_capability_*.py
docs/retargeting_v3/STEP2_3_CAPABILITY_REPORT.md
docs/retargeting_v3/subagents/capability_agent_e_handoff.md
```

**必须实现**

- `capability_limited_passed` terminal status；
- compact per-task certificate summary；
- no generic status reason；
- 44-model deterministic rerun；
- baseline 33 terminal passes no regression；
- failure before/after matrix；
- negative-control pairs 不运行 humanoid task equivalence；
- `not_eligible` 不计为 pair failure；
- capability-limited pairs 比较 shared task certificates；
- G1 same-source strict equivalence继续通过。

### Agent F — Independent Red Team, LFS, CI and Final Verdict

**目标**：防止 status gaming、threshold cheating、pointer-only validation 和错误 capability certificate。

**Owned files**

```text
scripts/audit_retargeting_v3_capability.py
.github/workflows/retargeting_v3_capability.yml
tests/v3/test_capability_acceptance_*.py
docs/retargeting_v3/STEP2_3_CAPABILITY_ACCEPTANCE.md
docs/retargeting_v3/subagents/capability_agent_f_red_team.md
artifacts/retargeting_v3_step2_capability/test_results/
```

**红队必须检查**

- exact thresholds 是否被修改；
- robot ID special cases；
- residual 超标但无 KKT certificate；
- rank-zero demand 被清零；
- compatible torso axis 被丢弃；
- prior 抵消 task gradient；
- local minimum seed 不一致；
- semantic target geometry 错误被包装成 capability；
- stress motion 被用来掩盖 ordinary failure；
- negative controls 被算作 humanoid pass；
- deterministic rerun只比较一部分 terminal models；
- LFS pointer 被当模型内容；
- `git lfs fsck` 未运行；
- full upstream/mesh/fetch-only 通过 LFS 进入 Git；
- clean worktree/provenance失败。

---

## 7. Integrator 接口冻结

Wave 1 后必须冻结：

```python
ProjectionSolveResult
ContinuationStep
SeedConsensusReport
TaskCapabilityDecomposition
ProjectionOptimalityCertificate
CapabilityTaskStatus
CapabilityProfileSummary
```

推荐核心结构：

```python
@dataclass
class ProjectionOptimalityCertificate:
    classification: str
    passed: bool
    exact_threshold_passed: bool
    task_residual: float
    normalized_residual: float
    reachable_residual_norm: float
    orthogonal_residual_norm: float
    reachable_residual_fraction: float
    projected_gradient_norm: float
    projected_gradient_tolerance: float
    active_lower_bounds: list[int]
    active_upper_bounds: list[int]
    complementarity_passed: bool
    seed_consensus_passed: bool
    continuation_passed: bool
    joint_limits_passed: bool
    numerical_gate_passed: bool
    evidence: dict
```

Schema 必须版本化，旧 artifacts 只能作为 baseline read-only。

---

## 8. 11-model 必须逐台回答

### Atlas DRC / Atlas V4

- 两个 variant 是否在相同 arm reach motion 失败？
- 手部 target distance 是否超过实际 chain envelope？
- semantic hand site 是否为 distal endpoint？
- solver 是否到达 joint-limit KKT optimum？

### Booster T1 URDF / MJCF

- 两个格式是否共享相同 capability limitation？
- 单轴 waist 的 incompatible torso demand 是否被正确分离？
- hand failures 是否来自 target geometry、joint limits 或 local optimum？

### Fourier N1

- URDF 当前通过而 MJCF 失败；必须进行 differential audit；
- 不能直接把 URDF map/limits复制给 MJCF；
- 必须定位 variant joint limits、neutral frame、site或solver差异。

### JAXON / Valkyrie

- fetch-only source 必须继续 cache-only；
- algorithm result必须能用 source hash复现；
- 检查高 DoF chain 的 local minimum 和 joint-limit saturation。

### MuJoCo Humanoid

- 检查抽象 humanoid 的 semantic site/chain定义；
- 不允许因为它是通用示例模型而放宽 gate。

### TALOS URDF / MJCF

- 检查两种来源是否为同一 variant；
- distal sole/hand site正确；
- torso/arm/leg capability分别认证；
- pair report不得宣称 strict equivalence，除非同源。

### ROBOTIS OP3

- fixed/rank-zero torso demand必须成为 `unsupported_rank_zero`；
- torso demand不泄漏到 legs；
- hand/foot普通任务仍单独求解；
- 如果手部超标，必须给 closest-point或target geometry结论。

---

## 9. Full validation outputs

新目录：

```text
artifacts/retargeting_v3_step2_capability/
  environment.json
  commands.txt
  lfs_state.json
  baseline_summary.json
  baseline_failure_ledger.json
  target_geometry_matrix.json
  certificate_thresholds.json
  before_after.json
  summary.json
  capability_matrix.json
  deterministic_rerun.json
  cross_format.json
  per_robot/
  per_task/
  failures/
  test_results/
```

`summary.json` 至少包含：

```text
manifest_total                 46
in_scope_total                 44
deferred_snapshot_count         2
source_unavailable               0
model_load_failed                0
semantic_failed                  0
passed
capability_limited_passed
partial_passed
negative_control_passed
algorithm_failed
ordinary_tasks_exact
ordinary_tasks_capability_limited
ordinary_tasks_failed
stress_tasks_certified
deterministic_compared
deterministic_matched
lfs_objects_verified
```

`before_after.json` 必须对 11 个 baseline failures逐台记录：

```text
baseline status/reason
baseline failed motion/task
new status
exact vs capability-limited
certificate class
residual before/after
projected gradient
reachable residual fraction
active limits
seed consensus
```

---

## 10. LFS Acceptance Contract

LFS 已接受，但必须通过：

```bash
git lfs pull
git lfs fsck
```

生成：

```text
artifacts/retargeting_v3_step2_capability/lfs_state.json
```

至少记录：

```text
git_lfs_version
fsck_returncode
pointer_files_detected
materialized_snapshot_count
missing_lfs_objects
```

CI：

```yaml
- uses: actions/checkout@v4
  with:
    lfs: true
```

必须有 smoke test 打开并解析所有 LFS-tracked snapshot XML，拒绝只含：

```text
version https://git-lfs.github.com/spec/v1
```

的 pointer 文件。

仍禁止：

- meshes；
- full upstream repositories；
- fetch-only models；
- 超过资产政策的 snapshots；
- 未锁定源。

---

## 11. CI 设计

### GitHub-hosted job

- checkout with LFS；
- `git lfs fsck`；
- install package；
- synthetic solver/certificate tests；
- all committed snapshot parse/load smoke；
- static audit；
- G1 strict gate；
- no asset leakage。

### Full 44-model job

使用：

```text
self-hosted/manual workflow_dispatch
```

因为 5 个 fetch-only models依赖外部 cache。

必须：

- 固定 cache source hashes；
- 44-model validation；
- deterministic rerun；
- upload compact artifacts；
- 记录 runner/environment；
- failure时不得使用旧 artifacts冒充新结果。

---

## 12. 测试要求

### Synthetic/math

```bash
python -m pytest -q \
  tests/v3/test_projection_continuation_*.py \
  tests/v3/test_projection_seed_consensus_*.py \
  tests/v3/test_projection_joint_limit_kkt_*.py \
  tests/v3/test_capability_projector_*.py \
  tests/v3/test_projection_certificate_*.py \
  tests/v3/test_rank_zero_capability_*.py \
  tests/v3/test_single_axis_torso_capability_*.py
```

### Regression

```bash
python -m pytest -q tests/v3
python -m pytest -q
```

### Full 44

```bash
python -m soma_retargeter.tools.validate_kinematic_profile_v3 \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --output-dir artifacts/retargeting_v3_step2_capability \
  --low-discrepancy-count 32 \
  --deterministic-rerun
```

### Audit

```bash
python scripts/audit_retargeting_v3_capability.py \
  --artifact-dir artifacts/retargeting_v3_step2_capability \
  --lock assets/robot_zoo/robot_zoo_lock.json
```

---

## 13. Acceptance Gates

### No regression

- [ ] 44 in-scope source/load/semantic closure保持；
- [ ] baseline 21 `passed` 不回归；
- [ ] baseline 3 `partial_passed` 不回归；
- [ ] baseline 9 negative controls 不回归；
- [ ] G1 same-source strict pass保持；
- [ ] epsilon-only failures仍为0；
- [ ] numerical thresholds未修改；
- [ ] exact projection thresholds未修改。

### Solver/certificate

- [ ] continuation solver通过synthetic；
- [ ] analytic residual Jacobian通过FD；
- [ ] bounded KKT正确；
- [ ] rank-zero certification正确；
- [ ] single-axis torso分解正确；
- [ ] seed consensus；
- [ ] no robot-specific tuning。

### 11-model closure

- [ ] 每台有 compact baseline oracle；
- [ ] 每个 failed motion/task有新结论；
- [ ] no generic failure reasons；
- [ ] target geometry error修复或排除；
- [ ] exact/capability/solver失败分类可复现；
- [ ] `algorithm_failed=0` 才能 PASS。

### Reproducibility

- [ ] terminal 44 models全部 deterministic compared；
- [ ] deterministic matched=44；
- [ ] clean source worktree；
- [ ] clean artifacts；
- [ ] no absolute paths/private assets；
- [ ] `git lfs fsck` pass；
- [ ] no pointer-only validation；
- [ ] six xhigh handoffs；
- [ ] full tests；
- [ ] CI green；
- [ ] Agent F PASS。

---

## 14. 明确禁止的捷径

以下任一项都算失败：

- 提高 hand/foot/torso exact residual threshold；
- 把所有 residual 超标直接改名为 capability-limited；
- 只检查 solver `success=True`；
- 不计算 projected gradient/KKT；
- 不分 reachable/orthogonal residual；
- rank-zero demand清零；
- compatible single-axis demand丢失；
- 用 prior 把梯度压小；
- 一个 seed成功就忽略其他local minima；
- 按 robot ID选择 solver steps、weights或tolerance；
- 修改 semantic map但不更新fingerprint/evidence；
- 把 negative controls算入 humanoid algorithm pass；
- deterministic只重跑33台；
- pointer-only XML通过CI；
- 用 LFS 提交 mesh/upstream/fetch-only；
- 进入Step3或改生产pipeline。

---

## 15. 推荐提交顺序

1. `test: freeze compact baseline for eleven capability failures`
2. `feat: add analytic task residual Jacobians`
3. `feat: add deterministic bounded continuation solver`
4. `test: certify joint-limit and rank-zero closest points`
5. `feat: add reachable tangent decomposition`
6. `feat: add bounded KKT and seed-consensus certificates`
7. `data: audit failure-model target geometry and variants`
8. `feat: add capability_limited_passed status`
9. `fix: classify negative-control cross-format pairs correctly`
10. `test: close eleven model capability failures`
11. `test: rerun full Assets44 deterministically`
12. `ci: add LFS-aware capability workflows`
13. `docs: publish six handoffs and capability report`
14. `artifacts: publish clean Step 2.3 evidence`

不要压成一个大提交。

---

## 16. Codex 启动步骤

1. 确认 branch/HEAD/worktree；
2. `git lfs pull && git lfs fsck`；
3. 激活正确 conda 环境；
4. 记录 Python/MuJoCo/Newton/Warp/NumPy/SciPy/LFS版本；
5. 一次创建6个xhigh subagents；
6. Agent A先生成compact 11-model baseline；
7. Integrator冻结接口和certificate schema；
8. B/C/D并行；
9. E集成status/full validation；
10. F首次红队；
11. blocker返给owner；
12. full tests；
13. clean 44-model rerun；
14. deterministic 44/44；
15. LFS/provenance audit；
16. Agent F最终判定；
17. push全部commits；
18. 停止，不进入Step3。

---

## 17. 完成定义

最终只能写：

```text
Step 2.3 Capability Projection: PASS
```

或：

```text
Step 2.3 Capability Projection: BLOCKED
```

PASS 必须同时满足：

- [ ] 44 in-scope source/load/semantic closure无回归；
- [ ] 21 baseline exact passes无回归；
- [ ] 3 partial passes无回归；
- [ ] 9 negative controls无回归；
- [ ] 11 baseline algorithm failures全部转为 `passed` 或 `capability_limited_passed`；
- [ ] `algorithm_failed=0`；
- [ ] exact thresholds未修改；
- [ ] numerical thresholds未修改；
- [ ] 每个 capability-limited task有完整certificate；
- [ ] no generic failure reason；
- [ ] ordinary motions全部exact或certified；
- [ ] stress motions全部finite/limit-safe/certified；
- [ ] deterministic compared=44；
- [ ] deterministic matched=44；
- [ ] G1 strict equivalence继续通过；
- [ ] cross-format negative controls正确分类；
- [ ] git lfs fsck通过；
- [ ] no pointer-only validation；
- [ ] full tests通过；
- [ ] clean artifacts；
- [ ] six xhigh handoffs完整；
- [ ] CI green；
- [ ] Agent F最终为PASS；
- [ ] 未进入Step3。

若任一 hard gate 未满足，必须写 BLOCKED，不得包装为“基本完成”。
