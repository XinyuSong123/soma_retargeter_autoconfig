# Goal：SOMA Retargeter v3 — 核心数学重建、通用自动配置、全机器人 Zoo 验证

> **Codex 执行契约**
>
> 本文件是当前分支唯一的总任务入口。Codex 必须从头读完，再开始修改代码。
> 本任务不是继续在旧 v2 实验分支上调权重，也不是只写计划或跑更多诊断组合。
> 必须完成实现、测试、机器人资产清单、许可证审计、全量验证、文档和可复现 artifacts，并将全部提交推送到当前分支。
>
> 目标仓库：`XinyuSong123/soma_retargeter_autoconfig`  
> 建议工作分支：`retargeting-v3-core-robot-zoo`  
> 干净基线：`aed9f9fbf09a1bf2816d8bfa5c8423ea2300cf2c`（任务创建时的 `dev` HEAD）  
> 旧实验参考：`retargeting-v2-design`，审计时 HEAD `63b84c265453fd3a2a46676d00c5499b27e319e1`  
>
> **严禁把 `retargeting-v2-design` 整体 merge 到当前分支。** 只能在审查后移植被明确列为“可保留”的小模块。

---

## 0. 用户最终需求

构建一套真正通用、形态感知、尽量无需人工调参的人形机器人动作重定向系统：

1. 新机器人输入 URDF 或 MJCF 后，自动分析运动学树、真实自由度、关节轴、关节范围、链长、末端 site、左右对称性及可达旋转/平移子空间。
2. 单 yaw 腰、无腕、无手臂、低自由度腿、特殊关节轴等形态不再被强制跟踪不可实现的 6D 目标。
3. 不依赖逐机器人手工 `joint_scales`、pose pairs、quaternion offsets 和大量权重搜索。
4. RoboParty RPO、Unitree G1 23/29DoF、H1/H1_2、OP3、TALOS、Booster T1、Fourier N1、ToddlerBot、Apollo、Adam Lite、Berkeley Humanoid、JVRC、ergoCub、Atlas、Draco、SigmaBan、Simple Humanoid 等公开模型进入统一验证矩阵。
5. 许可证允许的模型生成轻量、可复现、可提交的 kinematic snapshots；许可证不适合再分发的模型进入固定版本的 fetch-only 验证集。
6. 所有机器人全量运行模型加载、语义识别、profile 编译、neutral 校准、canonical motions 和真实 BVH 测试；不能只在 RPO 上调到“看起来可以”。
7. 全部自动决策、失败、置信度、许可证、上游 commit、模型 hash、指标和复现命令落盘。
8. Codex 应使用 subagents 并行处理独立模块，但由主 agent 严格集成，不允许各 agent 自行改变总方向。

---

## 1. 为什么停止 v2 方向

旧实验分支不是废物，但它已经偏向：

- 增加 direction、pole、collision、temporal 等 objective；
- 扫描 hand/foot/hips 权重；
- 切换 analytic/autodiff；
- 增加大量 benchmark compare modes；
- 在核心 target 定义和 reachability 尚不正确时优化 solver。

下一阶段必须先修数学定义，再谈 objective 数量和性能。

### 1.1 已确认的关键问题

#### A. Reachability 不是运行时真实可达性

旧实现从静态 MJCF XML 的 body `pos/quat` 和 joint axis 近似 Jacobian，没有以 Newton 实际加载模型的 FK 为真值，且存在：

- 未正确覆盖 MJCF defaults/includes/compiler orientation；
- joint anchor、slide joint、default q、多个 DoF 等处理不足；
- 平移 Jacobian 对所有关节使用旋转关节叉乘公式；
- 只在一个 rest pose 计算 rank；
- endpoint chain 从“直接语义父节点”开始，而不是完整控制链。

结果示例：RPO Hand 只把 elbow-yaw 计入 hand chain，忽略肩和其他肘关节；Foot 只计 ankle，忽略 hip/knee。

#### B. v2 segment-local target 仍依赖旧 scaler offset

旧 v2：

- 只替换了位置长度；
- 没有 source-rest → robot-rest 的 per-edge frame alignment；
- 保留 source 世界 quaternion；
- 继续应用旧 scaler translation/quaternion offsets；
- 编译出来的 root horizontal scale 没真正接入 target builder。

因此它没有消除 pose-pair/旧 offset 依赖。

#### C. “projected relative torso rotation”实际上不是 relative

旧 runtime 把 source Chest 的世界 quaternion 直接取 rotation vector、投影、再交给标准 world-space rotation objective，没有计算：

```text
Chest relative to Hips
source rest delta
robot rest reconstruction
```

这不是腰部子空间正确处理。

#### D. 无腕 virtual hand site 没真正进入 position IK

旧 profile 能推断 distal hand site，但 runtime 以 `orientation_supported` 作为是否应用 position offset 的条件，导致无腕 hand site 被忽略，仍追踪 elbow link 原点。

#### E. Priority 与 normalization 主要是 metadata

- profile 声称存在 priority bands；
- runtime 只要 `normalized_weight` 存在就直接使用；
- 所有 objective 一次性进入同一个 solve；
- `characteristic_length` 和 `robust_loss` 没有真正作用于 position residual；
- priority guard 主要只保护 joint limits。

#### F. 旧 benchmark 并非统一目标比较

legacy 和 v2 各自用自己的 scaler target 计算 RMSE，因此不是严格同一 desired motion；rank=0 时 reachable error 被置零，也容易产生误导。

---

## 2. 从旧分支允许移植的内容

主 agent 必须逐文件审查后才可移植：

1. `CompiledRetargetProfile` 的稳定 JSON/dataclass 思路；
2. explicit contact 在 warmup 前读取并 prepend 的修复；
3. per-environment contact position target/weight；
4. contact hysteresis、minimum duration、release blending；
5. range-normalized joint-limit barrier；
6. virtual site 来源记录、报告和 provenance 框架；
7. benchmark artifacts/环境信息/失败落盘框架；
8. deterministic hashing 和 schema validation 的通用部分。

不得直接移植：

- 当前 XML 静态 reachability；
- 当前 task compiler；
- 当前 segment-local runtime；
- 当前 world-quaternion torso projection；
- 当前 morphology-based weight schedule；
- 大量 compare-mode 权重 sweep；
- 未验证的 collision proxy；
- “只因 rank/robot profile 就选择 analytic/autodiff”的经验调度。

---

## 3. v3 目标架构

```text
URDF/MJCF
   │
   ▼
SourceResolver + LicenseAuditor
   │
   ▼
RuntimeRobotModelAdapter
   │  以 Newton 实际模型/FK 为真值
   ▼
SemanticResolver
   │  pelvis, torso, shoulder, elbow, hand site, hip, knee, sole
   ▼
FullChainReachabilityAnalyzer
   │  多姿态 finite-difference Jv/Jw + SVD
   ▼
RestPoseCalibrator
   │  source/robot semantic frames、segment lengths、virtual sites
   ▼
RetargetTargetBuilder
   │  robot-space targets，不使用 legacy offsets
   ▼
MinimalRetargetTaskCompiler
   │  root、hand/foot、relative torso、limits、contact
   ▼
NewtonPipelineV3
   │  per-env state、真实 normalization、可审计优先级
   ▼
IndependentSemanticEvaluator
   │  同一 desired motion 下比较 legacy/v3
```

---

## 4. RuntimeRobotModelAdapter

### 4.1 真值来源

运动学真值必须来自 Newton 实际加载后的 model 和 FK：

- body labels；
- joint labels；
- body/joint parent；
- q/qd starts；
- joint types；
- joint limits；
- neutral/default `joint_q`；
- body transforms；
- site/local offsets；
- DoF 对 body 的影响关系。

XML/URDF parser 仅用于：

- source metadata；
- license/provenance；
- 显式 site/geom 名称；
- 生成模型之前的错误提示。

不得再用自写 XML walker 的静态 body 位置作为最终 FK。

### 4.2 URDF 与 MJCF

实现统一适配层：

```python
class RuntimeRobotModelAdapter(Protocol):
    model_format: Literal["urdf", "mjcf"]
    def neutral_q(self) -> np.ndarray: ...
    def forward_kinematics(self, q: np.ndarray) -> RobotKinematicState: ...
    def body_path(self, root: str, tip: str) -> list[str]: ...
    def active_dofs(self, root: str, tip: str) -> list[int]: ...
    def site_pose(self, state, site: SemanticSite) -> Pose: ...
```

URDF 可先通过可靠转换路径进入 Newton/MuJoCo，但必须同时保留原 URDF hash 和转换后的 MJCF hash。转换不得改变关节树、轴、limits、neutral pose 和 mimic/fixed semantics而不记录 warning。

### 4.3 模型 fingerprint

fingerprint 至少包含：

- 原始 URDF/MJCF bytes；
- include/xacro resolved bytes；
- 所有实际引用的 assets hash；
- adapter/compiler version；
- canonicalization settings；
- semantic override；
- neutral q；
- source format。

变量复用不得导致 MJCF hash被最后一个 mesh hash覆盖。

---

## 5. Full-chain reachability

### 5.1 控制根定义

不要将 semantic parent 当作 reachability root。

默认 full chains：

- torso：pelvis/root → torso site；
- left/right hand：torso/chest anchor → distal hand site；
- left/right foot：pelvis/root → sole site；
- head：torso/chest → head；
- 中间 segment 只用于 calibration/direction，不作为 endpoint reachability 根。

Hand 即使 semantic hierarchy 是 Arm→ForeArm→Hand，也要把 shoulder、elbow、wrist 全部计入 hand position reachability。

### 5.2 numerical Jacobian

先保证正确，不先优化解析 Jacobian。

对于每个 active DoF：

\[
J_v[:,i] =
\frac{p(q+\epsilon e_i)-p(q-\epsilon e_i)}{2\epsilon}
\]

相对姿态：

\[
R^{rel}(q)=R_{reference}(q)^{-1}R_{target}(q)
\]

\[
J_\omega[:,i] =
\frac{\log(R^{rel}(q-\epsilon e_i)^{-1}R^{rel}(q+\epsilon e_i))}
{2\epsilon}
\]

要求：

- revolute、prismatic、复合 joint 均通过 FK 差分自然处理；
- quaternion sign canonicalization；
- central difference；
- epsilon 根据 joint range/type 自适应；
- 记录 finite-difference conditioning。

### 5.3 多姿态 rank

至少评估：

- neutral；
- 每个 DoF 在安全 range 的 ±10%；
- 固定 seed 的 low-discrepancy 组合姿态；
- 左右镜像姿态。

把多姿态 Jacobian 横向堆叠后做 SVD，避免 T-pose singularity 误判 rank。

### 5.4 必须通过的 synthetic tests

1. yaw-only torso：
   - rotational rank = 1；
   - basis 与真实 joint screw axis 一致。
2. 5DoF shoulder/elbow、无 wrist：
   - hand position aggregate rank 应反映完整手臂，而不是只有 elbow-yaw；
   - hand orientation capability按真实链计算，不能只靠名称中是否有 `wrist`。
3. 6DoF leg：
   - foot position rank包含 hip/knee/ankle贡献。
4. prismatic torso：
   - Jv 正确、Jω 不虚构。
5. rest-pose singular arm：
   - 单姿态 rank低，但多姿态 aggregate rank恢复。

---

## 6. RestPoseCalibrator

### 6.1 v3 禁止 legacy offset

v3 路径不得读取或应用旧 scaler 的：

- per-joint geocentric scales；
- optimized translation offsets；
- optimized quaternion offsets；
- pose-pair结果。

legacy 仅用于 baseline，不得影响 v3 target。

### 6.2 semantic rest frames

为 source 和 robot 构造：

- pelvis frame；
- chest frame；
- shoulder/upper-arm frame；
- elbow bend-plane frame；
- hip/thigh frame；
- knee bend-plane frame；
- foot/sole frame。

frame 构造优先：

1. 显式 site/frame；
2. 多个稳定 semantic directions 的 Kabsch/Wahba；
3. primary direction + bend-plane normal；
4. 对称链镜像；
5. 有置信度的 fallback。

每个 frame 保存：

- rotation；
- position；
- source；
- confidence；
- degeneracy/fallback原因。

### 6.3 segment-local targets

对 semantic edge：

\[
u_h(t)=\frac{p_h^c(t)-p_h^p(t)}{\|p_h^c(t)-p_h^p(t)\|}
\]

\[
u_r(t)=A_{edge}u_h(t)
\]

\[
p_r^{c,*}(t)=p_r^{p,*}(t)+L_r u_r(t)
\]

要求：

- \(A_{edge}\) 来自 rest calibration；
- \(L_r > 0\)；
- 左右对称机器人默认长度 tying；
- root horizontal translation按 robot/source leg length；
- vertical root结合 nominal pelvis height、ground 和 stance；
- zero-length source edge使用上一有效方向或 rest direction，不直接崩溃整条 clip。

### 6.4 neutral exactness

source neutral/T-pose 输入必须重建 robot neutral semantic sites：

- position max error < 1 mm；
- relative rotation error < 0.01 rad；
- 左右镜像误差可解释；
- 不依赖任何 pose-pair optimizer。

这是所有动态 benchmark 之前的硬门槛。

---

## 7. 真正的 torso relative rotation

必须实现相对 Hips/Pelvis 的目标：

\[
R_h^{rel}=R_h^{hips^{-1}}R_h^{chest}
\]

\[
\Delta R_h=(R_{h,rest}^{rel})^{-1}R_h^{rel}
\]

\[
e_h=\log(\Delta R_h)
\]

\[
e_h^{reachable}=P_R e_h
\]

\[
R_{r,target}^{rel}=R_{r,rest}^{rel}\exp(e_h^{reachable})
\]

objective 比较：

\[
R_r^{rel}(q)=R_r^{hips}(q)^{-1}R_r^{chest}(q)
\]

不得把 source world quaternion直接投影后当 robot world target。

第一版允许 autodiff或 numerical Jacobian，正确后再优化 analytic。

测试：

- pure pitch；
- pure roll；
- pure yaw；
- mixed rotation；
- root/world yaw同时变化；
- robot waist axis非世界 Z；
- pelvis frame初始旋转不为 identity。

对 yaw-only robot，pure pitch/roll：

- projected torso target不可达分量 ≤ 1%；
- waist yaw响应 ≤ 对 pure yaw 响应的 5%；
- lower-body compensation显著低于 legacy。

---

## 8. Virtual sites 与 endpoint 语义

### 8.1 position 与 orientation capability解耦

是否支持 hand orientation，不能决定是否应用 hand position site offset。

无腕机器人：

- distal hand site必须进入 position objective；
- orientation根据 full-chain rotational capability决定；
- 可只跟踪 reachable orientation；
- rank=0 时关闭 orientation并记录原因。

### 8.2 site 推断顺序

1. explicit MJCF/URDF frame/site；
2. distal child joint anchor；
3. collision/visual geom bounds；
4. symmetric side mirror；
5. kinematic segment extrapolation。

生成 site时保存：

- local pose；
- source；
- confidence；
- original mesh/geom hash；
- whether used for runtime；
- whether used only for reporting。

### 8.3 foot anchors

自动从 sole geometry/site生成：

- heel；
- toe；
- inner edge；
- outer edge；
- sole center。

不要求用户手工提供 `virtual_sole_anchors`。

---

## 9. 最小 runtime objective 集

第一阶段只保留：

1. pelvis/root position；
2. 可选 root reachable orientation/upright feasibility；
3. left/right distal hand position；
4. left/right foot/sole position；
5. true torso relative projected rotation；
6. joint limits；
7. per-env contact anchors；
8. 可选 very-low neutral prior。

暂时关闭：

- direction；
- pole vector；
- collision；
- temporal acceleration；
- projected endpoint position；
- morphology-based analytic/autodiff经验调度；
- 多组权重 sweep。

只有当最小集在 RPO、G1和 synthetic fixtures通过后，才能逐项重新引入。

---

## 10. 无量纲 residual 与权重

position residual：

\[
\hat e_p = e_p/\max(L_{chain},\epsilon)
\]

joint residual：

\[
\hat e_q=(q-q_{ref})/\max(q_{upper}-q_{lower},\epsilon)
\]

rotation使用 rad。

每个 residual按维数除以 \(\sqrt d\)。

注意 objective 的 `weight` 是 residual gain，least-squares 中会被平方。禁止直接使用 500、1000 这类未校准 gain。

第一版统一 gain建议：

```text
hard safety/contact: 10–30
root/stance: 5–15
hand/swing foot: 2–8
torso style: 0.5–3
neutral: 0.01–0.2
```

最终数值由 normalization 后的小型 calibration suite决定，不做逐机器人 sweep。

`robust_loss` 若写进 profile就必须在 runtime生效；否则从 schema删除，不能只保留 metadata。

---

## 11. 优先级策略

第一版不要求完整 HQP。

最低可接受实现：

1. feasibility pass：
   - joint safety；
   - stance contact；
   - root/upright；
2. tracking pass：
   - 保留 feasibility；
   - hand、swing foot、torso；
3. optional style pass：
   - neutral；
   - 后续 direction/pole。

每个 pass结束后检查高优先级 residual。若退化超过 tolerance：

- 回滚；
- 记录；
- 不通过 gate。

禁止把“priority bands写进 JSON”当作 priority solver已完成。

---

## 12. Contact

保留并完善旧分支中正确的：

- raw buffer explicit contacts优先；
- warmup prepend；
- per-env target和weight；
- hysteresis；
- min contact frames；
- release blending；
- missing foot guard；
- exception-safe reset。

新增：

- 自动 sole anchors；
- contact confidence；
- minimum off frames；
- touchdown velocity检查；
- stance target lock；
- batch parity；
- full clip reset；
- contact metrics。

当前所有 Tier-A humanoid必须在 profile中明确：

- contact enabled/disabled；
- source；
- anchor provenance；
- reason；
- coverage。

---

## 13. Robot Zoo 总体策略

### 13.1 “全部验证”的定义

不是把互联网所有机器人仓库无筛选复制进来，而是：

1. 建立公开、固定版本、许可证可审计的 humanoid/biped catalog；
2. 所有 catalog entry都必须产生结构化结果；
3. 宽松许可证模型生成轻量 kinematic snapshot并纳入常规 CI；
4. GPL/LGPL/CC/NASA或条款不明确模型进入 fetch-only suite，不在 Apache仓库中重新分发；
5. 非人形机器人作为 negative controls，必须安全拒绝或降级，而不是误识别；
6. 任何未运行 entry都必须明确写原因，不能静默跳过。

### 13.2 上游聚合源

固定使用：

```text
robot_descriptions == 2.0.0
MuJoCo Menagerie reference commit:
feadf76d42f8a2162426f7d226a3b539556b3bf5
```

`robot_descriptions` 的 description modules已固定各上游仓库 commit，并公开格式与许可证。不得在执行时跟踪浮动 `main`。

### 13.3 资产层级

#### Tier A：permissive kinematic snapshots

许可证为 Apache-2.0/MIT/BSD 类，允许再分发时：

```text
assets/robot_zoo/<robot_id>/
  model.urdf 或 model.xml
  SOURCE.json
  LICENSE
  NOTICE（如需要）
  semantic_expectations.json
```

默认不提交大型 mesh。生成 mesh-free/collision-light kinematic snapshot：

- 保留 joint tree；
- joint type/axis/limit；
- body transform；
- inertial可保留；
- site/frame；
- mimic/fixed joint语义；
- 用简单 primitive替代视觉 mesh；
- 保存 original model hash；
- snapshot generator必须确定性。

#### Tier B：fetch-only full models

对于 GPL/LGPL/CC-BY-SA/NASA或不确定条款：

- 只在 `robot_zoo_manifest.json` 中记录；
- 使用固定 commit下载到 `.cache/robot_zoo`；
- `.gitignore` 排除；
- 手动/full-zoo workflow运行；
- 仍输出完整 benchmark；
- 不把受限资产复制进仓库。

#### Tier C：negative controls

至少：

- Cassie/Upkie/Bolt：biped无完整上肢；
- Unitree Go2：quadruped；
- Franka Panda：单臂；
- Stretch：移动操作臂。

预期：

- 不崩溃；
- 明确 capability status；
- 不伪造 humanoid semantic map；
- 返回需要人工语义或 unsupported morphology。

### 13.4 必须覆盖的 positive humanoids

以 `assets/robot_zoo/robot_zoo_manifest.json` 为机器真值。至少包括：

- local RoboParty RPO；
- Unitree G1 URDF + MJCF；
- Unitree H1 URDF + MJCF；
- Unitree H1_2 URDF + MJCF；
- ROBOTIS OP3 MJCF；
- PAL TALOS MJCF；
- Booster T1 URDF + MJCF；
- ToddlerBot URDF、2XC MJCF、2XM MJCF；
- PNDbotics Adam Lite MJCF；
- Apptronik Apollo MJCF；
- Berkeley Humanoid URDF + MJCF；
- Fourier N1 URDF + MJCF；
- JVRC-1 URDF + MJCF；
- BXI Elf2 URDF + MJCF；
- ergoCub URDF；
- Draco3 URDF；
- SigmaBan URDF；
- Simple Humanoid URDF；
- Atlas DRC/V4 URDF；
- Romeo URDF；
- MuJoCo Humanoid MJCF。

### 13.5 许可证规则

实现 `tools/audit_robot_zoo_licenses.py`：

- 每个 entry必须有 SPDX-like license id；
- source repo/ref/path；
- license file hash；
- redistribution policy；
- generated snapshot lineage；
- NOTICE要求；
- 不允许 `unknown` entry进入 Tier A；
- CI验证 snapshot与 manifest一致；
- source hash变化必须显式更新。

---

## 14. Robot Zoo 工具

实现：

```bash
python -m soma_retargeter.tools.sync_robot_zoo \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --tier A \
  --verify-license \
  --write-snapshots

python -m soma_retargeter.tools.validate_robot_zoo \
  --all \
  --output artifacts/retargeting_v3/robot_zoo

python -m soma_retargeter.tools.audit_robot_zoo_licenses \
  --strict
```

`sync_robot_zoo` 必须：

- 支持 offline cache；
- lock file；
- SHA256；
- fixed ref；
- retry但不吞错；
- 不执行上游任意代码；
- xacro只在隔离环境、固定参数运行；
- path traversal防护；
- 最大下载/解压大小限制；
- 保存命令和版本；
- 生成确定性 snapshot。

---

## 15. SemanticResolver

输入信号优先级：

1. explicit minimal override；
2. explicit sites/frames；
3. body/joint names；
4. topology；
5. neutral pose空间位置；
6. bilateral symmetry；
7. chain capability。

输出：

- mapping；
- confidence；
- alternatives；
- supporting evidence；
- rejection reason。

不允许仅因为 body存在就给 confidence=1.0。

每个 Tier-A机器人都要有：

```text
semantic_resolution_report.json
expected_semantics.json
```

自动结果与人工确认 ground truth对比，统计 precision/recall和置信度 calibration。

---

## 16. IndependentSemanticEvaluator

不能用每个 pipeline自己的 target作为最终横向比较。

从 source motion和统一 calibration生成一次 canonical desired semantics：

- pelvis trajectory；
- torso relative reachable delta；
- normalized hand position relative chest；
- normalized foot position relative pelvis；
- limb direction；
- contact labels。

legacy和v3都对同一 canonical desired semantics评估。

rank=0时：

- 不得把 error直接置零并算作通过；
- 标记 `unobservable/unreachable`；
- 分开报告 reachable error与unreachable demand；
- gate根据 capability-aware规则判断。

---

## 17. 测试层级

### 17.1 数学单元测试

- SO(3) log/exp；
- relative quaternion；
- non-world-axis single-yaw projection；
- numerical Jacobian vs known analytic fixture；
- prismatic joint；
- multi-pose rank；
- full-chain active DoF；
- rest frame Kabsch；
- mirror tying；
- virtual site；
- zero-length source edge；
- normalization；
- deterministic profile；
- fingerprint；
- URDF↔canonical MJCF topology preservation。

### 17.2 synthetic fixtures

至少：

1. yaw-only torso + 5DoF no-wrist arm；
2. 3DoF torso + wrist；
3. prismatic torso；
4. asymmetric humanoid；
5. biped no arms；
6. non-humanoid negative model；
7. rotated base frame；
8. joint axis not aligned with XYZ；
9. neutral pose away from joint center。

### 17.3 canonical motions

固定生成：

1. neutral；
2. torso pitch；
3. torso roll；
4. torso yaw；
5. mixed torso；
6. arms forward；
7. elbow bend；
8. overhead reach；
9. squat；
10. single support；
11. step；
12. turn in place。

### 17.4 real BVH

至少：

- walk；
- squat；
- dance/high-motion；
- torso twist；
- arm reach。

不只跑最大运动80帧；必须同时有 full-length运行。

### 17.5 batch

- single env；
- 2 identical；
- 2不同 contact phase；
- mixed clip lengths；
- reset/reuse pipeline。

---

## 18. 验收门槛

### 18.1 核心正确性

- synthetic yaw torso rank=1；
- 5DoF no-wrist full hand chain aggregate position rank不能被误判为只含一个 distal joint；
- 6DoF leg foot position chain包含 hip/knee/ankle；
- neutral max position error < 1 mm；
- neutral relative rotation error < 0.01 rad；
- profile deterministic；
- no NaN/Inf；
- no negative segment length。

### 18.2 yaw-only leakage

pure source pitch/roll：

- projected unreachable residual ≤ source demand 1%；
- robot waist yaw ≤ pure-yaw响应5%；
- hip/knee compensation norm比 legacy降低至少50%。

### 18.3 RPO

在共同 canonical desired motion 下：

- reachable hand RMSE不差于 legacy 5%；
- reachable foot RMSE不差于 legacy 5%；
- root tilt不增加；
- penetration不增加；
- velocity/acceleration p95不增加；
- runtime目标 ≤ legacy 1.25x；
- no-wrist distal site实际被 position IK使用；
- contact anchors实际启用并有 coverage报告。

### 18.4 G1

- 23DoF与29DoF都通过；
- 29DoF不显著退化；
- waist 3DoF rank正确；
- wrist orientation按能力启用；
- runtime ≤ legacy 1.25x。

### 18.5 Robot Zoo

Tier A每个 entry：

- source sync/hash/license通过；
- model load通过；
- canonicalization通过；
- semantic resolution有结果；
- profile compile通过或结构化 capability rejection；
- neutral测试有结果；
- canonical motions有结果；
- 无 crash；
- artifacts完整。

Positive humanoid Tier A：

- 不允许 `unsupported`；
- 允许 `needs_semantic_confirmation`，但必须有候选和置信度；
- 最终提交前必须提供 verified minimal semantic map。

Negative controls：

- 必须结构化拒绝或降级；
- 不能误报为 full humanoid ready。

### 18.6 CI

建立：

```text
fast-core
robot-zoo-tier-a-compile
robot-zoo-tier-a-canonical
rpo-runtime
g1-runtime
license-audit
full-zoo-manual
```

PR必须跑 fast-core、Tier-A compile、license。GPU/full motion可单独 workflow，但结果必须可下载。

---

## 19. Subagent 架构

主 Codex 作为 **Integrator Agent**，拥有最终方向和合并权。

### Agent A：Runtime Kinematics

负责：

- Newton model adapter；
- URDF/MJCF canonicalization接口；
- FK；
- active DoF；
- numerical Jv/Jw；
- multi-pose rank；
- fingerprint。

可再派生：

- A1 Newton API Probe；
- A2 Numerical Validation。

不得修改 benchmark或robot registry业务逻辑。

### Agent B：Calibration & Targets

负责：

- semantic rest frames；
- Kabsch/Wahba；
- segment-local robot-space targets；
- root scaling；
- true relative torso target；
- virtual sites。

可再派生：

- B1 Source skeleton frames；
- B2 Robot rest calibration。

不得读取 legacy optimized offsets作为v3输入。

### Agent C：IK Runtime

负责：

- minimal objectives；
- relative torso objective；
- normalized position objective；
- contact integration；
- joint limits；
- staged solve；
- batch state。

可再派生：

- C1 Objective correctness；
- C2 Batch/contact state。

不得新增 direction/pole/collision，除非 Integrator在核心 gates通过后解锁。

### Agent D：Robot Zoo & Licensing

负责：

- manifest；
- sync tool；
- source pin；
- license audit；
- kinematic snapshots；
- semantic expectation files。

可再派生：

- D1 License Scout；
- D2 Asset Normalizer；
- D3 Semantic Ground Truth。

禁止提交不明许可证或大型未经审查 meshes。

### Agent E：Benchmark & QA

负责：

- independent semantic evaluator；
- canonical motion generator；
- full-length BVH；
- gates；
- artifacts；
- CI matrix；
- performance profiling。

可再派生：

- E1 Metrics Validator；
- E2 Performance/CI。

不得用 pipeline自身 target作为最终比较真值。

### Agent F：Adversarial Reviewer

只审查，不实现主功能：

- 查 robot-name special case；
- 查静态 XML 被误当 FK；
- 查 metadata 未接 runtime；
- 查 rank0伪通过；
- 查 license；
- 查 benchmark apples-to-oranges；
- 查未运行测试却声称通过；
- 查性能回归。

### 19.1 Subagent 协作规则

1. 每个 agent必须先读 `goal.md`、`CODE_AUDIT.md`、manifest。
2. 每个 agent使用独立 worktree/branch或明确文件 ownership。
3. 每个 agent提交小 commits。
4. 每个 agent输出：
   - changed files；
   - tests；
   - commands；
   - results；
   - assumptions；
   - unresolved risks；
   - handoff。
5. Integrator按 A → B → C → D/E → F 顺序集成。
6. 并行上限3个重任务，避免同文件冲突。
7. 子 agent不得修改总 acceptance gates。
8. 任何 agent发现架构问题，先写 ADR，不得直接扩大 scope。
9. 最终由 F 对全部完成声明做反证审查。

---

## 20. 推荐提交顺序

1. `docs: add v3 execution contract and robot zoo manifest`
2. `test: add runtime kinematics synthetic fixtures`
3. `feat: add Newton runtime model adapter`
4. `feat: add full-chain multi-pose reachability`
5. `feat: add source and robot rest frame calibration`
6. `feat: add offset-free v3 target builder`
7. `feat: add true relative torso objective`
8. `feat: wire minimal normalized v3 IK runtime`
9. `feat: auto-generate sole and distal sites`
10. `feat: add robot zoo source sync and license audit`
11. `data: add permissive kinematic robot snapshots`
12. `test: add robot zoo compile and neutral matrix`
13. `feat: add independent semantic benchmark`
14. `test: add RPO and G1 runtime gates`
15. `ci: add tiered robot zoo workflows`
16. `docs: publish implementation and validation reports`

每个提交后运行相关测试。不要累积几十个实验性 weight commits。

---

## 21. 明确禁止

- merge整个 `retargeting-v2-design`；
- 继续当前 hand/foot/hips权重 sweep；
- 机器人名字 if/else；
- 只靠 joint名称识别能力；
- 静态 XML body位置替代 runtime FK；
- immediate semantic edge替代 full endpoint chain；
- v3应用 legacy scaler offset；
- world Chest quaternion冒充 relative torso；
- orientation support控制 position site offset；
- rank0 error置零后宣称通过；
- profile写字段但 runtime不使用；
- 只跑 synthetic，不跑真实模型；
- 只跑 RPO；
- 盲目复制所有 Menagerie meshes；
- 提交 GPL/CC/NASA资产却不处理许可证；
- 下载浮动 main；
- benchmark失败仍写“完成”；
- tests因缺依赖未运行却宣称通过；
- 只写计划、不实现。

---

## 22. 必须落盘的信息

```text
docs/retargeting_v3/
  CODE_AUDIT.md
  SUBAGENTS.md
  ASSET_POLICY.md
  architecture.md
  design_decisions.md
  implementation_report.md
  known_limitations.md
  migration.md
  subagents/
    agent_a_handoff.md
    agent_b_handoff.md
    agent_c_handoff.md
    agent_d_handoff.md
    agent_e_handoff.md
    agent_f_review.md

assets/robot_zoo/
  robot_zoo_manifest.json
  lock.json
  <tier-a snapshots>

artifacts/retargeting_v3/
  environment.json
  commands.txt
  source_inventory.json
  license_audit.json
  robot_zoo_summary.json
  benchmark_summary.json
  benchmark_gates.json
  per_robot/
  failures/
```

---

## 23. Codex 启动步骤

1. `git status --short`
2. `git rev-parse HEAD`
3. 确认当前分支为 `retargeting-v3-core-robot-zoo`
4. 确认基线祖先含 `aed9f9fbf09a1bf2816d8bfa5c8423ea2300cf2c`
5. 阅读本文件和 `docs/retargeting_v3/*`
6. 查看旧分支但不得 merge：
   ```bash
   git show retargeting-v2-design:docs/retargeting_v2/implementation_report.md
   ```
7. 运行基线 tests并保存环境
8. 创建 subagent worktrees和任务清单
9. 先完成 synthetic full-chain/reachability tests
10. 在这些 tests失败前，不得修改 production weights
11. 核心 gates通过后再同步 Tier-A robot zoo
12. 持续推送小 commits到当前分支
13. 最后运行 Agent F审查
14. 只有所有硬门槛通过才标记完成

---

## 24. 完成定义

只有以下全部满足，任务才完成：

- [ ] runtime FK为运动学真值；
- [ ] full-chain multi-pose reachability；
- [ ] no legacy offsets in v3；
- [ ] true relative torso objective；
- [ ] distal hand/sole sites实际进入 IK；
- [ ] normalized residual真正生效；
- [ ] contact自动 anchors且batch正确；
- [ ] RPO通过硬 gates；
- [ ] G1 23/29通过硬 gates；
- [ ] Tier-A robot zoo全量结构化验证；
- [ ] fetch-only模型有可复现结果或明确外部限制；
- [ ] negative controls安全拒绝；
- [ ] independent semantic benchmark；
- [ ] full-length BVH；
- [ ] license audit通过；
- [ ] CI矩阵；
- [ ] tests实际运行；
- [ ] artifacts完整；
- [ ] Agent F无阻断项；
- [ ] implementation report诚实列出未完成项；
- [ ] 当前分支全部 commits已push。

