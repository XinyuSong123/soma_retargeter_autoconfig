# Goal：SOMA Retargeter v2 — 通用、形态感知、无需人工调参的机器人动作重定向

> **给 Codex 的执行契约**
>
> 本文件不是讨论稿，也不是只做调研的任务。请直接完成代码、测试、文档、基准评测和可复现报告，并把全部提交推送到当前分支。不得只提交计划、伪代码或部分原型后停止。
>
> 当前工作分支：`retargeting-v2-design`  
> 基线提交：`a53a45dce0d5f8a05cdecf5c796b046031368c6a`（`dev` 上合入接触感知足部 IK 的 PR #2）  
> 目标仓库：`XinyuSong123/soma_retargeter_autoconfig`

---

## 1. 最终目标

将当前依赖每台机器人手工调整 scaler、姿态对、四元数偏移和 IK 权重的流程，升级为一个 **morphology-aware retargeting compiler**：

1. 新机器人最低只需提供 MJCF/XML；URDF 可选。
2. 用户仅需确认少量语义映射，例如 pelvis、torso、左右手、左右脚对应哪个 robot body/link。常见人形机器人应自动识别并给出置信度。
3. 系统自动分析机器人运动学、自由度、关节轴、关节范围、链长、左右对称性、末端是否有腕关节、腰部实际可达旋转子空间。
4. 系统自动生成比例、坐标系对齐、虚拟末端 site、IK 任务、可达轴掩码、优先级、接触参数、时序正则参数。
5. 单 yaw 腰、无腕、低自由度手臂等机器人不再被强迫跟踪不可实现的 6D 目标。
6. 默认不要求人工制作 robot pose pair；姿态对只作为可选的、有边界的微调数据，不能破坏解析标定结果。
7. 同一套算法应覆盖 RoboParty RPO、Unitree G1 29DoF、G1 23DoF、E3 v2，以及资产可用时的逐际动力 OLI；不得通过机器人名字写专用算法分支。
8. 批处理和单条处理必须语义一致；接触状态、任务权重和锁定目标必须按 environment 独立。
9. 所有自动决策、置信度、指标、失败原因和版本信息必须落盘，便于下一轮分析。

### 1.1 “无需人工调参”的定义

允许：

- 提供或确认语义 link 映射；
- 对自动识别置信度低的 link 做一次性覆盖；
- 对硬件安全范围做显式 override；
- 可选地提供姿态对用于小幅 residual calibration。

不允许作为正常接入步骤：

- 手工逐机器人调 `joint_scales`；
- 手工猜 quaternion offset；
- 手工调每个 link 的位置/旋转权重；
- 为单 yaw 腰反复降低 Chest 权重直到“看起来能用”；
- 依赖一组人工 robot pose pair 才能得到正常结果；
- 在代码中按 `robot_type == "roboparty_rpo"`、`"unitree_g1"` 等写算法例外。

---

## 2. 当前已知事实、缺陷和必须先修的问题

### 2.1 现有 scaler 的模型错误

`HumanToRobotScaler.wp_compute_scaled_effectors()` 当前对每个映射关节点执行“相对人体根节点的独立 geocentric scaling”。这不是逐段骨骼缩放，会造成各关节点彼此独立漂移，优化器可以用负尺度和巨大 offset 拟合少量静态姿态。

RoboParty 当前配置已经出现：

- `RightForeArm` 负尺度；
- `RightHand` 负尺度；
- 左右链严重不对称；
- 多个 translation offset 远超合理的 residual calibration 范围。

这些值不得作为 v2 的初始真值。RPO 的 v2 profile 必须从机器人 FK/几何重新生成。

### 2.2 当前 optimizer 无物理约束

`app/optimize_scaler_config.py` 直接对 scale、translation offset、quaternion offset 做梯度更新，缺少：

- scale 正值参数化；
- 上下界；
- 左右对称 tying；
- identity/zero prior；
- 鲁棒损失；
- train/validation；
- 改善不足时回滚；
- 参数可辨识性检查。

结果是多组静态 pose pair 也可能把配置优化坏。

### 2.3 当前 IK 对所有映射 link 创建完整 position + quaternion objective

当前 runtime parser 将简单字符串 `ik_map` 统一展开为：

- `t_body == r_body`；
- 固定的人体部位默认权重；
- 每个映射点完整 3D position；
- 每个映射点完整 3D orientation。

它不理解机器人链少了哪些自由度，也不区分：

- 世界姿态与相对姿态；
- 末端位置与中间 link 方向；
- 可达旋转与不可达旋转；
- 有腕机器人与无腕机器人；
- 支撑脚与摆动脚；
- 低优先级风格目标与高优先级接触目标。

### 2.4 单 yaw 腰的核心失败模式

人体 `Chest relative to Hips` 包含 pitch、roll、yaw；单 yaw 腰机器人只可实现一个旋转方向。现有完整 quaternion objective 会把不可实现误差分摊给：

- floating base；
- 髋关节；
- 膝关节；
- 肩关节；
- 脚部姿态。

必须在进入 IK 前或 objective 内将旋转误差投影到该链的可达旋转子空间。

### 2.5 RoboParty 无腕却把人体 Hand 映射到 elbow-yaw link

RPO 机器人姿态文件显示手臂链为肩部 3 DoF + 肘部 2 DoF，没有独立腕关节。现有配置把 `LeftHand/RightHand` 映射到 elbow-yaw link 原点，同时追踪完整位置和姿态，目标定义不正确。

v2 必须：

- 自动识别末端缺少 wrist；
- 在末端 link 上生成 distal virtual site；
- Hand 默认仅追踪 position；
- orientation 在不可达时关闭或只投影到可达子空间；
- 用 upper-arm/forearm direction 与 elbow pole-vector 消除肘弯曲歧义。

### 2.6 已合入 `dev` 的接触感知足部 IK 仍有未解决缺陷

PR #1 和 PR #2 已加入 contact-aware virtual toe/heel IK，但 Codex review 留下的事项必须在第一批提交修复：

1. explicit NPZ contacts 在插入 initialization frames 后会丢失，因为读取的是替换后的普通 `AnimationBuffer`，而不是原始 clip buffer；
2. multi-env 下 objective weight 是共享的，因此当前实现直接禁用 batch contact-aware IK；v2 不允许禁用，必须实现 per-environment residual weight；
3. `execute()` 中临时修改 `self.contact_aware_foot_ik_enabled`，异常时不能保证恢复；
4. contact objective 的 missing-foot guard、状态重置、clip 长度对齐必须完整覆盖；
5. initialization/stabilization frames 与 contact label 的 prepend 策略必须明确且测试；
6. 当前接触启发式只使用高度与水平速度，应增加可信度、hysteresis、最短接触持续时间和 release blending。

---

## 3. 设计原则和硬性约束

1. **先投影可达性，再求 IK。** 不允许通过降低权重掩盖不可实现目标。
2. **逐段标定，不做关节点独立 root-centric scaling。**
3. **相对任务优先于不必要的世界任务。** Torso 使用 relative-to-pelvis；中间 limb 使用 direction/pole；末端使用 site position。
4. **任务全部无量纲化。** 位置残差除以对应链长，旋转残差使用 rad，关节残差除以 joint range。
5. **优先级明确。** 接触、根和关节安全不得被手部风格目标破坏。
6. **左右对称默认强制。** 只有模型几何或显式配置证明不对称时才放开。
7. **自动结果可解释。** 每个自动 task 都要保存来源、链、可达 rank、basis、权重、置信度和禁用原因。
8. **向后兼容。** 现有字符串 `ik_map`、现有 CLI、现有 CSV 输出继续可用。
9. **不得静默失败。** 低置信度、缺失语义、无法生成 site、病态 Jacobian 都要产生结构化 warning。
10. **不依赖机器人专名。** 测试可以点名机器人，算法不可点名。
11. **确定性。** 相同模型、配置和 seed 必须生成 byte-stable 或语义等价的 profile。
12. **四元数顺序显式。** 内部统一 `xyzw`；MJCF `quat` 解析为 `wxyz` 后转换；JSON 必须写 `quaternion_order`。
13. **不能靠后处理掩盖 IK 错误。** contact-aware IK 是主路径，feet stabilizer/grounding 是可配置 fallback。
14. **不修改或重写 Git LFS 机器人资产。** 只读分析；除非确实需要新增小型测试 fixture。

---

## 4. v2 总体架构

新增一条编译链：

```text
MJCF/URDF + minimal semantic map
        │
        ▼
RobotMorphologyAnalyzer
        │  joints / axes / limits / tree / geoms / symmetry
        ▼
SemanticChainResolver
        │  pelvis→torso, shoulder→hand, hip→foot, sites
        ▼
RestPoseCalibrator
        │  segment lengths / frame alignment / virtual sites
        ▼
ReachabilityAnalyzer
        │  Jv/Jw SVD, reachable basis, rank, confidence
        ▼
RetargetTaskCompiler
        │  normalized tasks, masks, priority, contact settings
        ▼
CompiledRetargetProfile v2
        │
        ▼
NewtonPipeline v2
        │  per-env tasks + temporal/contact state
        ▼
CSV + metrics + diagnostic report
```

### 4.1 建议新增模块

```text
soma_retargeter/
  robotics/
    morphology.py
    semantic_chains.py
    rest_pose_calibration.py
    reachability.py
    task_compiler.py
    retarget_profile.py
    config_validation.py
  pipelines/
    projected_ik_objectives.py
    direction_objectives.py
    temporal_objectives.py
    contact_manager.py
    retarget_metrics.py
  tools/
    autoconfigure_robot.py
    benchmark_retargeting.py
```

可按现有项目风格调整文件划分，但职责必须保持分离，不能把全部逻辑继续堆进 `robot_registry_parser.py` 或 `newton_pipeline.py`。

---

## 5. 数据模型

使用 typed dataclass；JSON 序列化字段稳定排序。

### 5.1 `JointDofInfo`

至少包含：

```python
@dataclass(frozen=True)
class JointDofInfo:
    joint_name: str
    body_name: str
    parent_body_name: str | None
    joint_type: str
    axis_local: np.ndarray
    axis_world_rest: np.ndarray
    q_index: int
    dof_index: int
    lower: float
    upper: float
    neutral: float
    continuous: bool
```

### 5.2 `SemanticSite`

```python
@dataclass
class SemanticSite:
    semantic_name: str
    body_name: str
    local_position: np.ndarray
    local_rotation_xyzw: np.ndarray
    source: str                 # explicit / inferred_child / geom_bounds / fallback
    confidence: float
    orientation_supported: bool
```

### 5.3 `KinematicChainProfile`

```python
@dataclass
class KinematicChainProfile:
    name: str
    root_body: str
    tip_site: str
    joint_names: list[str]
    segment_lengths: list[float]
    total_length: float
    translational_basis: np.ndarray   # 3 x rank
    rotational_basis: np.ndarray      # 3 x rank
    translational_rank: int
    rotational_rank: int
    singular_values_translation: list[float]
    singular_values_rotation: list[float]
    confidence: float
```

### 5.4 `TaskSpec`

```python
@dataclass
class TaskSpec:
    name: str
    task_type: str
    source_semantic: str
    target_site: str
    reference_site: str | None
    priority: int
    position_mask_or_basis: list[list[float]] | None
    rotation_mask_or_basis: list[list[float]] | None
    normalized_weight: float
    characteristic_length: float
    robust_loss: str
    enabled: bool
    reason: str
```

### 5.5 `CompiledRetargetProfile`

必须保存：

- `schema_version`;
- `compiler_version`;
- robot model fingerprint；
- source skeleton fingerprint；
- canonical quaternion order；
- morphology summary；
- semantic sites；
- chains；
- rest frame alignment；
- segment ratios；
- task specs；
- contact config；
- solver config；
- warnings；
- confidence；
- generation timestamp仅写入 report，不写入决定 cache hash 的 profile；
- source raw config path/hash。

---

## 6. 运动学分析

### 6.1 统一从运行时模型提取真实运动学

优先使用 Newton 已加载的模型和 FK 结果，避免只靠 XML 名称。MJCF/URDF parser 仅用于补充 geom、mesh、site 和 metadata。

需要提取：

- body tree；
- 每个 movable joint 的类型、轴、range、q/qd 索引；
- neutral/default q；
- rest FK；
- geom bounds；
- actuator 与 joint 对应关系；
- floating base；
- fixed body；
- 左右镜像候选。

### 6.2 语义链推断

输入优先级：

1. 用户明确 `ik_map`；
2. 已存在 profile/alias；
3. link/joint 名称 token；
4. body tree 拓扑；
5. rest pose 中的空间位置；
6. 左右对称匹配。

输出至少包含：

- pelvis/root；
- torso/chest；
- left/right shoulder；
- left/right elbow；
- left/right hand site；
- left/right hip；
- left/right knee；
- left/right foot site；
- 可选 head、toe、heel、sole edges。

置信度低于阈值时：

- 不猜测危险 task；
- 生成报告；
- CLI 返回“需要确认 semantic override”，但其余可安全生成部分仍写出。

### 6.3 左右对称检测

通过 joint subtree 拓扑、range、链长和 rest pose 镜像误差匹配左右链。对判定为对称的机器人：

- left/right segment length 使用平均值；
- scale tying；
- task weight tying；
- joint neutral prior tying；
- virtual site offset 镜像生成；
- 报告原始不对称误差。

---

## 7. 解析式 rest-pose 标定

### 7.1 禁止继续使用独立 geocentric joint scale

v2 的位置 target 必须沿语义骨骼逐段递归生成。

对于 source semantic edge `parent → child`：

\[
u_h(t) =
\frac{p_h^{child}(t)-p_h^{parent}(t)}
{\|p_h^{child}(t)-p_h^{parent}(t)\|+\epsilon}
\]

用 rest-frame 对齐矩阵 \(A_{edge}\) 映射方向：

\[
u_r(t)=A_{edge}u_h(t)
\]

使用机器人对应 semantic segment 的固定长度 \(L_r\)：

\[
p_r^{child,*}(t)=p_r^{parent,*}(t)+L_r u_r(t)
\]

不能让一个 joint 的 scale 改变另一侧链，也不能允许负 segment length。

### 7.2 frame alignment

从 source 和 robot rest pose 的稳定方向构造局部正交 frame：

- primary axis：parent→child；
- secondary axis：左右横向、bend plane 或另一条已知 edge；
- third axis：叉乘；
- 退化时使用 Kabsch/Wahba 多向量对齐；
- quaternion sign canonicalize，使 `w >= 0` 或使用固定 tie-break。

对齐：

\[
R_{offset}=R_{robot,rest}R_{source,rest}^{-1}
\]

不得用无界 quaternion gradient descent 来寻找本可解析求得的 offset。

### 7.3 虚拟末端 site

若 semantic hand/foot 不对应真实末端 body 原点：

1. 优先使用模型中显式 site；
2. 否则使用 distal child joint anchor；
3. 否则使用 geom/mesh bounds 沿链主轴的远端点；
4. 否则根据上一段方向和对称链估计；
5. 保存来源和 confidence。

无 wrist 的 Hand：

- site 位于 forearm distal end；
- position task 开启；
- orientation 默认关闭；
- 可达 rank > 0 时才允许 projected orientation；
- 必须同时生成 forearm direction 和 elbow pole task。

### 7.4 root motion scaling

root 水平位移使用机器人 leg length / source leg length 的稳健比例；竖直分量同时考虑：

- robot nominal pelvis height；
- stance foot；
- ground；
- crouch ratio。

不得仅用 mesh 总高度做所有部位统一比例。

---

## 8. 可达子空间分析

### 8.1 Jacobian

对每条 semantic chain，在 neutral/rest q 以及少量确定性采样姿态计算：

- translational Jacobian \(J_v\)；
- rotational Jacobian \(J_\omega\)。

采样至少包括：

- neutral；
- 每个 DoF 在 range 的 ±10% 或安全小扰动；
- 左右对称姿态。

对拼接后的 Jacobian 做 SVD：

\[
J = U\Sigma V^\top
\]

rank 阈值：

```python
sigma > max(abs_threshold, rel_threshold * sigma_max)
```

默认建议：

- `abs_threshold = 1e-6`;
- `rel_threshold = 1e-3`;

阈值进入 profile，并有单元测试。

### 8.2 旋转投影

人体相对旋转：

\[
R_h^{rel}=R_h^{parent^{-1}}R_h^{child}
\]

去掉 rest rotation 后取 Lie algebra：

\[
e_h=\log\left(R_{h,rest}^{rel^{-1}}R_h^{rel}\right)
\]

链的可达 projector：

\[
P_R=U_rU_r^\top
\]

只保留：

\[
e_h^{reachable}=P_Re_h
\]

robot target：

\[
R_r^{rel,*}=R_{r,rest}^{rel}\exp(e_h^{reachable})
\]

单 yaw 腰应自然得到 rank=1；不得通过 joint 名字或假定世界 Z 轴来实现。

### 8.3 projected relative rotation objective

首选实现 `IKObjectiveProjectedRelativeRotation`：

- child body；
- reference/parent body；
- per-env target relative quaternion；
- `3 x rank` basis 或 `3 x 3` projector；
- per-env weight；
- 3D rotation-vector residual；
- 支持 analytic Jacobian；若 Newton API 限制，允许 Warp autodiff，但不能退回完整 quaternion objective。

允许的工程 fallback：

- 先生成 projected relative target；
- 再根据 parent 的 target world rotation 合成 child world target；
- 使用标准 rotation objective。

fallback 仍必须有测试证明不可达 pitch/roll 不进入 target。

### 8.4 position subspace

当链对某 site 的平移 rank < 3 时，同样使用 \(P_T\) 投影位置误差。典型用途：

- 极低自由度 head；
- 简化手臂；
- 特殊脚链；
- torso position 不充分可控。

---

## 9. Task compiler

### 9.1 默认 semantic task

| 任务 | 表达 | 优先级 |
|---|---|---:|
| joint hard limits | barrier / clamp safeguard | 0 |
| stance sole anchors | world position, per-env contact lock | 0 |
| root/pelvis | world translation + reachable orientation | 1 |
| stance foot orientation | toe/heel/edge anchor geometry，不强制完整 quaternion | 1 |
| swing foot | position + reachable orientation | 2 |
| hand site | normalized position | 2 |
| torso | relative projected rotation | 2 |
| upper/lower limb | unit direction | 3 |
| elbow/knee bend | pole vector / bend-plane | 3 |
| head | relative projected rotation | 3 |
| temporal velocity | normalized q delta | 3 |
| temporal acceleration | normalized q second difference | 4 |
| neutral pose | range-normalized posture prior | 4 |
| symmetry/style | optional | 4 |
| self-collision | soft barrier，安全优先时提升 | 1–3 |

### 9.2 不再默认追踪中间 link 的绝对位置

`LeftArm`、`LeftForeArm`、`LeftLeg`、`LeftShin` 等中间语义点默认用于：

- direction；
- pole/bend plane；
- relative rotation 的可达部分。

只有末端、root、contact site 默认使用 absolute position。

### 9.3 direction objective

\[
e_d = u_r(q)\times u_r^*
\]

或使用 `1 - dot` 的稳健版本。必须：

- 归一化；
- 对接近零长度防护；
- 支持 per-env target；
- 左右链一致。

### 9.4 pole-vector objective

从 shoulder-elbow-hand 或 hip-knee-foot 构造 bend plane normal，避免肘/膝反向跳变。source bend plane 退化时：

- 使用上一帧；
- 再退化使用 robot neutral bend plane；
- 保存 fallback 计数。

### 9.5 任务无量纲化

位置：

\[
\hat e_p = e_p / \max(L_{chain}, \epsilon)
\]

关节：

\[
\hat e_q = (q-q_{ref})/\max(q_{upper}-q_{lower}, \epsilon)
\]

旋转直接使用 rad。每个 task 按 residual dimension 再除以 \(\sqrt{d}\)，防止三维 task 因维数天然占优。

### 9.6 robust loss

对普通 tracking 使用 Huber/Cauchy，减少异常 source frame 对整条链的冲击。接触和 hard safety task 不使用会放松约束的 robust tail。

---

## 10. 优先级求解

Newton 当前是加权 least squares。v2 实现可落地的 prioritized scheduler：

### 10.1 默认：分阶段 soft-lexicographic

每帧至少三阶段，均 warm-start：

1. **Feasibility pass**  
   joint limits、root、stance contact、ground/self-collision；
2. **Tracking pass**  
   保留第一阶段 task，高权重加入 hand、swing foot、torso projected rotation；
3. **Style pass**  
   保留前两层，加入 direction、pole、temporal、neutral。

任务残差已无量纲化，优先级 weight band 建议：

```text
priority 0: 1e4
priority 1: 1e3
priority 2: 1e2
priority 3: 1e1
priority 4: 1
```

编译器可以按 conditioning 自动整体缩放，但相邻 priority ratio 不得小于 10。保存最终 weight。

### 10.2 防止高优先级回退

每阶段结束记录高优先级 residual；下一阶段若导致其超过：

```text
previous_residual * (1 + tolerance) + absolute_tolerance
```

则：

- 回滚到上一阶段 q；
- 提高高优先级 band；
- 最多 retry 固定次数；
- report 记录。

如果 Newton 能稳定暴露 Jacobian/nullspace，可实现严格 HQP；但不得为追求 HQP 延迟基础 v2 完成。分阶段 residual guard 是本任务的最低要求。

---

## 11. 接触感知与足部稳定

### 11.1 contact source

支持：

- explicit `buffer.foot_contacts`;
- explicit `buffer.contacts`;
- NPZ `[L heel, L toe, R heel, R toe]`;
- SOMA heuristic；
- `auto` 优先 explicit，失败才 heuristic。

必须先从 **原始 buffer** 读取 explicit labels，再插入 initialization frames。

### 11.2 initialization frame 对齐

若原 motion contact score 为 `[T]`，插入 `N_init + N_stabilize` 后：

- prepend 第一真实帧的 contact score；
- 如果第一帧 source foot 高于 ground 且 explicit score 不可信，则使用 heuristic 决定；
- augmented score 长度必须与 augmented target 帧一致；
- 最终移除 warmup frames 后输出与原 motion 等长；
- 写测试覆盖。

### 11.3 per-environment objective weight

当前 `IKObjectivePosition.set_weight()` 是共享状态，不能在 batch 内按 env 更新。

实现新的 contact position objective：

- target position `[n_env, 3]`;
- weight `[n_env]`;
- link offset；
- residual 和 Jacobian 内乘对应 env weight；
- CUDA graph capture 后仍可更新 target/weight buffer；
- 不允许因为 `num_envs > 1` 禁用 contact-aware IK。

### 11.4 状态机

每个 env、每只脚、每个 anchor 独立保存：

- `contact_score`;
- `active`;
- `locked_world_target`;
- `frames_in_state`;
- `release_blend`;
- last valid target。

支持：

- on/off hysteresis；
- minimum on/off frames；
- touchdown target lock；
- release blend；
- clip reset；
- exception-safe state；
- missing anchor no-op。

### 11.5 sole anchors

优先 toe、heel，存在时加入 inner/outer edge。通过多个 position anchor 约束 foot yaw/roll，避免强制一个完整 foot quaternion。

### 11.6 与现有 post-processing 的关系

顺序：

1. contact-aware IK；
2. joint-limit safeguard；
3. virtual foot grounding 仅修正统一高度漂移；
4. legacy FeetStabilizer 仅作为显式 fallback。

默认不得同时让两个模块独立锁定同一 foot target。若共同启用，runtime 必须打印 warning 并选择单一 authority。

---

## 12. 时序、速度与关节安全

### 12.1 temporal objective

新增 per-env：

\[
e_v = D(q_t-q_{t-1})
\]

\[
e_a = D(q_t-2q_{t-1}+q_{t-2})
\]

其中 \(D\) 按 joint range、sample rate 和可选 motor velocity limit 归一化。floating root 与 actuated joints 使用不同配置。

### 12.2 velocity limit

如果模型有 actuator velocity limit，使用模型值；缺失时使用保守通用值，保存来源。限制要按 `dt` 生效，不能把统一 rad/s 常量直接当每帧 delta。

### 12.3 joint-limit objective

替换或修正当前 smooth filter，使用 joint-range normalized barrier。要求：

- 在安全 margin 内 residual 近零；
- 接近上下限单调增大；
- lower/upper 不对称时仍正确；
- continuous joint 不加有限 limit barrier；
- 解析 Jacobian 与数值差分匹配；
- 最终 hard clamp 只做数值安全，不应成为常态。

### 12.4 neutral prior

对弱可观测或低 leverage DoF 自动提高 neutral prior；对高优先级 task 必需的 DoF降低 neutral prior。neutral 来源：

1. MJCF/robot pose；
2. joint range center；
3. 显式 override。

---

## 13. 自碰撞与地面约束

实现轻量、可关闭、机器人无关的 soft collision：

1. 从 geom bounds 为主要 torso、upper/lower arm、thigh、shin 构造 sphere/capsule proxy；
2. 排除父子相邻 body；
3. 生成左右手臂与 torso、左右腿之间、手与腿等高风险 pair；
4. 距离小于 margin 时使用 smooth barrier；
5. 所有 pair、proxy、margin 写入 profile/report；
6. 解析失败时安全禁用并 warning，不能阻断 retargeting。

地面约束：

- stance sole anchors 不得穿地；
- swing foot 使用 soft ground barrier；
- root 高度不得被不可达 torso rotation 拉偏；
- ground height 来源明确。

---

## 14. 自动权重和置信度

不得按机器人保存手工 weight table。编译器根据以下因素生成：

- priority；
- task residual normalization；
- chain rank；
- chain conditioning；
- semantic confidence；
- contact state；
- site confidence；
- weakly observable DoF。

规则：

- rank=0 的 orientation task 直接禁用，不是把 weight 调小；
- rank=1/2 使用 projector；
- position site confidence 低时保留但降低 tracking priority，并输出 warning；
- semantic confidence 低于 hard threshold 时禁用危险 task；
- 左右 task weight 默认完全相同；
- 所有自动 weight 可被高级用户 override，但 override 不是正常接入要求。

---

## 15. v2 配置格式和向后兼容

### 15.1 最小用户配置继续支持

```json
{
  "ik_map": {
    "Hips": "base_link",
    "Chest": "torso_link",
    "LeftHand": "left_elbow_yaw_link",
    "RightHand": "right_elbow_yaw_link",
    "LeftFoot": "left_ankle_roll_link",
    "RightFoot": "right_ankle_roll_link"
  }
}
```

### 15.2 可选 v2 override

```json
{
  "schema_version": 2,
  "ik_map": {
    "Hips": "base_link",
    "Chest": "torso_link"
  },
  "semantic_overrides": {
    "LeftHand": {
      "body": "left_elbow_yaw_link",
      "site": "auto_distal",
      "orientation": "reachable_only"
    }
  },
  "autoconfig": {
    "enabled": true,
    "force_rebuild": false,
    "symmetry": "auto",
    "collision": "auto"
  }
}
```

### 15.3 runtime compiled profile 示例

```json
{
  "schema_version": 2,
  "compiler_version": "2.0",
  "quaternion_order": "xyzw",
  "robot_fingerprint": "...",
  "semantic_sites": {},
  "chains": {},
  "tasks": [],
  "contact": {},
  "solver": {},
  "warnings": []
}
```

### 15.4 cache

cache key 至少包含：

- MJCF bytes/hash；
- URDF bytes/hash（如使用）；
- minimal retarget config；
- robot T-pose/default q；
- compiler version；
- source skeleton version。

模型或代码变化必须自动失效旧 profile。

### 15.5 legacy scaler

- 读取 v1 scaler 继续支持；
- v2 默认不使用 v1 geocentric scaling；
- 检测负 scale、非有限值、巨大 asymmetry 时拒绝作为 v2 input；
- 提供一次性 migration report；
- 不自动覆盖用户旧文件，除非 `--force`；
- included demo robot 可以提交确定性生成的 v2 profile。

---

## 16. 具体文件改动

### 16.1 `soma_retargeter/robotics/human_to_robot_scaler.py`

重构：

- 保留 v1 path 为 `LegacyHumanToRobotScaler` 或明确 `mode="legacy"`；
- 新增 segment-local target builder；
- 不再修改加载后的 JSON dict in place；
- 不允许负长度；
- 支持 semantic sites 和 virtual sites；
- batch 与 single frame 共享同一数学实现；
- 添加 shape/type validation。

### 16.2 `soma_retargeter/robot_registry_parser.py`

职责缩减为：

- registry/profile path；
- raw config load；
- v1/v2 schema dispatch；
- compiled profile cache；
- legacy passthrough。

不要继续在这里实现复杂几何与 Jacobian 分析。

必须修复：

- contact config passthrough；
- teacher metadata 不进入推荐 runtime；
- virtual sole anchor → contact config；
- deterministic fingerprint；
- invalid config diagnostics。

### 16.3 `soma_retargeter/pipelines/newton_pipeline.py`

重构为：

- compiled task profile 驱动；
- task factory；
- per-env target/weight buffers；
- prioritized passes；
- contact manager；
- temporal state；
- metrics hooks；
- exception-safe execute；
- `clear()` 完整清理 per-env state；
- 不再对所有 mapped joints 无条件创建 position+rotation pair；
- 不再在 batch 模式禁用 contact；
- CUDA graph capture 支持动态 buffer 更新。

### 16.4 `soma_retargeter/pipelines/ik_objectives.py`

保留兼容类，新增或拆分：

- normalized joint-limit barrier；
- per-env weighted position；
- projected relative rotation；
- direction；
- pole vector；
- temporal q；
- ground/collision barrier。

每个自定义 objective 要有：

- residual dimension；
- analytic/autodiff 支持声明；
- finite-difference Jacobian test；
- CPU/CUDA smoke test（CI 无 GPU 时允许 skip CUDA）。

### 16.5 `app/optimize_scaler_config.py`

降级为可选 residual calibration：

- 默认使用 v2 compiled profile；
- scale 使用 `exp(raw)` 或直接固定解析 segment length；
- left/right tie；
- bounded translation offset；
- tangent-space rotation delta；
- identity/zero prior；
- robust loss；
- train/validation；
- 验证失败自动回滚；
- report before/after；
- 不允许输出负 scale。

### 16.6 `app/pose_optimizer_ui.py`

默认入口改为：

- 显示 morphology；
- 显示 semantic confidence；
- 显示链 rank/basis；
- 生成 v2 profile；
- 运行 benchmark；
- pose-pair refinement 放到 Advanced/Legacy 区域。

### 16.7 新 CLI

实现：

```bash
python -m soma_retargeter.tools.autoconfigure_robot --robot roboparty_rpo
python -m soma_retargeter.tools.autoconfigure_robot --robot unitree_g1 --force
python -m soma_retargeter.tools.autoconfigure_robot --robot e3_v2 --validate-only
```

参数：

- `--dry-run`;
- `--force`;
- `--validate-only`;
- `--write-report`;
- `--benchmark`;
- `--seed`;
- `--strict`;
- `--output`。

返回码：

- 0 成功；
- 2 需要用户确认低置信度语义；
- 3 模型/配置无效；
- 4 benchmark gate 失败。

### 16.8 README

更新新机器人接入流程：

1. 注册 MJCF/XML；
2. 填 minimal `ik_map`；
3. 运行 autoconfigure；
4. 查看 report；
5. 运行 demo。

不得再把“制作多组 pose pair 并手动优化”写成默认路径。

---

## 17. RoboParty RPO 必须达到的行为

1. 丢弃当前负尺度 scaler 作为 v2 初值；
2. 解析生成正 segment length；
3. 左右 upper arm、forearm、thigh、shin 默认 tied；
4. `Chest` 使用 pelvis-relative projected rotation；
5. 自动识别 torso chain rotational rank；若为 1，只跟踪真实轴分量；
6. Hand 使用 forearm distal virtual site；
7. Hand orientation rank=0 时关闭；
8. elbow 使用 direction + pole；
9. foot 使用 sole anchors/contact；
10. source torso pitch/roll 不得通过髋、膝和 floating root 大幅补偿；
11. report 明确打印：
    - torso rank；
    - torso basis；
    - wrist absent；
    - hand site source；
    - disabled orientation reason；
    - scale symmetry；
    - contact mode。

---

## 18. G1 23DoF、G1 29DoF、E3 v2、OLI

### 18.1 G1 29DoF

作为较高 DoF baseline：

- v2 不得显著劣化现有 sample motion；
- 腰 3DoF 时 torso rank 应正确检测；
- wrist 存在时 hand projected orientation 可开启；
- joint limits 和 contact 正常。

### 18.2 G1 23DoF

禁止写 G1 23 特例。通过模型自动识别：

- 缺少的 waist/wrist DoF；
- chain rank；
- orientation projector；
- virtual site；
- neutral prior。

若当前仓库只有 29DoF asset，添加一个小型测试 fixture 或通过锁定/移除 DoF 构造 23DoF capability test；不要下载或提交未经许可的资产。

### 18.3 E3 v2

用于验证：

- 简化 hand map；
- torso capability；
- minimal map；
- deterministic compile。

### 18.4 逐际动力 OLI

执行时先检查 `params.py` 和资产目录：

- 资产已存在：完成注册、autoconfig、测试和 demo；
- 资产不存在：不得伪造或联网下载；在 report 中保存所需文件清单和 semantic mapping 模板，核心算法完成不受阻。

---

## 19. 测试要求

使用 `pytest`；保留现有 `unittest` 测试也可以，但统一纳入 CI。

### 19.1 数学单元测试

新增：

- quaternion order/roundtrip；
- SO(3) log/exp；
- projector idempotence；
- rank threshold；
- single-yaw chain rank=1；
- projected pitch/roll≈0；
- segment-local scaling 长度恒定；
- positive scale；
- symmetry tying；
- direction residual；
- pole-vector退化；
- joint-limit barrier 单调性；
- analytic Jacobian vs finite difference；
- per-env weight isolation；
- temporal residual；
- cache fingerprint。

### 19.2 contact 回归

必须覆盖所有未解决 review：

- explicit contact 从原始 buffer 读取；
- initialization frames 后长度正确；
- multi-env 每个 clip contact state 不串扰；
- batch 不禁用 contact；
- execute 异常后状态不污染；
- missing LeftFoot/RightFoot 不崩；
- airborne stationary foot 低 contact；
- grounded fast-moving foot 低 contact；
- hysteresis 和 minimum duration；
- release blend；
- clear/reset。

### 19.3 synthetic robot fixtures

创建不含大资产的最小 MJCF：

1. floating base + yaw-only torso；
2. 3DoF torso；
3. 3DoF shoulder + 1/2DoF elbow，无 wrist；
4. 有 wrist；
5. 简化腿 + ankle；
6. 非对称测试机器人。

这些 fixture 用于验证算法，不依赖真实 robot mesh。

### 19.4 integration

对仓库已有 sample motions：

- RPO；
- G1；
- E3 v2；
- 其它已注册机器人。

至少运行：

- T-pose；
- natural down；
- arms forward；
- elbow bend；
- squat；
- walking/dance sample；
- torso twist；
- 大幅 pitch/roll source motion。

### 19.5 batch parity

同一个 clip：

- 单 env；
- 2 env 相同 clip；
- 2 env 不同 contact phase。

在确定性设置下，单 env 与 batch 对应结果最大 joint 差异应小于 `1e-4` 或给出合理的数值容差说明。

---

## 20. 指标与验收门槛

所有指标以机器人链长、身高和 rad 归一化，避免不同尺寸不可比。

### 20.1 配置健康

硬门槛：

- 无 NaN/Inf；
- 无负 segment length/scale；
- joint range 合法；
- quaternion 单位化；
- symmetric robot 的左右 segment mismatch，经 tying 后 ≤ 2%；
- cache 可复现；
- profile schema validation 通过。

### 20.2 不可达误差泄漏

yaw-only torso fixture：

- source 纯 pitch 或 roll 时，compiled torso target 的不可达分量范数 ≤ 原始分量的 1%；
- waist yaw response ≤ 对纯 yaw response 的 5%；
- lower-body compensation norm 相比 legacy 降低至少 50%。

### 20.3 末端 tracking

相对于 legacy baseline：

- hand/foot normalized position RMSE 不得整体恶化超过 5%；
- RPO hand 使用 virtual site 后应改善至少 15%，或 report 给出受限于链 reach 的解释；
- joint limit violation frame count不得增加。

### 20.4 接触

在有明确 contact 的 motion 上：

- stance foot slide distance 至少降低 30%；
- foot penetration frame count 不增加；
- touchdown/release 无明显单帧跳变；
- batch 与 single 指标一致。

### 20.5 平滑

- q velocity/acceleration 的 95th percentile 不高于 legacy；
- 不允许通过强低通导致末端误差显著增加；
- contact transition 处无高频抖动。

### 20.6 性能

记录 CPU/GPU：

- compile 是离线步骤；
- runtime 相比 legacy 单帧 IK 平均耗时增幅目标 ≤ 25%；
- 若超过，必须 profile 并在 report 解释；
- 不以牺牲正确性换取门槛。

---

## 21. 基准评测工具

实现：

```bash
python -m soma_retargeter.tools.benchmark_retargeting \
  --robots roboparty_rpo unitree_g1 e3_v2 \
  --motions assets/motions/bvh \
  --compare legacy v2 \
  --output artifacts/retargeting_v2
```

输出：

```text
artifacts/retargeting_v2/
  benchmark_summary.json
  benchmark_frames.csv
  environment.json
  commands.txt
  per_robot/
    roboparty_rpo.json
    unitree_g1.json
    e3_v2.json
  failures/
    <case>.json
```

不提交大型 CSV motion、视频或二进制缓存；提交小型指标 JSON/CSV、必要的图和失败摘要。

指标至少包括：

- task residual by type/priority；
- joint limit margin；
- foot slide；
- penetration；
- root tilt；
- torso reachable/unreachable residual；
- hand/foot RMSE；
- velocity/acceleration；
- solver iterations；
- runtime；
- fallback counts；
- confidence/warnings。

---

## 22. 必须保存的下一步分析信息

Codex 完成后，分支上必须有：

```text
docs/retargeting_v2/
  architecture.md
  implementation_report.md
  design_decisions.md
  known_limitations.md
  migration.md
artifacts/retargeting_v2/
  benchmark_summary.json
  environment.json
  commands.txt
  per_robot/*.json
```

### 22.1 `implementation_report.md`

记录：

- 完成的 commit；
- 修改文件；
- 每个设计项实现位置；
- 未完成项及原因；
- 测试命令和结果；
- benchmark 摘要；
- before/after；
- 性能；
- warning；
- 后续优先级。

### 22.2 `design_decisions.md`

每个重要决策使用 ADR 风格：

- 问题；
- 候选方案；
- 选择；
- 原因；
- 风险；
- 验证方法。

至少包含：

- projected target vs custom relative objective；
- priority scheduler；
- per-env contact weight；
- virtual site 推断；
- symmetry tying；
- collision proxy；
- cache。

### 22.3 `environment.json`

包含：

- OS；
- Python；
- package versions；
- Newton/Warp；
- GPU/driver；
- git commit；
- seed；
- CLI 参数；
- asset fingerprint。

### 22.4 失败也要落盘

任何机器人或 motion 失败时保存：

- exception；
- stack；
- semantic profile；
- task compilation warnings；
- frame index；
- solver residual；
- reproduction command。

不得仅在终端打印后丢失。

---

## 23. 迁移和默认行为

1. v2 autoconfig 为新注册机器人的默认路径；
2. 已有 robot 可通过 config flag 暂时选 legacy；
3. CLI 输出清楚标注当前使用 `legacy` 或 `v2`；
4. 老 CSV 字段顺序不变；
5. 老 viewer 不崩；
6. 老 raw config 字符串 `ik_map` 可直接编译；
7. 旧 scaler 不删除；
8. RPO/G1/E3 提供 generated v2 profile 或首次运行自动生成；
9. 对当前 RPO 的负 scale 发出明确 migration warning；
10. pose pair optimizer 不再在 README 中作为默认接入步骤。

---

## 24. 推荐提交顺序

Codex 应以可审查的小提交推进，但不要停在中间：

1. `fix: harden contact labels and per-env contact state`
2. `test: add synthetic morphology fixtures`
3. `feat: add morphology and semantic chain analysis`
4. `feat: add rest calibration and virtual sites`
5. `feat: add reachability analysis and projected tasks`
6. `feat: add segment-local target generation`
7. `feat: add normalized prioritized IK tasks`
8. `feat: add temporal, limit, ground and collision objectives`
9. `feat: compile and cache retarget profile v2`
10. `feat: add autoconfig and benchmark CLI`
11. `refactor: integrate v2 pipeline with legacy fallback`
12. `docs: add migration, reports and benchmark results`

每个提交后运行相关测试。最终 push 当前分支全部提交。

---

## 25. 明确禁止的“捷径”

以下不算完成：

- 只把 RPO 两个负 scale 改成正数；
- 只给 Chest 降权；
- 只增加更多 pose pair；
- 只复制 GMR 的手工 per-robot JSON；
- 用机器人名字 if/else；
- batch 时关闭 contact；
- 将 Hand 从 map 删除而不生成 virtual site/direction/pole；
- 只做后处理滤波；
- 只写 config schema 不接入 runtime；
- 只加 unit tests 不跑 integration；
- 只生成漂亮 demo，不保存指标；
- 测试因缺依赖失败后仍宣称通过；
- 只提交计划或 TODO。

---

## 26. 完成定义

只有同时满足以下条件才算完成：

- [ ] 所有第 2.6 节 contact review 缺陷已修；
- [ ] v2 morphology compiler 可从 minimal map 生成 profile；
- [ ] segment-local scaling 替代默认 geocentric scaling；
- [ ] yaw-only torso 使用真实 Jacobian basis 投影；
- [ ] 无 wrist 自动生成 virtual hand site；
- [ ] direction 和 pole-vector task 接入；
- [ ] per-env contact weight 在 batch 正常；
- [ ] temporal、joint limit、ground、collision 至少有可测试实现；
- [ ] prioritized scheduler 接入；
- [ ] v1 backward compatibility；
- [ ] RPO、G1、E3 integration 完成；
- [ ] OLI 资产存在时完成；不存在时留结构化接入报告；
- [ ] 所有测试实际运行，失败被修复或诚实记录；
- [ ] benchmark 和分析 artifacts 已提交；
- [ ] README 和 migration 更新；
- [ ] 当前分支所有提交已 push；
- [ ] 最终 implementation report 能让下一位工程师无需重新阅读聊天记录继续分析。

---

## 27. Codex 开始执行时的第一步

1. 检查当前 branch 和 HEAD；
2. 检查 `dev` 上 PR #1/#2 的实现与 unresolved review；
3. 运行现有 tests，记录真实环境；
4. 对当前 RPO scaler 运行 validator，保存负 scale/asymmetry baseline；
5. 建 synthetic yaw-only/no-wrist fixture；
6. 先修 contact 数据和 batch correctness；
7. 再按推荐提交顺序完成全链路；
8. 不需要等待额外确认，遇到资产缺失时按本文件 fallback 继续推进。
