# SOMA Retargeter vNext：覆盖 `main` 的一次性大版本重构目标

> 本文件是交给 Codex 的**执行规格**，不是讨论稿，也不是待拆分的路线图。
>
> 基线：审计时 `main` 的父提交为 `68a3d701b99477ddbb2d99df4d38fc034e1f0d28`。实际实施时以 `git pull` 后的最新工作树为准，**不要 reset 回该 SHA**。

---

## 0. Codex 执行约束

1. 直接完成整个 vNext 迭代，不要只输出计划、建议、伪代码、TODO、空壳接口或“下一步再做”。
2. 不要把工作拆成“先只优化 scale、再只优化 translation、最后只优化 rotation”的参数冻结阶段。所有可训练的 scaler 参数必须在每次联合优化迭代中同时处于激活状态。
3. 可以在内部组织多个提交或模块，但交付结果必须是一个可从当前 `main` 直接运行、测试和迁移的完整版本。
4. 允许重构内部 API；当前用户入口必须保留兼容包装：
   - `python ./app/bvh_to_csv_converter.py --viewer gl --robot <name>`
   - `python ./app/bvh_to_csv_converter.py --viewer null --robot <name>`
   - `python ./app/pose_optimizer_ui.py --viewer gl --robot <name>`
5. 不能静默吞掉错误或用默认值掩盖坏输入。配置、骨架、姿态、BVH、CSV、MJCF 映射失败时，必须给出可定位的错误和机器可读报告。
6. 不要破坏或重写现有 Git LFS 大文件；不要把运行生成的巨型数据、缓存、模型、临时 CSV、图像或 benchmark 全量输出提交进仓库。
7. 所有写文件操作必须使用临时文件 + 原子替换；覆盖已有 scaler/config 前必须生成可识别备份，或默认写入新版本路径并由用户显式 `--in-place` 才覆盖。
8. CPU 环境必须完成核心测试。GPU 不可用时允许跳过 GPU 专项测试，但必须有明确 skip 原因；不得伪称 GPU 测试已通过。
9. 新增依赖必须有明确用途，锁文件同步更新，许可证和安装文档同步更新。优先复用 NumPy、SciPy、Warp、Newton；不要为了简单校验引入重型框架。
10. 完成后生成 `docs/vnext/implementation_report.md`，记录实现摘要、迁移说明、实际执行的测试、基准结果、未验证的硬件项。不得把本规格中的必做项降级为“已知限制”。

---

## 1. 最终产品目标

把当前项目从“能针对少数机器人运行的脚本集合”升级为一个具备以下性质的统一重定向系统：

- 单一、明确且可测试的坐标系/单位/四元数数学契约；
- 同一套 target generator 同时服务运行时重定向、静态姿态标定、报告与测试；
- scaler 的比例、平移 offset、旋转 offset 联合优化，带物理约束、验证集、最佳 checkpoint 和回滚；
- 严格的版本化配置与迁移层，既支持内置机器人，也支持用户机器人；
- 稳定的多目标 IK、正确 Jacobian、流形感知的关节处理、真正基于接触状态的足底锁定；
- 可恢复、可并行、逐文件隔离失败的批处理；
- 完整 CLI、GUI 兼容入口、测试、CI、benchmark、数学文档和迁移文档；
- 任何一次优化或批处理都能产出包含输入指纹、配置、指标、失败项和产物哈希的可复现实验报告。

不追求对现有内部实现做最小改动；优先保证数学正确性、可验证性和新机器人泛化。

---

## 2. 必须固定下来的数学契约

新增 `docs/vnext/math.md`，以下定义必须与代码、配置 schema、测试一致。代码中不得再散落互相矛盾的隐式约定。

### 2.1 基本表示

统一采用：

- 长度：米；
- 时间：秒；
- 角度：内部弧度；
- 四元数：`xyzw`；
- 四元数必须单位化，并采用同半球规范化以保证时间连续性；
- 刚体变换记作 `T = (p, q)`。

变换复合定义为：

```text
T1 ⊗ T2 = (p1 + R(q1) p2, normalize(q1 q2))
T^-1 = (-R(q)^T p, q^-1)
```

禁止在“主动旋转/被动换基”“左乘世界旋转/共轭换坐标系”之间含糊切换。所有转换 API 名称必须表明语义，例如：

- `rotate_world_transform`
- `change_basis_transform`
- `convert_point_between_frames`

### 2.2 BVH 局部姿态

轴角到四元数：

```text
q(u, θ) = [u sin(θ/2), cos(θ/2)]
```

BVH 的旋转必须严格按每个 joint 在 `CHANNELS` 中声明的旋转通道顺序复合：

```text
q_local = q_axis_1(θ1) q_axis_2(θ2) ... q_axis_k(θk)
```

平移通道也必须按通道名称映射到 X/Y/Z，不能把读取到的前三个值直接假设成 XYZ。支持 0～3 个平移通道、0～3 个旋转通道，并对不支持的重复轴/异常布局给出明确错误。

BVH 文件没有单位元数据。vNext 必须通过配置显式声明 `source_length_unit`，默认兼容现有 SOMA 的 `centimeter`，并允许 `meter`、`millimeter`；报告中必须记录最终换算比例。

### 2.3 正向运动学与反向局部化

对拓扑有序骨架：

```text
G_root = X_world ⊗ L_root
G_i    = G_parent(i) ⊗ L_i

L_root = X_world^-1 ⊗ G_root
L_i    = G_parent(i)^-1 ⊗ G_i
```

`Skeleton` 构造时必须验证：名称唯一、父索引合法、恰有预期根节点、无环、父节点先于子节点或内部完成拓扑排序、transform 为 `(N, 7)`、数值有限、四元数可归一化。

### 2.4 坐标系转换

若 `C` 是源基到内部基的旋转：

```text
p_internal = R(C) p_source
q_internal = C q_source C^-1
```

完整 transform 的换基是共轭语义，而不是只左乘：

```text
T_internal = (0, C) ⊗ T_source ⊗ (0, C)^-1
```

若只想给整个场景施加世界旋转，应提供另一个明确 API。现有 Maya/Mujoco 字符串需大小写归一化，并纳入 versioned convention 配置。

### 2.5 当前 scaler 公式与兼容模式

现有实现的 target effector 为：

```text
ρ = model_height / human_height_assumption
s_i = ρ * configured_scale_i

r = p_source_root
r_scaled = r ⊙ s_root
q_target_i = q_source_i q_offset_i
p_target_i = (p_source_i - r) ⊙ s_i
             + r_scaled
             + R(q_target_i) t_offset_i
```

当 `scale_animation=False` 时，现有实现实际使用 `(1, 1, s_i)`，这仍是按 joint 的 Z 缩放，不是“完全不缩放”。

vNext 必须保留只读的 `geocentric_v1` 兼容模式用于旧配置复现，但新配置和新优化器不得继续默认使用这种“每个 effector 独立相对根缩放”的模型。

### 2.6 vNext 默认的层级 scaler

默认使用 `hierarchical_v2`。对映射后的有向树，设 `a_i > 0` 是从映射祖先到 joint `i` 的 segment scale：

```text
p_hat_root = S_root(p_source_root)

v_i = p_source_i - p_source_parent_mapped(i)
p_hat_i_without_offset = p_hat_parent_mapped(i) + a_i v_i

q_hat_i = q_source_i q_offset_i
p_hat_i = p_hat_i_without_offset + R(q_hat_i) t_offset_i
T_hat_i = (p_hat_i, q_hat_i)
```

要求：

- 同一 segment 只缩放一次，保持层级闭合；
- 映射跳过中间 source joint 时，`v_i` 使用最近 mapped ancestor；
- root 平移策略独立配置，不再复用某个普通 joint scale；
- `height_only` 模式使用单一全局尺度/竖直根策略，不允许不同 joint 各自缩放 Z；
- runtime、optimizer、preview、metrics 必须调用同一个实现，禁止复制公式。

### 2.7 IK 残差与 Levenberg–Marquardt

每帧求解变量为机器人 configuration `q`。位置残差：

```text
r_pos_i(q) = w_pos_i (p_target_i - p_body_i(q))
```

旋转残差必须在 SO(3) 切空间中计算。令：

```text
q_err = q_body_i(q) q_target_i^-1
```

先选择短弧同半球，再计算：

```text
φ = Log(q_err)
r_rot_i(q) = w_rot_i φ
```

关节限位残差：

```text
r_limit_j(q) = w_limit [max(0, q_j-u_j) + max(0, l_j-q_j)]
```

总目标：

```text
min_q  1/2 ||r(q)||²
```

LM 步：

```text
(JᵀJ + λD) Δ = -Jᵀr
q_next = IntegrateOnManifold(q, Δ)
```

`D` 可为单位阵或对角缩放，但必须记录；接受/拒绝步后自适应阻尼。不得把自由关节或球关节四元数当普通标量数组逐元素线性更新/截断。

### 2.8 平滑关节过滤器的正确导数

当前代码的自定义 smooth filter 近似为：

```text
c = (l+u)/2
e = q-c
s(e) = 0                                      if e in safe interval
     = 1-exp(-(m d(e))^p)                    otherwise
r(e) = w mask e s(e)
```

其精确导数应为：

```text
dr/de = 0                                    in safe interval
dr/de = w mask [s(e) + e ds/de]              otherwise
```

当前 analytic Jacobian 无条件写 `w*mask`，与残差不一致，是 P0。vNext 必须：

- 修复为精确解析导数，或移除该 objective，使用经验证的平滑 barrier；
- 用中心有限差分和 autodiff 双重测试 analytic Jacobian；
- padding 用关节范围的相对比例，不使用固定 `1.02 rad`；
- 在读取 `coord_masks[coord_idx]` 前先检查 `coord_idx >= 0`。

### 2.9 两段式 IK

保留时必须处理退化。当前核心关系为：

```text
l_ab = ||b-a||
l_bc = ||c-b||
l_at = clamp(||t-a||, ε, l_ab+l_bc-ε)

α = acos((l_ab²+l_at²-l_bc²)/(2 l_ab l_at))
β = acos((l_ab²+l_bc²-l_at²)/(2 l_ab l_bc))
```

弯曲轴来自 hint 或叉积。零长度骨骼、共线 hint、叉积近零、目标不可达都必须有稳定 fallback，不能归一化零向量。

### 2.10 当前标定损失与 vNext 联合目标

当前三阶段标定分别使用：

```text
L_bone = Σ_i w_i (||p_i-p_parent||-||p_i*-p_parent*||)²

L_pos = Σ_i w_i ||(p_i-p_root)-(p_i*-p_root*)||²

L_rot = Σ_i w_i [1-(q_i·q_i*)²]
```

最后一项等于 `sin²(Δθ/2)`，在 180°附近梯度退化。vNext 改为 SO(3) geodesic/robust loss。

联合标定变量必须重新参数化：

```text
a_i = a_min + (a_max-a_min) sigmoid(z_i)
t_i = t_i0 + bounded_delta(u_i)
Q_i = Q_i0 Exp(ω_i)
```

所有 `z_i, u_i, ω_i` 在每次迭代中同时更新。默认边界可配置，必须保证：

- scale 严格为正；
- offset 有每 joint/全局幅度上限；
- rotation 在切空间更新，避免直接对四元数四维分量做欧氏梯度后归一化；
- 左右对称 joint 可共享先验而不是被强制完全相同。

统一损失至少包含：

```text
L = λp L_position_robust
  + λR L_rotation_geodesic_robust
  + λb L_bone
  + λsym L_left_right_symmetry
  + λprior L_parameter_prior
  + λrange L_soft_bounds
  + λval L_runtime_validation_score
```

其中 `L_runtime_validation_score` 可以作为定期黑盒评估和最佳 checkpoint 选择指标，不强制对 Newton IK 反向传播。训练损失与 runtime target generator 必须一致。

### 2.11 接触与足滑

当前 `FeetStabilizer` 只是把腿再做一次 two-bone IK，并不等同于接触锁定。vNext 必须实现显式接触状态机：

- 进入条件：足端高度、线速度、可选垂直速度低于 enter 阈值；
- 退出条件使用更宽的 exit 阈值，形成 hysteresis；
- 接触开始时保存世界 anchor；
- 接触期间目标位置/朝向向 anchor 约束，并在可行范围内调整 pelvis/root；
- 双脚接触冲突时按置信度与可达性加权；
- 输出 contact mask、足滑距离、足滑速度、穿地深度指标。

---

## 3. 已确认的问题清单

以下不是“建议”，而是必须在本次大版本中解决并用回归测试锁住的缺陷。

### P0：会产生错误结果、崩溃或不可相信配置

- [ ] `soma_retargeter/configs/roboparty_rpo/soma_to_rpo_scaler_config.json` 已存在负 scale（右前臂、右手），证明优化器缺少正值约束；迁移时必须拒绝或修复并报告。
- [ ] `pipelines/ik_objectives.py` 的 smooth residual 与 analytic Jacobian 不一致。
- [ ] `utils/newton_utils.py::create_buffer_with_initialization_frames` 在初始化帧为 1～3 时存在空列表访问或除零；覆盖 `N=0..5` 及更大值测试。
- [ ] `robot_registry_parser.build_runtime_retargeter_config` 丢弃 raw config 中除字符串 `ik_map` 外的 solver/postprocess/weight 设置，并强制关闭 postprocess；改成 schema merge，不能静默覆盖用户配置。
- [ ] BVH 平移通道顺序被隐式当作 XYZ，且固定假设三旋转通道；重写严格解析层。
- [ ] BVH 固定乘 `0.01`，没有单位契约；改为显式 source unit。
- [ ] CUDA graph/solver warm-up 在真实首帧 target 设置前执行 `step()`，可能改变初值并造成 CPU/GPU 首帧差异；capture 前后必须保存/恢复状态，并有一致性测试。
- [ ] batch 只比较 joint 数量，并把缺失 joint 静默填 reference pose；必须使用 skeleton fingerprint/兼容性报告，默认严格拒绝。
- [ ] `JointLimitClamper` 通过 `coord0+k` 把 manifold configuration 当标量，对 free/ball joint 可能截断四元数分量；改成 joint-type-aware 限位/投影。
- [ ] UI 默认 `output_file == input scaler config`，会直接覆盖输入；默认改为新文件 + backup/显式 in-place。
- [ ] 运行时 scaler 和优化器有公式复制，未来容易漂移；合并为唯一 target generator。
- [ ] 无 tests、无 CI、无 golden baseline，现有结果无法证明没有回归。

### P1：泛化、稳定性、性能和可维护性问题

- [ ] `HumanToRobotScaler` 假设 mapped joint 第 0 个是 root，并硬编码 Toe/ToeBase alias；改为显式 root 和可配置 alias。
- [ ] `HumanToRobotScaler` 的 mapped parent 用 list `.index()`，配置不完整时晚崩；初始化时一次性验证整个映射图。
- [ ] `scale_animation=False` 仍按 joint 缩放 Z，语义错误。
- [ ] `create_joint_coord_masks` 把 body index 当 joint index；使用 Newton 明确的 body↔joint 映射。
- [ ] `FeetStabilizer.reset_state` 复制到 `self.joint_q` 后，FK 却使用 `self.model.joint_q`；明确数组别名/复制契约并测试。
- [ ] 两段式 IK 对零长度、共线和不可达目标缺少稳定 fallback。
- [ ] pipeline 在 offsets 数量不匹配时静默把全部 offset 换成 identity；改为严格校验。
- [ ] postprocess 硬编码 `LeftFoot`/`RightFoot` 名称；从 semantic role/config 解析。
- [ ] 空输入时 `execute()` 返回语义不一致；统一返回空列表和完整报告。
- [ ] 较短 batch env 在结束后继续使用陈旧 target 求解；做 active mask/按长度分组。
- [ ] 每帧/每阶段频繁 `.numpy()` 再传回设备；减少 host-device round trip，预分配 target buffers。
- [ ] collision 配置字段存在但未生效；实现可验证的 collision objective/diagnostic，或迁移时明确报不支持，不能保留死字段。
- [ ] 初始化 pose 的 world/root transform 没有完整参与插帧。
- [ ] `Skeleton` 缺少唯一名、拓扑、有限值、shape、四元数校验。
- [ ] `SkeletonInstance.set_local_transforms` 直接持有调用者数组，存在别名修改风险。
- [ ] CSV 读写没有列数、header、有限值、四元数、joint order schema 校验。
- [ ] legacy Euler CSV 经过 quaternion↔Euler 会有分支/gimbal ambiguity；报告中必须声明并优先推荐 quaternion NPZ-compatible schema。
- [ ] optimizer 的 `scale_translation` 接受多个 BVH 却只使用第一个，并仅取第 0 帧；重做 dataset/sampler。
- [ ] 无 retargeter config 时使用虚构的 `+5 cm` target；删除该行为，缺目标应明确失败或进入显式 synthetic-demo 模式。
- [ ] optimizer 没有 finite guard、gradient clipping、best checkpoint、early stopping、验证集、rollback、seed、前后指标。
- [ ] optimizer 报告只记录元数据，不记录 loss 曲线、每 joint 误差、端到端指标、输入哈希。
- [ ] optimizer/UI 的 rotation 参数字段存在，但调用时又硬编码为 `iterations/lr/1.0`。
- [ ] GUI 在主线程同步优化，界面冻结；改为可取消 worker，并实时显示线程安全的进度/日志。不要让 viewer/Newton 对象跨线程不安全使用。
- [ ] `params.py` 通过 import 执行任意 Python，且 `lru_cache` 让进程内修改不生效；明确本地可信边界，新增数据化 profile，legacy adapter 支持显式 reload。
- [ ] alias 递归无环检测。
- [ ] “读取配置”会隐式生成/修改文件；把 pure read 与 explicit `init/generate` 命令分开。
- [ ] `link_scaler_config_to_retargeter` 名称像写入，实际什么也没写；改名或真正更新 profile。
- [ ] robot model height 推断只覆盖部分 mesh/STL，不覆盖 primitive geom、include、所有姿态；输出置信度/来源，无法可靠推断时要求显式配置，而不是静默回退 1.70。
- [ ] batch 一个坏文件会中止整批；逐文件隔离、manifest、resume、失败汇总。
- [ ] batch size 未验证、扩展名大小写敏感、输入输出重叠无保护、写出非原子。

### P2：产品化与工程完整性

- [ ] wheel 安装只打包 `soma_retargeter*`，却把主要入口放在排除的 `app/`，并依赖仓库根 `params.py`/`assets`；修复 clean-wheel 安装。
- [ ] 增加正式 console scripts，不再要求用户知道源码目录布局。
- [ ] README 末尾临时命令、旧格式说明、中文手册需要统一。
- [ ] 所有 public config/pose/report 增加 `schema_version`。
- [ ] 所有错误增加稳定 error code；CLI 支持 `--json-report`。
- [ ] 为大 BVH/XML/STL 加可配置资源上限，拒绝 NaN/Inf 和异常尺寸，避免不受控内存/计算消耗。
- [ ] 对 batch 输出路径做 resolve 后的根目录约束，避免 symlink/路径逃逸。

---

## 4. 目标架构与单一事实来源

允许等价命名，但职责必须清楚。推荐结构：

```text
soma_retargeter/
  core/
    conventions.py       # units, frames, quaternion/SE(3) contract
    skeleton.py          # validated immutable skeleton + fingerprint
    transforms.py        # pure NumPy/Warp-compatible math
  config/
    models.py             # typed v2 config dataclasses
    validation.py         # semantic validation + stable error codes
    migration.py          # v1 -> v2 and legacy params adapter
    registry.py           # built-in/workspace robot profiles
  io/
    bvh.py                # strict parser + streaming/chunked reader
    csv.py                # versioned robot motion schema
    atomic.py             # atomic writes and backup policy
  retarget/
    target_generator.py   # hierarchical_v2 + geocentric_v1 compatibility
    objectives.py         # tested residuals/Jacobians
    solver.py             # frame/batch IK orchestration
    contact.py            # contact detector and foot locking
    metrics.py            # static/runtime/contact metrics
    pipeline.py           # public retarget API
  optimize/
    dataset.py            # pose pairs + sampled motion frames + split
    parameterization.py   # bounded scale/translation/SO(3)
    optimizer.py          # all-parameter joint optimization
    report.py             # checkpoints, metrics, JSON/plots
  cli.py
```

现有模块可作为兼容 wrapper，最终不能保留两套互相独立的数学实现。

核心对象必须支持依赖注入和纯函数测试，不要在 import 时调用 `wp.init()`、下载资产、生成配置或打开 viewer。

---

## 5. Versioned 配置、profile 与迁移

### 5.1 v2 profile

定义单一 robot profile，至少包含：

```json
{
  "schema_version": 2,
  "robot": {
    "name": "robot_name",
    "mjcf": "...",
    "urdf": "...",
    "model_height_m": 1.70
  },
  "source": {
    "type": "soma",
    "reference_bvh": "...",
    "length_unit": "centimeter",
    "coordinate_convention": "mujoco"
  },
  "mapping": {
    "Hips": {
      "position_body": "base_link",
      "rotation_body": "base_link",
      "position_weight": 10.0,
      "rotation_weight": 2.0,
      "semantic_role": "pelvis"
    }
  },
  "scaler": {
    "mode": "hierarchical_v2",
    "config": "..."
  },
  "solver": {},
  "postprocess": {},
  "io": {},
  "pose_pairs": []
}
```

要求：

- 继续接受 `"Hips": "base_link"` 字符串 shorthand；迁移后转为完整对象。
- 支持 position/rotation 使用不同 body 和独立权重。
- raw config 中未知字段默认报错；可在显式 `allow_extensions` 命名空间中扩展。
- 相对路径相对于 profile 文件，而不是随当前工作目录漂移。
- 内置 profile 可随 wheel 打包；workspace profile 可通过 `--profile`、环境变量或明确的项目配置目录加载。
- `params.py` 保留兼容 adapter，但不再是唯一数据源；提供 `soma-retarget migrate-config` 自动生成 v2 profile。
- getter 不生成文件。`soma-retarget init-robot` 才创建模板。
- alias 使用迭代解析 + visited set，检测循环并报错。

### 5.2 scaler v2

scaler 配置必须带：

- `schema_version`、`mode`、`source_skeleton_fingerprint`、`robot_model_fingerprint`；
- root policy；
- mapped hierarchy；
- 每 segment 的 bounded scale；
- offset translation/rotation；
- 参数边界、优化先验和 provenance；
- 生成该配置的 pose/motion 数据哈希、优化器版本、seed 和指标摘要。

对 v1 负 scale：默认迁移失败并指出 joint；提供显式 `--repair-invalid-scales`，将其投影到正区间后标记为 repaired，随后必须重新优化和验证，不能悄悄取绝对值。

### 5.3 输入指纹

至少实现：

- skeleton fingerprint：joint 名、父关系、reference transforms、约定版本；
- robot fingerprint：MJCF 主文件及递归 include/关键资产元数据哈希；
- config hash、pose-pair hash、motion hash。

报告和缓存都基于指纹失效，不能只看文件名。

---

## 6. 严格 I/O 与数据模型

### 6.1 BVH

重写或封装 parser，满足：

- 精确验证 `HIERARCHY`、`MOTION`、声明 Frames、每行 channel 数、Frame Time > 0；
- 不把 FPS 四舍五入到两位；保存原始 frame time；
- joint 名唯一；namespace 去除后冲突必须报错或通过显式映射解决；
- 通道按名称映射；支持任意合法顺序；
- 数值 finite；可配置最大 joints/frames/channels/file size；
- 大文件支持 chunk/stream 路径，至少避免构造多份 Python 嵌套列表；
- 可选 strict skeleton conformance，输出详细 diff：缺失、额外、父关系、rest offset、channel order；
- 兼容 remap 必须显式选择策略，绝不默认把缺失 joint 静默设 reference pose。

### 6.2 CSV/机器人 motion

定义带 schema 的首选格式：

- root position meters；
- root quaternion xyzw；
- joint coordinates radians；
- 明确 joint names/order、sample rate、robot fingerprint。

可使用 sidecar JSON 或自描述 header；保留现有 headerless NPZ-compatible CSV 和 legacy headered CSV adapter。

所有 loader/saver 校验列数、finite、quat norm、joint count/order、sample rate。写出原子化。legacy Euler 格式在报告中标记为 lossy。

### 6.3 Pose JSON

验证：schema version、单位、向量长度、finite、四元数 norm、source/robot fingerprint、关节集合。零四元数不能静默替换 identity。

Robot pose 必须验证 joint 是否存在、是否支持单坐标设置、是否在限位内；超限默认失败，可显式 clamp 并记录。

---

## 7. 联合标定系统

### 7.1 Dataset

统一支持：

- 静态 human/robot pose pairs；
- 多个 BVH；
- 从 BVH 按覆盖度采样 frames，而不是只用第 0 帧；
- deterministic train/validation split；
- 对对称、下蹲、伸展、转体、单脚/双脚接触等 pose coverage 输出报告；
- 重复/近重复样本检测；
- 每样本权重和每 semantic role 权重。

没有有效 target 时必须失败；删除默认 `+5 cm` 虚构 target。

### 7.2 优化器

要求所有 scaler 参数联合更新：

- bounded scale latent；
- bounded translation latent；
- SO(3) tangent rotation latent；
- 可选 root policy 参数。

可以使用 Adam、L-BFGS 或可信赖域，但不能按参数类型冻结成三个阶段。若切换优化算法，所有参数仍持续激活。

必须实现：

- deterministic seed；
- loss normalization，避免 joint 数和单位改变导致权重失真；
- robust loss（Huber/Cauchy 等）；
- gradient finite check、gradient clipping；
- NaN/Inf 时回滚到 last-good；
- validation best checkpoint；
- early stopping；
- 参数越界前通过参数化保证，而不是事后才 clamp；
- 可恢复 checkpoint；
- 训练前、训练中、训练后 runtime validation；
- 只有新配置通过 validation gate 才标记为 deployable。

### 7.3 Validation gate

至少报告：

- mapped position RMSE/median/P95（米）；
- rotation geodesic mean/median/P95（度）；
- per-joint/per-pose 指标；
- scale min/max、左右差异、offset norm；
- IK 收敛率、残差、迭代次数；
- joint-limit violation；
- NaN/Inf count；
- root trajectory error；
- contact precision/稳定性、foot slip、penetration；
- output temporal velocity/acceleration/jerk；
- 相对旧配置的 baseline delta。

real-data gate 使用 baseline-relative 规则：

- 所有输出 finite；
- scale 全部在声明正区间；
- clamp 后关节限位违规不超过数值容差；
- validation 综合目标必须优于输入配置；
- 任何 P95 关键指标不得无说明恶化超过 10%；
- 未通过时保留报告和 checkpoint，但不得自动覆盖 deployable config。

另设 synthetic ground-truth 测试，要求在已知可辨识数据上恢复 scale/offset/rotation 到明确容差。

---

## 8. Runtime IK、时序和接触

### 8.1 Solver lifecycle

- capture/warm-up 不得改变可见初始状态；
- CPU 与 GPU 在同 fixture 上首帧 target、初值、输出误差需在容差内一致；
- target buffer 预分配；减少 Python per-effector 循环与 host/device round-trip；
- 支持 active env mask 或按长度 bucketing；结束的 clip 不再继续求解；
- 空输入返回 `[]` 和 report，不返回不一致的 `None`；
- solver failure 每 frame 可记录并按配置选择 fail-fast、last-good 或 fallback，不能静默产生垃圾。

### 8.2 Objectives

- position、SO(3) rotation、joint limit、temporal、contact、collision 的 residual 和 Jacobian 必须有单元测试；
- analytic Jacobian 对有限差分；可用 autodiff 对照；
- weight 语义统一：明确是 residual weight 还是 squared-cost weight；文档和配置一致；
- temporal residual 使用 joint manifold difference，而不是对 quaternion 坐标线性相减；
- collision 字段必须真的进入目标或 diagnostic，不留死配置。

### 8.3 Postprocess

顺序必须显式：

```text
IK solve -> contact-aware correction -> manifold-aware limit projection
         -> optional temporal filter -> FK/metrics -> export
```

若某一步会破坏前一步目标，报告中必须重新计算最终 FK 误差，而不是只报告 pre-process 残差。

### 8.4 Root 与输出 blend

default-pose blend 只能作用在明确的标量 hinge/slide coordinates；free/ball joint 必须使用 transform interpolation/manifold integration。mask 必须按 joint type 生成。

---

## 9. 批处理引擎

重做 headless 批处理，满足：

- `batch_size >= 1` 严格校验；
- 扩展名大小写无关；
- import/export resolve 后检测重叠、递归输出、symlink escape；
- 每文件独立 try/catch，失败不阻断其他文件；
- 产出 manifest：输入 hash、配置 hash、状态、耗时、帧数、输出 hash、错误 code/message；
- 支持 `--resume`，仅跳过 hash 全匹配且成功的项；
- 支持 `--fail-fast`；默认继续并最终返回非零 exit code 表示部分失败；
- 输出先写临时文件再原子 rename；
- 同骨架/同配置 clip 分桶，避免每文件重复解析 reference/model；
- 进度、吞吐、峰值内存、设备信息进入报告；
- 一条坏 BVH、一个 skeleton mismatch、一个写权限错误分别有测试。

不要用 `exit()` 深埋库函数；库层抛 typed exception，CLI 决定 exit code。

---

## 10. GUI 与 CLI

### 10.1 CLI

新增 console script：

```text
soma-retarget validate
soma-retarget init-robot
soma-retarget migrate-config
soma-retarget inspect-bvh
soma-retarget optimize
soma-retarget retarget
soma-retarget batch
soma-retarget benchmark
```

要求：

- `--profile`、`--robot`、`--device`、`--seed`、`--json-report` 语义一致；
- stable exit codes；
- 可完全 headless；
- 旧 app 脚本变成薄 wrapper；
- help 中提供可复制示例；
- 不在 import 时初始化 GPU/下载资产/写文件。

### 10.2 GUI

- 优化在可取消 worker 中运行；
- worker 只处理线程安全的数据，viewer/Newton 上下文按其线程约束设计；
- 实时显示当前 iteration、train/val loss、best checkpoint、参数范围、错误；
- UI 字段实际传入 optimizer，不再硬编码 rotation 参数；
- 默认不覆盖输入；显示目标输出和 backup；
- optimization 失败不留下半写 config；
- 支持任意数量/名称 pose pairs，不固定只能五个 slot；保留五个标准快捷槽；
- 优化结束后可预览 before/after，并显示 per-joint 误差与 contact 指标。

---

## 11. 打包、版本与兼容

- 把真正的 CLI/核心入口放入 package；`pip install .` 后在仓库外也能运行。
- package data 包含所需小型 configs/schema/标准 pose；大资产通过明确 asset manager 或用户路径，不依赖源码 checkout 的偶然目录。
- `params.py` 兼容只在 workspace 模式使用。
- 添加 `[project.scripts]`。
- 增加 dev dependency group：至少 pytest、pytest-cov、ruff；可选 hypothesis 用于数学 property tests。
- 生成 wheel，在干净临时 venv 中安装并运行 `soma-retarget --help`、config validation 和纯 CPU smoke test。
- 对 Newton/Warp 当前 pin 做兼容测试。不要仅因为有新版本就盲目升级；若升级，必须更新 lock、迁移说明和 CPU/GPU regression。
- 成功后按语义版本更新 `_version.py`、CHANGELOG。重大 config/runtime 行为变化建议至少升到 `0.2.0`；若决定 `1.0.0`，必须在报告中说明 API 稳定承诺。

---

## 12. 测试矩阵

新增 `tests/`，最少覆盖以下内容。

### 12.1 数学与数据结构

- SE(3) 复合/逆/换基 property tests；
- quaternion 同半球、slerp、SO(3) log/exp 的 0、π、接近 π 情况；
- FK→local roundtrip；
- skeleton 重名、环、越界父节点、非拓扑输入、NaN、零四元数；
- hierarchical scaler 在链、分叉、mapped-joint skip 下的闭合性；
- `geocentric_v1` golden compatibility。

### 12.2 BVH

构造最小 fixture 覆盖：

- XYZ、ZXY 等旋转顺序；
- X/Z/Y 等平移通道顺序；
- 无平移、部分通道；
- cm/m/mm；
- frame count 不符、channel 数不符、Frame Time=0、NaN、重复名、namespace collision；
- skeleton fingerprint match/mismatch；
- 大文件资源上限。

### 12.3 IK 与 Jacobian

- position/rotation/joint-limit/smooth/contact residual；
- analytic vs central finite difference；
- smooth safe interval 导数为 0；
- 180° rotation 不发生梯度完全失效；
- two-bone IK 的可达、过远、过近、共线、零长度 fallback；
- warm-up/capture 前后 joint state 不变；
- CPU/GPU（可用时）首帧一致性。

### 12.4 初始化与时序

- initialization frame count `0,1,2,3,4,5,10`；
- 输出帧数、采样率、首尾 pose；
- root transform 确实参与；
- quaternion/manifold interpolation；
- 不产生重复越界、空 stack、除零。

### 12.5 配置与迁移

- string shorthand/full mapping；
- raw solver settings 不被丢弃；
- unknown field、alias cycle、缺资产、坏路径；
- v1 正常 config 迁移；
- 负 scale 默认失败、显式 repair 有审计记录；
- getter 无文件写副作用；
- profile relative path 相对 profile 解析；
- cache reload。

### 12.6 优化器

- synthetic known transform 回收；
- 所有参数每 iteration 都有 gradient/update（除显式 frozen mask）；
- scale 永远为正且有界；
- 旋转接近 180°仍可下降；
- NaN gradient rollback；
- seed 可复现；
- train/validation split、best checkpoint、early stop；
- 多 BVH、多 frame 真正被使用；
- report 含 before/after、曲线、hash、per-joint metrics；
- validation 不通过时不覆盖 deployable config。

### 12.7 Contact 与输出

- 合成站立、迈步、双脚交替 contact；
- hysteresis 防抖；
- planted foot slip 相对无锁定 baseline 显著下降；
- 不穿地或在声明容差内；
- joint limits 最终满足；
- CSV/sidecar roundtrip；
- legacy Euler adapter 标记 lossy。

### 12.8 Batch、CLI、package

- mixed good/bad files；
- resume hash；
- import/export overlap、symlink escape；
- atomic write interruption；
- stable exit codes/JSON report；
- clean wheel install；
- 旧三个 app 命令 smoke test；
- 内置/当前注册的 `unitree_g1`、`roboparty_rpo`、`e3_v2` profile validation。无法在 CI 获取大 LFS 资产时使用小 fixture，并在 GPU/asset job 覆盖真实模型。

---

## 13. CI 与质量门

新增 GitHub Actions，至少分成：

1. `lint-and-core-tests`：不依赖 GPU/LFS 大资产；
2. `runtime-cpu-smoke`：安装 pinned runtime，执行 CPU smoke；
3. `package-test`：build wheel + clean install；
4. 可选/自托管 `gpu-integration`：真实 Newton/Warp、真实机器人和样例 BVH。

合并前必须通过：

```bash
uv sync --all-groups
uv run ruff check .
uv run pytest -q
uv build
```

新核心模块分支覆盖率目标不低于 85%；总仓库覆盖率不得只靠排除旧代码伪造。对暂时无法在普通 CI 跑的 GPU 路径，必须有 CPU 逻辑测试和独立 integration marker。

禁止：

- `except Exception: pass`；
- 无原因的 `# noqa`/coverage exclude；
- 测试只断言“不崩”；
- 用放宽到无意义的容差让错误公式通过；
- 在测试中依赖开发者绝对路径。

---

## 14. Benchmark 与回归基线

新增小型、可提交的 benchmark manifest，不提交大输出。至少记录：

- parser FPS/峰值内存；
- target generation frames/s；
- IK frames/s，单 clip 和多 env；
- host-device copy 次数或可观测代理；
- batch throughput；
- 优化器每 iteration 时间；
- before/after 精度和 contact 指标。

`benchmark` 命令输出 JSON，包含硬件、OS、Python、Newton、Warp、device、commit、config/input hash。

性能目标：重构后不得因明显 Python 循环/重复设备拷贝造成超过 20% 的无解释回退；若精度提升需要成本，报告 accuracy/performance tradeoff，并提供 `quality`/`balanced`/`fast` preset。

---

## 15. 文档交付

必须新增/更新：

- `docs/vnext/math.md`：本文件第 2 节的正式推导、frame 图和符号表；
- `docs/vnext/architecture.md`：模块边界、数据流、线程/设备所有权；
- `docs/vnext/config_schema.md`：v2 profile/scaler/pose/motion/report；
- `docs/vnext/migration.md`：params/v1 config/旧命令/负 scale repair；
- `docs/vnext/validation_metrics.md`；
- `docs/vnext/implementation_report.md`；
- README 与 `操作手册.md`：新机器人从零注册、优化、验证、批处理的完整命令；
- CHANGELOG。

文档中的命令必须由测试或 smoke script 验证，不能写不存在的选项。

---

## 16. 完成定义（Definition of Done）

只有以下全部成立才算完成：

- [ ] 所有 P0、P1 项有实现和回归测试；P2 产品化项完成。
- [ ] runtime 与 optimizer 共享唯一 target generator。
- [ ] 默认新 scaler 是 `hierarchical_v2`；旧配置可显式复现或迁移。
- [ ] 不再可能训练出负 scale。
- [ ] scale、translation、rotation 在同一联合优化循环中同时更新。
- [ ] smooth objective Jacobian 与数值导数一致。
- [ ] 初始化帧 `0..5` 不崩且语义正确。
- [ ] raw solver/postprocess 配置不会被 registry 丢弃。
- [ ] BVH 通道顺序和单位被严格处理。
- [ ] solver capture 不改变首帧初值。
- [ ] 最终关节处理是 manifold-aware。
- [ ] 足底后处理有显式 contact state、anchor 和 slip 指标。
- [ ] batch 可恢复、逐文件隔离失败、原子写出。
- [ ] 默认优化不覆盖输入，未通过 validation gate 不部署。
- [ ] CLI、GUI 兼容 wrapper、clean wheel 都可运行。
- [ ] tests、CI、benchmark、数学/迁移文档完整。
- [ ] `docs/vnext/implementation_report.md` 如实列出实际执行结果；GPU 未验证时明确写未验证，不能假装通过。

实施完成后，最后一次本地检查应至少执行：

```bash
git status --short
uv sync --all-groups
uv run ruff check .
uv run pytest -q
uv build
```

并在 implementation report 中粘贴命令、退出码和摘要。工作树应只包含有意提交的源码、测试、配置 schema、迁移工具和文档。