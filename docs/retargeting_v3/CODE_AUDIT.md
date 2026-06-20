# Retargeting v2 Code Audit

审计参考分支：`retargeting-v2-design`  
审计参考 HEAD：`63b84c265453fd3a2a46676d00c5499b27e319e1`

## 总结

v2 建立了有价值的 profile、contact、diagnostics 和 artifacts 基础，但核心 target/reachability 数学尚未正确，随后工作逐渐偏向 objective 扩张和权重/解析 Jacobian sweep。

## 阻断问题

### 1. 静态 XML morphology 不是 runtime kinematics

`robotics/morphology.py` 自己解析 body pos/quat 和 joint axis；`task_compiler.py` 用静态 cross(axis, tip-origin)构造 Jv。没有以 Newton运行时 FK为最终真值。

风险：

- defaults/includes；
- joint pos；
- prismatic；
- non-zero neutral；
- orientation representations；
- compiled model差异；
- q/dof index错误。

### 2. endpoint reachability chain root错误

`_semantic_root_body()` 使用 immediate semantic parent：

- Hand root=ForeArm；
- Foot root=Shin。

这导致 RPO Hand只包含 elbow-yaw，Foot只包含 ankles，进而污染 rank、basis、weights、metrics和runtime schedule。

### 3. v2 scaler没有 rest-frame alignment

`SegmentLocalTargetBuilder` 直接保留 source world segment direction和source quaternion。`HumanToRobotScaler` 随后继续应用 legacy joint offsets。

### 4. root scale未接入

compiled profile记录 root horizontal scale，但 `enable_segment_local_from_profile()` 没把它传给 builder。

### 5. relative torso名不副实

runtime `_project_rotation_target()` 投影的是 source world quaternion，而不是 Chest relative Hips 的 rest delta。

### 6. virtual no-wrist hand site被忽略

runtime parser把 position link offset是否生效绑到 `orientation_supported`；无腕 hand offset不生效。

### 7. priority未实现

profile有bands，但 `_task_priority_weight()`优先返回 task normalized weight。runtime是单次 solve，guard主要保护joint limits。

### 8. normalization与robust loss未实现

`characteristic_length`、`robust_loss`多为metadata；position objective仍使用米制error乘大权重。

### 9. benchmark不统一 desired target

legacy/v2分别对自己生成的 target测RMSE。rank0 projection把 residual清零。

### 10. contact功能没有进入主要机器人配置

RPO/G1 raw config没有 anchors，compiled profile contact为空，许多接触工作未真正验证核心机器人。

### 11. fingerprint潜在错误

旧 morphology代码复用 `digest` 变量，可能让 MJCF fingerprint被最后一个mesh digest覆盖。

### 12. post-solve motion limit与warm-start状态不一致

motion-limited output没有明确回写同一个 solver seed，可能让下一帧 warm-start使用未限制状态。

## 可保留

- profile schema和稳定序列化；
- explicit contact warmup；
- per-env contact；
- contact state machine；
- range-normalized limit barrier；
- provenance/artifacts；
- virtual site来源框架；
- structured failures。

## v3 评审问题

每次 PR 都必须回答：

1. 数学真值来自runtime FK吗？
2. full endpoint chain包含所有proximal DoF吗？
3. neutral pose能精确重建吗？
4. 是否完全不依赖legacy offsets？
5. relative task真的是relative吗？
6. profile字段是否runtime使用？
7. benchmark target是否独立统一？
8. 许可证和source hash是否完整？
9. 是否新增robot-name special case？
10. 是否在核心未过时增加了新objective？
