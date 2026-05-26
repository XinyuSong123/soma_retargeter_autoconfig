# SOMA 动捕输入骨架说明

这套重定向工具的输入侧标准骨架是：

```text
soma_retargeter/configs/soma/soma_zero_frame0.bvh
```

我已经把这份 BVH 抽成机器可读规格：

```text
soma_retargeter/configs/soma/soma_skeleton_spec.json
```

如果你们的动捕设备一开始就导出成这个 BVH 骨架格式，后面基本可以直接放进重定向流程里用。

## 直接可用的条件

动捕导出的 `.bvh` 最好满足这些规则：

- 骨架层级和关节名与 `soma_zero_frame0.bvh` 一致。
- 平移单位按 BVH 文件里的厘米导出；程序会自动乘 `0.01` 转成米。
- 旋转单位是 degree。
- 旋转通道顺序是 `Zrotation Yrotation Xrotation`，也就是 `zyx`。
- 坐标约定按程序内部使用：`+Z` 向上，`-Y` 为前方。
- 每个关节都可以导出 6 个通道：`Xposition Yposition Zposition Zrotation Yrotation Xrotation`。

如果你们的 BVH 只有 root 有位置通道、其他关节只有旋转通道，解析器也能读；但为了减少设备导出差异，最稳的是按模板全关节 6 通道导出。

## 最少需要保留的关节

完整模板里有 78 个关节，包含手指和面部。当前默认重定向最核心依赖下面这些人体部位：

```text
Hips
Chest
LeftArm
LeftForeArm
LeftHand
RightArm
RightForeArm
RightHand
LeftLeg
LeftShin
LeftFoot
RightLeg
RightShin
RightFoot
```

做 scaler 学习时，脚趾和脖子也有帮助：

```text
Neck1
LeftToe
LeftToeBase
RightToe
RightToeBase
```

如果动捕设备只能导出简化人体骨架，至少要保证上面这些名字能对上；否则重定向时会出现某些目标缺失或者姿态比例不稳。

## 推荐工作流

1. 让动捕系统导出一条静止 T-pose BVH，骨架按 `soma_zero_frame0.bvh` 对齐。
2. 把导出的 BVH 放进 converter config 的 `import_folder`，或者直接替换默认输入目录中的 BVH。
3. 运行可视化重定向：

```bash
python ./app/bvh_to_csv_converter.py --viewer gl --robot roboparty_rpo
```

4. 确认效果没问题后跑 headless：

```bash
python ./app/bvh_to_csv_converter.py --viewer null --robot roboparty_rpo
```

## 骨架数据文件怎么用

`soma_skeleton_spec.json` 里包含：

- `joint_count`：标准骨架关节数量。
- `joints`：所有关节的名字、父节点、BVH 路径、offset。
- `required_joints_for_default_retarget_ik_map`：默认 IK 映射必须关心的关节。
- `recommended_joints_for_scaler_learning`：做姿态比例学习时推荐保留的关节。
- `coordinate_convention` 和 `channel_convention`：动捕导出时最关键的单位、坐标轴、通道顺序。

实际动捕导出时，以 `soma_zero_frame0.bvh` 作为权威模板；`soma_skeleton_spec.json` 是给程序、脚本或者动捕软件配置界面对照用的清单。
