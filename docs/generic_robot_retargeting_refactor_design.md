# 通用机器人 Retarget Config 生成重构设计

## 背景

当前仓库已经切到通用机器人注册方式：

1. `params.py` 只保存用户必须准备的机器人注册信息。
2. `app/pose_optimizer_ui.py` 通过 `--robot` 读取注册信息，并用 human/robot 静态姿态优化 scaler config。
3. `app/bvh_to_csv_converter.py` 同样通过 `--robot` 自动选择机器人和运行时 config。
4. `soma_retargeter/robot_registry_parser.py` 负责把最小注册信息展开成 Newton pipeline 需要的完整运行时字典。

新增机器人时，用户不需要改 app 或 pipeline 代码；准备文件并在 `params.py` 注册后，就能打开 config 优化窗口和 BVH 重定向窗口。

## 目标

1. 根目录 `params.py` 只保留四个注册字典：`ROBOT_XML_DICT`、`ROBOT_URDF_DICT`、`RETARGETER_CONFIG_DICT`、`POSE_PAIR_JSON_DICT`。
2. UI 默认可只使用 T-pose 一组 paired pose 生成 config；效果不够时再追加 natural down、arms forward 等姿态 JSON。
3. `pose_optimizer_ui.py` 按通用机器人注册名/profile 工作。
4. retargeter config 对用户只暴露 human body part 到 robot link 的映射；其余 Newton 参数在运行时自动生成。
5. scaler config、BVH converter config 在第一次启动时自动生成到 retargeter config 同一目录，后续自动复用。
6. `bvh_to_csv_converter.py` 可从 `params.py` 或 converter config 中识别新机器人，并用同一套 pipeline 一键重定向。
7. 现有 RoboParty / Unitree G1 作为普通注册机器人继续工作。

## 新架构

### 1. `params.py`

根目录新增用户可编辑的 `params.py`：

- `ROBOT_XML_DICT`：机器人运行时 MJCF/XML 入口。
- `ROBOT_URDF_DICT`：机器人 URDF 参考文件，可留空。
- `RETARGETER_CONFIG_DICT`：最小 retargeter link map config。
- `POSE_PAIR_JSON_DICT`：每个机器人只注册 robot pose JSON。标准 human pose JSON 走硬编码目录 `soma_retargeter/assets/standard_human_pos`，可以留空；窗口仍能打开，并提示缺少哪些机器人姿态文件。

路径默认按仓库根目录解析，也允许绝对路径。

最小 retargeter config 格式如下：

```json
{
    "ik_map": {
        "Hips": "base_link",
        "Chest": "torso_link",
        "LeftForeArm": "left_elbow_link"
    }
}
```

这里不要写 `t_weight`、`r_weight`、`human_robot_scaler_config`、初始化帧数、IK 迭代次数等内部参数；parser 会在运行时自动展开。

### 2. 运行时 profile 解析层

新增 `soma_retargeter/robot_registry_parser.py`，负责：

1. 读取根目录 `params.py`。
2. 将最小注册字典规范化为运行时 profile。
3. 硬编码注入标准 human pose 目录 `soma_retargeter/assets/standard_human_pos` 与参考 BVH `soma_retargeter/configs/soma/soma_zero_frame0.bvh`。
4. 在第一次运行时自动生成 scaler config 与 bvh converter config，并保存在 retargeter config 的同一文件夹。
5. 加载最小 retargeter config，并在内存里展开为 Newton pipeline 需要的完整运行时 config。
6. 将 repo 相对路径、configs 相对路径、绝对路径统一解析成真实文件路径。

### 3. Pipeline 动态目标机器人

`soma_retargeter/pipelines/utils.py` 保留原来的 `TargetType` 枚举，但允许 `get_target_type_from_str()` 对 `params.py` 中声明的机器人返回字符串目标名。

运行时：

- 如果目标机器人在 `params.py` 中存在，MJCF 从 profile 的 `mjcf_path` 读取。
- 如果目标机器人在 `params.py` 中存在，retargeter config 从 profile 的 `retargeter_config` 读取。
- 否则继续走原有 Unitree / RPO 内置分支。

这样新增机器人无需修改枚举。

### 4. Pose Optimizer UI 通用化

`app/pose_optimizer_ui.py` 是通用配置优化入口。

默认参数来自命令行 `--robot` 指定的注册名：

- `--robot` 支持别名，例如 `g1`、`rpo`
- `--config` 默认自动生成，并保存在对应 retargeter config 的同目录。
- `--scaler-config` 默认自动生成，并保存在对应 retargeter config 的同目录。
- `--retargeter-config` 默认来自 `RETARGETER_CONFIG_DICT[robot]`
- `--output` 默认同样指向自动生成的 scaler config，也就是优化生成的 scaler 和重定向使用的 scaler 是同一个文件。

姿态槽位保持 5 组语义槽：

1. `t_pose`
2. `natural_down`
3. `both_arms_forward`
4. `both_elbows_forward_90`
5. `arms_forward_squat_hip_yaw_out_45`

但只有硬编码 human pose 与注册好的 robot pose 都存在且已填完的槽位才默认勾选训练。因此新机器人只准备 T-pose JSON 也能开始优化。

### 5. BVH Converter 通用化

`app/bvh_to_csv_converter.py`：

- Target Robot 下拉列表来自内置机器人 + `params.py` robots。
- `--robot` 可临时覆盖 converter config 的 `retarget_target`。
- batch 和 viewer 都通过 `pipeline_utils` 找 MJCF / retargeter config。
- NPZ-compatible CSV 对自定义机器人使用通用透传格式：`[root position, root quaternion, joint q...]`。

### 6. 一键流程

新机器人最小流程：

1. 准备 MJCF、URDF、retargeter config、T-pose robot JSON。
2. 在 `params.py` 的 4 个注册字典中增加机器人路径。
3. 运行：

```bash
python ./app/pose_optimizer_ui.py --viewer gl --robot roboparty_rpo
```

4. 第一次打开窗口时，会先自动生成默认 scaler config 和 bvh converter config；点击“开始优化”后，UI 会直接覆盖更新这个 scaler config。retargeter link map 保持用户填写的最小格式，不写入内部运行时字段。
   优化窗口只显示一组迭代次数/学习率；rotation 学习在后端自动沿用相同参数。
5. 运行：

```bash
python ./app/bvh_to_csv_converter.py --viewer gl --robot roboparty_rpo
```

也可以显式传自动生成的 converter config：

```bash
python ./app/bvh_to_csv_converter.py --config ./soma_retargeter/configs/<robot>/<robot>_bvh_to_csv_converter_config.json --viewer gl
```

Headless batch 同样支持注册名：

```bash
python ./app/bvh_to_csv_converter.py --viewer null --robot roboparty_rpo
```

## 迁移策略

1. 新入口统一为 `app/pose_optimizer_ui.py`。
2. 原有内置机器人仍可不依赖完整 pose 注册工作；缺少姿态 JSON 时 UI 会提示。
3. 自定义机器人使用 headerless NPZ-compatible CSV，避免每个机器人必须新增一份 CSV header class。
