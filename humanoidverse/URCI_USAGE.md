# URCI (Unified Robot Control Interface) 使用文档

## 概述

`urci.py` 是 HumanoidVerse 的统一机器人控制接口，用于在 MuJoCo 仿真器中部署训练好的策略模型。支持单策略和多策略模式，可以实时切换不同的策略进行测试。

## 目录

- [基本用法](#基本用法)
- [参数说明](#参数说明)
- [使用示例](#使用示例)
- [键盘控制](#键盘控制)
- [输出文件](#输出文件)
- [常见问题](#常见问题)

---

## 基本用法

### 基本命令格式

```bash
python humanoidverse/urci.py [参数]
```

### 必需参数

- `+simulator=mujoco` - 指定使用 MuJoCo 仿真器
- `+checkpoint=<路径>` - 指定 ONNX 模型检查点路径

### 可选参数

- `+opt=record` - 启用动作数据保存
- 其他配置参数（见下方详细说明）

---

## 参数说明

### 1. 仿真器参数 (`+simulator`)

#### `+simulator=mujoco`
指定使用 MuJoCo 仿真器（目前唯一支持的仿真器）

**相关子参数：**

- `simulator.config.sim.fps` (默认: 500)
  - 仿真频率（Hz）
  - 示例: `simulator.config.sim.fps=500`

- `simulator.config.sim.control_decimation` (默认: 10)
  - 控制降采样率（每 N 个仿真步执行一次控制）
  - 实际控制频率 = fps / control_decimation
  - 示例: `simulator.config.sim.control_decimation=10` (控制频率 50Hz)

- `simulator.config.sim.render_mode` (默认: "human")
  - 渲染模式: `None`, `"human"`, `"rgb_array"`
  - 示例: `simulator.config.sim.render_mode="human"`

### 2. 检查点参数 (`+checkpoint`)

#### 单策略模式
```bash
+checkpoint=path/to/model.onnx
```

#### 多策略模式
```bash
+checkpoint='[path/to/model1.onnx,path/to/model2.onnx,path/to/model3.onnx]'
```

**说明：**
- 检查点必须是 `.onnx` 格式
- 系统会自动从检查点目录或其父目录加载 `config.yaml` 训练配置
- 多策略模式下，所有策略的 `robot` 和 `obs` 配置必须兼容

### 3. 动作保存参数 (`+opt=record`)

启用动作数据保存功能。

**相关子参数：**

- `env.config.save_motion` (默认: True，当使用 `+opt=record` 时)
  - 是否保存动作数据
  - 示例: `env.config.save_motion=True`

- `env.config.save_note` (默认: null)
  - 保存目录的备注名称（会出现在目录名中）
  - 示例: `env.config.save_note="my_experiment"`
  - 保存路径格式: `{checkpoint父目录的父目录}/motions/{save_note}_URCI_MujocoRobot_{时间戳}/`

- `env.config.save_total_steps` (默认: 10000)
  - 保存的总步数
  - 示例: `env.config.save_total_steps=20000`

- `env.config.ckpt_dir` (默认: checkpoint 的父目录)
  - 检查点目录（用于保存动作数据）
  - 通常自动设置，无需手动指定

### 4. 评估日志参数

- `eval_name` (默认: "TEST")
  - 评估名称（用于日志目录命名）
  - 示例: `eval_name=my_eval`
  - 日志路径: `logs_eval/{eval_name}/{eval_timestamp}/`

- `eval_timestamp` (自动生成)
  - 评估时间戳，格式: `YYYYMMDD_HHMMSS`
  - 通常自动生成，无需手动设置

### 5. 机器人配置参数 (`robot.*`)

- `robot.asset.xml_file`
  - 机器人模型 XML 文件路径
  - 示例: `robot.asset.xml_file="g1/g1_23dof_lock_wrist_phys_inertia.xml"`

- `robot.motion.motion_file`
  - 动作文件路径（用于动作跟踪任务）
  - 示例: `robot.motion.motion_file="path/to/motion_data"`

### 6. 部署参数 (`deploy.*`)

这些参数在 `humanoidverse/config/deploy/single.yaml` 中定义，可通过命令行覆盖：

- `deploy.render` (默认: True)
  - 是否启用渲染
  - 示例: `deploy.render=True`

- `deploy.defcmd` (默认: [0.0, 0.0, 0.0, 0.0])
  - 默认命令 [vx, vy, vz, heading]
  - 示例: `deploy.defcmd=[0.0,0.0,0.0,0.0]`

- `deploy.heading_cmd` (默认: True)
  - 是否使用航向角命令（True）或角速度命令（False）
  - 示例: `deploy.heading_cmd=True`

- `deploy.ctrl_dt` (默认: 0.02)
  - 控制周期（秒），通常为 `control_decimation / fps`
  - 示例: `deploy.ctrl_dt=0.02`

- `deploy.BYPASS_ACT` (默认: False)
  - 是否绕过动作输出（用于测试）
  - 示例: `deploy.BYPASS_ACT=False`

- `deploy.SWITCH_EMA` (默认: True)
  - 是否启用 EMA（指数移动平均）切换
  - 示例: `deploy.SWITCH_EMA=True`

### 7. 环境配置参数 (`env.config.*`)

- `env.config.env_spacing` (默认: 从训练配置继承)
  - 环境间距（多环境时使用）
  - 单环境部署时通常不需要

- `env.config.save_rendering_dir`
  - 渲染视频保存目录
  - 通常自动设置为: `{checkpoint父目录}/renderings/ckpt_{ckpt_num}/`

### 8. 外部策略 (`_external_*`)

支持使用外部策略（以 `_external` 开头的检查点名称）：

- `_external_zero` - 零策略（静止）
- `_external_sin` - 正弦波策略
- 自定义外部策略（需要在 `humanoidverse/deploy/external/core.py` 中实现）

---

## 使用示例

### 示例 1: 基本单策略部署

```bash
python humanoidverse/urci.py \
  +simulator=mujoco \
  +checkpoint=example/pretrained_horse_stance_pose/exported/model_50000.onnx
```

### 示例 2: 启用动作保存

```bash
python humanoidverse/urci.py \
  +opt=record \
  +simulator=mujoco \
  +checkpoint=example/pretrained_horse_stance_pose/exported/model_50000.onnx
```

### 示例 3: 自定义保存备注和步数

```bash
python humanoidverse/urci.py \
  +opt=record \
  +simulator=mujoco \
  +checkpoint=example/pretrained_horse_stance_pose/exported/model_50000.onnx \
  env.config.save_note="horse_stance_test" \
  env.config.save_total_steps=20000
```

### 示例 4: 调整仿真参数

```bash
python humanoidverse/urci.py \
  +simulator=mujoco \
  +checkpoint=example/pretrained_horse_stance_pose/exported/model_50000.onnx \
  simulator.config.sim.fps=500 \
  simulator.config.sim.control_decimation=10
```

### 示例 5: 指定机器人模型

```bash
python humanoidverse/urci.py \
  +opt=record \
  +simulator=mujoco \
  +checkpoint=example/pretrained_horse_stance_pose/exported/model_50000.onnx \
  robot.asset.xml_file="g1/g1_23dof_lock_wrist_phys_inertia.xml"
```

### 示例 6: 多策略部署

```bash
# 方式 1: 直接在命令行中指定
python humanoidverse/urci.py \
  +simulator=mujoco \
  +checkpoint='[example/pretrained_horse_stance_pose/exported/model_50000.onnx,example/pretrained_horse_stance_pose/exported/model_50000.onnx]' \
  +opt=record

# 方式 2: 使用环境变量（推荐，避免引号问题）
export CKPTS="[example/pretrained_horse_stance_pose/exported/model_50000.onnx,example/pretrained_horse_stance_pose/exported/model_50000.onnx]"
python humanoidverse/urci.py +simulator=mujoco +checkpoint=$CKPTS +opt=record
```

### 示例 7: 混合内部和外部策略

```bash
export CKPTS="[example/pretrained_horse_stance_pose/exported/model_50000.onnx,_external_zero]"
python humanoidverse/urci.py +simulator=mujoco +checkpoint=$CKPTS +opt=record
```

### 示例 8: 自定义评估名称

```bash
python humanoidverse/urci.py \
  +opt=record \
  +simulator=mujoco \
  +checkpoint=example/pretrained_horse_stance_pose/exported/model_50000.onnx \
  eval_name="horse_stance_eval"
```

### 示例 9: 通用动作策略部署（需要指定动作文件）

```bash
python humanoidverse/urci.py \
  +opt=record \
  +simulator=mujoco \
  +checkpoint="<path to your onnx>" \
  robot.motion.motion_file="<path to your motion data>" \
  robot.asset.xml_file=g1/g1_23dof_lock_wrist_rev_2.xml
```

---

## 键盘控制

在 MuJoCo 查看器窗口中，可以使用以下键盘快捷键进行控制：

### 运动控制（ViewerPlugin 模式）

**键盘映射：**
```
  ----------------
  |   K L ; '    |
  |   , . /      |
  ----------------
```

- **`,` (逗号)**: 增加 vy（前进速度）
- **`/` (斜杠)**: 减少 vy（后退速度）
- **`L`**: 增加 vx（左移速度）
- **`.` (句号)**: 减少 vx（右移速度）
- **`K`**: 
  - 如果 `heading_cmd=True`: 增加航向角（heading）
  - 如果 `heading_cmd=False`: 增加 vz（上升速度）
- **`;` (分号)**: 
  - 如果 `heading_cmd=True`: 减少航向角（heading）
  - 如果 `heading_cmd=False`: 减少 vz（下降速度）
- **`'` (单引号)**: 重置命令为默认值 (`deploy.defcmd`)
- **`Enter`**: 重置动作参考（`_ref_pid = -2`）
- **`[` (左方括号)**: 切换到上一个策略（`_ref_pid -= 1`）
- **`]` (右方括号)**: 切换到下一个策略（`_ref_pid += 1`）
- **`0-9` (数字键)**: 直接切换到指定策略索引（0-9）

### 运动控制（MViewerPlugin 模式）

- **`W`**: 增加 vy（前进速度）
- **`S`**: 减少 vy（后退速度）
- **`A`**: 增加 vx（左移速度）
- **`D`**: 减少 vx（右移速度）
- **`E`**: 
  - 如果 `heading_cmd=True`: 增加航向角（heading）
  - 如果 `heading_cmd=False`: 增加 vz（上升速度）
- **`Q`**: 
  - 如果 `heading_cmd=True`: 减少航向角（heading）
  - 如果 `heading_cmd=False`: 减少 vz（下降速度）
- **`R`**: 重置命令为默认值
- **`Space`**: 暂停/继续仿真
- **`Esc`**: 退出查看器
- **`Enter`**: 重置动作参考
- **`[` / `]`**: 切换策略

### 命令格式

命令格式为 `[vx, vy, vz, heading]`：
- `vx`: X 方向速度（左右）
- `vy`: Y 方向速度（前后）
- `vz`: Z 方向速度（上下，仅在 `heading_cmd=False` 时使用）
- `heading`: 航向角（仅在 `heading_cmd=True` 时使用）

---

## 输出文件

### 输出目录结构

部署输出保存在以下目录：

```
{checkpoint父目录的父目录}/
├── motions/
│   └── {save_note}_URCI_MujocoRobot_{时间戳}/
│       ├── 0_pid0_frame{帧数}_{时间戳}.pkl
│       ├── 1_pid0_frame{帧数}_{时间戳}.pkl
│       └── ...
└── renderings/
    └── ckpt_{ckpt_num}/
        └── video_{时间戳}.mp4
```

### 文件说明

- **动作数据文件 (`*.pkl`)**: 
  - 包含机器人状态、动作、观察等信息
  - 文件格式: `{motion_id}_pid{policy_id}_frame{帧数}_{时间戳}.pkl`
  - 包含字段：
    - `root_trans_offset`: 根位置偏移
    - `root_rot`: 根旋转（四元数，XYZW格式）
    - `dof`: 关节角度
    - `pose_aa`: 姿态（轴角表示）
    - `action`: 动作输出
    - `actor_obs`: Actor 观察
    - `terminate`: 终止标志
    - `dof_vel`: 关节速度
    - `root_lin_vel`: 根线速度
    - `root_ang_vel`: 根角速度
    - `motion_times`: 时间戳
    - `fps`: 帧率
    - `tau`: 力矩
    - `cmd`: 命令

- **渲染视频 (`*.mp4`)**: 
  - 如果启用录制，会保存渲染视频
  - 保存在 `renderings/ckpt_{ckpt_num}/` 目录

- **评估日志**: 
  - 保存在 `logs_eval/{eval_name}/{eval_timestamp}/`
  - 包含 `config.yaml` 和 `eval.log`

---

## 常见问题

### Q1: 找不到检查点配置文件

**解决方案：**
- 确保检查点目录或其父目录包含 `config.yaml` 文件
- 检查点路径格式应为: `.../exported/model_50000.onnx`
- 配置文件应在: `.../exported/config.yaml` 或 `.../config.yaml`

### Q2: 多策略模式下配置不兼容

**解决方案：**
- 确保所有策略使用相同的 `robot` 配置（除了 `robot.motion.motion_file`）
- 确保所有策略的 `obs` 配置兼容（观察维度、缩放等必须一致）
- 检查错误信息，了解具体哪个配置项不兼容

### Q3: 动作保存失败

**解决方案：**
- 确保使用 `+opt=record` 参数
- 检查 `env.config.save_motion=True`
- 确保有足够的磁盘空间
- 检查保存目录的写入权限

### Q4: 键盘控制无响应

**解决方案：**
- 确保 MuJoCo 查看器窗口处于焦点状态
- 检查是否使用了正确的键盘映射（ViewerPlugin 或 MViewerPlugin）
- 尝试重新启动部署

### Q5: 策略切换不工作

**解决方案：**
- 确保使用多策略模式（检查点路径为列表格式）
- 使用数字键 `0-9` 直接切换策略
- 使用 `[` 和 `]` 键切换策略
- 检查控制台输出的 `_ref_pid` 值

### Q6: 仿真速度太快或太慢

**解决方案：**
- 调整 `simulator.config.sim.fps`（默认 500）
- 调整 `simulator.config.sim.control_decimation`（默认 10）
- 实际控制频率 = fps / control_decimation

### Q7: 机器人模型加载失败

**解决方案：**
- 检查 `robot.asset.xml_file` 路径是否正确
- 确保 XML 文件存在于 `robot.asset.asset_root` 目录
- 检查 XML 文件格式是否正确

### Q8: 外部策略无法使用

**解决方案：**
- 确保外部策略名称以 `_external` 开头
- 检查 `humanoidverse/deploy/external/core.py` 中是否实现了对应的策略
- 常见外部策略：`_external_zero`, `_external_sin`

### Q9: 动作数据文件格式

**解决方案：**
- 动作数据保存为 `.pkl` 格式
- 使用 `joblib.load()` 加载文件
- 文件包含字典，键为 `motion{id}`，值为包含所有状态信息的字典

### Q10: 如何禁用渲染以提高性能

**解决方案：**
```bash
deploy.render=False
```

---

## 相关文档

- [README.md](README.md) - 项目总体说明
- [TRAIN_AGENT_USAGE.md](TRAIN_AGENT_USAGE.md) - 训练脚本使用文档
- 配置文件位置：
  - 部署配置: `humanoidverse/config/deploy/`
  - 机器人配置: `humanoidverse/config/robot/`

---

## 注意事项

1. **模型格式**: 部署需要使用 ONNX 格式的模型，可通过 `eval_agent.py` 导出
2. **配置兼容性**: 多策略模式下，所有策略的配置必须兼容
3. **键盘控制**: 确保查看器窗口处于焦点状态才能使用键盘控制
4. **动作保存**: 使用 `+opt=record` 启用动作保存，数据会保存在 motions 目录
5. **性能**: 禁用渲染可以提高性能，但无法看到可视化效果
6. **安全**: 在真实机器人上部署前，务必在仿真环境中充分测试

---

**最后更新**: 2024年

