# train_agent.py 使用文档

## 概述

`train_agent.py` 是 HumanoidVerse 的策略训练脚本，用于在 IsaacGym 仿真器中训练强化学习策略。支持两种训练模式：基础策略训练（Motion Tracking）和通用动作策略训练（General Motion Tracking），后者支持 Teacher-Student 训练范式。

## 目录

- [基本用法](#基本用法)
- [参数说明](#参数说明)
- [训练模式](#训练模式)
- [使用示例](#使用示例)
- [输出文件](#输出文件)
- [训练参数调整](#训练参数调整)
- [常见问题](#常见问题)

---

## 基本用法

### 基本命令格式

```bash
python humanoidverse/train_agent.py [参数]
```

### 必需参数

- `+simulator=isaacgym` - 指定使用 IsaacGym 仿真器（目前唯一支持的训练仿真器）
- `+exp=<实验类型>` - 指定实验类型（`motion_tracking` 或 `general_tracking`）
- `robot.motion.motion_file=<路径>` - 指定动作数据文件路径（`.pkl` 格式）

### 常用参数

- `project_name=<名称>` - 项目名称（用于日志目录）
- `experiment_name=<名称>` - 实验名称（用于日志目录）
- `num_envs=<数量>` - 并行环境数量（训练用 4096，调试用 128）
- `seed=<种子>` - 随机种子
- `+device=cuda:0` - 指定 GPU 设备

---

## 参数说明

### 1. 仿真器参数 (`+simulator`)

#### `+simulator=isaacgym`
指定使用 IsaacGym 仿真器进行训练。

**注意：** 目前仅支持 IsaacGym，IsaacSim 支持正在开发中。

### 2. 实验类型参数 (`+exp`)

#### `+exp=motion_tracking`
基础策略训练模式，用于训练简单的动作跟踪策略。

**相关配置：**
- 使用 `mh_ppo` 算法
- 观察空间：`motion_tracking/main` 或 `motion_tracking/benchmark`
- 适用于固定动作模式（如马步姿势）

#### `+exp=general_tracking`
通用动作策略训练模式，支持 Teacher-Student 训练。

**相关配置：**
- 使用 `ppo_mimic` 算法
- 观察空间：`motion_tracking/obs_ppo_teacher` 或 `motion_tracking/obs_ppo_student`
- 适用于复杂动作序列，支持动作重采样

### 3. 观察空间参数 (`+obs`)

#### 基础策略训练观察空间

- `+obs=motion_tracking/main` - 标准观察空间（可部署）
  - 包含：`base_ang_vel`, `projected_gravity`, `dof_pos`, `dof_vel`, `actions`, `ref_motion_phase`, `history_actor`
  - 历史长度：4 步

- `+obs=motion_tracking/benchmark` - 基准测试观察空间（不可部署）
  - 包含特权信息（privileged information）
  - 用于评估动作数据质量和难度
  - 无域随机化

#### 通用策略训练观察空间

- `+obs=motion_tracking/obs_ppo_teacher` - Teacher 观察空间
  - 包含：`base_lin_vel`, `base_ang_vel`, `dof_pos`, `dof_vel`, `actions`, `roll_pitch`, `local_key_body_pos`, `local_key_body_rot`, `anchor_ref_pos`, `anchor_ref_rot`, `next_step_ref_motion`
  - 历史长度：10 步
  - 包含未来动作目标（future motion targets）

- `+obs=motion_tracking/obs_ppo_student` - Student 观察空间
  - 移除部分信息（如 `base_lin_vel`, `anchor_ref_pos`）
  - 更接近真实部署场景
  - 用于 DAgger 训练

### 4. 机器人配置参数 (`+robot`)

#### `+robot=g1/g1_23dof_lock_wrist`
基础策略训练使用的机器人配置（锁定手腕）。

**相关子参数：**
- `robot.motion.motion_file` - 动作数据文件路径
  - 示例: `robot.motion.motion_file="example/motion_data/Horse-stance_pose.pkl"`
  - 支持 `.pkl` 格式的动作数据

#### `+robot=g1/g1_23dof_general`
通用策略训练使用的机器人配置（通用配置，更灵活）。

### 5. 域随机化参数 (`+domain_rand`)

#### `+domain_rand=main`
标准域随机化配置，用于提高策略的鲁棒性。

**随机化内容：**
- 机器人质量、质心位置
- 关节刚度、阻尼
- 地面摩擦系数
- 控制延迟

#### `+domain_rand=dr_nil`
无域随机化，用于基准测试。

### 6. 奖励函数参数 (`+rewards`)

#### `+rewards=motion_tracking/main`
基础策略训练的奖励函数配置。

#### `+rewards=motion_tracking/general_main`
通用策略训练的奖励函数配置，包含额外的锚点奖励项。

### 7. 地形参数 (`+terrain`)

#### `+terrain=terrain_locomotion_plane`
平面地形（最常用）。

**其他可选地形：**
- `terrain_locomotion_rough` - 粗糙地形
- `terrain_locomotion_flat` - 平坦地形

### 8. 训练算法参数 (`algo.config.*`)

这些参数可以通过命令行覆盖，也可以在配置文件中修改。

#### 迭代相关参数

- `algo.config.num_learning_iterations` (默认: 1000000)
  - 总训练迭代次数
  - 论文中训练 50000 次迭代
  - 示例: `algo.config.num_learning_iterations=50000`

- `algo.config.save_interval` (默认: 1000)
  - 模型保存间隔（每 N 次迭代保存一次）
  - 示例: `algo.config.save_interval=500`

- `algo.config.logging_interval` (默认: 25)
  - 日志记录间隔（每 N 次迭代记录一次）
  - 示例: `algo.config.logging_interval=10`

#### 学习相关参数

- `algo.config.learning_rate` (默认: 1.e-4 for ppo_mimic, 1.e-3 for mh_ppo)
  - 学习率
  - 示例: `algo.config.learning_rate=1.e-3`

- `algo.config.num_learning_epochs` (默认: 5)
  - 每次迭代的学习轮数
  - 示例: `algo.config.num_learning_epochs=10`

- `algo.config.num_mini_batches` (默认: 4)
  - 小批量数量
  - 示例: `algo.config.num_mini_batches=8`

- `algo.config.num_steps_per_env` (默认: 24)
  - 每个环境每次迭代的步数
  - 示例: `algo.config.num_steps_per_env=48`

#### PPO 算法参数

- `algo.config.clip_param` (默认: 0.2)
  - PPO 裁剪参数
  - 示例: `algo.config.clip_param=0.2`

- `algo.config.gamma` (默认: 0.99)
  - 折扣因子
  - 示例: `algo.config.gamma=0.99`

- `algo.config.lam` (默认: 0.95)
  - GAE (Generalized Advantage Estimation) 参数
  - 示例: `algo.config.lam=0.95`

- `algo.config.entropy_coef` (默认: 0.005 for ppo_mimic, 0.01 for mh_ppo)
  - 熵系数（鼓励探索）
  - 示例: `algo.config.entropy_coef=0.01`

- `algo.config.value_loss_coef` (默认: 1.0)
  - 价值损失系数
  - 示例: `algo.config.value_loss_coef=1.0`

#### Teacher-Student 训练参数（仅通用策略训练）

- `algo.config.dagger_only` (默认: False)
  - 是否仅使用 DAgger 训练（不进行 RL 训练）
  - 示例: `algo.config.dagger_only=True`

- `algo.config.teacher_model_path` (默认: null)
  - Teacher 模型路径（用于 Student 训练）
  - 示例: `algo.config.teacher_model_path="logs/MotionTracking/.../model_50000.pt"`

- `algo.config.dagger_update_freq` (默认: 20)
  - DAgger 更新频率（每 N 次迭代更新一次）
  - 示例: `algo.config.dagger_update_freq=20`

### 9. 环境参数 (`env.config.*`)

- `env.config.enforce_randomize_motion_start_eval` (默认: False)
  - 评估时是否随机化动作起始点
  - 示例: `env.config.enforce_randomize_motion_start_eval=False`

### 10. 全局参数

- `project_name` (默认: "TEST")
  - 项目名称，用于日志目录
  - 示例: `project_name=MotionTracking`

- `experiment_name` (默认: "TEST")
  - 实验名称，用于日志目录
  - 示例: `experiment_name=debug`

- `num_envs` (默认: 4096)
  - 并行环境数量
  - 训练用 4096，调试用 128
  - 示例: `num_envs=128`

- `seed` (默认: 0)
  - 随机种子
  - 示例: `seed=1`

- `+device` (默认: "cuda:0" if available, else "cpu")
  - GPU 设备
  - 示例: `+device=cuda:0` 或 `+device=cuda:1`

- `headless` (默认: True)
  - 是否无头模式（不显示 GUI）
  - 示例: `headless=True`

- `use_wandb` (默认: False)
  - 是否使用 Weights & Biases 记录训练
  - 示例: `use_wandb=True`

- `checkpoint` (默认: null)
  - 检查点路径（用于继续训练）
  - 示例: `checkpoint=logs/MotionTracking/.../model_10000.pt`

---

## 训练模式

### 模式 1: 基础策略训练（Motion Tracking）

用于训练简单的动作跟踪策略，适用于固定动作模式。

**特点：**
- 使用 `mh_ppo` 算法
- 单阶段训练
- 观察空间简单（相位+重力）
- 适用于简单固定动作

**基本命令：**
```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=128 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=debug \
robot.motion.motion_file="example/motion_data/Horse-stance_pose.pkl" \
seed=1 \
+device=cuda:0
```

### 模式 2: 基准测试训练（Benchmark）

用于评估动作数据质量，不可部署。

**特点：**
- 无域随机化
- 包含特权信息
- 用于检查动作质量和难度

**基本命令：**
```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=128 \
+obs=motion_tracking/benchmark \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_nil \
+rewards=motion_tracking/main \
experiment_name=benchmark \
robot.motion.motion_file="example/motion_data/Horse-stance_pose.pkl" \
seed=1 \
+device=cuda:0
```

### 模式 3: 通用动作策略训练（General Motion Tracking）

用于训练复杂的通用动作策略，支持 Teacher-Student 训练。

#### 3.1 Teacher 策略训练

**特点：**
- 使用 `ppo_mimic` 算法
- 观察空间包含更多信息
- 支持动作重采样

**基本命令：**
```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=general_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=128 \
+obs=motion_tracking/obs_ppo_teacher \
+robot=g1/g1_23dof_general \
+domain_rand=main \
+rewards=motion_tracking/general_main \
experiment_name=debug-teacher \
robot.motion.motion_file="<path to your motion data>" \
seed=1 \
+device=cuda:0
```

#### 3.2 Student 策略训练（DAgger）

**特点：**
- 使用 Teacher 模型指导训练
- 观察空间更接近真实部署
- 使用 DAgger 算法

**基本命令：**
```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=general_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=128 \
+obs=motion_tracking/obs_ppo_student \
+robot=g1/g1_23dof_general \
+domain_rand=main \
+rewards=motion_tracking/general_main \
experiment_name=debug-student \
robot.motion.motion_file="<path to your motion data>" \
algo.config.dagger_only=True \
algo.config.teacher_model_path="<path to your teacher ckpt>" \
seed=1 \
+device=cuda:0
```

---

## 使用示例

### 示例 1: 基础策略训练（调试模式）

```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=128 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=debug \
robot.motion.motion_file="example/motion_data/Horse-stance_pose.pkl" \
seed=1 \
+device=cuda:0
```

### 示例 2: 基础策略训练（正式训练，50000 迭代）

```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=horse_stance_pose \
robot.motion.motion_file="example/motion_data/Horse-stance_pose.pkl" \
seed=1 \
+device=cuda:0 \
algo.config.num_learning_iterations=50000 \
algo.config.save_interval=1000
```

### 示例 3: 自定义训练参数

```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=256 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=custom_training \
robot.motion.motion_file="example/motion_data/Horse-stance_pose.pkl" \
seed=42 \
+device=cuda:0 \
algo.config.num_learning_iterations=10000 \
algo.config.save_interval=500 \
algo.config.logging_interval=10 \
algo.config.learning_rate=5.e-4 \
algo.config.num_learning_epochs=10
```

### 示例 4: 从检查点继续训练

```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=continue_training \
robot.motion.motion_file="example/motion_data/Horse-stance_pose.pkl" \
seed=1 \
+device=cuda:0 \
checkpoint=logs/MotionTracking/20240101_120000-debug-motion_tracking-g1_23dof_lock_wrist/model_10000.pt \
algo.config.num_learning_iterations=50000
```

### 示例 5: Teacher 策略训练

```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=general_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/obs_ppo_teacher \
+robot=g1/g1_23dof_general \
+domain_rand=main \
+rewards=motion_tracking/general_main \
experiment_name=teacher_training \
robot.motion.motion_file="example/motion_data/Complex_motion.pkl" \
seed=1 \
+device=cuda:0 \
algo.config.num_learning_iterations=50000
```

### 示例 6: Student 策略训练（DAgger）

```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=general_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/obs_ppo_student \
+robot=g1/g1_23dof_general \
+domain_rand=main \
+rewards=motion_tracking/general_main \
experiment_name=student_training \
robot.motion.motion_file="example/motion_data/Complex_motion.pkl" \
algo.config.dagger_only=True \
algo.config.teacher_model_path="logs/MotionTracking/20240101_120000-teacher_training-general_tracking-g1_23dof_general/model_50000.pt" \
seed=1 \
+device=cuda:0 \
algo.config.num_learning_iterations=50000
```

### 示例 7: 使用 Weights & Biases 记录训练

```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=wandb_training \
robot.motion.motion_file="example/motion_data/Horse-stance_pose.pkl" \
seed=1 \
+device=cuda:0 \
use_wandb=True
```

---

## 输出文件

### 输出目录结构

训练输出保存在以下目录：

```
logs/
└── {project_name}/
    └── {timestamp}-{experiment_name}-{log_task_name}-{robot_type}/
        ├── config.yaml              # 训练配置文件（完整配置）
        ├── train.log                 # 训练日志
        ├── model_1000.pt            # 检查点文件（每 save_interval 次迭代保存）
        ├── model_2000.pt
        ├── ...
        ├── model_50000.pt           # 最终模型
        └── .hydra/                   # Hydra 配置目录
            ├── config.yaml
            └── hydra.yaml
```

### 文件说明

- **`config.yaml`**: 完整的训练配置，包含所有参数设置
- **`train.log`**: 训练过程的详细日志
- **`model_*.pt`**: 模型检查点文件，可用于：
  - 继续训练
  - 模型评估
  - 导出为 ONNX 格式用于部署

### 检查点命名规则

检查点文件命名格式：`model_{iteration}.pt`

例如：
- `model_1000.pt` - 第 1000 次迭代的模型
- `model_50000.pt` - 第 50000 次迭代的模型

---

## 训练参数调整

### 方法 1: 命令行参数覆盖（推荐用于临时调整）

在训练命令中直接添加参数：

```bash
algo.config.num_learning_iterations=50000 \
algo.config.save_interval=500 \
algo.config.learning_rate=1.e-3
```

### 方法 2: 修改配置文件（推荐用于长期设置）

编辑对应的算法配置文件：

- **基础策略训练**: `humanoidverse/config/algo/mh_ppo.yaml`
- **通用策略训练**: `humanoidverse/config/algo/ppo_mimic.yaml`

例如，修改迭代次数：

```yaml
config:
  num_learning_iterations: 50000  # 从 1000000 改为 50000
  save_interval: 500              # 从 1000 改为 500
```

### 常用参数调整建议

| 参数 | 调试值 | 训练值 | 说明 |
|------|--------|--------|------|
| `num_envs` | 128 | 4096 | 环境数量，影响训练速度和稳定性 |
| `num_learning_iterations` | 1000 | 50000 | 迭代次数，论文中训练 50000 次 |
| `save_interval` | 100 | 1000 | 保存间隔，调试时可设置更小 |
| `logging_interval` | 10 | 25 | 日志间隔，调试时可设置更小 |
| `learning_rate` | 1.e-3 | 1.e-4 (ppo_mimic) / 1.e-3 (mh_ppo) | 学习率，根据算法类型调整 |

---

## 常见问题

### Q1: 训练时出现 CUDA 内存不足错误

**解决方案：**
- 减少 `num_envs`（例如从 4096 改为 2048 或 1024）
- 减少 `num_steps_per_env`（例如从 24 改为 12）
- 减少 `num_mini_batches`（例如从 4 改为 2）

### Q2: 如何从检查点继续训练？

**解决方案：**
在训练命令中添加 `checkpoint` 参数：

```bash
checkpoint=logs/MotionTracking/.../model_10000.pt
```

系统会自动加载检查点并继续训练。

### Q3: 训练速度很慢怎么办？

**解决方案：**
- 确保使用 GPU：`+device=cuda:0`
- 增加 `num_envs`（如果内存允许）
- 减少 `num_learning_epochs`（例如从 5 改为 3）
- 使用 `headless=True` 关闭 GUI

### Q4: 如何监控训练进度？

**解决方案：**
- 查看 `train.log` 文件
- 使用 TensorBoard（如果启用）
- 使用 Weights & Biases：`use_wandb=True`

### Q5: 训练中断后如何恢复？

**解决方案：**
- 使用 `checkpoint` 参数加载最新的检查点
- 检查点保存在 `logs/{project_name}/{experiment_dir}/model_*.pt`

### Q6: 如何调整训练迭代次数？

**解决方案：**
- 命令行方式：`algo.config.num_learning_iterations=50000`
- 配置文件方式：编辑 `humanoidverse/config/algo/mh_ppo.yaml` 或 `ppo_mimic.yaml`

### Q7: Teacher-Student 训练时找不到 Teacher 模型

**解决方案：**
- 确保 `algo.config.teacher_model_path` 路径正确
- 确保 Teacher 模型已训练完成
- 检查路径中的时间戳和实验名称是否正确

### Q8: 训练时出现 "motion_file not found" 错误

**解决方案：**
- 检查 `robot.motion.motion_file` 路径是否正确
- 确保动作文件是 `.pkl` 格式
- 使用绝对路径或相对于项目根目录的路径

### Q9: 如何选择训练模式？

**解决方案：**
- **简单固定动作** → 使用 `+exp=motion_tracking`（基础策略训练）
- **复杂动作序列** → 使用 `+exp=general_tracking`（通用策略训练）
- **评估动作质量** → 使用 `+obs=motion_tracking/benchmark`（基准测试）

### Q10: 训练输出保存在哪里？

**解决方案：**
- 默认保存在 `logs/{project_name}/{timestamp}-{experiment_name}-{log_task_name}-{robot_type}/`
- 可以通过 `base_dir` 参数修改基础目录（默认 `logs`）

---

## 相关文档

- [README.md](README.md) - 项目总体说明
- [URCI_USAGE.md](URCI_USAGE.md) - 模型部署使用文档
- 配置文件位置：
  - 算法配置: `humanoidverse/config/algo/`
  - 实验配置: `humanoidverse/config/exp/`
  - 观察空间配置: `humanoidverse/config/obs/`
  - 机器人配置: `humanoidverse/config/robot/`

---

## 注意事项

1. **GPU 要求**: 训练需要 CUDA 支持的 GPU，建议至少 8GB 显存
2. **训练时间**: 50000 次迭代可能需要数小时到数天，取决于硬件配置
3. **内存要求**: 建议至少 16GB 系统内存
4. **动作数据**: 确保动作数据文件格式正确（`.pkl` 格式）
5. **检查点**: 定期保存检查点，避免训练中断导致进度丢失
6. **实验命名**: 使用有意义的 `experiment_name`，便于后续查找和管理

---

**最后更新**: 2024年

