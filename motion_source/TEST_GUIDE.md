# GVHMR 测试指南：从视频到PKL

本文档说明如何使用GVHMR从视频提取人体运动数据，并转换为机器人可用的PKL格式。

## 快速参考（gangster.mp4示例）

```bash
# 1. 视频提取（在motion_source目录）
cd motion_source
conda activate gvhmr
python demo.py --video videos/gangster.mp4 --output_root outputs/gangster

# 2. 准备数据（在项目根目录）
cd ..
mkdir -p test_motion_data/video_motion
cp motion_source/outputs/gangster/gangster/smpl.npz test_motion_data/video_motion/gangster.npz

# 3. 重定向（在项目根目录）
python smpl_retarget/mink_retarget/convert_fit_motion.py test_motion_data --robot-type g1 --humanoid-type smpl --force-retarget

# 4. 结果位置
# PKL文件：smpl_retarget/retargeted_motion_data/mink/gangster.pkl
```

## 前置条件

1. ✅ 已创建conda环境：`conda create -y -n gvhmr python=3.10`
2. ✅ 已安装GVHMR依赖：`pip install -r requirements.txt` 和 `pip install -e .`
3. ✅ 已替换demo.py（使用 `motion_source/demo.py` 替换 `GVHMR/tools/demo/demo.py`）
   - 或者直接使用 `motion_source/demo.py`，无需替换
4. ✅ 已下载GVHMR所需的模型权重（参考 `GVHMR/docs/INSTALL.md`）

> **注意**：`motion_source/demo.py` 是修改后的版本，会自动保存SMPL格式的npz文件。如果使用GVHMR原始的demo.py，需要手动修改代码来保存npz文件。

## 完整流程

### 步骤1：视频 → SMPL格式（.npz）

**运行位置**：在 `motion_source/` 目录下

```bash
cd motion_source
python demo.py --video videos/gangster.mp4 --output_root outputs/gangster
```

**参数说明**：
- `--video`: 输入视频路径（相对于motion_source目录）
- `--output_root`: 输出目录（可选，默认为 `outputs/demo`）

**输出**：
- 输出目录：`outputs/gangster/gangster/`（或 `outputs/demo/gangster/`）
- 关键文件：`smpl.npz` - SMPL格式的运动数据
- 其他输出：渲染视频等（可选）

**验证输出**：
```bash
# 检查npz文件是否存在
# 如果指定了 --output_root outputs/gangster
ls outputs/gangster/gangster/smpl.npz

# 如果使用默认输出目录
ls outputs/demo/gangster/smpl.npz

# 如果路径不对，可以搜索文件
find outputs -name "smpl.npz" -type f
```

### 步骤2：SMPL格式 → 机器人重定向 → PKL

**运行位置**：在项目根目录 `PBHC/` 下

#### 2.1 准备数据目录结构

将npz文件组织到合适的目录结构中：

```bash
# 在项目根目录创建临时数据目录
mkdir -p test_motion_data/video_motion

# 复制npz文件到数据目录
cp motion_source/outputs/gangster/gangster/smpl.npz test_motion_data/video_motion/gangster.npz
```

#### 2.2 安装poselib（如果未安装）

```bash
cd smpl_retarget/poselib
pip install -e .
cd ../..
```

#### 2.3 运行重定向

```bash
# 在项目根目录执行
python smpl_retarget/mink_retarget/convert_fit_motion.py test_motion_data --robot-type g1 --humanoid-type smpl --force-retarget
```

**参数说明**：
- `test_motion_data`: 运动数据根目录（位置参数，包含npz文件的文件夹）
- `--robot-type g1`: 目标机器人类型（Unitree G1，默认：g1）
- `--humanoid-type smpl`: 人体模型类型（SMPL，默认：smpl）
- `--force-retarget`: 强制重定向（对于g1机器人必须设置）

**输出**：
- 重定向后的npy文件：`test_motion_data/video_motion-g1_retargeted_npy/gangster.npy`
- **PKL文件**：`smpl_retarget/retargeted_motion_data/mink/gangster.pkl` ⭐

**验证输出**：
```bash
# 检查pkl文件
ls smpl_retarget/retargeted_motion_data/mink/gangster.pkl

# 查看pkl内容（可选）
python robot_motion_process/motion_readpkl.py smpl_retarget/retargeted_motion_data/mink/gangster.pkl
```

## 快速测试命令（完整流程）

```bash
# ===== 步骤1：视频提取 =====
cd motion_source
conda activate gvhmr  # 确保在正确的环境中
python demo.py --video videos/gangster.mp4 --output_root outputs/gangster

# 检查输出
ls outputs/gangster/gangster/smpl.npz  # 或 outputs/demo/gangster/smpl.npz

# ===== 步骤2：准备数据目录 =====
cd ..  # 回到项目根目录
mkdir -p test_motion_data/video_motion

# 复制npz文件（根据实际输出路径调整）
cp motion_source/outputs/gangster/gangster/smpl.npz test_motion_data/video_motion/gangster.npz
# 或者如果使用默认输出：
# cp motion_source/outputs/demo/gangster/smpl.npz test_motion_data/video_motion/gangster.npz

# ===== 步骤3：重定向 =====
# 确保在项目根目录，并激活包含poselib的环境
python smpl_retarget/mink_retarget/convert_fit_motion.py test_motion_data --robot-type g1 --humanoid-type smpl --force-retarget

# ===== 步骤4：验证结果 =====
ls smpl_retarget/retargeted_motion_data/mink/gangster.pkl
```

## 输出文件说明

### SMPL格式（.npz）
包含以下字段：
- `betas`: (10,) - SMPL形状参数
- `gender`: str - 性别（'neutral'）
- `poses`: (frames, 66) - 关节旋转（轴角表示）
- `trans`: (frames, 3) - 根节点全局平移
- `mocap_framerate`: float - 帧率（30.0）

### PKL格式（.pkl）
包含以下字段：
- `root_trans_offset`: 根节点平移
- `pose_aa`: 轴角表示的姿态
- `dof`: 自由度数据
- `root_rot`: 根节点旋转（四元数）
- `contact_mask`: 接触掩码
- `fps`: 帧率

## 常见问题

### 1. 找不到smpl.npz文件
- 检查输出目录路径是否正确
- 查看demo.py运行日志，确认输出目录

### 2. 重定向失败
- 确认poselib已正确安装
- 检查npz文件格式是否正确
- 查看错误日志

### 3. 输出目录结构
- demo.py的输出路径：`outputs/{output_root}/{video_name}/smpl.npz`
- 重定向输出：`smpl_retarget/retargeted_motion_data/mink/{filename}.pkl`

## 下一步

生成的PKL文件可以用于：
- 机器人运动跟踪训练
- 运动可视化
- 进一步的运动处理

参考 `example/motion_data/` 目录查看其他示例数据格式。

