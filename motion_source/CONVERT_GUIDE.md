# GVHMR 测试指南：从视频到PKL

本文档说明如何使用GVHMR从视频提取人体运动数据，并转换为机器人可用的PKL格式。


## 快速参考（gangster.mp4示例）

```bash
# 1. 视频提取（在GVHMR目录）
cd motion_source/GVHMR
conda activate gvhmr
python ../demo.py --video ../videos/gangster.mp4 --output_root    outputs/gangster --gpu 0

# 2. 准备数据（在项目根目录）
cd ../../  # 从 motion_source/GVHMR 回到项目根目录
mkdir -p test_motion_data/video_motion
cp motion_source/GVHMR/outputs/gangster/gangster/smpl.npz test_motion_data/video_motion/gangster.npz

# 3. 重定向（在 smpl_retarget 目录）
cd smpl_retarget
python mink_retarget/convert_fit_motion.py ../test_motion_data \
    --robot-type g1 \
    --humanoid-type smpl \
    --force-retarget \
    --humanoid-mjcf-path ../description/robots/g1/smpl_humanoid.xml \
    --correct \
    --correct-mode force  # 可选: "force" (强制贴地，推荐) 或 "contact" (基于接触检测)
    # 注意：默认会自动修复瞬移异常（--fix-flying），速度阈值默认10m/s
   
# 4. 结果位置
# PKL文件：smpl_retarget/retargeted_motion_data/mink/gangster.pkl
```



## 为什么使用 Mink Retarget 而不是 PHC Retarget？

本项目使用 **Mink Retarget** 方法进行运动重定向，而不是 PHC Retarget。主要理由如下：

### 1. **技术原理差异**

- **Mink Retarget**：基于**微分逆运动学框架**（differential inverse kinematics）
  - 逐帧实时求解，每帧只需少量优化步骤（通常2步）
  - 使用速度限制和配置限制保证物理合理性
  - 计算效率高，适合批量处理大量运动数据

- **PHC Retarget**：基于**梯度优化**（gradient-based optimization）
  - 需要多次迭代优化整个序列（默认1000次迭代）
  - 使用前向运动学计算损失，然后反向传播
  - 可能在某些情况下更精确，但计算时间显著更长

### 2. **性能优势**

- **速度**：Mink 方法逐帧处理，速度更快，适合处理大量视频数据
- **实时性**：Mink 方法可以实时处理，PHC 需要等待整个序列优化完成
- **资源消耗**：Mink 方法内存占用更少，适合批量处理

### 3. **项目选择**

根据 `smpl_retarget/README.md` 的说明：
> "Both methods can be used to retarget human motion to the robot with slightly different results. **We use Mink pipeline in our experiments.**"

项目在实验中选择了 Mink 方法，因为它在速度和质量之间取得了更好的平衡。

### 4. **何时考虑使用 PHC Retarget**

如果你需要：
- 对特定运动序列进行更精细的优化
- 不介意更长的处理时间
- 需要最大化重定向精度

可以考虑使用 PHC Retarget。但通常情况下，Mink Retarget 已经能够提供足够好的结果。



## 前置条件

### 视频提取部分（步骤1）
1. ✅ 已创建conda环境：`conda create -y -n gvhmr python=3.10`
2. ✅ 已安装GVHMR依赖：`pip install -r requirements.txt` 和 `pip install -e .`
3. ✅ 已替换demo.py（使用 `motion_source/demo.py` 替换 `GVHMR/tools/demo/demo.py`）
   - 或者直接使用 `motion_source/demo.py`，无需替换
4. ✅ 已下载GVHMR所需的模型权重（参考 `GVHMR/docs/INSTALL.md`）

### 重定向部分（步骤2）
5. ✅ 已下载SMPL模型文件并放置在 `smpl_retarget/smpl_model/smpl/` 目录下（见步骤2.2）
6. ✅ 已安装 `poselib`：`cd smpl_retarget/poselib && pip install -e .`
7. ✅ 已安装 `smpl_sim`：`pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master`

> **注意**：`motion_source/demo.py` 是修改后的版本，会自动保存SMPL格式的npz文件。如果使用GVHMR原始的demo.py，需要手动修改代码来保存npz文件。

## 完整流程

### 步骤1：视频 → SMPL格式（.npz）

**运行位置**：在 `motion_source/GVHMR/` 目录下

> **重要**：必须在 `GVHMR` 目录运行，因为 `demo.py` 中的路径（如 `hmr4d/utils/body_model/...` 和配置文件中的 `inputs/checkpoints/...`）都是相对于 GVHMR 目录的。

```bash
cd motion_source/GVHMR
conda activate gvhmr
python ../demo.py --video ../videos/gangster.mp4 --output_root outputs/gangster --gpu 0
```

**参数说明**：
- `--video`: 输入视频路径（相对于GVHMR目录，所以使用 `../videos/gangster.mp4`）
- `--output_root`: 输出目录（可选，默认为 `outputs/demo`）
- `--gpu`: GPU设备ID（可选，默认为0）。如果有多张显卡，可以指定不同的GPU来并行处理多条视频

**输出**：
- 输出目录：`motion_source/GVHMR/outputs/gangster/gangster/`（或 `motion_source/GVHMR/outputs/demo/gangster/`）
- 关键文件：`smpl.npz` - SMPL格式的运动数据
- 其他输出：渲染视频等（可选）

**验证输出**：
```bash
# 检查npz文件是否存在（在GVHMR目录下）
# 如果指定了 --output_root outputs/gangster
ls outputs/gangster/gangster/smpl.npz

# 如果使用默认输出目录
ls outputs/demo/gangster/smpl.npz

# 如果路径不对，可以搜索文件
find outputs -name "smpl.npz" -type f
```

### 步骤2：SMPL格式 → 机器人重定向 → PKL

**运行位置**：在 `smpl_retarget/` 目录下

> **重要**：根据 `smpl_retarget/README.md`，重定向命令应该在 `smpl_retarget/` 目录下运行，而不是项目根目录。

#### 2.1 准备数据目录结构

将npz文件组织到合适的目录结构中：

```bash
# 在项目根目录创建临时数据目录
mkdir -p test_motion_data/video_motion

# 复制npz文件到数据目录
cp motion_source/GVHMR/outputs/gangster/gangster/smpl.npz test_motion_data/video_motion/gangster.npz
```

> **注意**：数据目录可以在项目根目录，也可以在 `smpl_retarget/` 目录下。如果放在项目根目录，运行命令时需要使用相对于 `smpl_retarget/` 的路径（如 `../test_motion_data`）。

#### 2.2 准备SMPL模型文件（必需）

重定向脚本需要SMPL模型文件。如果还没有下载，请按照以下步骤：

1. **下载SMPL模型**：
   - 访问 [SMPL官网](https://smpl.is.tue.mpg.de/) 注册并下载 SMPL v1.1.0
   - 下载文件：`basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl`

2. **创建目录并放置文件**：
```bash
# 在 smpl_retarget 目录下创建目录结构
cd smpl_retarget
mkdir -p smpl_model/smpl

# 将下载的文件复制到 smpl_model/smpl/ 并重命名
# basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl -> SMPL_NEUTRAL.pkl
# basicmodel_m_lbs_10_207_0_v1.1.0.pkl -> SMPL_MALE.pkl
# basicmodel_f_lbs_10_207_0_v1.1.0.pkl -> SMPL_FEMALE.pkl
```

目录结构应该是：
```
smpl_retarget/
└── smpl_model/
    └── smpl/
        ├── SMPL_NEUTRAL.pkl
        ├── SMPL_MALE.pkl
        └── SMPL_FEMALE.pkl
```

#### 2.3 安装依赖包

**安装 poselib**（如果未安装）：
```bash
cd smpl_retarget/poselib
pip install -e .
cd ../..
```

**安装 smpl_sim**（必需）：
```bash
# 在项目根目录执行
# 方式1：使用HTTPS（推荐，适用于大多数情况）
pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master

# 方式2：使用SSH（如果你配置了SSH密钥）
pip install git+ssh://git@github.com/ZhengyiLuo/SMPLSim.git@master
```

> **注意**：`smpl_sim` 是运行重定向脚本的必需依赖，但它在 `setup.py` 中被注释掉了，需要手动安装。你可以选择HTTPS或SSH方式安装，SSH方式需要先配置GitHub的SSH密钥。

#### 2.4 运行重定向

```bash
# 切换到 smpl_retarget 目录
cd smpl_retarget

# 运行重定向命令
# 注意：如果数据目录在项目根目录，使用 ../test_motion_data
# 如果数据目录在 smpl_retarget 目录下，直接使用 test_motion_data
python mink_retarget/convert_fit_motion.py ../test_motion_data \
    --robot-type g1 \
    --humanoid-type smpl \
    --force-retarget \
    --humanoid-mjcf-path ../description/robots/g1/smpl_humanoid.xml \
    --correct \
    --correct-mode force  # 可选: "force" (强制贴地，推荐) 或 "contact" (基于接触检测)
    # 注意：默认会自动修复瞬移异常（--fix-flying），速度阈值默认10m/s
```

> **重要**：
> - 必须在 `smpl_retarget/` 目录下运行命令（根据 `smpl_retarget/README.md`）
> - `--humanoid-mjcf-path` 路径是相对于**当前工作目录**（`smpl_retarget/`）的，所以使用 `../description/robots/g1/smpl_humanoid.xml`
> - 如果数据目录在项目根目录，使用 `../test_motion_data`；如果在 `smpl_retarget/` 目录下，直接使用 `test_motion_data`

**参数说明**：
- `test_motion_data`: 运动数据根目录（位置参数，包含npz文件的文件夹）
- `--robot-type g1`: 目标机器人类型（Unitree G1，默认：g1）
- `--humanoid-type smpl`: 人体模型类型（SMPL，默认：smpl）
- `--force-retarget`: 强制重定向（对于g1机器人必须设置）
- `--humanoid-mjcf-path`: SMPL人形模型的MJCF文件路径（相对于项目根目录，必需）
- `--correct`: **重要！**启用运动修正，确保机器人脚部贴地，避免浮空问题（强烈推荐使用）
- `--correct-mode`: 修正模式，可选值：
  - `"force"`（默认）：强制贴地模式，使用零相位巴特沃斯滤波器去噪
    - 适用于GVHMR等容易产生Z轴漂移的数据
    - 无相位滞后，动作与视频同步
    - 强去噪能力，精确切掉高频抖动（>3Hz）
    - 可调节 `cutoff` 参数（推荐3.0）控制平滑度 vs 贴地紧密度
  - `"contact"`：接触检测模式，基于脚部速度和高度检测接触状态，适用于包含跳跃动作的高质量动捕数据
- `--fix-flying`: **新增！** 启用瞬移异常修复（默认：`True`）
  - 自动检测并修复速度超过阈值的异常帧（瞬移）
  - 防止物理模拟器因极高速度而崩溃
  - 如果检测到异常帧，会在控制台输出警告信息
  - 建议保持默认开启，除非数据质量极高且不需要此保护
- `--flying-threshold`: 速度阈值（默认：`10.0` m/s）
  - 当两帧之间的根节点速度超过此阈值时，会被判定为瞬移异常
  - 默认值 10.0 m/s 相当于人类极限冲刺速度（博尔特级别）
  - 异常帧会被强制锁定在上一帧的位置（原地踏步）
  - 可根据数据质量调整：数据质量高可适当提高（如 15.0），数据质量差可降低（如 5.0）

**输出**：
- 重定向后的npy文件：`test_motion_data/video_motion-g1_retargeted_npy/gangster.npy`（相对于数据目录）
- **PKL文件**：`smpl_retarget/retargeted_motion_data/mink/gangster.pkl` ⭐（相对于 smpl_retarget 目录）

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
cd motion_source/GVHMR
conda activate gvhmr  # 确保在正确的环境中
python ../demo.py --video ../videos/gangster.mp4 --output_root outputs/gangster --gpu 0

# 检查输出（在GVHMR目录下）
ls outputs/gangster/gangster/smpl.npz  # 或 outputs/demo/gangster/smpl.npz

# ===== 步骤2：准备数据目录 =====
cd ../../  # 从 motion_source/GVHMR 回到项目根目录
mkdir -p test_motion_data/video_motion

# 复制npz文件（根据实际输出路径调整）
cp motion_source/GVHMR/outputs/gangster/gangster/smpl.npz test_motion_data/video_motion/gangster.npz
# 或者如果使用默认输出：
# cp motion_source/GVHMR/outputs/demo/gangster/smpl.npz test_motion_data/video_motion/gangster.npz

# ===== 步骤3：准备SMPL模型（如果未准备） =====
# 确保已下载SMPL模型并放置在 smpl_retarget/smpl_model/smpl/ 目录下
# 参考步骤2.2的说明

# ===== 步骤4：安装依赖（如果未安装） =====
# 安装 poselib
cd smpl_retarget/poselib
pip install -e .
cd ../..

# 安装 smpl_sim（必需）
# 方式1：使用HTTPS
pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master
# 方式2：使用SSH（如果配置了SSH密钥）
# pip install git+ssh://git@github.com/ZhengyiLuo/SMPLSim.git@master

# ===== 步骤5：重定向 =====
# 切换到 smpl_retarget 目录（重要！）
cd smpl_retarget
python mink_retarget/convert_fit_motion.py ../test_motion_data \
    --robot-type g1 \
    --humanoid-type smpl \
    --force-retarget \
    --humanoid-mjcf-path ../description/robots/g1/smpl_humanoid.xml \
    --correct \
    --correct-mode force  # 可选: "force" (强制贴地，推荐) 或 "contact" (基于接触检测)
    # 注意：默认会自动修复瞬移异常（--fix-flying），速度阈值默认10m/s

# ===== 步骤6：验证结果 =====
# PKL文件保存在 smpl_retarget/retargeted_motion_data/mink/ 目录下
ls retargeted_motion_data/mink/gangster.pkl
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
- 确认 `poselib` 已正确安装：`cd smpl_retarget/poselib && pip install -e .`
- 确认 `smpl_sim` 已正确安装：
  - HTTPS方式：`pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master`
  - SSH方式：`pip install git+ssh://git@github.com/ZhengyiLuo/SMPLSim.git@master`
- **确认SMPL模型文件已下载并正确放置**：
  - 检查 `smpl_retarget/smpl_model/smpl/` 目录是否存在
  - 确认文件已重命名为 `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`, `SMPL_FEMALE.pkl`
  - 如果遇到 `AssertionError: Path ./smpl_model/smpl does not exist!`，请参考步骤2.2设置SMPL模型
- 检查npz文件格式是否正确
- 查看错误日志
- 如果遇到 `ModuleNotFoundError: No module named 'smpl_sim'`，请安装 smpl_sim（见步骤2.3）
- 如果遇到 `FileNotFoundError: No such file or directory: '../description/robots/g1/smpl_humanoid.xml'`：
  - 确保在 `smpl_retarget/` 目录运行命令
  - 使用 `--humanoid-mjcf-path ../description/robots/g1/smpl_humanoid.xml` 参数指定正确的路径

### 3. 输出目录结构
- demo.py的输出路径：`motion_source/GVHMR/outputs/{output_root}/{video_name}/smpl.npz`
- 重定向输出：`smpl_retarget/retargeted_motion_data/mink/{filename}.pkl`

### 4. 机器人浮空问题与高度修正原理

**问题现象**：可视化检查pkl文件时，发现除了最初的几帧外，后面机器人都是浮空舞蹈的。

**原因**：没有使用`--correct`参数，导致retargeting过程中没有进行地面接触修正。`correct_motion`函数会根据接触掩码（contact_mask）调整z轴偏移，确保有接触时脚部贴地。

**解决方案**：
- 在重定向命令中添加`--correct`参数
- 该参数会：
  1. 检测脚部接触状态（通过`foot_detect`函数）
  2. 对于有接触的帧，计算顶点最低z坐标作为偏移
  3. 对于没有接触的帧，使用前一帧的偏移
  4. 从所有帧的z坐标中减去偏移，确保脚部贴地
  5. 使用EMA平滑处理，使运动更自然

**示例命令**：
```bash
# 使用 "force" 模式（推荐，适用于GVHMR等容易浮空的数据）
python mink_retarget/convert_fit_motion.py ../test_motion_data \
    --robot-type g1 \
    --humanoid-type smpl \
    --force-retarget \
    --humanoid-mjcf-path ../description/robots/g1/smpl_humanoid.xml \
    --correct \
    --correct-mode force

# 或使用 "contact" 模式（适用于包含跳跃动作的高质量动捕数据）
python mink_retarget/convert_fit_motion.py ../test_motion_data \
    --robot-type g1 \
    --humanoid-type smpl \
    --force-retarget \
    --humanoid-mjcf-path ../description/robots/g1/smpl_humanoid.xml \
    --correct \
    --correct-mode contact
```

#### 高度修正代码详解

高度修正的核心代码位于 `smpl_retarget/mink_retarget/convert_fit_motion.py`：

**1. 脚部接触检测**（```28:46:smpl_retarget/mink_retarget/convert_fit_motion.py```）：
```python
def foot_detect(positions, thres=0.002):
    fid_r, fid_l = [8, 11], [7, 10]  # 左右脚关节索引
    velfactor, heightfactor = np.array([thres, thres]), np.array([0.15, 0.1])
    # 检测脚部速度和高度，判断是否接触地面
    # 返回左右脚的接触掩码
```

**2. "contact" 模式高度修正函数**（```70:82:smpl_retarget/mink_retarget/convert_fit_motion.py```）：
```python
def correct_motion(contact_mask, verts, trans):
    # 找到有接触和无接触的帧索引
    contact_indices = np.where(np.any(contact_mask != [0, 0], axis=1))[0]
    no_contact_indices = np.where(np.all(contact_mask == [0, 0], axis=1))[0]
    
    # 计算z轴偏移
    z_offset = np.zeros_like(trans[:, :, 2])
    # 对于有接触的帧：使用顶点最低z坐标作为偏移
    z_offset[contact_indices] = torch.min(
        verts[contact_indices, :, 2], dim=1, keepdim=True
    )[0]
    # 对于无接触的帧：使用前一帧的偏移（保持连续性）
    for idx in no_contact_indices:
        z_offset[idx] = z_offset[idx - 1]
    
    # 从所有帧的z坐标中减去偏移，使脚部贴地
    trans[:, :, 2] -= z_offset
    # 使用EMA平滑处理，使运动更自然
    trans[:, :, 2] = torch.from_numpy(EMA_smooth(trans[:, :, 2]))
    return trans
```

**3. 巴特沃斯滤波器工具函数**（```69:100:smpl_retarget/mink_retarget/convert_fit_motion.py```）：
```python
def butterworth_filter(data, cutoff=10, fs=30, order=4):
    """
    零相位巴特沃斯低通滤波
    
    这是专业的动捕数据平滑函数，比 EMA 更强：
    - 无相位滞后：通过正向和反向两次滤波，消除相位偏移
    - 频率截断：精确切掉高频抖动，只保留人体运动的低频有效信息
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    
    # 设计滤波器
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # 使用 filtfilt 进行零相位滤波 (forward-backward filtering)
    y = signal.filtfilt(b, a, data, axis=0)
    
    return y
```

**4. "force" 模式强制贴地函数**（```116:177:smpl_retarget/mink_retarget/convert_fit_motion.py```）：
```python
def correct_motion_force_ground(verts, trans, fps=30, floor_offset=0.0, cutoff=3.0):
    """
    修正模式 'force': 强制每一帧的最低点贴地，使用零相位巴特沃斯滤波去噪。
    
    核心改进：使用巴特沃斯滤波器处理最低点轨迹，提取低频漂移趋势，忽略高频抖动。
    这样既能让人物贴地，又不会引入脚部的抖动噪声。
    """
    # 1. 获取每一帧所有顶点的最低 Z 值 [Batch]
    # 这是原始的、充满抖动的"脚底板高度曲线"
    min_z, _ = torch.min(verts[:, :, 2], dim=1)
    min_z_np = min_z.cpu().numpy()
    
    # 2. 核心改进：使用巴特沃斯滤波处理最低点轨迹
    # 我们假设 Z 轴漂移是缓慢变化的，所以用极低的截止频率（例如 2Hz - 5Hz）
    # 这样可以忽略单帧的脚部抖动，只保留"整体是不是飘起来了"这个信息
    smooth_min_z = butterworth_filter(min_z_np, cutoff=cutoff, fs=fps, order=4)
    
    # 3. 计算偏移量
    offset = smooth_min_z - floor_offset
    
    # 4. 应用修正：从所有关节的z坐标中减去偏移
    trans_corrected = trans.clone()
    offset_tensor = torch.from_numpy(offset).to(trans.device).float()
    trans_corrected[:, :, 2] -= offset_tensor.unsqueeze(1)
    
    # 5. 二次修正：防止穿模（可选）
    # 如果修正后脚还在地下，就硬性提上来
    with torch.no_grad():
        corrected_verts_z = verts[:, :, 2] - offset_tensor.unsqueeze(1)
        current_min_z, _ = torch.min(corrected_verts_z, dim=1)
        penetration = current_min_z.cpu().numpy()
        extra_fix = np.clip(penetration, -np.inf, 0)
        if np.any(extra_fix < 0):
            trans_corrected[:, :, 2] -= torch.from_numpy(extra_fix).to(trans.device).unsqueeze(1)
    
    return trans_corrected
```

**4. 调用流程**（```420:446:smpl_retarget/mink_retarget/convert_fit_motion.py```）：
```python
# 检测脚部接触（用于 "contact" 模式）
feet_l, feet_r = foot_detect(origin_global_trans[::skip])
contact_mask = np.concatenate([feet_l, feet_r], axis=-1)

if correct:
    if correct_mode == "contact":
        # "contact" 模式：基于接触检测
        correct_global_trans = correct_motion(
            contact_mask, 
            origin_verts[::skip], 
            global_trans[::skip]
        )
    
    elif correct_mode == "force":
        # "force" 模式：强制贴地，使用零相位巴特沃斯滤波去噪（推荐）
        correct_global_trans = correct_motion_force_ground(
            origin_verts[::skip],
            global_trans[::skip],
            fps=fps,        # 传入当前视频的 FPS (通常是30)
            floor_offset=0.0, # 地面高度，通常为0
            cutoff=3.0     # 截止频率，推荐值3.0，可根据需要调整
        )
else:
    # 不修正，直接使用原始数据（可能导致浮空）
    correct_global_trans = global_trans[::skip]
```

**修正原理**：

**"contact" 模式**：
- **有接触帧**：计算SMPL模型顶点的最低z坐标，将其作为地面高度，从所有关节的z坐标中减去这个偏移，使最低点（通常是脚部）贴地
- **无接触帧**：使用前一帧的偏移值，保持高度连续性，避免突然跳跃
- **平滑处理**：使用指数移动平均（EMA，alpha=0.3）平滑z坐标变化，使运动更自然

**"force" 模式**（使用零相位巴特沃斯滤波）：
- **核心改进**：使用专业的零相位巴特沃斯滤波器处理最低点轨迹，提取低频漂移趋势，忽略高频抖动
- **技术优势**：
  - **无相位滞后**：零相位滤波确保动作与视频同步，不会"慢半拍"
  - **强去噪能力**：精确切掉高频抖动（>3Hz），只保留人体运动的低频有效信息
  - **可调节参数**：`cutoff` 参数控制平滑度 vs 贴地紧密度
    - 推荐值：3.0（平衡平滑度和贴地紧密度）
    - 如果仍有轻微浮动，可提高到 5.0
    - 如果抖动明显，可降低到 1.5-2.0
- **技术对比**：
  | 特性 | EMA 方法 | Butterworth 方法 |
  |------|---------|-----------------|
  | 原理 | 当前帧参考上一帧的加权平均 | 频域信号处理，切除高频分量 |
  | 相位/延迟 | 有明显的滞后（机器人反应慢） | 零相位（动作与视频同步） |
  | 抗抖动 | 差（为了减少滞后必须减小力度） | 极强（可以设置很低的截止频率） |
  | 适用场景 | 简单的平滑 | 专业的动捕数据清洗 |

### 5. GPU选择与并行处理

**指定GPU设备**：
- 使用 `--gpu` 参数指定GPU设备ID（默认：0）
- 示例：`--gpu 0` 使用第一张显卡，`--gpu 1` 使用第二张显卡

**并行处理多条视频**：
如果您有多张显卡，可以在不同的GPU上并行处理多条视频：

**方法1：使用后台进程（Windows PowerShell）**
```powershell
# 在GPU 0上处理视频1
Start-Process python -ArgumentList "../demo.py --video ../videos/video1.mp4 --output_root outputs/video1 --gpu 0" -NoNewWindow

# 在GPU 1上处理视频2
Start-Process python -ArgumentList "../demo.py --video ../videos/video2.mp4 --output_root outputs/video2 --gpu 1" -NoNewWindow
```

**方法2：创建批处理脚本**
创建 `process_parallel.bat` 文件：
```batch
@echo off
cd motion_source/GVHMR
conda activate gvhmr
start "GPU0" cmd /c "python ../demo.py --video ../videos/gangster.mp4 --output_root outputs/gangster --gpu 0"
start "GPU1" cmd /c "python ../demo.py --video ../videos/video2.mp4 --output_root outputs/video2 --gpu 1"
```

**方法3：使用Python脚本并行处理**
创建 `run_parallel.py` 脚本：
```python
import subprocess
import sys
import os

# 切换到GVHMR目录
os.chdir("motion_source/GVHMR")

videos = [
    ("../videos/gangster.mp4", "outputs/gangster", 0),
    ("../videos/video2.mp4", "outputs/video2", 1),
    ("../videos/video3.mp4", "outputs/video3", 2),
]

processes = []
for video, output, gpu in videos:
    cmd = [
        sys.executable, "../demo.py",
        "--video", video,
        "--output_root", output,
        "--gpu", str(gpu)
    ]
    p = subprocess.Popen(cmd)
    processes.append(p)

# 等待所有进程完成
for p in processes:
    p.wait()
```

**注意事项**：
- 确保指定的GPU ID存在（例如，如果有2张显卡，只能使用 `--gpu 0` 和 `--gpu 1`）
- 并行处理时注意每张GPU的显存使用情况
- 建议先测试单条视频的处理时间，以便合理分配任务

### 6. 为什么必须在GVHMR目录运行？
- `demo.py` 中有硬编码的相对路径，如 `"hmr4d/utils/body_model/smplx2smpl_sparse.pt"`，这些路径是相对于 GVHMR 目录的
- 配置文件 `demo.yaml` 中的路径（如 `inputs/checkpoints/...`）也是相对于 GVHMR 目录的
- 如果不在 GVHMR 目录运行，会找不到这些文件和模型权重

## 下一步

生成的PKL文件可以用于：
- 机器人运动跟踪训练
- 运动可视化
- 进一步的运动处理

参考 `example/motion_data/` 目录查看其他示例数据格式。

