# GVHMR 模型文件下载指南

本文档说明GVHMR运行所需的模型文件，以及哪些是**必须下载**的，哪些是**可选的**。

## 📋 必须下载的模型文件

### 1. GVHMR主模型 ⭐ **必须**

**文件**：`gvhmr_siga24_release.ckpt`  
**位置**：`inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt`  
**用途**：GVHMR的核心模型，用于从视频提取人体运动  
**下载**：从 [Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link) 下载

### 2. YOLO检测模型 ⭐ **必须**

**文件**：`yolov8x.pt`  
**位置**：`inputs/checkpoints/yolo/yolov8x.pt`  
**用途**：用于检测和跟踪视频中的人体（bounding box）  
**下载**：从 [Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link) 下载

### 3. ViTPose模型 ⭐ **必须**

**文件**：`vitpose-h-multi-coco.pth`  
**位置**：`inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth`  
**用途**：用于提取2D关键点（pose estimation）  
**下载**：从 [Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link) 下载

### 4. SMPL/SMPLX模型 ⭐ **必须**

**文件结构**：
```
inputs/checkpoints/body_models/
├── smplx/
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_MALE.npz
│   └── SMPLX_FEMALE.npz
└── smpl/
    ├── SMPL_NEUTRAL.pkl
    ├── SMPL_MALE.pkl
    └── SMPL_FEMALE.pkl
```

**用途**：用于渲染和评估人体模型  
**下载**：
1. 访问 [SMPL官网](https://smpl.is.tue.mpg.de/) 注册并下载SMPL模型
2. 访问 [SMPLX官网](https://smpl-x.is.tue.mpg.de/) 注册并下载SMPLX模型
3. 将文件重命名并放置到上述目录结构中

**注意**：至少需要 `SMPLX_NEUTRAL.npz` 和 `SMPL_NEUTRAL.pkl`（demo.py主要使用neutral模型）

## 🔧 可选的模型文件

### DPVO模型（可选）

**文件**：`dpvo.pth`  
**位置**：`inputs/checkpoints/dpvo/dpvo.pth`  
**用途**：用于估计相机运动（Visual Odometry）  
**是否需要**：❌ **不需要**  
**原因**：GVHMR默认使用SimpleVO（更高效），DPVO是可选的且不推荐使用（影响速度）

### HMR2模型（可选）

**文件**：`epoch=10-step=25000.ckpt`  
**位置**：`inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt`  
**用途**：可能用于某些评估任务  
**是否需要**：❌ **demo不需要**

## 📁 目录结构设置

在 `motion_source/GVHMR/` 目录下创建以下结构：

```bash
cd motion_source/GVHMR
mkdir -p inputs/checkpoints/{gvhmr,yolo,vitpose,body_models/{smplx,smpl},dpvo,hmr2}
```

## 📥 下载步骤

### 方法1：从Google Drive下载（推荐）

1. **访问Google Drive**：
   - 链接：https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link
   - 同意相应的许可证

2. **下载必须的文件**：
   ```bash
   cd motion_source/GVHMR/inputs/checkpoints
   
   # 下载gvhmr模型
   # 将 gvhmr_siga24_release.ckpt 放到 gvhmr/ 目录
   
   # 下载yolo模型
   # 将 yolov8x.pt 放到 yolo/ 目录
   
   # 下载vitpose模型
   # 将 vitpose-h-multi-coco.pth 放到 vitpose/ 目录
   ```

3. **下载SMPL/SMPLX模型**：
   - 访问 https://smpl.is.tue.mpg.de/ 注册并下载
   - 访问 https://smpl-x.is.tue.mpg.de/ 注册并下载
   - 将文件重命名并放置到 `body_models/` 目录

### 方法2：使用命令行下载（如果提供直接链接）

```bash
cd motion_source/GVHMR/inputs/checkpoints

# 下载gvhmr模型（需要从Google Drive获取实际下载链接）
# wget -O gvhmr/gvhmr_siga24_release.ckpt <下载链接>

# 下载yolo模型
# wget -O yolo/yolov8x.pt <下载链接>

# 下载vitpose模型
# wget -O vitpose/vitpose-h-multi-coco.pth <下载链接>
```

## ✅ 验证安装

运行以下命令检查模型文件是否存在：

```bash
cd motion_source/GVHMR

# 检查必须的模型文件
ls inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
ls inputs/checkpoints/yolo/yolov8x.pt
ls inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
ls inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
ls inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
```

如果所有文件都存在，说明模型文件已正确安装。

## 📝 总结

**必须下载的文件**（4类）：
1. ✅ `gvhmr_siga24_release.ckpt` - GVHMR主模型
2. ✅ `yolov8x.pt` - YOLO检测模型
3. ✅ `vitpose-h-multi-coco.pth` - ViTPose模型
4. ✅ SMPL/SMPLX模型文件（至少NEUTRAL版本）

**不需要的文件**：
- ❌ `dpvo.pth` - DPVO模型（默认不使用）
- ❌ `hmr2/epoch=10-step=25000.ckpt` - HMR2模型（demo不需要）

## 🔗 相关链接

- GVHMR Google Drive：https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link
- SMPL官网：https://smpl.is.tue.mpg.de/
- SMPLX官网：https://smpl-x.is.tue.mpg.de/

