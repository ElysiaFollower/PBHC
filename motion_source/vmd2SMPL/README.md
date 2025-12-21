# VMD to SMPL Converter

This directory contains scripts to convert VMD (MikuMikuDance) motion files to SMPL format.

## Overview

This converter transforms VMD (MikuMikuDance) motion files into SMPL format. The conversion process involves:

1. **Parsing VMD file**: Extract bone animation data from VMD binary format
2. **Bone mapping**: Map MMD bone names to SMPL joint structure
3. **SMPL conversion**: Convert VMD bone rotations to SMPL pose format (axis-angle representation)
4. **Output**: Save as `.npz` file in SMPL format

## Requirements

### 1. Install basic dependencies:

```bash
pip install numpy scipy torch joblib
```

### 2. Install SMPLSim (REQUIRED):

The `smpl_sim` module is required for SMPL processing and retargeting. Install it from GitHub:

```bash
pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master
```

Or if you have SSH access configured:

```bash
pip install git+ssh://git@github.com/ZhengyiLuo/SMPLSim.git@master
```

**Note**: This package is not included in the main `setup.py` requirements, so it must be installed separately.

### 3. Install poselib (REQUIRED):

```bash
cd smpl_retarget/poselib
pip install -e .
cd ../../..
```

### 4. Optional: VMD parsing library

For VMD parsing, you can use one of these libraries:
- `vmd-python`: `pip install vmd-python`
- Or the script will fall back to manual parsing

**Important**: Make sure you're running from the project root directory, or the script will automatically change to the project root.

## Usage

### Windows PowerShell

When using PowerShell with filenames containing special characters (like parentheses), use quotes:

```powershell
python vmd_to_smpl.py --input "vmd/愛包ダンスホール_DanceMotion_HimeTanaka(MMD)_v1.1.vmd" --output output.npz
```

### Linux/Mac

```bash
python vmd_to_smpl.py --input vmd/input.vmd --output output.npz
```

### With custom frame rate:

```bash
python vmd_to_smpl.py --input vmd/input.vmd --output output.npz --fps 30
```

### Using example script:

```bash
python example_usage.py
```

## Arguments (vmd_to_smpl.py)

- `--input`: Path to input VMD file (required)
- `--output`: Path to output SMPL file (.npz format, required)
- `--fps`: Frame rate (default: 30.0)

## Output Format (SMPL .npz file)

The output `.npz` file contains a dictionary with the following keys:

- `pose_aa`: SMPL pose in axis-angle format [num_frames, 72] (24 joints * 3)
- `poses`: SMPL pose in axis-angle format [num_frames, 66] (22 joints * 3, for compatibility)
- `trans`: Root translation [num_frames, 3]
- `betas`: SMPL shape parameters [10] (set to zeros)
- `gender`: Gender string (default: 'neutral')
- `fps`: Frame rate (float)
- `mocap_framerate`: Frame rate (int)

## Bone Mapping

The script maps MMD bone names to SMPL joints:

| MMD Bone Name | SMPL Joint |
|--------------|------------|
| センター (Center) | Pelvis |
| 上半身 (Upper body) | Spine1 |
| 上半身2 (Upper body 2) | Spine2 |
| 首 (Neck) | Neck |
| 頭 (Head) | Head |
| 左肩 (Left shoulder) | L_Shoulder |
| 左腕 (Left arm) | L_UpperArm |
| 左ひじ (Left elbow) | L_ForeArm |
| 左手首 (Left wrist) | L_Hand |
| 右肩 (Right shoulder) | R_Shoulder |
| 右腕 (Right arm) | R_UpperArm |
| 右ひじ (Right elbow) | R_ForeArm |
| 右手首 (Right wrist) | R_Hand |
| 左足 (Left leg) | L_Hip |
| 左ひざ (Left knee) | L_Knee |
| 左足首 (Left ankle) | L_Ankle |
| 右足 (Right leg) | R_Hip |
| 右ひざ (Right knee) | R_Knee |
| 右足首 (Right ankle) | R_Ankle |

## Notes

- The script handles VMD files with Japanese character encoding (Shift-JIS)
- If a bone is not found in the mapping, it will be skipped
- **The script automatically changes the working directory to the project root** to ensure correct imports

## Troubleshooting

### ModuleNotFoundError: No module named 'smpl_sim'

This error means `smpl_sim` is not installed. Install it using:

```bash
pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master
```

**Important**: `smpl_sim` is a required dependency that must be installed separately. It's not included in the main project's `setup.py` requirements.

After installation, verify it works:
```bash
python -c "import smpl_sim; print('smpl_sim installed successfully')"
```

### PowerShell filename issues

If PowerShell interprets parentheses in filenames as commands, use quotes around the filename:

```powershell
python vmd_to_smpl.py --input "vmd/file(MMD).vmd" --output output.npz
```

### Import errors

If you encounter import errors, make sure:
- You're running from the project root directory
- All dependencies are installed: `pip install -r requirements.txt`
- The `smpl_sim` module is available (check if `smpl_retarget/poselib` exists)

If VMD parsing fails, try installing the `vmd-python` library:
```bash
pip install vmd-python
```
