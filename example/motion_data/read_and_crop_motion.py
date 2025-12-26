"""
简单的动作文件读取和裁剪脚本

使用方法：
1. 修改脚本开头的全局变量（文件路径和裁剪范围）
2. 运行脚本：python read_and_crop_motion.py
"""

import pickle
import joblib
import torch
import numpy as np
from pathlib import Path
import os
import pathlib
if os.name == 'nt':  # 只有在 Windows 下才执行
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# ==================== 全局配置 ====================
# 要读取的pkl文件路径（相对于项目根目录的路径）
PKL_FILE_PATH = "example/motion_data/rin-dancehall.pkl"

# 裁剪范围（帧索引，从0开始）
# 如果 CROP_START 为 None 或 -1，则从第0帧开始
# 如果 CROP_END 为 None 或 -1，则裁剪到最后一帧
CROP_START = 330  # 例如: 10 表示从第10帧开始
CROP_END = 4980    # 例如: 100 表示裁剪到第100帧（不包含）

# 是否保存裁剪后的文件
SAVE_CROPPED = True

# 裁剪后文件的保存路径（如果为None，则在原文件同目录下生成新文件）
OUTPUT_FILE_PATH = None  # None 表示在原文件同目录下生成新文件
# ================================================


def load_pkl(pkl_path):
    """加载pkl文件，支持pickle、joblib和torch三种格式"""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"文件不存在: {pkl_path}")
    
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e1:
        try:
            print(f"Pickle加载失败，尝试使用joblib加载: {pkl_path}")
            return joblib.load(pkl_path)
        except Exception as e2:
            try:
                print(f"Joblib加载失败，尝试使用torch.load加载: {pkl_path}")
                return torch.load(pkl_path, map_location="cpu")
            except Exception as e3:
                raise RuntimeError(
                    f"无法加载 {pkl_path}，已尝试pickle、joblib和torch三种方式。\n"
                    f"错误信息:\n- pickle: {e1}\n- joblib: {e2}\n- torch: {e3}"
                )


def get_motion_length(motion):
    """获取动作的帧数"""
    if "dof" in motion:
        return motion["dof"].shape[0]
    elif "root_trans_offset" in motion:
        return motion["root_trans_offset"].shape[0]
    elif "pose_aa" in motion:
        return motion["pose_aa"].shape[0]
    else:
        return None


def crop_motion(motion, start_frame, end_frame):
    """裁剪动作数据"""
    cropped = {}
    
    for key, value in motion.items():
        if isinstance(value, np.ndarray):
            # numpy数组，按第一维裁剪
            if len(value.shape) > 0:
                cropped[key] = value[start_frame:end_frame].copy()
            else:
                cropped[key] = value
        elif isinstance(value, (list, tuple)):
            # 列表或元组，按索引裁剪
            cropped[key] = value[start_frame:end_frame]
        elif torch.is_tensor(value):
            # torch张量，按第一维裁剪
            if len(value.shape) > 0:
                cropped[key] = value[start_frame:end_frame].clone()
            else:
                cropped[key] = value
        else:
            # 其他类型（如fps、字符串等），直接复制
            cropped[key] = value
    
    return cropped


def print_motion_info(motion, motion_name="动作"):
    """打印动作信息"""
    print(f"\n{'='*60}")
    print(f"{motion_name} 信息:")
    print(f"{'='*60}")
    
    length = get_motion_length(motion)
    if length is not None:
        print(f"总帧数: {length}")
    
    if "fps" in motion:
        fps = motion["fps"]
        duration = length / fps if length else 0
        print(f"帧率: {fps} fps")
        print(f"时长: {duration:.2f} 秒")
    
    print(f"\n包含的键:")
    for key, value in motion.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            if torch.is_tensor(value):
                shape = tuple(value.shape)
                dtype = str(value.dtype)
            else:
                shape = value.shape
                dtype = str(value.dtype)
            print(f"  - {key}: shape={shape}, dtype={dtype}")
        elif isinstance(value, (list, tuple)):
            print(f"  - {key}: list/tuple, length={len(value)}")
        else:
            print(f"  - {key}: {type(value).__name__} = {value}")


def main():
    # 获取项目根目录（脚本所在目录的父目录的父目录，因为脚本在 example/motion_data/ 下）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # 解析文件路径：如果是相对路径，则相对于项目根目录
    if Path(PKL_FILE_PATH).is_absolute():
        pkl_path = Path(PKL_FILE_PATH)
    else:
        pkl_path = project_root / PKL_FILE_PATH
    
    print(f"项目根目录: {project_root}")
    print(f"正在读取文件: {pkl_path}")
    print(f"文件是否存在: {pkl_path.exists()}")
    
    # 加载pkl文件
    data = load_pkl(pkl_path)
    
    # 用于保存文件名的裁剪范围
    crop_start = None
    crop_end = None
    
    # 判断文件结构：可能是单个动作字典，也可能是包含多个动作的字典
    if isinstance(data, dict):
        # 检查是否是包含多个动作的字典（每个值都是字典且包含'dof'键）
        if all(isinstance(v, dict) and "dof" in v for v in data.values()):
            print(f"\n检测到包含 {len(data)} 个动作的文件")
            print(f"动作键: {list(data.keys())}")
            
            # 处理每个动作
            for key, motion in data.items():
                print_motion_info(motion, f"动作 '{key}'")
                
                length = get_motion_length(motion)
                if length is None:
                    print(f"  警告: 无法确定动作 '{key}' 的帧数，跳过裁剪")
                    continue
                
                # 确定裁剪范围
                start = CROP_START if CROP_START is not None and CROP_START >= 0 else 0
                end = CROP_END if CROP_END is not None and CROP_END >= 0 else length
                
                # 验证裁剪范围
                if start >= length:
                    print(f"  警告: 起始帧 {start} 超出范围 [0, {length})，跳过裁剪")
                    continue
                if end > length:
                    print(f"  警告: 结束帧 {end} 超出范围，调整为 {length}")
                    end = length
                if start >= end:
                    print(f"  警告: 起始帧 {start} >= 结束帧 {end}，跳过裁剪")
                    continue
                
                # 记录第一个成功裁剪的范围（用于文件名）
                if crop_start is None:
                    crop_start = start
                    crop_end = end
                
                print(f"\n裁剪范围: [{start}, {end}) (共 {end - start} 帧)")
                
                # 执行裁剪
                cropped_motion = crop_motion(motion, start, end)
                data[key] = cropped_motion
                
                print(f"裁剪完成: {length} 帧 -> {end - start} 帧")
        else:
            # 单个动作字典
            print("\n检测到单个动作文件")
            motion = data
            print_motion_info(motion, "动作")
            
            length = get_motion_length(motion)
            if length is None:
                print("警告: 无法确定动作的帧数，跳过裁剪")
                return
            
            # 确定裁剪范围
            start = CROP_START if CROP_START is not None and CROP_START >= 0 else 0
            end = CROP_END if CROP_END is not None and CROP_END >= 0 else length
            
            # 验证裁剪范围
            if start >= length:
                print(f"警告: 起始帧 {start} 超出范围 [0, {length})，跳过裁剪")
                return
            if end > length:
                print(f"警告: 结束帧 {end} 超出范围，调整为 {length}")
                end = length
            if start >= end:
                print(f"警告: 起始帧 {start} >= 结束帧 {end}，跳过裁剪")
                return
            
            # 记录裁剪范围（用于文件名）
            crop_start = start
            crop_end = end
            
            print(f"\n裁剪范围: [{start}, {end}) (共 {end - start} 帧)")
            
            # 执行裁剪
            data = crop_motion(motion, start, end)
            
            print(f"裁剪完成: {length} 帧 -> {end - start} 帧")
            print_motion_info(data, "裁剪后的动作")
    else:
        print(f"警告: 文件格式不符合预期，类型为 {type(data)}")
        return
    
    # 保存裁剪后的文件
    if SAVE_CROPPED:
        if OUTPUT_FILE_PATH is not None:
            # 如果指定了输出路径，解析为相对于项目根目录的路径
            if Path(OUTPUT_FILE_PATH).is_absolute():
                output_path = Path(OUTPUT_FILE_PATH)
            else:
                output_path = project_root / OUTPUT_FILE_PATH
        else:
            # 在原文件同目录下生成新文件，使用 start-end 格式
            if crop_start is not None and crop_end is not None:
                output_path = pkl_path.parent / f"{pkl_path.stem}_{crop_start}-{crop_end}.pkl"
            else:
                # 如果没有裁剪（不应该发生），使用默认名称
                output_path = pkl_path.parent / f"{pkl_path.stem}_cropped.pkl"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 尝试使用joblib保存（更常用）
        try:
            joblib.dump(data, output_path)
            print(f"\n裁剪后的文件已保存到: {output_path}")
        except Exception as e:
            try:
                # 如果joblib失败，尝试pickle
                with open(output_path, "wb") as f:
                    pickle.dump(data, f)
                print(f"\n裁剪后的文件已保存到: {output_path} (使用pickle)")
            except Exception as e2:
                print(f"\n保存文件失败: {e}")
                print(f"尝试pickle也失败: {e2}")
    else:
        print("\n未启用保存功能（SAVE_CROPPED = False）")


if __name__ == "__main__":
    main()

