# check_pkl.py
import pickle
import sys
import joblib
import torch
import pathlib
import os
import numpy as np

# Windows 路径兼容性处理
if os.name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

def load_pkl(pkl_path):
    """加载pkl文件，支持pickle、joblib和torch三种格式"""
    # 确保路径是字符串格式
    pkl_path_str = str(pkl_path)
    try:
        with open(pkl_path_str, "rb") as f:
            return pickle.load(f)
    except Exception as e1:
        try:
            print(f"Pickle加载失败，尝试使用joblib加载: {pkl_path_str}")
            return joblib.load(pkl_path_str)
        except Exception as e2:
            try:
                print(f"Joblib加载失败，尝试使用torch.load加载: {pkl_path_str}")
                return torch.load(pkl_path_str, map_location="cpu")
            except Exception as e3:
                raise RuntimeError(f"无法加载 {pkl_path_str}（已尝试pickle、joblib和torch）\n错误信息:\n- pickle: {e1}\n- joblib: {e2}\n- torch: {e3}")


def print_array_info(arr, name, max_items=10):
    """打印numpy数组的信息"""
    print(f"  [{name}]")
    print(f"    形状: {arr.shape}, 数据类型: {arr.dtype}")
    
    if arr.size == 0:
        print(f"    空数组")
        return
    
    # 标量
    if arr.ndim == 0:
        print(f"    值: {arr.item()}")
        return
    
    # 一维数组
    if arr.ndim == 1:
        print(f"    前{min(max_items, len(arr))}个值: {arr[:max_items]}")
        if len(arr) > max_items:
            print(f"    ... (共{len(arr)}个值)")
    
    # 二维数组
    elif arr.ndim == 2:
        print(f"    第一行前{min(max_items, arr.shape[1])}个值: {arr[0, :max_items]}")
        print(f"    第二行前{min(max_items, arr.shape[1])}个值: {arr[1, :max_items]}")
        print(f"    值范围: min={arr.min():.6f}, max={arr.max():.6f}")
    
    # 三维数组
    elif arr.ndim == 3:
        print(f"    第一帧第一行: {arr[0, 0, :]}")
        print(f"    值范围: min={arr.min():.6f}, max={arr.max():.6f}")
    
    # 特殊处理 dof
    if name == 'dof':
        print(f"    【DOF分析】关节角度值，{arr.shape[1]}个关节，{arr.shape[0]}帧")
        print(f"    平均值: {arr.mean():.6f}, 标准差: {arr.std():.6f}")

# 文件路径
file_path = "example/motion_data/gangster.pkl"

# PKL文件结构定义
MOTION_KEYS = [
    'root_trans_offset',  # 根节点平移 (帧数, 3)
    'root_rot',            # 根节点旋转四元数 (帧数, 4)
    'dof',                 # 关节角度 (帧数, 23)
    'fps',                 # 帧率 (标量)
    'contact_mask',        # 接触掩码 (帧数, 2)
    'pose_aa',             # 轴角姿态 (帧数, 27, 3)
]

try:
    data = load_pkl(file_path)
    
    print(f"成功加载文件，共包含 {len(data)} 个动作：")
    print("=" * 80)
    
    for key in data.keys():
        key_str = str(key)
        print(f"\n动作名 (Key): {key_str}")
        print("-" * 80)
        
        motion = data[key]
        
        # 先打印所有键名
        print(f"\n包含的键: {list(motion.keys())}")
        print(f"\n各键的详细信息:")
        print("-" * 80)
        
        # 按定义的顺序打印
        for key_name in MOTION_KEYS:
            if key_name in motion:
                print_array_info(motion[key_name], key_name)
        
        # 打印其他未定义的键
        for key_name in motion.keys():
            if key_name not in MOTION_KEYS:
                print_array_info(motion[key_name], key_name)
        
        print("=" * 80)
    
except Exception as e:
    print(f"读取失败: {e}")
    import traceback
    traceback.print_exc()