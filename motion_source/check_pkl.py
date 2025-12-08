# check_pkl.py
import pickle
import sys

# 你的文件路径
file_path = "example/motion_data/merged_motion.pkl"

try:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        
    print(f"成功加载文件，共包含 {len(data)} 个动作：")
    print("-" * 30)
    for key in data.keys():
        # 打印动作名称和帧数
        frames = data[key]['dof'].shape[0] if 'dof' in data[key] else "未知"
        print(f"动作名 (Key): {key:<30} | 帧数: {frames}")
    print("-" * 30)
    
except Exception as e:
    print(f"读取失败: {e}")