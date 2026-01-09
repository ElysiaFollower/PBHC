"""
在动作数据尾帧添加GoHome复位数据

功能：
1. 读取input文件夹中的pkl文件
2. 在动作数据尾帧添加数据，使得机器人在3s内GoHome复位
3. 输出到output文件夹
"""

import pickle
import joblib
import torch
import numpy as np
from pathlib import Path
import os
import pathlib
from scipy.spatial.transform import Rotation as R, Slerp

if os.name == 'nt':  # Windows系统
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# ==================== 配置 ====================
# GoHome关节角度（按照XML中actuator的顺序，共23个关节）
GOHOME_JOINT_ANGLES = np.array([
    -0.1,   # left_hip_pitch_joint
    0.0,    # left_hip_roll_joint
    0.0,    # left_hip_yaw_joint
    0.3,    # left_knee_joint
    -0.2,   # left_ankle_pitch_joint
    0.0,    # left_ankle_roll_joint
    -0.1,   # right_hip_pitch_joint
    0.0,    # right_hip_roll_joint
    0.0,    # right_hip_yaw_joint
    0.3,    # right_knee_joint
    -0.2,   # right_ankle_pitch_joint
    0.0,    # right_ankle_roll_joint
    0.0,    # waist_yaw_joint
    0.0,    # waist_roll_joint
    0.0,    # waist_pitch_joint
    0.2,    # left_shoulder_pitch_joint
    0.2,    # left_shoulder_roll_joint
    0.0,    # left_shoulder_yaw_joint
    0.9,    # left_elbow_joint
    0.2,    # right_shoulder_pitch_joint
    -0.2,   # right_shoulder_roll_joint
    0.0,    # right_shoulder_yaw_joint
    0.9,    # right_elbow_joint
], dtype=np.float32)

# GoHome持续时间（秒）
GOHOME_DURATION = 1.5

# 输出文件后缀
OUTPUT_SUFFIX = "_gohome"

# 输入和输出文件夹路径
INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"

# dof_axis文件路径（相对于项目根目录）
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.parent.resolve()
DOF_AXIS_FILE = project_root / "description" / "robots" / "g1" / "dof_axis.npy"
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


def interpolate_linear(start, end, num_frames):
    """线性插值"""
    t = np.linspace(0, 1, num_frames)
    return start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]


def interpolate_smooth(start, end, num_frames):
    """平滑插值（使用sigmoid函数）"""
    if num_frames <= 1:
        return end[None, :]
    
    t = np.linspace(0, 1, num_frames)
    # 使用sigmoid函数实现平滑过渡
    smooth_t = 1 / (1 + np.exp(-10 * (t - 0.5)))
    smooth_t = (smooth_t - smooth_t[0]) / (smooth_t[-1] - smooth_t[0] + 1e-8)  # 归一化到[0,1]，避免除以0
    return start[None, :] * (1 - smooth_t[:, None]) + end[None, :] * smooth_t[:, None]


def compute_pose_aa(root_rot, dof, dof_axis):
    """
    根据root_rot和dof计算pose_aa
    
    Args:
        root_rot: (N, 4) 四元数 [x, y, z, w] 或 [w, x, y, z]
        dof: (N, 23) 关节角度
        dof_axis: (23, 3) 关节轴向量
    
    Returns:
        pose_aa: (N, 27, 3) 轴角表示，包含root(1) + joints(23) + zeros(3)
    """
    # 将root_rot转换为轴角
    # 注意：需要确认root_rot的格式是[x,y,z,w]还是[w,x,y,z]
    # 从探索结果看，root_rot是(3060, 4)，示例值是[0.02568594, 0.02049165, -0.6951257, 0.71813685]
    # 这看起来像是[x, y, z, w]格式
    root_aa = R.from_quat(root_rot).as_rotvec()  # (N, 3)
    
    # 计算关节的轴角：dof_axis * dof
    # dof_axis: (23, 3), dof: (N, 23) -> 需要broadcast
    joint_aa = dof_axis[None, :, :] * dof[:, :, None]  # (N, 23, 3)
    
    # 创建零填充 (N, 3, 3)
    zeros = np.zeros((root_aa.shape[0], 3, 3), dtype=np.float32)
    
    # 拼接：root(1) + joints(23) + zeros(3) = 27
    pose_aa = np.concatenate([
        root_aa[:, None, :],  # (N, 1, 3)
        joint_aa,              # (N, 23, 3)
        zeros                  # (N, 3, 3)
    ], axis=1).astype(np.float32)
    
    return pose_aa


def add_gohome_motion(motion, fps=30):
    """在动作数据末尾添加GoHome复位数据
    
    简单实现：对每个关节角进行插值，最后重新计算pose_aa
    """
    # 获取当前动作的帧数
    length = get_motion_length(motion)
    if length is None:
        raise ValueError("无法确定动作的帧数")
    
    # 获取fps（如果存在）
    if "fps" in motion:
        fps_val = motion["fps"]
        if isinstance(fps_val, np.ndarray):
            fps = int(fps_val.item())
        else:
            fps = int(fps_val)
    else:
        fps = fps
    
    # 计算需要添加的帧数
    num_gohome_frames = int(GOHOME_DURATION * fps)
    print(f"  调试: fps={fps}, GOHOME_DURATION={GOHOME_DURATION}, num_gohome_frames={num_gohome_frames}")
    
    # 获取最后一帧的dof数据
    if "dof" not in motion:
        raise ValueError("动作数据中没有找到'dof'字段")
    
    dof_data = motion["dof"]
    if torch.is_tensor(dof_data):
        last_frame_dof = dof_data[-1].cpu().numpy()
    else:
        last_frame_dof = dof_data[-1].copy()
    
    # 确保是一维数组
    if len(last_frame_dof.shape) == 0:
        last_frame_dof = last_frame_dof.reshape(1)
    elif len(last_frame_dof.shape) > 1:
        last_frame_dof = last_frame_dof.flatten()
    
    if last_frame_dof.shape[0] != len(GOHOME_JOINT_ANGLES):
        raise ValueError(f"关节数量不匹配: 期望{len(GOHOME_JOINT_ANGLES)}, 实际{last_frame_dof.shape[0]}")
    
    # 简单插值：对每个关节角从最后一帧的值插值到GoHome值
    gohome_dof = interpolate_smooth(last_frame_dof, GOHOME_JOINT_ANGLES, num_gohome_frames)
    
    print(f"  调试信息: 原始帧数={length}, 需要添加={num_gohome_frames}帧, gohome_dof.shape={gohome_dof.shape}")
    
    # 创建新的motion数据
    new_motion = {}
    
    for key, value in motion.items():
        if key == "dof":
            # 追加dof数据
            if torch.is_tensor(value):
                new_dof = torch.cat([value, torch.from_numpy(gohome_dof).to(value.device)], dim=0)
                new_motion[key] = new_dof
                print(f"  调试信息: dof追加后 shape={new_dof.shape}")
            else:
                new_dof = np.concatenate([value, gohome_dof], axis=0)
                new_motion[key] = new_dof
                print(f"  调试信息: dof追加后 shape={new_dof.shape}")
        elif key == "root_trans_offset":
            # GoHome时：保持最后一帧的x,y位置，但z坐标插值到标准高度0.8
            last_root_trans = value[-1:].copy() if isinstance(value, np.ndarray) else value[-1:].clone()
            if torch.is_tensor(value):
                last_root_trans_np = last_root_trans.cpu().numpy()
            else:
                last_root_trans_np = last_root_trans
            
            # 目标位置：保持x,y，z设为0.8（标准站立高度）
            target_root_trans = last_root_trans_np.copy()
            target_root_trans[0, 2] = 0.8  # z坐标设为0.8
            
            # 插值root_trans_offset
            gohome_root_trans = interpolate_smooth(last_root_trans_np[0], target_root_trans[0], num_gohome_frames)
            
            if torch.is_tensor(value):
                gohome_root_trans_tensor = torch.from_numpy(gohome_root_trans).to(value.device)
                new_motion[key] = torch.cat([value, gohome_root_trans_tensor], dim=0)
            else:
                new_motion[key] = np.concatenate([value, gohome_root_trans], axis=0)
        elif key == "root_rot":
            # GoHome时：保持yaw（朝向）不变，只重置pitch和roll到0（直立状态）
            last_root_rot = value[-1:].copy() if isinstance(value, np.ndarray) else value[-1:].clone()
            if torch.is_tensor(value):
                last_root_rot_np = last_root_rot.cpu().numpy()
            else:
                last_root_rot_np = last_root_rot
            
            # 从最后一帧的旋转中提取yaw角度
            last_rot = R.from_quat(last_root_rot_np[0])  # [x,y,z,w]格式
            euler = last_rot.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
            yaw = euler[2]  # 提取yaw角度
            
            # 创建目标旋转：只有yaw旋转，pitch=0, roll=0（直立但保持朝向）
            target_rot = R.from_euler('xyz', [0.0, 0.0, yaw], degrees=False)
            target_root_rot = target_rot.as_quat().astype(np.float32)  # [x,y,z,w]格式
            
            # 使用四元数球面插值（slerp）从当前旋转插值到目标旋转
            rotations = R.from_quat([last_root_rot_np[0], target_root_rot])
            slerp = Slerp([0, 1], rotations)
            times = np.linspace(0, 1, num_gohome_frames)
            interpolated_rots = slerp(times)
            gohome_root_rot = interpolated_rots.as_quat().astype(np.float32)  # [x,y,z,w]格式
            
            if torch.is_tensor(value):
                gohome_root_rot_tensor = torch.from_numpy(gohome_root_rot).to(value.device)
                new_motion[key] = torch.cat([value, gohome_root_rot_tensor], dim=0)
            else:
                new_motion[key] = np.concatenate([value, gohome_root_rot], axis=0)
        elif key == "pose_aa":
            # 删除pose_aa，让系统自动从dof和root_rot计算
            # 这样就不需要手动更新pose_aa了
            pass  # 不添加pose_aa，系统会自动计算
        elif key == "contact_mask":
            # 保持最后一帧的接触状态（GoHome时通常双脚接触地面）
            last_contact = value[-1:].copy() if isinstance(value, np.ndarray) else value[-1:].clone()
            if torch.is_tensor(value):
                repeated_contact = last_contact.repeat(num_gohome_frames, 1)
                new_motion[key] = torch.cat([value, repeated_contact], dim=0)
            else:
                repeated_contact = np.repeat(last_contact, num_gohome_frames, axis=0)
                new_motion[key] = np.concatenate([value, repeated_contact], axis=0)
        elif key == "smpl_joints":
            # 保持最后一帧的SMPL关节
            last_smpl = value[-1:].copy() if isinstance(value, np.ndarray) else value[-1:].clone()
            if torch.is_tensor(value):
                repeated_smpl = last_smpl.repeat(num_gohome_frames, 1, 1)
                new_motion[key] = torch.cat([value, repeated_smpl], dim=0)
            else:
                repeated_smpl = np.repeat(last_smpl, num_gohome_frames, axis=0)
                new_motion[key] = np.concatenate([value, repeated_smpl], axis=0)
        elif key == "fps":
            # 保持fps不变
            new_motion[key] = value
        else:
            # 其他字段，如果是数组则重复最后一帧，否则直接复制
            if isinstance(value, (np.ndarray, torch.Tensor)):
                if len(value.shape) > 0 and value.shape[0] == length:
                    last_val = value[-1:].copy() if isinstance(value, np.ndarray) else value[-1:].clone()
                    if torch.is_tensor(value):
                        repeated_val = last_val.repeat(num_gohome_frames, *([1] * (len(value.shape) - 1)))
                        new_motion[key] = torch.cat([value, repeated_val], dim=0)
                    else:
                        repeated_val = np.repeat(last_val, num_gohome_frames, axis=0)
                        new_motion[key] = np.concatenate([value, repeated_val], axis=0)
                else:
                    new_motion[key] = value
            else:
                new_motion[key] = value
    
    # 重新计算pose_aa（在所有数据都更新后）
    # 注意：MotionLib会直接使用pose_aa，所以必须更新它以保证一致性
    if "pose_aa" in motion:
        # 加载dof_axis
        if not DOF_AXIS_FILE.exists():
            raise FileNotFoundError(f"dof_axis文件不存在: {DOF_AXIS_FILE}")
        dof_axis = np.load(DOF_AXIS_FILE, allow_pickle=True).astype(np.float32)
        
        # 获取更新后的root_rot和dof
        new_root_rot = new_motion["root_rot"]
        new_dof = new_motion["dof"]
        
        # 转换为numpy（如果是torch tensor）
        if torch.is_tensor(new_root_rot):
            new_root_rot_np = new_root_rot.cpu().numpy()
        else:
            new_root_rot_np = new_root_rot
        
        if torch.is_tensor(new_dof):
            new_dof_np = new_dof.cpu().numpy()
        else:
            new_dof_np = new_dof
        
        # 计算pose_aa
        new_pose_aa = compute_pose_aa(new_root_rot_np, new_dof_np, dof_axis)
        
        # 转换回原始格式
        if torch.is_tensor(motion["pose_aa"]):
            new_motion["pose_aa"] = torch.from_numpy(new_pose_aa).to(motion["pose_aa"].device)
        else:
            new_motion["pose_aa"] = new_pose_aa
    
    return new_motion


def process_file(input_path, output_path):
    """处理单个pkl文件"""
    print(f"\n处理文件: {input_path.name}")
    
    # 加载数据
    data = load_pkl(input_path)
    
    # 判断文件结构：可能是单个动作字典，也可能是包含多个动作的字典
    if isinstance(data, dict):
        # 检查是否是包含多个动作的字典（每个值都是字典且包含'dof'键）
        if all(isinstance(v, dict) and "dof" in v for v in data.values()):
            print(f"检测到包含 {len(data)} 个动作的文件")
            print(f"动作键: {list(data.keys())}")
            
            # 处理每个动作
            for key, motion in data.items():
                print(f"\n处理动作: {key}")
                original_length = get_motion_length(motion)
                print(f"原始帧数: {original_length}")
                
                try:
                    new_motion = add_gohome_motion(motion)
                    new_length = get_motion_length(new_motion)
                    print(f"添加GoHome后帧数: {new_length} (增加了 {new_length - original_length} 帧)")
                    data[key] = new_motion
                except Exception as e:
                    print(f"处理动作 '{key}' 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        else:
            # 单个动作字典
            print("检测到单个动作文件")
            original_length = get_motion_length(data)
            print(f"原始帧数: {original_length}")
            
            try:
                data = add_gohome_motion(data)
                new_length = get_motion_length(data)
                print(f"添加GoHome后帧数: {new_length} (增加了 {new_length - original_length} 帧)")
            except Exception as e:
                print(f"处理文件时出错: {e}")
                import traceback
                traceback.print_exc()
                return False
    else:
        print(f"警告: 文件格式不符合预期，类型为 {type(data)}")
        return False
    
    # 保存处理后的数据
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        joblib.dump(data, output_path)
        print(f"\n处理后的文件已保存到: {output_path}")
        return True
    except Exception as e:
        try:
            with open(output_path, "wb") as f:
                pickle.dump(data, f)
            print(f"\n处理后的文件已保存到: {output_path} (使用pickle)")
            return True
        except Exception as e2:
            print(f"\n保存文件失败: {e}")
            print(f"尝试pickle也失败: {e2}")
            return False


def main():
    """主函数"""
    print("=" * 60)
    print("GoHome动作数据添加脚本")
    print("=" * 60)
    
    # 检查输入文件夹
    if not INPUT_DIR.exists():
        print(f"错误: 输入文件夹不存在: {INPUT_DIR}")
        return
    
    # 创建输出文件夹
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 查找所有pkl文件
    pkl_files = list(INPUT_DIR.glob("*.pkl"))
    
    if len(pkl_files) == 0:
        print(f"警告: 在 {INPUT_DIR} 中没有找到pkl文件")
        return
    
    print(f"\n找到 {len(pkl_files)} 个pkl文件")
    
    # 处理每个文件
    success_count = 0
    for pkl_file in pkl_files:
        # 添加后缀到输出文件名
        output_name = pkl_file.stem + OUTPUT_SUFFIX + pkl_file.suffix
        output_file = OUTPUT_DIR / output_name
        if process_file(pkl_file, output_file):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"处理完成: {success_count}/{len(pkl_files)} 个文件成功处理")
    print("=" * 60)


if __name__ == "__main__":
    main()

