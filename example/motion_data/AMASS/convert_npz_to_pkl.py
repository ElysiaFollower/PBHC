"""
将AMASS的npz格式转换为pkl格式，以便使用vis_q_mj.py进行可视化

npz格式说明（来自huggingface）:
- dof_positions: [frames, 29] - 29个DOF角度
- body_positions: [frames, 30, 3] - 30个body的3D位置，第一个是pelvis
- body_rotations: [frames, 30, 4] - 30个body的旋转四元数，第一个是pelvis
- fps: 帧率

pkl格式说明:
- 外层字典：键是动作名称，值是动作数据字典
- 动作数据字典包含：
  - root_trans_offset: [frames, 3] - 根位置
  - root_rot: [frames, 4] - 根旋转四元数（xyzw格式）
  - dof: [frames, 23] - 23个DOF角度
  - pose_aa: [frames, num_joints, 3] - 姿态轴角表示
  - smpl_joints: [frames, ...] - SMPL关节位置（可选）
  - fps: 帧率
  - contact_mask: [frames, 2] - 接触掩码（可选）
"""

import numpy as np
import joblib
import pickle
from scipy.spatial.transform import Rotation as R
import argparse
from pathlib import Path
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

DOF_AXIS_FILE = "./description/robots/g1/dof_axis.npy"

# 29dof到23dof的映射（去掉6个手腕DOF）
# 29dof顺序: 左腿6 + 右腿6 + 腰部3 + 左肩4 + 左腕3 + 右肩4 + 右腕3
# 23dof顺序: 左腿6 + 右腿6 + 腰部3 + 左肩4 + 右肩4
DOF_29_TO_23_MAPPING = list(range(0, 19)) + list(range(22, 26))  # 前19个 + 跳过3个左腕 + 取4个右肩


def generate_contact_mask(body_positions, thres=0.002, height_thres=0.12):
    """
    根据脚部位置生成接触掩码
    
    Args:
        body_positions: [frames, 30, 3] body位置
        thres: 速度阈值
        height_thres: 高度阈值
    
    Returns:
        contact_mask: [frames, 2] 左右脚接触掩码
    """
    # 找到脚部body的索引（根据body_names，应该是left_ankle_roll_link和right_ankle_roll_link）
    # 从body_names来看，left_ankle_roll_link是索引6，right_ankle_roll_link是索引12
    fid_l, fid_r = 6, 12
    
    positions = body_positions
    num_frames = positions.shape[0]
    
    # 计算速度
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[1:, fid_l, 2]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < thres).astype(int) & 
              (feet_l_h < height_thres).astype(int)).astype(np.float32)
    feet_l = np.expand_dims(feet_l, axis=1)
    feet_l = np.concatenate([np.array([[1.]]), feet_l], axis=0)
    
    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[1:, fid_r, 2]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < thres).astype(int) & 
              (feet_r_h < height_thres).astype(int)).astype(np.float32)
    feet_r = np.expand_dims(feet_r, axis=1)
    feet_r = np.concatenate([np.array([[1.]]), feet_r], axis=0)
    
    contact_mask = np.concatenate([feet_l, feet_r], axis=-1)
    return contact_mask


def convert_npz_to_pkl(npz_file, output_file=None, generate_contact=True, apply_upright_start=False):
    """
    将npz文件转换为pkl格式
    
    Args:
        npz_file: npz文件路径
        output_file: 输出pkl文件路径（如果为None，则自动生成）
        generate_contact: 是否生成contact_mask
        apply_upright_start: 是否应用upright_start坐标系转换（默认False，因为AMASS数据已经是retargeted到G1的）
    
    Returns:
        输出文件路径
    """
    # 加载npz文件
    data = np.load(npz_file)
    
    # 提取数据
    dof_positions = data['dof_positions'].astype(np.float32)  # [frames, 29]
    body_positions = data['body_positions'].astype(np.float32)  # [frames, 30, 3]
    body_rotations = data['body_rotations'].astype(np.float32)  # [frames, 30, 4]
    fps = float(data['fps'][0]) if 'fps' in data else 30.0
    
    num_frames = dof_positions.shape[0]
    print(f"加载数据: {num_frames} 帧, FPS: {fps}")
    
    # 提取根位置和旋转（pelvis是第一个body）
    root_trans_offset = body_positions[:, 0, :].copy()  # [frames, 3]
    root_rot = body_rotations[:, 0, :].copy()  # [frames, 4]
    
    # 重要：根据mujoco_player.py的实现，npz文件中的body_rotations是wxyz格式
    # （mujoco_player.py第375行直接使用：data.qpos[3:7] = self.joint_positions[step, 3:7]）
    # 而vis_q_mj.py需要xyzw格式（第206行：mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]）
    # 所以需要将wxyz转换为xyzw格式
    # 转换方法：wxyz -> xyzw: [w,x,y,z] -> [x,y,z,w]
    root_rot = root_rot[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    print("已将root_rot从wxyz格式转换为xyzw格式（pkl格式要求）")
    
    # 注意：AMASS数据已经是retargeted到G1的，body_rotations应该可以直接使用
    # 不需要应用额外的坐标系转换，因为数据已经是正确的坐标系
    # 如果应用upright_start转换，会导致机器人倒过来
    # 默认不应用upright_start转换
    if apply_upright_start:
        print("警告：AMASS数据已经是retargeted的，不建议应用upright_start转换")
        print("如果机器人倒过来，请使用 --no-upright 参数")
        upright_quat = np.array([0.5, 0.5, 0.5, 0.5])  # xyzw格式
        root_rot_corrected = []
        for i in range(num_frames):
            rot_amass = R.from_quat(root_rot[i])
            rot_upright_inv = R.from_quat(upright_quat).inv()
            rot_corrected = (rot_amass * rot_upright_inv).as_quat()
            root_rot_corrected.append(rot_corrected)
        root_rot = np.array(root_rot_corrected).astype(np.float32)
        print("已应用upright_start坐标系转换（可能导致机器人倒过来）")
    else:
        print("直接使用AMASS的body_rotations（已经是retargeted到G1的）")
    
    # 从29个DOF中选择23个DOF（去掉6个手腕DOF）
    dof_23 = dof_positions[:, DOF_29_TO_23_MAPPING].copy()  # [frames, 23]
    
    print(f"DOF形状: {dof_23.shape}")
    print(f"根位置形状: {root_trans_offset.shape}")
    print(f"根旋转形状: {root_rot.shape}")
    
    # 加载dof_axis文件
    dof_axis_path = os.path.join(os.path.dirname(__file__), '../../../description/robots/g1/dof_axis.npy')
    if not os.path.exists(dof_axis_path):
        # 尝试相对路径
        dof_axis_path = DOF_AXIS_FILE
    dof_axis = np.load(dof_axis_path, allow_pickle=True).astype(np.float32)
    
    # 将根旋转四元数转换为轴角
    root_aa = R.from_quat(root_rot).as_rotvec()  # [frames, 3]
    
    # 构建pose_aa: [frames, num_joints, 3]
    # 格式: [root_aa, dof_axis * dof, zeros(3, 3)]
    pose_aa = np.concatenate(
        (np.expand_dims(root_aa, axis=1),  # [frames, 1, 3]
         dof_axis * np.expand_dims(dof_23, axis=2),  # [frames, 23, 3]
         np.zeros((num_frames, 3, 3))),  # [frames, 3, 3] 用于额外的关节
        axis=1
    ).astype(np.float32)  # [frames, 27, 3]
    
    # 生成contact_mask
    contact_mask = None
    if generate_contact:
        contact_mask = generate_contact_mask(body_positions).astype(np.float32)  # 确保是float32
        print(f"接触掩码形状: {contact_mask.shape}")
    
    # 构建数据字典
    data_dump = {
        "root_trans_offset": root_trans_offset,
        "pose_aa": pose_aa,
        "dof": dof_23,
        "root_rot": root_rot,
        "smpl_joints": np.zeros_like(pose_aa),  # 占位符
        "fps": fps,
    }
    
    if contact_mask is not None:
        data_dump["contact_mask"] = contact_mask
    
    # 构建外层字典（pkl格式要求）
    motion_name = Path(npz_file).stem
    all_data = {motion_name: data_dump}
    
    # 确定输出文件路径
    if output_file is None:
        output_file = npz_file.replace('.npz', '.pkl')
    
    # 保存pkl文件
    # 使用pickle保存以与convert_fit_motion.py的格式完全兼容
    # 注意：convert_fit_motion.py使用pickle.dump，但joblib.dump也应该可以工作
    # 为了完全兼容，我们使用pickle
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"转换完成！输出文件: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='将AMASS的npz格式转换为pkl格式')
    parser.add_argument('input', type=str, help='输入的npz文件路径或包含npz文件的目录')
    parser.add_argument('--output', type=str, default=None, help='输出pkl文件路径（如果输入是目录，则忽略此参数）')
    parser.add_argument('--no-contact', action='store_true', help='不生成contact_mask')
    parser.add_argument('--upright', action='store_true', help='应用upright_start坐标系转换（默认不应用，因为AMASS数据已经是retargeted的）')
    parser.add_argument('--recursive', action='store_true', help='递归处理目录中的所有npz文件')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 处理单个文件
        if not input_path.suffix == '.npz':
            print(f"错误: {input_path} 不是npz文件")
            return
        convert_npz_to_pkl(str(input_path), args.output, 
                          generate_contact=not args.no_contact,
                          apply_upright_start=args.upright)
    elif input_path.is_dir():
        # 处理目录
        if args.recursive:
            npz_files = list(input_path.rglob('*.npz'))
        else:
            npz_files = list(input_path.glob('*.npz'))
        
        if len(npz_files) == 0:
            print(f"在 {input_path} 中未找到npz文件")
            return
        
        print(f"找到 {len(npz_files)} 个npz文件")
        for npz_file in npz_files:
            print(f"\n处理: {npz_file}")
            try:
                convert_npz_to_pkl(str(npz_file), None, 
                                  generate_contact=not args.no_contact,
                                  apply_upright_start=args.upright)
            except Exception as e:
                print(f"错误: 处理 {npz_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"错误: {input_path} 不存在")


if __name__ == "__main__":
    main()

