"""
测试脚本：比对test.npz和test.pkl的FK一致性
使用数据集提供的XML文件
"""

import numpy as np
import mujoco
import joblib
import xml.etree.ElementTree as ET
import os
import tempfile
from scipy.spatial.transform import Rotation as R

def test_fk_consistency():
    """测试AMASS原始数据（test.npz）和转换后的pkl（test.pkl）的FK一致性"""
    
    npz_file = "example/motion_data/AMASS/test/test.npz"
    pkl_file = "example/motion_data/AMASS/test/test_fixed.pkl"  # 使用修复后的pkl文件
    
    print("=" * 80)
    print("FK一致性测试：test.npz vs test.pkl")
    print("=" * 80)
    
    # 加载数据
    print("\n1. 加载数据...")
    amass_data = np.load(npz_file)
    pkl_data = joblib.load(pkl_file)
    pkl_key = list(pkl_data.keys())[0]
    pkl_motion = pkl_data[pkl_key]
    
    print(f"   AMASS帧数: {amass_data['dof_positions'].shape[0]}")
    print(f"   PKL帧数: {pkl_motion['dof'].shape[0]}")
    
    # 加载模型（使用数据集提供的XML文件）
    print("\n2. 加载MuJoCo模型...")
    print("  使用数据集提供的XML文件: example/motion_data/AMASS/robots/g1_description/g1_29dof_rev_1_0.xml")
    
    # 数据集提供的XML的meshdir是"meshes"，但mesh文件在项目的description/robots/g1/meshes目录
    # 创建一个临时XML文件，修改meshdir路径
    dataset_xml_path = "example/motion_data/AMASS/robots/g1_description/g1_29dof_rev_1_0.xml"
    tree = ET.parse(dataset_xml_path)
    root = tree.getroot()
    
    # 修改meshdir路径为绝对路径
    compiler = root.find('.//compiler')
    if compiler is not None:
        # 使用绝对路径指向mesh文件目录
        mesh_path = os.path.abspath("description/robots/g1/meshes")
        compiler.set('meshdir', mesh_path)
        print(f"  修改meshdir路径为绝对路径: {mesh_path}")
    
    # 保存临时XML文件到项目目录（而不是临时目录）
    temp_xml_path = "example/motion_data/AMASS/temp_g1_29dof.xml"
    tree.write(temp_xml_path, encoding='utf-8', xml_declaration=True)
    
    try:
        model_29 = mujoco.MjModel.from_xml_path(temp_xml_path)
        print("  ✓ 成功加载数据集提供的XML（已修正meshdir路径）")
    except Exception as e:
        print(f"  ✗ 无法加载XML: {e}")
        if os.path.exists(temp_xml_path):
            os.unlink(temp_xml_path)
        raise
    
    data_29 = mujoco.MjData(model_29)
    
    model_23 = mujoco.MjModel.from_xml_path("description/robots/g1/g1_23dof_lock_wrist.xml")
    data_23 = mujoco.MjData(model_23)
    
    # 清理临时文件
    if os.path.exists(temp_xml_path):
        os.unlink(temp_xml_path)
    
    # 关键body ID
    pelvis_29_id = model_29.body("pelvis").id
    pelvis_23_id = model_23.body("pelvis").id
    left_foot_29_id = model_29.body("left_ankle_roll_link").id
    left_foot_23_id = model_23.body("left_ankle_roll_link").id
    right_foot_29_id = model_29.body("right_ankle_roll_link").id
    right_foot_23_id = model_23.body("right_ankle_roll_link").id
    
    # 测试关键点
    key_bodies = [
        ('pelvis', pelvis_29_id, pelvis_23_id),
        ('left_ankle_roll_link', left_foot_29_id, left_foot_23_id),
        ('right_ankle_roll_link', right_foot_29_id, right_foot_23_id),
    ]
    
    # 尝试添加更多测试点（下半身关键点）
    additional_bodies = [
        ('left_knee_link', 'left_knee_link'),
        ('right_knee_link', 'right_knee_link'),
        ('left_hip_pitch_link', 'left_hip_pitch_link'),
        ('right_hip_pitch_link', 'right_hip_pitch_link'),
    ]
    
    for body_name_29, body_name_23 in additional_bodies:
        try:
            body_29_id = model_29.body(body_name_29).id
            body_23_id = model_23.body(body_name_23).id
            key_bodies.append((body_name_29, body_29_id, body_23_id))
        except:
            pass
    
    print("\n3. FK一致性检查...")
    print("-" * 80)
    
    num_frames = min(amass_data['dof_positions'].shape[0], pkl_motion['dof'].shape[0])
    
    max_pos_diffs = {}
    max_rot_diffs = {}
    
    for body_name, body_29_id, body_23_id in key_bodies:
        max_pos_diffs[body_name] = 0
        max_rot_diffs[body_name] = 0
    
    # 检查所有帧
    check_frames = list(range(num_frames))
    
    # 特别追踪右脚位姿
    right_foot_tracking = []
    
    for frame_idx in check_frames:
        # AMASS原始数据（29dof模型）
        root_pos_29 = amass_data['body_positions'][frame_idx, 0]
        root_quat_xyzw_29 = amass_data['body_rotations'][frame_idx, 0]
        root_quat_wxyz_29 = np.array([root_quat_xyzw_29[3], root_quat_xyzw_29[0], 
                                      root_quat_xyzw_29[1], root_quat_xyzw_29[2]])
        dof_29 = amass_data['dof_positions'][frame_idx]
        
        qpos_29 = np.concatenate([root_pos_29, root_quat_wxyz_29, dof_29])
        data_29.qpos[:] = qpos_29
        mujoco.mj_forward(model_29, data_29)
        
        # PKL转换后（23dof模型）
        root_pos_23 = pkl_motion['root_trans_offset'][frame_idx]
        root_quat_xyzw_23 = pkl_motion['root_rot'][frame_idx]
        root_quat_wxyz_23 = np.array([root_quat_xyzw_23[3], root_quat_xyzw_23[0],
                                      root_quat_xyzw_23[1], root_quat_xyzw_23[2]])
        dof_23 = pkl_motion['dof'][frame_idx]
        
        qpos_23 = np.concatenate([root_pos_23, root_quat_wxyz_23, dof_23])
        data_23.qpos[:] = qpos_23
        mujoco.mj_forward(model_23, data_23)
        
        # 比较关键点
        for body_name, body_29_id, body_23_id in key_bodies:
            pos_29 = data_29.xpos[body_29_id]
            pos_23 = data_23.xpos[body_23_id]
            
            pos_diff = np.linalg.norm(pos_29 - pos_23) * 1000  # mm
            
            if pos_diff > max_pos_diffs[body_name]:
                max_pos_diffs[body_name] = pos_diff
            
            # 如果是右脚，也检查旋转
            if body_name == 'right_ankle_roll_link':
                rot_29 = data_29.xquat[body_29_id]
                rot_23 = data_23.xquat[body_23_id]
                
                # 计算旋转差异（角度）
                rot_29_obj = R.from_quat(rot_29[[1, 2, 3, 0]])  # wxyz to xyzw
                rot_23_obj = R.from_quat(rot_23[[1, 2, 3, 0]])  # wxyz to xyzw
                rot_diff = (rot_29_obj.inv() * rot_23_obj).magnitude() * 180 / np.pi  # degrees
                
                if rot_diff > max_rot_diffs.get(body_name, 0):
                    max_rot_diffs[body_name] = rot_diff
                
                # 记录右脚位姿
                right_foot_tracking.append({
                    'frame': frame_idx,
                    'pos_29': pos_29.copy(),
                    'pos_23': pos_23.copy(),
                    'pos_diff': pos_diff,
                    'rot_29': rot_29.copy(),
                    'rot_23': rot_23.copy(),
                    'rot_diff': rot_diff,
                })
    
    # 输出结果
    print(f"{'Body':<25} {'最大位置差异(mm)':<20} {'最大旋转差异(deg)':<20}")
    print("-" * 80)
    
    # 关键body（必须完全一致）
    critical_bodies = ['pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link', 
                       'left_knee_link', 'right_knee_link']
    
    all_passed = True
    for body_name, body_29_id, body_23_id in key_bodies:
        max_pos_diff = max_pos_diffs[body_name]
        max_rot_diff = max_rot_diffs.get(body_name, 0)
        is_critical = body_name in critical_bodies
        
        if is_critical:
            status = "✓" if max_pos_diff < 1.0 and max_rot_diff < 1.0 else "✗"
            if max_pos_diff >= 1.0 or max_rot_diff >= 1.0:
                all_passed = False
        else:
            status = "○" if max_pos_diff < 10.0 else "⚠"  # 非关键body允许更大差异
        
        rot_str = f"{max_rot_diff:.2f}" if max_rot_diff > 0 else "N/A"
        print(f"{status} {body_name:<23} {max_pos_diff:>15.2f} {rot_str:>18}")
    
    # 详细追踪右脚位姿
    print("\n5. 右脚位姿追踪（前20帧和每20帧）...")
    print("-" * 80)
    print(f"{'帧':<6} {'位置差异(mm)':<15} {'旋转差异(deg)':<15} {'右脚位置(29dof)':<30} {'右脚位置(23dof)':<30}")
    print("-" * 80)
    
    display_frames = list(range(min(20, len(right_foot_tracking)))) + \
                     list(range(20, len(right_foot_tracking), 20))
    
    for idx in display_frames:
        if idx < len(right_foot_tracking):
            track = right_foot_tracking[idx]
            pos_29_str = f"({track['pos_29'][0]:6.3f},{track['pos_29'][1]:6.3f},{track['pos_29'][2]:6.3f})"
            pos_23_str = f"({track['pos_23'][0]:6.3f},{track['pos_23'][1]:6.3f},{track['pos_23'][2]:6.3f})"
            print(f"{track['frame']:<6} {track['pos_diff']:>12.2f}  {track['rot_diff']:>12.2f}  "
                  f"{pos_29_str:<30} {pos_23_str:<30}")
    
    # 右脚位姿统计
    if right_foot_tracking:
        pos_diffs = [t['pos_diff'] for t in right_foot_tracking]
        rot_diffs = [t['rot_diff'] for t in right_foot_tracking]
        
        print(f"\n6. 右脚位姿统计...")
        print("-" * 80)
        print(f"  位置差异: 最大={max(pos_diffs):.2f}mm, 平均={np.mean(pos_diffs):.2f}mm, "
              f"最小={min(pos_diffs):.2f}mm")
        print(f"  旋转差异: 最大={max(rot_diffs):.2f}deg, 平均={np.mean(rot_diffs):.2f}deg, "
              f"最小={min(rot_diffs):.2f}deg")
        
        # 检查是否有异常
        if max(pos_diffs) > 1.0:
            print(f"  ⚠ 警告：最大位置差异{max(pos_diffs):.2f}mm超过1mm")
            max_frame = right_foot_tracking[np.argmax(pos_diffs)]['frame']
            print(f"     最大差异出现在帧{max_frame}")
        
        if max(rot_diffs) > 1.0:
            print(f"  ⚠ 警告：最大旋转差异{max(rot_diffs):.2f}度超过1度")
            max_frame = right_foot_tracking[np.argmax(rot_diffs)]['frame']
            print(f"     最大差异出现在帧{max_frame}")
    
    # 检查root_trans和root_rot
    print("\n4. root_trans和root_rot检查...")
    print("-" * 80)
    
    frame_idx = 0
    root_pos_29 = amass_data['body_positions'][frame_idx, 0]
    root_quat_wxyz_29 = amass_data['body_rotations'][frame_idx, 0]  # wxyz格式
    root_pos_23 = pkl_motion['root_trans_offset'][frame_idx]
    root_quat_xyzw_23 = pkl_motion['root_rot'][frame_idx]  # xyzw格式
    root_quat_wxyz_23 = root_quat_xyzw_23[[3, 0, 1, 2]]  # 转换为wxyz进行比较

    root_trans_diff = np.linalg.norm(root_pos_29 - root_pos_23) * 1000
    root_rot_diff = np.linalg.norm(root_quat_wxyz_29 - root_quat_wxyz_23)
    
    print(f"  root_trans差异: {root_trans_diff:.2f} mm")
    print(f"  root_rot差异: {root_rot_diff:.6f}")
    
    if root_trans_diff < 0.1 and root_rot_diff < 0.001:
        print(f"  ✓ root_trans和root_rot一致")
    else:
        print(f"  ✗ root_trans或root_rot不一致")
        all_passed = False
    
    # 总结
    print("\n" + "=" * 80)
    print("测试结果:")
    print("=" * 80)
    
    if all_passed:
        print("✓ 所有检查通过！FK一致性验证成功。")
        print("  使用数据集提供的XML文件进行验证")
    else:
        print("✗ 发现问题，需要修复。")
    
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = test_fk_consistency()
    sys.exit(0 if success else 1)
