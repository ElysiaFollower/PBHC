"""VMD to PKL Converter.

This script converts VMD (MikuMikuDance) motion files to the project's required
PKL format. The conversion process involves:
1. Parsing VMD file to extract bone animation data
2. Mapping MMD bones to SMPL format
3. Retargeting SMPL motion to robot DOF
4. Converting to project PKL format

Usage:
    python vmd_to_pkl.py --input vmd/input.vmd --output output.pkl
    python vmd_to_pkl.py --input vmd/input.vmd --output output.pkl --fps 30
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports (must be done before other imports)
# Get the project root (3 levels up from this file)
_project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(_project_root))
sys.path.append(str(_project_root))  # Also append for compatibility
# Change to project root directory (like other scripts do)
_original_cwd = os.getcwd()
os.chdir(_project_root)
sys.path.append(os.getcwd())  # Add current working directory to path

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from smpl_retarget.mink_retarget.convert_fit_motion import foot_detect
from smpl_retarget.mink_retarget.retargeting.mink_retarget import retarget_fit_motion
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
)
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree

# VMD bone name to SMPL joint mapping
# MMD uses Japanese bone names, this maps them to SMPL joints
VMD_TO_SMPL_BONE_MAP = {
    # Root/center bones
    "センター": "Pelvis",  # Center
    "上半身": "Spine1",  # Upper body
    "上半身2": "Spine2",  # Upper body 2
    "首": "Neck",  # Neck
    "頭": "Head",  # Head
    
    # Left arm
    "左肩": "L_Shoulder",  # Left shoulder
    "左腕": "L_UpperArm",  # Left arm
    "左ひじ": "L_ForeArm",  # Left elbow
    "左手首": "L_Hand",  # Left wrist
    
    # Right arm
    "右肩": "R_Shoulder",  # Right shoulder
    "右腕": "R_UpperArm",  # Right arm
    "右ひじ": "R_ForeArm",  # Right elbow
    "右手首": "R_Hand",  # Right wrist
    
    # Left leg
    "左足": "L_Hip",  # Left leg
    "左ひざ": "L_Knee",  # Left knee
    "左足首": "L_Ankle",  # Left ankle
    
    # Right leg
    "右足": "R_Hip",  # Right leg
    "右ひざ": "R_Knee",  # Right knee
    "右足首": "R_Ankle",  # Right ankle
}

# SMPL joint order will be obtained from SMPL_BONE_ORDER_NAMES
# This is defined in smpl_sim.smpllib.smpl_joint_names


class VMDParser:
    """Parser for VMD (MikuMikuDance) motion files.
    
    VMD files contain bone animation data in a binary format. This parser
    extracts bone rotation and position data from VMD files.
    """
    
    def __init__(self, vmd_path: str):
        """Initialize VMD parser.
        
        Args:
            vmd_path: Path to VMD file.
        """
        self.vmd_path = vmd_path
        self.bone_frames = {}
        self.max_frame = 0
        
    def parse(self) -> Dict:
        """Parse VMD file and extract motion data.
        
        Returns:
            Dictionary containing parsed motion data with keys:
            - bone_rotations: Dict mapping bone names to rotation quaternions
            - root_translation: Root bone translation over time
            - fps: Frame rate (default 30 for VMD)
        """
        try:
            # Try using vmd-python library
            import vmd
            vmd_data = vmd.load(self.vmd_path)
            
            # Extract bone frames
            bone_frames = {}
            for bone_frame in vmd_data.bones:
                bone_name = bone_frame.bone_name.decode('shift-jis', errors='ignore')
                if bone_name not in bone_frames:
                    bone_frames[bone_name] = []
                
                # Convert rotation from VMD format (quaternion) to numpy array
                # VMD uses (x, y, z, w) format
                if hasattr(bone_frame, 'rotation'):
                    rot = np.array([
                        bone_frame.rotation.x,
                        bone_frame.rotation.y,
                        bone_frame.rotation.z,
                        bone_frame.rotation.w
                    ])
                else:
                    # Fallback if rotation format is different
                    rot = np.array([0, 0, 0, 1])
                
                if hasattr(bone_frame, 'position'):
                    pos = np.array([
                        bone_frame.position.x,
                        bone_frame.position.y,
                        bone_frame.position.z
                    ])
                else:
                    pos = np.array([0, 0, 0])
                
                bone_frames[bone_name].append({
                    'frame': bone_frame.frame,
                    'rotation': rot,
                    'position': pos
                })
            
            # Sort frames by frame number
            for bone_name in bone_frames:
                bone_frames[bone_name].sort(key=lambda x: x['frame'])
                self.max_frame = max(self.max_frame, 
                                    max(f['frame'] for f in bone_frames[bone_name]))
            
            self.bone_frames = bone_frames
            
            return {
                'bone_frames': bone_frames,
                'fps': 30,  # VMD default fps
                'max_frame': self.max_frame
            }
            
        except ImportError:
            # Fallback: Use manual parsing
            return self._parse_manual()
        except Exception as e:
            print(f"Error parsing VMD with library: {e}")
            print("Attempting manual parsing...")
            return self._parse_manual()
    
    def _parse_manual(self) -> Dict:
        """Manually parse VMD file (fallback method).
        
        This is a simplified parser that handles basic VMD structure.
        For full compatibility, use the vmd-python library.
        
        Returns:
            Dictionary containing parsed motion data.
        """
        with open(self.vmd_path, 'rb') as f:
            # Read VMD header
            header = f.read(30)
            
            # Read model name (skip)
            model_name_len = int.from_bytes(f.read(4), byteorder='little')
            f.read(model_name_len)
            
            # Read bone frame count
            bone_count = int.from_bytes(f.read(4), byteorder='little')
            
            bone_frames = {}
            for _ in range(bone_count):
                # Read bone name (15 bytes, null-terminated)
                bone_name_bytes = f.read(15)
                bone_name = bone_name_bytes.split(b'\x00')[0].decode('shift-jis', errors='ignore')
                
                # Read frame number
                frame = int.from_bytes(f.read(4), byteorder='little')
                
                # Read position (3 floats)
                pos_x = np.frombuffer(f.read(4), dtype=np.float32)[0]
                pos_y = np.frombuffer(f.read(4), dtype=np.float32)[0]
                pos_z = np.frombuffer(f.read(4), dtype=np.float32)[0]
                
                # Read rotation quaternion (4 floats)
                rot_bytes = f.read(16)
                if len(rot_bytes) == 16:
                    rot_x = np.frombuffer(rot_bytes[0:4], dtype=np.float32)[0]
                    rot_y = np.frombuffer(rot_bytes[4:8], dtype=np.float32)[0]
                    rot_z = np.frombuffer(rot_bytes[8:12], dtype=np.float32)[0]
                    rot_w = np.frombuffer(rot_bytes[12:16], dtype=np.float32)[0]
                else:
                    rot_x, rot_y, rot_z, rot_w = 0.0, 0.0, 0.0, 1.0
                
                # Read interpolation data (skip 64 bytes)
                f.read(64)
                
                if bone_name not in bone_frames:
                    bone_frames[bone_name] = []
                
                bone_frames[bone_name].append({
                    'frame': frame,
                    'rotation': np.array([rot_x, rot_y, rot_z, rot_w]),
                    'position': np.array([pos_x, pos_y, pos_z])
                })
                
                self.max_frame = max(self.max_frame, frame)
            
            # Sort frames
            for bone_name in bone_frames:
                bone_frames[bone_name].sort(key=lambda x: x['frame'])
            
            self.bone_frames = bone_frames
            
            return {
                'bone_frames': bone_frames,
                'fps': 30,
                'max_frame': self.max_frame
            }


def vmd_to_smpl_poses(
    vmd_data: Dict,
    num_frames: int,
    fps: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert VMD bone data to SMPL pose format.
    
    Args:
        vmd_data: Parsed VMD data dictionary.
        num_frames: Number of frames in output.
        fps: Frame rate.
        
    Returns:
        Tuple of (pose_aa, trans):
        - pose_aa: SMPL pose in axis-angle format [num_frames, 72] (24 joints * 3)
        - trans: Root translation [num_frames, 3]
    """
    bone_frames = vmd_data['bone_frames']
    
    # Initialize SMPL pose (24 joints * 3 = 72)
    pose_aa = np.zeros((num_frames, 72))
    trans = np.zeros((num_frames, 3))
    
    # Get root bone (usually "センター" or "センター")
    root_bone_name = None
    for name in ["センター", "センター"]:
        if name in bone_frames:
            root_bone_name = name
            break
    
    if root_bone_name is None:
        print("Warning: Root bone not found, using first bone")
        root_bone_name = list(bone_frames.keys())[0] if bone_frames else None
    
    # Process each frame
    for frame_idx in range(num_frames):
        # Extract root translation
        if root_bone_name and root_bone_name in bone_frames:
            root_frames = bone_frames[root_bone_name]
            # Find closest frame
            closest_frame = min(root_frames, 
                              key=lambda x: abs(x['frame'] - frame_idx))
            trans[frame_idx] = closest_frame['position']
        
        # Extract joint rotations
        for vmd_bone_name, smpl_joint_name in VMD_TO_SMPL_BONE_MAP.items():
            if vmd_bone_name in bone_frames:
                frames = bone_frames[vmd_bone_name]
                closest_frame = min(frames, 
                                   key=lambda x: abs(x['frame'] - frame_idx))
                
                # Convert quaternion to axis-angle
                quat = closest_frame['rotation']
                # VMD uses (x, y, z, w) format, scipy uses (w, x, y, z)
                quat_scipy = np.array([quat[3], quat[0], quat[1], quat[2]])
                rot = sRot.from_quat(quat_scipy)
                axis_angle = rot.as_rotvec()
                
                # Map to SMPL joint index using SMPL_BONE_ORDER_NAMES
                if smpl_joint_name in SMPL_BONE_ORDER_NAMES:
                    joint_idx = SMPL_BONE_ORDER_NAMES.index(smpl_joint_name)
                    pose_aa[frame_idx, joint_idx * 3:(joint_idx + 1) * 3] = axis_angle
    
    return pose_aa, trans


def convert_to_pkl_format(
    pose_aa: np.ndarray,
    trans: np.ndarray,
    fps: float,
    robot_type: str = 'g1',
    humanoid_mjcf_path: Optional[str] = None
) -> Dict:
    """Convert SMPL format to project PKL format.
    
    Args:
        pose_aa: SMPL pose in axis-angle format [num_frames, 72].
        trans: Root translation [num_frames, 3].
        fps: Frame rate.
        robot_type: Robot type ('g1' or 'h1').
        humanoid_mjcf_path: Path to humanoid MJCF file.
        
    Returns:
        Dictionary in PKL format with keys:
        - root_trans_offset: Root translation
        - root_rot: Root rotation quaternion
        - dof: Robot DOF positions
        - pose_aa: Pose in axis-angle format
        - smpl_joints: SMPL joint positions (zeros)
        - fps: Frame rate
        - contact_mask: Contact mask for feet
    """
    from smpl_sim.smpllib.smpl_parser import SMPL_Parser
    
    num_frames = pose_aa.shape[0]
    
    # Convert to torch tensors
    pose_aa_tensor = torch.from_numpy(pose_aa).float()
    trans_tensor = torch.from_numpy(trans).float()
    
    # Initialize SMPL parser
    # Try multiple possible paths for SMPL model
    # Script is in motion_source/vmd2SMPL/, project root is 3 levels up
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    smpl_model_paths = [
        str(project_root / "smpl_retarget" / "smpl_model" / "smpl"),
        str(project_root / "smpl_model" / "smpl"),
        "./smpl_retarget/smpl_model/smpl",
        "../smpl_retarget/smpl_model/smpl",
        "../../smpl_retarget/smpl_model/smpl",
        "./smpl_model/smpl",
        "../smpl_model/smpl"
    ]
    smpl_model_path = None
    for path in smpl_model_paths:
        if os.path.exists(path):
            smpl_model_path = path
            break
    
    if smpl_model_path is None:
        raise FileNotFoundError(
            f"Could not find SMPL model directory. Tried: {smpl_model_paths}"
        )
    
    smpl_parser = SMPL_Parser(
        model_path=smpl_model_path,
        gender="neutral"
    )
    
    # Get SMPL joint names
    joint_names = SMPL_BONE_ORDER_NAMES
    mujoco_joint_names = SMPL_MUJOCO_NAMES
    
    # Convert pose to SMPL format
    smpl_2_mujoco = [
        joint_names.index(q) for q in mujoco_joint_names if q in joint_names
    ]
    
    # Reshape pose to [num_frames, 24, 3]
    pose_aa_24 = pose_aa.reshape(num_frames, 24, 3)[:, smpl_2_mujoco]
    
    # Convert to quaternion
    pose_quat = (
        sRot.from_rotvec(pose_aa_24.reshape(-1, 3))
        .as_quat()
        .reshape(num_frames, 24, 4)
    )
    
    # Load skeleton tree
    if humanoid_mjcf_path is None:
        # Try multiple possible paths
        # Script is in motion_source/vmd2SMPL/, project root is 3 levels up
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent.parent
        possible_paths = [
            str(project_root / "description" / "robots" / "g1" / "smpl_humanoid.xml"),
            "../description/robots/g1/smpl_humanoid.xml",
            "../../description/robots/g1/smpl_humanoid.xml",
            "./description/robots/g1/smpl_humanoid.xml"
        ]
        humanoid_mjcf_path = None
        for path in possible_paths:
            if os.path.exists(path):
                humanoid_mjcf_path = path
                break
        
        if humanoid_mjcf_path is None:
            raise FileNotFoundError(
                f"Could not find humanoid MJCF file. Tried: {possible_paths}"
            )
    
    skeleton_tree = SkeletonTree.from_mjcf(humanoid_mjcf_path)
    
    # Get SMPL joints and vertices
    betas = torch.zeros((1, 10))
    with torch.no_grad():
        verts, joints = smpl_parser.get_joints_verts(
            pose_aa_tensor, betas, trans_tensor
        )
    
    # Get root position
    root_pos = joints[:, 0]
    global_trans = joints[:, smpl_2_mujoco]
    
    # Convert to global rotation
    pose_walk_quat = (
        sRot.from_rotvec(pose_aa_24.reshape(-1, 3))
        .as_quat()
        .reshape(num_frames, 24, 4)
    )
    
    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_walk_quat),
        root_pos,
        is_local=True,
    )
    
    pose_quat_global = sk_state.global_rotation.numpy()
    
    # Retarget to robot
    print("Retargeting motion to robot...")
    skip = max(1, int(fps // 30))
    feet_l, feet_r = foot_detect(global_trans[::skip])
    contact_mask = np.concatenate([feet_l, feet_r], axis=-1)
    
    # Interpolate contact_mask to match original frame count
    if skip > 1:
        from scipy.interpolate import interp1d
        original_indices = np.arange(0, len(contact_mask)) * skip
        target_indices = np.arange(num_frames)
        contact_mask_interp = np.zeros((num_frames, 2))
        for i in range(2):
            f = interp1d(original_indices, contact_mask[:, i], 
                        kind='linear', fill_value='extrapolate')
            contact_mask_interp[:, i] = f(target_indices)
        contact_mask = contact_mask_interp
    
    # Retarget motion
    new_sk_motion = retarget_fit_motion(
        global_trans[::skip],
        pose_quat_global[::skip],
        fps=30.0,
        robot_type=robot_type,
        render=False
    )
    
    # Extract DOF positions
    dof_pos = new_sk_motion['dof_pos']
    
    # Interpolate DOF to match original frame count if needed
    if skip > 1:
        from scipy.interpolate import interp1d
        original_indices = np.arange(0, len(dof_pos)) * skip
        target_indices = np.arange(num_frames)
        dof_interp = np.zeros((num_frames, dof_pos.shape[1]))
        for i in range(dof_pos.shape[1]):
            f = interp1d(original_indices, dof_pos[:, i],
                        kind='linear', fill_value='extrapolate')
            dof_interp[:, i] = f(target_indices)
        dof_pos = dof_interp
    
    # Get root rotation and translation
    root_rot_quat = new_sk_motion['global_rotation'][:, 0, :]
    root_trans_offset = new_sk_motion['global_translation'][:, 0, :]
    
    # Interpolate root data if needed
    if skip > 1:
        from scipy.interpolate import interp1d
        original_indices = np.arange(0, len(root_rot_quat)) * skip
        target_indices = np.arange(num_frames)
        
        root_rot_interp = np.zeros((num_frames, 4))
        root_trans_interp = np.zeros((num_frames, 3))
        
        for i in range(4):
            f = interp1d(original_indices, root_rot_quat[:, i],
                        kind='linear', fill_value='extrapolate')
            root_rot_interp[:, i] = f(target_indices)
        
        for i in range(3):
            f = interp1d(original_indices, root_trans_offset[:, i],
                        kind='linear', fill_value='extrapolate')
            root_trans_interp[:, i] = f(target_indices)
        
        root_rot_quat = root_rot_interp
        root_trans_offset = root_trans_interp
    
    # Convert root rotation to axis-angle for pose_aa
    root_aa = sRot.from_quat(root_rot_quat).as_rotvec()
    
    # Load DOF axis
    # Script is in motion_source/vmd2SMPL/, project root is 3 levels up
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    dof_axis_paths = [
        str(project_root / "description" / "robots" / "g1" / "dof_axis.npy"),
        "../description/robots/g1/dof_axis.npy",
        "../../description/robots/g1/dof_axis.npy",
        "./description/robots/g1/dof_axis.npy"
    ]
    dof_axis_path = None
    for path in dof_axis_paths:
        if os.path.exists(path):
            dof_axis_path = path
            break
    
    if dof_axis_path is None:
        raise FileNotFoundError(
            f"Could not find DOF axis file. Tried: {dof_axis_paths}"
        )
    
    dof_axis = np.load(dof_axis_path, allow_pickle=True).astype(np.float32)
    
    # Create pose_aa
    pose_aa_output = np.concatenate(
        (
            np.expand_dims(root_aa, axis=1),
            dof_axis * np.expand_dims(dof_pos, axis=2),
            np.zeros((num_frames, 3, 3))
        ),
        axis=1
    ).astype(np.float32)
    
    # Create output dictionary
    output = {
        'root_trans_offset': root_trans_offset.astype(np.float32),
        'root_rot': root_rot_quat.astype(np.float32),
        'dof': dof_pos.astype(np.float32),
        'pose_aa': pose_aa_output.astype(np.float32),
        'smpl_joints': np.zeros_like(pose_aa_output).astype(np.float32),
        'fps': fps,
        'contact_mask': contact_mask.astype(np.float32)
    }
    
    return output


def main():
    """Main function to handle argument parsing and conversion."""
    parser = argparse.ArgumentParser(
        description='Convert VMD motion files to PKL format.'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input VMD file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output PKL file path'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Frame rate (default: 30.0)'
    )
    parser.add_argument(
        '--robot_type',
        type=str,
        default='g1',
        choices=['g1', 'h1'],
        help='Robot type (default: g1)'
    )
    parser.add_argument(
        '--humanoid_mjcf',
        type=str,
        default=None,
        help='Path to humanoid MJCF file'
    )
    
    args = parser.parse_args()
    
    # Parse VMD file
    print(f"Parsing VMD file: {args.input}")
    vmd_parser = VMDParser(args.input)
    vmd_data = vmd_parser.parse()
    
    num_frames = vmd_data['max_frame'] + 1
    print(f"Found {num_frames} frames")
    
    # Convert VMD to SMPL format
    print("Converting VMD to SMPL format...")
    pose_aa, trans = vmd_to_smpl_poses(vmd_data, num_frames, args.fps)
    
    # Convert to PKL format
    print("Converting to PKL format...")
    pkl_data = convert_to_pkl_format(
        pose_aa,
        trans,
        args.fps,
        robot_type=args.robot_type,
        humanoid_mjcf_path=args.humanoid_mjcf
    )
    
    # Save PKL file
    print(f"Saving PKL file: {args.output}")
    output_dict = {Path(args.input).stem: pkl_data}
    joblib.dump(output_dict, args.output)
    
    print("Conversion completed!")
    print(f"Output shape: DOF={pkl_data['dof'].shape}, "
          f"Root trans={pkl_data['root_trans_offset'].shape}")


if __name__ == '__main__':
    main()

