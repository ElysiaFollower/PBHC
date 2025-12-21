"""VMD to SMPL Converter.

This script converts VMD (MikuMikuDance) motion files to SMPL format.
The conversion process involves:
1. Parsing VMD file to extract bone animation data
2. Mapping MMD bones to SMPL format
3. Converting to SMPL pose format (axis-angle representation)

Usage:
    python vmd_to_smpl.py --input vmd/input.vmd --output output.npz
    python vmd_to_smpl.py --input vmd/input.vmd --output output.npz --fps 30
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add project root to path for imports (must be done before other imports)
# Get the project root (3 levels up from this file)
_project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(_project_root))
sys.path.append(str(_project_root))  # Also append for compatibility
# Change to project root directory (like other scripts do)
_original_cwd = os.getcwd()
os.chdir(_project_root)
sys.path.append(os.getcwd())  # Add current working directory to path

import numpy as np
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES

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
            - bone_frames: Dict mapping bone names to list of frame data
            - fps: Frame rate (default 30 for VMD)
            - max_frame: Maximum frame number
        """
        try:
            # Try using vmd library (may be vmd-python or other VMD parser)
            print("DEBUG: Attempting to import vmd library...")
            import sys
            sys.stdout.flush()  # Force flush output
            
            try:
                import vmd
                print(f"DEBUG: vmd module imported successfully from: {getattr(vmd, '__file__', 'unknown')}")
                sys.stdout.flush()
            except ImportError as e:
                print(f"DEBUG: Failed to import vmd: {e}")
                sys.stdout.flush()
                raise
            
            # Debug: Print vmd module info to understand its API
            vmd_attrs = [attr for attr in dir(vmd) if not attr.startswith('_')]
            print(f"DEBUG: vmd module found. Available attributes: {vmd_attrs[:20]}...")  # Show first 20
            print(f"DEBUG: Total attributes: {len(vmd_attrs)}")
            sys.stdout.flush()
            
            # Try different possible API methods based on common VMD parser libraries
            vmd_data = None
            
            # Method 1: VMDReader class (common in MMD VMD parsers)
            if hasattr(vmd, 'VMDReader'):
                print("Using VMDReader API")
                reader = vmd.VMDReader(self.vmd_path)
                if hasattr(reader, 'read'):
                    vmd_data = reader.read()
                elif hasattr(reader, 'parse'):
                    vmd_data = reader.parse()
                else:
                    vmd_data = reader
            # Method 2: Direct load function
            elif hasattr(vmd, 'load'):
                print("Using load() API")
                vmd_data = vmd.load(self.vmd_path)
            # Method 3: read function
            elif hasattr(vmd, 'read'):
                print("Using read() API")
                vmd_data = vmd.read(self.vmd_path)
            # Method 4: VMD class with from_file
            elif hasattr(vmd, 'VMD'):
                print("Using VMD class API")
                if hasattr(vmd.VMD, 'from_file'):
                    vmd_data = vmd.VMD.from_file(self.vmd_path)
                else:
                    vmd_data = vmd.VMD(self.vmd_path)
            # Method 5: parse function
            elif hasattr(vmd, 'parse'):
                print("Using parse() API")
                vmd_data = vmd.parse(self.vmd_path)
            else:
                # Print more info about the module
                print(f"DEBUG: vmd module type: {type(vmd)}")
                print(f"DEBUG: vmd module file: {getattr(vmd, '__file__', 'unknown')}")
                print(f"DEBUG: vmd module __name__: {getattr(vmd, '__name__', 'unknown')}")
                print(f"DEBUG: All vmd attributes: {vmd_attrs}")
                import sys
                sys.stdout.flush()
                raise AttributeError(f"vmd module does not have expected API. Available methods: {vmd_attrs}")
            
            print(f"vmd_data type: {type(vmd_data)}")
            print(f"vmd_data attributes: {[attr for attr in dir(vmd_data) if not attr.startswith('_')][:20]}")
            
            # Extract bone frames - try different data structures
            bone_frames = {}
            
            # Check how bone data is structured
            bones = None
            if hasattr(vmd_data, 'bones'):
                bones = vmd_data.bones
            elif hasattr(vmd_data, 'bone_frames'):
                bones = vmd_data.bone_frames
            elif isinstance(vmd_data, dict):
                bones = vmd_data.get('bones') or vmd_data.get('bone_frames')
            elif isinstance(vmd_data, list):
                # Maybe it's a list of bone frames directly
                bones = vmd_data
            
            if bones is None:
                raise AttributeError(f"Could not find bones in vmd_data. Type: {type(vmd_data)}, attributes: {dir(vmd_data)}")
            
            print(f"Found {len(bones)} bone frames")
            
            # Parse bone frames
            for i, bone_frame in enumerate(bones):
                try:
                    # Extract bone name
                    bone_name_raw = None
                    if hasattr(bone_frame, 'bone_name'):
                        bone_name_raw = bone_frame.bone_name
                    elif hasattr(bone_frame, 'name'):
                        bone_name_raw = bone_frame.name
                    elif isinstance(bone_frame, dict):
                        bone_name_raw = bone_frame.get('bone_name') or bone_frame.get('name')
                    
                    if bone_name_raw is None:
                        print(f"Warning: Could not extract bone name from frame {i}")
                        continue
                    
                    # Decode bone name
                    if isinstance(bone_name_raw, bytes):
                        bone_name = bone_name_raw.decode('shift-jis', errors='ignore').rstrip('\x00')
                    elif isinstance(bone_name_raw, str):
                        bone_name = bone_name_raw.rstrip('\x00')
                    else:
                        bone_name = str(bone_name_raw)
                    
                    if not bone_name:
                        continue
                    
                    if bone_name not in bone_frames:
                        bone_frames[bone_name] = []
                    
                    # Extract frame number
                    frame = 0
                    if hasattr(bone_frame, 'frame'):
                        frame = bone_frame.frame
                    elif hasattr(bone_frame, 'frame_no'):
                        frame = bone_frame.frame_no
                    elif isinstance(bone_frame, dict):
                        frame = bone_frame.get('frame') or bone_frame.get('frame_no', 0)
                    
                    # Extract rotation
                    rot = np.array([0, 0, 0, 1])  # Default quaternion
                    if hasattr(bone_frame, 'rotation'):
                        rot_obj = bone_frame.rotation
                        if hasattr(rot_obj, 'x') and hasattr(rot_obj, 'y') and hasattr(rot_obj, 'z') and hasattr(rot_obj, 'w'):
                            rot = np.array([rot_obj.x, rot_obj.y, rot_obj.z, rot_obj.w])
                        elif isinstance(rot_obj, (list, tuple, np.ndarray)) and len(rot_obj) >= 4:
                            rot = np.array(rot_obj[:4])
                    elif isinstance(bone_frame, dict) and 'rotation' in bone_frame:
                        rot = np.array(bone_frame['rotation'][:4])
                    
                    # Extract position
                    pos = np.array([0, 0, 0])
                    if hasattr(bone_frame, 'position'):
                        pos_obj = bone_frame.position
                        if hasattr(pos_obj, 'x') and hasattr(pos_obj, 'y') and hasattr(pos_obj, 'z'):
                            pos = np.array([pos_obj.x, pos_obj.y, pos_obj.z])
                        elif isinstance(pos_obj, (list, tuple, np.ndarray)) and len(pos_obj) >= 3:
                            pos = np.array(pos_obj[:3])
                    elif isinstance(bone_frame, dict) and 'position' in bone_frame:
                        pos = np.array(bone_frame['position'][:3])
                    
                    bone_frames[bone_name].append({
                        'frame': frame,
                        'rotation': rot,
                        'position': pos
                    })
                except Exception as e:
                    print(f"Warning: Error parsing bone frame {i}: {e}")
                    continue
            
            # Sort frames by frame number
            for bone_name in bone_frames:
                bone_frames[bone_name].sort(key=lambda x: x['frame'])
                if bone_frames[bone_name]:
                    self.max_frame = max(self.max_frame, 
                                        max(f['frame'] for f in bone_frames[bone_name]))
            
            self.bone_frames = bone_frames
            
            print(f"Successfully parsed {len(bone_frames)} unique bones with vmd library")
            print(f"Total frames: {self.max_frame + 1}")
            
            return {
                'bone_frames': bone_frames,
                'fps': 30,  # VMD default fps
                'max_frame': self.max_frame
            }
            
        except ImportError as e:
            # Fallback: Use manual parsing
            print(f"DEBUG: vmd library not available (ImportError: {e}), using manual parsing...")
            import traceback
            traceback.print_exc()
            import sys
            sys.stdout.flush()
            return self._parse_manual()
        except AttributeError as e:
            # vmd module exists but doesn't have expected API
            print(f"DEBUG: vmd module API mismatch: {e}")
            import traceback
            traceback.print_exc()
            import sys
            sys.stdout.flush()
            print("DEBUG: Attempting manual parsing...")
            return self._parse_manual()
        except Exception as e:
            print(f"DEBUG: Error parsing VMD with library: {e}")
            import traceback
            traceback.print_exc()
            import sys
            sys.stdout.flush()
            print("DEBUG: Attempting manual parsing...")
            return self._parse_manual()
    
    def _parse_manual(self) -> Dict:
        """Manually parse VMD file (fallback method).
        
        VMD file format (based on MMD specification):
        - Header: 30 bytes ("Vocaloid Motion Data 0002")
        - Model name: 4 bytes (length) + name (variable length, max 20 bytes)
        - Bone frame count: 4 bytes (unsigned int, little endian)
        - For each bone frame:
          - Bone name: 15 bytes (null-terminated, Shift-JIS encoding)
          - Frame number: 4 bytes (unsigned int, little endian)
          - Position: 12 bytes (3 floats, little endian)
          - Rotation: 16 bytes (4 floats as quaternion, little endian)
          - Interpolation: 64 bytes
        
        Returns:
            Dictionary containing parsed motion data.
        """
        with open(self.vmd_path, 'rb') as f:
            # Read VMD header (30 bytes: "Vocaloid Motion Data 0002")
            header = f.read(30)
            if not header.startswith(b'Vocaloid Motion Data'):
                print(f"Warning: VMD header doesn't match expected format: {header[:20]}")
                # Try to continue anyway
            
            # Read model name length (unsigned int, little endian)
            model_name_len_bytes = f.read(4)
            if len(model_name_len_bytes) < 4:
                raise ValueError("Unexpected end of file while reading model name length")
            
            # Debug: print raw bytes to understand the format
            current_debug_pos = f.tell()
            debug_bytes = f.read(50)
            f.seek(current_debug_pos)  # Go back to original position
            print(f"Debug: Bytes at offset 34-84: {debug_bytes.hex()}")
            print(f"Debug: As ASCII (where printable): {''.join(chr(b) if 32 <= b < 127 else '.' for b in debug_bytes[:30])}")
            
            model_name_len = int.from_bytes(model_name_len_bytes, byteorder='little', signed=False)
            
            # Sanity check: model name should be reasonable length (VMD spec says max 20 bytes)
            if model_name_len > 0 and model_name_len <= 256:  # Allow some flexibility
                model_name = f.read(model_name_len)
                print(f"Model name length: {model_name_len}")
            else:
                # The value is unreasonable - try to recover
                print(f"Warning: Invalid model name length: {model_name_len}")
                print("Attempting recovery: trying to find bone count by scanning...")
                
                # Save current position
                current_pos = f.tell()
                
                # Try to find bone count by scanning ahead
                # Read a chunk and look for reasonable values
                scan_chunk = f.read(200)
                f.seek(current_pos)  # Go back
                
                # Search for reasonable bone count values (should be < 100000)
                found_offset = None
                found_count = None
                
                for offset in range(0, len(scan_chunk) - 4, 4):
                    test_bytes = scan_chunk[offset:offset+4]
                    test_le = int.from_bytes(test_bytes, byteorder='little', signed=False)
                    test_be = int.from_bytes(test_bytes, byteorder='big', signed=False)
                    
                    # Look for reasonable bone count (typically 100-100000 for dance motions)
                    if 10 <= test_le <= 100000:
                        found_offset = offset
                        found_count = test_le
                        print(f"Found potential bone count at offset {offset} (LE): {test_le}")
                        break
                    elif 10 <= test_be <= 100000:
                        found_offset = offset
                        found_count = test_be
                        print(f"Found potential bone count at offset {offset} (BE): {test_be}")
                        break
                
                if found_offset is not None and found_count is not None:
                    # Calculate the actual position
                    # If model_name_len was wrong, we need to adjust
                    # Assume model name section is from offset 30 to 30+4+found_offset
                    actual_model_name_len = found_offset - 4  # Subtract the 4 bytes we read
                    if actual_model_name_len < 0:
                        actual_model_name_len = 0
                    
                    # Reposition to read bone count
                    f.seek(30 + 4 + actual_model_name_len, 0)
                    bone_count_bytes = f.read(4)
                    bone_count = found_count
                    print(f"Using recovered bone count: {bone_count}")
                else:
                    # Last resort: assume no model name and try reading bone count
                    print("Could not find bone count by scanning, assuming no model name")
                    f.seek(30 + 4, 0)  # Skip header and model_name_len (assume it's 0)
                    bone_count_bytes = f.read(4)
                    if len(bone_count_bytes) < 4:
                        raise ValueError("Unexpected end of file while reading bone count")
                    bone_count = int.from_bytes(bone_count_bytes, byteorder='little', signed=False)
                    if bone_count > 1000000:
                        raise ValueError(f"Still invalid bone count: {bone_count}. File format may be unsupported.")
            
            # If we haven't read bone_count_bytes yet, read it now
            if 'bone_count_bytes' not in locals():
                bone_count_bytes = f.read(4)
                if len(bone_count_bytes) < 4:
                    raise ValueError("Unexpected end of file while reading bone count")
                
                bone_count_le = int.from_bytes(bone_count_bytes, byteorder='little', signed=False)
                bone_count_be = int.from_bytes(bone_count_bytes, byteorder='big', signed=False)
                
                # Choose the more reasonable value
                if 0 <= bone_count_le <= 1000000:
                    bone_count = bone_count_le
                elif 0 <= bone_count_be <= 1000000:
                    bone_count = bone_count_be
                    print(f"Using big-endian byte order for bone count")
                else:
                    raise ValueError(f"Invalid bone count (LE: {bone_count_le}, BE: {bone_count_be}). File may be corrupted.")
            
            print(f"Found {bone_count} bone frames in VMD file")
            
            bone_frames = {}
            for i in range(bone_count):
                try:
                    # Read bone name (15 bytes, null-terminated)
                    bone_name_bytes = f.read(15)
                    if len(bone_name_bytes) < 15:
                        print(f"Warning: Unexpected end of file at bone frame {i+1}/{bone_count}")
                        break
                    
                    # Extract null-terminated string
                    null_pos = bone_name_bytes.find(b'\x00')
                    if null_pos >= 0:
                        bone_name_bytes = bone_name_bytes[:null_pos]
                    
                    try:
                        bone_name = bone_name_bytes.decode('shift-jis', errors='ignore')
                    except:
                        bone_name = bone_name_bytes.decode('utf-8', errors='ignore')
                    
                    # Read frame number (unsigned int, little endian)
                    frame_bytes = f.read(4)
                    if len(frame_bytes) < 4:
                        print(f"Warning: Unexpected end of file at bone frame {i+1}/{bone_count}")
                        break
                    frame = int.from_bytes(frame_bytes, byteorder='little', signed=False)
                    
                    # Sanity check: frame number should be reasonable (VMD files typically have < 10000 frames)
                    if frame > 100000:
                        print(f"Warning: Frame number {frame} seems too large at bone frame {i+1}. "
                              f"Bytes: {frame_bytes.hex()}. Skipping this frame.")
                        # Try big-endian
                        frame_be = int.from_bytes(frame_bytes, byteorder='big', signed=False)
                        if frame_be <= 100000:
                            frame = frame_be
                            print(f"  Using big-endian: {frame}")
                        else:
                            # Skip this frame if both are unreasonable
                            print(f"  Big-endian also unreasonable: {frame_be}. Skipping frame.")
                            # Skip the rest of this frame's data
                            f.read(12 + 16 + 64)  # Skip position, rotation, interpolation
                            continue
                    
                    # Read position (3 floats, little endian)
                    pos_bytes = f.read(12)
                    if len(pos_bytes) < 12:
                        print(f"Warning: Unexpected end of file at bone frame {i+1}/{bone_count}")
                        break
                    pos_array = np.frombuffer(pos_bytes, dtype=np.float32)
                    pos_x, pos_y, pos_z = pos_array[0], pos_array[1], pos_array[2]
                    
                    # Read rotation quaternion (4 floats, little endian)
                    rot_bytes = f.read(16)
                    if len(rot_bytes) < 16:
                        print(f"Warning: Unexpected end of file at bone frame {i+1}/{bone_count}")
                        break
                    rot_array = np.frombuffer(rot_bytes, dtype=np.float32)
                    rot_x, rot_y, rot_z, rot_w = rot_array[0], rot_array[1], rot_array[2], rot_array[3]
                    
                    # Read interpolation data (skip 64 bytes)
                    interp_bytes = f.read(64)
                    if len(interp_bytes) < 64:
                        print(f"Warning: Unexpected end of file at bone frame {i+1}/{bone_count}")
                        break
                    
                    if bone_name not in bone_frames:
                        bone_frames[bone_name] = []
                    
                    bone_frames[bone_name].append({
                        'frame': frame,
                        'rotation': np.array([rot_x, rot_y, rot_z, rot_w]),
                        'position': np.array([pos_x, pos_y, pos_z])
                    })
                    
                    self.max_frame = max(self.max_frame, frame)
                    
                except Exception as e:
                    print(f"Error parsing bone frame {i+1}/{bone_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Try to continue with next frame
                    continue
            
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


def main():
    """Main function to handle argument parsing and conversion."""
    parser = argparse.ArgumentParser(
        description='Convert VMD motion files to SMPL format.'
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
        help='Output SMPL file path (.npz format)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Frame rate (default: 30.0)'
    )
    
    args = parser.parse_args()
    
    # Parse VMD file
    print(f"Parsing VMD file: {args.input}")
    vmd_parser = VMDParser(args.input)
    vmd_data = vmd_parser.parse()
    
    # Calculate actual number of frames from bone frames
    # max_frame is the maximum frame number, but we need to check all frames
    all_frames = set()
    for bone_name, frames in vmd_data['bone_frames'].items():
        for frame_data in frames:
            all_frames.add(frame_data['frame'])
    
    if all_frames:
        actual_max_frame = max(all_frames)
        num_frames = actual_max_frame + 1
        print(f"Found {len(all_frames)} unique frame numbers, max frame: {actual_max_frame}, total frames: {num_frames}")
    else:
        num_frames = vmd_data['max_frame'] + 1
        print(f"Warning: No frames found, using max_frame: {num_frames}")
    
    # Sanity check: VMD files typically have reasonable frame counts
    if num_frames > 100000:
        print(f"Warning: Frame count {num_frames} seems too large. This may indicate a parsing error.")
        print(f"  max_frame from parser: {vmd_data['max_frame']}")
        print(f"  Number of bone frames: {sum(len(frames) for frames in vmd_data['bone_frames'].values())}")
        raise ValueError(f"Invalid frame count: {num_frames}. File may be corrupted or format is unsupported.")
    
    # Convert VMD to SMPL format
    print("Converting VMD to SMPL format...")
    pose_aa, trans = vmd_to_smpl_poses(vmd_data, num_frames, args.fps)
    
    # Create SMPL format data dictionary
    # According to motion_source/README.md, SMPL format should contain:
    # - poses: (frames, 66) - 22 joints * 3, but we use 72 (24 joints * 3) for consistency
    # - trans: (frames, 3) - root translation
    # - betas: (10,) - shape parameters (set to zeros)
    # - gender: str - gender (default to 'neutral')
    # - mocap_framerate: int - fps
    smpl_data = {
        'pose_aa': pose_aa.astype(np.float32),  # [num_frames, 72]
        'poses': pose_aa[:, :66].astype(np.float32),  # [num_frames, 66] for compatibility
        'trans': trans.astype(np.float32),  # [num_frames, 3]
        'betas': np.zeros(10, dtype=np.float32),  # Default shape parameters
        'gender': 'neutral',  # Default gender
        'fps': args.fps,
        'mocap_framerate': int(args.fps)
    }
    
    # Save SMPL file
    print(f"Saving SMPL file: {args.output}")
    np.savez_compressed(args.output, **smpl_data)
    
    print("Conversion completed!")
    print(f"Output shape: pose_aa={pose_aa.shape}, trans={trans.shape}")
    print(f"Frame rate: {args.fps} fps")


if __name__ == '__main__':
    main()

