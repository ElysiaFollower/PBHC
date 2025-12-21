"""Example usage of VMD to SMPL converter.

This script demonstrates how to use the VMD to SMPL converter.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run example conversion."""
    # Get paths
    script_dir = Path(__file__).parent  # motion_source/vmd2SMPL
    project_root = script_dir.parent.parent  # project root (PBHC)
    
    # Example: Convert a VMD file to SMPL
    # Use quotes around filename to handle parentheses in PowerShell
    vmd_file = r"motion_source/vmd2SMPL/vmd/愛包ダンスホール_DanceMotion_HimeTanaka(MMD)_v1.1.vmd"
    output_file = "motion_source/vmd2SMPL/output_dance.npz"
    
    vmd_path = project_root / vmd_file
    
    if not vmd_path.exists():
        print(f"Error: VMD file not found: {vmd_path}")
        print("Please ensure the VMD file exists in the vmd/ directory")
        return
    
    # Use absolute paths and quote the input file to handle special characters
    cmd = [
        sys.executable,
        str(script_dir / "vmd_to_smpl.py"),
        "--input", str(vmd_path),
        "--output", str(project_root / output_file),
        "--fps", "30"
    ]
    
    print("Running VMD to SMPL conversion:")
    print(f"Working directory: {project_root}")
    print(" ".join(cmd))
    print()
    
    try:
        # Run from project root so relative paths work correctly
        subprocess.run(cmd, check=True, cwd=str(project_root))
        print(f"\nConversion completed! Output saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"\nError during conversion: {e}")
    except FileNotFoundError:
        print("Error: vmd_to_smpl.py not found. Make sure you're running from the correct directory.")

if __name__ == "__main__":
    main()

