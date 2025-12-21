"""Example usage of VMD to PKL converter.

This script demonstrates how to use the VMD to PKL converter.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run example conversion."""
    # Example: Convert a VMD file to PKL
    # Use quotes around filename to handle parentheses in PowerShell
    vmd_file = r"vmd/愛包ダンスホール_DanceMotion_HimeTanaka(MMD)_v1.1.vmd"
    output_file = "output_dance.pkl"
    
    script_dir = Path(__file__).parent
    vmd_path = script_dir / vmd_file
    
    if not vmd_path.exists():
        print(f"Error: VMD file not found: {vmd_path}")
        print("Please ensure the VMD file exists in the vmd/ directory")
        return
    
    # Use absolute paths and quote the input file to handle special characters
    cmd = [
        sys.executable,
        str(script_dir / "vmd_to_pkl.py"),
        "--input", str(vmd_path),
        "--output", str(script_dir / output_file),
        "--fps", "30",
        "--robot_type", "g1"
    ]
    
    print("Running conversion command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nConversion completed! Output saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"\nError during conversion: {e}")
    except FileNotFoundError:
        print("Error: vmd_to_pkl.py not found. Make sure you're running from the correct directory.")

if __name__ == "__main__":
    main()

