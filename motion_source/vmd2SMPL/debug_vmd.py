"""Debug script to examine VMD file binary structure."""

import sys
from pathlib import Path

vmd_file = Path("vmd/愛包ダンスホール_DanceMotion_HimeTanaka(MMD)_v1.1.vmd")

if not vmd_file.exists():
    print(f"File not found: {vmd_file}")
    sys.exit(1)

with open(vmd_file, 'rb') as f:
    # Read header
    header = f.read(30)
    print(f"Header (30 bytes): {header}")
    print(f"Header as string: {header.decode('ascii', errors='ignore')}")
    print()
    
    # Read next 100 bytes
    next_bytes = f.read(100)
    print(f"Next 100 bytes (hex): {next_bytes.hex()}")
    print()
    
    # Try to interpret as different formats
    f.seek(30)
    
    # Try reading model name length
    model_len_bytes = f.read(4)
    print(f"Model name length bytes: {model_len_bytes.hex()}")
    print(f"  As little-endian uint32: {int.from_bytes(model_len_bytes, 'little', signed=False)}")
    print(f"  As big-endian uint32: {int.from_bytes(model_len_bytes, 'big', signed=False)}")
    print()
    
    # Try reading bone count from different positions
    for offset in [0, 4, 8, 12, 16, 20, 24]:
        f.seek(30 + offset)
        test_bytes = f.read(4)
        test_le = int.from_bytes(test_bytes, 'little', signed=False)
        test_be = int.from_bytes(test_bytes, 'big', signed=False)
        if 10 <= test_le <= 100000:
            print(f"Offset {offset}: Found potential bone count (LE): {test_le}")
            # Check if next bytes look like bone name
            f.seek(30 + offset + 4)
            bone_name_test = f.read(15)
            print(f"  Next 15 bytes (potential bone name): {bone_name_test}")
            print(f"  Has non-zero bytes: {any(b != 0 for b in bone_name_test)}")
        if 10 <= test_be <= 100000:
            print(f"Offset {offset}: Found potential bone count (BE): {test_be}")
            f.seek(30 + offset + 4)
            bone_name_test = f.read(15)
            print(f"  Next 15 bytes (potential bone name): {bone_name_test}")
            print(f"  Has non-zero bytes: {any(b != 0 for b in bone_name_test)}")

