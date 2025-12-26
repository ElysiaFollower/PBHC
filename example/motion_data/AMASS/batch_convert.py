"""
批量转换AMASS的npz文件为pkl格式

遍历collection目录下的所有.npz文件，调用convert_npz_to_pkl.py进行转换，
并将转换后的pkl文件保存到collection/pkl目录中。
"""

import os
import sys
import subprocess
from pathlib import Path

def batch_convert():
    """批量转换npz文件为pkl格式"""
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    collection_dir = script_dir / "collection"
    output_dir = collection_dir / "pkl"
    
    # 检查collection目录是否存在
    if not collection_dir.exists():
        print(f"错误: collection目录不存在: {collection_dir}")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 查找所有npz文件
    npz_files = list(collection_dir.glob("*.npz"))
    
    if len(npz_files) == 0:
        print(f"在 {collection_dir} 中未找到npz文件")
        return
    
    print(f"找到 {len(npz_files)} 个npz文件")
    print("-" * 80)
    
    # 转换脚本路径（使用绝对路径）
    convert_script = script_dir / "convert_npz_to_pkl.py"
    
    if not convert_script.exists():
        print(f"错误: 转换脚本不存在: {convert_script}")
        return
    
    # 遍历转换每个文件
    success_count = 0
    fail_count = 0
    
    for i, npz_file in enumerate(npz_files, 1):
        print(f"\n[{i}/{len(npz_files)}] 处理: {npz_file.name}")
        
        # 生成输出文件路径
        output_file = output_dir / f"{npz_file.stem}.pkl"
        
        # 如果输出文件已存在，跳过
        if output_file.exists():
            print(f"  跳过（文件已存在）: {output_file.name}")
            success_count += 1
            continue
        
        try:
            # 调用转换脚本
            # 使用绝对路径确保能找到脚本
            cmd = [
                sys.executable,
                str(convert_script.absolute()),
                str(npz_file.absolute()),
                "--output",
                str(output_file.absolute())
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(script_dir.absolute()),
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"  ✓ 转换成功: {output_file.name}")
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"  ✗ 转换失败: {npz_file.name}")
            print(f"    错误信息: {e.stderr}")
            fail_count += 1
        except Exception as e:
            print(f"  ✗ 转换失败: {npz_file.name}")
            print(f"    错误信息: {str(e)}")
            fail_count += 1
    
    # 输出总结
    print("\n" + "=" * 80)
    print("转换完成!")
    print("=" * 80)
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"输出目录: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    batch_convert()

