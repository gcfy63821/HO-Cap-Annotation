#!/usr/bin/env python3
"""
检查H5文件的数据结构
"""
import h5py
import numpy as np
from pathlib import Path

def check_h5_structure(h5_path):
    """
    检查H5文件的结构和形状
    """
    h5_path = Path(h5_path)
    
    if not h5_path.exists():
        print(f"[ERROR] File not found: {h5_path}")
        return
    
    print(f"[INFO] Opening H5 file: {h5_path}")
    print("=" * 80)
    
    with h5py.File(h5_path, 'r') as f:
        print(f"[INFO] File size: {h5_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"[INFO] Keys in root: {list(f.keys())}")
        print("=" * 80)
        
        def print_structure(name, obj):
            """递归打印H5文件结构"""
            indent = "  " * (name.count('/'))
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}Dataset: {name}")
                print(f"{indent}  Shape: {obj.shape}")
                print(f"{indent}  Dtype: {obj.dtype}")
                print(f"{indent}  Size: {obj.size}")
                
                # 显示一些统计信息
                if obj.size > 0:
                    try:
                        data = obj[:]
                        if data.size < 100:  # 如果数据很小，显示所有值
                            print(f"{indent}  Values: {data}")
                        else:
                            print(f"{indent}  Min: {np.min(data)}")
                            print(f"{indent}  Max: {np.max(data)}")
                            print(f"{indent}  Mean: {np.mean(data):.4f}")
                            if data.dtype in [np.int32, np.int64, np.uint8, np.uint16, np.uint32]:
                                unique_vals = np.unique(data)
                                if len(unique_vals) <= 20:
                                    print(f"{indent}  Unique values: {unique_vals}")
                                else:
                                    print(f"{indent}  Unique values count: {len(unique_vals)}")
                                    print(f"{indent}  Unique value range: [{unique_vals.min()}, {unique_vals.max()}]")
                    except Exception as e:
                        print(f"{indent}  [Could not read data: {e}]")
                print()
            elif isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
                print(f"{indent}  Keys: {list(obj.keys())}")
                print()
        
        # 遍历所有键
        f.visititems(print_structure)
        
        # 如果有特定的数据集，显示更详细的信息
        print("=" * 80)
        print("[DETAILED INFO]")
        print("=" * 80)
        
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                print(f"\nDataset '{key}':")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                if len(obj.shape) > 0:
                    print(f"  First few values:")
                    if obj.size < 100:
                        print(f"    {obj[:]}")
                    else:
                        # 显示第一个元素或切片
                        if len(obj.shape) == 1:
                            print(f"    First 10: {obj[:10]}")
                        elif len(obj.shape) == 2:
                            print(f"    First row: {obj[0, :]}")
                            print(f"    First col: {obj[:, 0]}")
                        elif len(obj.shape) == 3:
                            print(f"    First frame shape: {obj[0].shape}")
                            print(f"    First frame sample (top-left 5x5):")
                            print(f"    {obj[0, :5, :5]}")
                        elif len(obj.shape) == 4:
                            print(f"    First element shape: {obj[0].shape}")
                            print(f"    First element sample (first frame, top-left 5x5):")
                            print(f"    {obj[0, 0, :5, :5]}")
            elif isinstance(obj, h5py.Group):
                print(f"\nGroup '{key}':")
                for subkey in obj.keys():
                    subobj = obj[subkey]
                    if isinstance(subobj, h5py.Dataset):
                        print(f"  Dataset '{subkey}':")
                        print(f"    Shape: {subobj.shape}")
                        print(f"    Dtype: {subobj.dtype}")

if __name__ == "__main__":
    h5_path = "/home/ruoqu/crq_ws/HO-Cap-Annotation/data/videos_1121_annotated/fork_slice_dough/20251121_bigwoodenfork_slice_dough_half_1/tool_masks/masks.h5"
    check_h5_structure(h5_path)

