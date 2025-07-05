import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def visualize_masks(mask_dir, output_dir=None, cmap=cv2.COLORMAP_JET):
    """
    将指定目录下的所有npy mask文件可视化为png图片。
    mask_dir: 存放npy mask的目录
    output_dir: 输出png的目录（默认为mask_dir/png_vis）
    cmap: 可选的OpenCV colormap
    """
    mask_dir = Path(mask_dir)
    if output_dir is None:
        output_dir = mask_dir / "png_vis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(mask_dir.glob("*.npy"))
    if not npy_files:
        print(f"[WARN] No npy files found in {mask_dir}")
        return

    for npy_file in tqdm(npy_files, desc="Visualizing masks"):
        mask = np.load(npy_file)
        mask = np.squeeze(mask)
        # 归一化到0-255
        if mask.max() > 1:
            mask_vis = (mask / mask.max() * 255).astype(np.uint8)
        else:
            mask_vis = (mask * 255).astype(np.uint8)
        mask_color = cv2.applyColorMap(mask_vis, cmap)
        out_path = output_dir / f"{npy_file.stem}.png"
        cv2.imwrite(str(out_path), mask_color)
        # 可选：直接显示
        # cv2.imshow("mask", mask_color)
        # cv2.waitKey(0)
    print(f"[INFO] All masks visualized to {output_dir}")

def visualize_single_npy_mask(npy_path, output_dir=None, cmap=cv2.COLORMAP_JET):
    """
    可视化单个npy文件中每个唯一取值对应的mask，并保存为png。
    npy_path: 单个npy文件路径
    output_dir: 输出png的目录（默认为npy文件同目录下的vis_{stem}文件夹）
    """
    npy_path = Path(npy_path)
    mask = np.load(npy_path)
    mask = np.squeeze(mask)
    unique_vals = np.unique(mask)
    print(f"[INFO] Unique values in mask: {unique_vals}")

    if output_dir is None:
        output_dir = npy_path.parent / f"vis_{npy_path.stem}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for val in unique_vals:
        mask_bin = (mask == val).astype(np.uint8) * 255
        mask_color = cv2.applyColorMap(mask_bin, cmap)
        out_path = output_dir / f"{npy_path.stem}_val{val}.png"
        cv2.imwrite(str(out_path), mask_color)
        print(f"[INFO] Saved mask for value {val} to {out_path}")

if __name__ == "__main__":
    '''
        usage: view all labels in a single npy mask file
    '''
    mask_dir = "/home/wys/learning-compliant/crq_ws/data/0513data/2379b837_coffee_1/masks/cam00.mp4/0.npy"
    output_dir = "debug_outputs/mask_vis"
    visualize_single_npy_mask(mask_dir, output_dir)
