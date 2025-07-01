import h5py
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def save_depth_mm_png(depth_m, save_path):
    """将深度（米）转为16-bit PNG（毫米）"""
    depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(str(save_path), depth_mm)

def save_depth_vis(depth_m, save_path, percentile_range=(5, 95)):
    """保存伪彩色可视化图"""
    depth = depth_m.copy()
    valid_mask = depth > 0
    if not np.any(valid_mask):
        print(f"[WARN] Empty depth map: {save_path}")
        return

    vmin, vmax = np.percentile(depth[valid_mask], percentile_range)
    depth_norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    depth_colored = plt.cm.jet(depth_norm)[:, :, :3]  # 去除 alpha 通道
    depth_colored = (depth_colored * 255).astype(np.uint8)
    cv2.imwrite(str(save_path), cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))

def export_depths_from_h5(
    h5_path,
    output_dir,
    max_frames=10,
    camera_ids=None,
):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        depths = f["depths"][:]  # shape = (N, 8, H, W)
        print(f"[INFO] Loaded depths, shape: {depths.shape}, dtype: {depths.dtype}")

    num_frames, num_cams, H, W = depths.shape

    if camera_ids is None:
        camera_ids = list(range(num_cams))

    for frame_idx in range(min(num_frames, max_frames)):
        for cam_id in camera_ids:
            depth_m = depths[frame_idx, cam_id]  # 单位是米

            save_dir = Path(output_dir) / f"cam{cam_id:02d}"
            save_dir.mkdir(parents=True, exist_ok=True)

            depth_png_path = save_dir / f"depth_{frame_idx:06d}.png"
            depth_vis_path = save_dir / f"depth_{frame_idx:06d}_vis.jpg"

            save_depth_mm_png(depth_m, depth_png_path)
            save_depth_vis(depth_m, depth_vis_path)

            print(f"[Frame {frame_idx} | Cam {cam_id}] Min: {depth_m.min():.4f} m, "
                  f"Max: {depth_m.max():.4f} m, Mean: {depth_m.mean():.4f} m")

# 示例调用
if __name__ == "__main__":
    h5_path = "/home/wys/learning-compliant/crq_ws/data/raw_h5s/0b99324a_wooden_spoon_small_6/data00000000.h5"
    output_dir = "debug_outputs/exported_depths"
    export_depths_from_h5(h5_path, output_dir, max_frames=5)
