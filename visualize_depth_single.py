#!/usr/bin/env python3
"""
可视化单个深度图，生成彩色深度图
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def visualize_depth_png(depth_path, output_path=None, colormap='viridis', percentile_range=(5, 95)):
    """
    读取深度图并生成彩色可视化
    
    Args:
        depth_path: 深度图路径
        output_path: 输出路径（可选），如果不指定则显示图像
        colormap: 颜色映射，可选 'viridis', 'jet', 'plasma', 'turbo' 等
        percentile_range: 用于自动剪裁的百分位范围
    """
    # 读取深度图（支持16位和8位）
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Cannot load image: {depth_path}")
    
    print(f"[INFO] Depth image shape: {depth_raw.shape}, dtype: {depth_raw.dtype}")
    
    # 转换为浮点数
    depth = depth_raw.astype(np.float32)
    
    # 处理有效深度值
    valid_mask = depth > 0
    
    if not np.any(valid_mask):
        print("[WARN] No valid depth values found. Using all pixels.")
        valid_mask = np.ones_like(depth, dtype=bool)
    
    # 自动剪裁上下限（排除极端值）
    if np.any(valid_mask):
        vmin, vmax = np.percentile(depth[valid_mask], percentile_range)
        print(f"[INFO] Depth range: min={depth.min():.2f}, max={depth.max():.2f}")
        print(f"[INFO] Using percentile range: {vmin:.2f} - {vmax:.2f}")
    else:
        vmin, vmax = depth.min(), depth.max()
    
    # 归一化到 [0, 1]
    depth_vis = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    # 应用颜色映射
    if colormap == 'viridis':
        # 使用matplotlib的viridis colormap
        depth_colored = plt.cm.viridis(depth_vis)[:, :, :3]  # RGB，去除alpha通道
        depth_colored = (depth_colored * 255).astype(np.uint8)
    elif colormap == 'jet':
        depth_colored = plt.cm.jet(depth_vis)[:, :, :3]
        depth_colored = (depth_colored * 255).astype(np.uint8)
    elif colormap == 'plasma':
        depth_colored = plt.cm.plasma(depth_vis)[:, :, :3]
        depth_colored = (depth_colored * 255).astype(np.uint8)
    elif colormap == 'turbo':
        depth_colored = plt.cm.turbo(depth_vis)[:, :, :3]
        depth_colored = (depth_colored * 255).astype(np.uint8)
    else:
        # 使用OpenCV的colormap
        depth_vis_uint8 = (depth_vis * 255).astype(np.uint8)
        if colormap == 'cv_viridis':
            depth_colored = cv2.applyColorMap(depth_vis_uint8, cv2.COLORMAP_VIRIDIS)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        elif colormap == 'cv_jet':
            depth_colored = cv2.applyColorMap(depth_vis_uint8, cv2.COLORMAP_JET)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unknown colormap: {colormap}")
    
    # 保存或显示
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 保存为PNG（RGB格式）
        cv2.imwrite(str(output_path), cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved colored depth image to {output_path}")
    else:
        # 显示图像
        plt.figure(figsize=(12, 8))
        plt.imshow(depth_colored)
        plt.title(f"Colored Depth Visualization ({colormap})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return depth_colored


if __name__ == "__main__":
    depth_path = "/home/ruoqu/crq_ws/HO-Cap-Annotation/data/videos_1121_annotated/fork_slice_dough/20251121_bigwoodenfork_slice_dough_half_1/processed/fd_pose_solver/debug/wooden_fork_2/depth.png"
    
    # 生成输出路径（在同一目录下）
    depth_path_obj = Path(depth_path)
    output_path = depth_path_obj.parent / "depth_colored.png"
    
    # 可视化深度图（使用viridis colormap）
    visualize_depth_png(depth_path, output_path=str(output_path), colormap='viridis')
    
    print(f"\n[SUCCESS] Colored depth image saved to: {output_path}")


