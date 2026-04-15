#!/usr/bin/env python3
"""
将所有相机第一帧的深度图转换为点云并投影到一起，检查深度是否正常
"""
import numpy as np
import open3d as o3d
from pathlib import Path
import sys
import argparse
from hocap_annotation.loaders.my_cluster_loader import MyClusterLoader

def visualize_first_frame_depth_pcd(sequence_folder, output_path=None, filter_outliers=True):
    """
    将所有相机第一帧的深度图转换为点云并投影到一起
    
    Args:
        sequence_folder: 序列文件夹路径
        output_path: 输出点云文件路径（可选）
        filter_outliers: 是否过滤离群点
    """
    print(f"[INFO] Loading data from: {sequence_folder}")
    
    # 加载数据
    loader = MyClusterLoader(sequence_folder)
    
    frame_id = 100
    print(f"[INFO] Processing frame {frame_id}")
    print(f"[INFO] Number of cameras: {len(loader.rs_serials)}")
    print(f"[INFO] Camera serials: {loader.rs_serials}")
    
    # 收集所有相机的点云
    all_points = []
    all_colors = []
    camera_colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
        [1, 0.5, 0],  # Orange
        [0.5, 0, 1],  # Purple
    ]
    
    depth_stats = []
    
    for cam_idx, serial in enumerate(loader.rs_serials):
        print(f"\n[INFO] Processing camera {serial} (index {cam_idx})")
        
        # 获取深度图
        depth = loader.get_depth(serial, frame_id)
        print(f"  Depth shape: {depth.shape}")
        print(f"  Depth range: [{depth.min():.4f}, {depth.max():.4f}] m")
        print(f"  Depth mean: {depth.mean():.4f} m")
        print(f"  Valid pixels: {(depth > 0).sum()} / {depth.size}")
        
        depth_stats.append({
            'serial': serial,
            'min': depth.min(),
            'max': depth.max(),
            'mean': depth.mean(),
            'valid_ratio': (depth > 0).sum() / depth.size
        })
        
        # 获取相机内参和外参
        K = loader.rs_Ks[cam_idx]
        T = loader.extr2world[cam_idx]
        
        print(f"  Camera intrinsics K:")
        print(f"    {K}")
        print(f"  Camera extrinsics T (to world):")
        print(f"    Translation: {T[:3, 3]}")
        
        # 将深度图转换为点云（使用正确的齐次坐标转换）
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth.flatten()
        
        # 过滤无效深度值
        valid_mask = depth_flat > 0
        u_flat = u_flat[valid_mask]
        v_flat = v_flat[valid_mask]
        depth_flat = depth_flat[valid_mask]
        
        # 从像素坐标和深度计算相机坐标系下的3D点
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_cam = (u_flat - cx) * depth_flat / fx
        y_cam = (v_flat - cy) * depth_flat / fy
        z_cam = depth_flat
        
        # 相机坐标系下的点
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)
        
        # 转换为齐次坐标
        ones = np.ones((points_cam.shape[0], 1))
        points_cam_hom = np.hstack([points_cam, ones])  # (N, 4)
        
        # 使用外参矩阵转换到世界坐标系
        # T 是 4x4 矩阵，从相机坐标系到世界坐标系
        points_world_hom = (T @ points_cam_hom.T).T  # (N, 4)
        points = points_world_hom[:, :3]  # (N, 3)
        
        print(f"  Valid points: {len(points)}")
        if len(points) > 0:
            print(f"  Point cloud range:")
            print(f"    X: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]")
            print(f"    Y: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")
            print(f"    Z: [{points[:, 2].min():.4f}, {points[:, 2].max():.4f}]")
        
        # 为每个相机的点云分配颜色
        color = camera_colors[cam_idx % len(camera_colors)]
        colors = np.tile(color, (len(points), 1))
        
        all_points.append(points)
        all_colors.append(colors)
    
    # 合并所有点云
    print(f"\n[INFO] Merging point clouds from all cameras...")
    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)
    
    print(f"[INFO] Total points: {len(merged_points)}")
    print(f"[INFO] Merged point cloud range:")
    print(f"  X: [{merged_points[:, 0].min():.4f}, {merged_points[:, 0].max():.4f}]")
    print(f"  Y: [{merged_points[:, 1].min():.4f}, {merged_points[:, 1].max():.4f}]")
    print(f"  Z: [{merged_points[:, 2].min():.4f}, {merged_points[:, 2].max():.4f}]")
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    
    # 过滤离群点（可选）
    if filter_outliers and len(merged_points) > 0:
        print(f"\n[INFO] Filtering outliers...")
        print(f"  Points before filtering: {len(pcd.points)}")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)
        print(f"  Points after filtering: {len(pcd.points)}")
    
    # 保存点云
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"\n[SUCCESS] Saved point cloud to: {output_path}")
    else:
        # 默认保存路径
        sequence_path = Path(sequence_folder)
        output_path = sequence_path / "debug" / "first_frame_depth_pcd.ply"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"\n[SUCCESS] Saved point cloud to: {output_path}")
    
    # 打印深度统计信息
    print("\n" + "=" * 80)
    print("[DEPTH STATISTICS]")
    print("=" * 80)
    for stat in depth_stats:
        print(f"Camera {stat['serial']}:")
        print(f"  Min depth: {stat['min']:.4f} m")
        print(f"  Max depth: {stat['max']:.4f} m")
        print(f"  Mean depth: {stat['mean']:.4f} m")
        print(f"  Valid pixel ratio: {stat['valid_ratio']:.2%}")
    
    # 可视化点云
    print("\n[INFO] Opening point cloud viewer...")
    print("[INFO] Each camera's points are colored differently:")
    for cam_idx, serial in enumerate(loader.rs_serials):
        color = camera_colors[cam_idx % len(camera_colors)]
        color_name = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange", "Purple"][cam_idx % len(camera_colors)]
        print(f"  Camera {serial}: {color_name} {color}")
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="First Frame Depth Point Cloud (All Cameras)",
        width=1920,
        height=1080,
    )
    
    return pcd, depth_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize first frame depth point cloud from all cameras")
    parser.add_argument(
        "sequence_folder",
        type=str,
        nargs="?",
        default="/home/ruoqu/crq_ws/HO-Cap-Annotation/data/videos_0115/mallet_crush_banana/20260116_mallet_crush_pealed_banana_50",
        help="Path to sequence folder"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PLY file path (default: sequence_folder/debug/first_frame_depth_pcd.ply)"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable outlier filtering"
    )
    
    args = parser.parse_args()
    
    visualize_first_frame_depth_pcd(
        args.sequence_folder,
        output_path=args.output,
        filter_outliers=not args.no_filter
    )

