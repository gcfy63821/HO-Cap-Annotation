#!/usr/bin/env python
"""
Align each cached point cloud to z=0 plane and crop to reasonable bounds.

This script:
1. Loads cached point clouds from 00-0_align_cameras.py output (cached_pc/cam*_uncropped.ply)
2. For each camera's point cloud:
   - Fits a plane using RANSAC
   - Transforms the point cloud to align the plane with z=0
   - Crops to reasonable X/Y bounds
3. Saves each flattened point cloud as PLY file
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import copy
from tqdm import tqdm


def fit_plane_ransac(pc, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Fit a plane to point cloud using RANSAC.
    
    Args:
        pc: Open3D point cloud
        distance_threshold: Maximum distance from point to plane to be considered inlier
        ransac_n: Number of points to sample for plane fitting
        num_iterations: Number of RANSAC iterations
    
    Returns:
        plane_model: [a, b, c, d] where ax + by + cz + d = 0
        inliers: List of inlier indices
    """
    plane_model, inliers = pc.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return plane_model, inliers


def plane_to_transform(plane_model):
    """
    Compute transformation matrix to align plane to z=0.
    
    Args:
        plane_model: [a, b, c, d] where ax + by + cz + d = 0
    
    Returns:
        4x4 transformation matrix
    """
    a, b, c, d = plane_model
    
    # Normalize plane normal
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # Target normal is [0, 0, 1] (pointing up)
    target_normal = np.array([0, 0, 1])
    
    # Compute rotation axis and angle
    axis = np.cross(normal, target_normal)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:
        # Planes are already aligned
        if np.dot(normal, target_normal) > 0:
            return np.eye(4)
        else:
            # Flip 180 degrees around x or y
            return np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
    
    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(normal, target_normal), -1, 1))
    
    # Compute rotation matrix using Rodrigues' formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # Compute translation to move plane to z=0
    # We need to find a point on the plane and translate it so its z becomes 0
    # For simplicity, use the point closest to origin projected onto the plane
    # Then translate by -d/c in z direction (if c != 0)
    if abs(c) > 1e-6:
        # Point on plane closest to origin: project origin onto plane
        point_on_plane = -d * normal / (np.dot(normal, normal))
        # After rotation, we want z=0, so translate by -z_component
        rotated_point = R @ point_on_plane
        t = np.array([0, 0, -rotated_point[2]])
    else:
        t = np.array([0, 0, 0])
    
    # Build transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def crop_pointcloud_xy(pc, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), z_range=(-0.1, 0.1)):
    """
    Crop point cloud to specified X/Y/Z bounds.
    
    Args:
        pc: Open3D point cloud
        x_range: Tuple (min, max) for X coordinate
        y_range: Tuple (min, max) for Y coordinate
        z_range: Tuple (min, max) for Z coordinate (for z=0 plane, should be small)
    
    Returns:
        Cropped Open3D point cloud
    """
    points = np.asarray(pc.points)
    
    x_mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
    y_mask = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    z_mask = (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    valid_mask = x_mask & y_mask & z_mask
    
    return pc.select_by_index(np.where(valid_mask)[0])


def process_single_pointcloud(pc, cam_idx, x_range, y_range, plane_distance_threshold):
    """
    Process a single point cloud: fit plane, align to z=0, and crop.
    
    Args:
        pc: Open3D point cloud
        cam_idx: Camera index (for logging)
        x_range: X coordinate bounds for cropping
        y_range: Y coordinate bounds for cropping
        plane_distance_threshold: Distance threshold for plane fitting
    
    Returns:
        Aligned and cropped point cloud
    """
    if len(pc.points) == 0:
        print(f"  [WARNING] Camera {cam_idx}: Empty point cloud, skipping")
        return None
    
    # Remove outliers
    pc, ind = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    if len(pc.points) < 10:
        print(f"  [WARNING] Camera {cam_idx}: Too few points after outlier removal, skipping")
        return None
    
    # Downsample for faster processing
    pc = pc.voxel_down_sample(voxel_size=0.005)
    
    if len(pc.points) < 3:
        print(f"  [WARNING] Camera {cam_idx}: Too few points after downsampling, skipping")
        return None
    
    # Fit plane using RANSAC
    try:
        plane_model, inliers = fit_plane_ransac(
            pc,
            distance_threshold=plane_distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
    except Exception as e:
        print(f"  [ERROR] Camera {cam_idx}: Failed to fit plane: {e}")
        return None
    
    a, b, c, d = plane_model
    inlier_ratio = len(inliers) / len(pc.points) if len(pc.points) > 0 else 0
    print(f"  Camera {cam_idx}: Plane normal=[{a:.4f}, {b:.4f}, {c:.4f}], "
          f"inliers={len(inliers)}/{len(pc.points)} ({100*inlier_ratio:.1f}%)")
    
    # Compute transformation to align plane to z=0
    T = plane_to_transform(plane_model)
    
    # Apply transformation
    pc_aligned = copy.deepcopy(pc)
    pc_aligned.transform(T)
    
    # Verify plane alignment (check z values of inliers)
    if len(inliers) > 0:
        inlier_points_aligned = np.asarray(pc_aligned.select_by_index(inliers).points)
        z_mean = np.mean(inlier_points_aligned[:, 2])
        z_std = np.std(inlier_points_aligned[:, 2])
        print(f"  Camera {cam_idx}: After alignment - z: mean={z_mean:.4f}, std={z_std:.4f}")
    
    # Crop to reasonable bounds
    pc_cropped = crop_pointcloud_xy(
        pc_aligned,
        x_range=x_range,
        y_range=y_range,
        z_range=(-0.1, 0.1)  # Small range around z=0
    )
    
    return pc_cropped


def main(cached_pc_dir, out_path=None,
         x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
         plane_distance_threshold=0.01):
    """
    Align each cached point cloud to z=0 plane and crop.
    
    Args:
        cached_pc_dir: Directory containing cached point clouds (cached_pc/)
        out_path: Output directory
        x_range: X coordinate bounds for cropping
        y_range: Y coordinate bounds for cropping
        plane_distance_threshold: Distance threshold for plane fitting
    """
    cached_pc_dir = Path(cached_pc_dir)
    
    if not cached_pc_dir.exists():
        raise FileNotFoundError(f"Cached PC directory not found: {cached_pc_dir}")
    
    # Determine output path
    if out_path is None:
        out_path = cached_pc_dir.parent / "flat_pc"
    else:
        out_path = Path(out_path)
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find all cached point cloud files
    cached_files = sorted(cached_pc_dir.glob("cam*_uncropped.ply"))
    
    if len(cached_files) == 0:
        raise FileNotFoundError(f"No cached point cloud files found in {cached_pc_dir}")
    
    print(f"[INFO] Found {len(cached_files)} cached point cloud files")
    print(f"[INFO] Output directory: {out_path}")
    
    # Process each point cloud
    processed_pcs = []
    for cached_file in tqdm(cached_files, desc="Processing point clouds"):
        # Extract camera index from filename (e.g., "cam0_uncropped.ply" -> 0)
        try:
            cam_idx = int(cached_file.stem.split("_")[0].replace("cam", ""))
        except:
            print(f"[WARNING] Could not extract camera index from {cached_file.name}, skipping")
            continue
        
        print(f"\n[INFO] Processing camera {cam_idx} from {cached_file.name}")
        pc = o3d.io.read_point_cloud(str(cached_file))
        
        pc_processed = process_single_pointcloud(
            pc, cam_idx, x_range, y_range, plane_distance_threshold
        )
        
        if pc_processed is not None and len(pc_processed.points) > 0:
            # Save individual camera point cloud
            output_file = out_path / f"cam{cam_idx}_flat_z0.ply"
            o3d.io.write_point_cloud(str(output_file), pc_processed)
            print(f"  [SUCCESS] Saved to {output_file}")
            
            processed_pcs.append(pc_processed)
    
    # Merge all processed point clouds
    if len(processed_pcs) > 0:
        print(f"\n[INFO] Merging {len(processed_pcs)} point clouds...")
        merged_pc = o3d.geometry.PointCloud()
        for pc in processed_pcs:
            merged_pc += pc
        
        # Save merged point cloud
        merged_file = out_path / "merged_flat_z0.ply"
        o3d.io.write_point_cloud(str(merged_file), merged_pc)
        print(f"[SUCCESS] Merged point cloud saved to: {merged_file}")
        print(f"[INFO] Total points: {len(merged_pc.points)}")
        
        # Print bounds
        bbox = merged_pc.get_axis_aligned_bounding_box()
        print(f"[INFO] Point cloud bounds:")
        print(f"  X: [{bbox.min_bound[0]:.3f}, {bbox.max_bound[0]:.3f}]")
        print(f"  Y: [{bbox.min_bound[1]:.3f}, {bbox.max_bound[1]:.3f}]")
        print(f"  Z: [{bbox.min_bound[2]:.3f}, {bbox.max_bound[2]:.3f}]")
    else:
        print("[WARNING] No point clouds were successfully processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align each cached point cloud to z=0 plane and crop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cached_pc_dir",
        type=str,
        required=True,
        help="Directory containing cached point clouds (e.g., cached_pc/ from 00-0_align_cameras.py output)"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output directory (default: flat_pc/ in parent of cached_pc_dir)"
    )
    parser.add_argument(
        "--x_range",
        type=float,
        nargs=2,
        default=[-1.0, 1.0],
        metavar=("MIN", "MAX"),
        help="X coordinate bounds for cropping"
    )
    parser.add_argument(
        "--y_range",
        type=float,
        nargs=2,
        default=[-1.0, 1.0],
        metavar=("MIN", "MAX"),
        help="Y coordinate bounds for cropping"
    )
    parser.add_argument(
        "--plane_distance_threshold",
        type=float,
        default=0.01,
        help="Distance threshold for plane fitting (meters)"
    )
    
    args = parser.parse_args()
    
    main(
        cached_pc_dir=args.cached_pc_dir,
        out_path=args.out_path,
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        plane_distance_threshold=args.plane_distance_threshold
    )

