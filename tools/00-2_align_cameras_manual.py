#!/usr/bin/env python
"""
Interactive manual point cloud alignment using 3-point correspondence.

This script:
1. Loads cached point clouds from 00-0_align_cameras.py
2. Selects a reference point cloud
3. For each other point cloud:
   - Displays reference and current point cloud side by side
   - User clicks 3 corresponding points in each
   - Computes transformation matrix from 3-point correspondence
   - Applies transformation and updates extrinsics
4. Saves corrected extrinsics and merged point cloud
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import yaml
import copy
from tqdm import tqdm


def pick_points_interactive(pcd, num_points=3):
    """
    Pick points from point cloud using Open3D's built-in picker.
    
    Args:
        pcd: Open3D point cloud
        num_points: Number of points to pick
    
    Returns:
        List of selected point indices
    """
    print(f"\n[INSTRUCTIONS]")
    print(f"1. A window will open showing the point cloud")
    print(f"2. Press 'K' to lock the screen and enable point picking")
    print(f"3. Left click on {num_points} points to select them")
    print(f"4. Selected points will be highlighted")
    print(f"5. Press 'Q' or close the window when done")
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"Point Picker - Select {num_points} points")
    vis.add_geometry(pcd)
    
    # Set point size for better visibility
    opt = vis.get_render_option()
    opt.point_size = 5.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    vis.run()  # User picks points here
    vis.destroy_window()
    
    # Get picked points
    picked_indices = vis.get_picked_points()
    
    if len(picked_indices) == 0:
        print("[WARNING] No points were picked!")
        return []
    
    print(f"[INFO] Picked {len(picked_indices)} points")
    return picked_indices


def compute_transformation_3points(src_points, dst_points):
    """
    Compute transformation matrix from 3 point correspondences.
    
    Uses SVD-based method to compute optimal rigid transformation.
    
    Args:
        src_points: 3x3 array, each row is a point in source coordinate system
        dst_points: 3x3 array, each row is a point in destination coordinate system
    
    Returns:
        4x4 transformation matrix
    """
    if src_points.shape != (3, 3) or dst_points.shape != (3, 3):
        raise ValueError("Both src_points and dst_points must be 3x3 arrays")
    
    # Center the points
    src_center = src_points.mean(axis=0)
    dst_center = dst_points.mean(axis=0)
    
    src_centered = src_points - src_center
    dst_centered = dst_points - dst_center
    
    # Compute covariance matrix
    H = src_centered.T @ dst_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = dst_center - R @ src_center
    
    # Build transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def visualize_single_pointcloud(pcd, title="Point Cloud"):
    """
    Visualize a single point cloud.
    
    Args:
        pcd: Open3D point cloud
        title: Window title
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    
    # Set point size
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    print(f"[INFO] Displaying {title}")
    print("[INFO] Close the window to continue...")
    
    vis.run()
    vis.destroy_window()


def visualize_two_pointclouds(pcd1, pcd2, title1="Reference", title2="Current"):
    """
    Visualize two point clouds side by side for comparison.
    
    Args:
        pcd1: First point cloud (reference)
        pcd2: Second point cloud (current)
        title1: Title for first point cloud
        title2: Title for second point cloud
    """
    # Color the point clouds differently
    pcd1_colored = copy.deepcopy(pcd1)
    pcd2_colored = copy.deepcopy(pcd2)
    
    # Reference: blue
    pcd1_colored.paint_uniform_color([0, 0, 1])
    # Current: red
    pcd2_colored.paint_uniform_color([1, 0, 0])
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{title1} (Blue) vs {title2} (Red)")
    
    # Add geometries
    vis.add_geometry(pcd1_colored)
    vis.add_geometry(pcd2_colored)
    
    # Set point size
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    print(f"\n[INFO] Displaying {title1} (blue) and {title2} (red)")
    print("[INFO] Close the window to continue...")
    
    vis.run()
    vis.destroy_window()


def main(cached_pc_dir, extrinsic_file, out_path=None, ref_cam_idx=0):
    """
    Interactive manual alignment of point clouds.
    
    Args:
        cached_pc_dir: Directory containing cached point clouds
        extrinsic_file: Path to calibration YAML file
        out_path: Output directory
        ref_cam_idx: Index of reference camera (default: 0)
    """
    cached_pc_dir = Path(cached_pc_dir)
    extrinsic_file = Path(extrinsic_file)
    
    if not cached_pc_dir.exists():
        raise FileNotFoundError(f"Cached PC directory not found: {cached_pc_dir}")
    if not extrinsic_file.exists():
        raise FileNotFoundError(f"Extrinsic file not found: {extrinsic_file}")
    
    # Determine output path
    if out_path is None:
        out_path = cached_pc_dir.parent / "manual_aligned"
    else:
        out_path = Path(out_path)
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load cached point clouds
    cached_files = sorted(cached_pc_dir.glob("cam*_uncropped.ply"))
    
    if len(cached_files) == 0:
        raise FileNotFoundError(f"No cached point cloud files found in {cached_pc_dir}")
    
    print(f"[INFO] Found {len(cached_files)} cached point cloud files")
    
    # Load all point clouds
    point_clouds = {}
    for cached_file in cached_files:
        try:
            cam_idx = int(cached_file.stem.split("_")[0].replace("cam", ""))
            pc = o3d.io.read_point_cloud(str(cached_file))
            if len(pc.points) > 0:
                point_clouds[cam_idx] = pc
                print(f"  Camera {cam_idx}: {len(pc.points)} points")
        except Exception as e:
            print(f"[WARNING] Failed to load {cached_file}: {e}")
    
    if len(point_clouds) == 0:
        raise ValueError("No valid point clouds loaded")
    
    # Check reference camera
    if ref_cam_idx not in point_clouds:
        print(f"[WARNING] Reference camera {ref_cam_idx} not found, using first available")
        ref_cam_idx = min(point_clouds.keys())
    
    print(f"\n[INFO] Using camera {ref_cam_idx} as reference")
    ref_pc = point_clouds[ref_cam_idx]
    
    # Load extrinsics
    with open(extrinsic_file, "r") as f:
        extrinsics_yaml = yaml.safe_load(f)
    
    # Support both formats
    if isinstance(extrinsics_yaml, dict) and "extrinsics" in extrinsics_yaml:
        cam_list = []
        for k, v in extrinsics_yaml["extrinsics"].items():
            if not k.startswith("tag_"):
                cam_list.append(v)
        extrinsics_yaml = cam_list
    elif isinstance(extrinsics_yaml, list):
        extrinsics_yaml = sorted(extrinsics_yaml, key=lambda x: x.get("camera_id", 0))
    
    # Create mapping from camera_id to index
    cam_id_to_idx = {}
    for i, cam_info in enumerate(extrinsics_yaml):
        cam_id = cam_info.get("camera_id", i)
        cam_id_to_idx[cam_id] = i
    
    # Store original extrinsics
    original_extrinsics = [cam["transformation"] for cam in extrinsics_yaml]
    updated_extrinsics = copy.deepcopy(original_extrinsics)
    
    # Create per-camera output directories
    per_cam_output_dir = out_path / "per_camera"
    per_cam_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load previous alignment results if they exist
    print(f"\n{'='*60}")
    print("Checking for previous alignment results...")
    print(f"{'='*60}")
    
    previous_transformations = {}
    for cam_idx in sorted(point_clouds.keys()):
        if cam_idx == ref_cam_idx:
            previous_transformations[cam_idx] = np.eye(4)
            continue
        
        cam_output_dir = per_cam_output_dir / f"cam{cam_idx}"
        transform_file = cam_output_dir / "transformation_matrix.txt"
        
        if transform_file.exists():
            try:
                T_prev = np.loadtxt(transform_file)
                previous_transformations[cam_idx] = T_prev
                print(f"  [FOUND] Previous alignment for camera {cam_idx}")
            except Exception as e:
                print(f"  [WARNING] Failed to load previous alignment for camera {cam_idx}: {e}")
    
    if len(previous_transformations) > 1:  # More than just the reference
        print(f"[INFO] Found {len(previous_transformations) - 1} previous alignment(s)")
    
    # Process each camera
    transformations = {}  # Store transformations for each camera
    
    for cam_idx in sorted(point_clouds.keys()):
        if cam_idx == ref_cam_idx:
            print(f"\n[Skipping] Camera {cam_idx} is the reference")
            transformations[cam_idx] = np.eye(4)
            continue
        
        current_pc = point_clouds[cam_idx]
        
        # Create output directory for this camera
        cam_output_dir = per_cam_output_dir / f"cam{cam_idx}"
        cam_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if there's a previous alignment result
        has_previous_alignment = cam_idx in previous_transformations
        
        if has_previous_alignment:
            print(f"\n{'='*60}")
            print(f"Camera {cam_idx} - Previous alignment found")
            print(f"{'='*60}")
            
            # Load previous transformation
            T_prev = previous_transformations[cam_idx]
            
            # Apply previous transformation to show aligned result
            current_pc_aligned_prev = copy.deepcopy(current_pc)
            current_pc_aligned_prev.transform(T_prev)
            
            # Visualize previous alignment result
            print(f"\n[INFO] Displaying previous alignment result for camera {cam_idx}...")
            visualize_two_pointclouds(
                ref_pc, current_pc_aligned_prev,
                title1=f"Reference (Cam {ref_cam_idx})",
                title2=f"Previously Aligned (Cam {cam_idx})"
            )
            
            # Ask if user wants to re-align
            re_align = input(f"\n[QUESTION] Re-align camera {cam_idx}? (y/n): ").strip().lower()
            
            if re_align != 'y':
                print(f"[INFO] Keeping previous alignment for camera {cam_idx}")
                transformations[cam_idx] = T_prev
                
                # Update extrinsic matrix using previous transformation
                if cam_idx in cam_id_to_idx:
                    idx = cam_id_to_idx[cam_idx]
                    original_ext = np.array(original_extrinsics[idx]).reshape(4, 4)
                    updated_ext = T_prev @ original_ext
                    updated_extrinsics[idx] = updated_ext.tolist()
                    print(f"[INFO] Using previous extrinsic matrix for camera {cam_idx}")
                
                continue  # Skip to next camera
        
        # If no previous alignment or user wants to re-align, proceed with manual alignment
        # Loop until user confirms alignment
        alignment_confirmed = False
        attempt = 1
        
        while not alignment_confirmed:
            print(f"\n{'='*60}")
            print(f"Processing Camera {cam_idx} (Attempt {attempt})")
            print(f"{'='*60}")
            
            # First, show the current camera's point cloud alone
            print(f"\n[STEP 0] Displaying Camera {cam_idx} point cloud")
            visualize_single_pointcloud(
                current_pc,
                title=f"Camera {cam_idx} Point Cloud"
            )
            
            # Then visualize both point clouds for comparison
            print(f"\n[STEP 0.5] Displaying reference and current point clouds together")
            visualize_two_pointclouds(
                ref_pc, current_pc,
                title1=f"Reference (Cam {ref_cam_idx})",
                title2=f"Current (Cam {cam_idx})"
            )
            
            # Pick 3 points from reference
            print(f"\n[STEP 1] Pick 3 points from REFERENCE point cloud (Camera {ref_cam_idx})")
            ref_indices = pick_points_interactive(ref_pc, num_points=3)
            
            if len(ref_indices) < 3:
                print(f"[ERROR] Need exactly 3 points, got {len(ref_indices)}.")
                retry = input("[QUESTION] Retry? (y/n): ").strip().lower()
                if retry != 'y':
                    print(f"[INFO] Skipping camera {cam_idx}")
                    break
                continue
            
            ref_indices = ref_indices[:3]  # Take first 3
            ref_points = np.asarray(ref_pc.points)[ref_indices]
            print(f"[INFO] Selected reference points:\n{ref_points}")
            
            # Pick 3 corresponding points from current
            print(f"\n[STEP 2] Pick 3 CORRESPONDING points from CURRENT point cloud (Camera {cam_idx})")
            print("[INFO] Make sure the order matches the reference points!")
            curr_indices = pick_points_interactive(current_pc, num_points=3)
            
            if len(curr_indices) < 3:
                print(f"[ERROR] Need exactly 3 points, got {len(curr_indices)}.")
                retry = input("[QUESTION] Retry? (y/n): ").strip().lower()
                if retry != 'y':
                    print(f"[INFO] Skipping camera {cam_idx}")
                    break
                continue
            
            curr_indices = curr_indices[:3]  # Take first 3
            curr_points = np.asarray(current_pc.points)[curr_indices]
            print(f"[INFO] Selected current points:\n{curr_points}")
            
            # Compute transformation
            print("\n[STEP 3] Computing transformation...")
            try:
                T = compute_transformation_3points(curr_points, ref_points)
                print(f"[INFO] Transformation matrix computed successfully")
                
                # Apply transformation to current point cloud for verification
                current_pc_transformed = copy.deepcopy(current_pc)
                current_pc_transformed.transform(T)
                
                # Visualize result
                print("\n[STEP 4] Visualizing alignment result...")
                visualize_two_pointclouds(
                    ref_pc, current_pc_transformed,
                    title1=f"Reference (Cam {ref_cam_idx})",
                    title2=f"Aligned (Cam {cam_idx})"
                )
                
                # Ask for confirmation
                confirm = input("\n[QUESTION] Does the alignment look good? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    # Save results for this camera
                    print(f"\n[INFO] Saving results for camera {cam_idx}...")
                    
                    # Save transformation matrix
                    transform_file = cam_output_dir / "transformation_matrix.txt"
                    np.savetxt(transform_file, T, fmt='%.8f')
                    print(f"  [SAVED] Transformation matrix: {transform_file}")
                    
                    # Save selected points
                    points_file = cam_output_dir / "selected_points.txt"
                    with open(points_file, 'w') as f:
                        f.write("Reference points (from cam {}):\n".format(ref_cam_idx))
                        for i, pt in enumerate(ref_points):
                            f.write(f"  Point {i+1}: [{pt[0]:.6f}, {pt[1]:.6f}, {pt[2]:.6f}]\n")
                        f.write("\nCurrent points (from cam {}):\n".format(cam_idx))
                        for i, pt in enumerate(curr_points):
                            f.write(f"  Point {i+1}: [{pt[0]:.6f}, {pt[1]:.6f}, {pt[2]:.6f}]\n")
                    print(f"  [SAVED] Selected points: {points_file}")
                    
                    # Save aligned point cloud
                    aligned_pc_file = cam_output_dir / "aligned_pointcloud.ply"
                    o3d.io.write_point_cloud(str(aligned_pc_file), current_pc_transformed)
                    print(f"  [SAVED] Aligned point cloud: {aligned_pc_file}")
                    
                    # Save original point cloud for reference
                    original_pc_file = cam_output_dir / "original_pointcloud.ply"
                    o3d.io.write_point_cloud(str(original_pc_file), current_pc)
                    print(f"  [SAVED] Original point cloud: {original_pc_file}")
                    
                    # Store transformation
                    transformations[cam_idx] = T
                    
                    # Update extrinsic matrix
                    if cam_idx in cam_id_to_idx:
                        idx = cam_id_to_idx[cam_idx]
                        original_ext = np.array(original_extrinsics[idx]).reshape(4, 4)
                        updated_ext = T @ original_ext
                        updated_extrinsics[idx] = updated_ext.tolist()
                        
                        # Save updated extrinsic
                        extrinsic_file_cam = cam_output_dir / "updated_extrinsic.txt"
                        np.savetxt(extrinsic_file_cam, updated_ext, fmt='%.8f')
                        print(f"  [SAVED] Updated extrinsic: {extrinsic_file_cam}")
                        print(f"[INFO] Updated extrinsic matrix for camera {cam_idx}")
                    else:
                        print(f"[WARNING] Camera {cam_idx} not found in extrinsics, skipping update")
                    
                    alignment_confirmed = True
                    print(f"[SUCCESS] Camera {cam_idx} alignment confirmed and saved!")
                
                else:
                    print(f"[INFO] Re-calibrating camera {cam_idx}...")
                    attempt += 1
                    # Continue loop to retry
                    
            except Exception as e:
                print(f"[ERROR] Failed to compute transformation: {e}")
                retry = input("[QUESTION] Retry? (y/n): ").strip().lower()
                if retry != 'y':
                    print(f"[INFO] Skipping camera {cam_idx}")
                    break
                continue
    
    # Save updated extrinsics
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")
    
    # Update YAML
    updated_yaml = []
    for i, cam_info in enumerate(extrinsics_yaml):
        cam_dict = copy.deepcopy(cam_info)
        cam_dict["transformation"] = updated_extrinsics[i]
        updated_yaml.append(cam_dict)
    
    # Save updated extrinsics
    extrinsic_dir = extrinsic_file.parent
    extrinsic_basename = extrinsic_file.stem
    out_yaml = extrinsic_dir / f"{extrinsic_basename}_manual_aligned.yaml"
    
    with open(out_yaml, "w") as f:
        yaml.dump(updated_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"[INFO] Updated extrinsics saved to: {out_yaml}")
    
    # Apply transformations and merge point clouds
    print("\n[INFO] Merging transformed point clouds...")
    merged_pc = o3d.geometry.PointCloud()
    
    for cam_idx in sorted(point_clouds.keys()):
        pc = copy.deepcopy(point_clouds[cam_idx])
        if cam_idx in transformations:
            pc.transform(transformations[cam_idx])
        
        # Color by camera for visualization
        color = np.random.rand(3)
        pc.paint_uniform_color(color)
        
        merged_pc += pc
    
    # Save merged point cloud
    merged_file = out_path / "merged_manual_aligned.ply"
    o3d.io.write_point_cloud(str(merged_file), merged_pc)
    print(f"[INFO] Merged point cloud saved to: {merged_file}")
    print(f"[INFO] Total points: {len(merged_pc.points)}")
    
    # Visualize final result
    print("\n[INFO] Displaying final merged point cloud...")
    o3d.visualization.draw_geometries([merged_pc], window_name="Final Merged Point Cloud")
    
    print("\n[SUCCESS] Manual alignment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive manual point cloud alignment using 3-point correspondence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cached_pc_dir",
        type=str,
        required=True,
        help="Directory containing cached point clouds (cached_pc/)"
    )
    parser.add_argument(
        "--extrinsic_file",
        type=str,
        required=True,
        help="Path to calibration YAML file"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output directory (default: manual_aligned/ in parent of cached_pc_dir)"
    )
    parser.add_argument(
        "--ref_cam_idx",
        type=int,
        default=0,
        help="Index of reference camera"
    )
    
    args = parser.parse_args()
    
    main(
        cached_pc_dir=args.cached_pc_dir,
        extrinsic_file=args.extrinsic_file,
        out_path=args.out_path,
        ref_cam_idx=args.ref_cam_idx
    )

