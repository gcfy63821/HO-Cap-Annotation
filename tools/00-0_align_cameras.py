import h5py
import yaml
import trimesh
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import copy
import math
from sklearn.neighbors import NearestNeighbors

REF_CENTER = np.array([0, 0, 0], dtype=np.float64)
CROP_RADIUS = 0.5

# Default world coordinate thresholds (can be overridden via command line)
# These define the valid range for points in world coordinates
DEFAULT_X_THRESHOLD = (-0.5, 0.4)
DEFAULT_Y_THRESHOLD = (-0.5, 0.4)
DEFAULT_Z_THRESHOLD = (-0.2, 0.6)


# ========== Point Cloud Utility Functions ==========

def filter_pointcloud_by_world_bounds(pc, x_threshold, y_threshold, z_threshold):
    """
    Filter point cloud to keep only points within the specified world coordinate bounds.
    
    Args:
        pc: Open3D point cloud (already in world coordinates)
        x_threshold: Tuple (min, max) for X coordinate
        y_threshold: Tuple (min, max) for Y coordinate
        z_threshold: Tuple (min, max) for Z coordinate
    
    Returns:
        Filtered Open3D point cloud
    """
    points = np.asarray(pc.points)
    
    # Create mask for points within bounds
    x_mask = (points[:, 0] > x_threshold[0]) & (points[:, 0] < x_threshold[1])
    y_mask = (points[:, 1] > y_threshold[0]) & (points[:, 1] < y_threshold[1])
    z_mask = (points[:, 2] > z_threshold[0]) & (points[:, 2] < z_threshold[1])
    valid_mask = x_mask & y_mask & z_mask
    
    # Filter point cloud
    filtered_pc = pc.select_by_index(np.where(valid_mask)[0])
    
    return filtered_pc

def distance_map_to_point_cloud(distances, fov, width, height):
    """
    Converts from a depth map to a point cloud.
    
    Args:
        distances: An numpy array which has the shape of (height, width) that
            denotes a distance map. The unit is meter.
        fov: The field of view of the camera in the vertical direction. The unit
            is radian.
        width: The width of the image resolution of the camera.
        height: The height of the image resolution of the camera.
    Returns:
        point_cloud: The converted point cloud from the distance map. It is a numpy
            array of shape (height, width, 3).
    """
    f = height / (2 * math.tan(fov / 2.0))
    px = np.tile(np.arange(width), [height, 1])
    x = (2 * (px + 0.5) - width) / f * distances / 2
    py = np.tile(np.arange(height), [width, 1]).T
    y = (2 * (py + 0.5) - height) / f * distances / 2
    point_cloud = np.stack((x, y, distances), axis=-1)
    return point_cloud


def estimate_normals(points, k_neighbors=10):
    """
    Estimate normals for a point cloud using nearest neighbors.
    
    Args:
        points: Nx3 array of 3D points
        k_neighbors: Number of neighbors to consider for normal estimation
    Returns:
        Nx3 array of normals
    """
    # Remove invalid points (e.g., zeros or NaNs)
    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isclose(points, 0).all(axis=1)
    valid_points = points[valid_mask]
    
    if len(valid_points) == 0:
        return np.zeros_like(points)
    
    # Use nearest neighbors to find the k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(valid_points)), algorithm='auto').fit(valid_points)
    _, indices = nbrs.kneighbors(valid_points)
    
    # Estimate normals using PCA on the neighbors
    normals = np.zeros_like(valid_points)
    for i, neighbors in enumerate(indices):
        neighbor_points = valid_points[neighbors]
        if len(neighbor_points) > 1:
            # Center the points
            centered = neighbor_points - neighbor_points.mean(axis=0)
            # Compute covariance matrix
            cov_matrix = np.cov(centered.T)
            # Get eigenvector corresponding to smallest eigenvalue
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            normals[i] = eigvecs[:, 0]  # Smallest eigenvalue
    
    # Normalize normals to unit length
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normals = normals / norms
    
    # Fill in normals for all points
    full_normals = np.zeros_like(points)
    full_normals[valid_mask] = normals
    
    return full_normals


def convert_rgbd_img_to_pc_with_normals(rgb, depth, intrinsic, extrinsic,
                                        raw_depth_scale, raw_depth_trunc,
                                        tf_base_world=None, k_neighbors=10):
    """
    Convert an RGB-D image to a point cloud with normals.
    
    Args:
        rgb: RGB image (H, W, 3), values in [0, 1]
        depth: Depth image (H, W), values in raw units (will be scaled by raw_depth_scale)
        intrinsic: Camera intrinsic matrix (3x3) - can be list or numpy array
        extrinsic: Camera extrinsic matrix (4x4, tf_world_cam) - can be list or numpy array
        raw_depth_scale: Scale factor for depth values (typically 1000.0 for mm to m, or 1.0 if already in meters)
        raw_depth_trunc: Truncation value for depth in meters
        tf_base_world: Transformation to a base frame (optional)
        k_neighbors: Number of neighbors to consider for normal estimation
    Returns:
        Point cloud with xyz, rgb, and normals (N, 9)
    """
    # Convert inputs to numpy arrays if needed
    intrinsic = np.array(intrinsic, dtype=np.float32)
    extrinsic = np.array(extrinsic, dtype=np.float32)
    
    # Preprocess depth: convert from raw units to meters
    depth = depth.astype(np.float32) / raw_depth_scale
    depth = np.clip(depth, 0, raw_depth_trunc)
    
    img_size = rgb.shape[:2]
    
    # Compute field of view (fov) from the intrinsic
    # fov = 2 * arctan(height / (2 * fy))
    fy = intrinsic[1][1]  # Use fy instead of fx for vertical FOV
    fov = 2 * np.arctan(img_size[0] / (2 * fy))
    
    # Convert depth to point cloud
    xyz = distance_map_to_point_cloud(depth, fov, img_size[1], img_size[0])
    xyz = xyz.reshape(-1, 3)
    
    # Transform to world coordinates
    if tf_base_world is None:
        tf = extrinsic
    else:
        tf = np.array(tf_base_world, dtype=np.float32) @ extrinsic
    
    xyz = trimesh.transform_points(xyz, tf)
    
    # Reshape RGB
    rgb = rgb.reshape(-1, 3)
    
    # Estimate normals (only for valid points)
    valid_mask = (depth.flatten() > 0.01) & (depth.flatten() < raw_depth_trunc)
    if valid_mask.sum() > 0:
        normals = estimate_normals(xyz, k_neighbors)
    else:
        normals = np.zeros_like(xyz)
    
    # Concatenate xyz, rgb, normals
    xyzrgb_normals = np.concatenate([xyz, rgb, normals], axis=1)
    
    return xyzrgb_normals


def crop_within_radius(
    pc: o3d.geometry.PointCloud,
    center: np.ndarray = REF_CENTER,
    radius: float = CROP_RADIUS,
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pc.points)
    keep_idx = np.where(np.linalg.norm(pts - center, axis=1) <= radius)[0]
    return pc.select_by_index(keep_idx, invert=False)


def colored_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    iters: int = 20,
    dist_thr: float = 0.1,
) -> np.ndarray:
    reg = o3d.pipelines.registration.registration_colored_icp(
        source,
        target,
        max_correspondence_distance=dist_thr,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=5e-7,
            relative_rmse=5e-7,
            max_iteration=iters,
        ),
    )
    return reg.transformation


def main(h5_file: str, extrinsic_file: str, out_path: str = None, frame_idx: int = 100,
         x_threshold: tuple = None, y_threshold: tuple = None, z_threshold: tuple = None):
    """
    Align cameras using colored ICP.
    
    Args:
        h5_file: Path to HDF5 file containing RGB and depth images
        extrinsic_file: Path to calibration YAML file (supports both old and new formats)
        out_path: Output directory (if None, uses same directory as extrinsic_file)
        frame_idx: Frame index to use for alignment
        x_threshold: Tuple (min, max) for X coordinate filtering in world frame
        y_threshold: Tuple (min, max) for Y coordinate filtering in world frame
        z_threshold: Tuple (min, max) for Z coordinate filtering in world frame
    """
    # Use default thresholds if not provided
    if x_threshold is None:
        x_threshold = DEFAULT_X_THRESHOLD
    if y_threshold is None:
        y_threshold = DEFAULT_Y_THRESHOLD
    if z_threshold is None:
        z_threshold = DEFAULT_Z_THRESHOLD
    # Determine output path
    if out_path is None:
        extrinsic_dir = os.path.dirname(extrinsic_file)
        out_path = extrinsic_dir
    
    os.makedirs(out_path, exist_ok=True)

    with open(extrinsic_file, "r") as f:
        extrinsics_yaml = yaml.safe_load(f)

    # Support both old format (dict with "extrinsics" key) and new format (list)
    if isinstance(extrinsics_yaml, dict) and "extrinsics" in extrinsics_yaml:
        # Old format: {"extrinsics": {...}}
        cam_list = []
        for k, v in extrinsics_yaml["extrinsics"].items():
            if not k.startswith("tag_"):
                cam_list.append(v)
        extrinsics_yaml = cam_list
    elif isinstance(extrinsics_yaml, list):
        # New format: list of camera dicts
        # Sort by camera_id to ensure consistent ordering
        extrinsics_yaml = sorted(extrinsics_yaml, key=lambda x: x.get("camera_id", 0))
    else:
        raise ValueError(f"Unsupported YAML format in {extrinsic_file}")

    camera_intrinsics, camera_depth, camera_extrinsics, used_serials, camera_ids = [], [], [], [], []
    for cam_info in extrinsics_yaml:
        used_serials.append(cam_info["serial_number"])
        camera_extrinsics.append(cam_info["transformation"])
        camera_intrinsics.append(cam_info["color_intrinsic_matrix"])
        camera_depth.append(cam_info["depth_intrinsic_matrix"])
        # Store camera_id if present (for new format)
        camera_ids.append(cam_info.get("camera_id", len(camera_ids)))


    cache_dir = os.path.join(out_path, "cached_pc")
    os.makedirs(cache_dir, exist_ok=True)

    with h5py.File(h5_file, "r") as f:
        # - imgs: (num_frames, num_cams, H, W, 3), dtype=uint8, RGB format
        # - depths: (num_frames, num_cams, H, W), dtype=uint16, values in millimeters
        rgbs = f["imgs"][frame_idx]   # (N_CAMS, H, W, 3), uint8, RGB
        depths = f["depths"][frame_idx]  # (N_CAMS, H, W), uint16, millimeters

    N_CAMERAS = rgbs.shape[0]
    pc_raw_list = []  # list[open3d.geometry.PointCloud]

    for cam_idx in tqdm(range(N_CAMERAS), desc="Loading/creating point clouds"):
        uncropped_cache = os.path.join(cache_dir, f"cam{cam_idx}_uncropped.ply")

        # load from cache if saved
        if os.path.exists(uncropped_cache):
            pc = o3d.io.read_point_cloud(uncropped_cache)
            # Apply cropping even for cached point clouds
            pc = filter_pointcloud_by_world_bounds(
                pc, 
                x_threshold=x_threshold,
                y_threshold=y_threshold,
                z_threshold=z_threshold
            )
        else:
            rgb = rgbs[cam_idx].astype(np.float32) / 255.0
            
            depth = depths[cam_idx].astype(np.float32)
            
            intrinsic = np.array(camera_intrinsics[cam_idx], dtype=np.float32)
            extrinsic = np.array(camera_extrinsics[cam_idx], dtype=np.float32)

            xyzrgbnor = convert_rgbd_img_to_pc_with_normals(
                rgb, depth, intrinsic, extrinsic, 
                raw_depth_scale=1000.0,  # Convert mm to meters
                raw_depth_trunc=2.0,      # Truncate at 2 meters
            )

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(xyzrgbnor[:, :3])
            pc.colors = o3d.utility.Vector3dVector(xyzrgbnor[:, 3:6])
            pc.normals = o3d.utility.Vector3dVector(xyzrgbnor[:, 6:9])

            # Crop point cloud to world coordinate bounds
            # Points are already in world coordinates (transformed by extrinsic)
            # ob_in_world = cam_RT @ ob_in_cam (done inside convert_rgbd_img_to_pc_with_normals)
            pc = filter_pointcloud_by_world_bounds(
                pc, 
                x_threshold=x_threshold,
                y_threshold=y_threshold,
                z_threshold=z_threshold
            )

            o3d.io.write_point_cloud(uncropped_cache, pc)

        pc_raw_list.append(pc)

    prealign_full = o3d.geometry.PointCloud()
    for pc in pc_raw_list:
        prealign_full += pc
    o3d.io.write_point_cloud(os.path.join(out_path, "prealign_full.ply"), prealign_full)

    # Create posts directory for aligned point clouds
    posts_dir = os.path.join(out_path, "posts")
    os.makedirs(posts_dir, exist_ok=True)
    
    # Create directory for reference point clouds at each stage
    ref_pc_dir = os.path.join(out_path, "ref_pc_stages")
    os.makedirs(ref_pc_dir, exist_ok=True)

    T_dict = {i: np.eye(4) for i in range(N_CAMERAS)}
    # old order: 6, [3, 5] 20, [2, 4, 7] 20 , 7 =0, [0, 1] =3

    # previous order: 6 [5, 3] [7, 1] [0,2,4]
    # stage 1
    print("\n[STAGE 1] Starting with camera 6 as reference...")
    ref_pc = pc_raw_list[6].voxel_down_sample(0.0005)
    o3d.io.write_point_cloud(os.path.join(ref_pc_dir, "stage1_initial_ref_cam6.ply"), ref_pc)
    print(f"  [SAVED] Initial reference (cam6): {len(ref_pc.points)} points")
    
    for idx in [5,3]:
        src = pc_raw_list[idx].voxel_down_sample(0.0005)
        T = colored_icp(src, ref_pc, iters=40)
        T_dict[idx] = T
        aligned_src = copy.deepcopy(pc_raw_list[idx])
        aligned_src.transform(T)
        ref_pc += aligned_src  # grow reference cloud
        ref_pc=ref_pc.voxel_down_sample(0.001)
    
    o3d.io.write_point_cloud(os.path.join(ref_pc_dir, "stage1_after_cam5_cam3.ply"), ref_pc)
    print(f"  [SAVED] After Stage 1 (cam6 + cam5 + cam3): {len(ref_pc.points)} points")
    print("Stage 1 complete")
    
    # stage 2
    print("\n[STAGE 2] Adding cameras 7 and 4...")
    # ref_pc = ref_pc.voxel_down_sample(0.001)  # lightly resample for speed
    for idx in [7,4]:
        src = pc_raw_list[idx].voxel_down_sample(0.001)
        iters = 20
        T = colored_icp(src, ref_pc, iters=iters)
        T_dict[idx] = T
        aligned_src = copy.deepcopy(pc_raw_list[idx])
        aligned_src.transform(T)
        ref_pc += aligned_src
        # Save after each camera in stage 2
        o3d.io.write_point_cloud(os.path.join(ref_pc_dir, f"stage2_after_cam{idx}.ply"), ref_pc)
        print(f"  [SAVED] After adding cam{idx}: {len(ref_pc.points)} points")
    
    print("Stage 2 complete")
    
    # stage 3
    print("\n[STAGE 3] Adding cameras 0, 1, and 2...")
    ref_pc = ref_pc.voxel_down_sample(0.001)
    o3d.io.write_point_cloud(os.path.join(ref_pc_dir, "stage3_initial_downsampled.ply"), ref_pc)
    print(f"  [SAVED] Initial downsampled for Stage 3: {len(ref_pc.points)} points")
    
    for idx in [0,1,2]:
        src = pc_raw_list[idx].voxel_down_sample(0.001)
        T = colored_icp(src, ref_pc, iters=40)
        T_dict[idx] = T
        aligned_src = copy.deepcopy(pc_raw_list[idx])
        aligned_src.transform(T)
        ref_pc += aligned_src
        # Save after each camera in stage 3
        o3d.io.write_point_cloud(os.path.join(ref_pc_dir, f"stage3_after_cam{idx}.ply"), ref_pc)
        print(f"  [SAVED] After adding cam{idx}: {len(ref_pc.points)} points")
    
    print("Stage 3 complete")
    # identity for cam‑6 already in T_dict
    
    # Save final reference point cloud
    o3d.io.write_point_cloud(os.path.join(ref_pc_dir, "final_ref_all_cameras.ply"), ref_pc)
    print(f"\n[SAVED] Final reference point cloud (all cameras): {len(ref_pc.points)} points")


    merged_aligned = o3d.geometry.PointCloud()
    for idx, pc in enumerate(pc_raw_list):
        pc_aligned = copy.deepcopy(pc)
        pc_aligned.transform(T_dict[idx])
        merged_aligned += pc_aligned
        o3d.io.write_point_cloud(os.path.join(posts_dir, f"post_{idx}.ply"), pc_aligned)

    o3d.io.write_point_cloud(os.path.join(out_path, "postalign_full.ply"), merged_aligned)

    # Update YAML with aligned extrinsics
    # Maintain the same format as input (with camera_id if present)
    updated_extrinsics = []
    for cam_idx, (cam_id, serial, intrinsic, depth_intr, ext_orig) in enumerate(
        zip(camera_ids, used_serials, camera_intrinsics, camera_depth, camera_extrinsics)
    ):
        ext_orig_np = np.array(ext_orig).reshape(4, 4)
        ext_updated = T_dict[cam_idx] @ ext_orig_np
        
        # Create output dict matching input format
        cam_dict = {
            "camera_id": cam_id,
            "serial_number": serial,
            "transformation": ext_updated.tolist(),
            "color_intrinsic_matrix": intrinsic,
            "depth_intrinsic_matrix": depth_intr,
        }
        updated_extrinsics.append(cam_dict)

    # Save updated extrinsics in the same directory as input file
    extrinsic_dir = os.path.dirname(extrinsic_file)
    extrinsic_basename = os.path.basename(extrinsic_file)
    # Create output filename: add "_aligned" before extension
    name_without_ext = os.path.splitext(extrinsic_basename)[0]
    ext = os.path.splitext(extrinsic_basename)[1]
    out_yaml = os.path.join(extrinsic_dir, f"{name_without_ext}_aligned{ext}")
    
    with open(out_yaml, "w") as f:
        yaml.dump(updated_extrinsics, f, default_flow_style=False, sort_keys=False)

    print(f"\n[INFO] Hierarchical alignment finished!")
    print(f"[INFO] Updated extrinsics saved to: {out_yaml}")
    print(f"[INFO] Aligned point clouds saved to: {out_path}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Align cameras using colored ICP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--h5_file",
        type=str,
        required=True,
        help="Path to HDF5 file containing RGB and depth images"
    )
    parser.add_argument(
        "--extrinsic_file",
        type=str,
        required=True,
        help="Path to calibration YAML file (supports both old and new formats)"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output directory for point clouds (default: same as extrinsic_file directory)"
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=100,
        help="Frame index to use for alignment"
    )
    parser.add_argument(
        "--x_threshold",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="X coordinate bounds in world frame (default: -0.3 0.3)"
    )
    parser.add_argument(
        "--y_threshold",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Y coordinate bounds in world frame (default: -0.3 0.3)"
    )
    parser.add_argument(
        "--z_threshold",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Z coordinate bounds in world frame (default: -0.2 0.4)"
    )
    
    args = parser.parse_args()
    
    # Convert threshold lists to tuples
    x_thresh = tuple(args.x_threshold) if args.x_threshold else None
    y_thresh = tuple(args.y_threshold) if args.y_threshold else None
    z_thresh = tuple(args.z_threshold) if args.z_threshold else None
    
    main(
        args.h5_file, 
        args.extrinsic_file, 
        args.out_path, 
        args.frame_idx,
        x_threshold=x_thresh,
        y_threshold=y_thresh,
        z_threshold=z_thresh
    )
