import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

import copy
from itertools import combinations
from hocap_annotation.utils import *
from hocap_annotation.loaders import MyClusterLoader as HOCapLoader
from hocap_annotation.wrappers.foundationpose import (
    FoundationPose,
    ScorePredictor,
    PoseRefinePredictor,
    set_logging_format,
    set_seed,
    dr,
)

from pathlib import Path
import cv2
import logging
import numpy as np
import open3d as o3d
import trimesh
import argparse
import time
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

# Thresholds for pose validation
X_THRESHOLD = (-0.3, 0.3)
Y_THRESHOLD = (-0.3, 0.3)
Z_THRESHOLD = (-0.2, 0.4)


def depth_to_pointcloud(depth, mask, K, max_depth=3.0, min_depth=0.1):
    """
    Convert depth image and mask to point cloud in camera coordinates.
    
    Args:
        depth: Depth image (H, W)
        mask: Binary mask (H, W)
        K: Camera intrinsic matrix (3, 3)
        max_depth: Maximum depth value
        min_depth: Minimum depth value
    
    Returns:
        pointcloud: Open3D point cloud
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    
    # Filter by mask and valid depth
    valid = (mask > 0) & (depth >= min_depth) & (depth <= max_depth)
    u_valid = u[valid]
    v_valid = v[valid]
    depth_valid = depth[valid]
    
    if len(depth_valid) == 0:
        return None
    
    # Convert to 3D points
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid
    
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Remove outliers
    if len(points) > 100:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return pcd


def mesh_to_pointcloud(mesh, num_points=10000):
    """
    Convert mesh to point cloud by sampling points on the mesh surface.
    
    Args:
        mesh: Trimesh mesh object
        num_points: Number of points to sample
    
    Returns:
        pointcloud: Open3D point cloud
    """
    # Sample points on mesh surface
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    points = points.astype(np.float32)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Compute normals
    pcd.estimate_normals()
    pcd.normalize_normals()
    
    return pcd


def refine_pose_with_icp(
    initial_pose,
    model_pcd,
    scene_pcd,
    max_correspondence_distance=0.01,
    max_iteration=30,
    relative_fitness=1e-6,
    relative_rmse=1e-6,
):
    """
    Refine pose using ICP (Iterative Closest Point).
    
    Args:
        initial_pose: Initial 4x4 transformation matrix (object to camera)
        model_pcd: Model point cloud (Open3D)
        scene_pcd: Scene point cloud (Open3D)
        max_correspondence_distance: Maximum distance for correspondence
        max_iteration: Maximum ICP iterations
        relative_fitness: Relative fitness threshold
        relative_rmse: Relative RMSE threshold
    
    Returns:
        refined_pose: Refined 4x4 transformation matrix
        fitness: ICP fitness score
        inlier_rmse: Inlier RMSE
    """
    if model_pcd is None or scene_pcd is None:
        return initial_pose, 0.0, float('inf')
    
    if len(model_pcd.points) == 0 or len(scene_pcd.points) == 0:
        return initial_pose, 0.0, float('inf')
    
    # Transform model point cloud to camera space using initial pose
    model_pcd_transformed = copy.deepcopy(model_pcd)
    model_pcd_transformed.transform(initial_pose)
    
    # Perform ICP
    try:
        result = o3d.pipelines.registration.registration_icp(
            model_pcd_transformed,
            scene_pcd,
            max_correspondence_distance,
            np.eye(4),  # Initial transformation (identity since model is already transformed)
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=relative_fitness,
                relative_rmse=relative_rmse,
                max_iteration=max_iteration
            )
        )
        
        # Compose transformations: initial_pose @ result.transformation
        refined_pose = initial_pose @ result.transformation
        
        return refined_pose, result.fitness, result.inlier_rmse
    except Exception as e:
        print(f"[WARNING] ICP failed: {e}, using initial pose")
        return initial_pose, 0.0, float('inf')


def refine_pose_with_icp_point_to_plane(
    initial_pose,
    model_pcd,
    scene_pcd,
    max_correspondence_distance=0.01,
    max_iteration=30,
    relative_fitness=1e-6,
    relative_rmse=1e-6,
):
    """
    Refine pose using ICP with point-to-plane distance (more accurate but requires normals).
    
    Args:
        initial_pose: Initial 4x4 transformation matrix (object to camera)
        model_pcd: Model point cloud with normals (Open3D)
        scene_pcd: Scene point cloud with normals (Open3D)
        max_correspondence_distance: Maximum distance for correspondence
        max_iteration: Maximum ICP iterations
        relative_fitness: Relative fitness threshold
        relative_rmse: Relative RMSE threshold
    
    Returns:
        refined_pose: Refined 4x4 transformation matrix
        fitness: ICP fitness score
        inlier_rmse: Inlier RMSE
    """
    if model_pcd is None or scene_pcd is None:
        return initial_pose, 0.0, float('inf')
    
    if len(model_pcd.points) == 0 or len(scene_pcd.points) == 0:
        return initial_pose, 0.0, float('inf')
    
    # Ensure normals are computed
    if not model_pcd.has_normals():
        model_pcd.estimate_normals()
        model_pcd.normalize_normals()
    if not scene_pcd.has_normals():
        scene_pcd.estimate_normals()
        scene_pcd.normalize_normals()
    
    # Transform model point cloud to camera space using initial pose
    model_pcd_transformed = copy.deepcopy(model_pcd)
    model_pcd_transformed.transform(initial_pose)
    
    # Perform ICP with point-to-plane
    try:
        result = o3d.pipelines.registration.registration_icp(
            model_pcd_transformed,
            scene_pcd,
            max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=relative_fitness,
                relative_rmse=relative_rmse,
                max_iteration=max_iteration
            )
        )
        
        # Compose transformations
        refined_pose = initial_pose @ result.transformation
        
        return refined_pose, result.fitness, result.inlier_rmse
    except Exception as e:
        print(f"[WARNING] ICP point-to-plane failed: {e}, using initial pose")
        return initial_pose, 0.0, float('inf')


# Import helper functions from original script
def slerp(q1, q2, t):
    """Spherical Linear Interpolation (SLERP) between two quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot_product = np.dot(q1, q2)
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product
    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta_0 = np.arccos(dot_product)
    sin_theta_0 = np.sin(theta_0)
    if sin_theta_0 < 1e-6:
        return (1 - t) * q1 + t * q2
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    s1 = np.sin(theta_0 - theta_t) / sin_theta_0
    s2 = sin_theta_t / sin_theta_0
    return s1 * q1 + s2 * q2


def predict_current_rotation(prev_quats, prev_flags):
    """Predict the current frame rotation based on previous quaternions."""
    valid_quats = []
    weights = []
    for i, (q, flag) in enumerate(zip(prev_quats, prev_flags)):
        if flag == 1:
            valid_quats.append(q)
            weights.append(1 / (len(prev_quats) - i))
    if len(valid_quats) == 0:
        return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    if len(valid_quats) == 1:
        return valid_quats[0]
    weights = np.array(weights) / np.sum(weights)
    weighted_quat = np.zeros(4)
    for q, w in zip(valid_quats, weights):
        weighted_quat += w * np.array(q)
    weighted_quat /= np.linalg.norm(weighted_quat)
    most_recent_valid_quat = valid_quats[-1]
    predicted_quat = slerp(weighted_quat, most_recent_valid_quat, t=0.5)
    return predicted_quat


def predict_current_position(prev_positions, prev_flags):
    """Predict the current frame position using Cubic Spline Interpolation."""
    valid_positions = []
    valid_times = []
    for i, (pos, flag) in enumerate(zip(prev_positions, prev_flags)):
        if flag == 1:
            valid_positions.append(pos)
            valid_times.append(i)
    if len(valid_positions) == 0:
        return np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    if len(valid_positions) == 1:
        return np.array(valid_positions[0])
    valid_positions = np.array(valid_positions)
    x_coords = valid_positions[:, 0]
    y_coords = valid_positions[:, 1]
    z_coords = valid_positions[:, 2]
    spline_x = CubicSpline(valid_times, x_coords)
    spline_y = CubicSpline(valid_times, y_coords)
    spline_z = CubicSpline(valid_times, z_coords)
    t_current = len(prev_positions)
    x_pred = spline_x(t_current)
    y_pred = spline_y(t_current)
    z_pred = spline_z(t_current)
    return np.array([x_pred, y_pred, z_pred], dtype=np.float32)


def calculate_pairwise_distances(poses):
    """Calculate pairwise distances for rotations and translations."""
    num_poses = len(poses)
    rot_dists = []
    trans_dists = []
    pairwise_indices = []
    for i in range(num_poses):
        for j in range(i + 1, num_poses):
            q1, q2 = poses[i][:4], poses[j][:4]
            t1, t2 = poses[i][4:], poses[j][4:]
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)
            dot_product = np.dot(q1, q2)
            theta = 2 * np.arccos(np.clip(abs(dot_product), -1, 1))
            rot_dists.append(theta)
            trans_dists.append(np.linalg.norm(np.array(t1) - np.array(t2)))
            pairwise_indices.append((i, j))
    rot_dists = np.array(rot_dists, dtype=np.float32)
    trans_dists = np.array(trans_dists, dtype=np.float32)
    return rot_dists, trans_dists, pairwise_indices


def analyze_distances(distances, threshold_factor=2.0, outlier_ratio=0.2):
    """Analyze distance distribution to identify noise and inliers."""
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + threshold_factor * std_dist
    inlier_indices = np.where(distances <= threshold)[0]
    outlier_fraction = 1 - (len(inlier_indices) / len(distances))
    is_noisy = outlier_fraction > outlier_ratio
    return is_noisy, inlier_indices


def detect_pose_outliers(poses, threshold_factor=2.0, outlier_ratio=0.2):
    """Detect outlier rotations and translations, and return inliers."""
    rot_dists, trans_dists, pairwise_indices = calculate_pairwise_distances(poses)
    is_rot_noisy, rot_inlier_dist_indices = analyze_distances(rot_dists, threshold_factor, outlier_ratio)
    is_trans_noisy, trans_inlier_dist_indices = analyze_distances(trans_dists, threshold_factor, outlier_ratio)
    rot_inlier_indices = set(
        idx for pair_idx in rot_inlier_dist_indices for idx in pairwise_indices[pair_idx]
    )
    trans_inlier_indices = set(
        idx for pair_idx in trans_inlier_dist_indices for idx in pairwise_indices[pair_idx]
    )
    rot_inlier_indices = sorted(rot_inlier_indices)
    trans_inlier_indices = sorted(trans_inlier_indices)
    inlier_rots = [poses[i][:4] for i in rot_inlier_indices]
    inlier_trans = [poses[i][4:] for i in trans_inlier_indices]
    return inlier_rots, inlier_trans, is_rot_noisy, is_trans_noisy


def is_valid_pose(pose_w, x_threshold, y_threshold, z_threshold):
    """Check if pose in world space is valid."""
    x, y, z = pose_w[-3:]
    return (x_threshold[0] < x < x_threshold[1] and
            y_threshold[0] < y < y_threshold[1] and
            z_threshold[0] < z < z_threshold[1])


def transform_poses_to_world(mat_poses_c, cam_RTs, x_threshold, y_threshold, z_threshold):
    """Transform poses from camera to world coordinates and filter valid ones."""
    poses_w = []
    for mat_pose, cam_RT in zip(mat_poses_c, cam_RTs):
        if np.all(mat_pose == -1):
            continue
        mat_pose_w = cam_RT @ mat_pose
        quat_pose_w = mat_to_quat(mat_pose_w)
        if is_valid_pose(quat_pose_w, x_threshold, y_threshold, z_threshold):
            poses_w.append(quat_pose_w)
    return poses_w


def ransac_consistent_rotation(inlier_rots, threshold):
    """Estimate the consistent rotation using RANSAC on inlier rotations."""
    if len(inlier_rots) == 1:
        return inlier_rots[0]
    best_rotation = None
    max_inliers = 0
    for r in range(1, len(inlier_rots) + 1):
        for comb in combinations(inlier_rots, r):
            candidate = np.mean(comb, axis=0)
            candidate /= np.linalg.norm(candidate)
            inlier_count = 0
            for rot in inlier_rots:
                dot_product = np.dot(candidate, rot)
                loss = 2 * np.arccos(np.clip(abs(dot_product), -1, 1))
                if loss <= threshold:
                    inlier_count += 1
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_rotation = candidate
    return best_rotation


def ransac_consistent_translation(inlier_trans, threshold):
    """Estimate the consistent translation using RANSAC on inlier translations."""
    if len(inlier_trans) == 1:
        return inlier_trans[0]
    best_translation = None
    max_inliers = 0
    for r in range(1, len(inlier_trans) + 1):
        for comb in combinations(inlier_trans, r):
            candidate = np.mean(comb, axis=0)
            inlier_count = 0
            for trans in inlier_trans:
                loss = np.linalg.norm(candidate - trans)
                if loss <= threshold:
                    inlier_count += 1
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_translation = candidate
    return best_translation


def is_valid_ob_pose(ob_in_cam, x_threshold, y_threshold, z_threshold, cam_RT=None):
    """Check if object pose in camera is valid."""
    if np.all(ob_in_cam == -1):
        return False
    elif cam_RT is None:
        x, y, z = ob_in_cam[:3, 3]
    else:
        ob_in_world = cam_RT @ ob_in_cam
        x, y, z = ob_in_world[:3, 3]
    return (x_threshold[0] < x < x_threshold[1] and
            y_threshold[0] < y < y_threshold[1] and
            z_threshold[0] < z < z_threshold[1])


def get_consistent_pose_w(
    mat_poses_c,
    cam_RTs,
    prev_poses_w,
    rot_thresh=5.0,
    trans_thresh=0.01,
    thresh_factor=2.0,
    outlier_ratio=0.2,
    x_threshold=(-0.3, 0.3),
    y_threshold=(-0.3, 0.3),
    z_threshold=(-0.2, 0.4),
):
    """Get consistent pose in world space using RANSAC on inlier rotations and translations."""
    rot_thresh = np.deg2rad(rot_thresh)
    curr_rot = None
    curr_trans = None
    flag = 1
    
    poses_w = transform_poses_to_world(mat_poses_c, cam_RTs, x_threshold, y_threshold, z_threshold)
    
    if len(poses_w) < 3:
        curr_rot = predict_current_rotation(
            [pose[:4] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        curr_trans = predict_current_position(
            [pose[4:7] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
        return np.concatenate([curr_rot, curr_trans, [flag]], axis=0)
    
    poses_w = np.stack(poses_w, axis=0)
    
    inlier_rots, inlier_trans, is_rot_noisy, is_trans_noisy = detect_pose_outliers(
        poses_w, thresh_factor, outlier_ratio
    )
    
    if is_rot_noisy and is_trans_noisy:
        curr_rot = predict_current_rotation(
            [pose[:4] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        curr_trans = predict_current_position(
            [pose[4:7] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
    elif is_rot_noisy:
        curr_rot = predict_current_rotation(
            [pose[:4] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        curr_trans = ransac_consistent_translation(inlier_trans, trans_thresh)
        flag = 0
    elif is_trans_noisy:
        curr_rot = ransac_consistent_rotation(inlier_rots, rot_thresh)
        curr_trans = predict_current_position(
            [pose[4:7] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
    else:
        curr_rot = ransac_consistent_rotation(inlier_rots, rot_thresh)
        curr_trans = ransac_consistent_translation(inlier_trans, trans_thresh)
    
    if curr_rot is None:
        curr_rot = predict_current_rotation(
            [pose[:4] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
    if curr_trans is None:
        curr_trans = predict_current_position(
            [pose[4:7] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
    
    return np.concatenate([curr_rot, curr_trans, [flag]], axis=0)


def run_pose_estimation(
    sequence_folder,
    object_idx,
    est_refine_iter,
    track_refine_iter,
    start_frame,
    end_frame,
    rot_thresh,
    trans_thresh,
    use_icp=True,
    icp_after_register=True,
    icp_after_track=True,
    icp_after_world_merge=False,
    icp_max_distance=0.01,
    icp_max_iterations=30,
    icp_use_point_to_plane=False,
):
    """
    Run pose estimation with ICP refinement at key steps.
    
    Args:
        sequence_folder: Path to sequence folder
        object_idx: Object index (1-based)
        est_refine_iter: Iterations for registration refinement
        track_refine_iter: Iterations for tracking refinement
        start_frame: Start frame index
        end_frame: End frame index
        rot_thresh: Rotation threshold for RANSAC
        trans_thresh: Translation threshold for RANSAC
        use_icp: Whether to use ICP refinement
        icp_after_register: Apply ICP after register
        icp_after_track: Apply ICP after track_one
        icp_after_world_merge: Apply ICP after world coordinate merge
        icp_max_distance: Maximum correspondence distance for ICP
        icp_max_iterations: Maximum ICP iterations
        icp_use_point_to_plane: Use point-to-plane ICP (requires normals)
    """
    sequence_folder = Path(sequence_folder)
    object_idx = object_idx - 1  # 0-based index

    # Load parameters from data_loader
    data_loader = HOCapLoader(sequence_folder)
    rs_width = data_loader.rs_width
    rs_height = data_loader.rs_height
    num_frames = data_loader.num_frames
    object_id = data_loader.object_ids[object_idx]
    rs_serials = data_loader.rs_serials
    cam_Ks = data_loader.rs_Ks
    cam_RTs = data_loader.extr2world
    valid_serials = data_loader.get_valid_seg_serials()
    valid_serial_indices = [rs_serials.index(serial) for serial in valid_serials]
    valid_Ks = data_loader.rs_Ks[valid_serial_indices]
    valid_RTs = data_loader.extr2world[valid_serial_indices]
    valid_RTs_inv = data_loader.extr2world_inv[valid_serial_indices]
    object_mesh_textured = trimesh.load(data_loader.object_textured_files[object_idx])
    object_mesh_cleaned = trimesh.load(data_loader.object_cleaned_files[object_idx])
    empty_mat_pose = np.full((4, 4), -1.0, dtype=np.float32)

    x_threshold = data_loader._thresholds[:2]
    y_threshold = data_loader._thresholds[2:4]
    z_threshold = data_loader._thresholds[4:]
    print(f"[DEBUG] x_threshold: {x_threshold}, y_threshold: {y_threshold}, z_threshold: {z_threshold}")

    # Process mesh
    other_mesh = trimesh.load(data_loader.object_cleaned_files[object_idx], process=True)
    USE_TEXTURE = False
    
    if len(other_mesh.vertices) > 200000:
        mesh = other_mesh.simplify_quadric_decimation(0.8)
        print("Decim mesh.")
        del other_mesh
    else:
        mesh = copy.deepcopy(other_mesh)

    print(f"[DEBUG] Mesh vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")

    # Check start and end frame_idx
    start_frame = max(start_frame, 0)
    end_frame = num_frames if end_frame < start_frame else end_frame

    logging.info(f"start_frame: {start_frame}, end_frame: {end_frame}")
    print(f"[DEBUG] start_frame: {start_frame}, end_frame: {end_frame}")

    save_folder = Path(f"{data_loader._data_folder.parent.parent.parent}/{data_loader._folder_name}_annotated/{data_loader._task_name}/{data_loader._sequence_name}/processed/fd_pose_solver_icp")
    save_folder.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.WARNING)
    set_seed(0)
    debug = 0

    # Initialize FoundationPose estimator
    estimator = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(),
        debug=debug,
        debug_dir=save_folder / "debug" / object_id,
        rotation_grid_min_n_views=120,
        rotation_grid_inplane_step=60,
    )

    # Pre-compute model point cloud for ICP
    model_pcd = None
    if use_icp:
        print("[INFO] Pre-computing model point cloud for ICP...")
        model_pcd = mesh_to_pointcloud(mesh, num_points=min(10000, len(mesh.vertices)))
        print(f"[INFO] Model point cloud: {len(model_pcd.points)} points")

    # Initialize poses
    ob_in_world_refined = empty_mat_pose.copy()
    ob_in_cam_poses = [empty_mat_pose.copy()] * len(valid_serials)
    all_poses_w = []
    print(f"[DEBUG] valid_serials: {valid_serials}")

    # Configuration flags
    REVERSE = False
    MASKED_DEPTH = True
    MASKED_IMAGE = False
    CROP_VIEW = False
    MASKED_OBJECT = True

    for frame_id in range(start_frame, end_frame, 1):
        frame_idx = frame_id
        for serial_idx, serial in enumerate(valid_serials):
            if not REVERSE:
                color = data_loader.get_color(serial, frame_id)
                depth = data_loader.get_depth(serial, frame_id)
                mask = data_loader.get_mask(serial, frame_id, object_idx)
                frame_idx = frame_id
            else:
                color = data_loader.get_color(serial, num_frames - frame_id - 1)
                depth = data_loader.get_depth(serial, num_frames - frame_id - 1)
                mask = data_loader.get_mask(serial, num_frames - frame_id - 1, object_idx)
                frame_idx = num_frames - frame_id - 1
            
            # Apply mask to depth
            if MASKED_DEPTH:
                depth = depth.copy()
                depth[mask == 0] = 0

            if MASKED_IMAGE:
                color = color.copy()
                color[mask == 0] = 0

            K = valid_Ks[serial_idx]

            if mask.sum() < 10:
                ob_in_cam_mat = empty_mat_pose.copy()
                print(f"[DEBUG] Frame {frame_idx}, Cam {serial}: mask.sum() = {mask.sum()} is less than 10, skipping.")
            elif is_valid_ob_pose(ob_in_world_refined, x_threshold, y_threshold, z_threshold):
                # Track using refined world pose
                ob_in_cam_mat = estimator.track_one(
                    rgb=color,
                    depth=depth,
                    K=K,
                    iteration=track_refine_iter,
                    prev_pose=valid_RTs_inv[serial_idx] @ ob_in_world_refined,
                )
                
                # ICP refinement after track_one
                if use_icp and icp_after_track and not np.all(ob_in_cam_mat == -1):
                    scene_pcd = depth_to_pointcloud(depth, mask, K)
                    if scene_pcd is not None and len(scene_pcd.points) > 100:
                        if icp_use_point_to_plane:
                            ob_in_cam_mat, fitness, rmse = refine_pose_with_icp_point_to_plane(
                                ob_in_cam_mat, model_pcd, scene_pcd,
                                max_correspondence_distance=icp_max_distance,
                                max_iteration=icp_max_iterations
                            )
                        else:
                            ob_in_cam_mat, fitness, rmse = refine_pose_with_icp(
                                ob_in_cam_mat, model_pcd, scene_pcd,
                                max_correspondence_distance=icp_max_distance,
                                max_iteration=icp_max_iterations
                            )
                        if frame_idx % 10 == 0:
                            print(f"[ICP] Frame {frame_idx}, Cam {serial} after track: fitness={fitness:.4f}, rmse={rmse:.6f}")
                
            elif is_valid_ob_pose(ob_in_cam_poses[serial_idx], x_threshold, y_threshold, z_threshold, valid_RTs[serial_idx]):
                # Track using previous camera pose
                ob_in_cam_mat = estimator.track_one(
                    rgb=color,
                    depth=depth,
                    K=K,
                    iteration=track_refine_iter,
                    prev_pose=ob_in_cam_poses[serial_idx],
                )
                
                # ICP refinement after track_one
                if use_icp and icp_after_track and not np.all(ob_in_cam_mat == -1):
                    scene_pcd = depth_to_pointcloud(depth, mask, K)
                    if scene_pcd is not None and len(scene_pcd.points) > 100:
                        if icp_use_point_to_plane:
                            ob_in_cam_mat, fitness, rmse = refine_pose_with_icp_point_to_plane(
                                ob_in_cam_mat, model_pcd, scene_pcd,
                                max_correspondence_distance=icp_max_distance,
                                max_iteration=icp_max_iterations
                            )
                        else:
                            ob_in_cam_mat, fitness, rmse = refine_pose_with_icp(
                                ob_in_cam_mat, model_pcd, scene_pcd,
                                max_correspondence_distance=icp_max_distance,
                                max_iteration=icp_max_iterations
                            )
                        if frame_idx % 10 == 0:
                            print(f"[ICP] Frame {frame_idx}, Cam {serial} after track: fitness={fitness:.4f}, rmse={rmse:.6f}")
            else:
                # Register new pose
                init_ob_pos_center = data_loader.get_init_translation(
                    frame_idx, [serial], object_idx, kernel_size=5
                )[0][0]

                if init_ob_pos_center is not None:
                    print(f"[DEBUG] Frame {frame_idx}, Cam {serial}: init_ob_pos_center = {init_ob_pos_center}")
                    ob_in_cam_mat = estimator.register(
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        K=K,
                        iteration=est_refine_iter,
                        init_ob_pos_center=init_ob_pos_center,
                    )
                    
                    # ICP refinement after register
                    if use_icp and icp_after_register and not np.all(ob_in_cam_mat == -1):
                        scene_pcd = depth_to_pointcloud(depth, mask, K)
                        if scene_pcd is not None and len(scene_pcd.points) > 100:
                            if icp_use_point_to_plane:
                                ob_in_cam_mat, fitness, rmse = refine_pose_with_icp_point_to_plane(
                                    ob_in_cam_mat, model_pcd, scene_pcd,
                                    max_correspondence_distance=icp_max_distance,
                                    max_iteration=icp_max_iterations
                                )
                            else:
                                ob_in_cam_mat, fitness, rmse = refine_pose_with_icp(
                                    ob_in_cam_mat, model_pcd, scene_pcd,
                                    max_correspondence_distance=icp_max_distance,
                                    max_iteration=icp_max_iterations
                                )
                            print(f"[ICP] Frame {frame_idx}, Cam {serial} after register: fitness={fitness:.4f}, rmse={rmse:.6f}")
                    
                    if not is_valid_ob_pose(ob_in_cam_mat, x_threshold, y_threshold, z_threshold, valid_RTs[serial_idx]):
                        print(f"[DEBUG]!!! Frame {frame_idx}, Cam {serial}: Register failed! using empty pose.")
                        debug_ob_in_world = valid_RTs[serial_idx] @ ob_in_cam_mat
                        print(debug_ob_in_world[:3,3])
                        ob_in_cam_mat = empty_mat_pose.copy()
                else:
                    print(f"[DEBUG] Frame {frame_idx}, Cam {serial}: init_ob_pos_center is None, using empty pose.")
                    ob_in_cam_mat = empty_mat_pose.copy()

            ob_in_cam_poses[serial_idx] = ob_in_cam_mat

            # Save pose to file
            save_pose_folder = save_folder / object_id / "ob_in_cam" / serial
            save_pose_folder.mkdir(parents=True, exist_ok=True)
            write_pose_to_txt(
                save_pose_folder / f"{frame_idx:06d}.txt", mat_to_quat(ob_in_cam_mat)
            )

        # Refine object pose in world coordinate system
        curr_pose_w = get_consistent_pose_w(
            mat_poses_c=ob_in_cam_poses,
            cam_RTs=valid_RTs,
            prev_poses_w=all_poses_w,
            rot_thresh=rot_thresh,
            trans_thresh=trans_thresh,
            thresh_factor=2.0,
            outlier_ratio=0.2,
            x_threshold=x_threshold,
            y_threshold=y_threshold,
            z_threshold=z_threshold,
        )

        # Optional: ICP refinement after world coordinate merge
        if use_icp and icp_after_world_merge and curr_pose_w[-1] == 1:  # Only if pose is valid
            # Collect point clouds from all valid cameras
            all_scene_pcds = []
            for serial_idx, serial in enumerate(valid_serials):
                if not np.all(ob_in_cam_poses[serial_idx] == -1):
                    color = data_loader.get_color(serial, frame_idx)
                    depth = data_loader.get_depth(serial, frame_idx)
                    mask = data_loader.get_mask(serial, frame_idx, object_idx)
                    if MASKED_DEPTH:
                        depth = depth.copy()
                        depth[mask == 0] = 0
                    K = valid_Ks[serial_idx]
                    scene_pcd = depth_to_pointcloud(depth, mask, K)
                    if scene_pcd is not None and len(scene_pcd.points) > 100:
                        # Transform to world coordinates (camera to world)
                        cam2world = valid_RTs[serial_idx]
                        scene_pcd.transform(cam2world)
                        all_scene_pcds.append(scene_pcd)
            
            if len(all_scene_pcds) > 0:
                # Merge all scene point clouds
                merged_scene_pcd = all_scene_pcds[0]
                for pcd in all_scene_pcds[1:]:
                    merged_scene_pcd += pcd
                
                # Downsample merged point cloud if too large
                if len(merged_scene_pcd.points) > 50000:
                    merged_scene_pcd = merged_scene_pcd.voxel_down_sample(voxel_size=0.005)
                
                # Transform model to world coordinates
                pose_w_mat = quat_to_mat(curr_pose_w[:7])
                model_pcd_world = copy.deepcopy(model_pcd)
                model_pcd_world.transform(pose_w_mat)
                
                # Perform ICP in world coordinates
                if icp_use_point_to_plane:
                    refined_pose_w, fitness, rmse = refine_pose_with_icp_point_to_plane(
                        np.eye(4), model_pcd_world, merged_scene_pcd,
                        max_correspondence_distance=icp_max_distance,
                        max_iteration=icp_max_iterations
                    )
                else:
                    refined_pose_w, fitness, rmse = refine_pose_with_icp(
                        np.eye(4), model_pcd_world, merged_scene_pcd,
                        max_correspondence_distance=icp_max_distance,
                        max_iteration=icp_max_iterations
                    )
                
                # Compose with original pose
                refined_pose_w = pose_w_mat @ refined_pose_w
                curr_pose_w = np.concatenate([mat_to_quat(refined_pose_w), [curr_pose_w[-1]]])
                
                if frame_idx % 10 == 0:
                    print(f"[ICP] Frame {frame_idx} after world merge: fitness={fitness:.4f}, rmse={rmse:.6f}")

        all_poses_w.append(curr_pose_w)
        print(f"[RESULT] ob_in_world (Frame {frame_idx}): {curr_pose_w[4:7]}")

        # Save pose to file
        save_pose_folder = save_folder / object_id / "ob_in_world"
        save_pose_folder.mkdir(parents=True, exist_ok=True)
        write_pose_to_txt(save_pose_folder / f"{frame_idx:06d}.txt", curr_pose_w)

        ob_in_world_refined = quat_to_mat(curr_pose_w[:7])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder", type=str, default=None, help="Path to the sequence folder."
    )
    parser.add_argument(
        "--object_idx",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="object index",
    )
    parser.add_argument(
        "--est_refine_iter",
        type=int,
        default=10,
        help="number of iterations for estimation",
    )
    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=10,
        help="number of iterations for tracking",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="start frame")
    parser.add_argument("--end_frame", type=int, default=-1, help="end frame")
    parser.add_argument(
        "--rot_thresh",
        type=float,
        default=2.0,
        help="rotation threshold, degree",
    )
    parser.add_argument(
        "--trans_thresh",
        type=float,
        default=0.03,
        help="translation threshold, meters",
    )
    parser.add_argument(
        "--use_icp",
        action="store_true",
        default=True,
        help="Whether to use ICP refinement",
    )
    parser.add_argument(
        "--icp_after_register",
        action="store_true",
        default=True,
        help="Apply ICP after register",
    )
    parser.add_argument(
        "--icp_after_track",
        action="store_true",
        default=True,
        help="Apply ICP after track_one",
    )
    parser.add_argument(
        "--icp_after_world_merge",
        action="store_true",
        default=False,
        help="Apply ICP after world coordinate merge",
    )
    parser.add_argument(
        "--icp_max_distance",
        type=float,
        default=0.01,
        help="Maximum correspondence distance for ICP (meters)",
    )
    parser.add_argument(
        "--icp_max_iterations",
        type=int,
        default=30,
        help="Maximum ICP iterations",
    )
    parser.add_argument(
        "--icp_use_point_to_plane",
        action="store_true",
        default=False,
        help="Use point-to-plane ICP (requires normals, more accurate)",
    )
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please specify the sequence folder.")
    if args.object_idx is None:
        raise ValueError("Please specify the object index.")

    set_logging_format()
    t_start = time.time()
    logger = logging.getLogger("register")
    logger.setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)

    run_pose_estimation(
        args.sequence_folder,
        args.object_idx,
        args.est_refine_iter,
        args.track_refine_iter,
        args.start_frame,
        args.end_frame,
        args.rot_thresh,
        args.trans_thresh,
        use_icp=args.use_icp,
        icp_after_register=args.icp_after_register,
        icp_after_track=args.icp_after_track,
        icp_after_world_merge=args.icp_after_world_merge,
        icp_max_distance=args.icp_max_distance,
        icp_max_iterations=args.icp_max_iterations,
        icp_use_point_to_plane=args.icp_use_point_to_plane,
    )

    print(f"done!!! time: {time.time() - t_start:.3f}s.")

