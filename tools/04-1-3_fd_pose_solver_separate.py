import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

import copy
from itertools import combinations
from hocap_annotation.utils import *
# from hocap_annotation.loaders import HOCapLoader
from hocap_annotation.loaders import MyLoader as HOCapLoader
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

from scipy.spatial.transform import Rotation as R

def geodesic_distance(R1, R2):
    """Compute geodesic distance (in radians) between two 3x3 rotation matrices."""
    return np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))

# -0.2 < x < 0.4 and 0.0 < y < 0.8 and 0.0 < z < 0.8
# X_THRESHOLD = (0.2, 0.4)
# Y_THRESHOLD = (0.4, 0.8)
# Z_THRESHOLD = (0.4, 0.8)

# to origin
# spoon
# X_THRESHOLD = (-0.18, 0.1)
# Y_THRESHOLD = (0.0, 0.2)
# Z_THRESHOLD = (0.6, 0.9)

# scooper
# X_THRESHOLD = (-0.25, 0.1)
# Y_THRESHOLD = (0.0, 0.25)
# Z_THRESHOLD = (0.6, 0.9)

# more general

# X_THRESHOLD = (-0.3, 0.2)
# Y_THRESHOLD = (-0.3, 0.3)
# Z_THRESHOLD = (0.5, 0.95)

# new trinsics

X_THRESHOLD = (-0.3, 0.3)
Y_THRESHOLD = (-0.3, 0.3)
Z_THRESHOLD = (-0.2, 0.4)


def slerp(q1, q2, t):
    """
    Spherical Linear Interpolation (SLERP) between two quaternions.

    Args:
        q1: Starting quaternion as [qx, qy, qz, qw].
        q2: Ending quaternion as [qx, qy, qz, qw].
        t: Interpolation factor, 0 <= t <= 1.

    Returns:
        Interpolated quaternion as [qx, qy, qz, qw].
    """
    # Normalize input quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute the dot product
    dot_product = np.dot(q1, q2)

    # Ensure the shortest path is used
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product

    # Clamp the dot product to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute the angle between quaternions
    theta_0 = np.arccos(dot_product)  # Angle between q1 and q2
    sin_theta_0 = np.sin(theta_0)

    if sin_theta_0 < 1e-6:
        # Quaternions are almost identical
        return (1 - t) * q1 + t * q2

    # Perform SLERP
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s1 = np.sin(theta_0 - theta_t) / sin_theta_0
    s2 = sin_theta_t / sin_theta_0

    return s1 * q1 + s2 * q2


def predict_current_rotation(prev_quats, prev_flags):
    """
    Predict the current frame rotation based on previous quaternions and flags, weighted by temporal proximity.

    Args:
        prev_quats: List of previous frame quaternions [[qx, qy, qz, qw], ...].
        prev_flags: List of flags corresponding to each quaternion (1 = valid, 0 = invalid).

    Returns:
        Predicted quaternion [qx, qy, qz, qw] for the current frame.
    """
    # Step 1: Filter valid quaternions and assign temporal weights
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

    # Step 4: Compute the weighted mean quaternion
    weights = np.array(weights) / np.sum(weights)  # Normalize weights
    weighted_quat = np.zeros(4)
    for q, w in zip(valid_quats, weights):
        weighted_quat += w * np.array(q)
    weighted_quat /= np.linalg.norm(weighted_quat)  # Normalize the mean quaternion

    # Step 5: Interpolate between the weighted mean quaternion and the most recent valid quaternion
    most_recent_valid_quat = valid_quats[-1]
    predicted_quat = slerp(weighted_quat, most_recent_valid_quat, t=0.5)

    return predicted_quat


def predict_current_position(prev_positions, prev_flags):
    """
    Predict the current frame position using Cubic Spline Interpolation.

    Args:
        prev_positions: List of previous frame positions [[x, y, z], ...].
        prev_flags: List of flags corresponding to each position (1 = valid, 0 = invalid).

    Returns:
        Predicted position [x, y, z] for the current frame.
    """
    # Step 1: Filter valid positions and their corresponding time indices
    valid_positions = []
    valid_times = []
    for i, (pos, flag) in enumerate(zip(prev_positions, prev_flags)):
        if flag == 1:
            valid_positions.append(pos)
            valid_times.append(i)  # Use the index as the "time" axis

    if len(valid_positions) == 0:
        return np.array([-1.0, -1.0, -1.0], dtype=np.float32)

    if len(valid_positions) == 1:
        return np.array(valid_positions[0])

    # Step 4: Fit cubic splines for x, y, z components
    valid_positions = np.array(valid_positions)
    x_coords = valid_positions[:, 0]
    y_coords = valid_positions[:, 1]
    z_coords = valid_positions[:, 2]

    spline_x = CubicSpline(valid_times, x_coords)
    spline_y = CubicSpline(valid_times, y_coords)
    spline_z = CubicSpline(valid_times, z_coords)

    # Step 5: Predict position for the current frame (next time index)
    t_current = len(prev_positions)  # Current frame index
    x_pred = spline_x(t_current)
    y_pred = spline_y(t_current)
    z_pred = spline_z(t_current)

    return np.array([x_pred, y_pred, z_pred], dtype=np.float32)


# Helper Function 1: Calculate Pairwise Distances
def calculate_pairwise_distances(poses):
    """
    Calculate pairwise distances for rotations and translations.

    Args:
        poses: List of poses, where each pose is [qx, qy, qz, qw, x, y, z].

    Returns:
        rot_dists: Pairwise rotation distances.
        trans_dists: Pairwise translation distances.
        pairwise_indices: List of tuples (i, j) indicating which poses were used to calculate each distance.
    """
    num_poses = len(poses)
    rot_dists = []
    trans_dists = []
    pairwise_indices = []  # To track which poses are involved in each pair

    for i in range(num_poses):
        for j in range(i + 1, num_poses):
            # Extract rotations and translations
            q1, q2 = poses[i][:4], poses[j][:4]
            t1, t2 = poses[i][4:], poses[j][4:]

            # Normalize quaternions
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)

            # Calculate rotation geodesic distance
            dot_product = np.dot(q1, q2)
            theta = 2 * np.arccos(np.clip(abs(dot_product), -1, 1))
            rot_dists.append(theta)

            # Calculate translation Euclidean distance
            trans_dists.append(np.linalg.norm(np.array(t1) - np.array(t2)))

            # Save pairwise indices
            pairwise_indices.append((i, j))
    rot_dists = np.array(rot_dists, dtype=np.float32)
    trans_dists = np.array(trans_dists, dtype=np.float32)
    return rot_dists, trans_dists, pairwise_indices


# Helper Function 2: Analyze Distances
def analyze_distances(distances, threshold_factor=2.0, outlier_ratio=0.2):
    """
    Analyze distance distribution to identify noise and inliers.

    Args:
        distances: Array of pairwise distances.
        threshold_factor: Factor to determine outlier threshold (default 2.0 for 95% confidence).
        outlier_ratio: Maximum acceptable ratio of outliers (default 0.2).

    Returns:
        is_noisy: Boolean, True if distances are noisy.
        inlier_distance_indices: Indices of distances considered inliers.
    """
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + threshold_factor * std_dist

    # Identify inliers
    inlier_indices = np.where(distances <= threshold)[0]
    outlier_fraction = 1 - (len(inlier_indices) / len(distances))

    is_noisy = outlier_fraction > outlier_ratio
    return is_noisy, inlier_indices


def detect_pose_outliers(poses, threshold_factor=2.0, outlier_ratio=0.2):
    """
    Detect outlier rotations and translations, and return inliers.

    Args:
        poses: List of poses, where each pose is [qx, qy, qz, qw, x, y, z].
        threshold_factor: Factor to determine outlier threshold (default 2.0 for 95% confidence).
        outlier_ratio: Maximum acceptable ratio of outliers (default 0.2).

    Returns:
        inlier_rots: List of inlier rotations as [qx, qy, qz, qw].
        inlier_trans: List of inlier translations as [x, y, z].
        is_rot_noisy: Boolean, True if rotations are noisy.
        is_trans_noisy: Boolean, True if translations are noisy.
    """
    # Step 1: Calculate pairwise distances
    rot_dists, trans_dists, pairwise_indices = calculate_pairwise_distances(poses)

    # Step 2: Analyze rotation distances
    is_rot_noisy, rot_inlier_dist_indices = analyze_distances(
        rot_dists, threshold_factor, outlier_ratio
    )

    # Step 3: Analyze translation distances
    is_trans_noisy, trans_inlier_dist_indices = analyze_distances(
        trans_dists, threshold_factor, outlier_ratio
    )

    # Step 4: Find pose inliers
    rot_inlier_indices = set(
        idx
        for pair_idx in rot_inlier_dist_indices
        for idx in pairwise_indices[pair_idx]
    )
    trans_inlier_indices = set(
        idx
        for pair_idx in trans_inlier_dist_indices
        for idx in pairwise_indices[pair_idx]
    )

    print(f"rot_inlier_indices: {rot_inlier_indices}")
    print(f"trans_inlier_indices: {trans_inlier_indices}")

    # Convert sets to sorted lists for consistent output
    rot_inlier_indices = sorted(rot_inlier_indices)
    trans_inlier_indices = sorted(trans_inlier_indices)

    # Extract inlier rotations and translations
    inlier_rots = [poses[i][:4] for i in rot_inlier_indices]
    inlier_trans = [poses[i][4:] for i in trans_inlier_indices]

    return inlier_rots, inlier_trans, is_rot_noisy, is_trans_noisy


def is_valid_pose(pose_w, x_threshold, y_threshold, z_threshold):
    """
    Check if pose in world space is valid.

    Args:
        pose_w: Pose in world space as [qx, qy, qz, qw, x, y, z].

    Returns:
        Boolean, True if pose is valid.
    """
    x, y, z = pose_w[-3:]
    # return -1 < x < 1 and -1 < y < 1 and -1 < z < 1

    # return -0.2 < x < 0.4 and 0.0 < y < 0.8 and 0.0 < z < 0.8
    return (x_threshold[0] < x < x_threshold[1] and
            y_threshold[0] < y < y_threshold[1] and
            z_threshold[0] < z < z_threshold[1])


def transform_poses_to_world(mat_poses_c, cam_RTs, x_threshold, y_threshold, z_threshold):
    poses_w = []
    for mat_pose, cam_RT in zip(mat_poses_c, cam_RTs):
        if np.all(mat_pose == -1):  # invalid pose
            continue
        mat_pose_w = cam_RT @ mat_pose
        quat_pose_w = mat_to_quat(mat_pose_w)
        if is_valid_pose(quat_pose_w, x_threshold, y_threshold, z_threshold):
            poses_w.append(quat_pose_w)
    return poses_w


def ransac_consistent_rotation(inlier_rots, threshold):
    """
    Estimate the consistent rotation using RANSAC on inlier rotations.

    Args:
        inlier_rots: List of inlier rotations as [qx, qy, qz, qw].
        threshold: Geodesic distance threshold for inlier classification (in radians).

    Returns:
        Consistent rotation quaternion [qx, qy, qz, qw].
    """
    if len(inlier_rots) == 1:
        return inlier_rots[0]  # Return directly if only one inlier exists

    best_rotation = None
    max_inliers = 0

    # Step 1: Generate candidate rotations by averaging all possible combinations
    for r in range(1, len(inlier_rots) + 1):  # From 1 to all inliers
        for comb in combinations(inlier_rots, r):
            candidate = np.mean(comb, axis=0)
            candidate /= np.linalg.norm(candidate)  # Normalize quaternion

            # Step 2: Evaluate candidate using RANSAC
            inlier_count = 0
            for rot in inlier_rots:
                dot_product = np.dot(candidate, rot)
                loss = 2 * np.arccos(np.clip(abs(dot_product), -1, 1))
                if loss <= threshold:
                    inlier_count += 1

            # Step 3: Update best candidate
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_rotation = candidate

    return best_rotation


def ransac_consistent_translation(inlier_trans, threshold):
    """
    Estimate the consistent translation using RANSAC on inlier translations.

    Args:
        inlier_trans: List of inlier translations as [x, y, z].
        threshold: Euclidean distance threshold for inlier classification.

    Returns:
        Consistent translation [x, y, z].
    """
    if len(inlier_trans) == 1:
        return inlier_trans[0]  # Return directly if only one inlier exists

    best_translation = None
    max_inliers = 0

    # Step 1: Generate candidate translations by averaging all possible combinations
    for r in range(1, len(inlier_trans) + 1):  # From 1 to all inliers
        for comb in combinations(inlier_trans, r):
            candidate = np.mean(comb, axis=0)

            # Step 2: Evaluate candidate using RANSAC
            inlier_count = 0
            for trans in inlier_trans:
                loss = np.linalg.norm(candidate - trans)
                if loss <= threshold:
                    inlier_count += 1

            # Step 3: Update best candidate
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_translation = candidate

    return best_translation


def project_points_to_image(camera_intrinsics, camera_extrinsics, points_3d):
    """
    Project a group of 3D points back to 2D image points using the camera intrinsics and extrinsics.
    
    Args:
    - camera_intrinsics (np.ndarray): 3x3 intrinsic matrix of the camera.
    - camera_extrinsics (np.ndarray): 4x4 extrinsic matrix of the camera (rotation and translation).
    - points_3d (np.ndarray): Nx3 array of 3D points in world coordinates.
    
    Returns:
    - image_points (np.ndarray): Nx2 array of 2D points in image coordinates (u, v).
    """
    # Convert the 3D points to homogeneous coordinates (Nx4)
    points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # (778, 4)
    
    # Apply the inverse of the extrinsics to convert points to the camera frame
    extrinsics_inv = np.linalg.inv(camera_extrinsics)  # (4, 4)
    points_camera_frame = (extrinsics_inv @ points_3d_homogeneous.T).T  # (778, 4)
    
    # Perspective projection: project onto the image plane using the intrinsics
    points_projected_homogeneous = (camera_intrinsics @ points_camera_frame[:, :3].T).T  # (778, 3)
    
    # Convert from homogeneous coordinates to 2D
    u = points_projected_homogeneous[:, 0] / points_projected_homogeneous[:, 2]
    v = points_projected_homogeneous[:, 1] / points_projected_homogeneous[:, 2]
    
    # Stack into Nx2 array
    image_points = np.vstack([u, v]).T  # (778, 2)
    
    return image_points

def debug_save_poses(poses_w, save_dir="debug_pose_w"):
    os.makedirs(save_dir, exist_ok=True)
    for i, pose in enumerate(poses_w):
        pose_dict = {
            "rotation": pose[:4].tolist(),  # [qx, qy, qz, qw]
            "translation": pose[4:7].tolist()
        }
        with open(f"{save_dir}/pose_cam{i}.json", "w") as f:
            json.dump(pose_dict, f, indent=2)


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
    """
    Get consistent pose in world space using RANSAC on inlier rotations and translations.

    Args:
        mat_poses_c: List of poses in camera space as 4x4 matrix.
        cam_RTs: List of camera extrinsics as 4x4 matrix.
        prev_poses_w: List of previous poses in world space as [qx, qy, qz, qw, x, y, z, flag].
        rot_thresh: Rotational threshold in degrees.
        trans_thresh: Translation threshold in meters.
        thresh_factor: Factor to determine outlier threshold (default 2.0 for 95% confidence).
        outlier_ratio: Maximum acceptable ratio of outliers (default 0.2).

    Returns:
        Consistent pose in world space as [qx, qy, qz, qw, x, y, z, flag].
    """
    rot_thresh = np.deg2rad(rot_thresh)
    curr_rot = None
    curr_trans = None
    flag = 1

    # Step 1: transform all poses to world space
    poses_w = transform_poses_to_world(mat_poses_c, cam_RTs, x_threshold, y_threshold, z_threshold)

    
    # if len(poses_w) == 0:
    if len(poses_w) < 3:
        curr_rot = predict_current_rotation(
            [pose[:4] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        curr_trans = predict_current_position(
            [pose[4:7] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
        return np.concatenate([curr_rot, curr_trans, [flag]], axis=0)
    # elif len(poses_w) == 1:
    #     curr_rot = poses_w[0][:4]
    #     curr_trans = poses_w[0][4:]
    #     return np.concatenate([curr_rot, curr_trans, [flag]], axis=0)

    # Stack poses for processing
    poses_w = np.stack(poses_w, axis=0)

    #####debug
    debug_save_poses(poses_w)


    # Step 2: detect check if poses are noisy
    inlier_rots, inlier_trans, is_rot_noisy, is_trans_noisy = detect_pose_outliers(
        poses_w, thresh_factor, outlier_ratio
    )
    print(f"is_rot_noisy: {is_rot_noisy}, is_trans_noisy: {is_trans_noisy}")
    # print(f"inlier_rots: {inlier_rots}, inlier_trans: {inlier_trans}")

    # Step 3: Handle noisy scenarios
    if is_rot_noisy and is_trans_noisy:
        # Predict both rotation and translation
        curr_rot = predict_current_rotation(
            [pose[:4] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        curr_trans = predict_current_position(
            [pose[4:7] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
    elif is_rot_noisy:
        # Predict rotation, estimate translation via RANSAC
        curr_rot = predict_current_rotation(
            [pose[:4] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        curr_trans = ransac_consistent_translation(inlier_trans, trans_thresh)
        flag = 0
    elif is_trans_noisy:
        # Predict translation, estimate rotation via RANSAC
        curr_rot = ransac_consistent_rotation(inlier_rots, rot_thresh)
        curr_trans = predict_current_position(
            [pose[4:7] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
    else:
        # Use RANSAC for both rotation and translation
        curr_rot = ransac_consistent_rotation(inlier_rots, rot_thresh)
        curr_trans = ransac_consistent_translation(inlier_trans, trans_thresh)

    # Ensure both rotation and translation are defined
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


def is_valid_ob_pose(ob_in_cam, x_threshold, y_threshold, z_threshold, cam_RT=None):
    if np.all(ob_in_cam == -1):
        return False
    elif cam_RT is None:
        x, y, z = ob_in_cam[:3, 3]
    else:
        ob_in_world = cam_RT @ ob_in_cam
        x, y, z = ob_in_world[:3, 3]

    # print(f"DEBUG: ob_in_world: {ob_in_world if cam_RT is not None else ob_in_cam[:3, 3]}")

    # return -0.2 < x < 0.4 and 0.0 < y < 0.8 and 0.0 < z < 0.8
    return (x_threshold[0] < x < x_threshold[1] and
            y_threshold[0] < y < y_threshold[1] and
            z_threshold[0] < z < z_threshold[1])


def initialize_fd_pose_estimator(textured_mesh_path, cleaned_mesh_path, debug_dir):
    textured_mesh = trimesh.load(textured_mesh_path, process=False)
    cleaned_mesh = trimesh.load(cleaned_mesh_path, process=False)
    return FoundationPose(
        model_pts=cleaned_mesh.vertices.astype(np.float32),
        model_normals=cleaned_mesh.vertex_normals.astype(np.float32),
        mesh=textured_mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(),
        debug=0,
        debug_dir=debug_dir,
        rotation_grid_min_n_views=120,
        rotation_grid_inplane_step=60,
    )


def crop_mask_and_adjust_intrinsics(mask, K):
    """
    Given a binary mask (shape [480, 640]) and a camera intrinsic matrix K (3x3),
    returns the bounding box (ymin, ymax, xmin, xmax) for a 200x200 region (with 20px padding)
    centered on the mask==1 region, and a new intrinsic matrix K_new that incorporates the cropping.
    Handles edge cases where the crop would go out of bounds.
    """
    H, W = mask.shape
    ys, xs = np.where(mask == 255)
    if len(xs) == 0 or len(ys) == 0:
        # No foreground, return full image and original K
        return (0, H, 0, W), K.copy()
    # Center of the mask region
    y_center = int(np.round(ys.mean()))
    x_center = int(np.round(xs.mean()))
    crop_size = 200
    pad = 20

    # Compute crop bounds
    ymin = y_center - crop_size // 2 - pad
    ymax = y_center + crop_size // 2 + pad
    xmin = x_center - crop_size // 2 - pad
    xmax = x_center + crop_size // 2 + pad

    # Adjust crop to take the appropriate corner if at the borders
    if ymin < 0:
        ymax = min(H, ymax - ymin)
        ymin = 0
    if xmin < 0:
        xmax = min(W, xmax - xmin)
        xmin = 0
    if ymax > H:
        ymin = max(0, ymin - (ymax - H))
        ymax = H
    if xmax > W:
        xmin = max(0, xmin - (xmax - W))
        xmax = W

    # If crop is smaller than 200+2*pad due to image edge, adjust to get as close as possible
    # (optional: could shift crop to keep size, but here we just clamp)
    K_new = K.copy().astype(np.float32)
    K_new[0, 2] -= xmin
    K_new[1, 2] -= ymin
    return (ymin, ymax, xmin, xmax), K_new

def flip_z_to_match_ref(pose, ref_z):
    """
    Given a 4x4 pose (world), flip z axis (and x to keep right-handed) if needed so that z is closer to ref_z.
    Returns the possibly flipped pose.
    """
    R0 = pose[:3, :3]
    t0 = pose[:3, 3]
    z0 = R0[:, 2]
    # Mirrored rotation: flip z axis
    R_flip = R0.copy()
    R_flip[:, 2] *= -1
    R_flip[:, 0] *= -1
    # Compare angle between z0 and ref_z
    angle_orig = np.arccos(np.clip(np.dot(z0, ref_z) / (np.linalg.norm(z0) * np.linalg.norm(ref_z)), -1, 1))
    z_flip = R_flip[:, 2]
    angle_flip = np.arccos(np.clip(np.dot(z_flip, ref_z) / (np.linalg.norm(z_flip) * np.linalg.norm(ref_z)), -1, 1))
    if angle_flip < angle_orig:
        T_new = np.eye(4)
        T_new[:3, :3] = R_flip
        T_new[:3, 3] = t0
        return T_new
    else:
        return pose

def threshold_iterative_outlier_removal(poses, trans_thresh=0.03, rot_thresh_deg=15):
    """
    Iteratively remove the pose farthest from the mean pose (translation and rotation),
    until all remaining poses are within the thresholds from the mean.
    Args:
        poses: list of [qx, qy, qz, qw, x, y, z]
        trans_thresh: translation threshold (meters)
        rot_thresh_deg: rotation threshold (degrees)
    Returns:
        inliers: list of indices of inlier poses
        outliers: list of indices of outlier poses
    """
    poses = np.array(poses)
    indices = list(range(len(poses)))
    outliers = []
    rot_thresh = np.deg2rad(rot_thresh_deg)
    while len(indices) > 1:
        # Compute mean pose (translation and rotation)
        mean_trans = np.mean([poses[i][4:] for i in indices], axis=0)
        mean_rot = np.mean([poses[i][:4] / np.linalg.norm(poses[i][:4]) for i in indices], axis=0)
        mean_rot = mean_rot / np.linalg.norm(mean_rot)
        # Compute distances to mean
        trans_dists = np.array([np.linalg.norm(poses[i][4:] - mean_trans) for i in indices])
        rot_dists = np.array([2 * np.arccos(np.clip(np.abs(np.dot(poses[i][:4] / np.linalg.norm(poses[i][:4]), mean_rot)), -1, 1)) for i in indices])
        # Check if all within threshold
        if np.all(trans_dists <= trans_thresh) and np.all(rot_dists <= rot_thresh):
            break
        # Remove the pose with the largest (normalized) distance
        norm_trans = trans_dists / trans_thresh
        norm_rot = rot_dists / rot_thresh
        norm_total = norm_trans + norm_rot
        farthest_idx = np.argmax(norm_total)
        outliers.append(indices[farthest_idx])
        indices.pop(farthest_idx)
    inliers = indices
    return inliers, outliers


def run_pose_estimation(
    sequence_folder,
    object_idx,
    est_refine_iter,
    track_refine_iter,
    start_frame,
    end_frame,
    rot_thresh,
    trans_thresh,
    crop_view,
):
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

    object_mesh_small = object_mesh_cleaned.simplify_quadric_decimation(0.4)
    object_mesh_small.vertices *= 0.001

    #### process mesh ###


    other_mesh = trimesh.load(data_loader.object_cleaned_files[object_idx], process=True)

    # load in texture information
    # print(f"Register: {config['foundation_pose']['register']}")
    # USE_TEXTURE= True
    USE_TEXTURE = False

    if USE_TEXTURE:
        try:
            # texture_file = args.mesh_file.replace('decim_mesh_files', 'textures').replace('.obj', '.jpg')
            texture_file = data_loader._texture_files[object_idx]
            texture = cv2.imread(texture_file)
            from PIL import Image
            im = Image.open(texture_file)
            uv = other_mesh.visual.uv
            material = trimesh.visual.texture.SimpleMaterial(image=im)
            color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
            other_mesh.visual = color_visuals
        except:
            print(f"Error loading texture file: {texture_file}")
    mesh=None 
    # print(len(other_mesh.vertices))
    if len(other_mesh.vertices) > 200000: # fix
        mesh = other_mesh.simplify_quadric_decimation(0.8)
        print("Decim mesh.")
        # mesh = other_mesh.simplify_quadric_decimation(200000) #trimesh.Trimesh(vertices=samples, process=True)
        del other_mesh
    else:
        mesh = copy.deepcopy(other_mesh)

    # mesh.vertices *= 0.001
    # mesh_copy = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy())
    print(f"[DEBUG] : Mesh vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")
    # debug = config['foundation_pose']['debug']
    # debug_dir = args.debug_dir
    # os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    # to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    # mesh_copy.apply_transform(to_origin)
    # mesh.apply_transform(to_origin)
    # bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  
    # print(f"[DEBUG] : To Origin:\ {to_origin}, Extents: {extents}, BBox: {bbox}")
    ###########

    # Check start and end frame_idx
    start_frame = max(start_frame, 0)
    end_frame = num_frames if end_frame < start_frame else end_frame

    logging.info(f"start_frame: {start_frame}, end_frame: {end_frame}")

    save_folder = sequence_folder / "processed" / "fd_pose_solver"
    save_folder.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.WARNING)
    set_seed(0)
    debug = 3
    # estimator = FoundationPose(
    #     # model_pts=object_mesh_cleaned.vertices.astype(np.float32),
    #     # model_normals=object_mesh_cleaned.vertex_normals.astype(np.float32),
    #     # mesh=trimesh.load(data_loader.object_textured_files[object_idx]),
    #     model_pts=object_mesh_small.vertices.astype(np.float32),
    #     model_normals=object_mesh_small.vertex_normals.astype(np.float32),
    #     mesh = object_mesh_small,
    #     scorer=ScorePredictor(),
    #     refiner=PoseRefinePredictor(),
    #     glctx=dr.RasterizeCudaContext(),
    #     debug=3,
    #     debug_dir=save_folder / "debug" / object_id,
    #     rotation_grid_min_n_views=120,
    #     rotation_grid_inplane_step=60,
    # )

    GENERATE_OB_IN_WORLD_ONLY = False
    if not GENERATE_OB_IN_WORLD_ONLY:

        estimator = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),
            debug=debug,
            # debug=0,
            debug_dir=save_folder / "debug" / object_id,
            rotation_grid_min_n_views=120,
            rotation_grid_inplane_step=60,
        )

    # estimator = FoundationPose(
    #     model_pts=mesh_copy.vertices, model_normals=mesh_copy.vertex_normals, mesh=mesh_copy,
    #     scorer=ScorePredictor(),
    #     refiner=PoseRefinePredictor(),
    #     glctx=dr.RasterizeCudaContext(),
    #     debug=debug,
    #     # debug=0,
    #     debug_dir=save_folder / "debug" / object_id,
    #     rotation_grid_min_n_views=120,
    #     rotation_grid_inplane_step=60,
    # )


    # Initialize poses
    ob_in_world_refined = empty_mat_pose.copy()
    # For each camera, keep a list of its own per-frame poses
    ob_in_cam_poses_per_cam = [ [] for _ in range(len(valid_serials)) ]  # list of lists: [cam][frame]
    all_poses_w = []  # integrated world poses per frame


    ################ Tricks #################
    REVERSE = False
    MASKED_DEPTH = True  # whether to use masked depth or not
    MASKED_IMAGE = False  # whether to use masked image or not
    CROP_VIEW = False  # whether to crop the view or not
    MASKED_OBJECT = True
    CROP_VIEW = crop_view
    if GENERATE_OB_IN_WORLD_ONLY:
        end_frame = start_frame
    for frame_id in range(start_frame, end_frame, 1):
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

            # 只保留mask区域的深度
            if MASKED_DEPTH:
                depth = depth.copy()
                depth[mask == 0] = 0
                if MASKED_OBJECT:
                    object_mask = data_loader.get_object_mask(serial, frame_idx) # 如果没有会返回0
                    depth[object_mask != 0] = 0
                    if frame_idx == 0:
                        # print(f"[DEBUG] Frame {frame_idx}, Cam {serial}: object_mask.sum() = {object_mask.sum()}")
                        # 保存depth图片debug
                        debug_depth_path = save_folder / "debug" / f"processed_depth_{object_id}" / serial
                        debug_depth_path.mkdir(parents=True, exist_ok=True)
                        view_depth = depth.copy()
                        # 让深度图可视化
                        view_depth = (view_depth - np.min(view_depth)) / (np.max(view_depth) - np.min(view_depth)) * 255
                        view_depth = view_depth.astype(np.uint8)
                        cv2.imwrite(str(debug_depth_path / f"depth_{frame_idx:06d}_object_mask.png"), view_depth)
                        cv2.imwrite(str(debug_depth_path / f"depth_{frame_idx:06d}.png"), depth)

            if MASKED_IMAGE:
                color = color.copy()
                color[mask == 0] = 0
            
            # --- CROP_VIEW logic ---
            K = valid_Ks[serial_idx]
            if CROP_VIEW:
                coords, cropped_K = crop_mask_and_adjust_intrinsics(mask, K)
                ymin, ymax, xmin, xmax = coords
                cropped_color = color[ymin:ymax, xmin:xmax]
                cropped_depth = depth[ymin:ymax, xmin:xmax]
                cropped_mask = mask[ymin:ymax, xmin:xmax]
                use_color = cropped_color
                use_depth = cropped_depth
                use_mask = cropped_mask
                use_K = cropped_K
            else:
                use_color = color
                use_depth = depth
                use_mask = mask
                use_K = K
            # --- END CROP_VIEW logic ---

            # print("depth shape:", depth.shape)
            # print("depth stats: min =", np.min(depth), ", max =", np.max(depth), ", mean =", np.mean(depth))

            # print(f"[DEBUG] Cam {serial}: mask.sum() = {mask.sum()}, depth.mean() = {np.mean(depth):.2f}")


        # --- SEPARATE TRACKING FOR EACH CAMERA ---
            # Get previous pose for this camera (or empty_mat_pose if first frame)
            prev_pose = ob_in_cam_poses_per_cam[serial_idx][-1] if ob_in_cam_poses_per_cam[serial_idx] else empty_mat_pose.copy()

            if use_mask.sum() < 10:
                ob_in_cam_mat = empty_mat_pose.copy()
                print(f"[DEBUG] Frame {frame_idx}, Cam {serial}: mask.sum() = {use_mask.sum()} is less than 100, skipping.")
            elif is_valid_ob_pose(prev_pose, x_threshold, y_threshold, z_threshold, valid_RTs[serial_idx]):
                ob_in_cam_mat = estimator.track_one(
                    rgb=use_color,
                    depth=use_depth,
                    K=use_K,
                    iteration=track_refine_iter,
                    prev_pose=prev_pose,
                )
                curr_ob_in_world = valid_RTs[serial_idx] @ ob_in_cam_mat
                print(f"Frame {frame_idx}, Cam {serial}: ob in world: {curr_ob_in_world[:3,3]}")
            else:
                print(f"[DEBUG] Frame {frame_idx}, Cam {serial}: estimating new pose.")
                init_ob_pos_center = data_loader.get_init_translation(
                    frame_idx, [serial], object_idx, kernel_size=5
                )[0][0]
                if init_ob_pos_center is not None:
                    ob_in_cam_mat = estimator.register(
                        rgb=use_color,
                        depth=use_depth,
                        ob_mask=use_mask,
                        K=use_K,
                        iteration=est_refine_iter,
                        init_ob_pos_center=init_ob_pos_center,
                    )
                    if not is_valid_ob_pose(ob_in_cam_mat, x_threshold, y_threshold, z_threshold, valid_RTs[serial_idx]):
                        print(f"[DEBUG]!!! Frame {frame_idx}, Cam {serial}: Register failed! using empty pose.")
                        debug_ob_in_world = valid_RTs[serial_idx] @ ob_in_cam_mat
                        print(debug_ob_in_world[:3,3])
                        ob_in_cam_mat = empty_mat_pose.copy()
                else:
                    print(f"[DEBUG] Frame {frame_idx}, Cam {serial}: init_ob_pos_center is None, using empty pose.")
                    ob_in_cam_mat = empty_mat_pose.copy()

            ob_in_cam_poses_per_cam[serial_idx].append(ob_in_cam_mat)

            
        # --- END SEPARATE TRACKING ---

        # --- Direction consistency check for all cams (including cam 7 except in frame 0) ---
        # Gather all tracked ob_in_cam poses for this frame
        tracked_poses = [ob_in_cam_poses_per_cam[serial_idx][-1] for serial_idx in range(len(valid_serials))]
        # Transform to world
        ob_in_world_poses = [valid_RTs[serial_idx] @ tracked_poses[serial_idx] for serial_idx in range(len(valid_serials))]
        # Use cam 7's last frame's z direction as reference
        ref_cam_idx = 7
        if frame_id == 0:
            ref_z = ob_in_world_poses[ref_cam_idx][:3, 2]
        else:
            # Use cam 7's last frame's z direction (from previous frame)
            ref_pose_prev = ob_in_cam_poses_per_cam[ref_cam_idx][-2] if len(ob_in_cam_poses_per_cam[ref_cam_idx]) > 1 else ob_in_cam_poses_per_cam[ref_cam_idx][-1]
            ref_pose_prev_world = valid_RTs[ref_cam_idx] @ ref_pose_prev
            ref_z = ref_pose_prev_world[:3, 2]
        for serial_idx in range(len(valid_serials)):
            # For all cams, including cam 7 (except in frame 0)
            if frame_id == 0 and serial_idx == ref_cam_idx:
                continue  # skip cam 7 in frame 0
            pose_world = ob_in_world_poses[serial_idx]
            pose_flipped = flip_z_to_match_ref(pose_world, ref_z)
            if not np.allclose(pose_flipped, pose_world):
                # If flipped, update the ob_in_cam pose accordingly
                ob_in_cam_poses_per_cam[serial_idx][-1] = np.linalg.inv(valid_RTs[serial_idx]) @ pose_flipped
                print(f"[INFO] Frame {frame_id}, Cam {serial_idx}: flipped z axis to match cam 7's last frame direction.")

        # --- Position consistency check: iterative outlier removal ---
        POSITION_THRESH = 0.03  # meters
        ROT_THRESH_DEG = 15
        # Convert world poses to [qx,qy,qz,qw,x,y,z]
        poses_quat = [mat_to_quat(T) for T in ob_in_world_poses]
        inlier_idx, outlier_idx = threshold_iterative_outlier_removal(
            poses_quat, trans_thresh=POSITION_THRESH, rot_thresh_deg=ROT_THRESH_DEG)
        if len(outlier_idx) > 0:
            print(f"[Iterative threshold] Inlier cams: {[valid_serials[i] for i in inlier_idx]}")
            print(f"[Iterative threshold] Outlier cams: {[valid_serials[i] for i in outlier_idx]}")
            # Compute mean inlier pose (world)
            inlier_poses = [ob_in_world_poses[i] for i in inlier_idx]
            mean_trans = np.mean([T[:3,3] for T in inlier_poses], axis=0)
            # Average quaternion (naive mean, then normalize)
            quats = np.array([R.from_matrix(T[:3,:3]).as_quat() for T in inlier_poses])
            mean_quat = np.mean(quats, axis=0)
            mean_quat = mean_quat / np.linalg.norm(mean_quat)
            mean_rot = R.from_quat(mean_quat).as_matrix()
            mean_pose_world = np.eye(4)
            mean_pose_world[:3,:3] = mean_rot
            mean_pose_world[:3,3] = mean_trans
            for serial_idx in outlier_idx:
                new_ob_in_cam = np.linalg.inv(valid_RTs[serial_idx]) @ mean_pose_world
                ob_in_cam_poses_per_cam[serial_idx][-1] = new_ob_in_cam
                print(f"[Iterative threshold] Frame {frame_id}, Cam {serial_idx}: replaced with mean inlier world pose.")

        # --- END Position consistency check ---

        # --- re-register cam if far from mask center ---
        MASK_CENTER_THRESH = 10  # pixels
        for serial_idx in range(len(valid_serials)):
            ob_in_cam_mat = ob_in_cam_poses_per_cam[serial_idx][-1]
            if np.all(ob_in_cam_mat == -1):
                continue
            # Project 3D position to 2D
            K = valid_Ks[serial_idx]
            t = ob_in_cam_mat[:3, 3]
            pt3d = t.reshape(3, 1)
            pt2d_h = K @ pt3d
            pt2d = (pt2d_h[:2] / pt2d_h[2]).flatten()
            # Compute mask center
            mask = data_loader.get_mask(valid_serials[serial_idx], frame_id, object_idx)
            ys, xs = np.where(mask == 255)
            if len(xs) == 0 or len(ys) == 0:
                continue
            mask_center = np.array([xs.mean(), ys.mean()])
            dist = np.linalg.norm(pt2d - mask_center)
            if dist > MASK_CENTER_THRESH:
                print(f"[Re-register] Frame {frame_id}, Cam {serial_idx}: 3D->2D proj {pt2d} far from mask center {mask_center}, dist={dist:.1f} > {MASK_CENTER_THRESH}. Re-registering.")
                color = data_loader.get_color(valid_serials[serial_idx], frame_id)
                depth = data_loader.get_depth(valid_serials[serial_idx], frame_id)
                mask = data_loader.get_mask(valid_serials[serial_idx], frame_id, object_idx)
                K = valid_Ks[serial_idx]
                # Optionally crop if CROP_VIEW is enabled
                if CROP_VIEW:
                    coords, cropped_K = crop_mask_and_adjust_intrinsics(mask, K)
                    ymin, ymax, xmin, xmax = coords
                    color = color[ymin:ymax, xmin:xmax]
                    depth = depth[ymin:ymax, xmin:xmax]
                    mask = mask[ymin:ymax, xmin:xmax]
                    K = cropped_K
                init_ob_pos_center = data_loader.get_init_translation(
                    frame_id, [valid_serials[serial_idx]], object_idx, kernel_size=5
                )[0][0]
                if init_ob_pos_center is not None:
                    new_pose = estimator.register(
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        K=K,
                        iteration=est_refine_iter,
                        init_ob_pos_center=init_ob_pos_center,
                    )
                    ob_in_cam_poses_per_cam[serial_idx][-1] = new_pose
                    print(f"[Re-register] Frame {frame_id}, Cam {serial_idx}: replaced tracked pose with re-registered pose.")

        # --- Save per-camera pose ---
        for serial_idx in range(len(valid_serials)):
        # Save per-camera pose
            save_pose_folder = save_folder / object_id / "ob_in_cam" / serial
            save_pose_folder.mkdir(parents=True, exist_ok=True)
            write_pose_to_txt(
                save_pose_folder / f"{frame_idx:06d}.txt", mat_to_quat(ob_in_cam_mat)
            )
            if debug > 1:
                debug_image_path = save_folder / "debug" / object_id / serial
                debug_image_path.mkdir(parents=True, exist_ok=True)
        


    # # --- INTEGRATE ALL VIEWS AT THE END ---
    # # For each frame, collect all camera poses and integrate
    # if GENERATE_OB_IN_WORLD_ONLY:
    #     end_frame = num_frames
    #     # load ob_in_cam_poses_per_cam from file
    #     for serial_idx in range(len(valid_serials)):
    #         for frame_idx in range(end_frame - start_frame):
    #             ob_in_cam_poses_per_cam[serial_idx].append(np.loadtxt(save_folder / object_id / "ob_in_cam" / valid_serials[serial_idx] / f"{frame_idx:06d}.txt"))

    for frame_idx in range(end_frame - start_frame):
        ob_in_cam_poses = [ob_in_cam_poses_per_cam[serial_idx][frame_idx] for serial_idx in range(len(valid_serials))]
        curr_pose_w = get_consistent_pose_w(
            mat_poses_c=ob_in_cam_poses,
            cam_RTs=valid_RTs,
            prev_poses_w=all_poses_w,
            rot_thresh=rot_thresh,
            trans_thresh=trans_thresh,
            thresh_factor=1.0,
            outlier_ratio=0.4,
            x_threshold=x_threshold,
            y_threshold=y_threshold,
            z_threshold=z_threshold,
        )
        all_poses_w.append(curr_pose_w)
        print(f"[RESULT] ob_in_world (Frame {frame_idx}): {curr_pose_w[4:7]}")
        save_pose_folder = save_folder / object_id / "ob_in_world"
        save_pose_folder.mkdir(parents=True, exist_ok=True)
        write_pose_to_txt(save_pose_folder / f"{frame_idx:06d}.txt", curr_pose_w)
        ob_in_world_refined = quat_to_mat(curr_pose_w[:7])
        # --- END INTEGRATION ---

        # # --- Outlier rejection and replacement ---
        # ob_in_cam_poses = [ob_in_cam_poses_per_cam[serial_idx][-1] for serial_idx in range(len(valid_serials))]
        # # Transform to world
        # ob_in_world_poses = [valid_RTs[serial_idx] @ ob_in_cam_poses[serial_idx] for serial_idx in range(len(valid_serials))]
        # translations = np.array([pose[:3, 3] for pose in ob_in_world_poses])
        # rotations = np.array([pose[:3, :3] for pose in ob_in_world_poses])

        # # Use the integrated world result as reference
        # ob_in_world_combined = get_consistent_pose_w(
        #     mat_poses_c=ob_in_cam_poses,
        #     cam_RTs=valid_RTs,
        #     prev_poses_w=all_poses_w,
        #     rot_thresh=rot_thresh,
        #     trans_thresh=trans_thresh,
        #     thresh_factor=1.0,
        #     outlier_ratio=0.4,
        #     x_threshold=x_threshold,
        #     y_threshold=y_threshold,
        #     z_threshold=z_threshold,
        # )
        # print(f"ob_in_world_combined: {ob_in_world_combined}")
        # # Convert to 4x4
        # q = ob_in_world_combined[:4]
        # t = ob_in_world_combined[4:7]
        # R_combined = R.from_quat(q).as_matrix()
        # T_combined = np.eye(4)
        # T_combined[:3, :3] = R_combined
        # T_combined[:3, 3] = t


        # # Compute reference
        # trans_ref = T_combined[:3, 3]
        # rot_ref = T_combined[:3, :3]

        # trans_thresh_val = trans_thresh  # from args
        # # rot_thresh_rad = np.deg2rad(rot_thresh)  # from args
        # rot_thresh_rad = rot_thresh

        # for serial_idx in range(len(valid_serials)):
        #     trans = translations[serial_idx]
        #     rot = rotations[serial_idx]
        #     trans_dist = np.linalg.norm(trans - trans_ref)
        #     rot_dist = geodesic_distance(rot, rot_ref)
        #     if (trans_dist > trans_thresh_val or rot_dist > rot_thresh_rad) and frame_idx > 0:
        #         print(f"Outlier detected! Frame {frame_idx}, Cam {serial_idx}: trans_dist: {trans_dist}, rot_dist: {rot_dist} , rot_thresh: {rot_thresh}, trans_thresh: {trans_thresh}")
            
        #         # print(f"Frame {frame_idx}, Cam {serial}: trans_dist: {trans_dist}, rot_dist: {rot_dist}")
        #         # Outlier: replace with combined world result projected to this camera
        #         ob_in_cam_poses_per_cam[serial_idx][-1] = np.linalg.inv(valid_RTs[serial_idx]) @ T_combined

        

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
        default=15,
        help="number of iterations for estimation",
    )
    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=20, # 50 5
        help="number of iterations for tracking",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="start frame")
    parser.add_argument("--end_frame", type=int, default=-1, help="end frame")
    parser.add_argument(
        "--rot_thresh",
        type=float,
        default=1.0, # 2.0
        help="rotation threshold, degree",
    )
    parser.add_argument(
        "--trans_thresh",
        type=float,
        default=0.02, # 0.03
        help="translation threshold, meters",
    )

    parser.add_argument("--crop_view", type=bool, default=False, help="crop view")
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please specify the sequence folder.")
    if args.object_idx is None:
        raise ValueError("Please specify the object index.")

    set_logging_format()
    t_start = time.time()
    logger = logging.getLogger("register")
    # 
    logger.setLevel(logging.WARNING)  # 只显示WARNING及以上
    logging.getLogger().setLevel(logging.WARNING)  # 全局也设为WARNING

    loader = HOCapLoader(args.sequence_folder)
    x_threshold = loader.x_threshold
    y_threshold = loader.y_threshold
    z_threshold = loader.z_threshold

    run_pose_estimation(
        args.sequence_folder,
        args.object_idx,
        args.est_refine_iter,
        args.track_refine_iter,
        args.start_frame,
        args.end_frame,
        args.rot_thresh,
        args.trans_thresh,
        args.crop_view,
    )

    # logging.info(f"done!!! time: {time.time() - t_start:.3f}s.")
    print(f"done!!! time: {time.time() - t_start:.3f}s.")
