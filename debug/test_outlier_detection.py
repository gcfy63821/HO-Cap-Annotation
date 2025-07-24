import numpy as np
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN


def preprocess_flip_z_to_match_ref(tool_poses_world, ref_cam=7):
    """
    For each cam's pose in world, flip z axis if needed so that its z direction is closer to ref_cam's z direction.
    Input: tool_poses_world: list of 4x4 np.arrays, order matches serials.
           ref_cam: int (index in list)
    Returns: new list with possibly flipped poses, same order as input.
    """
    processed = []
    ref_pose = tool_poses_world[ref_cam]
    ref_z = ref_pose[:3, 2]
    for cam, pose in enumerate(tool_poses_world):
        if cam == ref_cam:
            processed.append(pose)
            continue
        R0 = pose[:3, :3]
        t0 = pose[:3, 3]
        z0 = R0[:, 2]
        # Mirrored rotation: flip z axis
        R_flip = R0.copy()
        R_flip[:, 2] *= -1
        # To keep right-handed, also flip x
        R_flip[:, 0] *= -1
        # Compare angle between z0 and ref_z
        angle_orig = np.arccos(np.clip(np.dot(z0, ref_z) / (np.linalg.norm(z0) * np.linalg.norm(ref_z)), -1, 1))
        z_flip = R_flip[:, 2]
        angle_flip = np.arccos(np.clip(np.dot(z_flip, ref_z) / (np.linalg.norm(z_flip) * np.linalg.norm(ref_z)), -1, 1))
        if angle_flip < angle_orig:
            T_new = np.eye(4)
            T_new[:3, :3] = R_flip
            T_new[:3, 3] = t0
            processed.append(T_new)
            print(f"[INFO] Cam {cam}: flipped z axis to match cam {ref_cam} (angle_orig={np.rad2deg(angle_orig):.2f} deg, angle_flip={np.rad2deg(angle_flip):.2f} deg)")
        else:
            processed.append(pose)
    return processed

# --- Outlier detection code from 04-1-3_fd_pose_solver_separate.py ---
def calculate_pairwise_distances(poses):
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
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + threshold_factor * std_dist
    inlier_indices = np.where(distances <= threshold)[0]
    outlier_fraction = 1 - (len(inlier_indices) / len(distances))
    is_noisy = outlier_fraction > outlier_ratio
    return is_noisy, inlier_indices

def detect_pose_outliers(poses, threshold_factor=2.0, outlier_ratio=0.2):
    rot_dists, trans_dists, pairwise_indices = calculate_pairwise_distances(poses)
    is_rot_noisy, rot_inlier_dist_indices = analyze_distances(rot_dists, threshold_factor, outlier_ratio)
    is_trans_noisy, trans_inlier_dist_indices = analyze_distances(trans_dists, threshold_factor, outlier_ratio)
    rot_inlier_indices = set(idx for pair_idx in rot_inlier_dist_indices for idx in pairwise_indices[pair_idx])
    trans_inlier_indices = set(idx for pair_idx in trans_inlier_dist_indices for idx in pairwise_indices[pair_idx])
    rot_inlier_indices = sorted(rot_inlier_indices)
    trans_inlier_indices = sorted(trans_inlier_indices)
    inlier_rots = [poses[i][:4] for i in rot_inlier_indices]
    inlier_trans = [poses[i][4:] for i in trans_inlier_indices]
    return inlier_rots, inlier_trans, is_rot_noisy, is_trans_noisy, rot_inlier_indices, trans_inlier_indices

# Add get_consistent_pose_w from 04-1-3_fd_pose_solver_separate.py

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
    rot_thresh = np.deg2rad(rot_thresh)
    curr_rot = None
    curr_trans = None
    flag = 1
    # Step 1: transform all poses to world space
    poses_w = [cam_RTs[i] @ mat_poses_c[i] for i in range(len(mat_poses_c))]
    poses_quat = [mat_to_quat(T) for T in poses_w]
    if len(poses_w) < 3:
        curr_rot = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        curr_trans = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        flag = 0
        return np.concatenate([curr_rot, curr_trans, [flag]], axis=0)
    poses_w = np.stack(poses_w, axis=0)
    inlier_rots, inlier_trans, is_rot_noisy, is_trans_noisy, _, _ = detect_pose_outliers(
        [mat_to_quat(T) for T in poses_w], thresh_factor, outlier_ratio)
    # Use mean of inlier rotations and translations
    if len(inlier_rots) > 0:
        mean_rot = np.mean(np.stack(inlier_rots), axis=0)
        mean_rot = mean_rot / np.linalg.norm(mean_rot)
    else:
        mean_rot = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    if len(inlier_trans) > 0:
        mean_trans = np.mean(np.stack(inlier_trans), axis=0)
    else:
        mean_trans = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    return np.concatenate([mean_rot, mean_trans, [flag]], axis=0)

def geodesic_dist_matrix(quats):
    """Compute NxN geodesic distance matrix for quaternions."""
    N = len(quats)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            q1 = quats[i] / np.linalg.norm(quats[i])
            q2 = quats[j] / np.linalg.norm(quats[j])
            dot = np.abs(np.dot(q1, q2))
            D[i, j] = 2 * np.arccos(np.clip(dot, -1, 1))
    return D

def robust_outlier_detection_dbscan(poses, trans_eps=0.03, rot_eps_deg=15, min_samples=2):
    # Translation clustering
    translations = np.stack([p[4:] for p in poses])
    dbscan_trans = DBSCAN(eps=trans_eps, min_samples=min_samples)
    trans_labels = dbscan_trans.fit_predict(translations)
    # Rotation clustering (using geodesic distance)
    quats = np.stack([p[:4] / np.linalg.norm(p[:4]) for p in poses])
    D_rot = geodesic_dist_matrix(quats)
    dbscan_rot = DBSCAN(eps=np.deg2rad(rot_eps_deg), min_samples=min_samples, metric='precomputed')
    rot_labels = dbscan_rot.fit_predict(D_rot)
    # Find largest cluster in both
    def largest_cluster(labels):
        vals, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) == 0:
            return np.array([], dtype=int)
        main_label = vals[np.argmax(counts)]
        return np.where(labels == main_label)[0]
    inlier_trans_idx = largest_cluster(trans_labels)
    inlier_rot_idx = largest_cluster(rot_labels)
    # Intersection: only those in both clusters
    inlier_idx = np.intersect1d(inlier_trans_idx, inlier_rot_idx)
    outlier_idx = np.setdiff1d(np.arange(len(poses)), inlier_idx)
    return inlier_idx, outlier_idx, trans_labels, rot_labels

def threshold_iterative_outlier_removal(poses, trans_thresh=0.03, rot_thresh_deg=15):
    """
    Iteratively remove the pose farthest from the mean pose (translation and rotation),
    until all remaining poses are within the thresholds from the mean.
    Returns inlier and outlier indices.
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

# --- Helper functions ---
def load_pose_txt(txt_path):
    arr = np.loadtxt(txt_path)
    if arr.shape[0] == 7:
        q = arr[:4]
        t = arr[4:7]
    elif arr.shape[0] == 8:
        q = arr[:4]
        t = arr[4:7]
    else:
        raise ValueError(f"Unexpected pose txt shape: {arr.shape}")
    R_mat = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

def load_extrinsics_yaml(yaml_path, serials):
    def create_mat(values):
        return np.array([values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]], dtype=np.float32)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    extr = data["extrinsics"]
    return {s: create_mat(extr[s]) for s in serials}

def mat_to_quat(T):
    # Returns [qx, qy, qz, qw, x, y, z]
    q = R.from_matrix(T[:3, :3]).as_quat()  # xyzw
    t = T[:3, 3]
    return np.concatenate([q, t])

if __name__ == "__main__":
    # User settings
    frame_idx = 156  # Set your frame index here
    ob_in_cam_dir = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/videos_0713/coffee_1_1/processed/fd_pose_solver/green_straw/ob_in_cam"
    extrinsics_yaml = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml"
    serials = [f"{i:02d}" for i in range(8)]
    threshold_factor = 0.05  # Try different values
    outlier_ratio = 0.2     # Try different values

    # Load all ob_in_cam for this frame
    ob_in_cam_poses = []
    for cam in serials:
        pose_txt = Path(ob_in_cam_dir) / cam / f"{frame_idx:06d}.txt"
        if not pose_txt.exists():
            print(f"[WARN] {pose_txt} not found, skipping.")
            continue
        T = load_pose_txt(pose_txt)
        ob_in_cam_poses.append(T)
    if len(ob_in_cam_poses) != len(serials):
        print(f"[WARN] Not all cameras have pose for frame {frame_idx}.")

    # Load extrinsics
    extrinsics_dict = load_extrinsics_yaml(extrinsics_yaml, serials)
    valid_RTs = [extrinsics_dict[cam] for cam in serials]

    # Transform to world
    ob_in_world_poses = [valid_RTs[i] @ ob_in_cam_poses[i] for i in range(len(ob_in_cam_poses))]
    # Preprocess: flip z axis if needed to match cam7 (index 7)
    ref_cam_idx = 7  # index in list, not string
    ob_in_world_poses = preprocess_flip_z_to_match_ref(ob_in_world_poses, ref_cam=ref_cam_idx)
    # Convert to [qx,qy,qz,qw,x,y,z] for each
    poses_quat = [mat_to_quat(T) for T in ob_in_world_poses]

    # Run outlier detection and print all math details
    rot_dists, trans_dists, pairwise_indices = calculate_pairwise_distances(poses_quat)
    print("\nAll pairwise rotation distances (rad):")
    for idx, (i, j) in enumerate(pairwise_indices):
        print(f"Cam {serials[i]} <-> Cam {serials[j]}: {rot_dists[idx]:.4f} rad ({np.rad2deg(rot_dists[idx]):.2f} deg)")
    print("\nAll pairwise translation distances (m):")
    for idx, (i, j) in enumerate(pairwise_indices):
        print(f"Cam {serials[i]} <-> Cam {serials[j]}: {trans_dists[idx]:.4f} m")
    print(f"\nRotation mean: {np.mean(rot_dists):.4f} rad, std: {np.std(rot_dists):.4f} rad")
    print(f"Translation mean: {np.mean(trans_dists):.4f} m, std: {np.std(trans_dists):.4f} m")

    # Print mean distance to others for each cam
    print("\nMean rotation/translation distance to others for each cam:")
    n = len(serials)
    rot_matrix = np.zeros((n, n))
    trans_matrix = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            rot_matrix[i, j] = rot_matrix[j, i] = rot_dists[idx]
            trans_matrix[i, j] = trans_matrix[j, i] = trans_dists[idx]
            idx += 1
    for i in range(n):
        print(f"Cam {serials[i]}: mean rot dist = {np.mean(rot_matrix[i, :]):.4f} rad, mean trans dist = {np.mean(trans_matrix[i, :]):.4f} m")

    # --- Old outlier detection ---
    inlier_rots, inlier_trans, is_rot_noisy, is_trans_noisy, rot_inlier_indices, trans_inlier_indices = detect_pose_outliers(
        poses_quat, threshold_factor, outlier_ratio)
    rot_thresh = np.mean(rot_dists) + threshold_factor * np.std(rot_dists)
    trans_thresh = np.mean(trans_dists) + threshold_factor * np.std(trans_dists)
    print(f"\n[Old method] Rotation inlier threshold: {rot_thresh:.4f} rad ({np.rad2deg(rot_thresh):.2f} deg)")
    print(f"[Old method] Translation inlier threshold: {trans_thresh:.4f} m")
    print(f"[Old method] Rotation outlier detected: {is_rot_noisy}")
    print(f"[Old method] Translation outlier detected: {is_trans_noisy}")
    print(f"[Old method] Rotation inlier indices: {rot_inlier_indices}")
    print(f"[Old method] Translation inlier indices: {trans_inlier_indices}")
    print(f"[Old method] Rotation outlier cams: {[serials[i] for i in range(len(serials)) if i not in rot_inlier_indices]}")
    print(f"[Old method] Translation outlier cams: {[serials[i] for i in range(len(serials)) if i not in trans_inlier_indices]}")

    # --- New DBSCAN-based outlier detection ---
    inlier_idx, outlier_idx, trans_labels, rot_labels = robust_outlier_detection_dbscan(
        poses_quat, trans_eps=0.03, rot_eps_deg=15, min_samples=2
    )
    print(f"\n[DBSCAN method] Inlier indices: {inlier_idx}")
    print(f"[DBSCAN method] Outlier indices: {outlier_idx}")
    print(f"[DBSCAN method] Inlier cams: {[serials[i] for i in inlier_idx]}")
    print(f"[DBSCAN method] Outlier cams: {[serials[i] for i in outlier_idx]}")
    print(f"[DBSCAN method] Translation cluster labels: {trans_labels}")
    print(f"[DBSCAN method] Rotation cluster labels: {rot_labels}")

    # --- New iterative threshold-based outlier removal ---
    inlier_idx_thr, outlier_idx_thr = threshold_iterative_outlier_removal(
        poses_quat, trans_thresh=0.03, rot_thresh_deg=15)
    print(f"\n[Iterative threshold method] Inlier indices: {inlier_idx_thr}")
    print(f"[Iterative threshold method] Outlier indices: {outlier_idx_thr}")
    print(f"[Iterative threshold method] Inlier cams: {[serials[i] for i in inlier_idx_thr]}")
    print(f"[Iterative threshold method] Outlier cams: {[serials[i] for i in outlier_idx_thr]}")

    # Compute consistent pose in world frame (mean of inliers)
    consistent_pose_w = get_consistent_pose_w(
        ob_in_cam_poses, valid_RTs, [],
        rot_thresh=5.0, trans_thresh=0.01, thresh_factor=threshold_factor, outlier_ratio=outlier_ratio)
    print("\nConsistent pose in world frame (mean of inliers):")
    print(f"[qx,qy,qz,qw,x,y,z,flag]: {consistent_pose_w}")
    print(f"Translation: {consistent_pose_w[4:7]}")
