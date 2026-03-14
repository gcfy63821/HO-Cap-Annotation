"""
FoundationPose++ tracking with Kalman filter for each camera, then integrate to world frame.

Combines:
- track_obj_plus.py: FoundationPose++ tracking with 2D tracker and Kalman filter
- 04-1-4_fd_pose_solver_separate_cluster.py: Separate tracking per camera + world integration
"""

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import copy
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R

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

# Import FoundationPose++ components
FP_PLUS_PLUS_PATH = Path(__file__).resolve().parent.parent.parent / "FoundationPose-plus-plus" / "src"
import sys
if str(FP_PLUS_PLUS_PATH) not in sys.path:
    sys.path.insert(0, str(FP_PLUS_PLUS_PATH))

from VOT import Cutie, Tracker_2D
from utils.kalman_filter_6d import KalmanFilter6D

# Import pose utility functions from track_obj_plus
# These functions are defined in track_obj_plus.py
def adjust_pose_to_image_point(ob_in_cam, K, x=-1., y=-1.):
    """Adjust pose to match 2D image point. Imported from track_obj_plus logic."""
    from scipy.spatial.transform import Rotation
    device = ob_in_cam.device
    dtype = ob_in_cam.dtype
    is_batched = ob_in_cam.ndim == 3
    if not is_batched:
        ob_in_cam = ob_in_cam.unsqueeze(0)
    B = ob_in_cam.shape[0]
    ob_in_cam_new = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)
    for i in range(B):
        R = ob_in_cam[i, :3, :3]
        t = ob_in_cam[i, :3, 3]
        tx, ty = get_pose_xy_from_image_point(ob_in_cam[i], K, x, y)
        t_new = torch.tensor([tx, ty, t[2]], device=device, dtype=dtype)
        ob_in_cam_new[i, :3, :3] = R
        ob_in_cam_new[i, :3, 3] = t_new
    return ob_in_cam_new if is_batched else ob_in_cam_new[0]

def get_pose_xy_from_image_point(ob_in_cam, K, x=-1., y=-1.):
    """Get pose xy from image point. Returns Python floats."""
    is_batched = ob_in_cam.ndim == 3
    if is_batched:
        ob_in_cam_new = ob_in_cam[0].cpu()
    else:
        ob_in_cam_new = ob_in_cam.cpu()
    if x == -1. or y == -1.:
        return x, y
    t = ob_in_cam_new[:3, 3]
    if torch.is_tensor(K):
        K_cpu = K.cpu() if K.is_cuda else K
        fx, fy, cx, cy = float(K_cpu[0, 0]), float(K_cpu[1, 1]), float(K_cpu[0, 2]), float(K_cpu[1, 2])
    else:
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    tz = float(t[2])
    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy
    return tx, ty

def get_mat_from_6d_pose_arr(pose_arr):
    """Convert 6D pose array to 4x4 matrix."""
    from scipy.spatial.transform import Rotation
    xyz = pose_arr[:3]
    euler_angles = pose_arr[3:]
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation.as_matrix()
    transformation_matrix[:3, 3] = xyz
    return transformation_matrix

def get_6d_pose_arr_from_mat(pose):
    """Convert 4x4 matrix to 6D pose array."""
    from scipy.spatial.transform import Rotation
    if torch.is_tensor(pose):
        is_batched = pose.ndim == 3
        if is_batched:
            pose_np = pose[0].cpu().numpy()
        else:
            pose_np = pose.cpu().numpy()
    else:
        pose_np = pose
    xyz = pose_np[:3, 3]
    rotation_matrix = pose_np[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    return np.r_[xyz, euler_angles]

# Import world integration functions - need to import from the separate_cluster file
# Since we can't easily import, we'll use a simpler approach: import the module and call functions
import importlib.util
separate_cluster_path = Path(__file__).parent / "04-1-4_fd_pose_solver_separate_cluster.py"
spec = importlib.util.spec_from_file_location("separate_cluster", separate_cluster_path)
separate_cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(separate_cluster)

# Get functions from the imported module
get_consistent_pose_w = separate_cluster.get_consistent_pose_w
is_valid_ob_pose = separate_cluster.is_valid_ob_pose


def run_tracking_with_kalman_and_integration(
    sequence_folder: str,
    object_idx: int,
    est_refine_iter: int = 15,
    track_refine_iter: int = 20,
    start_frame: int = 0,
    end_frame: int = -1,
    activate_2d_tracker: bool = False,
    activate_kalman_filter: bool = True,
    kf_measurement_noise_scale: float = 0.05,
    use_masked_depth: bool = True,
    use_masked_image: bool = True,
    rot_thresh: float = 1.0,
    trans_thresh: float = 0.02,
    output_suffix: str = "fd_pose_solver",
):
    """
    Run FoundationPose++ tracking with Kalman filter for each camera, then integrate to world frame.
    
    Args:
        sequence_folder: Path to the sequence folder
        object_idx: Object index (1-based)
        est_refine_iter: Iterations for initial pose estimation
        track_refine_iter: Iterations for tracking refinement
        start_frame: Starting frame index
        end_frame: Ending frame index (-1 for all frames)
        activate_2d_tracker: Whether to use Cutie 2D tracker
        activate_kalman_filter: Whether to use Kalman filter for smoothing
        kf_measurement_noise_scale: Kalman filter measurement noise scale
        use_masked_depth: Whether to mask depth with object mask
        use_masked_image: Whether to mask color image with object mask
        rot_thresh: Rotation threshold for world pose integration (degrees)
        trans_thresh: Translation threshold for world pose integration (meters)
        output_suffix: Suffix for output folder name
    """
    sequence_folder = Path(sequence_folder)
    object_idx_0 = object_idx - 1  # Convert to 0-based index
    
    # Load data
    data_loader = HOCapLoader(sequence_folder)
    num_frames = data_loader.num_frames
    object_id = data_loader.object_ids[object_idx_0]
    valid_serials = data_loader.get_valid_seg_serials()
    rs_serials = data_loader.rs_serials
    valid_serial_indices = [rs_serials.index(serial) for serial in valid_serials]
    valid_Ks = data_loader.rs_Ks[valid_serial_indices]
    valid_RTs = data_loader.extr2world[valid_serial_indices]
    
    # Get thresholds for pose validation
    x_threshold = data_loader._thresholds[:2]
    y_threshold = data_loader._thresholds[2:4]
    z_threshold = data_loader._thresholds[4:]
    print(f"[INFO] Thresholds - X: {x_threshold}, Y: {y_threshold}, Z: {z_threshold}")
    
    # Load and prepare mesh
    mesh = trimesh.load(data_loader.object_cleaned_files[object_idx_0], process=True)
    if len(mesh.vertices) > 200000:
        mesh = mesh.simplify_quadric_decimation(0.8)
        print("[INFO] Mesh decimated due to high vertex count.")
    print(f"[INFO] Mesh - vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")
    
    # Set frame range
    start_frame = max(start_frame, 0)
    end_frame = num_frames if end_frame < 0 or end_frame > num_frames else end_frame
    total_frames = end_frame - start_frame
    print(f"[INFO] Processing frames {start_frame} to {end_frame} ({total_frames} frames)")
    
    # Setup output folder
    save_folder = Path(
        f"{data_loader._data_folder.parent.parent.parent}/"
        f"{data_loader._folder_name}_annotated/{data_loader._task_name}/"
        f"{data_loader._sequence_name}/processed/{output_suffix}"
    )
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output folder: {save_folder}")
    SKIP_TRACKING = False
    
    # Initialize pose estimator
    set_seed(0)
    if not SKIP_TRACKING:
        estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),
            debug=0,
            debug_dir=save_folder / "debug" / object_id,
            rotation_grid_min_n_views=120,
            rotation_grid_inplane_step=60,
        )
        
        empty_mat_pose = np.full((4, 4), -1.0, dtype=np.float32)
        
        # Initialize 2D tracker ONCE outside the camera loop (Hydra can only be initialized once)
        if activate_2d_tracker:
            try:
                from hydra.core.global_hydra import GlobalHydra
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
            except ImportError:
                pass
            tracker_2D = Cutie()
            print(f"[INFO] Cutie 2D tracker initialized")
        else:
            tracker_2D = Tracker_2D()
    
        # ========== STEP 1: Track each camera separately ==========
        print("\n" + "="*60)
        print("STEP 1: Separate tracking for each camera")
        print("="*60)
    
    # Process each camera
    for cam_idx in range(len(valid_serials)):
        if SKIP_TRACKING:
            continue
        serial = valid_serials[cam_idx]
        K = valid_Ks[cam_idx]
        K_tensor = torch.tensor(K, dtype=torch.float32)
        cam_RT = valid_RTs[cam_idx]
        
        print(f"\n[INFO] ===== Processing Camera {serial} ({cam_idx + 1}/{len(valid_serials)}) =====")
        
        # Create output directories
        cam_pose_folder = save_folder / object_id / "ob_in_cam" / serial
        cam_pose_folder.mkdir(parents=True, exist_ok=True)
        
        # Reset 2D tracker's internal state for new camera sequence
        if activate_2d_tracker and hasattr(tracker_2D, 'cutie_processor'):
            from cutie.inference.inference_core import InferenceCore
            tracker_2D.cutie_processor = InferenceCore(tracker_2D.cutie, cfg=tracker_2D.cutie.cfg)
            tracker_2D.cutie_processor.max_internal_size = -1
            print(f"  [INFO] Cutie tracker processor reinitialized for camera {serial}")
        
        # Initialize Kalman filter if enabled
        if activate_kalman_filter:
            kf = KalmanFilter6D(kf_measurement_noise_scale)
            kf_mean, kf_covariance = None, None
            print(f"  [INFO] Kalman filter initialized (noise_scale={kf_measurement_noise_scale})")
        
        # Initialize pose storage
        pose_seq = [None] * total_frames
        prev_pose = empty_mat_pose.copy()
        
        for frame_id in range(start_frame, end_frame):
            frame_idx = frame_id - start_frame
            
            # Load frame data
            color = data_loader.get_color(serial, frame_id)
            depth = data_loader.get_depth(serial, frame_id)
            mask = data_loader.get_mask(serial, frame_id, object_idx_0)
            
            # Store original color for visualization
            color_orig = color.copy()
            
            # Apply masking if enabled
            if use_masked_depth:
                depth = depth.copy()
                depth[mask == 0] = 0
            
            if use_masked_image:
                color = color.copy()
                color[mask == 0] = 0
            
            # First frame: register pose and initialize tracker
            if frame_idx == 0:
                if mask.sum() < 10:
                    print(f"  Frame {frame_id}: Invalid mask (sum={mask.sum()}), cannot register.")
                    pose_seq[frame_idx] = empty_mat_pose.copy()
                    continue
                
                # Get initial translation
                init_ob_pos_center = data_loader.get_init_translation(
                    frame_id, [serial], object_idx_0, kernel_size=5
                )[0][0]
                
                if init_ob_pos_center is not None:
                    pose = estimator.register(
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        K=K,
                        iteration=est_refine_iter,
                        init_ob_pos_center=init_ob_pos_center,
                    )
                    print(f"  Frame {frame_id}: Registered initial pose.")
                else:
                    pose = estimator.register(
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        K=K,
                        iteration=est_refine_iter,
                    )
                    print(f"  Frame {frame_id}: Registered initial pose (no init center).")
                
                # Initialize Kalman filter
                if activate_kalman_filter:
                    kf_mean, kf_covariance = kf.initiate(get_6d_pose_arr_from_mat(pose))
                
                # Initialize 2D tracker
                if activate_2d_tracker:
                    init_mask = (mask > 0).astype(bool)
                    tracker_2D.initialize(
                        color_orig,
                        init_info={"mask": init_mask},
                        mask_visualization_path=None,
                        bbox_visualization_path=None,
                    )
                
                pose_seq[frame_idx] = pose.reshape(4, 4)
                prev_pose = pose.reshape(4, 4)
            
            else:
                # Subsequent frames: track with optional 2D guidance
                
                # 2D tracker guidance
                if activate_2d_tracker:
                    bbox_2d = tracker_2D.track(
                        color_orig,
                        mask_visualization_path=None,
                        bbox_visualization_path=None,
                    )
                    
                    # Adjust pose based on 2D bbox center
                    if bbox_2d[0] != -1 and estimator.pose_last is not None:
                        bbox_center_x = bbox_2d[0] + bbox_2d[2] / 2
                        bbox_center_y = bbox_2d[1] + bbox_2d[3] / 2
                        
                        if not activate_kalman_filter:
                            estimator.pose_last = adjust_pose_to_image_point(
                                ob_in_cam=estimator.pose_last,
                                K=K_tensor,
                                x=bbox_center_x,
                                y=bbox_center_y,
                            )
                        else:
                            # Update Kalman filter with 2D measurement
                            kf_mean, kf_covariance = kf.update(
                                kf_mean, kf_covariance,
                                get_6d_pose_arr_from_mat(estimator.pose_last)
                            )
                            measurement_xy = np.array(get_pose_xy_from_image_point(
                                ob_in_cam=estimator.pose_last,
                                K=K_tensor,
                                x=bbox_center_x,
                                y=bbox_center_y,
                            ))
                            kf_mean, kf_covariance = kf.update_from_xy(
                                kf_mean, kf_covariance, measurement_xy
                            )
                            estimator.pose_last = torch.from_numpy(
                                get_mat_from_6d_pose_arr(kf_mean[:6])
                            ).unsqueeze(0).float().to(estimator.pose_last.device)
                
                # Check if we have valid previous pose
                if mask.sum() < 10:
                    pose = empty_mat_pose.copy()
                    print(f"  Frame {frame_id}: Invalid mask, using empty pose.")
                elif is_valid_ob_pose(prev_pose, x_threshold, y_threshold, z_threshold, cam_RT):
                    # Track from previous pose
                    pose = estimator.track_one(
                        rgb=color,
                        depth=depth,
                        K=K,
                        iteration=track_refine_iter,
                        prev_pose=prev_pose,
                    )
                    
                    # Update Kalman filter prediction
                    if activate_2d_tracker and activate_kalman_filter:
                        kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)
                    
                    if frame_id % 50 == 0:
                        print(f"  Frame {frame_id}: Tracked from previous pose.")
                else:
                    # Re-register if previous pose is invalid
                    print(f"  Frame {frame_id}: Re-registering due to invalid previous pose...")
                    ob_in_world = cam_RT @ prev_pose
                    x, y, z = ob_in_world[:3, 3]
                    print(f"debug:{x}, {y}, {z}")
                    init_ob_pos_center = data_loader.get_init_translation(
                        frame_id, [serial], object_idx_0, kernel_size=5
                    )[0][0]
                    
                    if init_ob_pos_center is not None:
                        pose = estimator.register(
                            rgb=color,
                            depth=depth,
                            ob_mask=mask,
                            K=K,
                            iteration=est_refine_iter,
                            init_ob_pos_center=init_ob_pos_center,
                        )
                    else:
                        pose = empty_mat_pose.copy()
                        print(f"    No init position, using empty pose.")
                
                pose_seq[frame_idx] = pose.reshape(4, 4) if not np.all(pose == -1) else pose
                prev_pose = pose_seq[frame_idx]
            
            # Save pose to file
            pose_quat = mat_to_quat(pose_seq[frame_idx])
            write_pose_to_txt(cam_pose_folder / f"{frame_id:06d}.txt", pose_quat)
        
        print(f"  [INFO] Camera {serial} tracking complete. Saved to {cam_pose_folder}")
        
        # Clear GPU memory after each camera
        torch.cuda.empty_cache()
    
    # ========== STEP 2: Integrate all camera poses to world frame ==========
    print("\n" + "="*60)
    print("STEP 2: Integrate camera poses to world frame")
    print("="*60)
    print(f"[INFO] Valid serials: {valid_serials}")
    # Load all per-camera poses
    ob_in_cam_poses_per_cam = [[] for _ in range(len(valid_serials))]
    for serial_idx, serial in enumerate(valid_serials):
        print(f"[INFO] Processing camera: {serial}")
        for frame_id in range(start_frame, end_frame):
            pose_path = save_folder / object_id / "ob_in_cam" / serial / f"{frame_id:06d}.txt"
            if pose_path.exists():
                pose_quat = np.loadtxt(pose_path)
                ob_in_cam_poses_per_cam[serial_idx].append(pose_quat)
            else:
                # If file is missing, append a placeholder
                ob_in_cam_poses_per_cam[serial_idx].append(np.full(7, -1.0, dtype=np.float32))
    
    # Integrate poses frame by frame
    all_poses_w = []
    save_pose_folder = save_folder / object_id / "ob_in_world"
    save_pose_folder.mkdir(parents=True, exist_ok=True)
    
    for frame_idx in range(total_frames):
        frame_id = start_frame + frame_idx
        
        # Collect poses from all cameras for this frame
        ob_in_cam_poses = [ob_in_cam_poses_per_cam[serial_idx][frame_idx] for serial_idx in range(len(valid_serials))]
        
        # Convert quaternion poses to 4x4 matrices
        ob_in_cam_poses_mat = [quat_to_mat(pose[:7]) for pose in ob_in_cam_poses]
        
        # Get consistent pose in world frame
        curr_pose_w = get_consistent_pose_w(
            mat_poses_c=ob_in_cam_poses_mat,
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
        print(f"[RESULT] ob_in_world (Frame {frame_id}): {curr_pose_w[4:7]}")
        
        # Save world pose
        write_pose_to_txt(save_pose_folder / f"{frame_id:06d}.txt", curr_pose_w)
    
    print(f"\n[INFO] Integration complete! Results saved to {save_folder}")
    print(f"[INFO] Per-camera poses: {save_folder / object_id / 'ob_in_cam'}")
    print(f"[INFO] World poses: {save_pose_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FoundationPose++ tracking with Kalman filter + world integration"
    )
    parser.add_argument(
        "--sequence_folder",
        type=str,
        required=True,
        help="Path to the sequence folder"
    )
    parser.add_argument(
        "--object_idx",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Object index (1-based)"
    )
    parser.add_argument(
        "--est_refine_iter",
        type=int,
        default=15,
        help="Iterations for initial pose estimation"
    )
    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=20,
        help="Iterations for tracking refinement"
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Starting frame index"
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=-1,
        help="Ending frame index (-1 for all frames)"
    )
    parser.add_argument(
        "--activate_2d_tracker",
        action="store_true",
        help="Enable Cutie 2D tracker for pose guidance"
    )
    parser.add_argument(
        "--activate_kalman_filter",
        action="store_true",
        default=True,
        help="Enable Kalman filter for pose smoothing (default: True)"
    )
    parser.add_argument(
        "--kf_measurement_noise_scale",
        type=float,
        default=0.05,
        help="Kalman filter measurement noise scale"
    )
    parser.add_argument(
        "--no_masked_depth",
        action="store_true",
        help="Disable depth masking"
    )
    parser.add_argument(
        "--no_masked_image",
        action="store_true",
        help="Disable image masking"
    )
    parser.add_argument(
        "--rot_thresh",
        type=float,
        default=1.0,
        help="Rotation threshold for world integration (degrees)"
    )
    parser.add_argument(
        "--trans_thresh",
        type=float,
        default=0.02,
        help="Translation threshold for world integration (meters)"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="fd_pose_solver",
        help="Suffix for output folder name"
    )
    
    args = parser.parse_args()
    
    set_logging_format()
    logging.getLogger().setLevel(logging.WARNING)
    
    t_start = time.time()
    
    run_tracking_with_kalman_and_integration(
        sequence_folder=args.sequence_folder,
        object_idx=args.object_idx,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        activate_2d_tracker=args.activate_2d_tracker,
        activate_kalman_filter=args.activate_kalman_filter,
        kf_measurement_noise_scale=args.kf_measurement_noise_scale,
        use_masked_depth=not args.no_masked_depth,
        use_masked_image=not args.no_masked_image,
        rot_thresh=args.rot_thresh,
        trans_thresh=args.trans_thresh,
        output_suffix=args.output_suffix,
    )
    
    print(f"\n[INFO] Total time: {time.time() - t_start:.2f}s")
    torch.cuda.empty_cache()

