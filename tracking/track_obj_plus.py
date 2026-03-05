"""
FoundationPose++ tracking adapted for HO-Cap.
Combines FoundationPose tracking with 2D tracker (Cutie) and Kalman filter.

Based on FoundationPose-plus-plus/src/obj_pose_track.py, 
adapted to use HO-Cap's FoundationPose interface.
"""

import os
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import logging
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
import trimesh
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation

# Add FoundationPose++ path for VOT and KalmanFilter
FP_PLUS_PLUS_PATH = Path(__file__).resolve().parent.parent.parent / "FoundationPose-plus-plus" / "src"
if str(FP_PLUS_PLUS_PATH) not in sys.path:
    sys.path.insert(0, str(FP_PLUS_PLUS_PATH))

# Import from FoundationPose++
from VOT import Cutie, Tracker_2D
from utils.kalman_filter_6d import KalmanFilter6D

# Import HO-Cap modules
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


# ==================== Pose Utility Functions ====================

def adjust_pose_to_image_point(
    ob_in_cam: torch.Tensor,
    K: torch.Tensor,
    x: float = -1.,
    y: float = -1.,
) -> torch.Tensor:
    """
    Adjusts the 6D pose(s) so that the projection matches the given 2D coordinate (x, y).

    Parameters:
    - ob_in_cam: Original 6D pose(s) as [4,4] or [B,4,4] tensor.
    - K: Camera intrinsic matrix (3x3 tensor).
    - x, y: Desired 2D coordinates on the image plane.

    Returns:
    - ob_in_cam_new: Adjusted pose(s) in same shape as input (tensor).
    """
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


def get_pose_xy_from_image_point(
    ob_in_cam: torch.Tensor,
    K: torch.Tensor,
    x: float = -1.,
    y: float = -1.,
) -> tuple:
    """
    Computes new (tx, ty) in camera space such that the projection matches image point (x, y).
    Returns Python floats (not tensors).
    """
    is_batched = ob_in_cam.ndim == 3
    if is_batched:
        ob_in_cam_new = ob_in_cam[0].cpu()
    else:
        ob_in_cam_new = ob_in_cam.cpu()

    if x == -1. or y == -1.:
        return x, y

    t = ob_in_cam_new[:3, 3]

    # Convert K to numpy if it's a tensor, otherwise use directly
    if torch.is_tensor(K):
        K_cpu = K.cpu() if K.is_cuda else K
        fx = float(K_cpu[0, 0])
        fy = float(K_cpu[1, 1])
        cx = float(K_cpu[0, 2])
        cy = float(K_cpu[1, 2])
    else:
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
    
    tz = float(t[2])

    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy

    return tx, ty


def get_mat_from_6d_pose_arr(pose_arr: np.ndarray) -> np.ndarray:
    """Convert 6D pose array [x, y, z, rx, ry, rz] to 4x4 transformation matrix."""
    xyz = pose_arr[:3]
    euler_angles = pose_arr[3:]
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz
    
    return transformation_matrix


def get_6d_pose_arr_from_mat(pose) -> np.ndarray:
    """Convert 4x4 transformation matrix to 6D pose array [x, y, z, rx, ry, rz]."""
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


def is_valid_ob_pose(ob_in_cam, x_threshold, y_threshold, z_threshold, cam_RT=None):
    """Check if the pose is within valid range."""
    if np.all(ob_in_cam == -1):
        return False
    if cam_RT is None:
        x, y, z = ob_in_cam[:3, 3]
    else:
        ob_in_world = cam_RT @ ob_in_cam
        x, y, z = ob_in_world[:3, 3]
    return (x_threshold[0] < x < x_threshold[1] and
            y_threshold[0] < y < y_threshold[1] and
            z_threshold[0] < z < z_threshold[1])


# ==================== Main Tracking Function ====================

def run_tracking_plus(
    sequence_folder: str,
    object_idx: int,
    camera_serial: str = None,
    est_refine_iter: int = 15,
    track_refine_iter: int = 20,
    start_frame: int = 0,
    end_frame: int = -1,
    activate_2d_tracker: bool = False,
    activate_kalman_filter: bool = False,
    kf_measurement_noise_scale: float = 0.05,
    use_masked_depth: bool = True,
    use_masked_image: bool = True,
    output_suffix: str = "tracking_plus",
    save_visualization: bool = True,
):
    """
    Run FoundationPose++ tracking on HO-Cap data.
    
    Args:
        sequence_folder: Path to the sequence folder
        object_idx: Object index (1-based)
        camera_serial: Camera serial to track (None for all cameras)
        est_refine_iter: Iterations for initial pose estimation
        track_refine_iter: Iterations for tracking refinement
        start_frame: Starting frame index
        end_frame: Ending frame index (-1 for all frames)
        activate_2d_tracker: Whether to use Cutie 2D tracker
        activate_kalman_filter: Whether to use Kalman filter for smoothing
        kf_measurement_noise_scale: Kalman filter measurement noise scale
        use_masked_depth: Whether to mask depth with object mask
        use_masked_image: Whether to mask color image with object mask
        output_suffix: Suffix for output folder name
        save_visualization: Whether to save visualization images
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
    
    # Filter cameras if specific serial is provided
    if camera_serial is not None:
        if camera_serial not in valid_serials:
            raise ValueError(f"Camera {camera_serial} not in valid serials: {valid_serials}")
        cam_indices = [valid_serials.index(camera_serial)]
        process_serials = [camera_serial]
    else:
        cam_indices = list(range(len(valid_serials)))
        process_serials = valid_serials
    
    # Load and prepare mesh
    mesh = trimesh.load(data_loader.object_cleaned_files[object_idx_0], process=True)
    if len(mesh.vertices) > 200000:
        mesh = mesh.simplify_quadric_decimation(0.8)
        print("[INFO] Mesh decimated due to high vertex count.")
    print(f"[INFO] Mesh - vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")
    
    # Compute bounding box for visualization
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    
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
    
    # Initialize pose estimator
    set_seed(0)
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
        # Clear Hydra global state if already initialized
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
    
    # Process each camera
    for cam_idx in cam_indices[3:]:
        serial = valid_serials[cam_idx]
        K = valid_Ks[cam_idx]
        K_tensor = torch.tensor(K, dtype=torch.float32)
        cam_RT = valid_RTs[cam_idx]
        
        print(f"\n[INFO] ===== Processing Camera {serial} =====")
        
        # Create output directories
        cam_pose_folder = save_folder / object_id / "ob_in_cam" / serial
        cam_pose_folder.mkdir(parents=True, exist_ok=True)
        
        if save_visualization:
            vis_folder = save_folder / object_id / "visualization" / serial
            vis_folder.mkdir(parents=True, exist_ok=True)
            if activate_2d_tracker:
                mask_vis_folder = save_folder / object_id / "mask_vis" / serial
                mask_vis_folder.mkdir(parents=True, exist_ok=True)
                bbox_vis_folder = save_folder / object_id / "bbox_vis" / serial
                bbox_vis_folder.mkdir(parents=True, exist_ok=True)
        
        # Reset 2D tracker's internal state for new camera sequence
        if activate_2d_tracker and hasattr(tracker_2D, 'cutie_processor'):
            # Fully reinitialize the processor (clear_memory doesn't reset object_manager)
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
                    mask_vis_path = str(mask_vis_folder / f"{frame_id:06d}.png") if save_visualization else None
                    bbox_vis_path = str(bbox_vis_folder / f"{frame_id:06d}.png") if save_visualization else None
                    tracker_2D.initialize(
                        color_orig,
                        init_info={"mask": init_mask},
                        mask_visualization_path=mask_vis_path,
                        bbox_visualization_path=bbox_vis_path,
                    )
                
                pose_seq[frame_idx] = pose.reshape(4, 4)
                prev_pose = pose.reshape(4, 4)
            
            else:
                # Subsequent frames: track with optional 2D guidance
                
                # 2D tracker guidance
                if activate_2d_tracker:
                    mask_vis_path = str(mask_vis_folder / f"{frame_id:06d}.png") if save_visualization else None
                    bbox_vis_path = str(bbox_vis_folder / f"{frame_id:06d}.png") if save_visualization else None
                    bbox_2d = tracker_2D.track(
                        color_orig,
                        mask_visualization_path=mask_vis_path,
                        bbox_visualization_path=bbox_vis_path,
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
                    ob_in_world = cam_RT @ prev_pose
                    print(f"  Frame {frame_id}: Previous pose in world: {ob_in_world}")
                    print(f"  Frame {frame_id}: Re-registering due to invalid previous pose...")
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
            
            # Save visualization
            if save_visualization and not np.all(pose_seq[frame_idx] == -1):
                try:
                    # Import visualization functions
                    from hocap_annotation.wrappers.foundationpose import (
                        draw_posed_3d_box,
                        draw_xyz_axis,
                    )
                except ImportError:
                    # Try alternative import
                    try:
                        from third_party.FoundationPose.Utils import draw_posed_3d_box, draw_xyz_axis
                    except ImportError:
                        save_visualization = False
                        print("  [WARN] Visualization functions not available.")
                
                if save_visualization:
                    center_pose = pose_seq[frame_idx] @ np.linalg.inv(to_origin)
                    vis_color = draw_posed_3d_box(K, img=color_orig, ob_in_cam=center_pose, bbox=bbox)
                    vis_color = draw_xyz_axis(
                        vis_color,
                        ob_in_cam=center_pose,
                        scale=0.1,
                        K=K,
                        thickness=3,
                        transparency=0,
                        is_input_rgb=True,
                    )
                    imageio.imwrite(str(vis_folder / f"{frame_id:06d}.png"), vis_color)
            
            # Save pose to file
            pose_quat = mat_to_quat(pose_seq[frame_idx])
            write_pose_to_txt(cam_pose_folder / f"{frame_id:06d}.txt", pose_quat)
        
        # Save pose sequence as npy
        pose_seq_array = np.array(pose_seq)
        np.save(cam_pose_folder / "poses.npy", pose_seq_array)
        
        print(f"  [INFO] Camera {serial} complete. Poses saved to {cam_pose_folder}")
        
        # Clear GPU memory after each camera
        torch.cuda.empty_cache()
    
    print(f"\n[INFO] All cameras processed. Results saved to {save_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FoundationPose++ tracking adapted for HO-Cap"
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
        "--camera_serial",
        type=str,
        default=None,
        help="Camera serial to track (None for all cameras)"
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
        help="Enable Kalman filter for pose smoothing"
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
        "--output_suffix",
        type=str,
        default="tracking_plus",
        help="Suffix for output folder name"
    )
    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="Disable visualization saving"
    )
    
    args = parser.parse_args()
    
    set_logging_format()
    logging.getLogger().setLevel(logging.WARNING)
    
    t_start = time.time()
    
    run_tracking_plus(
        sequence_folder=args.sequence_folder,
        object_idx=args.object_idx,
        camera_serial=args.camera_serial,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        activate_2d_tracker=args.activate_2d_tracker,
        activate_kalman_filter=args.activate_kalman_filter,
        kf_measurement_noise_scale=args.kf_measurement_noise_scale,
        use_masked_depth=not args.no_masked_depth,
        use_masked_image=not args.no_masked_image,
        output_suffix=args.output_suffix,
        save_visualization=not args.no_visualization,
    )
    
    print(f"\n[INFO] Total time: {time.time() - t_start:.2f}s")
    torch.cuda.empty_cache()

