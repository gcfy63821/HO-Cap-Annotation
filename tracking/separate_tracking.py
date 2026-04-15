"""
Simplified separate tracking script for each camera view.
Based on 04-1-4_fd_pose_solver_separate_cluster.py but with simplified logic.
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
import trimesh

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


def run_separate_tracking(
    sequence_folder: str,
    object_idx: int,
    est_refine_iter: int = 15,
    track_refine_iter: int = 20,
    start_frame: int = 0,
    end_frame: int = -1,
    use_masked_depth: bool = True,
    use_masked_image: bool = True,
    output_suffix: str = "separate_tracking",
):
    """
    Run separate tracking for each camera view.
    
    Args:
        sequence_folder: Path to the sequence folder
        object_idx: Object index (1-based)
        est_refine_iter: Iterations for initial pose estimation
        track_refine_iter: Iterations for tracking refinement
        start_frame: Starting frame index
        end_frame: Ending frame index (-1 for all frames)
        use_masked_depth: Whether to mask depth with object mask
        use_masked_image: Whether to mask color image with object mask
        output_suffix: Suffix for output folder name
    """
    sequence_folder = Path(sequence_folder)
    object_idx = object_idx - 1  # Convert to 0-based index
    
    # Load data
    data_loader = HOCapLoader(sequence_folder)
    num_frames = data_loader.num_frames
    object_id = data_loader.object_ids[object_idx]
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
    mesh = trimesh.load(data_loader.object_cleaned_files[object_idx], process=True)
    if len(mesh.vertices) > 200000:
        mesh = mesh.simplify_quadric_decimation(0.8)
        print("[INFO] Mesh decimated due to high vertex count.")
    print(f"[INFO] Mesh - vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")
    
    # Set frame range
    start_frame = max(start_frame, 0)
    end_frame = num_frames if end_frame < 0 or end_frame > num_frames else end_frame
    print(f"[INFO] Processing frames {start_frame} to {end_frame}")
    
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
    
    # Track each camera separately
    for serial_idx, serial in enumerate(valid_serials):
        print(f"\n[INFO] ===== Processing Camera {serial} ({serial_idx + 1}/{len(valid_serials)}) =====")
        
        K = valid_Ks[serial_idx]
        cam_RT = valid_RTs[serial_idx]
        prev_pose = empty_mat_pose.copy()
        
        # Create output directory for this camera
        cam_save_folder = save_folder / object_id / "ob_in_cam" / serial
        cam_save_folder.mkdir(parents=True, exist_ok=True)
        
        for frame_id in range(start_frame, end_frame):
            # Load frame data
            color = data_loader.get_color(serial, frame_id)
            depth = data_loader.get_depth(serial, frame_id)
            mask = data_loader.get_mask(serial, frame_id, object_idx)
            
            # Apply masking if enabled
            if use_masked_depth:
                depth = depth.copy()
                depth[mask == 0] = 0
            
            if use_masked_image:
                color = color.copy()
                color[mask == 0] = 0
            
            # Determine tracking mode
            if mask.sum() < 10:
                # Invalid mask - skip this frame
                ob_in_cam_mat = empty_mat_pose.copy()
                print(f"  Frame {frame_id}: Invalid mask (sum={mask.sum()}), skipping.")
            elif is_valid_ob_pose(prev_pose, x_threshold, y_threshold, z_threshold, cam_RT):
                # Valid previous pose - track from it
                ob_in_cam_mat = estimator.track_one(
                    rgb=color,
                    depth=depth,
                    K=K,
                    iteration=track_refine_iter,
                    prev_pose=prev_pose,
                )
                if frame_id % 50 == 0:
                    print(f"  Frame {frame_id}: Tracked from previous pose.")
            else:
                # No valid previous pose - register new pose
                print(f"  Frame {frame_id}: Registering new pose...")
                init_ob_pos_center = data_loader.get_init_translation(
                    frame_id, [serial], object_idx, kernel_size=5
                )[0][0]
                
                if init_ob_pos_center is not None:
                    ob_in_cam_mat = estimator.register(
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        K=K,
                        iteration=est_refine_iter,
                        init_ob_pos_center=init_ob_pos_center,
                    )
                    # Validate registered pose
                    if not is_valid_ob_pose(ob_in_cam_mat, x_threshold, y_threshold, z_threshold, cam_RT):
                        print(f"    Registration failed validation, using empty pose.")
                        ob_in_cam_mat = empty_mat_pose.copy()
                else:
                    print(f"    No init position available, using empty pose.")
                    ob_in_cam_mat = empty_mat_pose.copy()
            
            # Update previous pose for next iteration
            prev_pose = ob_in_cam_mat
            
            # Save pose
            pose_quat = mat_to_quat(ob_in_cam_mat)
            write_pose_to_txt(cam_save_folder / f"{frame_id:06d}.txt", pose_quat)
        
        print(f"[INFO] Camera {serial} tracking complete. Saved to {cam_save_folder}")
    
    print(f"\n[INFO] All cameras processed. Results saved to {save_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simplified separate tracking for each camera view"
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
        default="separate_tracking",
        help="Suffix for output folder name"
    )
    
    args = parser.parse_args()
    
    set_logging_format()
    logging.getLogger().setLevel(logging.WARNING)
    
    t_start = time.time()
    
    run_separate_tracking(
        sequence_folder=args.sequence_folder,
        object_idx=args.object_idx,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        use_masked_depth=not args.no_masked_depth,
        use_masked_image=not args.no_masked_image,
        output_suffix=args.output_suffix,
    )
    
    print(f"\n[INFO] Total time: {time.time() - t_start:.2f}s")

