#!/usr/bin/env python3
"""
Real-time 6D object pose tracking with RealSense + FoundationPose++.

Run in the `hocap-annotation` conda env.

Prerequisite:
  Generate initial mask first using realtime_init_mask.py (in sam2 env).

Usage:
  conda activate hocap-annotation
  python tools/realtime/realtime_tracker.py \
    --mesh_path /path/to/object.obj \
    --init_dir /tmp/tracking_init \
    --apply_scale 0.01 \
    --track_refine_iter 5 \
    --activate_2d_tracker \
    --activate_kalman_filter

Controls:
  'r' - Re-register pose from current frame (using saved init mask projected)
  's' - Save current pose trajectory to init_dir/poses.npy
  'q' - Quit
"""

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import trimesh

# FoundationPose wrapper (from hocap_annotation)
from hocap_annotation.wrappers.foundationpose import (
    FoundationPose,
    PoseRefinePredictor,
    ScorePredictor,
    dr,
    set_logging_format,
    set_seed,
)

# FoundationPose++ components (Cutie 2D tracker + Kalman filter)
FP_PLUS_PLUS_PATH = Path(__file__).resolve().parent.parent.parent / "FoundationPose-plus-plus" / "src"
FP_PATH = Path(__file__).resolve().parent.parent.parent / "FoundationPose-plus-plus" / "FoundationPose"
if str(FP_PLUS_PLUS_PATH) not in sys.path:
    sys.path.insert(0, str(FP_PLUS_PLUS_PATH))
if str(FP_PATH) not in sys.path:
    sys.path.insert(0, str(FP_PATH))

from VOT import Cutie, Tracker_2D
from utils.kalman_filter_6d import KalmanFilter6D
from Utils import draw_posed_3d_box, draw_xyz_axis
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Pose conversion utilities (from 04-1-4_fd_pose_solver_kalman.py)
# ---------------------------------------------------------------------------
def adjust_pose_to_image_point(ob_in_cam, K, x=-1.0, y=-1.0):
    """Adjust pose translation so projection matches image point (x, y)."""
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


def get_pose_xy_from_image_point(ob_in_cam, K, x=-1.0, y=-1.0):
    """Compute (tx, ty) in camera space from desired image point."""
    is_batched = ob_in_cam.ndim == 3
    if is_batched:
        ob_in_cam_cpu = ob_in_cam[0].cpu()
    else:
        ob_in_cam_cpu = ob_in_cam.cpu()
    if x == -1.0 or y == -1.0:
        return x, y
    t = ob_in_cam_cpu[:3, 3]
    if torch.is_tensor(K):
        K_np = K.cpu() if K.is_cuda else K
        fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
        cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
    else:
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
    tz = float(t[2])
    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy
    return tx, ty


def get_6d_pose_arr_from_mat(pose):
    """Convert 4x4 matrix to [x, y, z, rx, ry, rz] array."""
    if torch.is_tensor(pose):
        is_batched = pose.ndim == 3
        pose_np = pose[0].cpu().numpy() if is_batched else pose.cpu().numpy()
    else:
        pose_np = pose
    xyz = pose_np[:3, 3]
    euler_angles = Rotation.from_matrix(pose_np[:3, :3]).as_euler("xyz", degrees=False)
    return np.r_[xyz, euler_angles]


def get_mat_from_6d_pose_arr(pose_arr):
    """Convert [x, y, z, rx, ry, rz] array to 4x4 matrix."""
    xyz = pose_arr[:3]
    rotation = Rotation.from_euler("xyz", pose_arr[3:], degrees=False)
    mat = np.eye(4)
    mat[:3, :3] = rotation.as_matrix()
    mat[:3, 3] = xyz
    return mat


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def init_realsense(width, height, fps):
    """Initialize RealSense pipeline with aligned depth. Returns (pipeline, align, K)."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1],
    ], dtype=np.float64)

    # Warm up for auto-exposure
    print("[INFO] Warming up camera (30 frames)...")
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, align, K


def get_frame(pipeline, align):
    """Get aligned RGB + depth from RealSense."""
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    color_bgr = np.asanyarray(color_frame.get_data())       # (H, W, 3) uint8 BGR
    depth = np.asanyarray(depth_frame.get_data()) / 1e3      # (H, W) float32 meters
    return color_bgr, depth.astype(np.float32)


# ---------------------------------------------------------------------------
# FoundationPose++ initialization
# ---------------------------------------------------------------------------
def init_foundation_pose(mesh_path, apply_scale, force_apply_color, apply_color):
    """Load mesh and create FoundationPose estimator."""
    mesh = trimesh.load(mesh_path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_scale(apply_scale)

    if force_apply_color:
        from FoundationPose.estimater import trimesh_add_pure_colored_texture
        mesh = trimesh_add_pure_colored_texture(
            mesh, color=np.array(apply_color), resolution=10
        )

    # Bounding box for visualization
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    print(f"[INFO] Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=0,
    )

    return est, mesh, to_origin, bbox


# ---------------------------------------------------------------------------
# Main tracking loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Real-time 6D object pose tracking with RealSense + FoundationPose++"
    )
    # Required
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to object mesh (.obj/.ply/.stl)")
    parser.add_argument("--init_dir", type=str, required=True,
                        help="Directory with init data (rgb.png, depth.npy, mask.npy, K.npy)")
    # Mesh
    parser.add_argument("--apply_scale", type=float, default=0.01,
                        help="Mesh scale factor (default 0.01 = cm to m)")
    parser.add_argument("--force_apply_color", action="store_true",
                        help="Force a texture color for colorless meshes")
    parser.add_argument("--apply_color", type=json.loads, default="[0, 159, 237]",
                        help="RGB color for colorless mesh")
    # Camera
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    # FoundationPose
    parser.add_argument("--est_refine_iter", type=int, default=15,
                        help="Refinement iterations for registration")
    parser.add_argument("--track_refine_iter", type=int, default=5,
                        help="Refinement iterations for tracking (lower = faster)")
    # FoundationPose++ modules
    parser.add_argument("--activate_2d_tracker", action="store_true",
                        help="Enable Cutie 2D tracker for xy guidance")
    parser.add_argument("--activate_kalman_filter", action="store_true",
                        help="Enable Kalman filter for pose smoothing")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05,
                        help="Kalman filter measurement noise scale")
    args = parser.parse_args()

    set_logging_format()
    logging.getLogger().setLevel(logging.WARNING)
    set_seed(0)

    init_dir = Path(args.init_dir)

    # --- Load init data ---
    print("[INFO] Loading init data from", init_dir)
    init_bgr = cv2.imread(str(init_dir / "rgb.png"))
    init_rgb = cv2.cvtColor(init_bgr, cv2.COLOR_BGR2RGB)
    init_depth = np.load(str(init_dir / "depth.npy"))
    init_mask = np.load(str(init_dir / "mask.npy"))
    init_K = np.load(str(init_dir / "K.npy"))

    print(f"  RGB: {init_rgb.shape}, Depth: {init_depth.shape}, Mask sum: {init_mask.sum()}")

    # --- Init FoundationPose ---
    print("[INFO] Initializing FoundationPose++...")
    est, mesh, to_origin, bbox = init_foundation_pose(
        args.mesh_path, args.apply_scale, args.force_apply_color, args.apply_color
    )

    # --- Register on init frame ---
    print("[INFO] Registering initial pose...")
    mask_uint8 = init_mask.astype(np.uint8) * 255
    pose = est.register(
        K=init_K, rgb=init_rgb, depth=init_depth,
        ob_mask=mask_uint8, iteration=args.est_refine_iter,
    )
    print(f"[INFO] Initial pose registered. Translation: {pose[:3, 3]}")

    # --- Init 2D tracker ---
    if args.activate_2d_tracker:
        tracker_2D = Cutie()
        tracker_2D.initialize(
            init_rgb,
            init_info={"mask": init_mask.astype(bool)},
            mask_visualization_path=None,
            bbox_visualization_path=None,
        )
        print("[INFO] Cutie 2D tracker initialized")
    else:
        tracker_2D = Tracker_2D()
        print("[INFO] 2D tracker disabled (passthrough)")

    # --- Init Kalman filter ---
    kf = None
    kf_mean, kf_cov = None, None
    if args.activate_kalman_filter:
        kf = KalmanFilter6D(args.kf_measurement_noise_scale)
        kf_mean, kf_cov = kf.initiate(get_6d_pose_arr_from_mat(pose))
        print("[INFO] Kalman filter initialized")

    # --- Init camera ---
    print("[INFO] Starting RealSense camera...")
    pipeline, align, live_K = init_realsense(args.width, args.height, args.fps)
    K_tensor = torch.tensor(live_K, dtype=torch.float32)
    print(f"[INFO] Live camera intrinsics:\n{live_K}")

    # --- Tracking state ---
    pose_history = [pose.copy()]
    frame_count = 0
    fps_smooth = 0.0

    win_name = "Real-time Tracking | r=re-register, s=save, q=quit"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    print("\n[INFO] Tracking started. Press 'q' to quit.\n")

    try:
        while True:
            t0 = time.time()

            # --- Capture frame ---
            color_bgr, depth = get_frame(pipeline, align)
            if color_bgr is None:
                continue
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            # --- 2D tracker guidance ---
            if args.activate_2d_tracker and est.pose_last is not None:
                bbox_2d = tracker_2D.track(
                    rgb,
                    mask_visualization_path=None,
                    bbox_visualization_path=None,
                )

                if bbox_2d[0] != -1:
                    bbox_cx = bbox_2d[0] + bbox_2d[2] / 2
                    bbox_cy = bbox_2d[1] + bbox_2d[3] / 2

                    if args.activate_kalman_filter and kf is not None:
                        # Kalman update with 2D measurement
                        kf_mean, kf_cov = kf.update(
                            kf_mean, kf_cov,
                            get_6d_pose_arr_from_mat(est.pose_last),
                        )
                        measurement_xy = np.array(get_pose_xy_from_image_point(
                            ob_in_cam=est.pose_last,
                            K=K_tensor,
                            x=bbox_cx,
                            y=bbox_cy,
                        ))
                        kf_mean, kf_cov = kf.update_from_xy(kf_mean, kf_cov, measurement_xy)
                        est.pose_last = (
                            torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6]))
                            .unsqueeze(0)
                            .float()
                            .to(est.pose_last.device)
                        )
                    else:
                        est.pose_last = adjust_pose_to_image_point(
                            ob_in_cam=est.pose_last,
                            K=K_tensor,
                            x=bbox_cx,
                            y=bbox_cy,
                        )

            # --- Track pose ---
            pose = est.track_one(
                rgb=rgb, depth=depth, K=live_K, iteration=args.track_refine_iter,
            )

            # --- Kalman predict (for next frame) ---
            if args.activate_kalman_filter and kf is not None:
                kf_mean, kf_cov = kf.predict(kf_mean, kf_cov)

            pose_history.append(pose.copy())

            # --- Visualize ---
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(
                live_K, img=rgb, ob_in_cam=center_pose, bbox=bbox
            )
            vis = draw_xyz_axis(
                vis, ob_in_cam=center_pose, scale=0.1, K=live_K,
                thickness=3, transparency=0, is_input_rgb=True,
            )

            # FPS counter
            dt = time.time() - t0
            fps_instant = 1.0 / max(dt, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant
            frame_count += 1

            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.putText(
                vis_bgr, f"FPS: {fps_smooth:.1f}  Frame: {frame_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            t_str = f"t=[{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]"
            cv2.putText(
                vis_bgr, t_str,
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )

            cv2.imshow(win_name, vis_bgr)

            # --- Key handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit.")
                break

            elif key == ord("r"):
                # Re-register using current frame + saved mask
                print("[INFO] Re-registering...")
                pose = est.register(
                    K=live_K, rgb=rgb, depth=depth,
                    ob_mask=mask_uint8, iteration=args.est_refine_iter,
                )
                if args.activate_kalman_filter and kf is not None:
                    kf_mean, kf_cov = kf.initiate(get_6d_pose_arr_from_mat(pose))
                if args.activate_2d_tracker:
                    # Re-init Cutie with the saved mask
                    tracker_2D.initialize(
                        rgb,
                        init_info={"mask": init_mask.astype(bool)},
                        mask_visualization_path=None,
                        bbox_visualization_path=None,
                    )
                print(f"[INFO] Re-registered. Translation: {pose[:3, 3]}")

            elif key == ord("s"):
                save_path = str(init_dir / "poses.npy")
                np.save(save_path, np.array(pose_history))
                print(f"[INFO] Saved {len(pose_history)} poses to {save_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        # Auto-save poses
        save_path = str(init_dir / "poses.npy")
        np.save(save_path, np.array(pose_history))
        print(f"[INFO] Saved {len(pose_history)} poses to {save_path}")
        print(f"[INFO] Average FPS: {fps_smooth:.1f}")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
