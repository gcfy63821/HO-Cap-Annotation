#!/usr/bin/env python3
"""
Capture a single frame from RealSense and generate an object mask via SAM2.

Run in the `sam2` conda env (requires pyrealsense2 + sam2).

Usage:
  conda activate sam2
  python tools/realtime/realtime_init_mask.py \
    --save_dir /tmp/tracking_init \
    --width 640 --height 480

Controls:
  - Live preview opens automatically
  - Press 's' to freeze the current frame
  - Click on the object (positive points). Right-click for negative points.
  - Press Enter to confirm selection and generate mask
  - Press 'r' to retry (unfreeze and pick a new frame)
  - Press 'q' to quit without saving

Output (saved to --save_dir):
  rgb.png       - Color image (H, W, 3) uint8 RGB
  depth.npy     - Depth map (H, W) float32 in meters
  mask.npy      - Binary mask (H, W) bool
  K.npy         - Camera intrinsic matrix (3, 3) float64
  mask_vis.png  - Visualization of mask overlaid on image
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Click handler state
# ---------------------------------------------------------------------------
click_points = []
click_labels = []  # 1 = positive, 0 = negative


def mouse_callback(event, x, y, flags, param):
    """Record left-clicks as positive, right-clicks as negative."""
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append([x, y])
        click_labels.append(1)
        print(f"  [+] point ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        click_points.append([x, y])
        click_labels.append(0)
        print(f"  [-] point ({x}, {y})")


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------
def init_realsense(width, height, fps):
    """Initialize RealSense pipeline with aligned depth."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Extract intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1],
    ], dtype=np.float64)

    # Warm up (auto-exposure stabilization)
    print("[INFO] Warming up camera (30 frames)...")
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, align, K


def get_frame(pipeline, align):
    """Capture one aligned RGB-D frame."""
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None

    color = np.asanyarray(color_frame.get_data())        # BGR uint8
    depth = np.asanyarray(depth_frame.get_data()) / 1e3   # uint16 mm -> float32 m
    depth = depth.astype(np.float32)
    return color, depth


# ---------------------------------------------------------------------------
# SAM2 mask generation
# ---------------------------------------------------------------------------
def generate_mask_sam2(image_rgb, points, labels, sam2_dir):
    """Run SAM2 image predictor with point prompts."""
    import torch

    # Add sam2 repo to path
    if sam2_dir not in sys.path:
        sys.path.insert(0, sam2_dir)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    checkpoint = os.path.join(sam2_dir, "checkpoints", "sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    print("[INFO] Loading SAM2 model...")
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    predictor.set_image(image_rgb)

    input_point = np.array(points)
    input_label = np.array(labels)

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Pick the best mask
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx].astype(bool)
    print(f"[INFO] Best mask score: {scores[best_idx]:.3f}")

    # Clean up GPU
    del predictor, sam2_model
    torch.cuda.empty_cache()

    return best_mask


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_points_on_image(img, points, labels):
    """Draw click points on image. Green = positive, Red = negative."""
    vis = img.copy()
    for (x, y), label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(vis, (x, y), 5, color, -1)
        cv2.circle(vis, (x, y), 7, (255, 255, 255), 1)
    return vis


def create_mask_visualization(image_rgb, mask):
    """Create a visualization with mask overlay."""
    vis = image_rgb.copy()
    overlay = vis.copy()
    overlay[mask] = [30, 144, 255]  # Blue overlay on mask region
    vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

    # Draw contours
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global click_points, click_labels

    parser = argparse.ArgumentParser(description="Capture frame + SAM2 mask for real-time tracking init")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sam2_dir", type=str,
                        default="/home/ruoqu/crq_ws/robotool/mesh_reconstruction/sam2",
                        help="Path to sam2 repo root")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Init camera
    pipeline, align, K = init_realsense(args.width, args.height, args.fps)
    print(f"[INFO] Camera ready. Intrinsics:\n{K}")

    # --- Phase 1: Live preview, freeze a frame ---
    win_name = "RealSense - Press 's' to freeze, 'q' to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    frozen_bgr = None
    frozen_depth = None

    print("\n[INFO] Live preview. Press 's' to freeze a frame.")

    while True:
        color_bgr, depth = get_frame(pipeline, align)
        if color_bgr is None:
            continue

        cv2.imshow(win_name, color_bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            frozen_bgr = color_bgr.copy()
            frozen_depth = depth.copy()
            print("[INFO] Frame frozen. Click on the object, then press Enter.")
            break
        elif key == ord('q'):
            print("[INFO] Quit without saving.")
            pipeline.stop()
            cv2.destroyAllWindows()
            return

    # Stop camera streaming (we have our frame)
    pipeline.stop()

    # --- Phase 2: Click on the frozen frame ---
    win_name = "Click object (+left, -right), Enter=confirm, r=retry"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, mouse_callback)

    click_points = []
    click_labels = []

    while True:
        vis = draw_points_on_image(frozen_bgr, click_points, click_labels)
        cv2.imshow(win_name, vis)
        key = cv2.waitKey(30) & 0xFF

        if key == 13:  # Enter
            if len(click_points) == 0:
                print("[WARNING] No points selected. Click on the object first.")
                continue
            break
        elif key == ord('r'):
            # Clear points
            click_points = []
            click_labels = []
            print("[INFO] Points cleared. Click again.")
        elif key == ord('q'):
            print("[INFO] Quit without saving.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # --- Phase 3: Generate mask with SAM2 ---
    frozen_rgb = cv2.cvtColor(frozen_bgr, cv2.COLOR_BGR2RGB)
    mask = generate_mask_sam2(frozen_rgb, click_points, click_labels, args.sam2_dir)

    # --- Phase 4: Save outputs ---
    # RGB image
    rgb_path = os.path.join(args.save_dir, "rgb.png")
    cv2.imwrite(rgb_path, cv2.cvtColor(frozen_rgb, cv2.COLOR_RGB2BGR))

    # Depth
    depth_path = os.path.join(args.save_dir, "depth.npy")
    np.save(depth_path, frozen_depth)

    # Mask
    mask_path = os.path.join(args.save_dir, "mask.npy")
    np.save(mask_path, mask)

    # Intrinsics
    k_path = os.path.join(args.save_dir, "K.npy")
    np.save(k_path, K)

    # Visualization
    vis = create_mask_visualization(frozen_rgb, mask)
    vis_path = os.path.join(args.save_dir, "mask_vis.png")
    cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"\n[INFO] Saved to {args.save_dir}/")
    print(f"  rgb.png      - Color image ({frozen_rgb.shape})")
    print(f"  depth.npy    - Depth map ({frozen_depth.shape}, meters)")
    print(f"  mask.npy     - Binary mask ({mask.shape})")
    print(f"  K.npy        - Intrinsics (3x3)")
    print(f"  mask_vis.png - Visualization")


if __name__ == "__main__":
    main()
