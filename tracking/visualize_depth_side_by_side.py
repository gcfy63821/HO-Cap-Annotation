#!/usr/bin/env python3
"""
Create side-by-side visualization videos for each camera view.
Left side: RGB video
Right side: Colorized depth video

Also supports edge alignment visualization to check RGB-Depth calibration.

Usage:
    python visualize_depth_side_by_side.py --input_folder /path/to/video_folder
    python visualize_depth_side_by_side.py --input_folder /path/to/video_folder --mode edge_overlay
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def extract_depth_edges(depth_frame: np.ndarray, threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
    """
    Extract edges from depth frame using Canny edge detection.
    
    Args:
        depth_frame: 16-bit depth frame
        threshold1: Lower threshold for Canny
        threshold2: Upper threshold for Canny
    
    Returns:
        Binary edge mask (255 for edges, 0 otherwise)
    """
    # Normalize depth to 8-bit for edge detection
    valid_mask = depth_frame > 0
    if not valid_mask.any():
        return np.zeros(depth_frame.shape, dtype=np.uint8)
    
    depth_normalized = np.zeros(depth_frame.shape, dtype=np.uint8)
    depth_valid = depth_frame[valid_mask]
    min_d, max_d = depth_valid.min(), depth_valid.max()
    if max_d > min_d:
        depth_normalized[valid_mask] = ((depth_frame[valid_mask] - min_d) / (max_d - min_d) * 255).astype(np.uint8)
    
    # Apply Gaussian blur to reduce noise
    depth_blurred = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
    
    # Extract edges
    edges = cv2.Canny(depth_blurred, threshold1, threshold2)
    
    # Also detect depth discontinuities (large depth jumps)
    # Compute gradient magnitude
    sobelx = cv2.Sobel(depth_frame.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(depth_frame.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Threshold for depth discontinuities (large depth changes indicate object boundaries)
    depth_edges = (gradient_mag > 100).astype(np.uint8) * 255
    
    # Combine both edge types
    combined_edges = cv2.bitwise_or(edges, depth_edges)
    
    return combined_edges


def overlay_edges_on_rgb(rgb_frame: np.ndarray, edges: np.ndarray, color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Overlay edge mask on RGB image.
    
    Args:
        rgb_frame: RGB image (BGR format)
        edges: Binary edge mask
        color: Color for edges (BGR)
    
    Returns:
        RGB image with edges overlaid
    """
    result = rgb_frame.copy()
    edge_mask = edges > 0
    result[edge_mask] = color
    return result


def blend_rgb_with_depth(rgb_frame: np.ndarray, depth_colored: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend RGB with colorized depth for alignment visualization.
    
    Args:
        rgb_frame: RGB image
        depth_colored: Colorized depth image
        alpha: Blend factor (0=RGB only, 1=depth only)
    
    Returns:
        Blended image
    """
    # Only blend where depth is valid
    valid_mask = np.any(depth_colored > 0, axis=2)
    
    result = rgb_frame.copy()
    result[valid_mask] = cv2.addWeighted(
        rgb_frame[valid_mask].reshape(-1, 3), 1 - alpha,
        depth_colored[valid_mask].reshape(-1, 3), alpha,
        0
    ).reshape(-1, 3)
    
    return result


def create_smooth_colormap():
    """
    Create a smooth continuous colormap for depth visualization.
    Transitions: Blue (near) -> Cyan -> Green -> Yellow -> Red (far)
    """
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    
    for i in range(256):
        t = i / 255.0
        
        if t < 0.25:
            # Blue to Cyan
            ratio = t / 0.25
            r, g, b = 0, int(255 * ratio), 255
        elif t < 0.5:
            # Cyan to Green
            ratio = (t - 0.25) / 0.25
            r, g, b = 0, 255, int(255 * (1 - ratio))
        elif t < 0.75:
            # Green to Yellow
            ratio = (t - 0.5) / 0.25
            r, g, b = int(255 * ratio), 255, 0
        else:
            # Yellow to Red
            ratio = (t - 0.75) / 0.25
            r, g, b = 255, int(255 * (1 - ratio)), 0
        
        colormap[i, 0] = [b, g, r]  # BGR format
    
    return colormap


def create_plasma_colormap():
    """
    Create a plasma-like colormap with very smooth transitions.
    Purple (near) -> Blue -> Pink -> Orange -> Yellow (far)
    """
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    
    colors = [
        (13, 8, 135),      # Dark purple
        (84, 2, 163),      # Purple
        (139, 10, 165),    # Magenta
        (185, 50, 137),    # Pink
        (219, 92, 104),    # Salmon
        (244, 136, 73),    # Orange
        (254, 188, 43),    # Yellow-orange
        (240, 249, 33),    # Yellow
    ]
    
    n_colors = len(colors)
    for i in range(256):
        t = i / 255.0 * (n_colors - 1)
        idx = min(int(t), n_colors - 2)
        frac = t - idx
        
        c1, c2 = colors[idx], colors[idx + 1]
        r = int(c1[0] * (1 - frac) + c2[0] * frac)
        g = int(c1[1] * (1 - frac) + c2[1] * frac)
        b = int(c1[2] * (1 - frac) + c2[2] * frac)
        
        colormap[i, 0] = [b, g, r]
    
    return colormap


def create_grayscale_colormap():
    """Create a simple grayscale colormap (near=white, far=black)."""
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        val = 255 - i
        colormap[i, 0] = [val, val, val]
    return colormap


# Pre-create custom colormaps
SMOOTH_COLORMAP = create_smooth_colormap()
PLASMA_COLORMAP = create_plasma_colormap()
GRAYSCALE_COLORMAP = create_grayscale_colormap()


def colorize_depth(
    depth_frame: np.ndarray, 
    min_depth: float = 0.1, 
    max_depth: float = 3.0,
    colormap: str = "smooth",
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Convert 16-bit depth frame to colorized visualization.
    
    Args:
        depth_frame: 16-bit depth frame (values in mm typically)
        min_depth: Minimum depth in meters for colormap normalization
        max_depth: Maximum depth in meters for colormap normalization
        colormap: Colormap to use:
            - "smooth": Custom smooth rainbow (blue->cyan->green->yellow->red)
            - "plasma": Plasma-like smooth gradient (purple->pink->orange->yellow)
            - "grayscale": Simple grayscale (near=white, far=black)
            - "turbo": OpenCV TURBO colormap (original)
            - "viridis": OpenCV VIRIDIS colormap (perceptually uniform)
        gamma: Gamma correction (>1 = emphasize near, <1 = emphasize far)
    
    Returns:
        Colorized depth image (BGR format)
    """
    # Convert to float meters (assuming depth is in mm)
    depth_m = depth_frame.astype(np.float32) / 1000.0
    
    # Clip to valid range
    depth_clipped = np.clip(depth_m, min_depth, max_depth)
    
    # Normalize to 0-1
    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)
    
    # Apply gamma correction for smoother appearance
    if gamma != 1.0:
        depth_normalized = np.power(depth_normalized, gamma)
    
    # Convert to 0-255
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Apply colormap
    if colormap == "smooth":
        depth_colored = cv2.applyColorMap(depth_uint8, SMOOTH_COLORMAP)
    elif colormap == "plasma":
        depth_colored = cv2.applyColorMap(depth_uint8, PLASMA_COLORMAP)
    elif colormap == "grayscale":
        depth_colored = cv2.applyColorMap(depth_uint8, GRAYSCALE_COLORMAP)
    elif colormap == "viridis":
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
    else:  # turbo (default/fallback)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    
    # Set invalid depth pixels (depth=0) to black
    depth_colored[depth_frame == 0] = [0, 0, 0]
    
    return depth_colored


def create_side_by_side_video(
    rgb_video_path: str,
    depth_video_path: str,
    output_path: str,
    min_depth: float = 0.1,
    max_depth: float = 3.0,
    fps: float = None,
    mode: str = "side_by_side",
    blend_alpha: float = 0.5,
    colormap: str = "smooth",
    gamma: float = 1.0,
):
    """
    Create visualization video for RGB and depth alignment checking.
    
    Args:
        rgb_video_path: Path to RGB video file
        depth_video_path: Path to depth video file
        output_path: Path to output video file
        min_depth: Minimum depth for colormap normalization (meters)
        max_depth: Maximum depth for colormap normalization (meters)
        fps: Output video FPS (None to use RGB video FPS)
        mode: Visualization mode:
            - "side_by_side": RGB on left, colorized depth on right
            - "edge_overlay": Depth edges overlaid on RGB (green lines)
            - "blend": RGB blended with colorized depth
            - "quad": 4-panel view (RGB, Depth, Edge overlay, Blend)
        blend_alpha: Blend factor for "blend" mode (0=RGB, 1=depth)
        colormap: Colormap for depth (smooth, plasma, grayscale, viridis, turbo)
        gamma: Gamma correction for depth (>1 = emphasize near, <1 = emphasize far)
    """
    # Open video captures
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    depth_cap = cv2.VideoCapture(depth_video_path)
    
    if not rgb_cap.isOpened():
        raise ValueError(f"Cannot open RGB video: {rgb_video_path}")
    if not depth_cap.isOpened():
        raise ValueError(f"Cannot open depth video: {depth_video_path}")
    
    # Get video properties
    width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    rgb_frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    depth_frame_count = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps is None:
        fps = rgb_fps
    
    print(f"  RGB: {width}x{height}, {rgb_fps:.2f} fps, {rgb_frame_count} frames")
    print(f"  Depth: {depth_frame_count} frames")
    print(f"  Mode: {mode}")
    
    # Determine output dimensions based on mode
    if mode == "side_by_side":
        output_width, output_height = width * 2, height
    elif mode in ["edge_overlay", "blend"]:
        output_width, output_height = width, height
    elif mode == "quad":
        output_width, output_height = width * 2, height * 2
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not writer.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")
    
    frame_idx = 0
    min_frames = min(rgb_frame_count, depth_frame_count)
    
    while True:
        ret_rgb, rgb_frame = rgb_cap.read()
        ret_depth, depth_frame = depth_cap.read()
        
        if not ret_rgb or not ret_depth:
            break
        
        # Handle depth frame (might be read as 3-channel, need to extract single channel)
        if len(depth_frame.shape) == 3:
            if depth_frame.dtype == np.uint8:
                depth_16bit = depth_frame[:, :, 0].astype(np.uint16) + (depth_frame[:, :, 1].astype(np.uint16) << 8)
            else:
                depth_16bit = depth_frame[:, :, 0]
        else:
            depth_16bit = depth_frame
        
        # Resize depth to match RGB if necessary
        if depth_16bit.shape[:2] != (height, width):
            depth_16bit = cv2.resize(depth_16bit, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Colorize depth
        depth_colored = colorize_depth(depth_16bit, min_depth, max_depth, colormap, gamma)
        
        # Create output based on mode
        if mode == "side_by_side":
            output_frame = np.hstack([rgb_frame, depth_colored])
            cv2.putText(output_frame, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(output_frame, "Depth", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        elif mode == "edge_overlay":
            # Extract depth edges and overlay on RGB
            depth_edges = extract_depth_edges(depth_16bit)
            output_frame = overlay_edges_on_rgb(rgb_frame, depth_edges, color=(0, 255, 0))
            cv2.putText(output_frame, "RGB + Depth Edges", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        elif mode == "blend":
            # Blend RGB with colorized depth
            output_frame = blend_rgb_with_depth(rgb_frame, depth_colored, blend_alpha)
            cv2.putText(output_frame, f"RGB+Depth Blend ({blend_alpha:.1f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        elif mode == "quad":
            # 4-panel view: RGB, Depth, Edge overlay, Blend
            depth_edges = extract_depth_edges(depth_16bit)
            edge_overlay = overlay_edges_on_rgb(rgb_frame, depth_edges, color=(0, 255, 0))
            blended = blend_rgb_with_depth(rgb_frame, depth_colored, blend_alpha)
            
            # Add labels
            rgb_labeled = rgb_frame.copy()
            cv2.putText(rgb_labeled, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            depth_labeled = depth_colored.copy()
            cv2.putText(depth_labeled, "Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(edge_overlay, "Depth Edges on RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(blended, f"Blend ({blend_alpha:.1f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            top_row = np.hstack([rgb_labeled, depth_labeled])
            bottom_row = np.hstack([edge_overlay, blended])
            output_frame = np.vstack([top_row, bottom_row])
        
        writer.write(output_frame)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{min_frames} frames")
    
    rgb_cap.release()
    depth_cap.release()
    writer.release()
    
    print(f"  Saved: {output_path} ({frame_idx} frames)")


def process_folder(
    input_folder: str,
    output_subfolder: str = "depth_visualization",
    min_depth: float = 0.1,
    max_depth: float = 3.0,
    camera_indices: list = None,
    mode: str = "side_by_side",
    blend_alpha: float = 0.5,
    colormap: str = "smooth",
    gamma: float = 1.0,
):
    """
    Process all camera videos in a folder and create visualizations.
    
    Args:
        input_folder: Folder containing camX_rgb.mp4 and camX_depth.mkv files
        output_subfolder: Name of subfolder for output videos
        min_depth: Minimum depth for colormap normalization (meters)
        max_depth: Maximum depth for colormap normalization (meters)
        camera_indices: List of camera indices to process (None for all found)
        mode: Visualization mode (side_by_side, edge_overlay, blend, quad)
        blend_alpha: Blend factor for blend mode
        colormap: Colormap for depth visualization (smooth, plasma, grayscale, viridis, turbo)
        gamma: Gamma correction for depth
    """
    input_path = Path(input_folder)
    output_path = input_path / output_subfolder
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Input folder: {input_path}")
    print(f"Output folder: {output_path}")
    
    # Find all camera pairs
    if camera_indices is None:
        # Auto-detect cameras
        camera_indices = []
        for i in range(20):  # Check cam0 to cam19
            rgb_file = input_path / f"cam{i}_rgb.mp4"
            depth_file = input_path / f"cam{i}_depth.mkv"
            if rgb_file.exists() and depth_file.exists():
                camera_indices.append(i)
    
    if not camera_indices:
        print("No camera video pairs found!")
        return
    
    print(f"Found cameras: {camera_indices}")
    print(f"Depth range: {min_depth}m - {max_depth}m")
    print(f"Visualization mode: {mode}")
    print()
    
    # Determine output filename suffix based on mode
    mode_suffix = {
        "side_by_side": "side_by_side",
        "edge_overlay": "edge_overlay",
        "blend": "blend",
        "quad": "quad_view",
    }
    suffix = mode_suffix.get(mode, mode)
    
    # Process each camera
    for cam_idx in camera_indices:
        rgb_video = input_path / f"cam{cam_idx}_rgb.mp4"
        depth_video = input_path / f"cam{cam_idx}_depth.mkv"
        output_video = output_path / f"cam{cam_idx}_{suffix}.mp4"
        
        if not rgb_video.exists():
            print(f"[WARN] RGB video not found: {rgb_video}")
            continue
        if not depth_video.exists():
            print(f"[WARN] Depth video not found: {depth_video}")
            continue
        
        print(f"Processing camera {cam_idx}...")
        try:
            create_side_by_side_video(
                str(rgb_video),
                str(depth_video),
                str(output_video),
                min_depth=min_depth,
                max_depth=max_depth,
                mode=mode,
                blend_alpha=blend_alpha,
                colormap=colormap,
                gamma=gamma,
            )
        except Exception as e:
            print(f"  [ERROR] Failed to process camera {cam_idx}: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create RGB and depth visualization videos for alignment checking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Visualization modes:
  side_by_side  - RGB on left, colorized depth on right (default)
  edge_overlay  - Depth edges (green) overlaid on RGB to check alignment
  blend         - RGB blended with colorized depth
  quad          - 4-panel view with all visualizations

Colormaps (for continuous depth visualization):
  smooth    - Smooth rainbow: blue->cyan->green->yellow->red (default, most continuous)
  plasma    - Plasma-like: purple->pink->orange->yellow (perceptually smooth)
  grayscale - Simple grayscale: near=white, far=black
  viridis   - Perceptually uniform (good for scientific visualization)
  turbo     - Original OpenCV TURBO (more banded)

Examples:
  # Basic side-by-side with smooth colormap
  python visualize_depth_side_by_side.py --input_folder /path/to/folder

  # Use plasma colormap for smoother visualization
  python visualize_depth_side_by_side.py --input_folder /path/to/folder --colormap plasma

  # Grayscale depth with gamma correction (emphasize near objects)
  python visualize_depth_side_by_side.py --input_folder /path/to/folder --colormap grayscale --gamma 1.5

  # Check edge alignment
  python visualize_depth_side_by_side.py --input_folder /path/to/folder --mode edge_overlay
        """
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing camX_rgb.mp4 and camX_depth.mkv files"
    )
    parser.add_argument(
        "--output_subfolder",
        type=str,
        default="depth_visualization",
        help="Name of subfolder for output videos (default: depth_visualization)"
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.1,
        help="Minimum depth for colormap normalization in meters (default: 0.1)"
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=3.0,
        help="Maximum depth for colormap normalization in meters (default: 3.0)"
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated camera indices to process (default: auto-detect all)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="side_by_side",
        choices=["side_by_side", "edge_overlay", "blend", "quad"],
        help="Visualization mode (default: side_by_side)"
    )
    parser.add_argument(
        "--blend_alpha",
        type=float,
        default=0.5,
        help="Blend factor for 'blend' mode: 0=RGB only, 1=depth only (default: 0.5)"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="smooth",
        choices=["smooth", "plasma", "grayscale", "viridis", "turbo"],
        help="Colormap for depth visualization (default: smooth)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction: >1 emphasizes near, <1 emphasizes far (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    camera_indices = None
    if args.cameras:
        camera_indices = [int(x.strip()) for x in args.cameras.split(",")]
    
    process_folder(
        input_folder=args.input_folder,
        output_subfolder=args.output_subfolder,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        camera_indices=camera_indices,
        mode=args.mode,
        blend_alpha=args.blend_alpha,
        colormap=args.colormap,
        gamma=args.gamma,
    )

