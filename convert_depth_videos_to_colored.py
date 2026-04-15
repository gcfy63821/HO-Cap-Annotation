#!/usr/bin/env python3
"""
将文件夹下的所有深度视频转换为彩色视频，并与RGB视频拼接在一起
"""
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys

def get_depth_colormap(image: np.ndarray, vmin=None, vmax=None, percentile_range=(5, 95)) -> np.ndarray:
    """
    Convert a depth image to a colormap representation.
    
    Args:
        image: 2D depth image
        vmin: Minimum depth value for normalization (if None, use percentile)
        vmax: Maximum depth value for normalization (if None, use percentile)
        percentile_range: Percentile range for clipping extreme values (used if vmin/vmax not provided)
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    
    depth = image.astype(np.float32)
    valid_mask = depth > 0
    
    if not np.any(valid_mask):
        # If no valid depth, return black image
        return np.zeros((*image.shape, 3), dtype=np.uint8)
    
    # Auto-clip using percentiles or provided range
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(depth[valid_mask], percentile_range)
    
    # Normalize to [0, 255]
    depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    
    # Apply colormap (VIRIDIS)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    return depth_colored


def read_depth_frame(cap, is_mkv=True):
    """
    Read a depth frame from video.
    
    Args:
        cap: cv2.VideoCapture object
        is_mkv: Whether the video is MKV format (may need special handling)
    
    Returns:
        np.ndarray: Depth frame as uint16 (in millimeters) or None if failed
    """
    ret, frame = cap.read()
    if not ret:
        return None
    
    # Handle different depth video encodings
    if frame.ndim == 3:
        # If depth is stored as 3-channel, extract depth from first channel
        depth_frame = frame[:, :, 0]
    else:
        depth_frame = frame
    
    # Convert to uint16 if needed
    if depth_frame.dtype == np.uint8:
        # Scale from uint8 to uint16 (millimeters)
        depth_uint16 = depth_frame.astype(np.uint16) * 256
    elif depth_frame.dtype == np.uint16:
        depth_uint16 = depth_frame
    else:
        depth_uint16 = depth_frame.astype(np.uint16)
    
    return depth_uint16


def process_video_pair(depth_video_path, rgb_video_path, output_path, fps=None):
    """
    Process a pair of depth and RGB videos:
    1. Convert depth video to colored depth video
    2. Concatenate with RGB video side by side
    3. Save the result
    
    Args:
        depth_video_path: Path to depth video (.mkv)
        rgb_video_path: Path to RGB video (.mp4)
        output_path: Path to save the concatenated video
        fps: FPS for output video (if None, use RGB video FPS)
    """
    print(f"\n[INFO] Processing: {Path(depth_video_path).name} + {Path(rgb_video_path).name}")
    
    # Open videos
    depth_cap = cv2.VideoCapture(str(depth_video_path))
    rgb_cap = cv2.VideoCapture(str(rgb_video_path))
    
    if not depth_cap.isOpened():
        raise ValueError(f"Cannot open depth video: {depth_video_path}")
    if not rgb_cap.isOpened():
        raise ValueError(f"Cannot open RGB video: {rgb_video_path}")
    
    # Get video properties
    depth_fps = depth_cap.get(cv2.CAP_PROP_FPS)
    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    depth_frame_count = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb_frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use the minimum frame count to ensure both videos are in sync
    frame_count = min(depth_frame_count, rgb_frame_count)
    
    # Get dimensions
    depth_width = int(depth_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    depth_height = int(depth_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rgb_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Depth video: {depth_width}x{depth_height}, {depth_fps} fps, {depth_frame_count} frames")
    print(f"[INFO] RGB video: {rgb_width}x{rgb_height}, {rgb_fps} fps, {rgb_frame_count} frames")
    print(f"[INFO] Processing {frame_count} frames")
    
    # Use RGB FPS if fps is not specified
    if fps is None:
        fps = rgb_fps
    
    # Resize if dimensions don't match (resize depth to match RGB)
    if depth_width != rgb_width or depth_height != rgb_height:
        print(f"[INFO] Resizing depth frames from {depth_width}x{depth_height} to {rgb_width}x{rgb_height}")
        resize_depth = True
    else:
        resize_depth = False
    
    # Output dimensions: side by side
    output_width = rgb_width * 2
    output_height = rgb_height
    
    # Scan all frames to find global depth range for consistent colormap
    print("[INFO] Scanning depth frames to find global depth range...")
    all_depths = []
    temp_cap = cv2.VideoCapture(str(depth_video_path))
    scan_count = 0
    while scan_count < min(frame_count, 100):  # Sample first 100 frames for speed
        depth_frame = read_depth_frame(temp_cap, is_mkv=True)
        if depth_frame is None:
            break
        if resize_depth:
            depth_frame = cv2.resize(depth_frame, (rgb_width, rgb_height), interpolation=cv2.INTER_NEAREST)
        valid_depths = depth_frame[depth_frame > 0]
        if len(valid_depths) > 0:
            all_depths.extend(valid_depths.tolist())
        scan_count += 1
    temp_cap.release()
    
    # Calculate global depth range
    if len(all_depths) > 0:
        all_depths = np.array(all_depths)
        global_vmin, global_vmax = np.percentile(all_depths, (5, 95))
        print(f"[INFO] Global depth range: {global_vmin:.2f} - {global_vmax:.2f} mm")
    else:
        global_vmin, global_vmax = None, None
        print("[WARNING] Could not determine depth range, using per-frame normalization")
    
    # Create output video writer using OpenCV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use mp4v codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    
    if not out_writer.isOpened():
        raise ValueError(f"Failed to create video writer for {output_path}")
    
    # Process frames
    frame_idx = 0
    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        while frame_idx < frame_count:
            # Read depth frame
            depth_frame = read_depth_frame(depth_cap, is_mkv=True)
            if depth_frame is None:
                break
            
            # Read RGB frame
            ret, rgb_frame = rgb_cap.read()
            if not ret:
                break
            
            # Resize depth if needed
            if resize_depth:
                depth_frame = cv2.resize(depth_frame, (rgb_width, rgb_height), interpolation=cv2.INTER_NEAREST)
            
            # Convert depth to colored depth (use global range if available)
            depth_colored = get_depth_colormap(depth_frame, vmin=global_vmin, vmax=global_vmax)
            
            # Convert RGB from BGR to RGB for concatenation
            rgb_frame_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            
            # Concatenate side by side: [RGB | Colored Depth]
            concatenated = np.concatenate([rgb_frame_rgb, depth_colored], axis=1)
            
            # Convert back to BGR for OpenCV VideoWriter
            concatenated_bgr = cv2.cvtColor(concatenated, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out_writer.write(concatenated_bgr)
            
            frame_idx += 1
            pbar.update(1)
    
    # Clean up
    depth_cap.release()
    rgb_cap.release()
    out_writer.release()
    
    print(f"[SUCCESS] Saved concatenated video to: {output_path}")


def process_all_videos(folder_path):
    """
    Process all depth and RGB video pairs in a folder.
    
    Args:
        folder_path: Path to folder containing videos
    """
    folder_path = Path(folder_path)
    
    # Find all depth videos
    depth_videos = sorted(folder_path.glob("cam*_depth.mkv"))
    
    if not depth_videos:
        print(f"[ERROR] No depth videos found in {folder_path}")
        return
    
    print(f"[INFO] Found {len(depth_videos)} depth videos")
    
    # Process each depth video with its corresponding RGB video
    for depth_video in depth_videos:
        # Extract camera ID (e.g., "cam0" from "cam0_depth.mkv")
        cam_id = depth_video.stem.replace("_depth", "")
        
        # Find corresponding RGB video
        rgb_video = folder_path / f"{cam_id}_rgb.mp4"
        
        if not rgb_video.exists():
            print(f"[WARNING] RGB video not found for {depth_video.name}: {rgb_video}")
            continue
        
        # Create output path
        output_path = folder_path / f"{cam_id}_rgb_depth_sidebyside.mp4"
        
        try:
            process_video_pair(depth_video, rgb_video, output_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {depth_video.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n[SUCCESS] All videos processed!")


if __name__ == "__main__":
    folder_path = "/home/ruoqu/crq_ws/HO-Cap-Annotation/data/videos_1121/fork_slice_dough/20251121_bigwoodenfork_slice_dough_half_1"
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    
    print(f"[INFO] Processing videos in: {folder_path}")
    process_all_videos(folder_path)

