import os
import cv2
import h5py
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm


def write_video_opencv(images, out_path, fps=25, is_depth=False):
    """Write a list of images to an MP4 video file using OpenCV."""
    if is_depth:
        # For depth videos, we need to handle uint16 data
        height, width = images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=False)
        
        for img in images:
            if img.dtype == np.uint16:
                # Convert uint16 to uint8 for video encoding (lose some precision but keep most)
                # Scale from 0-65535 to 0-255
                img_uint8 = (img / 256).astype(np.uint8)
            else:
                img_uint8 = img.astype(np.uint8)
            writer.write(img_uint8)
    else:
        # For RGB videos
        height, width, _ = images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        for img in images:
            if img.dtype != np.uint8:
                # Convert float images to uint8
                bgr = (img * 255).clip(0, 255).astype(np.uint8)
            else:
                bgr = img
            writer.write(bgr)
    
    writer.release()


def normalize_depth_to_uint16(depth_array):
    """Convert depth data to uint16 range while preserving precision."""
    # Convert depth from meters to millimeters and clip to reasonable range
    # Assuming depth is in meters (0.001 to 10.0 meters)
    depth_mm = (depth_array * 1000).clip(0, 65535)  # Convert to mm and clip to uint16 range
    return depth_mm.astype(np.uint16)


def process_depth_frame(depth_data, view_idx):
    """Process a single depth frame for a specific view."""
    if depth_data.ndim == 2:
        # Single view depth
        return normalize_depth_to_uint16(depth_data)
    else:
        # Multi-view depth
        return normalize_depth_to_uint16(depth_data[view_idx])


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate MP4 videos from H5 data including RGB and depth")
    parser.add_argument("--data_file", type=str, 
                       default="/viscam/projects/robotool/data/videos_0909/knife_cut_banana/20250909_1626_orangeknife_purpleplate_cut_banana_25_2/data00000000.h5",
                       help="Path to the H5 data file")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="Starting frame index")
    parser.add_argument("--end_frame", type=int, default=None,
                       help="Ending frame index (default: all frames)")
    parser.add_argument("--output_dir", type=str, 
                        default="/viscam/projects/robotool/data/videos_0909/knife_cut_banana/20250909_1626_orangeknife_purpleplate_cut_banana_25_2",
                       help="Output directory for videos")
    parser.add_argument("--save_rgb", action="store_true", default=True,
                       help="Save RGB videos")
    parser.add_argument("--save_depth", action="store_true", default=True,
                       help="Save depth videos")
    parser.add_argument("--fps", type=int, default=25,
                       help="Video frame rate")
    args = parser.parse_args()

    # Open H5 file
    session_h5 = h5py.File(args.data_file, "r")
    
    # Check available keys
    print("Available keys in H5 file:", list(session_h5.keys()))
    
    # Get frame information
    if "imgs" in session_h5:
        total_frames = len(session_h5["imgs"])
        print(f"Found {total_frames} RGB frames")
    else:
        print("Error: 'imgs' key not found in H5 file")
        session_h5.close()
        exit(1)
    
    if "depths" in session_h5:
        depth_frames = len(session_h5["depths"])
        print(f"Found {depth_frames} depth frames")
        if depth_frames != total_frames:
            print(f"Warning: Mismatch between RGB frames ({total_frames}) and depth frames ({depth_frames})")
    else:
        print("Warning: 'depths' key not found in H5 file")
        args.save_depth = False
    
    # Set frame range
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame is not None else total_frames
    print(f"Processing frames {start_frame} to {end_frame-1}")

    # Create output directories
    video_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    print("Saving videos to:", video_dir)

    # Initialize data storage
    rgb_view_to_images = defaultdict(list)
    depth_view_to_images = defaultdict(list)

    # Process frames
    for frame_idx in tqdm(range(start_frame, end_frame), desc="Processing frames"):
        # Process RGB images
        if args.save_rgb:
            rgbs = session_h5["imgs"][frame_idx]
            if rgbs.ndim == 3:
                # Single view
                rgb = rgbs.astype(np.float32) / 255.0
                rgb_view_to_images[0].append(rgb)
            else:
                # Multi-view
                for view_idx, rgb in enumerate(rgbs):
                    rgb = rgb.astype(np.float32) / 255.0
                    rgb_view_to_images[view_idx].append(rgb)
        
        # Process depth images
        if args.save_depth and "depths" in session_h5:
            depth_data = session_h5["depths"][frame_idx]
            if depth_data.ndim == 2:
                # Single view depth
                depth_processed = process_depth_frame(depth_data, 0)
                depth_view_to_images[0].append(depth_processed)
            else:
                # Multi-view depth
                for view_idx in range(depth_data.shape[0]):
                    depth_processed = process_depth_frame(depth_data, view_idx)
                    depth_view_to_images[view_idx].append(depth_processed)

    session_h5.close()

    # Generate RGB videos
    if args.save_rgb:
        print("Generating RGB videos...")
        for view, images in tqdm(rgb_view_to_images.items(), desc="RGB videos"):
            out_path = os.path.join(video_dir, f"rgb_cam{view:02d}.mp4")
            write_video_opencv(images, out_path, fps=args.fps, is_depth=False)
            print(f"Saved RGB video: {out_path}")

    # Generate depth videos
    if args.save_depth:
        print("Generating depth videos...")
        for view, images in tqdm(depth_view_to_images.items(), desc="Depth videos"):
            out_path = os.path.join(video_dir, f"depth_cam{view:02d}.mp4")
            write_video_opencv(images, out_path, fps=args.fps, is_depth=True)
            print(f"Saved depth video: {out_path}")

    print("Video generation completed!")
