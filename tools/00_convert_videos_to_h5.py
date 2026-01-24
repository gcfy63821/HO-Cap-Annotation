#!/usr/bin/env python
"""
Convert RGB and depth videos to HDF5 format for efficient data loading.

This script converts a set of camera videos (RGB and depth) into a single HDF5 file
with the following structure:
- 'imgs': np.ndarray of shape (num_frames, num_cams, height, width, 3), dtype=uint8
- 'depths': np.ndarray of shape (num_frames, num_cams, height, width), dtype=uint16

The depth values are stored in millimeters as uint16 for efficient storage.

Usage:
    python 00_convert_videos_to_h5.py --input_dir <path_to_videos> --output_file <output.h5>
"""

import argparse
import cv2
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


class VideoToHDF5Converter:
    """Convert a set of camera videos to HDF5 format."""
    
    def __init__(self, input_dir, output_file=None, video_name_pattern="cam{cam_id}_{type}.mp4"):
        """
        Initialize the converter.
        
        Args:
            input_dir (str or Path): Directory containing the video files.
            output_file (str or Path, optional): Output HDF5 file path. 
                                                 If None, saves as 'data00000000.h5' in input_dir.
            video_name_pattern (str): Pattern for video file names. 
                                     Use {cam_id} for camera ID and {type} for 'rgb' or 'depth'.
        """
        self.input_dir = Path(input_dir)
        if output_file is None:
            self.output_file = self.input_dir / "data00000000.h5"
        else:
            self.output_file = Path(output_file)
        
        self.video_name_pattern = video_name_pattern
        self.rgb_videos = []
        self.depth_videos = []
        self.num_cameras = 0
        self.num_frames = 0
        self.height = 0
        self.width = 0
        
    def discover_videos(self):
        """
        Discover all RGB and depth videos in the input directory.
        Returns:
            tuple: (rgb_video_paths, depth_video_paths) sorted by camera ID
        """
        print(f"[INFO] Discovering videos in {self.input_dir}...")
        
        # Look for RGB and depth videos
        rgb_videos = sorted(list(self.input_dir.glob("*_rgb.mp4")))
        depth_videos = sorted(list(self.input_dir.glob("*_depth.mp4")))
        
        # Also try alternative naming convention
        if len(rgb_videos) == 0:
            rgb_videos = sorted(list(self.input_dir.glob("rgb_cam*.mp4")))
            depth_videos = sorted(list(self.input_dir.glob("depth_cam*.mp4")))
        
        if len(rgb_videos) == 0:
            raise FileNotFoundError(f"No RGB videos found in {self.input_dir}")
        
        print(f"[INFO] Found {len(rgb_videos)} RGB videos and {len(depth_videos)} depth videos")
        
        self.rgb_videos = rgb_videos
        self.depth_videos = depth_videos
        self.num_cameras = len(rgb_videos)
        
        # Get video properties from first RGB video
        cap = cv2.VideoCapture(str(rgb_videos[0]))
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        
        print(f"[INFO] Video properties: {self.num_cameras} cameras, {self.num_frames} frames, "
              f"{self.width}x{self.height} resolution")
        
        return rgb_videos, depth_videos
    
    def load_video_frames(self, video_path, is_depth=False):
        """
        Load all frames from a video file.
        
        Args:
            video_path (Path): Path to the video file.
            is_depth (bool): Whether this is a depth video.
            
        Returns:
            np.ndarray: Array of frames, shape (num_frames, H, W, 3) for RGB 
                       or (num_frames, H, W) for depth
        """
        cap = cv2.VideoCapture(str(video_path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        if is_depth:
            frames = np.zeros((num_frames, height, width), dtype=np.uint16)
        else:
            frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        
        frame_idx = 0
        while frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if is_depth:
                # For depth videos, extract the depth information
                # Assuming depth was encoded as grayscale uint8 scaled down by 256
                # We need to scale back up to uint16 (millimeters)
                depth_uint8 = frame[:, :, 0] if frame.ndim == 3 else frame
                depth_uint16 = depth_uint8.astype(np.uint16) * 256
                frames[frame_idx] = depth_uint16
            else:
                # Convert BGR to RGB
                frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_idx += 1
        
        cap.release()
        
        if frame_idx < num_frames:
            print(f"[WARNING] Expected {num_frames} frames but got {frame_idx} from {video_path}")
            frames = frames[:frame_idx]
        
        return frames
    
    def convert(self):
        """
        Convert all videos to HDF5 format.
        Uses a memory-efficient approach by creating the HDF5 file first and then
        writing camera data one at a time.
        """
        # Discover videos
        self.discover_videos()
        
        # Create HDF5 file with empty datasets
        print(f"[INFO] Creating HDF5 file with shape ({self.num_frames}, {self.num_cameras}, "
              f"{self.height}, {self.width})...")
        with h5py.File(self.output_file, 'w') as f:
            imgs_dataset = f.create_dataset(
                'imgs', 
                shape=(self.num_frames, self.num_cameras, self.height, self.width, 3),
                dtype=np.uint8,
                compression='gzip',
                compression_opts=4
            )
            depths_dataset = f.create_dataset(
                'depths',
                shape=(self.num_frames, self.num_cameras, self.height, self.width),
                dtype=np.uint16,
                compression='gzip',
                compression_opts=4
            )
            
            # Load and write RGB videos one camera at a time
            print("[INFO] Loading and writing RGB videos...")
            for cam_idx, rgb_path in enumerate(tqdm(self.rgb_videos, desc="RGB videos")):
                frames = self.load_video_frames(rgb_path, is_depth=False)
                imgs_dataset[:len(frames), cam_idx, :, :, :] = frames
                del frames  # Free memory immediately
            
            # Load and write depth videos one camera at a time
            if len(self.depth_videos) > 0:
                print("[INFO] Loading and writing depth videos...")
                for cam_idx, depth_path in enumerate(tqdm(self.depth_videos, desc="Depth videos")):
                    frames = self.load_video_frames(depth_path, is_depth=True)
                    depths_dataset[:len(frames), cam_idx, :, :] = frames
                    del frames  # Free memory immediately
            else:
                print("[WARNING] No depth videos found. Depth array will be all zeros.")
        
        print(f"[SUCCESS] Conversion complete!")
        print(f"[INFO] Output file: {self.output_file}")
        print(f"[INFO] File size: {self.output_file.stat().st_size / (1024**3):.2f} GB")
        print(f"[INFO] Data shapes - imgs: ({self.num_frames}, {self.num_cameras}, {self.height}, "
              f"{self.width}, 3), depths: ({self.num_frames}, {self.num_cameras}, {self.height}, {self.width})")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert camera videos (RGB and depth) to HDF5 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing the video files (cam0_rgb.mp4, cam0_depth.mp4, etc.)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output HDF5 file path. If not specified, saves as data00000000.h5 in input_dir'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='cam{cam_id}_{type}.mp4',
        help='Naming pattern for video files'
    )
    
    args = parser.parse_args()
    
    # Create converter and run
    converter = VideoToHDF5Converter(
        input_dir=args.input_dir,
        output_file=args.output_file,
        video_name_pattern=args.pattern
    )
    
    converter.convert()


if __name__ == "__main__":
    main()

