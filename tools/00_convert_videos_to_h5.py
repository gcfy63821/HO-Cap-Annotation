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
import subprocess
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
        Supports formats: cam{cam_id}_rgb.mp4 and cam{cam_id}_depth.mkv (or .mp4)
        Returns:
            tuple: (rgb_video_paths, depth_video_paths) sorted by camera ID
        """
        print(f"[INFO] Discovering videos in {self.input_dir}...")
        
        # Look for RGB videos (cam{cam_id}_rgb.mp4)
        rgb_videos = sorted(list(self.input_dir.glob("cam*_rgb.mp4")))
        
        # Look for depth videos (cam{cam_id}_depth.mkv or .mp4)
        depth_videos_mkv = sorted(list(self.input_dir.glob("cam*_depth.mkv")))
        depth_videos_mp4 = sorted(list(self.input_dir.glob("cam*_depth.mp4")))
        depth_videos = depth_videos_mkv if len(depth_videos_mkv) > 0 else depth_videos_mp4
        
        # Also try alternative naming convention
        if len(rgb_videos) == 0:
            rgb_videos = sorted(list(self.input_dir.glob("rgb_cam*.mp4")))
            depth_videos = sorted(list(self.input_dir.glob("depth_cam*.mp4")))
        
        if len(rgb_videos) == 0:
            raise FileNotFoundError(f"No RGB videos found in {self.input_dir}")
        
        # Match depth videos to RGB videos by camera ID
        if len(depth_videos) > 0:
            # Match depth videos to RGB videos
            matched_depth_videos = []
            for rgb_path in rgb_videos:
                name = rgb_path.stem  # "cam0_rgb"
                cam_id = name.split('_')[0]  # "cam0"
                # Look for corresponding depth video
                depth_pattern = f"{cam_id}_depth"
                matching_depth = [d for d in depth_videos if depth_pattern in d.stem]
                if matching_depth:
                    matched_depth_videos.append(matching_depth[0])
                else:
                    matched_depth_videos.append(None)
            
            depth_videos = matched_depth_videos
        else:
            depth_videos = [None] * len(rgb_videos)
        
        print(f"[INFO] Found {len(rgb_videos)} RGB videos and {sum(1 for d in depth_videos if d is not None)} depth videos")
        
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
    
    def load_depth_frames_ffmpeg(self, video_path, num_frames, width, height):
        """
        Load all depth frames from a video file using ffmpeg to preserve 16-bit precision.
        
        Args:
            video_path (Path): Path to the depth video file.
            num_frames (int): Number of frames to read.
            width (int): Frame width.
            height (int): Frame height.
            
        Returns:
            np.ndarray: Array of depth frames, shape (num_frames, H, W), dtype=uint16
        """
        if video_path is None:
            return np.zeros((num_frames, height, width), dtype=np.uint16)
        
        # Use ffmpeg to decode all frames at once with 16-bit grayscale format
        command = [
            'ffmpeg',
            '-i', str(video_path),
            '-f', 'rawvideo',
            '-pix_fmt', 'gray16le',  # 16-bit little-endian grayscale
            '-'
        ]
        
        try:
            # Run ffmpeg and capture all output
            result = subprocess.run(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            raw_data = result.stdout
            
            # Calculate expected size
            bytes_per_frame = width * height * 2  # 2 bytes per pixel for uint16
            actual_frames = len(raw_data) // bytes_per_frame
            
            if actual_frames != num_frames:
                print(f"[WARNING] Expected {num_frames} frames but got {actual_frames} from {video_path}")
                num_frames = actual_frames
            
            # Convert raw bytes to numpy array
            depth_data = np.frombuffer(raw_data, dtype=np.uint16)
            depth_frames = depth_data.reshape((num_frames, height, width))
            
            return depth_frames
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ffmpeg failed for {video_path}: {e.stderr.decode()}")
            return np.zeros((num_frames, height, width), dtype=np.uint16)
        except Exception as e:
            print(f"[ERROR] Failed to load depth video {video_path}: {e}")
            return np.zeros((num_frames, height, width), dtype=np.uint16)
    
    def load_video_frames(self, video_path, is_depth=False):
        """
        Load all frames from a video file.
        
        Args:
            video_path (Path): Path to the video file. Can be None for depth if not available.
            is_depth (bool): Whether this is a depth video.
            
        Returns:
            np.ndarray: Array of frames, shape (num_frames, H, W, 3) for RGB 
                       or (num_frames, H, W) for depth
        """
        if video_path is None:
            # Return zeros if depth video is not available
            return np.zeros((self.num_frames, self.height, self.width), dtype=np.uint16)
        
        # For depth videos, use ffmpeg to preserve 16-bit precision
        if is_depth:
            return self.load_depth_frames_ffmpeg(video_path, self.num_frames, self.width, self.height)
        
        # For RGB videos, use OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        
        frame_idx = 0
        while frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
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
            valid_depth_count = sum(1 for d in self.depth_videos if d is not None)
            if valid_depth_count > 0:
                print("[INFO] Loading and writing depth videos...")
                for cam_idx, depth_path in enumerate(tqdm(self.depth_videos, desc="Depth videos")):
                    if depth_path is not None:
                        frames = self.load_video_frames(depth_path, is_depth=True)
                        depths_dataset[:len(frames), cam_idx, :, :] = frames
                        del frames  # Free memory immediately
                    else:
                        print(f"[WARNING] No depth video found for camera {cam_idx}, using zeros.")
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
        help='Directory containing the video files (cam0_rgb.mp4, cam0_depth.mkv, etc.)'
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

