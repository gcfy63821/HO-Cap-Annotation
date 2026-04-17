#!/usr/bin/env python3
"""
Integrate stereo camera (Fast-FoundationStereo) output into the HO-Cap pipeline.

This script:
  1. Reads existing H5 file (N, num_cams, H, W, ...) with RealSense data
  2. Reads stereo color images + depth npy files from FFS output
  3. Appends stereo data as an additional camera to the H5 file
  4. Creates a calibration YAML that includes both RealSense cameras and the stereo camera

Usage:
  python tools/integrate_stereo_to_h5.py \
    --h5_path /path/to/data00000000.h5 \
    --stereo_color_dir /path/to/stereo_output/color \
    --stereo_depth_dir /path/to/stereo_output/depth \
    --stereo_calib_file /path/to/stereo_calib.npz \
    --realsense_calib_yaml /path/to/realsense_calibration.yaml
"""

import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml


def load_stereo_frames(color_dir, depth_dir, num_frames, height, width, start_frame=0):
    """
    Load stereo color and depth frames, resize to match H5 resolution.

    Returns:
        colors: (N, H, W, 3) uint8
        depths: (N, H, W) uint16 (millimeters)
    """
    color_dir = Path(color_dir)
    depth_dir = Path(depth_dir)

    colors = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    depths = np.zeros((num_frames, height, width), dtype=np.uint16)

    for i in range(num_frames):
        frame_idx = i + start_frame

        # Load color
        color_path = color_dir / f"{frame_idx:06d}.png"
        if color_path.exists():
            img = cv2.imread(str(color_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height))
                colors[i] = img

        # Load depth (float32 meters -> uint16 millimeters)
        depth_path = depth_dir / f"{frame_idx:06d}.npy"
        if depth_path.exists():
            depth_m = np.load(depth_path)  # float32, meters
            if depth_m.shape[0] != height or depth_m.shape[1] != width:
                depth_m = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)
            # Convert to uint16 millimeters (same format as RealSense depth in H5)
            depth_mm = np.clip(depth_m * 1000, 0, 65535).astype(np.uint16)
            depths[i] = depth_mm

    loaded = sum(1 for i in range(num_frames) if (color_dir / f"{i+start_frame:06d}.png").exists())
    print(f"[INFO] Loaded {loaded}/{num_frames} stereo frames")

    return colors, depths


def create_stereo_calib_entry(stereo_calib_npz, realsense_calib_data, stereo_cam_id):
    """
    Create a calibration entry for the stereo camera compatible with the
    RealSense calibration YAML format.

    The stereo camera does not have a known world-frame extrinsic (it's not
    calibrated with the RealSense system). We set transformation to identity
    as a placeholder — the pipeline will use it for single-view tracking only.
    """
    calib = np.load(stereo_calib_npz)
    K_rect = calib["K_rectified"]  # (3, 3)

    # Use identity as extrinsic (stereo camera world pose unknown)
    # TODO: if you calibrate stereo-to-realsense extrinsics, replace this
    T_identity = np.eye(4).tolist()

    entry = {
        "camera_id": stereo_cam_id,
        "serial_number": f"stereo_{stereo_cam_id:02d}",
        "color_intrinsic_matrix": K_rect.tolist(),
        "depth_intrinsic_matrix": K_rect.tolist(),  # same K for stereo depth
        "transformation": T_identity,
    }

    return entry


def append_stereo_to_h5(h5_path, stereo_colors, stereo_depths):
    """
    Append stereo camera data to existing H5 file.
    Expands the camera dimension: (N, C, H, W, ...) -> (N, C+1, H, W, ...)
    """
    h5_path = Path(h5_path)
    tmp_path = h5_path.parent / "data00000000_with_stereo.h5"

    with h5py.File(h5_path, "r") as f_in:
        N, C_old, H, W = f_in["depths"].shape
        C_new = C_old + 1
        print(f"[INFO] Expanding H5: ({N}, {C_old}, {H}, {W}) -> ({N}, {C_new}, {H}, {W})")

        with h5py.File(tmp_path, "w") as f_out:
            # Create expanded datasets
            imgs_out = f_out.create_dataset(
                "imgs", shape=(N, C_new, H, W, 3), dtype=np.uint8,
                chunks=(1, 1, H, W, 3), compression="gzip", compression_opts=4,
            )
            depths_out = f_out.create_dataset(
                "depths", shape=(N, C_new, H, W), dtype=np.uint16,
                chunks=(1, 1, H, W), compression="gzip", compression_opts=4,
            )

            # Copy existing RealSense data in batches
            batch_size = 50
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                imgs_out[start:end, :C_old] = f_in["imgs"][start:end]
                depths_out[start:end, :C_old] = f_in["depths"][start:end]

            # Append stereo data as last camera
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                imgs_out[start:end, C_old] = stereo_colors[start:end]
                depths_out[start:end, C_old] = stereo_depths[start:end]

    # Replace original H5
    backup_path = h5_path.parent / "data00000000_realsense_only.h5"
    if not backup_path.exists():
        os.rename(h5_path, backup_path)
        print(f"[INFO] Backed up original H5 to {backup_path.name}")
    else:
        os.remove(h5_path)
    os.rename(tmp_path, h5_path)
    print(f"[INFO] Updated {h5_path.name} with stereo camera")


def save_combined_calib_yaml(realsense_yaml_path, stereo_entry, output_yaml_path):
    """
    Create a combined calibration YAML with all RealSense cameras + stereo camera.
    """
    with open(realsense_yaml_path, "r") as f:
        calib_data = yaml.safe_load(f)

    calib_data.append(stereo_entry)

    with open(output_yaml_path, "w") as f:
        yaml.dump(calib_data, f, default_flow_style=False)

    print(f"[INFO] Saved combined calibration to {output_yaml_path}")
    print(f"[INFO] Total cameras: {len(calib_data)} ({len(calib_data)-1} RealSense + 1 stereo)")


def main():
    parser = argparse.ArgumentParser(description="Integrate stereo depth into HO-Cap H5 pipeline")
    parser.add_argument("--h5_path", required=True, type=str)
    parser.add_argument("--stereo_color_dir", required=True, type=str)
    parser.add_argument("--stereo_depth_dir", required=True, type=str)
    parser.add_argument("--stereo_calib_file", required=True, type=str,
                        help="Path to stereo_calib.npz from stereo_calibrate.py")
    parser.add_argument("--realsense_calib_yaml", required=True, type=str,
                        help="Path to original RealSense calibration YAML")
    parser.add_argument("--start_frame", type=int, default=0)
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    sequence_folder = h5_path.parent

    # Read existing H5 dimensions
    with h5py.File(h5_path, "r") as f:
        N, C_old, H, W = f["depths"].shape
    print(f"[INFO] Existing H5: {N} frames, {C_old} cameras, {H}x{W}")

    # Check if stereo already appended
    with open(args.realsense_calib_yaml, "r") as f:
        rs_calib = yaml.safe_load(f)
    num_rs_cams = len(rs_calib)

    if C_old > num_rs_cams:
        print(f"[WARNING] H5 already has {C_old} cameras (> {num_rs_cams} RealSense). "
              f"Stereo may already be integrated. Skipping H5 append.")
    else:
        # Load stereo frames
        print(f"[INFO] Loading stereo frames...")
        stereo_colors, stereo_depths = load_stereo_frames(
            args.stereo_color_dir, args.stereo_depth_dir,
            N, H, W, start_frame=args.start_frame,
        )

        # Append to H5
        print(f"[INFO] Appending stereo data to H5...")
        append_stereo_to_h5(h5_path, stereo_colors, stereo_depths)

    # Create stereo calibration entry
    stereo_cam_id = num_rs_cams  # next camera ID (e.g., 8)
    stereo_entry = create_stereo_calib_entry(
        args.stereo_calib_file, rs_calib, stereo_cam_id,
    )

    # Save combined calibration YAML
    output_yaml = sequence_folder / "calibration_with_stereo.yaml"
    save_combined_calib_yaml(args.realsense_calib_yaml, stereo_entry, str(output_yaml))


if __name__ == "__main__":
    main()
