import os
import h5py
import yaml
import numpy as np
from pathlib import Path
import argparse

def load_masks_from_folder(mask_root_dir, num_frames, num_cams):
    """
    根据mask根目录读取所有摄像头的mask，返回形状 (N, 8, H, W)
    mask_root_dir路径格式:
    tool_masks/
      cam00.mp4/
        0.npy
        1.npy
        ...
      cam01.mp4/
      ...
      cam07.mp4/
    """
    H, W = 480, 640
    mask_root_dir = Path(mask_root_dir)
    all_masks = []
    for frame_idx in range(num_frames):
        frame_masks = []
        for cam_idx in range(num_cams):
            cam_folder = mask_root_dir / f"cam{cam_idx:02d}.mp4"
            npy_path = cam_folder / f"{frame_idx}.npy"
            if not npy_path.exists():
                frame_masks.append(np.zeros((1, H, W), dtype=np.uint8))
                # print("is none:", cam_folder, frame_idx)
                continue
            mask = np.load(npy_path)
            # print("mask_shape", mask.shape)
            frame_masks.append(mask)
        all_masks.append(frame_masks)
    # print("all_masks", len(all_masks))
    all_masks = np.array(all_masks)  # (N, 8, H, W)
    return all_masks

def save_masks_to_h5(masks, h5_path, dataset_name="masks"):
    """
    Save masks numpy array to an h5 file with the given dataset name.
    """
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset(dataset_name, data=masks, compression="gzip")
    print(f"[INFO] Saved {dataset_name} to {h5_path}")

def generate_meta_yaml(h5_path, mask_root_dir, calibration_yaml_path, output_root, subject_id="subject_5", tool_name="blue_scooper", models_folder="models", object_mask_dir=None):
    """
    Generate meta.yaml for a HO-Cap dataset sequence. Also saves masks as h5 files in their respective directories.
    Args:
        h5_path (str): Path to the .h5 file containing imgs and depths.
        mask_root_dir (str): Path to the tool_masks folder.
        calibration_yaml_path (str): Path to the original calibration YAML file.
        output_root (str): Output root directory for meta.yaml.
        subject_id (str): Subject ID.
        tool_name (str): Name of the tool/object.
        models_folder (str): Path to the models folder.
        object_mask_dir (str or None): Path to object_masks folder (optional).
    """
    # Load .h5 to get number of frames and cameras
    with h5py.File(h5_path, 'r') as f:
        imgs = f["imgs"][:]  # (N, num_cams, H, W, 3)
    num_frames, num_cams = imgs.shape[0], imgs.shape[1]

    # Save masks as h5 file in mask_root_dir
    masks = load_masks_from_folder(mask_root_dir, num_frames, num_cams)  # (N, 8, H, W)
    masks_h5_path = Path(mask_root_dir) / "masks.h5"
    save_masks_to_h5(masks, masks_h5_path, dataset_name="masks")

    # Save object masks if provided
    if object_mask_dir is not None:
        object_masks = load_masks_from_folder(object_mask_dir, num_frames, num_cams)
        object_masks_h5_path = Path(object_mask_dir) / "object_masks.h5"
        save_masks_to_h5(object_masks, object_masks_h5_path, dataset_name="object_masks")

    # Get camera serials from calibration YAML
    with open(calibration_yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    cam_serials = [str(cam['camera_id']).zfill(2) for cam in calib_data]
    # Assume width/height from imgs
    width = imgs.shape[3]
    height = imgs.shape[2]

    # Output folder
    output_folder = Path(output_root)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Compose meta.yaml
    meta = {
        "num_frames": int(num_frames),
        "object_ids": [tool_name],
        "mano_sides": ['left', 'right'],
        "subject_id": subject_id,
        "realsense": {
            "serials": cam_serials,
            "width": width,
            "height": height
        },
        "hololens": {
            "serial": "hololens_kv5h72",
            "pv_height": 720,
            "pv_width": 1280
        },
        "have_hololens": False,
        "have_mano": True,
        "task_id": 1,
        "thresholds": [-0.4, 0.3, -0.4, 0.3, -0.3, 0.4],
        "calibration_yaml_path": calibration_yaml_path,
        "models_folder": models_folder,
        "betas": [
            0.051946,
            0.023095,
            0.13714,
            0.039837,
            0.054446,
            0.03033,
            0.041728,
            0.006936,
            0.022853,
            0.010556
        ]
    }

    with open(output_folder / "meta.yaml", "w") as f:
        yaml.dump(meta, f)
    print(f"[INFO] meta.yaml written to {output_folder / 'meta.yaml'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate meta.yaml for HO-Cap dataset.")
    parser.add_argument('--h5_path', type=str, required=True, help='Path to the .h5 file (e.g. /data/folder_name/sequence_name/data00000000.h5)')
    parser.add_argument('--calibration_yaml_path', type=str, default='/path/to/calibration.yaml', help='Path to calibration YAML (fixed)')
    parser.add_argument('--models_folder', type=str, default='/path/to/models', help='Path to models folder (fixed)')
    parser.add_argument('--subject_id', type=str, default=None, help='Subject ID (default: sequence_name)')
    parser.add_argument('--tool_name', type=str, default=None, help='Tool/object name (default: sequence_name)')
    args = parser.parse_args()

    # Infer folder_name and sequence_name from h5_path
    h5_path = Path(args.h5_path)
    # /.../{folder_name}/{sequence_name}/data00000000.h5
    folder_name = h5_path.parts[-3]
    sequence_name = h5_path.parts[-2]

    # Infer mask_root_dir and object_mask_dir
    mask_root_dir = h5_path.parent.parent.parent / f"{folder_name}_annotated" / sequence_name / "tool_masks"
    object_mask_dir = h5_path.parent.parent.parent / f"{folder_name}_annotated" / sequence_name / "object_masks"
    if not mask_root_dir.exists():
        mask_root_dir = h5_path.parent.parent.parent / f"{folder_name}_annotated" / sequence_name / "masks"
    if not object_mask_dir.exists():
        object_mask_dir = None

    # Infer output_root
    output_root = h5_path.parent

    # Subject and tool name
    subject_id = args.subject_id if args.subject_id is not None else sequence_name
    tool_name = args.tool_name if args.tool_name is not None else sequence_name

    generate_meta_yaml(
        str(h5_path),
        str(mask_root_dir),
        args.calibration_yaml_path,
        str(output_root),
        subject_id=subject_id,
        tool_name=tool_name,
        models_folder=args.models_folder,
        object_mask_dir=str(object_mask_dir) if object_mask_dir is not None else None
    )
