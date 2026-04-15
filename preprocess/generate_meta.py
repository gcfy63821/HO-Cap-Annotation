import os
import h5py
import yaml
import numpy as np
from pathlib import Path
import argparse

def load_masks_from_folder(mask_root_dir, num_frames, num_cams, expected_H=None, expected_W=None, start_frame=0):
    """
    根据mask根目录读取所有摄像头的mask，返回形状 (N, num_cams, H, W)
    mask_root_dir路径格式支持:
    tool_masks/
      cam00.mp4/ 或 cam0_rgb/
        0.npy 或 0000.npy
        1.npy 或 0001.npy
        ...
      cam01.mp4/ 或 cam1_rgb/
      ...
      cam07.mp4/ 或 cam7_rgb/
    
    Args:
        mask_root_dir: mask根目录路径
        num_frames: 帧数
        num_cams: 相机数量
        expected_H: 期望的高度（如果提供，会resize mask）
        expected_W: 期望的宽度（如果提供，会resize mask）
    """
    mask_root_dir = Path(mask_root_dir)
    all_masks = []
    
    # 从第一帧推断mask的尺寸
    first_mask_shape = None
    for cam_idx in range(num_cams):
        # 尝试多种路径格式
        possible_cam_folders = [
            mask_root_dir / f"cam{cam_idx:02d}.mp4",
            mask_root_dir / f"cam{cam_idx}_rgb",
            mask_root_dir / f"cam{cam_idx}.mp4",
        ]
        
        for cam_folder in possible_cam_folders:
            if cam_folder.exists():
                # 尝试多种文件名格式
                for fmt in [f"{0:04d}.npy", f"{0}.npy"]:
                    npy_path = cam_folder / fmt
                    if npy_path.exists():
                        mask = np.load(npy_path)
                        # 处理mask形状
                        if mask.ndim == 3:
                            mask = mask.squeeze(0)  # (1, H, W) -> (H, W)
                        if first_mask_shape is None:
                            first_mask_shape = mask.shape
                        break
                if first_mask_shape is not None:
                    break
        if first_mask_shape is not None:
            break
    
    # 如果无法推断，使用默认值
    if first_mask_shape is None:
        H, W = expected_H or 480, expected_W or 640
        print(f"[WARNING] Could not infer mask shape, using default: ({H}, {W})")
    else:
        H, W = first_mask_shape
        print(f"[INFO] Inferred mask shape: ({H}, {W})")
    
    # 如果提供了期望尺寸，使用期望尺寸
    if expected_H is not None and expected_W is not None:
        H, W = expected_H, expected_W
        print(f"[INFO] Using expected mask shape: ({H}, {W})")
    
    all_masks = []
    for frame_idx in range(num_frames):
        # mask npy 文件使用原始视频帧号索引
        original_frame_idx = frame_idx + start_frame
        frame_masks = []
        for cam_idx in range(num_cams):
            # 尝试多种路径格式
            possible_cam_folders = [
                mask_root_dir / f"cam{cam_idx:02d}.mp4",
                mask_root_dir / f"cam{cam_idx}_rgb",
                mask_root_dir / f"cam{cam_idx}.mp4",
            ]

            npy_path = None
            for cam_folder in possible_cam_folders:
                if cam_folder.exists():
                    # 尝试多种文件名格式（使用原始帧号）
                    for fmt in [f"{original_frame_idx:04d}.npy", f"{original_frame_idx}.npy"]:
                        test_path = cam_folder / fmt
                        if test_path.exists():
                            npy_path = test_path
                            break
                    if npy_path is not None:
                        break
            
            if npy_path is None or not npy_path.exists():
                frame_masks.append(np.zeros((H, W), dtype=np.uint8))
                continue
            
            mask = np.load(npy_path)
            
            # 处理mask形状
            if mask.ndim == 3:
                mask = mask.squeeze(0)  # (1, H, W) -> (H, W)
            elif mask.ndim != 2:
                print(f"[WARNING] Unexpected mask shape {mask.shape} from {npy_path}, using zeros")
                frame_masks.append(np.zeros((H, W), dtype=np.uint8))
                continue
            
            # Resize mask if dimensions don't match
            if mask.shape[0] != H or mask.shape[1] != W:
                try:
                    import cv2
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                except ImportError:
                    from scipy.ndimage import zoom
                    scale_h = H / mask.shape[0]
                    scale_w = W / mask.shape[1]
                    mask = zoom(mask, (scale_h, scale_w), order=0).astype(mask.dtype)
            
            frame_masks.append(mask)
        all_masks.append(frame_masks)
    
    all_masks = np.array(all_masks)  # (N, num_cams, H, W)
    print(f"[INFO] Loaded masks with shape: {all_masks.shape}, dtype: {all_masks.dtype}")
    return all_masks
def save_masks_to_h5(masks, h5_path, dataset_name="masks"):
    """
    Save masks numpy array to an h5 file with the given dataset name.
    """
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset(dataset_name, data=masks, compression="gzip")
    print(f"[INFO] Saved {dataset_name} to {h5_path}")

def generate_meta_yaml(h5_path, mask_root_dir, calibration_yaml_path, output_root, subject_id="subject_5", tool_name="blue_scooper", models_folder="models", object_mask_dir=None, start_frame=0, thresholds=None):
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
    img_H, img_W = imgs.shape[2], imgs.shape[3]

    # Save masks as h5 file in mask_root_dir
    # 使用图像尺寸作为期望的mask尺寸，start_frame用于mask npy文件的偏移
    masks = load_masks_from_folder(mask_root_dir, num_frames, num_cams, expected_H=img_H, expected_W=img_W, start_frame=start_frame)  # (N, num_cams, H, W)
    masks_h5_path = Path(mask_root_dir) / "masks.h5"
    save_masks_to_h5(masks, masks_h5_path, dataset_name="masks")

    # Save object masks if provided
    if object_mask_dir is not None:
        object_masks = load_masks_from_folder(object_mask_dir, num_frames, num_cams, expected_H=img_H, expected_W=img_W, start_frame=start_frame)
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
        "start_frame": int(start_frame),
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
        "thresholds": thresholds if thresholds is not None else [-0.5, 0.35, -0.5, 0.4, -0.3, 0.4],
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
    parser.add_argument('--h5_path', type=str, required=True, help='Path to the .h5 file (e.g. /data/folder_name/task_name/sequence_name/data00000000.h5)')
    parser.add_argument('--calibration_yaml_path', type=str, default='/path/to/calibration.yaml', help='Path to calibration YAML (fixed)')
    parser.add_argument('--models_folder', type=str, default='/path/to/models', help='Path to models folder (fixed)')
    parser.add_argument('--subject_id', type=str, default=None, help='Subject ID (default: sequence_name)')
    parser.add_argument('--tool_name', type=str, default=None, help='Tool/object name (default: sequence_name)')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame index in original video (saved to meta.yaml for mask offset)')
    parser.add_argument('--x_min', type=float, default=-0.5, help='Threshold x min (default: -0.5)')
    parser.add_argument('--x_max', type=float, default=0.35, help='Threshold x max (default: 0.35)')
    parser.add_argument('--y_min', type=float, default=-0.5, help='Threshold y min (default: -0.5)')
    parser.add_argument('--y_max', type=float, default=0.4, help='Threshold y max (default: 0.4)')
    parser.add_argument('--z_min', type=float, default=-0.3, help='Threshold z min (default: -0.3)')
    parser.add_argument('--z_max', type=float, default=0.4, help='Threshold z max (default: 0.4)')
    args = parser.parse_args()

    # Infer folder_name, task_name, and sequence_name from h5_path
    h5_path = Path(args.h5_path)
    # Updated path parsing for new structure: /.../{folder_name}/{task_name}/{sequence_name}/data00000000.h5
    sequence_name = h5_path.parts[-2]  # xxxvideoname
    task_name = h5_path.parts[-3]      # taskname
    folder_name = h5_path.parts[-4]    # videos_0901
    
    print(f"[INFO] Parsed paths - folder_name: {folder_name}, task_name: {task_name}, sequence_name: {sequence_name}")

    # Infer mask_root_dir and object_mask_dir with taskname included
    # Structure: .../videos_0901_annotated/taskname/xxxvideoname/
    mask_root_dir = h5_path.parent.parent.parent.parent / f"{folder_name}_annotated" / task_name / sequence_name / "tool_masks"
    object_mask_dir = h5_path.parent.parent.parent.parent / f"{folder_name}_annotated" / task_name / sequence_name / "object_masks"
    if not mask_root_dir.exists():
        mask_root_dir = h5_path.parent.parent.parent.parent / f"{folder_name}_annotated" / task_name / sequence_name / "masks"
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
        object_mask_dir=str(object_mask_dir) if object_mask_dir is not None else None,
        start_frame=args.start_frame,
        thresholds=[args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max],
    )
