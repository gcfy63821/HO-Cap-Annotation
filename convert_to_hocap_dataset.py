import os
import h5py
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


# def save_image(image_array, path):
#     # 只对彩色图做BGR->RGB转换，深度和mask不转换
#     if image_array.ndim == 3 and image_array.shape[2] == 3:
#         image_array = image_array[..., ::-1]  # BGR->RGB
#     Image.fromarray(image_array).save(path)
def save_image(image_array, path):
    # 自动 squeeze 掉多余的维度
    image_array = np.squeeze(image_array)

    # 彩色图像：BGR to RGB
    if image_array.ndim == 3 and image_array.shape[2] == 3:
        image_array = image_array[..., ::-1]

    # 如果是 mask，确保是 uint8 格式
    if image_array.dtype != np.uint8:
        image_array = (image_array.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)

    Image.fromarray(image_array).save(path)


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
    mask_root_dir = Path(mask_root_dir)
    all_masks = []
    for frame_idx in range(num_frames):
        frame_masks = []
        for cam_idx in range(num_cams):
            cam_folder = mask_root_dir / f"cam{cam_idx:02d}.mp4"
            npy_path = cam_folder / f"{frame_idx}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"Mask file missing: {npy_path}")
            mask = np.load(npy_path)
            frame_masks.append(mask)
        all_masks.append(frame_masks)
    all_masks = np.array(all_masks)  # (N, 8, H, W)
    return all_masks


def convert_to_hocap_format(h5_path, mask_root_dir, extrinsics_yaml_path, output_root, subject_id="subject_5"):
    # Load data
    with h5py.File(h5_path, 'r') as f:
        imgs = f["imgs"][:]  # (N, 8, 480, 640, 3)
        depths = f["depths"][:]  # (N, 8, 480, 640)

    num_frames, num_cams = imgs.shape[0], imgs.shape[1]

    # Load masks from folder structure
    masks = load_masks_from_folder(mask_root_dir, num_frames, num_cams)  # (N, 8, H, W)

    # Load extrinsics from YAML
    with open(extrinsics_yaml_path, 'r') as f:
        extrinsics_yaml = yaml.safe_load(f)
    extrinsics_dict = extrinsics_yaml["extrinsics"]
    cam_serials = sorted(extrinsics_dict.keys())  # ['00', '01', ..., '07']

    # Check consistency
    assert len(cam_serials) == num_cams, f"Number of cameras mismatch: {len(cam_serials)} vs {num_cams}"

    # Time tag
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path(output_root) / subject_id / time_tag
    print(f"[INFO] Saving to {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    # For each camera
    for cam_idx in range(num_cams):
        cam_serial = f"{cam_idx:02d}"
        cam_dir = output_folder / cam_serial
        color_dir = cam_dir
        depth_dir = cam_dir
        mask_dir = output_folder / "processed" / "segmentation" / "sam2" / cam_serial / "mask"
        mask_dir.mkdir(parents=True, exist_ok=True)
        color_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in tqdm(range(num_frames), desc=f"Camera {cam_idx}"):
            # Save color
            rgb = imgs[frame_idx, cam_idx]
            save_image(rgb, color_dir / f"color_{frame_idx:06d}.jpg")

            # Save depth
            depth = depths[frame_idx, cam_idx]
            save_image(depth, depth_dir / f"depth_{frame_idx:06d}.png")

            # Save mask
            mask = masks[frame_idx, cam_idx].astype(np.uint8)
            save_image(mask * 255, mask_dir / f"mask_{frame_idx:06d}.png")

    # Save meta.yaml
    meta = {
        "num_frames": int(num_frames),
        "object_ids": ["blue_scooper"],
        "mano_sides": [],
        "subject_id": subject_id,
        "realsense": {
            "serials": cam_serials,
            "width": 640,
            "height": 480
        },
        "hololens": {
            "serial": "hololens_kv5h72",
            "pv_height": 720,
            "pv_width": 1280
        },
        "extrinsics": "extrinsics.yaml"
    }

    with open(output_folder / "meta.yaml", "w") as f:
        yaml.dump(meta, f)

    print("[INFO] Done writing images and meta.yaml!")


# 示例调用
if __name__ == "__main__":
    convert_to_hocap_format(
        h5_path="/home/wys/learning-compliant/crq_ws/data/0506data/blue_scooper/06c0c8e0_blue_scooper_mid_6/data00000000.h5",
        mask_root_dir="/home/wys/learning-compliant/crq_ws/data/0506data/blue_scooper_annotated/06c0c8e0_blue_scooper_mid_6/tool_masks",
        extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
        output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
        subject_id="subject_5"
    )
