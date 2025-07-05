import os
import h5py
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


# old one
# def save_image(image_array, path, is_depth=False):
#     image_array = np.squeeze(image_array)

#     # 彩色图：BGR -> RGB
#     if image_array.ndim == 3 and image_array.shape[2] == 3:
#         image_array = image_array[..., ::-1]

#     if is_depth:
#         # 深度图使用 uint16 保存
#         image_array = image_array.astype(np.uint16)
#     else:
#         # mask/灰度图转换为 uint8
#         if image_array.dtype != np.uint8:
#             image_array = (image_array.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)

#     Image.fromarray(image_array).save(path)

def save_image(image_array, path, is_depth=False):
    """
    保存图像为 PNG 格式：
    - 彩色图RGB为 8-bit 三通道 JPEG
    - 深度图为 16-bit 单通道 PNG
    - Mask 为灰度 8-bit PNG
    """
    image_array = np.squeeze(image_array)

    # 彩色图：BGR -> RGB（如果是彩色图）
    if image_array.ndim == 3 and image_array.shape[2] == 3:
        image_array = image_array[..., ::-1].astype(np.uint8)  # BGR to RGB
        img = Image.fromarray(image_array, mode='RGB')
        img.save(path, format='JPEG')

    elif is_depth:
        # 深度图保存为 16-bit PNG
        # image_array = image_array.astype(np.uint16)
        # img = Image.fromarray(image_array, mode='I;16')  # 单通道 16-bit
        # img.save(path, format='PNG')
        depth_uint16 = np.clip(image_array, 0, 65535).astype(np.uint16)
        img = Image.fromarray(depth_uint16, mode='I;16')
        img.save(path, format='PNG')


    else:
        # Mask 或灰度图，保存为 8-bit 灰度图
        if image_array.dtype != np.uint8:
            # image_array = (image_array.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
            image_array = (image_array.astype(np.float32)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(image_array, mode='L')  # 单通道 8-bit
        img.save(path, format='PNG')

# def save_image(image_array, path, is_depth=False):
#     """
#     保存图像为 PNG 格式：
#     - 彩色图RGB为 8-bit 三通道 JPEG
#     - 深度图为 16-bit 单通道 PNG，单位毫米（uint16）
#     - Mask 为灰度 8-bit PNG
#     """
#     image_array = np.squeeze(image_array)

#     # 彩色图：BGR -> RGB（如果是彩色图）
#     if image_array.ndim == 3 and image_array.shape[2] == 3:
#         image_array = image_array[..., ::-1].astype(np.uint8)  # BGR to RGB
#         img = Image.fromarray(image_array, mode='RGB')
#         img.save(path, format='JPEG')

#     elif is_depth:
#         # 深度图，假设输入单位是米，转成毫米并保存成 uint16 16-bit PNG
#         depth_mm = np.where(image_array > 0, image_array * 1000, 0).astype(np.uint16)
#         img = Image.fromarray(depth_mm, mode='I;16')  # 单通道 16-bit
#         img.save(path, format='PNG')

#     else:
#         # Mask 或灰度图，保存为 8-bit 灰度图
#         if image_array.dtype != np.uint8:
#             image_array = (image_array.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
#         img = Image.fromarray(image_array, mode='L')  # 单通道 8-bit
#         img.save(path, format='PNG')



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


def save_all_data_to_npz(output_folder, colors, depths, masks):
    """
    一次性保存所有 color/depth/mask 到一个 npz 文件
    """
    np.savez_compressed(
        Path(output_folder) / "all_data.npz",
        colors=colors,
        depths=depths,
        masks=masks
    )
    print(f"[INFO] All data saved to {Path(output_folder) / 'all_data.npz'}")

def save_all_data_to_h5(output_folder, colors, depths, masks):
    """
    一次性保存所有 color/depth/mask 到一个 h5 文件
    """
    import h5py
    h5_path = Path(output_folder) / "all_data.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("colors", data=colors, compression="gzip")
        f.create_dataset("depths", data=depths, compression="gzip")
        f.create_dataset("masks", data=masks, compression="gzip")
    print(f"[INFO] All data saved to {h5_path}")


def preprocess_masks(masks, kernel_size=1):
    """
    对所有mask进行squeeze、二值化和腐蚀（如需要），返回处理后的mask
    masks: (N, C, H, W) numpy array
    """
    print(masks.shape)
    N, C,  H, W = masks.shape
    processed = np.zeros_like(masks, dtype=np.uint8)
    for i in range(N):
        for j in range(C):
            mask = masks[i, j]
            mask = np.squeeze(mask)
            mask = mask.astype(np.uint8)
            if kernel_size > 1:
                from hocap_annotation.utils.cv_utils import erode_mask
                mask = erode_mask(mask, kernel_size)
            processed[i, j] = mask
    # debug: 输出mask的唯一值
    unique_vals = np.unique(processed)
    print(f"[DEBUG] Mask unique values after preprocess: {unique_vals}")
    return processed

def preprocess_depths(depths):
    """
    对所有深度图进行预处理：将小于1或无穷大的值置为0
    depths: (N, C, H, W) numpy array
    """
    depths = np.copy(depths)
    mask_invalid = (depths < 1) | (~np.isfinite(depths))
    depths[mask_invalid] = 0
    # debug: 输出每个相机的深度最大最小平均值
    N, C, H, W = depths.shape
    for cam in range(C):
        d = depths[:, cam]
        print(f"[DEBUG] Depth stats for camera {cam}: min={np.min(d):.4f}, max={np.max(d):.4f}, mean={np.mean(d):.4f}")
    return depths

def convert_to_hocap_format(h5_path, mask_root_dir, extrinsics_yaml_path, output_root, subject_id="subject_5", tool_name="blue_scooper", mask_kernel_size=1):
    # Load data
    with h5py.File(h5_path, 'r') as f:
        imgs = f["imgs"][:]  # (N, 8, 480, 640, 3)
        depths = f["depths"][:]  # (N, 8, 480, 640)

    num_frames, num_cams = imgs.shape[0], imgs.shape[1]

    # 深度预处理
    depths = preprocess_depths(depths)

    # Load masks from folder structure
    masks = load_masks_from_folder(mask_root_dir, num_frames, num_cams)  # (N, 8, H, W)

    # 预处理mask（squeeze、二值化、腐蚀等）
    masks = preprocess_masks(masks, kernel_size=mask_kernel_size)

    # Load extrinsics from YAML
    with open(extrinsics_yaml_path, 'r') as f:
        extrinsics_yaml = yaml.safe_load(f)
    extrinsics_dict = {
        k: v for k, v in extrinsics_yaml["extrinsics"].items()
        if not k.startswith("tag_")
    }
    cam_serials = sorted(extrinsics_dict.keys())

    # Check consistency
    assert len(cam_serials) == num_cams, f"Number of cameras mismatch: {len(cam_serials)} vs {num_cams}"

    # Time tag
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path(output_root) / subject_id / time_tag
    print(f"[INFO] Saving to {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    # 保存为大文件
    save_all_data_to_h5(output_folder, imgs, depths, masks)

    # Save meta.yaml
    meta = {
        "num_frames": int(num_frames),
        "object_ids": [tool_name],
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
        "extrinsics": "extrinsics.yaml",
        "have_hololens": False,
        "have_mano": False,
        "task_id": 1,
    }

    with open(output_folder / "meta.yaml", "w") as f:
        yaml.dump(meta, f)

    print("[INFO] Done writing all_data.npz and meta.yaml!")



# ====== 如何加载数据示例 ======
# data = np.load('/path/to/all_data.npz')
# imgs = data['colors']      # (N, 8, 480, 640, 3)
# depths = data['depths']    # (N, 8, 480, 640)
# masks = data['masks']      # (N, 8, H, W)
# ===========================
# 示例调用
if __name__ == "__main__":
    # blue scooper
    # test_1
    # convert_to_hocap_format(
    #     h5_path="/home/wys/learning-compliant/crq_ws/data/0506data/blue_scooper/06c0c8e0_blue_scooper_mid_6/data00000000.h5",
    #     mask_root_dir="/home/wys/learning-compliant/crq_ws/data/0506data/blue_scooper_annotated/06c0c8e0_blue_scooper_mid_6/tool_masks",
    #     extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
    #     output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
    #     subject_id="test_1"
    # )
    # spoon
    convert_to_hocap_format(
        h5_path="/home/wys/learning-compliant/crq_ws/data/0513data/28dfb756_pestie_1/data00000000.h5",
        mask_root_dir="/home/wys/learning-compliant/crq_ws/data/0513data/28dfb756_pestie_1/masks",
        extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
        output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
        subject_id="pestle_1",
        tool_name="pestle"
    )

    # wooden_spoon
    # convert_to_hocap_format(
    #     h5_path="/home/wys/learning-compliant/crq_ws/data/raw_h5s/0b99324a_wooden_spoon_small_6/data00000000.h5",
    #     mask_root_dir="/home/wys/learning-compliant/crq_ws/data/0b99324a_wooden_spoon_small_6/tool_masks",
    #     extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
    #     output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
    #     subject_id="test_2"
    # )
