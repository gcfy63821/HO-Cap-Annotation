import os
import h5py
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import cv2


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


def convert_to_hocap_format(
    h5_path, mask_root_dir, extrinsics_yaml_path, output_root,
    subject_id="null", tool_name="blue_scooper", object_mask_dir=None
):
    # Load data
    with h5py.File(h5_path, 'r') as f:
        imgs = f["imgs"][:]  # (N, 8, 480, 640, 3)
        depths = f["depths"][:]  # (N, 8, 480, 640)

    num_frames, num_cams = imgs.shape[0], imgs.shape[1]

    # Load masks from folder structure
    masks = load_masks_from_folder(mask_root_dir, num_frames, num_cams)  # (N, 8, H, W)
    if object_mask_dir is not None:
        object_masks = load_masks_from_folder(object_mask_dir, num_frames, num_cams)

    # Load extrinsics from YAML
    with open(extrinsics_yaml_path, 'r') as f:
        extrinsics_yaml = yaml.safe_load(f)
    extrinsics_dict = {
        k: v for k, v in extrinsics_yaml["extrinsics"].items()
        if not k.startswith("tag_")
    }
    cam_serials = sorted(extrinsics_dict.keys())

    assert len(cam_serials) == num_cams, f"Number of cameras mismatch: {len(cam_serials)} vs {num_cams}"

    # Time tag
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path(output_root) / subject_id / time_tag
    print(f"[INFO] Saving to {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    # 保存为新的数据集结构，并保存每个相机的视频
    for cam_idx in range(num_cams):
        cam_serial = f"cam{cam_idx:02d}"
        cam_dir = output_folder / cam_serial / "images"
        cam_dir.mkdir(parents=True, exist_ok=True)
        # 保存视频
        video_path = output_folder / f"{cam_serial}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        H, W = imgs[0, cam_idx].shape[:2]
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (W, H))
        for frame_idx in tqdm(range(num_frames), desc=f"Camera {cam_serial}"):
            rgb = imgs[frame_idx, cam_idx]
            # 保存为png，命名为0000.png, 0001.png, ...
            img_path = cam_dir / f"{frame_idx:04d}.png"
            save_image(rgb, img_path)
            # 写入视频（BGR格式）
            img_bgr = np.array(Image.open(img_path))  # 读取保存的PNG
            if img_bgr.shape[2] == 3:
                img_bgr = img_bgr[..., ::-1]  # RGB->BGR
            video_writer.write(img_bgr)
        video_writer.release()
        # 可选：保存depth和mask（如需要可扩展）

    # 其它文件和meta.yaml保持原有逻辑
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

    print("[INFO] Done writing images and meta.yaml!")


# 示例调用
if __name__ == "__main__":
    # dataset for HandReconstruction
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
    # convert_to_hocap_format(
    #     h5_path="/home/wys/learning-compliant/crq_ws/data/0607data/6c1264cf_wooden_spoon_dough_0/data00000000.h5",
    #     mask_root_dir="/home/wys/learning-compliant/crq_ws/data/0607data/6c1264cf_wooden_spoon_dough_0_annotated/tool_masks",
    #     extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
    #     output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
    #     subject_id="test_2",
    #     tool_name="wooden_spoon"
    # )
    # pestle
    convert_to_hocap_format(
        h5_path="/home/wys/learning-compliant/crq_ws/data/0513data/28dfb756_pestie_1/data00000000.h5",
        mask_root_dir="/home/wys/learning-compliant/crq_ws/data/0513data/28dfb756_pestie_1_annotated/masks",
        extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
        output_root="/home/wys/learning-compliant/crq_ws/robotool/HandReconstruction/data",
        subject_id="pestle_1",
        tool_name="pestle"
    )
    # squeegee
    # convert_to_hocap_format(
    #     h5_path="/home/wys/learning-compliant/crq_ws/data/0513data/2379b837_coffee_1/data00000000.h5",
    #     mask_root_dir="/home/wys/learning-compliant/crq_ws/data/0513data/2379b837_coffee_1/masks",
    #     extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
    #     output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
    #     subject_id="squeegee_1",
    #     tool_name="squeegee",

    # )
    # new scooper
    # convert_to_hocap_format(
    #     h5_path="/media/wys/1146f1fa-b0e1-4f4d-bd3c-924c33902dd7/crq_ws/data/0506data/blue_scooper/8f2cf90d_blue_scooper_small_1/data00000000.h5",
    #     mask_root_dir="/media/wys/1146f1fa-b0e1-4f4d-bd3c-924c33902dd7/crq_ws/data/0506data/blue_scooper_annotated/8f2cf90d_blue_scooper_small_1/tool_masks",
    #     extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
    #     output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
    #     subject_id="blue_scooper_1",
    #     tool_name="blue_scooper",
    #     object_mask_dir = "/media/wys/1146f1fa-b0e1-4f4d-bd3c-924c33902dd7/crq_ws/data/0506data/blue_scooper_annotated/8f2cf90d_blue_scooper_small_1/object_masks"  # 如果有物体mask目录，可以传入
    # )

    # hand
    # convert_to_hocap_format(
    #     h5_path="/media/wys/1146f1fa-b0e1-4f4d-bd3c-924c33902dd7/crq_ws/data/0607data/6c1264cf_wooden_spoon_dough_0/data00000000.h5",
    #     mask_root_dir="/media/wys/1146f1fa-b0e1-4f4d-bd3c-924c33902dd7/crq_ws/data/0607data/6c1264cf_wooden_spoon_dough_0_annotated/tool_masks",
    #     extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
    #     output_root="/home/wys/learning-compliant/crq_ws/robotool/HandReconstruction/data",
    #     subject_id="wooden_spoon_1",
    #     tool_name="wooden_spoon",
    #     # object_mask_dir = "/media/wys/1146f1fa-b0e1-4f4d-bd3c-924c33902dd7/crq_ws/data/0506data/blue_scooper_annotated/8f2cf90d_blue_scooper_small_1/object_masks"  # 如果有物体mask目录，可以传入
    # )

    # convert_to_hocap_format(
    #     h5_path="/home/wys/learning-compliant/crq_ws/data/0629data/b28bdb5c_wooden_spoon_banana_3/data00000000.h5",
    #     mask_root_dir="/home/wys/learning-compliant/crq_ws/data/0629data/b28bdb5c_wooden_spoon_banana_3/tool_masks",
    #     extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
    #     output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
    #     subject_id="wooden_spoon_1",
    #     tool_name="wooden_spoon",
    #     # object_mask_dir = "/home/wys/learning-compliant/crq_ws/data/0629data/b28bdb5c_wooden_spoon_banana_3/object_masks"  # 如果有物体mask目录，可以传入
    # )

    # wooden_spoon
    # convert_to_hocap_format(
    #     h5_path="/home/wys/learning-compliant/crq_ws/data/raw_h5s/0b99324a_wooden_spoon_small_6/data00000000.h5",
    #     mask_root_dir="/home/wys/learning-compliant/crq_ws/data/0b99324a_wooden_spoon_small_6/tool_masks",
    #     extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
    #     output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
    #     subject_id="test_2"
    # )
    #     subject_id="blue_scooper_1",
    #     tool_name="blue_scooper",
    #     object_mask_dir = "/media/wys/1146f1fa-b0e1-4f4d-bd3c-924c33902dd7/crq_ws/data/0506data/blue_scooper_annotated/8f2cf90d_blue_scooper_small_1/object_masks"  # 如果有物体mask目录，可以传入
    # )

    # new spoon with plate
    # convert_to_hocap_format(
    #     h5_path="/media/wys/1146f1fa-b0e1-4f4d-bd3c-924c33902dd7/crq_ws/data/0607data/6c1264cf_wooden_spoon_dough_0/data00000000.h5",
    #     mask_root_dir="/media/wys/1146f1fa-b0e1-4f4d-bd3c-924c33902dd7/crq_ws/data/0607data/6c1264cf_wooden_spoon_dough_0_annotated/tool_masks",
    #     extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
    #     output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
    #     subject_id="wooden_spoon_1",
    #     tool_name="wooden_spoon",
    #     # object_mask_dir = "/media/wys/1146f1fa-b0e1-4f4d-bd3c-924c33902dd7/crq_ws/data/0506data/blue_scooper_annotated/8f2cf90d_blue_scooper_small_1/object_masks"  # 如果有物体mask目录，可以传入
    # )

    # convert_to_hocap_format(
    #     h5_path="/home/wys/learning-compliant/crq_ws/data/0629data/b28bdb5c_wooden_spoon_banana_3/data00000000.h5",
    #     mask_root_dir="/home/wys/learning-compliant/crq_ws/data/0629data/b28bdb5c_wooden_spoon_banana_3/tool_masks",
    #     extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
    #     output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
    #     subject_id="wooden_spoon_1",
    #     tool_name="wooden_spoon",
    #     # object_mask_dir = "/home/wys/learning-compliant/crq_ws/data/0629data/b28bdb5c_wooden_spoon_banana_3/object_masks"  # 如果有物体mask目录，可以传入
    # )

    # wooden_spoon
    # convert_to_hocap_format(
    #     h5_path="/home/wys/learning-compliant/crq_ws/data/raw_h5s/0b99324a_wooden_spoon_small_6/data00000000.h5",
    #     mask_root_dir="/home/wys/learning-compliant/crq_ws/data/0b99324a_wooden_spoon_small_6/tool_masks",
    #     extrinsics_yaml_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml",
    #     output_root="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset",
    #     subject_id="test_2"
    # )
