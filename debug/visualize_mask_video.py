"""
可视化每个视角的彩色视频与 tool mask 叠加效果，用于检查 mask 是否正确覆盖目标物体。
支持显示 raw mask（所有 object index）或指定 object_idx 的 mask。
输出为一个 grid 视频，每个 cell 是某个相机的 [color+mask overlay] 画面。

用法:
    python debug/visualize_mask_video.py --data_path <sequence_folder>
    python debug/visualize_mask_video.py --data_path <sequence_folder> --object_idx 0
    python debug/visualize_mask_video.py --data_path <sequence_folder> --object_idx 0 --cam_idx 4
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math
import subprocess
import h5py


# Distinct colors for different mask indices (BGR)
MASK_COLORS = [
    (0, 0, 0),        # 0: background (not drawn)
    (0, 0, 255),      # 1: red
    (0, 255, 0),      # 2: green
    (255, 0, 0),      # 3: blue
    (0, 255, 255),    # 4: yellow
    (255, 0, 255),    # 5: magenta
    (255, 255, 0),    # 6: cyan
    (128, 0, 255),    # 7: orange-ish
    (255, 128, 0),    # 8: light blue
]


def overlay_mask_on_image(color_bgr, mask, alpha=0.4, object_idx=None):
    """
    将 mask 叠加到彩色图上。

    Args:
        color_bgr: (H, W, 3) uint8 BGR
        mask: (H, W) uint8，mask 值代表不同 object index
        alpha: 叠加透明度
        object_idx: 如果指定，只显示该 object 的 mask（0-based），mask 值为 object_idx+1
    Returns:
        (H, W, 3) uint8 BGR 叠加图
    """
    overlay = color_bgr.copy()
    if mask.ndim == 3:
        mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[:, :, 0]

    if object_idx is not None:
        # 只显示指定 object
        binary = (mask == (object_idx + 1))
        color = MASK_COLORS[min((object_idx + 1), len(MASK_COLORS) - 1)]
        overlay[binary] = (
            np.array(color, dtype=np.float32) * alpha +
            overlay[binary].astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)
        # 画轮廓
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 1)
    else:
        # 显示所有 object
        unique_vals = np.unique(mask)
        for val in unique_vals:
            if val == 0:
                continue
            binary = (mask == val)
            color = MASK_COLORS[min(int(val), len(MASK_COLORS) - 1)]
            overlay[binary] = (
                np.array(color, dtype=np.float32) * alpha +
                overlay[binary].astype(np.float32) * (1 - alpha)
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 1)

    return overlay


def concat_frames_grid(frames, grid_shape):
    """将帧拼成 grid"""
    rows, cols = grid_shape
    h, w = frames[0].shape[:2]
    while len(frames) < rows * cols:
        frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    row_imgs = []
    for i in range(rows):
        row = np.concatenate(frames[i * cols:(i + 1) * cols], axis=1)
        row_imgs.append(row)
    return np.concatenate(row_imgs, axis=0)


def load_masks_from_folder(mask_root_dir, num_frames, num_cams, rs_height, rs_width, start_frame=0):
    """
    从 h5 或 npy 文件加载 masks。
    Args:
        start_frame: mask npy 文件使用原始视频帧号索引，需要加上此偏移
    Returns:
        np.ndarray: shape (num_frames, num_cams, H, W)
    """
    mask_root_dir = Path(mask_root_dir)

    # Try h5 first (h5 already saved with correct offset during generate_meta)
    h5_path = mask_root_dir / "masks.h5"
    if h5_path.exists():
        with h5py.File(h5_path, 'r') as f:
            masks = f["masks"][:]
        print(f"[INFO] Loaded masks from {h5_path}, shape: {masks.shape}")
        return masks

    # Fallback to npy - use original frame index
    print(f"[INFO] Loading masks from npy files in {mask_root_dir} (start_frame={start_frame})...")
    all_masks = []
    for frame_idx in range(num_frames):
        original_frame_idx = frame_idx + start_frame
        frame_masks = []
        for cam_idx in range(num_cams):
            cam_folder = mask_root_dir / f"cam{cam_idx}_rgb"
            npy_path = None
            for fmt in [f"{original_frame_idx:04d}.npy", f"{original_frame_idx}.npy"]:
                test_path = cam_folder / fmt
                if test_path.exists():
                    npy_path = test_path
                    break

            if npy_path is None:
                frame_masks.append(np.zeros((rs_height, rs_width), dtype=np.uint8))
                continue

            mask = np.load(npy_path)
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            if mask.shape[0] != rs_height or mask.shape[1] != rs_width:
                mask = cv2.resize(mask, (rs_width, rs_height), interpolation=cv2.INTER_NEAREST)
            frame_masks.append(mask)
        all_masks.append(frame_masks)
    all_masks = np.array(all_masks)
    print(f"[INFO] Loaded masks shape: {all_masks.shape}")
    return all_masks


def main():
    import argparse

    parser = argparse.ArgumentParser(description="可视化每个视角的彩色图+mask叠加视频")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据路径（sequence folder）")
    parser.add_argument("--object_idx", type=int, default=None,
                        help="要显示的 object index（0-based），不指定则显示所有 object")
    parser.add_argument("--cam_idx", type=int, default=None,
                        help="只显示指定相机（0-based），不指定则显示所有")
    parser.add_argument("--fps", type=int, default=20, help="输出视频帧率")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="每个 cell 的缩放比例")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="mask 叠加透明度")
    parser.add_argument("--output_name", type=str, default="",
                        help="输出文件名后缀")
    args = parser.parse_args()

    data_folder = Path(args.data_path)

    # --- 读取 meta.yaml 获取基本信息 ---
    import yaml
    meta_path = data_folder / "meta.yaml"
    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)

    num_frames = meta["num_frames"]
    start_frame = meta.get("start_frame", 0)
    rs_width = meta["realsense"]["width"]
    rs_height = meta["realsense"]["height"]
    rs_serials = meta["realsense"]["serials"]
    num_cams = len(rs_serials)
    object_ids = meta["object_ids"]

    print(f"[INFO] {num_cams} cameras, {num_frames} frames, start_frame={start_frame}, resolution {rs_width}x{rs_height}")
    print(f"[INFO] Objects: {object_ids}")

    # --- 加载 h5 数据 ---
    h5_files = list(data_folder.glob("*.h5"))
    assert len(h5_files) > 0, f"No .h5 file found in {data_folder}"
    h5_path = h5_files[0]
    print(f"[INFO] Loading images from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        all_colors = f["imgs"][:]  # (N, num_cams, H, W, 3) RGB

    actual_frames = all_colors.shape[0]
    actual_cams = all_colors.shape[1]
    print(f"[INFO] H5 data: {actual_frames} frames, {actual_cams} cameras")

    if actual_frames < num_frames:
        print(f"[WARNING] H5 has {actual_frames} frames but meta says {num_frames}, using {actual_frames}")
        num_frames = actual_frames
    if actual_cams < num_cams:
        print(f"[WARNING] H5 has {actual_cams} cameras but meta says {num_cams}, using {actual_cams}")
        num_cams = actual_cams

    # --- 加载 masks ---
    # 查找 annotated 路径
    seq_name = data_folder.name
    task_name = data_folder.parent.name
    folder_name = data_folder.parent.parent.name
    annotated_path = data_folder.parent.parent.parent / f"{folder_name}_annotated" / task_name / seq_name

    tool_masks_folder = annotated_path / "tool_masks"
    masks_folder = annotated_path / "masks"

    if tool_masks_folder.exists():
        mask_folder = tool_masks_folder
    elif masks_folder.exists():
        mask_folder = masks_folder
    else:
        raise FileNotFoundError(f"No tool_masks or masks folder found in {annotated_path}")

    print(f"[INFO] Loading masks from {mask_folder} (start_frame={start_frame})...")
    all_masks = load_masks_from_folder(mask_folder, num_frames, num_cams, rs_height, rs_width, start_frame=start_frame)

    # --- 确定要显示的相机 ---
    if args.cam_idx is not None:
        cam_indices = [args.cam_idx]
    else:
        cam_indices = list(range(num_cams))

    display_cams = len(cam_indices)
    grid_cols = math.ceil(math.sqrt(display_cams))
    grid_rows = math.ceil(display_cams / grid_cols)
    grid_shape = (grid_rows, grid_cols)

    cell_w = int(rs_width * args.scale)
    cell_h = int(rs_height * args.scale)
    video_w = cell_w * grid_cols
    video_h = cell_h * grid_rows
    print(f"[INFO] Grid layout: {grid_rows}x{grid_cols}, output size: {video_w}x{video_h}")

    # --- 准备输出路径 ---
    output_dir = Path(__file__).resolve().parent.parent / "debug_output" / "mask_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    obj_suffix = f"_obj{args.object_idx}" if args.object_idx is not None else "_all"
    cam_suffix = f"_cam{args.cam_idx}" if args.cam_idx is not None else ""
    extra_suffix = f"_{args.output_name}" if args.output_name else ""
    video_path = output_dir / f"{task_name}_{seq_name}{obj_suffix}{cam_suffix}{extra_suffix}_mask.mp4"

    # --- 写视频 ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(str(video_path), fourcc, args.fps, (video_w, video_h))
    if not video_out.isOpened():
        raise RuntimeError("Failed to initialize VideoWriter")

    for frame_idx in tqdm(range(num_frames), desc="Generating video"):
        frame_tiles = []
        for cam_idx in cam_indices:
            # 彩色图 RGB -> BGR
            color = all_colors[frame_idx, cam_idx]
            color_bgr = color[..., ::-1].copy()

            # Mask
            mask = all_masks[frame_idx, cam_idx]
            if mask.ndim == 3:
                mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[:, :, 0]

            # 叠加
            vis = overlay_mask_on_image(color_bgr, mask, alpha=args.alpha, object_idx=args.object_idx)

            # 标签
            label = f"cam{cam_idx}"
            mask_sum = int(np.sum(mask > 0)) if args.object_idx is None else int(np.sum(mask == (args.object_idx + 1)))
            info = f"{label} f{frame_idx} px={mask_sum}"
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 缩放
            if args.scale != 1.0:
                vis = cv2.resize(vis, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

            frame_tiles.append(vis)

        grid_frame = concat_frames_grid(frame_tiles, grid_shape)
        video_out.write(grid_frame)

    video_out.release()

    # ffmpeg 重编码
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Raw video saved: {video_path} ({file_size_mb:.2f} MB)")

    h264_path = video_path.with_name(video_path.stem + "_h264.mp4")
    try:
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            str(h264_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and h264_path.exists():
            h264_size = h264_path.stat().st_size / (1024 * 1024)
            print(f"[INFO] H.264 video saved: {h264_path} ({h264_size:.2f} MB)")
        else:
            print(f"[WARNING] ffmpeg re-encoding failed, using raw video.")
    except Exception as e:
        print(f"[WARNING] ffmpeg error: {e}")

    print("[DONE]")


if __name__ == "__main__":
    main()
