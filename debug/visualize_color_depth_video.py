"""
可视化每个视角的彩色视频和深度视频并排对比，用于检查深度与彩色是否对齐一致。
输出为一个 grid 视频，每个 cell 是某个相机的 [color | depth] 并排画面。
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math
import subprocess

from hocap_annotation.loaders.my_cluster_loader import MyClusterLoader


def depth_to_colormap(depth, max_depth=2.0):
    """将深度图转换为伪彩色图用于可视化。

    Args:
        depth: (H, W) float32, 单位为米
        max_depth: 最大深度截断值（米）
    Returns:
        (H, W, 3) uint8 BGR 伪彩色图
    """
    depth_clipped = np.clip(depth, 0, max_depth)
    depth_norm = (depth_clipped / max_depth * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    # 无效深度（=0）显示为黑色
    depth_color[depth <= 0] = 0
    return depth_color


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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="可视化每个视角的彩色+深度并排视频")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据路径，如 videos_1121/fork_slice_dough/20251121_xxx")
    parser.add_argument("--max_depth", type=float, default=2.0,
                        help="深度可视化最大值（米），默认 2.0")
    parser.add_argument("--fps", type=int, default=20, help="输出视频帧率")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="每个 cell 的缩放比例，默认 0.5（减小输出尺寸）")
    parser.add_argument("--output_name", type=str, default="",
                        help="输出文件名后缀")
    args = parser.parse_args()

    # 初始化 loader
    loader = MyClusterLoader(args.data_path)
    serials = loader.rs_serials
    num_cams = len(serials)
    num_frames = loader.num_frames
    W, H = loader.rs_width, loader.rs_height
    print(f"[INFO] {num_cams} cameras, {num_frames} frames, resolution {W}x{H}")

    # 计算 grid 布局
    grid_cols = math.ceil(math.sqrt(num_cams))
    grid_rows = math.ceil(num_cams / grid_cols)
    grid_shape = (grid_rows, grid_cols)
    print(f"[INFO] Grid layout: {grid_rows}x{grid_cols}")

    # 单个 cell 尺寸：color 和 depth 并排
    cell_w = int(W * 2 * args.scale)  # 并排两张图
    cell_h = int(H * args.scale)
    video_w = cell_w * grid_cols
    video_h = cell_h * grid_rows
    print(f"[INFO] Output video size: {video_w}x{video_h}")

    # 准备输出路径
    script_dir = Path(__file__).resolve().parent.parent
    output_dir = script_dir / "debug_output" / "color_depth_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_name = loader.task_name
    seq_name = loader._data_folder.name
    suffix = f"_{args.output_name}" if args.output_name else ""
    video_path = output_dir / f"{task_name}_{seq_name}{suffix}_color_depth.mp4"

    # 初始化 VideoWriter
    codecs_to_try = [('H264',), ('avc1',), ('XVID',), ('mp4v',)]
    video_out = None
    for (codec,) in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_out = cv2.VideoWriter(str(video_path), fourcc, args.fps, (video_w, video_h))
            if video_out.isOpened():
                print(f"[INFO] VideoWriter: codec={codec}, output={video_path}")
                break
        except Exception:
            if video_out is not None:
                video_out.release()
            video_out = None

    if video_out is None or not video_out.isOpened():
        raise RuntimeError("Failed to initialize VideoWriter, no compatible codec found.")

    # 获取所有数据的引用（已在内存中）
    all_colors = loader.get_all_colors()   # (N, num_cams, H, W, 3) RGB
    all_depths = loader.get_all_depths()   # (N, num_cams, H, W) float32 meters

    # 逐帧处理
    for frame_idx in tqdm(range(num_frames), desc="Generating video"):
        frame_tiles = []
        for cam_idx in range(num_cams):
            # 彩色图：RGB -> BGR
            color = all_colors[frame_idx, cam_idx]
            color_bgr = color[..., ::-1].copy()

            # 深度图：转伪彩色
            depth = all_depths[frame_idx, cam_idx]
            depth_vis = depth_to_colormap(depth, max_depth=args.max_depth)

            # 添加相机标签
            label = f"cam{cam_idx}"
            cv2.putText(color_bgr, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(depth_vis, "depth", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 并排拼接
            pair = np.concatenate([color_bgr, depth_vis], axis=1)  # (H, 2W, 3)

            # 缩放
            if args.scale != 1.0:
                pair = cv2.resize(pair, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

            frame_tiles.append(pair)

        grid_frame = concat_frames_grid(frame_tiles, grid_shape)
        video_out.write(grid_frame)

    video_out.release()

    # ffmpeg 重编码为 H.264 确保兼容性
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
    except FileNotFoundError:
        print(f"[WARNING] ffmpeg not found, using raw video.")
    except Exception as e:
        print(f"[WARNING] ffmpeg error: {e}")

    print("[DONE]")


if __name__ == "__main__":
    main()
