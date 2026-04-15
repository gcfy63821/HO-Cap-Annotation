import os
import cv2
import numpy as np
from pathlib import Path
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import multiprocessing
import h5py
from hocap_annotation.loaders.my_cluster_loader import MyClusterLoader
from hocap_annotation.utils.cv_utils import get_depth_colormap


def load_pose_txt(pose_txt):
    """从txt文件加载位姿（四元数+平移）"""
    with open(pose_txt, 'r') as f:
        lines = f.readlines()
    # 读取前7个数字（前3个是平移，后4个是四元数）
    arr = np.array([float(x.strip()) for x in lines[:7]])
    t = np.array(arr[4:7])  # tx, ty, tz
    q = np.array(arr[:4])  # qx, qy, qz, qw
    R_mat = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


def project_points(vertices, K):
    """投影 3D 点到 2D 图像平面"""
    pts = vertices @ K[:3, :3].T
    pts = pts[:, :2] / pts[:, 2:]
    return pts


def extract_images_from_h5(h5_path, output_dir, num_frames, num_cams, cam_h5_indices=None):
    """
    从 h5 文件中提取所有图像并保存为图片文件，避免后续的 GPU/CPU 复制。
    Args:
        cam_h5_indices: h5 中的相机索引列表 (e.g. [2,3,4,5,6,7])。
                        如果为 None，使用 range(num_cams)。
    缓存目录结构: color_{active_idx:02d}/color_{frame:06d}.jpg
    其中 active_idx 是活动相机的顺序索引 (0, 1, 2...)。
    """
    if cam_h5_indices is None:
        cam_h5_indices = list(range(num_cams))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已经提取过，并验证参数是否匹配
    cache_key = f"{num_frames},{num_cams},{h5_path},{cam_h5_indices}"
    check_file = output_dir / ".extracted"
    if check_file.exists():
        try:
            cached = check_file.read_text().strip()
            if cached == cache_key:
                print(f"[INFO] Images already extracted to {output_dir}, skipping extraction.")
                return
            else:
                print(f"[INFO] Cache mismatch, re-extracting...")
                import shutil
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            print(f"[INFO] Invalid cache marker, re-extracting...")
            import shutil
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Extracting images from {h5_path} to {output_dir} (h5 indices: {cam_h5_indices})...")

    with h5py.File(h5_path, 'r') as f:
        colors = f["imgs"]  # (N, all_cams, H, W, 3)

        # 为每个活动相机创建目录，用顺序索引命名
        for active_idx in range(num_cams):
            cam_dir = output_dir / f"color_{active_idx:02d}"
            cam_dir.mkdir(parents=True, exist_ok=True)

        # 逐相机提取图像，从 h5 中读取正确的相机列
        batch_size = 100
        for active_idx, h5_idx in enumerate(cam_h5_indices):
            cam_dir = output_dir / f"color_{active_idx:02d}"
            for batch_start in tqdm(range(0, num_frames, batch_size),
                                     desc=f"Extracting cam {active_idx:02d} (h5 idx {h5_idx})"):
                batch_end = min(batch_start + batch_size, num_frames)
                # 从 h5 中读取正确的相机列
                batch_colors = colors[batch_start:batch_end, h5_idx]
                for i, frame_idx in enumerate(range(batch_start, batch_end)):
                    img_bgr = batch_colors[i][..., ::-1].copy()
                    cv2.imwrite(str(cam_dir / f"color_{frame_idx:06d}.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 创建标记文件，记录提取参数用于缓存校验
    check_file.write_text(cache_key)
    print(f"[INFO] Images extracted successfully to {output_dir}")


def get_depth_colormap_custom(depth, vmin=None, vmax=None, percentile_range=(5, 95)):
    """
    将深度图转换为彩色图，支持自定义范围或自动计算
    """
    if depth.ndim != 2:
        raise ValueError("Input depth must be a 2D array.")
    
    depth_float = depth.astype(np.float32)
    valid_mask = depth_float > 0
    
    if not np.any(valid_mask):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    
    # 自动计算范围或使用提供的范围
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(depth_float[valid_mask], percentile_range)
    
    # 归一化到 [0, 255]
    depth_norm = np.clip((depth_float - vmin) / (vmax - vmin + 1e-8), 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    
    # 应用颜色映射 (VIRIDIS)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    return depth_colored


def process_frame_single_cam(args):
    """
    处理单帧单相机：从txt文件加载位姿（在相机坐标系中），并可视化
    """
    (frame_idx, cam_idx, cam_h5_idx, image_dir, mask_data, depth_data, K,
     ob_in_cam_root, object_idx, orig_vertices, orig_mesh_faces,
     W, H, background_type, depth_min, depth_max) = args
    
    # 根据背景类型加载图像
    if background_type == 'rgb':
        # 加载RGB图像
        color_path = image_dir / f"color_{cam_idx:02d}" / f"color_{frame_idx:06d}.jpg"
        if not color_path.exists():
            return np.ones((H, W, 3), dtype=np.uint8) * 255
        
        color = cv2.imread(str(color_path))
        if color is None:
            return np.ones((H, W, 3), dtype=np.uint8) * 255
        # 转换为RGB格式
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    elif background_type == 'depth':
        # 从深度数据加载并转换为彩色
        if depth_data is not None:
            depth = depth_data[frame_idx, cam_idx]  # (H, W)
            # 转换为彩色深度图
            color = get_depth_colormap_custom(depth, vmin=depth_min, vmax=depth_max)
        else:
            return np.ones((H, W, 3), dtype=np.uint8) * 255
    else:
        raise ValueError(f"Unknown background type: {background_type}")
    
    # 获取 mask
    sam_mask = mask_data[frame_idx, cam_idx]
    if sam_mask.ndim == 3:
        sam_mask = sam_mask[0]
    # 如果 mask_data 存储的是原始 mask（包含 object_idx+1），需要转换
    if sam_mask.max() > 1:
        sam_mask = (sam_mask == (object_idx + 1)).astype(np.uint8)
    else:
        sam_mask = sam_mask.astype(np.uint8)
    
    # 创建 overlay
    vis = color.copy()
    sam_overlay = color.copy()
    sam_overlay[sam_mask > 0] = [0, 0, 255]
    vis = cv2.addWeighted(vis, 0.7, sam_overlay, 0.3, 0)
    
    # 加载位姿（在相机坐标系中），目录名使用校准索引
    pose_path = ob_in_cam_root / f"{cam_h5_idx:02d}" / f"{frame_idx:06d}.txt"
    if pose_path.exists():
        ob_in_cam = load_pose_txt(pose_path)
        
        # 物体顶点已经在相机坐标系中
        obj_vertices_cam = (ob_in_cam[:3, :3] @ orig_vertices.T).T + ob_in_cam[:3, 3]
        
        # 投影到2D
        pts = project_points(obj_vertices_cam, K)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]  # 降采样以减少绘制点数
        
        # 绘制物体顶点
        for x, y in pts:
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        
        # 绘制物体轮廓（使用mesh faces）
        if orig_mesh_faces is not None and len(orig_mesh_faces) > 0:
            # 只处理在图像范围内的面
            valid_faces = []
            for face in orig_mesh_faces[::50]:  # 降采样面以减少计算
                v0 = obj_vertices_cam[face[0]]
                v1 = obj_vertices_cam[face[1]]
                v2 = obj_vertices_cam[face[2]]
                
                # 检查是否在图像范围内
                if (v0[2] > 0 and v1[2] > 0 and v2[2] > 0):
                    p0 = project_points(v0.reshape(1, -1), K)[0]
                    p1 = project_points(v1.reshape(1, -1), K)[0]
                    p2 = project_points(v2.reshape(1, -1), K)[0]
                    
                    if (0 <= p0[0] < W and 0 <= p0[1] < H) or \
                       (0 <= p1[0] < W and 0 <= p1[1] < H) or \
                       (0 <= p2[0] < W and 0 <= p2[1] < H):
                        pts_2d = np.array([p0, p1, p2], dtype=np.int32)
                        cv2.polylines(vis, [pts_2d], isClosed=True, color=(0, 255, 0), thickness=1)
    
    # 添加帧号和相机信息
    cv2.putText(vis, f"Frame: {frame_idx:06d} | Cam: {cam_h5_idx:02d}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, 
                       help="数据路径，如 videos_0121/squeegee_collect_sand/20260122_squeegee_collect_sand_from_table_20")
    parser.add_argument("--tool_name", type=str, default="squeegee", help="工具名，如 squeegee")
    parser.add_argument("--output_idx", type=str, default="0", help="输出编号")
    parser.add_argument("--uuid", type=str, default="", help="唯一标识符，用于区分不同运行")
    parser.add_argument("--object_idx", type=int, default=1, help="物体索引，默认为1")
    parser.add_argument("--extract_images", action="store_true", help="是否提取 h5 图像为图片文件")
    parser.add_argument("--num_workers", type=int, default=1, 
                       help="并行处理的worker数量（默认自动计算，建议2-4以避免内存溢出）")
    parser.add_argument("--cam_idx", type=int, default=None, 
                       help="指定单个相机索引进行可视化（默认处理所有相机）")
    parser.add_argument("--background", type=str, default="rgb", choices=["rgb", "depth"],
                       help="选择背景图像类型：rgb 或 depth（默认：rgb）")
    args = parser.parse_args()

    # 初始化 loader
    sequence_folder = args.data_path
    loader = MyClusterLoader(sequence_folder)

    serials = loader.rs_serials
    Ks_list = loader.rs_Ks  # 列表格式
    W, H = loader.rs_width, loader.rs_height
    num_frames = loader.num_frames
    num_cams = len(serials)

    print(f"[INFO] Processing {num_frames} frames with {num_cams} cameras")
    print(f"[INFO] Camera serials: {serials}")

    # 提取图像（如果需要）
    h5_files = list(loader._data_folder.glob('*.h5'))
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No .h5 file found in {loader._data_folder}")
    h5_path = h5_files[0]
    
    # 图像缓存目录（仅在需要RGB背景时提取）
    image_cache_dir = loader._data_folder / "image_cache"
    
    if args.background == 'rgb':
        if args.extract_images or not (image_cache_dir / ".extracted").exists():
            extract_images_from_h5(h5_path, image_cache_dir, num_frames, num_cams,
                                   cam_h5_indices=loader.cam_h5_indices)
    else:
        print(f"[INFO] Using depth background, skipping RGB image extraction")
    
    # 加载 mask 数据到内存
    print("[INFO] Loading mask data into memory...")
    try:
        mask_data = loader._all_masks  # 直接从 loader 获取
    except AttributeError:
        # 如果无法直接访问，则批量加载
        print("[INFO] Loading masks via get_mask method...")
        mask_data = np.zeros((num_frames, num_cams, H, W), dtype=np.uint8)
        for frame_idx in tqdm(range(num_frames), desc="Loading masks"):
            for cam_idx, serial in enumerate(serials):
                mask_data[frame_idx, cam_idx] = loader.get_mask(serial, frame_idx, args.object_idx - 1)
    
    # 加载深度数据（如果使用深度背景）
    depth_data = None
    depth_min = None
    depth_max = None
    if args.background == 'depth':
        print("[INFO] Loading depth data into memory...")
        try:
            depth_data = loader._all_depths  # 直接从 loader 获取，单位是米
            # 计算全局深度范围（用于一致的颜色映射）
            valid_depths = depth_data[depth_data > 0]
            if len(valid_depths) > 0:
                depth_min, depth_max = np.percentile(valid_depths, [5, 95])
                print(f"[INFO] Depth range for colormap: {depth_min:.3f}m - {depth_max:.3f}m")
            else:
                print("[WARNING] No valid depth values found, using default range")
                depth_min, depth_max = 0.0, 5.0
        except AttributeError:
            print("[ERROR] Cannot access depth data from loader")
            depth_data = None
    
    # 加载 mesh
    orig_mesh = trimesh.load(str(loader.object_cleaned_files[args.object_idx - 1]), process=False)
    orig_vertices = orig_mesh.vertices.copy()
    orig_mesh_faces = orig_mesh.faces
    
    print(f"[INFO] Loaded mesh, vertices shape: {orig_vertices.shape}, faces shape: {orig_mesh_faces.shape}")

    # 获取路径
    task_name = loader._data_folder.parent.name
    sequence_name = loader._data_folder.name
    annotated_base = loader._data_folder.parent.parent.parent / f"{loader._folder_name}_annotated" / task_name / loader._sequence_name
    
    # 跟踪结果路径
    ob_in_cam_root = annotated_base / "processed" / "fd_pose_solver" / args.tool_name / "ob_in_cam"
    
    if not ob_in_cam_root.exists():
        raise FileNotFoundError(f"Tracking results directory not found: {ob_in_cam_root}")
    
    print(f"[INFO] Tracking results directory: {ob_in_cam_root}")
    
    # 计算mask_data的内存占用
    mask_memory_mb = mask_data.nbytes / (1024 * 1024)
    print(f"[INFO] Mask data memory: {mask_memory_mb:.2f} MB")
    
    # 根据用户指定或自动计算worker数量
    if args.num_workers is not None:
        num_workers = max(1, min(args.num_workers, os.cpu_count() or 4))
        print(f"[INFO] Using user-specified {num_workers} worker processes")
    else:
        # 根据可用内存调整worker数量
        max_memory_gb = 16  # 假设系统有16GB可用内存
        max_workers_by_memory = int(max_memory_gb * 1024 / (mask_memory_mb + 500))  # 500MB其他开销
        num_workers = min(4, max_workers_by_memory, os.cpu_count() or 4)  # 最多4个worker
        num_workers = max(1, num_workers)  # 至少1个worker
        print(f"[INFO] Auto-calculated {num_workers} worker processes (reduced to avoid OOM)")
    
    chunksize = max(1, num_frames // (num_workers * 2))
    print(f"[INFO] Using {num_workers} worker processes with chunksize {chunksize}")

    # 确保视频尺寸是偶数
    video_w = W
    video_h = H
    video_w = video_w - (video_w % 2)
    video_h = video_h - (video_h % 2)

    # 确定要处理的相机列表
    cam_indices = [args.cam_idx] if args.cam_idx is not None else list(range(num_cams))
    
    # 为每个相机生成视频
    cam_h5_indices = loader.cam_h5_indices
    for cam_idx in cam_indices:
        cam_h5_idx = cam_h5_indices[cam_idx]
        print(f"\n[INFO] Processing camera {cam_idx:02d} (h5 idx {cam_h5_idx})...")
        
        output_path = Path(f"debug_output/tracking_result_in_cam_new")
        output_path.mkdir(parents=True, exist_ok=True)
        
        video_path = str(output_path / f"{args.output_idx}{'_' + args.uuid if args.uuid else ''}_cam{cam_idx:02d}_tracking_{args.background}.mp4")
        
        # 尝试使用更兼容的编码器
        fourcc = None
        video_out = None
        codecs_to_try = [
            ('H264', 'H.264 (most compatible)'),
            ('avc1', 'H.264 AVC1'),
            ('XVID', 'XVID'),
            ('mp4v', 'MPEG-4'),
        ]
        
        for codec_name, codec_desc in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                video_out = cv2.VideoWriter(video_path, fourcc, 20, (video_w, video_h))
                if video_out.isOpened():
                    print(f"[INFO] VideoWriter initialized with {codec_desc}: {video_path}")
                    break
            except Exception as e:
                if video_out is not None:
                    video_out.release()
                video_out = None
        
        if video_out is None or not video_out.isOpened():
            print(f"[ERROR] Failed to initialize VideoWriter for {video_path}, skipping camera {cam_idx:02d}")
            continue
        
        K = Ks_list[cam_idx]
        
        # 如果只有1个worker，直接在主进程中处理
        if num_workers == 1:
            print(f"[INFO] Using single-process mode for camera {cam_idx:02d}")
            for frame_idx in tqdm(range(num_frames), desc=f"Processing cam {cam_idx:02d}"):
                args_tuple = (
                    frame_idx, cam_idx, cam_h5_idx, image_cache_dir, mask_data, depth_data, K,
                    ob_in_cam_root, args.object_idx, orig_vertices, orig_mesh_faces,
                    W, H, args.background, depth_min, depth_max
                )
                frame = process_frame_single_cam(args_tuple)
                if frame.shape[:2] != (video_h, video_w):
                    frame = cv2.resize(frame, (video_w, video_h))
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                # 转换为BGR格式（OpenCV格式）
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_out.write(frame)
        else:
            # 多进程处理
            multiprocessing.set_start_method('spawn', force=True)
            pool = multiprocessing.Pool(processes=num_workers)
            
            args_list = [
                (frame_idx, cam_idx, image_cache_dir, mask_data, depth_data, K,
                 ob_in_cam_root, args.object_idx, orig_vertices, orig_mesh_faces,
                 W, H, args.background, depth_min, depth_max)
                for frame_idx in range(num_frames)
            ]
            
            for frame in tqdm(pool.imap(process_frame_single_cam, args_list, chunksize=chunksize), 
                             total=num_frames, desc=f"Processing cam {cam_idx:02d}"):
                if frame.shape[:2] != (video_h, video_w):
                    frame = cv2.resize(frame, (video_w, video_h))
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                # 转换为BGR格式（OpenCV格式）
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_out.write(frame)
            
            pool.close()
            pool.join()
        
        video_out.release()
        
        # 验证视频文件
        video_file = Path(video_path)
        if video_file.exists():
            file_size_mb = video_file.stat().st_size / (1024 * 1024)
            print(f"[INFO] Camera {cam_idx:02d} video saved: {video_path}, size: {file_size_mb:.2f} MB")
            
            # 尝试使用ffmpeg重新编码为H.264以确保兼容性
            if file_size_mb > 0.1:
                print(f"[INFO] Attempting to re-encode camera {cam_idx:02d} video with H.264...")
                output_path_h264 = video_file.parent / f"{video_file.stem}_h264.mp4"
                try:
                    import subprocess
                    cmd = [
                        'ffmpeg', '-y', '-i', str(video_file),
                        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                        '-c:a', 'copy', '-pix_fmt', 'yuv420p',
                        str(output_path_h264)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode == 0 and output_path_h264.exists():
                        h264_size_mb = output_path_h264.stat().st_size / (1024 * 1024)
                        print(f"[SUCCESS] Re-encoded video saved: {output_path_h264}, size: {h264_size_mb:.2f} MB")
                except Exception as e:
                    print(f"[WARNING] Failed to re-encode video: {e}")
        else:
            print(f"[ERROR] Video file was not created: {video_path}")
    
    print("\n[INFO] All cameras processed!")
