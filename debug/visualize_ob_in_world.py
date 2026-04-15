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


def load_pose(pose_txt):
    """从txt文件加载位姿（四元数+平移）"""
    with open(pose_txt, 'r') as f:
        arr = np.array([float(x) for x in f.read().strip().split()])
    t = np.array(arr[4:7])
    q = np.array(arr[:4])  # xyzw
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


def concat_frames_grid(frames, grid_shape=(2, 4)):
    """将8帧拼成2x4 grid"""
    assert len(frames) == grid_shape[0] * grid_shape[1]
    h, w = frames[0].shape[:2]
    rows = []
    for i in range(grid_shape[0]):
        row = np.concatenate(frames[i*grid_shape[1]:(i+1)*grid_shape[1]], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    return grid


def extract_images_from_h5(h5_path, output_dir, num_frames, num_cams):
    """
    从 h5 文件中提取所有图像并保存为图片文件，避免后续的 GPU/CPU 复制。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已经提取过
    check_file = output_dir / ".extracted"
    if check_file.exists():
        print(f"[INFO] Images already extracted to {output_dir}, skipping extraction.")
        return
    
    print(f"[INFO] Extracting images from {h5_path} to {output_dir}...")
    
    with h5py.File(h5_path, 'r') as f:
        colors = f["imgs"]  # (N, num_cams, H, W, 3)
        
        # 为每个相机创建目录
        for cam_idx in range(num_cams):
            cam_dir = output_dir / f"color_{cam_idx:02d}"
            cam_dir.mkdir(parents=True, exist_ok=True)
        
        # 批量提取图像
        batch_size = 100
        for batch_start in tqdm(range(0, num_frames, batch_size), desc="Extracting images"):
            batch_end = min(batch_start + batch_size, num_frames)
            batch_colors = colors[batch_start:batch_end]  # (batch_size, num_cams, H, W, 3)
            
            for frame_idx in range(batch_start, batch_end):
                local_idx = frame_idx - batch_start
                for cam_idx in range(num_cams):
                    img = batch_colors[local_idx, cam_idx]  # (H, W, 3)
                    # 转换为 BGR (OpenCV 格式)
                    img_bgr = img[..., ::-1].copy()
                    cam_dir = output_dir / f"color_{cam_idx:02d}"
                    cv2.imwrite(str(cam_dir / f"color_{frame_idx:06d}.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 创建标记文件
    check_file.touch()
    print(f"[INFO] Images extracted successfully to {output_dir}")


def process_frame_world_to_cam(args):
    """
    处理单帧：从世界坐标系位姿文件加载位姿，并可视化到各个相机视图
    """
    (i, image_dir, mask_data, serials, Ks_list, extrinsics_list,
     ob_in_world_root, object_idx, orig_vertices, orig_mesh_faces,
     outlier_idxs, W, H) = args
    
    frame_tiles = []
    
    # 加载位姿文件
    pose_path = ob_in_world_root / f"{i:06d}.txt"
    if not pose_path.exists():
        frame_tiles = [np.ones((H, W, 3), dtype=np.uint8) * 255 for _ in serials]
        return concat_frames_grid(frame_tiles, (2, 4))
    
    ob_in_world = load_pose(pose_path)
    
    # 预计算物体顶点在世界坐标系中的位置
    obj_vertices_world = (ob_in_world[:3, :3] @ orig_vertices.T).T + ob_in_world[:3, 3]
    
    # 处理每个相机视图
    for serial_idx, serial in enumerate(serials):
        # 从图像缓存目录加载图像
        color_path = image_dir / f"color_{serial_idx:02d}" / f"color_{i:06d}.jpg"
        if not color_path.exists():
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        
        color = cv2.imread(str(color_path))
        if color is None:
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        
        # 获取 mask（从内存中的 mask_data）
        sam_mask = mask_data[i, serial_idx]
        if sam_mask.ndim == 3:
            sam_mask = sam_mask[0]
        # 如果 mask_data 存储的是原始 mask（包含 object_idx+1），需要转换
        if sam_mask.max() > 1:
            sam_mask = (sam_mask == (object_idx + 1)).astype(np.uint8)
        else:
            sam_mask = sam_mask.astype(np.uint8)
        
        # 创建 overlay
        sam_overlay = color.copy()
        sam_overlay[sam_mask > 0] = [0, 0, 255]
        
        # 可视化物体 - 使用预计算的顶点
        world2cam = extrinsics_list[serial_idx]
        obj_vertices_cam = (world2cam[:3, :3] @ obj_vertices_world.T).T + world2cam[:3, 3]
        
        K = Ks_list[serial_idx]
        pts = project_points(obj_vertices_cam, K)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]  # 降采样以减少绘制点数
        
        vis = sam_overlay.copy()
        color_dot = (0, 0, 255) if i in outlier_idxs else (255, 0, 0)
        for x, y in pts:
            cv2.circle(vis, (x, y), 2, color_dot, -1)
        
        frame_tiles.append(vis)
    
    return concat_frames_grid(frame_tiles, (2, 4))


def process_frame_pose_npy(args):
    """
    处理单帧：从npy文件加载位姿，并可视化到各个相机视图
    """
    (i, pose_data, image_dir, mask_data, serials, Ks_list, extrinsics_list,
     object_idx, orig_vertices, orig_mesh_faces,
     outlier_idxs, W, H) = args
    
    frame_tiles = []
    
    if i >= len(pose_data):
        frame_tiles = [np.ones((H, W, 3), dtype=np.uint8) * 255 for _ in serials]
        return concat_frames_grid(frame_tiles, (2, 4))
    
    # 读取物体位姿
    qx, qy, qz, qw, tx, ty, tz = pose_data[i]
    q = np.array([qx, qy, qz, qw])
    t = np.array([tx, ty, tz])
    R_mat = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    
    # 预计算物体顶点在世界坐标系中的位置
    obj_vertices_world = (T[:3, :3] @ orig_vertices.T).T + T[:3, 3]
    
    # 处理每个相机视图
    for serial_idx, serial in enumerate(serials):
        # 从图像缓存目录加载图像
        color_path = image_dir / f"color_{serial_idx:02d}" / f"color_{i:06d}.jpg"
        if not color_path.exists():
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        
        color = cv2.imread(str(color_path))
        if color is None:
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        
        # 获取 mask（从内存中的 mask_data）
        sam_mask = mask_data[i, serial_idx]
        if sam_mask.ndim == 3:
            sam_mask = sam_mask[0]
        # 如果 mask_data 存储的是原始 mask（包含 object_idx+1），需要转换
        if sam_mask.max() > 1:
            sam_mask = (sam_mask == (object_idx + 1)).astype(np.uint8)
        else:
            sam_mask = sam_mask.astype(np.uint8)
        
        # 创建 overlay
        sam_overlay = color.copy()
        sam_overlay[sam_mask > 0] = [0, 0, 255]
        
        # 可视化物体 - 使用预计算的顶点
        world2cam = extrinsics_list[serial_idx]
        obj_vertices_cam = (world2cam[:3, :3] @ obj_vertices_world.T).T + world2cam[:3, 3]
        
        K = Ks_list[serial_idx]
        pts = project_points(obj_vertices_cam, K)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]  # 降采样以减少绘制点数
        
        vis = sam_overlay.copy()
        color_dot = (0, 0, 255) if i in outlier_idxs else (255, 0, 0)
        for x, y in pts:
            cv2.circle(vis, (x, y), 2, color_dot, -1)
        
        frame_tiles.append(vis)
    
    return concat_frames_grid(frame_tiles, (2, 4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="数据路径，如 videos_1121/fork_slice_dough/20251121_bigwoodenfork_slice_dough_half_1")
    parser.add_argument("--tool_name", type=str, default="blue_scooper", help="工具名，如 blue_scooper")
    parser.add_argument("--output_idx", type=str, default="0", help="输出编号")
    parser.add_argument("--pose_file", type=str, default="fd", choices=["fd", "adaptive", "optimized"], help="选择foundation pose 或 optimized")
    parser.add_argument("--uuid", type=str, default="", help="唯一标识符，用于区分不同运行")
    parser.add_argument("--object_idx", type=int, default=1, help="物体索引，默认为1")
    parser.add_argument("--extract_images", action="store_true", help="是否提取 h5 图像为图片文件")
    parser.add_argument("--num_workers", type=int, default=None, help="并行处理的worker数量（默认自动计算，建议2-4以避免内存溢出）")
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
    
    # 图像缓存目录
    image_cache_dir = loader._data_folder / "image_cache"
    
    if args.extract_images or not (image_cache_dir / ".extracted").exists():
        extract_images_from_h5(h5_path, image_cache_dir, num_frames, num_cams)
    
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
    
    # 加载 mesh
    orig_mesh = trimesh.load(str(loader.object_cleaned_files[args.object_idx - 1]), process=False)
    orig_vertices = orig_mesh.vertices.copy()
    orig_mesh_faces = orig_mesh.faces
    
    print(f"[INFO] Loaded mesh, vertices shape: {orig_vertices.shape}")

    # 获取路径
    task_name = loader._data_folder.parent.name
    sequence_name = loader._data_folder.name
    annotated_base = loader._data_folder.parent.parent.parent / f"{loader._folder_name}_annotated" / task_name / loader._sequence_name
    
    # 预计算 extrinsics（避免在每个进程中重复计算）
    extrinsics_list = [loader.extr2world_inv[i] for i in range(len(serials))]
    
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
    video_w = W * 4
    video_h = H * 2
    video_w = video_w - (video_w % 2)
    video_h = video_h - (video_h % 2)

    outlier_idxs = []

    ###################
    # world_to_cam_tracking (从txt文件加载位姿)
    if args.pose_file == "fd":
        ob_in_world_root = annotated_base / "processed" / "fd_pose_solver" / args.tool_name / "ob_in_world"
        
        if ob_in_world_root.exists():
            output_path1 = Path(f"debug_output/pose_npy_in_cams_video")
            output_path1.mkdir(parents=True, exist_ok=True)
            
            video_path1 = str(output_path1 / f"{args.output_idx}{'_' + args.uuid if args.uuid else ''}_world_to_cam_tracking_2x4.mp4")
            
            # 尝试使用更兼容的编码器
            fourcc = None
            video_out1 = None
            codecs_to_try = [
                ('H264', 'H.264 (most compatible)'),
                ('avc1', 'H.264 AVC1'),
                ('XVID', 'XVID'),
                ('mp4v', 'MPEG-4'),
            ]
            
            for codec_name, codec_desc in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                    video_out1 = cv2.VideoWriter(video_path1, fourcc, 20, (video_w, video_h))
                    if video_out1.isOpened():
                        print(f"[INFO] VideoWriter initialized with {codec_desc}: {video_path1}")
                        break
                except Exception as e:
                    if video_out1 is not None:
                        video_out1.release()
                    video_out1 = None
            
            if video_out1 is None or not video_out1.isOpened():
                raise RuntimeError(f"Failed to initialize VideoWriter for {video_path1}.")
            
            # 如果只有1个worker，直接在主进程中处理
            if num_workers == 1:
                print("[INFO] Using single-process mode for world_to_cam_tracking")
                for i in tqdm(range(num_frames), desc="Processing world_to_cam frames"):
                    args_tuple = (
                        i, image_cache_dir, mask_data, serials, Ks_list, extrinsics_list,
                        ob_in_world_root, args.object_idx, orig_vertices, orig_mesh_faces,
                        outlier_idxs, W, H
                    )
                    frame = process_frame_world_to_cam(args_tuple)
                    if frame.shape[:2] != (video_h, video_w):
                        frame = cv2.resize(frame, (video_w, video_h))
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    video_out1.write(frame)
            else:
                # 多进程处理
                multiprocessing.set_start_method('spawn', force=True)
                pool = multiprocessing.Pool(processes=num_workers)
                
                args_list1 = [
                    (i, image_cache_dir, mask_data, serials, Ks_list, extrinsics_list,
                     ob_in_world_root, args.object_idx, orig_vertices, orig_mesh_faces,
                     outlier_idxs, W, H)
                    for i in range(num_frames)
                ]
                
                for frame in tqdm(pool.imap(process_frame_world_to_cam, args_list1, chunksize=chunksize), total=num_frames):
                    if frame.shape[:2] != (video_h, video_w):
                        frame = cv2.resize(frame, (video_w, video_h))
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    video_out1.write(frame)
                
                pool.close()
                pool.join()
            
            video_out1.release()
            
            # 验证视频文件
            video_file1 = Path(video_path1)
            if video_file1.exists():
                file_size_mb = video_file1.stat().st_size / (1024 * 1024)
                print(f"[INFO] world_to_cam_tracking video saved: {video_path1}, size: {file_size_mb:.2f} MB")
            else:
                print(f"[ERROR] Video file was not created: {video_path1}")
        else:
            print(f"[WARNING] ob_in_world directory not found: {ob_in_world_root}, skipping world_to_cam_tracking")

    ########################
    # pose_npy_in_cams (从npy文件加载位姿)
    
    # 加载 pose 数据
    if args.pose_file == "fd":
        pose_npy_path = str(annotated_base / "processed/fd_pose_solver/fd_poses_merged_fixed.npy")
    elif args.pose_file == "adaptive":
        pose_npy_path = str(annotated_base / "processed/fd_pose_solver/adaptive_fd_poses_merged_fixed.npy")
    elif args.pose_file == "optimized":
        pose_npy_path = str(annotated_base / "processed/joint_pose_solver/poses_o.npy")
    
    if not Path(pose_npy_path).exists():
        print(f"[ERROR] Pose file not found: {pose_npy_path}")
        exit(1)
    
    pose_data = np.load(pose_npy_path)
    print(f"[INFO] Loaded pose data from {pose_npy_path}, shape: {pose_data.shape}")
    if pose_data.ndim == 3:
        pose_data = pose_data[args.object_idx - 1]
        print(f"[INFO] Using pose_data[{args.object_idx - 1}], shape: {pose_data.shape}")
    pose_data = pose_data.reshape(-1, 7)
    
    output_path2 = Path(f"debug_output/pose_npy_in_cams_video")
    output_path2.mkdir(parents=True, exist_ok=True)
    
    video_path2 = str(output_path2 / f"{args.output_idx}{'_' + args.uuid if args.uuid else ''}_{args.pose_file}_pose_npy_in_cams_2x4.mp4")
    
    # 尝试使用更兼容的编码器
    fourcc = None
    video_out2 = None
    codecs_to_try = [
        ('H264', 'H.264 (most compatible)'),
        ('avc1', 'H.264 AVC1'),
        ('XVID', 'XVID'),
        ('mp4v', 'MPEG-4'),
    ]
    
    for codec_name, codec_desc in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            video_out2 = cv2.VideoWriter(video_path2, fourcc, 20, (video_w, video_h))
            if video_out2.isOpened():
                print(f"[INFO] VideoWriter initialized with {codec_desc}: {video_path2}")
                break
        except Exception as e:
            if video_out2 is not None:
                video_out2.release()
            video_out2 = None
    
    if video_out2 is None or not video_out2.isOpened():
        raise RuntimeError(f"Failed to initialize VideoWriter for {video_path2}.")
    
    # 如果只有1个worker，直接在主进程中处理
    if num_workers == 1:
        print("[INFO] Using single-process mode for pose_npy_in_cams")
        for i in tqdm(range(num_frames), desc="Processing pose_npy frames"):
            args_tuple = (
                i, pose_data, image_cache_dir, mask_data, serials, Ks_list, extrinsics_list,
                args.object_idx, orig_vertices, orig_mesh_faces,
                outlier_idxs, W, H
            )
            frame = process_frame_pose_npy(args_tuple)
            if frame.shape[:2] != (video_h, video_w):
                frame = cv2.resize(frame, (video_w, video_h))
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            video_out2.write(frame)
    else:
        # 多进程处理
        multiprocessing.set_start_method('spawn', force=True)
        pool = multiprocessing.Pool(processes=num_workers)
        
        args_list2 = [
            (i, pose_data, image_cache_dir, mask_data, serials, Ks_list, extrinsics_list,
             args.object_idx, orig_vertices, orig_mesh_faces,
             outlier_idxs, W, H)
            for i in range(num_frames)
        ]
        
        for frame in tqdm(pool.imap(process_frame_pose_npy, args_list2, chunksize=chunksize), total=num_frames):
            if frame.shape[:2] != (video_h, video_w):
                frame = cv2.resize(frame, (video_w, video_h))
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            video_out2.write(frame)
        
        pool.close()
        pool.join()
    
    video_out2.release()
    
    # 验证视频文件
    video_file2 = Path(video_path2)
    if video_file2.exists():
        file_size_mb = video_file2.stat().st_size / (1024 * 1024)
        print(f"[INFO] pose_npy_in_cams video saved: {video_path2}, size: {file_size_mb:.2f} MB")
    else:
        print(f"[ERROR] Video file was not created: {video_path2}")
