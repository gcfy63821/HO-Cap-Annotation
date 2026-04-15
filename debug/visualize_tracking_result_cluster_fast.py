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


def load_pose_from_txt(pose_txt):
    """从txt文件加载pose（quaternion格式：qx qy qz qw tx ty tz）"""
    with open(pose_txt, 'r') as f:
        arr = np.array([float(x) for x in f.read().strip().split()])
        t = np.array(arr[4:7])  # translation
        q = np.array(arr[:4])  # xyzw
        R_mat = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        return T


def load_pose_from_npy(pose_npy, frame_idx, object_idx=0):
    """从npy文件加载pose（形状为(num_objects, num_frames, 7)或(num_frames, 7)）"""
    if pose_npy.ndim == 3:
        pose = pose_npy[object_idx, frame_idx]  # (7,)
    else:
        pose = pose_npy[frame_idx]  # (7,)
    
    qx, qy, qz, qw, tx, ty, tz = pose
    q = np.array([qx, qy, qz, qw])
    t = np.array([tx, ty, tz])
    R_mat = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


def load_all_poses_from_txt(pose_folder, num_frames):
    """从txt文件加载所有pose"""
    poses = []
    for i in range(num_frames):
        pose_path = pose_folder / f"{i:06d}.txt"
        if not pose_path.exists():
            poses.append(None)
            continue
        try:
            poses.append(load_pose_from_txt(pose_path))
        except Exception as e:
            print(f"[WARNING] Failed to load pose from {pose_path}: {e}")
            poses.append(None)
    return poses


def linear_interpolate_poses(poses, num_frames):
    """对于缺失的pose使用线性插值填充"""
    interpolated_poses = []
    for i in range(num_frames):
        if poses[i] is not None:
            interpolated_poses.append(poses[i])
        else:
            # 找到前后非空的pose进行插值
            prev_pose = None
            next_pose = None
            prev_idx = None
            next_idx = None
            for j in range(i - 1, -1, -1):
                if poses[j] is not None:
                    prev_pose = poses[j]
                    prev_idx = j
                    break
            for j in range(i + 1, num_frames):
                if poses[j] is not None:
                    next_pose = poses[j]
                    next_idx = j
                    break
            
            if prev_pose is not None and next_pose is not None:
                # 线性插值：对旋转和平移分别插值
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                
                # 旋转矩阵插值（使用SLERP）
                prev_R = prev_pose[:3, :3]
                next_R = next_pose[:3, :3]
                prev_q = R.from_matrix(prev_R).as_quat()  # xyzw
                next_q = R.from_matrix(next_R).as_quat()
                
                # 简单的线性插值（更精确可以用SLERP）
                interp_q = (1 - alpha) * prev_q + alpha * next_q
                interp_q = interp_q / np.linalg.norm(interp_q)
                interp_R = R.from_quat(interp_q).as_matrix()
                
                # 平移插值
                prev_t = prev_pose[:3, 3]
                next_t = next_pose[:3, 3]
                interp_t = (1 - alpha) * prev_t + alpha * next_t
                
                interp_pose = np.eye(4)
                interp_pose[:3, :3] = interp_R
                interp_pose[:3, 3] = interp_t
                interpolated_poses.append(interp_pose)
            elif prev_pose is not None:
                interpolated_poses.append(prev_pose)
            elif next_pose is not None:
                interpolated_poses.append(next_pose)
            else:
                interpolated_poses.append(None)
    return interpolated_poses


def project_points(vertices, K):
    """投影3D点到2D图像平面"""
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


def process_frame_tracking(args):
    """
    处理单帧的跟踪结果可视化
    """
    (i, image_dir, mask_data, serial_idx, serial, K, 
     orig_vertices, pre_optim_poses, post_optim_poses,
     object_idx, W, H, outlier_idxs) = args
    
    # 加载图像
    color_path = image_dir / f"color_{serial_idx:02d}" / f"color_{i:06d}.jpg"
    if not color_path.exists():
        return None
    
    color = cv2.imread(str(color_path))
    if color is None:
        return None
    
    # 获取mask
    sam_mask = mask_data[i, serial_idx]
    if sam_mask.ndim == 3:
        sam_mask = sam_mask[0]
    if sam_mask.max() > 1:
        sam_mask = (sam_mask == (object_idx + 1)).astype(np.uint8)
    else:
        sam_mask = sam_mask.astype(np.uint8)
    
    # 创建SAM overlay
    sam_overlay = color.copy()
    sam_overlay[sam_mask > 0] = [0, 0, 255]
    
    # ---------- 优化前重投影 ----------
    T_pre = pre_optim_poses[i] if i < len(pre_optim_poses) else None
    if T_pre is not None:
        obj_vertices_cam = (T_pre[:3, :3] @ orig_vertices.T).T + T_pre[:3, 3]
        pts = project_points(obj_vertices_cam, K)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) &
                  (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]  # 降采样
        pre_frame = color.copy()
        # 添加mask overlay
        pre_frame[sam_mask > 0] = [0, 0, 255]
        for x, y in pts:
            c = (0, 0, 255) if i in outlier_idxs else (255, 0, 0)
            cv2.circle(pre_frame, (x, y), 2, c, -1)
    else:
        pre_frame = np.ones_like(color) * 255
        # 即使没有pose，也显示mask
        pre_frame[sam_mask > 0] = [0, 0, 255]
    
    # ---------- 优化后重投影 ----------
    T_post = post_optim_poses[i] if i < len(post_optim_poses) else None
    if T_post is not None:
        obj_vertices_cam = (T_post[:3, :3] @ orig_vertices.T).T + T_post[:3, 3]
        pts = project_points(obj_vertices_cam, K)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) &
                  (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]  # 降采样
        post_frame = color.copy()
        # 添加mask overlay
        post_frame[sam_mask > 0] = [0, 0, 255]
        for x, y in pts:
            c = (0, 0, 255) if i in outlier_idxs else (0, 255, 0)  # 优化后用绿色
            cv2.circle(post_frame, (x, y), 2, c, -1)
    else:
        post_frame = np.ones_like(color) * 255
        # 即使没有pose，也显示mask
        post_frame[sam_mask > 0] = [0, 0, 255]
    
    # 拼图（2x2）：原始图像、SAM mask、优化前、优化后
    top = np.concatenate((color, sam_overlay), axis=1)
    bottom = np.concatenate((pre_frame, post_frame), axis=1)
    final = np.concatenate((top, bottom), axis=0)
    
    return final


def visualize_tracking_for_camera(
    loader, serial_idx, serial, image_dir, mask_data,
    pre_optim_poses, post_optim_poses,
    orig_vertices, K, output_path,
    num_frames, object_idx, outlier_idxs=[]
):
    """
    为单个相机生成跟踪结果视频
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    W, H = loader.rs_width, loader.rs_height
    
    video_out = cv2.VideoWriter(
        str(output_path / f"tracking_result_cam{serial_idx:02d}.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (W * 2, H * 2)
    )
    
    # 使用多进程处理
    num_workers = min(4, os.cpu_count() or 4)
    chunksize = max(1, num_frames // (num_workers * 4))
    
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=num_workers)
    
    args_list = [
        (i, image_dir, mask_data, serial_idx, serial, K,
         orig_vertices, pre_optim_poses, post_optim_poses,
         object_idx, W, H, outlier_idxs)
        for i in range(num_frames)
    ]
    
    for frame in tqdm(pool.imap(process_frame_tracking, args_list, chunksize=chunksize), 
                      total=num_frames, desc=f"Camera {serial_idx:02d}"):
        if frame is not None:
            video_out.write(frame)
    
    pool.close()
    pool.join()
    video_out.release()
    
    print(f"[INFO] Video saved to {output_path / f'tracking_result_cam{serial_idx:02d}.mp4'}")


def process_frame_world_to_cam(args):
    """
    处理单帧的世界坐标系到相机坐标系的可视化
    """
    (i, image_dir, mask_data, serials, Ks_list, extrinsics_list,
     ob_in_world_pose, orig_vertices, object_idx, W, H, outlier_idxs) = args
    
    frame_tiles = []
    
    if ob_in_world_pose is None:
        frame_tiles = [np.ones((H, W, 3), dtype=np.uint8) * 255 for _ in serials]
        return concat_frames_grid(frame_tiles, (2, 4))
    
    # 处理每个相机视图
    for serial_idx, serial in enumerate(serials):
        # 从图片文件加载
        color_path = image_dir / f"color_{serial_idx:02d}" / f"color_{i:06d}.jpg"
        if not color_path.exists():
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        
        color = cv2.imread(str(color_path))
        if color is None:
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        
        # 获取mask
        sam_mask = mask_data[i, serial_idx]
        if sam_mask.ndim == 3:
            sam_mask = sam_mask[0]
        if sam_mask.max() > 1:
            sam_mask = (sam_mask == (object_idx + 1)).astype(np.uint8)
        else:
            sam_mask = sam_mask.astype(np.uint8)
        
        # 创建SAM overlay
        sam_overlay = color.copy()
        sam_overlay[sam_mask > 0] = [0, 0, 255]
        
        # 将世界坐标系的pose转换到相机坐标系
        world2cam = extrinsics_list[serial_idx]
        obj_vertices_world = (ob_in_world_pose[:3, :3] @ orig_vertices.T).T + ob_in_world_pose[:3, 3]
        obj_vertices_cam = (world2cam[:3, :3] @ obj_vertices_world.T).T + world2cam[:3, 3]
        
        K = Ks_list[serial_idx]
        pts = project_points(obj_vertices_cam, K)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) &
                  (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]  # 降采样
        
        vis = sam_overlay.copy()
        color_dot = (0, 0, 255) if i in outlier_idxs else (255, 0, 0)
        for x, y in pts:
            cv2.circle(vis, (x, y), 2, color_dot, -1)
        
        frame_tiles.append(vis)
    
    return concat_frames_grid(frame_tiles, (2, 4))


def visualize_world_to_cam_tracking(
    loader, image_dir, mask_data,
    ob_in_world_poses, orig_vertices,
    output_path, num_frames, object_idx, outlier_idxs=[]
):
    """
    可视化世界坐标系下的跟踪结果（所有相机视角的2x4 grid）
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    serials = loader.rs_serials
    Ks_list = loader.rs_Ks
    extrinsics_list = [loader.extr2world_inv[i] for i in range(len(serials))]
    W, H = loader.rs_width, loader.rs_height
    
    video_out = cv2.VideoWriter(
        str(output_path / "world_to_cam_tracking_2x4.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (W * 4, H * 2)
    )
    
    # 使用多进程处理
    num_workers = min(8, os.cpu_count() or 8)
    chunksize = max(1, num_frames // (num_workers * 4))
    
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=num_workers)
    
    args_list = [
        (i, image_dir, mask_data, serials, Ks_list, extrinsics_list,
         ob_in_world_poses[i] if i < len(ob_in_world_poses) else None,
         orig_vertices, object_idx, W, H, outlier_idxs)
        for i in range(num_frames)
    ]
    
    for frame in tqdm(pool.imap(process_frame_world_to_cam, args_list, chunksize=chunksize), 
                      total=num_frames, desc="World to cam tracking"):
        if frame is not None:
            video_out.write(frame)
    
    pool.close()
    pool.join()
    video_out.release()
    
    print(f"[INFO] World to cam tracking video saved to {output_path / 'world_to_cam_tracking_2x4.mp4'}")


def concat_videos_grid(video_paths, output_path, grid_shape=(2, 4)):
    """
    将多个视频拼接成grid大视频
    """
    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    if not all([c.isOpened() for c in caps]):
        print(f"[WARNING] Some videos could not be opened. Skipping grid concatenation.")
        return
    
    widths = [int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) for c in caps]
    heights = [int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
    fpss = [c.get(cv2.CAP_PROP_FPS) for c in caps]
    frame_counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    min_frames = min(frame_counts)
    w, h = widths[0], heights[0]
    fps = fpss[0]
    grid_h, grid_w = grid_shape
    out_h, out_w = h * grid_h, w * grid_w
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    
    for _ in tqdm(range(min_frames), desc="Concatenating videos"):
        frames = []
        for c in caps:
            ret, frame = c.read()
            if not ret:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            frames.append(frame)
        # 拼接
        rows = []
        for i in range(grid_h):
            row = np.concatenate(frames[i*grid_w:(i+1)*grid_w], axis=1)
            rows.append(row)
        grid = np.concatenate(rows, axis=0)
        out.write(grid)
    
    for c in caps:
        c.release()
    out.release()
    print(f"[INFO] Grid video saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, 
                       help="数据路径，如 videos_1121/fork_slice_dough/20251121_bigwoodenfork_slice_dough_half_1")
    parser.add_argument("--object_idx", type=int, default=1, help="物体索引，默认为1")
    parser.add_argument("--pose_type", type=str, default="fd", 
                       choices=["fd", "optimized"], 
                       help="选择foundation pose或optimized pose作为优化后的pose")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="输出目录，默认为debug_output/tracking_result_{data_path}")
    parser.add_argument("--concat_grid", action="store_true", 
                       help="是否生成所有相机的2x4 grid视频")
    parser.add_argument("--outlier_idxs", type=str, default="", 
                       help="异常帧索引，用逗号分隔，如 '10,20,30'")
    parser.add_argument("--visualize_world", action="store_true",
                       help="是否可视化世界坐标系下的跟踪结果（2x4 grid）")
    args = parser.parse_args()

    # 初始化loader
    sequence_folder = args.data_path
    loader = MyClusterLoader(sequence_folder)

    serials = loader.rs_serials
    Ks_list = loader.rs_Ks
    W, H = loader.rs_width, loader.rs_height
    num_frames = loader.num_frames
    num_cams = len(serials)

    # 图像缓存目录
    image_cache_dir = loader._data_folder / "image_cache"
    if not (image_cache_dir / ".extracted").exists():
        print(f"[ERROR] Image cache not found. Please run visualize_cluster_video_fast.py with --extract_images first.")
        exit(1)

    # 检查data loader的配置
    print("[INFO] ========== Checking Data Loader Configuration ==========")
    print(f"[INFO] Data folder: {loader._data_folder}")
    print(f"[INFO] Annotated folder: {loader._annotated_folder}")
    print(f"[INFO] Seg folder (mask source): {loader._seg_folder}")
    print(f"[INFO] Seg folder exists: {loader._seg_folder.exists()}")
    print(f"[INFO] Object masks folder: {loader._object_masks_folder}")
    print(f"[INFO] Object masks folder exists: {loader._object_masks_folder.exists()}")
    
    # 检查mask文件
    print("[INFO] Checking mask files...")
    masks_h5_path = loader._seg_folder / "masks.h5"
    print(f"[INFO] Masks h5 path: {masks_h5_path}")
    print(f"[INFO] Masks h5 exists: {masks_h5_path.exists()}")
    
    if masks_h5_path.exists():
        with h5py.File(masks_h5_path, 'r') as f:
            print(f"[INFO] Masks h5 datasets: {list(f.keys())}")
            if 'masks' in f:
                masks_ds = f['masks']
                print(f"[INFO] Masks dataset shape: {masks_ds.shape}")
                print(f"[INFO] Masks dataset dtype: {masks_ds.dtype}")
                # 检查前几个值
                sample_masks = masks_ds[:min(3, num_frames), :min(2, num_cams)]
                print(f"[INFO] Sample masks shape: {sample_masks.shape}")
                print(f"[INFO] Sample masks min/max: {sample_masks.min()}/{sample_masks.max()}")
                print(f"[INFO] Sample masks unique: {np.unique(sample_masks)}")
    else:
        # 检查npy文件
        print("[INFO] Checking npy mask files...")
        for cam_idx in range(min(2, num_cams)):
            cam_folder = loader._seg_folder / f"cam{cam_idx}_rgb"
            print(f"[INFO] Cam {cam_idx} folder: {cam_folder}, exists: {cam_folder.exists()}")
            if cam_folder.exists():
                npy_files = list(cam_folder.glob("*.npy"))
                print(f"[INFO] Cam {cam_idx} npy files count: {len(npy_files)}")
                if len(npy_files) > 0:
                    sample_npy = cam_folder / "0000.npy"
                    if sample_npy.exists():
                        sample_mask = np.load(sample_npy)
                        print(f"[INFO] Cam {cam_idx} sample mask shape: {sample_mask.shape}, dtype: {sample_mask.dtype}")
                        print(f"[INFO] Cam {cam_idx} sample mask min/max: {sample_mask.min()}/{sample_mask.max()}")
                        print(f"[INFO] Cam {cam_idx} sample mask unique: {np.unique(sample_mask)}")
    
    print("[INFO] ========================================================")
    
    # 加载mask数据
    print("[INFO] Loading mask data into memory...")
    print(f"[INFO] object_idx={args.object_idx}, object_id={loader.object_ids[args.object_idx - 1] if args.object_idx > 0 else 'N/A'}")
    
    # 首先检查原始mask数据
    try:
        raw_mask_data = loader._all_masks
        print(f"[INFO] Loaded raw mask data from loader._all_masks")
        print(f"  - Raw shape: {raw_mask_data.shape}")
        print(f"  - Raw dtype: {raw_mask_data.dtype}")
        print(f"  - Raw min/max: {raw_mask_data.min()}/{raw_mask_data.max()}")
        print(f"  - Raw unique values: {np.unique(raw_mask_data)}")
        
        # 检查原始mask的前几帧
        print("[INFO] Checking raw mask data for first few frames...")
        for frame_idx in range(min(3, num_frames)):
            for cam_idx in range(min(2, num_cams)):
                raw_mask = raw_mask_data[frame_idx, cam_idx]
                if raw_mask.ndim == 3:
                    raw_mask = raw_mask[0]
                print(f"  - Frame {frame_idx:03d}, Cam {cam_idx:02d}: sum={raw_mask.sum()}, max={raw_mask.max()}, unique={np.unique(raw_mask)}")
        
        # 使用get_mask方法获取二值mask（正确的方法）
        print("[INFO] Converting raw masks to binary masks using get_mask method...")
        mask_data = np.zeros((num_frames, num_cams, H, W), dtype=np.uint8)
        for frame_idx in tqdm(range(num_frames), desc="Loading binary masks"):
            for cam_idx, serial in enumerate(serials):
                binary_mask = loader.get_mask(serial, frame_idx, args.object_idx - 1)
                if binary_mask.ndim == 3:
                    binary_mask = binary_mask[0]
                mask_data[frame_idx, cam_idx] = binary_mask
        
    except AttributeError:
        print("[INFO] loader._all_masks not available, loading masks via get_mask method...")
        mask_data = np.zeros((num_frames, num_cams, H, W), dtype=np.uint8)
        for frame_idx in tqdm(range(num_frames), desc="Loading masks"):
            for cam_idx, serial in enumerate(serials):
                binary_mask = loader.get_mask(serial, frame_idx, args.object_idx - 1)
                if binary_mask.ndim == 3:
                    binary_mask = binary_mask[0]
                mask_data[frame_idx, cam_idx] = binary_mask
    
    # 检查处理后的mask数据
    print("[INFO] Checking processed mask data...")
    print(f"  - Shape: {mask_data.shape}")
    print(f"  - Dtype: {mask_data.dtype}")
    print(f"  - Min value: {mask_data.min()}")
    print(f"  - Max value: {mask_data.max()}")
    print(f"  - Unique values: {np.unique(mask_data)}")
    
    # 检查每个相机和帧的mask统计信息
    print("[INFO] Checking processed mask statistics for first few frames...")
    for frame_idx in range(min(5, num_frames)):
        for cam_idx in range(min(3, num_cams)):
            mask = mask_data[frame_idx, cam_idx]
            if mask.ndim == 3:
                mask = mask[0]
            mask_sum = mask.sum()
            mask_max = mask.max()
            unique_vals = np.unique(mask)
            print(f"  - Frame {frame_idx:03d}, Cam {cam_idx:02d}: sum={mask_sum}, max={mask_max}, unique={unique_vals}")
    
    # 检查object_idx对应的mask值
    expected_mask_value = args.object_idx  # 如果mask存储的是object_idx
    expected_mask_value_alt = args.object_idx + 1  # 如果mask存储的是object_idx+1
    print(f"[INFO] Expected mask values for object_idx={args.object_idx}: {expected_mask_value} or {expected_mask_value_alt}")
    
    # 检查是否有有效的mask数据
    valid_mask_count = 0
    for frame_idx in range(num_frames):
        for cam_idx in range(num_cams):
            mask = mask_data[frame_idx, cam_idx]
            if mask.ndim == 3:
                mask = mask[0]
            if mask.sum() > 0:
                valid_mask_count += 1
    print(f"[INFO] Found {valid_mask_count}/{num_frames * num_cams} frames with non-zero masks")
    
    if valid_mask_count == 0:
        print("[WARNING] No valid masks found! This might indicate:")
        print("  1. Mask files are missing or empty")
        print("  2. object_idx is incorrect")
        print("  3. Mask data format is different than expected")

    # 加载mesh
    object_id = loader.object_ids[args.object_idx - 1]
    orig_mesh = trimesh.load(str(loader.object_cleaned_files[args.object_idx - 1]), process=False)
    orig_vertices = orig_mesh.vertices.copy()
    # 注意：mesh顶点可能需要缩放（根据实际情况调整）
    # orig_vertices *= 0.001  # 如果需要从mm转换为m

    # 获取路径
    task_name = loader._data_folder.parent.name
    sequence_name = loader._data_folder.name
    annotated_base = loader._data_folder.parent.parent.parent / f"{loader._folder_name}_annotated" / task_name / loader._sequence_name

    # 加载优化前的pose（从txt文件）
    print("[INFO] Loading pre-optimization poses from txt files...")
    pre_optim_pose_folder = annotated_base / "processed/fd_pose_solver" / object_id / "ob_in_cam"
    pre_optim_poses_all_cams = []
    for serial_idx, serial in enumerate(serials):
        pose_folder = pre_optim_pose_folder / serial
        poses = load_all_poses_from_txt(pose_folder, num_frames)
        poses = linear_interpolate_poses(poses, num_frames)
        pre_optim_poses_all_cams.append(poses)
        print(f"[INFO] Loaded {sum(1 for p in poses if p is not None)}/{num_frames} poses for camera {serial}")

    # 加载优化后的pose（从npy文件）
    print("[INFO] Loading post-optimization poses from npy file...")
    if args.pose_type == "fd":
        # 使用fd_poses_merged_fixed.npy（已经是优化后的）
        post_optim_pose_file = annotated_base / "processed/fd_pose_solver/fd_poses_merged_fixed.npy"
    elif args.pose_type == "optimized":
        # 使用object_pose_solver的优化结果
        post_optim_pose_file = annotated_base / "processed/object_pose_solver/poses_o.npy"
    else:
        post_optim_pose_file = annotated_base / "processed/joint_pose_solver/poses_o.npy"
    
    if not post_optim_pose_file.exists():
        print(f"[WARNING] Post-optimization pose file not found: {post_optim_pose_file}")
        print("[INFO] Using pre-optimization poses as post-optimization poses.")
        post_optim_poses_all_cams = pre_optim_poses_all_cams
    else:
        post_optim_pose_data = np.load(post_optim_pose_file)
        if post_optim_pose_data.ndim == 3:
            post_optim_pose_data = post_optim_pose_data[args.object_idx - 1]
        post_optim_pose_data = post_optim_pose_data.reshape(-1, 7)
        
        # 将世界坐标系的pose转换到每个相机坐标系
        post_optim_poses_all_cams = []
        for serial_idx, serial in enumerate(serials):
            world2cam = loader.extr2world_inv[serial_idx]
            poses = []
            for i in range(num_frames):
                if i < len(post_optim_pose_data):
                    pose_world = load_pose_from_npy(post_optim_pose_data, i, 0)
                    pose_cam = world2cam @ pose_world
                    poses.append(pose_cam)
                else:
                    poses.append(None)
            post_optim_poses_all_cams.append(poses)
        print(f"[INFO] Loaded post-optimization poses from {post_optim_pose_file}")

    # 解析异常帧索引
    outlier_idxs = []
    if args.outlier_idxs:
        outlier_idxs = [int(x.strip()) for x in args.outlier_idxs.split(',') if x.strip()]

    # 加载世界坐标系下的pose（如果启用）
    ob_in_world_poses = None
    if args.visualize_world:
        print("[INFO] Loading poses in world coordinate system...")
        ob_in_world_pose_folder = annotated_base / "processed/fd_pose_solver" / object_id / "ob_in_world"
        if ob_in_world_pose_folder.exists():
            ob_in_world_poses = []
            for i in range(num_frames):
                pose_path = ob_in_world_pose_folder / f"{i:06d}.txt"
                if pose_path.exists():
                    try:
                        pose = load_pose_from_txt(pose_path)
                        ob_in_world_poses.append(pose)
                    except Exception as e:
                        if i < 5:
                            print(f"[WARNING] Failed to load pose from {pose_path}: {e}")
                        ob_in_world_poses.append(None)
                else:
                    ob_in_world_poses.append(None)
            valid_poses = sum(1 for p in ob_in_world_poses if p is not None)
            print(f"[INFO] Loaded {valid_poses}/{num_frames} poses from world coordinate system")
        else:
            print(f"[WARNING] World pose folder not found: {ob_in_world_pose_folder}")
            print("[INFO] Skipping world coordinate visualization.")
            args.visualize_world = False

    # 设置输出目录
    if args.output_dir is None:
        output_base = Path("debug_output") / "tracking_result" / args.data_path.replace('/', '_')
    else:
        output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # 为每个相机生成视频
    video_paths = []
    for serial_idx, serial in enumerate(serials):
        cam_output_dir = output_base / f"cam{serial_idx:02d}"
        visualize_tracking_for_camera(
            loader=loader,
            serial_idx=serial_idx,
            serial=serial,
            image_dir=image_cache_dir,
            mask_data=mask_data,
            pre_optim_poses=pre_optim_poses_all_cams[serial_idx],
            post_optim_poses=post_optim_poses_all_cams[serial_idx],
            orig_vertices=orig_vertices,
            K=Ks_list[serial_idx],
            output_path=cam_output_dir,
            num_frames=num_frames,
            object_idx=args.object_idx,
            outlier_idxs=outlier_idxs
        )
        video_paths.append(cam_output_dir / f"tracking_result_cam{serial_idx:02d}.mp4")

    # 可选：生成2x4 grid视频
    if args.concat_grid:
        grid_output_path = output_base / f"tracking_result_2x4_grid.mp4"
        concat_videos_grid(video_paths, grid_output_path, grid_shape=(2, 4))

    # 可视化世界坐标系下的跟踪结果
    if args.visualize_world and ob_in_world_poses is not None:
        world_output_dir = output_base / "world_to_cam"
        visualize_world_to_cam_tracking(
            loader=loader,
            image_dir=image_cache_dir,
            mask_data=mask_data,
            ob_in_world_poses=ob_in_world_poses,
            orig_vertices=orig_vertices,
            output_path=world_output_dir,
            num_frames=num_frames,
            object_idx=args.object_idx,
            outlier_idxs=outlier_idxs
        )

    print(f"[INFO] All tracking result videos saved to {output_base}")

