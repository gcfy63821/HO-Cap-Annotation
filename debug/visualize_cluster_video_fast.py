import os
import cv2
import numpy as np
from pathlib import Path
import trimesh
import yaml
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import multiprocessing
import h5py
import torch
from hocap_annotation.loaders.my_cluster_loader import MyClusterLoader
import pickle
from hocap_annotation.utils import CFG
from manopth.manolayer import ManoLayer
import math


def extract_images_from_h5(h5_path, output_dir, num_frames, num_cams):
    """
    从 h5 文件中提取所有图像并保存为图片文件，避免后续的 GPU/CPU 复制。
    
    Args:
        h5_path: h5 文件路径
        output_dir: 输出目录，将创建 color_XX 子目录
        num_frames: 帧数
        num_cams: 相机数量
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


def project_points(vertices, K):
    """投影 3D 点到 2D 图像平面"""
    pts = vertices @ K[:3, :3].T
    pts = pts[:, :2] / pts[:, 2:]
    return pts


def render_hand_mesh_fast(hand_vertices, hand_faces, K, W, H):
    """
    快速渲染手部网格（使用预变换的顶点）
    """
    if hand_vertices is None or len(hand_vertices) == 0:
        return np.ones((H, W, 3), dtype=np.uint8) * 255
    
    # 投影到2D图像
    hand_pts_2d = project_points(hand_vertices, K)
    
    # 过滤在图像范围内的点
    valid_mask = (hand_pts_2d[:, 0] >= 0) & (hand_pts_2d[:, 0] < W) & \
                 (hand_pts_2d[:, 1] >= 0) & (hand_pts_2d[:, 1] < H)
    
    if not valid_mask.any():
        return np.ones((H, W, 3), dtype=np.uint8) * 255

    # 创建空白图像
    hand_img = np.ones((H, W, 3), dtype=np.uint8) * 255

    # 只处理有效的面
    valid_faces = valid_mask[hand_faces].all(axis=1)
    valid_faces_idx = np.where(valid_faces)[0]
    
    if len(valid_faces_idx) > 0:
        valid_faces_pts = hand_pts_2d[hand_faces[valid_faces_idx]].astype(np.int32)
        for pts_2d in valid_faces_pts:
            if pts_2d.shape[0] == 3:
                cv2.polylines(hand_img, [pts_2d], isClosed=True, color=(0, 255, 0), thickness=1)

    return hand_img


def concat_frames_grid(frames, grid_shape=None):
    """将帧拼成 grid，自动计算布局"""
    n = len(frames)
    if grid_shape is None:
        # 自动计算最接近正方形的布局
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        grid_shape = (rows, cols)
    
    rows, cols = grid_shape
    h, w = frames[0].shape[:2]
    
    # 如果帧数不足，用白色填充
    while len(frames) < rows * cols:
        frames.append(np.ones((h, w, 3), dtype=np.uint8) * 255)
    
    row_imgs = []
    for i in range(rows):
        row = np.concatenate(frames[i*cols:(i+1)*cols], axis=1)
        row_imgs.append(row)
    return np.concatenate(row_imgs, axis=0)


def load_pkl_and_get_hand_data(pkl_file):
    """从 pkl 文件加载手部数据"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    if 'hand_pose' not in data:
        raise ValueError("No 'hand_pose' found in the .pkl file.")
    hand_pose = data['hand_pose']
    return {
        'left_hand_pose': np.array(hand_pose.get('left_hand_pose', [])),
        'left_hand_beta': np.array(hand_pose.get('left_hand_beta', [])),
        'left_hand_translation': np.array(hand_pose.get('left_hand_translation', [])),
        'left_hand_base_rot': np.array(hand_pose.get('left_hand_base_rot', [])),
        'right_hand_pose': np.array(hand_pose.get('right_hand_pose', [])),
        'right_hand_beta': np.array(hand_pose.get('right_hand_beta', [])),
        'right_hand_translation': np.array(hand_pose.get('right_hand_translation', [])),
        'right_hand_base_rot': np.array(hand_pose.get('right_hand_base_rot', [])),
    }


def get_betas(b):
    """提取 betas"""
    b = np.array(b)
    if b.ndim == 2 and b.shape[0] == 1:
        return b[0]
    return b.squeeze()


def init_mano_layers(hand_data):
    """初始化 MANO 层"""
    mano_betas_left = get_betas(hand_data['left_hand_beta'])
    mano_betas_right = get_betas(hand_data['right_hand_beta'])
    mano_layer_right = ManoLayer(side="right",
                                mano_root=CFG.mano.model_path, 
                                use_pca=False, 
                                ncomps=45).to('cuda')
    mano_layer_left = ManoLayer(side="left",
                                mano_root=CFG.mano.model_path, 
                                use_pca=False, 
                                ncomps=45).to('cuda')
    return mano_layer_left, mano_layer_right


def reconstruct_hand_mesh_fast(hand_data, frame_idx, mano_layer, pose, side='left'):
    """
    快速重建手部网格（优化版本，减少内存拷贝）
    """
    try:
        pose_tensor = torch.tensor(pose, dtype=torch.float32).to('cuda').unsqueeze(0)
        
        if side == 'left':
            translation = torch.tensor(hand_data['left_hand_translation'][frame_idx], dtype=torch.float32).to('cuda').unsqueeze(0)
            hand_beta = torch.tensor(hand_data['left_hand_beta'], dtype=torch.float32).to('cuda')
            if hand_data['left_hand_base_rot'].ndim == 3:
                base_rot = torch.tensor(hand_data['left_hand_base_rot'][frame_idx], dtype=torch.float32).to('cuda')
            else:
                base_rot = torch.eye(3, dtype=torch.float32).to('cuda')
        else:
            translation = torch.tensor(hand_data['right_hand_translation'][frame_idx], dtype=torch.float32).to('cuda').unsqueeze(0)
            hand_beta = torch.tensor(hand_data['right_hand_beta'], dtype=torch.float32).to('cuda')
            base_rot = torch.eye(3, dtype=torch.float32).to('cuda')
        
        verts, joints = mano_layer(pose_tensor, hand_beta.float())
        verts = verts[0] / 1000.0
        joints = joints[0] / 1000.0
        
        if verts.size(0) == 1:
            verts = verts.squeeze(0)
            joints = joints.squeeze(0)
        
        root_trans = joints[0].clone()
        verts = verts - root_trans
        
        if side == 'left':
            verts[:, 0] *= -1
            verts = verts @ base_rot.T
        
        verts = verts + translation
        
        # 直接返回顶点，不创建完整的 mesh 对象
        return verts.detach().cpu().numpy()
    except Exception as e:
        if frame_idx < 5:
            print(f"Error in reconstruct_hand_mesh_fast for frame {frame_idx}, side {side}: {e}")
        return None


def process_frame_fast(args):
    """
    快速处理单帧（优化版本）
    """
    (i, pose_data, orig_vertices, orig_mesh_faces, 
     image_dir, mask_data, serials, Ks_list, extrinsics_list,
     object_idx, render_hands, hand_data, mano_layer_left, mano_layer_right, poses_m,
     outlier_idxs, colors_m, W, H, grid_shape) = args
    
    frame_tiles = []
    
    if i >= len(pose_data):
        frame_tiles = [np.ones((H, W, 3), dtype=np.uint8) * 255 for _ in serials]
        return concat_frames_grid(frame_tiles, grid_shape)

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
    
    # 预计算手部顶点（如果启用）
    left_hand_vertices_world = None
    right_hand_vertices_world = None
    left_hand_faces = None
    right_hand_faces = None
    
    if render_hands and hand_data is not None and mano_layer_left is not None and mano_layer_right is not None:
        left_pose = poses_m[0][i]
        right_pose = poses_m[1][i]
        
        try:
            left_hand_vertices_world = reconstruct_hand_mesh_fast(hand_data, i, mano_layer_left, left_pose, 'left')
            right_hand_vertices_world = reconstruct_hand_mesh_fast(hand_data, i, mano_layer_right, right_pose, 'right')
            
            # 获取手部 faces（只需要获取一次，可以预计算）
            if left_hand_vertices_world is not None:
                left_hand_faces = mano_layer_left.th_faces.detach().cpu().numpy()
            if right_hand_vertices_world is not None:
                right_hand_faces = mano_layer_right.th_faces.detach().cpu().numpy()
        except Exception as e:
            if i < 5:
                print(f"Error reconstructing hand meshes for frame {i}: {e}")

    # 处理每个相机视图
    for serial_idx, serial in enumerate(serials):
        # 直接从图片文件加载（避免 GPU/CPU 复制）
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
        # mask_data 已经是处理过的，但我们需要确保格式正确
        if sam_mask.ndim == 3:
            sam_mask = sam_mask[0]
        # 如果 mask_data 存储的是原始 mask（包含 object_idx+1），需要转换
        # 否则如果已经是二值 mask，直接使用
        if sam_mask.max() > 1:
            sam_mask = (sam_mask == (object_idx + 1)).astype(np.uint8)
        else:
            sam_mask = sam_mask.astype(np.uint8)
        
        # 创建 overlay（使用视图而不是拷贝）
        sam_overlay = color.copy()  # 这个拷贝是必要的，因为我们要修改
        sam_overlay[sam_mask > 0] = [0, 0, 255]

        # 可视化物体 - 使用预计算的顶点
        world2cam = extrinsics_list[serial_idx]
        obj_vertices_cam = (world2cam[:3, :3] @ obj_vertices_world.T).T + world2cam[:3, 3]

        K = Ks_list[serial_idx]
        pts = project_points(obj_vertices_cam, K)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]  # 降采样以减少绘制点数
        
        vis = sam_overlay.copy()
        color_dot = (0, 0, 255) if i in outlier_idxs else colors_m[1]
        for x, y in pts:
            cv2.circle(vis, (x, y), 2, color_dot, -1)

        # 手部mesh渲染
        if render_hands:
            if left_hand_vertices_world is not None and left_hand_faces is not None:
                try:
                    left_hand_vertices_cam = (world2cam[:3, :3] @ left_hand_vertices_world.T).T + world2cam[:3, 3]
                    left_hand_img = render_hand_mesh_fast(left_hand_vertices_cam, left_hand_faces, K, W, H)
                    vis = cv2.addWeighted(vis, 0.6, left_hand_img, 0.4, 0)
                except Exception as e:
                    if i < 5:
                        print(f"Error processing left hand mesh for frame {i}, serial {serial}: {e}")
            
            if right_hand_vertices_world is not None and right_hand_faces is not None:
                try:
                    right_hand_vertices_cam = (world2cam[:3, :3] @ right_hand_vertices_world.T).T + world2cam[:3, 3]
                    right_hand_img = render_hand_mesh_fast(right_hand_vertices_cam, right_hand_faces, K, W, H)
                    vis = cv2.addWeighted(vis, 0.6, right_hand_img, 0.4, 0)
                except Exception as e:
                    if i < 5:
                        print(f"Error processing right hand mesh for frame {i}, serial {serial}: {e}")

        frame_tiles.append(vis)

    return concat_frames_grid(frame_tiles, grid_shape)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="数据路径，如 videos_1121/fork_slice_dough/20251121_bigwoodenfork_slice_dough_half_1")
    parser.add_argument("--tool_name", type=str, default="blue_scooper", help="工具名，如 blue_scooper")
    parser.add_argument("--output_idx", type=str, default="0", help="输出编号")
    parser.add_argument("--pose_file", type=str, default="fd", choices=["fd", "adaptive", "optimized"], help="选择foundation pose 或 optimized")
    parser.add_argument("--uuid", type=str, default="", help="唯一标识符，用于区分不同运行")
    parser.add_argument("--object_idx", type=int, default=1, help="物体索引，默认为1")
    parser.add_argument("--render_hands", action="store_true", help="是否渲染手部网格")
    parser.add_argument("--extract_images", action="store_true", help="是否提取 h5 图像为图片文件")
    parser.add_argument("--num_workers", type=int, default=None, help="并行处理的worker数量（默认自动计算，建议2-4以避免内存溢出）")
    args = parser.parse_args()

    # 初始化 loader
    sequence_folder = args.data_path
    loader = MyClusterLoader(sequence_folder)

    serials = loader.rs_serials
    Ks_list = loader.rs_Ks  # 列表格式
    Ks_dict = {s: loader.rs_Ks[i] for i, s in enumerate(serials)}
    W, H = loader.rs_width, loader.rs_height
    num_frames = loader.num_frames
    num_cams = len(serials)
    grid_cols = math.ceil(math.sqrt(num_cams))
    grid_rows = math.ceil(num_cams / grid_cols)
    grid_shape = (grid_rows, grid_cols)
    print(f"[INFO] Grid layout: {grid_rows}x{grid_cols} for {num_cams} cameras")

    # 提取图像（如果需要）
    h5_files = list(loader._data_folder.glob('*.h5'))
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No .h5 file found in {loader._data_folder}")
    h5_path = h5_files[0]
    
    # 图像缓存目录
    image_cache_dir = loader._data_folder / "image_cache"
    
    if args.extract_images or not (image_cache_dir / ".extracted").exists():
        extract_images_from_h5(h5_path, image_cache_dir, num_frames, num_cams)
    
    # 加载 mask 数据到内存（一次性加载，避免重复读取）
    print("[INFO] Loading mask data into memory...")
    # 注意：_all_masks 是私有属性，我们需要通过属性访问或使用 get_mask
    # 为了性能，我们直接访问（如果可能）或创建一个包装
    try:
        mask_data = loader._all_masks  # 直接从 loader 获取，避免重复加载
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

    # 获取路径
    task_name = loader._data_folder.parent.name
    sequence_name = loader._data_folder.name
    annotated_base = loader._data_folder.parent.parent.parent / f"{loader._folder_name}_annotated" / task_name / loader._sequence_name
    
    # 加载 pose 数据
    if args.pose_file == "fd":
        pose_npy_path = str(annotated_base / "processed/fd_pose_solver/fd_poses_merged_fixed.npy")
    elif args.pose_file == "adaptive":
        pose_npy_path = str(annotated_base / "processed/fd_pose_solver/adaptive_fd_poses_merged_fixed.npy")
    elif args.pose_file == "optimized":
        pose_npy_path = str(annotated_base / "processed/joint_pose_solver/poses_o.npy")
    
    pose_data = np.load(pose_npy_path)
    print(f"[INFO] Loaded pose data from {pose_npy_path}, shape: {pose_data.shape}")
    if pose_data.ndim == 3:
        pose_data = pose_data[args.object_idx - 1]
        print(f"[INFO] Using pose_data[{args.object_idx - 1}], shape: {pose_data.shape}")
    pose_data = pose_data.reshape(-1, 7)

    # 准备输出
    output_path2 = Path(f"debug_output/pose_npy_in_cams_video")
    output_path2.mkdir(parents=True, exist_ok=True)
    
    # 确保视频尺寸是偶数（某些编码器要求）
    video_w = W * grid_cols
    video_h = H * grid_rows
    # video_w = video_w - (video_w % 2)
    # video_h = video_h - (video_h % 2)
    
    video_path = str(output_path2 / f"{args.output_idx}{'_' + args.uuid if args.uuid else ''}_{args.pose_file}_pose_npy_in_cams_2x4.mp4")
    
    # 尝试使用更兼容的编码器
    # 优先尝试 H.264 ('avc1' 或 'H264')，这是最通用的编码器
    fourcc = None
    video_out2 = None
    
    # 尝试不同的编码器，按兼容性排序
    codecs_to_try = [
        ('H264', 'H.264 (most compatible)'),
        ('avc1', 'H.264 AVC1'),
        ('XVID', 'XVID'),
        ('mp4v', 'MPEG-4'),
    ]
    
    for codec_name, codec_desc in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            video_out2 = cv2.VideoWriter(video_path, fourcc, 20, (video_w, video_h))
            if video_out2.isOpened():
                print(f"[INFO] VideoWriter initialized with {codec_desc}: {video_path}, size: ({video_w}, {video_h}), fps: 20")
                break
        except Exception as e:
            print(f"[WARNING] Failed to initialize with {codec_name}: {e}")
            if video_out2 is not None:
                video_out2.release()
            video_out2 = None
    
    if video_out2 is None or not video_out2.isOpened():
        raise RuntimeError(f"Failed to initialize VideoWriter for {video_path}. No compatible codec found.")

    # 预计算 extrinsics（避免在每个进程中重复计算）
    extrinsics_list = [loader.extr2world_inv[i] for i in range(len(serials))]
    
    # 加载手部数据（如果启用）
    hand_data = None
    mano_layer_left = None
    mano_layer_right = None
    poses_m = None
    if args.render_hands:
        mano_file = str(annotated_base / "processed/joint_pose_solver/poses_m.npy")
        pkl_file_path = str(annotated_base / "result_hand_optimized.pkl")
        try:
            hand_data = load_pkl_and_get_hand_data(pkl_file_path)
            mano_layer_left, mano_layer_right = init_mano_layers(hand_data)
            poses_m = np.load(mano_file)
            # 重新排列为 [left, right] 顺序
            mano_sides = ['left', 'right']
            poses_m = np.stack(
                [poses_m[0 if side == "right" else 1] for side in mano_sides], axis=0
            )
            print(f"[INFO] Loaded hand data from {pkl_file_path}")
            print(f"[INFO] Loaded poses_m from {mano_file}, shape: {poses_m.shape}")
        except Exception as e:
            print(f"Warning: Failed to load hand data: {e}")
            print("Hand rendering will be disabled")
            args.render_hands = False
            hand_data = None
            mano_layer_left = None
            mano_layer_right = None
            poses_m = None

    colors_m = [(0.0, 1.0, 1.0), (0.9803921568627451, 0.2901960784313726, 0.16862745098039217)]
    outlier_idxs = []

    print(f"[INFO] Hand rendering enabled: {args.render_hands}")
    print(f"[INFO] Processing {num_frames} frames with {num_cams} cameras")
    
    # 减少进程数以避免内存溢出
    # 计算mask_data的内存占用（MB）
    mask_memory_mb = mask_data.nbytes / (1024 * 1024)
    print(f"[INFO] Mask data memory: {mask_memory_mb:.2f} MB")
    
    # 根据用户指定或自动计算worker数量
    if args.num_workers is not None:
        num_workers = max(1, min(args.num_workers, os.cpu_count() or 4))
        print(f"[INFO] Using user-specified {num_workers} worker processes")
    else:
        # 根据可用内存调整worker数量
        # 假设每个worker需要复制mask_data，限制总内存使用
        max_memory_gb = 16  # 假设系统有16GB可用内存
        max_workers_by_memory = int(max_memory_gb * 1024 / (mask_memory_mb + 500))  # 500MB其他开销
        num_workers = min(4, max_workers_by_memory, os.cpu_count() or 4)  # 最多4个worker
        num_workers = max(1, num_workers)  # 至少1个worker
        print(f"[INFO] Auto-calculated {num_workers} worker processes (reduced to avoid OOM)")
    
    chunksize = max(1, num_frames // (num_workers * 2))
    print(f"[INFO] Using {num_workers} worker processes with chunksize {chunksize}")
    
    # 如果只有1个worker，直接在主进程中处理，避免multiprocessing的内存开销
    if num_workers == 1:
        print("[INFO] Using single-process mode (no multiprocessing) to save memory")
        for i in tqdm(range(num_frames), desc="Processing frames"):
            args_tuple = (
                i, pose_data, orig_vertices, orig_mesh_faces,
                image_cache_dir, mask_data, serials, Ks_list, extrinsics_list,
                args.object_idx, args.render_hands, hand_data, mano_layer_left, mano_layer_right, poses_m,
                outlier_idxs, colors_m, W, H, grid_shape
            )
            frame = process_frame_fast(args_tuple)
            # 确保帧尺寸匹配，并转换为BGR格式（如果需要）
            if frame.shape[:2] != (video_h, video_w):
                frame = cv2.resize(frame, (video_w, video_h))
            # 确保是BGR格式（OpenCV格式）
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            video_out2.write(frame)
    else:
        # 多进程处理
        multiprocessing.set_start_method('spawn', force=True)
        pool = multiprocessing.Pool(processes=num_workers)
        
        args_list = [
            (i, pose_data, orig_vertices, orig_mesh_faces,
             image_cache_dir, mask_data, serials, Ks_list, extrinsics_list,
             args.object_idx, args.render_hands, hand_data, mano_layer_left, mano_layer_right, poses_m,
             outlier_idxs, colors_m, W, H, grid_shape)
            for i in range(num_frames)
        ]
        
        for frame in tqdm(pool.imap(process_frame_fast, args_list, chunksize=chunksize), total=num_frames):
            # 确保帧尺寸匹配，并转换为BGR格式（如果需要）
            if frame.shape[:2] != (video_h, video_w):
                frame = cv2.resize(frame, (video_w, video_h))
            # 确保是BGR格式（OpenCV格式）
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            video_out2.write(frame)
        
        pool.close()
        pool.join()
    video_out2.release()
    
    # 验证视频文件是否存在且大小合理
    video_file = Path(video_path)
    if video_file.exists():
        file_size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"[INFO] Video saved successfully: {video_path}")
        print(f"[INFO] Video file size: {file_size_mb:.2f} MB")
        if file_size_mb < 0.1:
            print(f"[WARNING] Video file is very small ({file_size_mb:.2f} MB), may be corrupted!")
        
        # 尝试使用ffmpeg重新编码为H.264以确保兼容性
        print(f"[INFO] Attempting to re-encode video with H.264 for better compatibility...")
        output_path_h264 = video_file.parent / f"{video_file.stem}_h264.mp4"
        try:
            import subprocess
            # 使用ffmpeg重新编码为H.264
            cmd = [
                'ffmpeg', '-y', '-i', str(video_file),
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'copy',  # 如果没有音频，这个会被忽略
                '-pix_fmt', 'yuv420p',  # 确保兼容性
                str(output_path_h264)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and output_path_h264.exists():
                h264_size_mb = output_path_h264.stat().st_size / (1024 * 1024)
                print(f"[SUCCESS] Re-encoded video saved: {output_path_h264}")
                print(f"[INFO] H.264 video file size: {h264_size_mb:.2f} MB")
                print(f"[INFO] Original video: {video_path}")
                print(f"[INFO] H.264 video (recommended): {output_path_h264}")
            else:
                print(f"[WARNING] ffmpeg re-encoding failed or not available. Using original video.")
                if result.stderr:
                    print(f"[DEBUG] ffmpeg error: {result.stderr[:200]}")
        except FileNotFoundError:
            print(f"[WARNING] ffmpeg not found. Install ffmpeg for better video compatibility.")
        except Exception as e:
            print(f"[WARNING] Failed to re-encode video: {e}")
    else:
        print(f"[ERROR] Video file was not created: {video_path}")

