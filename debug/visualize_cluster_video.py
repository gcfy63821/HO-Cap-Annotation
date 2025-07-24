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
from hocap_annotation.layers import MANOGroupLayer
from hocap_annotation.utils.color_info import *
from hocap_annotation.utils.mano_info import *
from hocap_annotation.loaders.my_cluster_loader import MyClusterLoader



def load_pose(pose_txt):
    with open(pose_txt, 'r') as f:
        arr = np.array([float(x) for x in f.read().strip().split()])
        # print(f"[DEBUG] Loaded pose from {pose_txt}: {arr}")
        t = np.array(arr[4:7])
        q = np.array(arr[:4])  # xyzw
        R_mat = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        return T

def load_mano_sequence(mano_file):
    mano_data = np.load(mano_file).astype(np.float32)
    print(f"[INFO] Loaded MANO sequence from {mano_file}, shape: {mano_data.shape}")
    return mano_data  # 返回形状为(2, N, 51)的数组

# Remove load_mano_beta and load_extrinsics_yaml, and update init_mano_group_layer to use loader.mano_beta

def init_mano_group_layer(betas):
    mano_group_layer = MANOGroupLayer(['left','right'], [betas] * 2).to('cuda')
    return mano_group_layer

# 1. 加载MANO姿态数据并传递给层
def mano_group_layer_forward(poses_m, layer, subset=None):
    p = torch.cat(poses_m, dim=1)
    v, j = layer(p, subset)
    if v.size(0) == 1:
        v = v.squeeze(0)
        j = j.squeeze(0)
    return v, j

def load_poses_m(pose_file):
    poses = np.load(pose_file).astype(np.float32)
    mano_sides = ['left','right']
    poses = np.stack(
        [poses[0 if side == "right" else 1] for side in mano_sides], axis=0
    )  # (num_hands, num_frames 51)
    return poses

def load_mano_data(mano_file, layer):
    poses_m = load_poses_m(mano_file)
    
    poses_m = [torch.from_numpy(p).to('cuda') for p in poses_m]
    verts_m, joints_m = mano_group_layer_forward(poses_m, layer)  # 获取verts_m和joints_m
    verts_m = verts_m.detach().clone().cpu().numpy()
    joints_m = joints_m.detach().clone().cpu().numpy()
    return verts_m, joints_m

def project_points(vertices, K):
    pts = vertices @ K[:3, :3].T
    pts = pts[:, :2] / pts[:, 2:]
    return pts


def render_hand_mesh(hand_mesh, K, W, H):
    """
    使用Trimesh渲染手的网格并将其投影到2D图像平面
    """
    # 投影到2D图像
    hand_pts_2d = project_points(hand_mesh.vertices, K)

    # 创建空白图像，用于显示手部网格
    hand_img = np.ones((H, W, 3), dtype=np.uint8) * 255  # 白色背景

    # 绘制手部网格的面
    for face in hand_mesh.faces:
        pts_2d = hand_pts_2d[face]
        pts_2d = pts_2d.astype(np.int32)
        if pts_2d.shape[0] == 3:
            cv2.polylines(hand_img, [pts_2d], isClosed=True, color=(0, 255, 0), thickness=1)

    return hand_img

# Restore concat_frames_grid function

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

def process_frame_pose_npy_h5(args):
    i, pose_data, verts_m, faces_m, outlier_idxs, orig_vertices, orig_mesh, dataloader, object_idx = args
    W, H = 640, 480
    frame_tiles = []

    if i >= len(pose_data):
        frame_tiles = [np.ones((H, W, 3), dtype=np.uint8) * 255 for _ in dataloader.rs_serials]
        return concat_frames_grid(frame_tiles, (2, 4))

    # 读取物体位姿
    qx, qy, qz, qw, tx, ty, tz = pose_data[i]
    q = np.array([qx, qy, qz, qw])
    t = np.array([tx, ty, tz])
    R_mat = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t

    # 获取当前帧的MANO手部数据
    mano_verts = verts_m[i, :, :]
    
    colors_m = [(0.0, 1.0, 1.0), (0.9803921568627451, 0.2901960784313726, 0.16862745098039217)]
    extrinsics_dict = {s: dataloader.extr2world_inv[i] for i, s in enumerate(dataloader.rs_serials)}

    for serial_idx, serial in enumerate(dataloader.rs_serials):
        color = dataloader.get_color(serial, i)
        sam_mask = dataloader.get_mask(serial, i, object_idx - 1)
        if color is None or sam_mask is None:
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        sam_overlay = color.copy()
        sam_overlay[sam_mask > 0] = [0, 0, 255]

        # 可视化物体
        mesh = orig_mesh.copy()
        mesh.vertices = orig_vertices.copy()
        mesh.apply_transform(T)
        world2cam = extrinsics_dict[serial]
        mesh.apply_transform(world2cam)
        K = dataloader.rs_Ks[serial_idx]
        pts = project_points(mesh.vertices, K)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]
        
        vis = sam_overlay.copy()
        color_dot = (0, 0, 255) if i in outlier_idxs else colors_m[1]
        for x, y in pts:
            cv2.circle(vis, (x, y), 2, color_dot, -1)
        # slow
        # for face in mesh.faces:
        #     pts_2d = pts[face]
        #     pts_2d = pts_2d.astype(np.int32)
        #     if pts_2d.shape[0] == 3:
        #         cv2.polylines(vis, [pts_2d], isClosed=True, color=colors_m[1], thickness=1)

        # 将MANO手部位姿可视化到图像上
        mano_verts_cpu = mano_verts.cpu().numpy() if torch.is_tensor(mano_verts) else mano_verts
        faces_m_cpu = faces_m.cpu().numpy().astype(np.int32) if torch.is_tensor(faces_m) else faces_m.astype(np.int32)
        # 创建Trimesh对象
        hand_mesh = trimesh.Trimesh(vertices=mano_verts_cpu, faces=faces_m_cpu, process=False)
        hand_mesh.apply_transform(world2cam)
        hand_img = render_hand_mesh(hand_mesh, K, W, H)

        # 合并结果
        vis = cv2.addWeighted(vis, 0.6, hand_img, 0.4, 0)

        frame_tiles.append(vis)
        # vis = visualize_mano_hand(mano_verts, faces_m, colors_m, serial_idx, i, outlier_idxs, dataloader)
        


    return concat_frames_grid(frame_tiles, (2, 4))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_1/20250701_012148", help="数据路径，如 test_1/20250701_012148")
    parser.add_argument("--tool_name", type=str, default="blue_scooper", help="工具名，如 blue_scooper")
    parser.add_argument("--output_idx", type=str, default="0", help="输出编号")
    parser.add_argument("--pose_file", type=str, default="fd", choices=["fd", "adaptive", "optimized"], help="选择foundation pose 或 optimized")
    parser.add_argument("--uuid", type=str, default="", help="唯一标识符，用于区分不同运行")
    parser.add_argument("--object_idx", type=int, default=1, help="物体索引，默认为0")
    args = parser.parse_args()

    # Use MyClusterLoader instead of IMGLoader
    sequence_folder = f"/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/{args.data_path}"
    loader = MyClusterLoader(sequence_folder)

    serials = loader.rs_serials
    K = loader.rs_Ks[0]  # Assume all cameras have the same intrinsics for now
    Ks = {s: loader.rs_Ks[i] for i, s in enumerate(serials)}
    W, H = loader.rs_width, loader.rs_height
    num_frames = loader.num_frames

    # Get color and mask access functions
    def get_color_img(frame_idx, serial_idx):
        serial = serials[serial_idx]
        return loader.get_color(serial, frame_idx)

    def get_mask_img(frame_idx, serial_idx):
        serial = serials[serial_idx]
        return loader.get_mask(serial, frame_idx, args.object_idx - 1)

    # Load mesh
    orig_mesh = trimesh.load(str(loader.object_cleaned_files[args.object_idx - 1]), process=False)
    orig_vertices = orig_mesh.vertices.copy()

    # pose_npy_in_cams
    # Use the same structure as MyClusterLoader for annotated data
    annotated_base = loader._data_folder.parent.parent / f"{loader._folder_name}_annotated" / loader._sequence_name
    if args.pose_file == "fd":
        pose_npy_path = str(annotated_base / "processed/fd_pose_solver/fd_poses_merged_fixed.npy")
    elif args.pose_file == "adaptive":
        pose_npy_path = str(annotated_base / "processed/fd_pose_solver/adaptive_fd_poses_merged_fixed.npy")
    elif args.pose_file == "optimized":
        pose_npy_path = str(annotated_base / "processed/object_pose_solver/poses_o.npy")
    output_path2 = Path(f"debug_output/{args.data_path}/pose_npy_in_cams_video")
    output_path2.mkdir(parents=True, exist_ok=True)
    video_out2 = cv2.VideoWriter(
        str(output_path2 / f"{args.output_idx}{'_' + args.uuid if args.uuid else ''}_{args.pose_file}_pose_npy_in_cams_2x4.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (W * 4, H * 2)
    )

    pose_data = np.load(pose_npy_path)
    print(f"[INFO] Loaded pose data from {pose_npy_path}, shape: {pose_data.shape}")
    if pose_data.ndim == 3:
        pose_data = pose_data[args.object_idx - 1]
        print(f"[INFO] Using pose_data[{args.object_idx - 1}], shape: {pose_data.shape}")
    pose_data = pose_data.reshape(-1, 7)

    # MANO hand sequence
    mano_file = str(annotated_base / "processed/joint_pose_solver/poses_m.npy")
    mano_layer = init_mano_group_layer(loader.mano_beta)
    verts_m, joints_m = load_mano_data(mano_file, mano_layer)
    subset_m = list(range(2))
    faces_m, _ = mano_layer.get_f_from_inds(subset_m)
    faces_m = [faces_m.detach().clone().cpu().numpy()]
    mano_sides = ['left','right']
    colors_m = []
    for i, side in enumerate(mano_sides):
        faces_m.append(np.array(NEW_MANO_FACES[side]) + i * NUM_MANO_VERTS)
        colors_m.append(HAND_COLORS[1 if side == "right" else 2].rgb_norm)
    faces_m = np.concatenate(faces_m, axis=0).astype(np.int64)

    print("verts_m:", verts_m.shape)
    print("joints_m:", joints_m.shape)
    print("faces_m:", faces_m.shape)
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=min(8, os.cpu_count() or 8))
    args_list2 = [
        (i, pose_data, verts_m, faces_m, [], orig_vertices, orig_mesh, loader, args.object_idx)
        for i in range(num_frames)
    ]
    for frame in tqdm(pool.imap(process_frame_pose_npy_h5, args_list2), total=num_frames):
        video_out2.write(frame)
    pool.close()
    pool.join()
    video_out2.release()
    print(f"[INFO] pose_npy_in_cams_2x4.mp4 saved to {output_path2}{'_' + args.uuid if args.uuid else ''}")
