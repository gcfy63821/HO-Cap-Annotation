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

def load_poses_m(pose_file):
    poses = np.load(pose_file).astype(np.float32)
    mano_sides = ['left','right']
    poses = np.stack(
        [poses[0 if side == "right" else 1] for side in mano_sides], axis=0
    )  # (num_hands, num_frames 51)
    return poses

def load_pkl_and_get_hand_data(pkl_file):
    # 加载 .pkl 文件
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    if 'hand_pose' not in data:
        raise ValueError("No 'hand_pose' found in the .pkl file.")
    hand_pose = data['hand_pose']
    # Extract all relevant fields, using None if not present
    left_hand_pose = np.array(hand_pose.get('left_hand_pose', []))
    left_hand_beta = np.array(hand_pose.get('left_hand_beta', []))
    left_hand_translation = np.array(hand_pose.get('left_hand_translation', []))
    left_hand_base_rot = np.array(hand_pose.get('left_hand_base_rot', []))
    right_hand_pose = np.array(hand_pose.get('right_hand_pose', []))
    right_hand_beta = np.array(hand_pose.get('right_hand_beta', []))
    right_hand_translation = np.array(hand_pose.get('right_hand_translation', []))
    # right_hand_base_rot is not always present
    right_hand_base_rot = np.array(hand_pose.get('right_hand_base_rot', []))
    return {
        'left_hand_pose': left_hand_pose,
        'left_hand_beta': left_hand_beta,
        'left_hand_translation': left_hand_translation,
        'left_hand_base_rot': left_hand_base_rot,
        'right_hand_pose': right_hand_pose,
        'right_hand_beta': right_hand_beta,
        'right_hand_translation': right_hand_translation,
        'right_hand_base_rot': right_hand_base_rot,
    }

def get_betas(b):
    b = np.array(b)
    if b.ndim == 2 and b.shape[0] == 1:
        return b[0]
    return b.squeeze()

def init_mano_layers(hand_data):
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

def reconstruct_left_hand_mesh(hand_data, frame_idx, mano_layer, left_pose, mano_layer_left):
    # Use pose from npy file, other data from pkl
    try:
        pose = torch.tensor(left_pose).to('cuda').unsqueeze(0)
        translation = torch.tensor(hand_data['left_hand_translation'][frame_idx]).to('cuda').unsqueeze(0)
        base_rot = torch.tensor(hand_data['left_hand_base_rot'][frame_idx]).to('cuda') if hand_data['left_hand_base_rot'].ndim == 3 else torch.eye(3).to('cuda')
        hand_beta = torch.tensor(hand_data['left_hand_beta']).to('cuda')
        
        verts, joints = mano_layer(pose, hand_beta.float())
        verts = verts[0] / 1000
        joints = joints[0] / 1000

        if verts.size(0) == 1:
            verts = verts.squeeze(0)
            joints = joints.squeeze(0)
        
        root_trans = joints[0].clone().detach()
        verts -= root_trans
        verts[:, 0] *= -1
        verts = verts @ base_rot.T
        verts += translation
        faces = mano_layer_left.th_faces.detach().cpu().numpy()
        
        mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces)
        return mesh
    except Exception as e:
        print(f"Error in reconstruct_left_hand_mesh for frame {frame_idx}: {e}")
        return None

def reconstruct_right_hand_mesh(hand_data, frame_idx, mano_layer_right, right_pose):
    # Use pose from npy file, other data from pkl
    try:
        pose = torch.tensor(right_pose).to('cuda').unsqueeze(0)
        translation = torch.tensor(hand_data['right_hand_translation'][frame_idx]).to('cuda').unsqueeze(0)
        hand_beta = torch.tensor(hand_data['right_hand_beta']).to('cuda')
        
        verts, joints = mano_layer_right(pose, hand_beta.float())
        
        verts = verts[0] / 1000
        joints = joints[0] / 1000
        if verts.size(0) == 1:
            verts = verts.squeeze(0)
            joints = joints.squeeze(0)
        
        root_trans = joints[0].clone().detach()
        verts -= root_trans
        verts += translation
        faces = mano_layer_right.th_faces.detach().cpu().numpy()
        
        mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces)
        return mesh
    except Exception as e:
        print(f"Error in reconstruct_right_hand_mesh for frame {frame_idx}: {e}")
        return None

def project_points(vertices, K):
    pts = vertices @ K[:3, :3].T
    pts = pts[:, :2] / pts[:, 2:]
    return pts


def render_hand_mesh(hand_mesh, K, W, H):
    """
    使用Trimesh渲染手的网格并将其投影到2D图像平面
    """
    # Check if mesh is valid
    if hand_mesh is None or len(hand_mesh.vertices) == 0 or len(hand_mesh.faces) == 0:
        # Return empty image if mesh is invalid
        return np.ones((H, W, 3), dtype=np.uint8) * 255
    
    # 投影到2D图像
    hand_pts_2d = project_points(hand_mesh.vertices, K)

    # 创建空白图像，用于显示手部网格
    hand_img = np.ones((H, W, 3), dtype=np.uint8) * 255  # 白色背景

    # 绘制手部网格的面
    for face in hand_mesh.faces:
        if len(face) == 3:  # Ensure face has 3 vertices
            try:
                pts_2d = hand_pts_2d[face]
                pts_2d = pts_2d.astype(np.int32)
                if pts_2d.shape[0] == 3:
                    cv2.polylines(hand_img, [pts_2d], isClosed=True, color=(0, 255, 0), thickness=1)
            except (IndexError, ValueError) as e:
                # Skip invalid faces
                continue

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
    i, pose_data, verts_m, faces_m, outlier_idxs, orig_vertices, orig_mesh, dataloader, object_idx, render_hands, hand_data, mano_layer_left, mano_layer_right, poses_m = args
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

    # 重建左右手mesh - use poses from npy, other data from pkl
    left_hand_mesh = None
    right_hand_mesh = None
    if render_hands and hand_data is not None and mano_layer_left is not None and mano_layer_right is not None:
        left_pose = poses_m[0][i]  # left hand pose for current frame
        right_pose = poses_m[1][i]  # right hand pose for current frame
        
        try:
            left_hand_mesh = reconstruct_left_hand_mesh(hand_data, i, mano_layer_right, left_pose, mano_layer_left)
            right_hand_mesh = reconstruct_right_hand_mesh(hand_data, i, mano_layer_right, right_pose)
            
            # Debug: check if meshes are valid
            if left_hand_mesh is None or len(left_hand_mesh.vertices) == 0:
                print(f"Warning: Left hand mesh is empty for frame {i}")
                left_hand_mesh = None
            if right_hand_mesh is None or len(right_hand_mesh.vertices) == 0:
                print(f"Warning: Right hand mesh is empty for frame {i}")
                right_hand_mesh = None
                
        except Exception as e:
            print(f"Error reconstructing hand meshes for frame {i}: {e}")
            left_hand_mesh = None
            right_hand_mesh = None
    
    colors_m = [(0.0, 1.0, 1.0), (0.9803921568627451, 0.2901960784313726, 0.16862745098039217)]
    Ks = dataloader.rs_Ks
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

        # 手部mesh - only process if meshes are valid and hands are enabled
        if render_hands:
            if left_hand_mesh is not None:
                try:
                    left_hand_mesh_copy = left_hand_mesh.copy()
                    left_hand_mesh_copy.vertices = left_hand_mesh_copy.vertices.copy()
                    left_hand_mesh_copy.apply_transform(world2cam)
                    left_hand_img = render_hand_mesh(left_hand_mesh_copy, K, W, H)
                    vis = cv2.addWeighted(vis, 0.6, left_hand_img, 0.4, 0)
                except Exception as e:
                    print(f"Error processing left hand mesh for frame {i}, serial {serial}: {e}")
            
            if right_hand_mesh is not None:
                try:
                    right_hand_mesh_copy = right_hand_mesh.copy()
                    right_hand_mesh_copy.vertices = right_hand_mesh_copy.vertices.copy()
                    right_hand_mesh_copy.apply_transform(world2cam)
                    right_hand_img = render_hand_mesh(right_hand_mesh_copy, K, W, H)
                    vis = cv2.addWeighted(vis, 0.6, right_hand_img, 0.4, 0)
                except Exception as e:
                    print(f"Error processing right hand mesh for frame {i}, serial {serial}: {e}")

        frame_tiles.append(vis)

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
    parser.add_argument("--render_hands", action="store_true", help="是否渲染手部网格")
    args = parser.parse_args()

    # Use MyClusterLoader instead of IMGLoader
    # sequence_folder = f"/viscam/projects/robotool/data/{args.data_path}"
    sequence_folder = args.data_path
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

    task_name = loader._data_folder.parent.name
    sequence_name = loader._data_folder.name
    annotated_base = loader._data_folder.parent.parent.parent / f"{loader._folder_name}_annotated" / task_name / loader._sequence_name
    if args.pose_file == "fd":
        pose_npy_path = str(annotated_base / "processed/fd_pose_solver/fd_poses_merged_fixed.npy")
    elif args.pose_file == "adaptive":
        pose_npy_path = str(annotated_base / "processed/fd_pose_solver/adaptive_fd_poses_merged_fixed.npy")
    elif args.pose_file == "optimized":
        pose_npy_path = str(annotated_base / "processed/joint_pose_solver/poses_o.npy")
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
    # Load hand data and initialize MANO layers if rendering hands
    hand_data = None
    mano_layer_left = None
    mano_layer_right = None
    poses_m = None
    if args.render_hands:
        # Load pkl hand data (for betas, translations, base_rot)
        pkl_file_path = str(annotated_base / "result_hand_optimized.pkl")
        try:
            hand_data = load_pkl_and_get_hand_data(pkl_file_path)
            mano_layer_left, mano_layer_right = init_mano_layers(hand_data)
            poses_m = load_poses_m(mano_file)
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

    print("Hand rendering enabled:", args.render_hands)
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=min(12, os.cpu_count() or 12))
    args_list2 = [
        (i, pose_data, None, None, [], orig_vertices, orig_mesh, loader, args.object_idx, args.render_hands, hand_data, mano_layer_left, mano_layer_right, poses_m)
        for i in range(num_frames)
    ]
    for frame in tqdm(pool.imap(process_frame_pose_npy_h5, args_list2), total=num_frames):
        # frame = frame[..., ::-1].copy()
        video_out2.write(frame)
    pool.close()
    pool.join()
    video_out2.release()
    print(f"[INFO] pose_npy_in_cams_2x4.mp4 saved to {output_path2}{'_' + args.uuid if args.uuid else ''}")
