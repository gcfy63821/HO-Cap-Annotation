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
from manopth.manolayer import ManoLayer
from hocap_annotation.utils.color_info import *
from hocap_annotation.utils.mano_info import *
import pickle
from hocap_annotation.utils import  CFG

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
    # mano_layer_left = MANOLayer('left', mano_betas_left).to('cuda')
    # mano_layer_right = MANOLayer('right', mano_betas_right).to('cuda')
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
        
        # Debug prints
        # print(f"Left hand - pose shape: {pose.shape}, translation shape: {translation.shape}")
        
        # verts, joints = mano_layer(pose, translation)
        verts, joints = mano_layer(pose, hand_beta.float())
        verts = verts[0] / 1000
        joints = joints[0] / 1000

        if verts.size(0) == 1:
            verts = verts.squeeze(0)
            joints = joints.squeeze(0)
        
        # print(f"Left hand - verts shape: {verts.shape}, joints shape: {joints.shape}")
        
        root_trans = joints[0].clone().detach()
        verts -= root_trans
        verts[:, 0] *= -1
        verts = verts @ base_rot.T
        verts += translation
        # faces = mano_layer.f.detach().cpu().numpy()
        faces = mano_layer_left.th_faces.detach().cpu().numpy()
        
        # print(f"Left hand - final verts shape: {verts.shape}, faces shape: {faces.shape}")
        
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
        hand_beta = torch.tensor(hand_data['left_hand_beta']).to('cuda')
        
        # Debug prints
        # print(f"Right hand - pose shape: {pose.shape}, translation shape: {translation.shape}")
        
        # verts, joints = mano_layer_right(pose, translation)
        verts, joints = mano_layer_right(pose, hand_beta.float())
        
        verts = verts[0] / 1000
        joints = joints[0] / 1000
        if verts.size(0) == 1:
            verts = verts.squeeze(0)
            joints = joints.squeeze(0)
        
        # print(f"Right hand - verts shape: {verts.shape}, joints shape: {joints.shape}")
        
        root_trans = joints[0].clone().detach()
        verts -= root_trans
        verts += translation
        # faces = mano_layer_right.f.detach().cpu().numpy()
        faces = mano_layer_right.th_faces.detach().cpu().numpy()
        
        # print(f"Right hand - final verts shape: {verts.shape}, faces shape: {faces.shape}")
        
        mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces)
        return mesh
    except Exception as e:
        print(f"Error in reconstruct_right_hand_mesh for frame {frame_idx}: {e}")
        return None


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

def load_poses_m(pose_file):
    poses = np.load(pose_file).astype(np.float32)
    mano_sides = ['left','right']
    poses = np.stack(
        [poses[0 if side == "right" else 1] for side in mano_sides], axis=0
    )  # (num_hands, num_frames 51)
    return poses

def load_mano_beta():
    file_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/mano/squeegee_1.yaml"
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return np.array(data['betas'], dtype=np.float32)

def init_mano_group_layer():
    betas = load_mano_beta()
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


def load_extrinsics_yaml(yaml_path, serials):
    def create_mat(values):
        return np.array([values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]], dtype=np.float32)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    extr = data["extrinsics"]
    return {s: create_mat(extr[s]) for s in serials}


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

class IMGLoader():
    def __init__(self,data_path):

        self.serials = [f"{i:02d}" for i in range(8)]
        self.K = np.array([[607.4, 0.0, 320.0],
                    [0.0, 607.4, 240.0],
                    [0.0, 0.0, 1.0]])
        serials = self.serials
        self.Ks = {s: K for s in serials}
        
        self.base_path = f"/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/{data_path}"
        self.sam_base = f"{self.base_path}/processed/segmentation/sam2"
        self.color_roots = {s: f"{self.base_path}/{s}" for s in serials}
        self.sam_mask_roots = {s: f"{self.sam_base}/{s}/mask" for s in serials}
        extrinsics_yaml = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml"
        self.extrinsics_dict = load_extrinsics_yaml(extrinsics_yaml, serials)

    # 修改frame读取逻辑
    def get_color_img(self, frame_idx, serial_idx):
        use_h5 = False
        if use_h5:
            img = self.colors_h5[frame_idx, serial_idx]
            return img[..., ::-1].copy()  # RGB->BGR for cv2
        else:
            serial = self.serials[serial_idx]
            return cv2.imread(str(Path(self.color_roots[serial]) / f"color_{frame_idx:06d}.jpg"))

    def get_mask_img(self, frame_idx, serial_idx):
        use_h5 = False
        if use_h5:
            mask = masks_h5[frame_idx, serial_idx]
            return (mask > 0).astype(np.uint8) * 255
        else:
            serial = self.serials[serial_idx]
            return cv2.imread(str(Path(self.sam_mask_roots[serial]) / f"mask_{frame_idx:06d}.png"), cv2.IMREAD_GRAYSCALE)

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

def process_frame_pose_npy_h5(args):
    i, pose_data, hand_data, mano_layer_left, mano_layer_right, poses_m, orig_vertices, orig_mesh, dataloader = args
    W, H = 640, 480
    frame_tiles = []

    if i >= len(pose_data):
        frame_tiles = [np.ones((H, W, 3), dtype=np.uint8) * 255 for _ in dataloader.serials]
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
    left_pose = poses_m[0][i]  # left hand pose for current frame
    right_pose = poses_m[1][i]  # right hand pose for current frame
    
    # Use correct MANO layers for each hand
    try:
        # left_hand_mesh = reconstruct_left_hand_mesh(hand_data, i, mano_layer_left, left_pose)
        left_hand_mesh = reconstruct_left_hand_mesh(hand_data, i, mano_layer_right, left_pose, mano_layer_left) # special setting
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
    Ks = dataloader.Ks

    for serial_idx, serial in enumerate(dataloader.serials):
        color = dataloader.get_color_img(i, serial_idx)
        sam_mask = dataloader.get_mask_img(i, serial_idx)
        if color is None or sam_mask is None:
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        sam_overlay = color.copy()
        sam_overlay[sam_mask > 0] = [0, 0, 255]

        # 可视化物体
        mesh = orig_mesh.copy()
        mesh.vertices = orig_vertices.copy()
        mesh.apply_transform(T)
        world2cam = np.linalg.inv(dataloader.extrinsics_dict[serial])
        mesh.apply_transform(world2cam)
        pts = project_points(mesh.vertices, Ks[serial])
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]
        
        vis = sam_overlay.copy()
        color_dot = colors_m[1]
        for x, y in pts:
            cv2.circle(vis, (x, y), 2, color_dot, -1)

        # 手部mesh - only process if meshes are valid
        if left_hand_mesh is not None:
            try:
                left_hand_mesh_copy = left_hand_mesh.copy()
                left_hand_mesh_copy.vertices = left_hand_mesh_copy.vertices.copy()
                left_hand_mesh_copy.apply_transform(world2cam)
                left_hand_img = render_hand_mesh(left_hand_mesh_copy, Ks[serial], W, H)
                vis = cv2.addWeighted(vis, 0.6, left_hand_img, 0.4, 0)
            except Exception as e:
                print(f"Error processing left hand mesh for frame {i}, serial {serial}: {e}")
        
        if right_hand_mesh is not None:
            try:
                right_hand_mesh_copy = right_hand_mesh.copy()
                right_hand_mesh_copy.vertices = right_hand_mesh_copy.vertices.copy()
                right_hand_mesh_copy.apply_transform(world2cam)
                right_hand_img = render_hand_mesh(right_hand_mesh_copy, Ks[serial], W, H)
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
    parser.add_argument("--pose_file", type=str, default="optimized", choices=["fd", "adaptive", "optimized"], help="选择foundation pose 或 optimized")
    parser.add_argument("--uuid", type=str, default="", help="唯一标识符，用于区分不同运行")
    parser.add_argument("--object_idx", type=int, default=1, help="物体索引，默认为0")
    args = parser.parse_args()

    serials = [f"{i:02d}" for i in range(8)]
    K = np.array([[607.4, 0.0, 320.0],
                  [0.0, 607.4, 240.0],
                  [0.0, 0.0, 1.0]])
    Ks = {s: K for s in serials}
    ################
    data_path = args.data_path
    tool_name = args.tool_name
    output_idx = args.output_idx
    pose_file = args.pose_file
    uuid = "_" + args.uuid if args.uuid else ""
    ################

    data_loader = IMGLoader(data_path)

    base_path = f"/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/{data_path}"
    sam_base = f"{base_path}/processed/segmentation/sam2"
    color_roots = {s: f"{base_path}/{s}" for s in serials}
    sam_mask_roots = {s: f"{sam_base}/{s}/mask" for s in serials}

   

    
    h5_path = Path(base_path) / "all_data.h5"
    use_h5 = h5_path.exists()
    if use_h5:
        h5_file = h5py.File(h5_path, "r")
        colors_h5 = h5_file["colors"]  # (N, 8, H, W, 3)
        masks_h5 = h5_file["masks"]    # (N, 8, H, W)

    extrinsics_yaml = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml"
    extrinsics_dict = load_extrinsics_yaml(extrinsics_yaml, serials)


    # 自动获取帧数
    if use_h5:
        num_frames = colors_h5.shape[0]
    else:
        color_dir = Path(color_roots[serials[0]])
        num_frames = len(list(color_dir.glob("color_*.jpg")))

    

    orig_mesh = trimesh.load(f"{base_path}/../../models/{tool_name}/cleaned_mesh_10000.obj", process=False)
    orig_vertices = orig_mesh.vertices.copy()
    W, H = 640, 480


    # pose_npy_in_cams
    if pose_file == "fd":
        pose_npy_path = f"{base_path}/processed/fd_pose_solver/fd_poses_merged_fixed.npy"
    elif pose_file == "adaptive":
        pose_npy_path = f"{base_path}/processed/fd_pose_solver/adaptive_fd_poses_merged_fixed.npy"
    elif pose_file == "optimized":
        pose_npy_path = f"{base_path}/processed/joint_pose_solver/poses_o.npy"
    output_path2 = Path(f"debug_output/{data_path}/hand_video")
    output_path2.mkdir(parents=True, exist_ok=True)
    video_out2 = cv2.VideoWriter(
        str(output_path2 / f"{output_idx}{uuid}_{pose_file}_hand_video.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (W * 4, H * 2)
    )
    
    pose_data = np.load(pose_npy_path)
    print(f"[INFO] Loaded pose data from {pose_npy_path}, shape: {pose_data.shape}")
    # 根据object_idx选择对应物体
    if pose_data.ndim == 3:
        pose_data = pose_data[args.object_idx - 1]
        print(f"[INFO] Using pose_data[{args.object_idx - 1}], shape: {pose_data.shape}")
    pose_data = pose_data.reshape(-1, 7)

    # 加载pkl手部数据 (for betas, translations, base_rot)
    pkl_file_path = f"{base_path}/processed/result_hand_optimized.pkl"
    hand_data = load_pkl_and_get_hand_data(pkl_file_path)

    # 加载MANO手部序列数据 (for poses)
    mano_file = f"{base_path}/processed/joint_pose_solver/poses_m.npy"
    poses_m = load_poses_m(mano_file)

    # 初始化MANO layers，分别为左右手加载betas
    mano_layer_left, mano_layer_right = init_mano_layers(hand_data)
    print(f"[INFO] Loaded mano_betas_left: {get_betas(hand_data['left_hand_beta']).shape}, mano_betas_right: {get_betas(hand_data['right_hand_beta']).shape}")
    print(f"[INFO] Loaded poses_m from {mano_file}, shape: {poses_m.shape}")
    print(f"[INFO] Loaded hand data from {pkl_file_path}")

    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=min(8, os.cpu_count()))
    args_list2 = [
        (i, pose_data, hand_data, mano_layer_left, mano_layer_right, poses_m, orig_vertices, orig_mesh, data_loader)
        for i in range(num_frames)
    ]
    for frame in tqdm(pool.imap(process_frame_pose_npy_h5, args_list2), total=num_frames):
        # 如果用h5，frame为RGB，需转为BGR再写入
        if use_h5:
            frame = frame[..., ::-1].copy()
        video_out2.write(frame)
    pool.close()
    pool.join()
    video_out2.release()
    print(f"[INFO] pose_npy_in_cams_2x4.mp4 saved to {output_path2}{uuid}")

    if use_h5:
        h5_file.close()
