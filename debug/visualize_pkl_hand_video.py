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
from hocap_annotation.layers import MANOGroupLayer, MANOLayer
from hocap_annotation.utils.color_info import *
from hocap_annotation.utils.mano_info import *
import pickle

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

def init_mano_layers(hand_data):
    mano_betas_left = get_betas(hand_data['left_hand_beta'])
    mano_betas_right = get_betas(hand_data['right_hand_beta'])
    mano_layer_left = MANOLayer('left', mano_betas_left).to('cuda')
    mano_layer_right = MANOLayer('right', mano_betas_right).to('cuda')
    return mano_layer_left, mano_layer_right

def reconstruct_left_hand_mesh(hand_data, frame_idx, mano_layer_left):
    pose = torch.tensor(hand_data['left_hand_pose'][frame_idx]).to('cuda').unsqueeze(0)
    translation = torch.tensor(hand_data['left_hand_translation'][frame_idx]).to('cuda').unsqueeze(0)
    base_rot = torch.tensor(hand_data['left_hand_base_rot'][frame_idx]).to('cuda') if hand_data['left_hand_base_rot'].ndim == 3 else torch.eye(3).to('cuda')
    # print(f"[DEBUG] left_hand pose shape: {pose.shape}, translation shape: {translation.shape}")
    # print(f"[DEBUG] left_hand pose: {pose}")
    # print(f"[DEBUG] left_hand translation: {translation}")
    hand_beta = torch.tensor(hand_data['left_hand_beta']).to('cuda')
    verts, joints = mano_layer_left(pose, translation)

    # verts, joints = mano_layer_left(pose, hand_beta)
    if verts.size(0) == 1:
        verts = verts.squeeze(0)
        joints = joints.squeeze(0)
    # print(f"[DEBUG] left_hand verts shape: {verts.shape}, joints shape: {joints.shape}")
    root_trans = joints[0].clone().detach()
    verts -= root_trans
    # joints -= root_trans
    verts[:, 0] *= -1
    # joints[:, 0] *= -1
    verts = verts @ base_rot.T
    # joints = joints @ base_rot.T
    # rotate 180 degree around x axis
    verts = verts @ R.from_euler('x', 180, degrees=True).as_matrix()
    verts += translation
    faces = mano_layer_left.f.detach().cpu().numpy()
    mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces)
    return mesh

def reconstruct_right_hand_mesh(hand_data, frame_idx, mano_layer_right):
    pose = torch.tensor(hand_data['right_hand_pose'][frame_idx]).to('cuda').unsqueeze(0)
    translation = torch.tensor(hand_data['right_hand_translation'][frame_idx]).to('cuda').unsqueeze(0)
    base_rot = torch.tensor(hand_data['right_hand_base_rot'][frame_idx]).to('cuda') if hand_data['right_hand_base_rot'].ndim == 3 else torch.eye(3).to('cuda')
    # print(f"[DEBUG] right_hand pose shape: {pose.shape}, translation shape: {translation.shape}")
    # print(f"[DEBUG] right_hand pose: {pose}")
    # print(f"[DEBUG] right_hand translation: {translation}")
    verts, joints = mano_layer_right(pose, translation)
    if verts.size(0) == 1:
        verts = verts.squeeze(0)
        joints = joints.squeeze(0)
    # print(f"[DEBUG] right_hand verts shape: {verts.shape}, joints shape: {joints.shape}")
    # verts = verts[0] / 1000
    # joints = joints[0] / 1000
    root_trans = joints[0].clone().detach()
    verts -= root_trans
    verts += translation
    faces = mano_layer_right.f.detach().cpu().numpy()
    mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces)
    return mesh


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

def process_frame_pkl_hand(args):
    i, pose_data, orig_vertices, orig_mesh, dataloader, hand_data, mano_layer_left, mano_layer_right = args
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
    Ks = dataloader.Ks
    colors_m = [(0.0, 1.0, 1.0), (0.9803921568627451, 0.2901960784313726, 0.16862745098039217)]
    left_hand_mesh = reconstruct_left_hand_mesh(hand_data, i, mano_layer_left)
    # left_hand_mesh = reconstruct_left_hand_mesh(hand_data, i, mano_layer_right)
    right_hand_mesh = reconstruct_right_hand_mesh(hand_data, i, mano_layer_right)
    
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
        
        # 手部mesh
        left_hand_mesh_copy = left_hand_mesh.copy()
        left_hand_mesh_copy.vertices = left_hand_mesh_copy.vertices.copy()
        right_hand_mesh_copy = right_hand_mesh.copy()
        right_hand_mesh_copy.vertices = right_hand_mesh_copy.vertices.copy()
        left_hand_mesh_copy.apply_transform(world2cam) # shape: 778,3
        right_hand_mesh_copy.apply_transform(world2cam) # shape: 778,3
        left_hand_img = render_hand_mesh(left_hand_mesh_copy, Ks[serial], W, H)
        right_hand_img = render_hand_mesh(right_hand_mesh_copy, Ks[serial], W, H)
        # Overlay both hands
        vis = cv2.addWeighted(vis, 0.6, left_hand_img, 0.4, 0)
        vis = cv2.addWeighted(vis, 0.6, right_hand_img, 0.4, 0)
        frame_tiles.append(vis)
    return concat_frames_grid(frame_tiles, (2, 4))

def get_betas(b):
    b = np.array(b)
    if b.ndim == 2 and b.shape[0] == 1:
        return b[0]
    return b.squeeze()

def mano_layer_forward(poses_m, layer, subset=None):
    p = torch.cat(poses_m, dim=1)
    v, j = layer(p, subset)
    if v.size(0) == 1:
        v = v.squeeze(0)
        j = j.squeeze(0)
    return v, j

# Add IMGLoader class from visualize_hand_video.py
class IMGLoader():
    def __init__(self, data_path):
        self.serials = [f"{i:02d}" for i in range(8)]
        self.K = np.array([[607.4, 0.0, 320.0],
                           [0.0, 607.4, 240.0],
                           [0.0, 0.0, 1.0]])
        serials = self.serials
        self.Ks = {s: self.K for s in serials}
        self.base_path = f"/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/{data_path}"
        self.sam_base = f"{self.base_path}/processed/segmentation/sam2"
        self.color_roots = {s: f"{self.base_path}/{s}" for s in serials}
        self.sam_mask_roots = {s: f"{self.sam_base}/{s}/mask" for s in serials}
        extrinsics_yaml = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml"
        self.extrinsics_dict = load_extrinsics_yaml(extrinsics_yaml, serials)

    def get_color_img(self, frame_idx, serial_idx):
        # use_h5 = False, so only disk loading is used
        serial = self.serials[serial_idx]
        return cv2.imread(str(Path(self.color_roots[serial]) / f"color_{frame_idx:06d}.jpg"))

    def get_mask_img(self, frame_idx, serial_idx):
        # use_h5 = False, so only disk loading is used
        serial = self.serials[serial_idx]
        return cv2.imread(str(Path(self.sam_mask_roots[serial]) / f"mask_{frame_idx:06d}.png"), cv2.IMREAD_GRAYSCALE)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_1/20250701_012148", help="数据路径，如 test_1/20250701_012148")
    parser.add_argument("--tool_name", type=str, default="blue_scooper", help="工具名，如 blue_scooper")
    parser.add_argument("--output_idx", type=str, default="1", help="输出编号")
    parser.add_argument("--pose_file", type=str, default="fd", choices=["fd", "adaptive", "optimized"], help="选择foundation pose 或 optimized")
    parser.add_argument("--uuid", type=str, default="", help="唯一标识符，用于区分不同运行")
    parser.add_argument("--object_idx", type=int, default=1, help="物体索引，默认为0")
    # parser.add_argument("--hand_file", type=str, default="", help="手部文件，如 /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/squeegee_1/20250704_151206/processed/result_hand_optimized.pkl")
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
        pose_npy_path = f"{base_path}/processed/object_pose_solver/poses_o.npy"
    output_path2 = Path(f"debug_output/{data_path}/pose_pkl_hand_in_cams_video")
    output_path2.mkdir(parents=True, exist_ok=True)
    video_out2 = cv2.VideoWriter(
        str(output_path2 / f"{output_idx}{uuid}_{pose_file}_pkl_hand_in_cams_2x4.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (W * 4, H * 2)
    )
    pose_data = np.load(pose_npy_path)
    print(f"[INFO] Loaded pose data from {pose_npy_path}, shape: {pose_data.shape}")
    if pose_data.ndim == 3:
        pose_data = pose_data[args.object_idx - 1]
        print(f"[INFO] Using pose_data[{args.object_idx - 1}], shape: {pose_data.shape}")
    pose_data = pose_data.reshape(-1, 7)

    # 加载pkl手部数据
    # pkl_file_path = f"/home/wys/learning-compliant/crq_ws/robotool/HandReconstruction/coffee/2379b837_coffee_1/result_hand_optimized.pkl"  # 可参数化
    # pkl_file_path = args.hand_file
    pkl_file_path = f"{base_path}/processed/result_hand_optimized.pkl"
    hand_data = load_pkl_and_get_hand_data(pkl_file_path)

    # 初始化MANOGroupLayer，分别为左右手加载betas
    mano_layer_left, mano_layer_right = init_mano_layers(hand_data)
    print(f"[INFO] Loaded mano_betas_left: {get_betas(hand_data['left_hand_beta']).shape}, mano_betas_right: {get_betas(hand_data['right_hand_beta']).shape}")
    # faces_m 处理
    # (optional: you can keep faces_m for mesh rendering if needed)

    # 处理和生成视频
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=min(8, os.cpu_count()))
    args_list2 = [
        (i, pose_data, orig_vertices, orig_mesh, data_loader, hand_data, mano_layer_left, mano_layer_right)
        for i in range(num_frames)
    ]
    for frame in tqdm(pool.imap(process_frame_pkl_hand, args_list2), total=num_frames):
        video_out2.write(frame)
    pool.close()
    pool.join()
    video_out2.release()
    if use_h5:
        h5_file.close()
    # print directory
    print(f"[INFO] Output video saved to {output_path2 / f'{output_idx}{uuid}_{pose_file}_pkl_hand_in_cams_2x4.mp4'}")