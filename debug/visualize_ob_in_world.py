import os
import cv2
import numpy as np
from pathlib import Path
import trimesh
import yaml
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import multiprocessing


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


def process_frame_world_to_cam(args):
    i, color_roots, sam_mask_roots, ob_in_world_root, extrinsics_dict, mesh_path, Ks, serials, outlier_idxs, orig_vertices, orig_mesh = args
    W, H = 640, 480
    frame_tiles = []
    pose_path = ob_in_world_root / f"{i:06d}.txt"
    if not pose_path.exists():
        frame_tiles = [np.ones((H, W, 3), dtype=np.uint8) * 255 for _ in serials]
        return concat_frames_grid(frame_tiles, (2, 4))
    ob_in_world = load_pose(pose_path)
    for serial in serials:
        color = cv2.imread(str(Path(color_roots[serial]) / f"color_{i:06d}.jpg"))
        sam_mask = cv2.imread(str(Path(sam_mask_roots[serial]) / f"mask_{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if color is None or sam_mask is None:
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        sam_overlay = color.copy()
        sam_overlay[sam_mask > 0] = [0, 0, 255]
        mesh = orig_mesh.copy()
        mesh.vertices = orig_vertices.copy()
        mesh.apply_transform(ob_in_world)
        world2cam = np.linalg.inv(extrinsics_dict[serial])
        mesh.apply_transform(world2cam)
        pts = project_points(mesh.vertices, Ks[serial])
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]
        vis = sam_overlay.copy()
        color_dot = (0, 0, 255) if i in outlier_idxs else (255, 0, 0)
        for x, y in pts:
            cv2.circle(vis, (x, y), 2, color_dot, -1)
        frame_tiles.append(vis)
    return concat_frames_grid(frame_tiles, (2, 4))


def process_frame_pose_npy(args):
    i, color_roots, sam_mask_roots, pose_data, extrinsics_dict, mesh_path, Ks, serials, outlier_idxs, orig_vertices, orig_mesh = args
    W, H = 640, 480
    frame_tiles = []
    if i >= len(pose_data):
        frame_tiles = [np.ones((H, W, 3), dtype=np.uint8) * 255 for _ in serials]
        return concat_frames_grid(frame_tiles, (2, 4))
    qx, qy, qz, qw, tx, ty, tz = pose_data[i]
    q = np.array([qx, qy, qz, qw])
    t = np.array([tx, ty, tz])
    R_mat = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    for serial in serials:
        color = cv2.imread(str(Path(color_roots[serial]) / f"color_{i:06d}.jpg"))
        sam_mask = cv2.imread(str(Path(sam_mask_roots[serial]) / f"mask_{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if color is None or sam_mask is None:
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        sam_overlay = color.copy()
        sam_overlay[sam_mask > 0] = [0, 0, 255]
        mesh = orig_mesh.copy()
        mesh.vertices = orig_vertices.copy()
        mesh.apply_transform(T)
        world2cam = np.linalg.inv(extrinsics_dict[serial])
        mesh.apply_transform(world2cam)
        pts = project_points(mesh.vertices, Ks[serial])
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]
        vis = sam_overlay.copy()
        color_dot = (0, 0, 255) if i in outlier_idxs else (255, 0, 0)
        for x, y in pts:
            cv2.circle(vis, (x, y), 2, color_dot, -1)
        frame_tiles.append(vis)
    return concat_frames_grid(frame_tiles, (2, 4))


def visualize_world_to_cam_tracking(
    color_roots, sam_mask_roots, ob_in_world_root,
    extrinsics_dict, mesh_path, Ks,
    output_path, serials, num_frames=246, outlier_idxs=[]
):
    ob_in_world_root = Path(ob_in_world_root)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    orig_mesh = trimesh.load(mesh_path, process=False)
    orig_mesh.vertices *= 0.001
    to_origin, _ = trimesh.bounds.oriented_bounds(orig_mesh)
    # debug
    orig_mesh.apply_transform(to_origin)

    orig_vertices = orig_mesh.vertices.copy()
    print(f"[INFO] Loaded mesh from {mesh_path}, vertices shape: {orig_vertices.shape}")
    W, H = 640, 480
    video_out = cv2.VideoWriter(
        str(output_path / "world_to_cam_tracking_2x4.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (W * 4, H * 2)
    )
    pool = multiprocessing.Pool(processes=min(8, os.cpu_count()))
    args_list = [
        (i, color_roots, sam_mask_roots, ob_in_world_root, extrinsics_dict, mesh_path, Ks, serials, outlier_idxs, orig_vertices, orig_mesh)
        for i in range(num_frames)
    ]
    for frame in tqdm(pool.imap(process_frame_world_to_cam, args_list), total=num_frames):
        video_out.write(frame)
    pool.close()
    pool.join()
    video_out.release()
    print(f"[INFO] world_to_cam_tracking_2x4.mp4 saved to {output_path}")


def visualize_pose_npy_in_cams(
    color_roots, sam_mask_roots, pose_npy_path,
    extrinsics_dict, mesh_path, Ks,
    output_path, serials, num_frames=246, outlier_idxs=[]
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    orig_mesh = trimesh.load(mesh_path, process=False)
    orig_mesh.vertices *= 0.001
    to_origin, _ = trimesh.bounds.oriented_bounds(orig_mesh)
    #
    orig_mesh.apply_transform(to_origin)
    orig_vertices = orig_mesh.vertices.copy()
    print(f"[INFO] Loaded mesh from {mesh_path}, vertices shape: {orig_vertices.shape}")
    W, H = 640, 480
    video_out = cv2.VideoWriter(
        str(output_path / "pose_npy_in_cams_2x4.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (W * 4, H * 2)
    )
    pose_data = np.load(pose_npy_path).reshape(-1, 7)
    pool = multiprocessing.Pool(processes=min(8, os.cpu_count()))
    args_list = [
        (i, color_roots, sam_mask_roots, pose_data, extrinsics_dict, mesh_path, Ks, serials, outlier_idxs, orig_vertices, orig_mesh)
        for i in range(num_frames)
    ]
    for frame in tqdm(pool.imap(process_frame_pose_npy, args_list), total=num_frames):
        video_out.write(frame)
    pool.close()
    pool.join()
    video_out.release()
    print(f"[INFO] pose_npy_in_cams_2x4.mp4 saved to {output_path}")


if __name__ == "__main__":
    serials = [f"{i:02d}" for i in range(8)]
    K = np.array([[607.4, 0.0, 320.0],
                  [0.0, 607.4, 240.0],
                  [0.0, 0.0, 1.0]])
    Ks = {s: K for s in serials}
    data_path = "test_1/20250701_012148"
    tool_name = "blue_scooper"
    # base_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250630_165212"
    base_path = f"/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/{data_path}"
    sam_base = f"{base_path}/processed/segmentation/sam2"
    color_roots = {s: f"{base_path}/{s}" for s in serials}
    sam_mask_roots = {s: f"{sam_base}/{s}/mask" for s in serials}

    extrinsics_yaml = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml"
    extrinsics_dict = load_extrinsics_yaml(extrinsics_yaml, serials)

    visualize_world_to_cam_tracking(
        color_roots=color_roots,
        sam_mask_roots=sam_mask_roots,
        ob_in_world_root=Path(f"{base_path}/processed/fd_pose_solver/{tool_name}/ob_in_world"),
        extrinsics_dict=extrinsics_dict,
        mesh_path=f"{base_path}/../../models/{tool_name}/cleaned_mesh_10000.obj",
        Ks=Ks,
        output_path=Path(f"debug_output/{data_path}/world_to_cam_video"),
        serials=serials,
        num_frames=246,
        outlier_idxs=[]
    )
    # 新增npy可视化
    pose_npy_path = f"{base_path}/processed/fd_pose_solver/fd_poses_merged_fixed.npy"
    visualize_pose_npy_in_cams(
        color_roots=color_roots,
        sam_mask_roots=sam_mask_roots,
        pose_npy_path=pose_npy_path,
        extrinsics_dict=extrinsics_dict,
        mesh_path=f"{base_path}/../../models/{tool_name}/cleaned_mesh_10000.obj",
        Ks=Ks,
        output_path=Path(f"debug_output/{data_path}/pose_npy_in_cams_video"),
        serials=serials,
        num_frames=246,
        outlier_idxs=[]
    )
