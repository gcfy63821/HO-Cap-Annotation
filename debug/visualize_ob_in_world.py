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

# 修改frame读取逻辑
def get_color_img(frame_idx, serial_idx):
    if use_h5:
        img = colors_h5[frame_idx, serial_idx]
        return img[..., ::-1].copy()  # RGB->BGR for cv2
    else:
        serial = serials[serial_idx]
        return cv2.imread(str(Path(color_roots[serial]) / f"color_{frame_idx:06d}.jpg"))

def get_mask_img(frame_idx, serial_idx):
    if use_h5:
        mask = masks_h5[frame_idx, serial_idx]
        return (mask > 0).astype(np.uint8) * 255
    else:
        serial = serials[serial_idx]
        return cv2.imread(str(Path(sam_mask_roots[serial]) / f"mask_{frame_idx:06d}.png"), cv2.IMREAD_GRAYSCALE)

# 修改process_frame_world_to_cam和process_frame_pose_npy的color/mask读取部分
def process_frame_world_to_cam_h5(args):
    i, Ks, serials, outlier_idxs, orig_vertices, orig_mesh = args
    W, H = 640, 480
    frame_tiles = []
    pose_path = Path(base_path) / "processed" / "fd_pose_solver" / tool_name / "ob_in_world" / f"{i:06d}.txt"
    if not pose_path.exists():
        frame_tiles = [np.ones((H, W, 3), dtype=np.uint8) * 255 for _ in serials]
        return concat_frames_grid(frame_tiles, (2, 4))
    ob_in_world = load_pose(pose_path)
    for serial_idx, serial in enumerate(serials):
        color = get_color_img(i, serial_idx)
        sam_mask = get_mask_img(i, serial_idx)
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

def process_frame_pose_npy_h5(args):
    i, pose_data, Ks, serials, outlier_idxs, orig_vertices, orig_mesh = args
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
    for serial_idx, serial in enumerate(serials):
        color = get_color_img(i, serial_idx)
        sam_mask = get_mask_img(i, serial_idx)
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
    # orig_mesh.vertices *= 0.001
    # to_origin, _ = trimesh.bounds.oriented_bounds(orig_mesh)
    # # debug
    # orig_mesh.apply_transform(to_origin)

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
    # orig_mesh.vertices *= 0.001
    # to_origin, _ = trimesh.bounds.oriented_bounds(orig_mesh)
    # #
    # orig_mesh.apply_transform(to_origin)
    orig_vertices = orig_mesh.vertices.copy()
    print(f"[INFO] Loaded mesh from {mesh_path}, vertices shape: {orig_vertices.shape}")
    W, H = 640, 480
    video_out = cv2.VideoWriter(
        str(output_path / "pose_npy_in_cams_2x4.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (W * 4, H * 2)
    )
    pose_data = np.load(pose_npy_path).reshape(-1, 7)
    import concurrent.futures
    args_list2 = [
        (i, pose_data, Ks, serials, [], orig_vertices, orig_mesh)
        for i in range(num_frames)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
        for frame in tqdm(executor.map(process_frame_pose_npy_h5, args_list2), total=num_frames):
            # 如果用h5，frame为RGB，需转为BGR再写入
            if use_h5:
                frame = frame[..., ::-1].copy()
            video_out2.write(frame)
    video_out2.release()
    print(f"[INFO] pose_npy_in_cams_2x4.mp4 saved to {output_path2}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_1/20250701_012148", help="数据路径，如 test_1/20250701_012148")
    parser.add_argument("--tool_name", type=str, default="blue_scooper", help="工具名，如 blue_scooper")
    parser.add_argument("--output_idx", type=str, default="0", help="输出编号")
    parser.add_argument("--pose_file", type=str, default="fd", choices=["fd", "adaptive", "optimized"], help="选择foundation pose 或 optimized")
    parser.add_argument("--uuid", type=str, default="", help="唯一标识符，用于区分不同运行")
    parser.add_argument("--object_idx", type=int, default=0, help="物体索引，默认为0")
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

    ###################
    if pose_file == "fd":
        # world_to_cam_tracking
        output_path1 = Path(f"debug_output/{data_path}/world_to_cam_video")
        output_path1.mkdir(parents=True, exist_ok=True)
        video_out1 = cv2.VideoWriter(
            str(output_path1 / f"{output_idx}{uuid}_world_to_cam_tracking_2x4.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            20, (W * 4, H * 2)
        )
        import multiprocessing
        pool = multiprocessing.Pool(processes=min(32, os.cpu_count()))
        args_list1 = [
            (i, Ks, serials, [], orig_vertices, orig_mesh)
            for i in range(num_frames)
        ]
        for frame in tqdm(pool.imap(process_frame_world_to_cam_h5, args_list1), total=num_frames):
            # 如果用h5，frame为RGB，需转为BGR再写入
            if use_h5:
                frame = frame[..., ::-1].copy()
            video_out1.write(frame)
        pool.close()
        pool.join()
        video_out1.release()
        print(f"[INFO] world_to_cam_tracking_2x4.mp4 saved to {output_path1}{uuid}")

    ########################

    # pose_npy_in_cams
    if pose_file == "fd":
        pose_npy_path = f"{base_path}/processed/fd_pose_solver/fd_poses_merged_fixed.npy"
    elif pose_file == "adaptive":
        pose_npy_path = f"{base_path}/processed/fd_pose_solver/adaptive_fd_poses_merged_fixed.npy"
    elif pose_file == "optimized":
        pose_npy_path = f"{base_path}/processed/object_pose_solver/poses_o.npy"
    output_path2 = Path(f"debug_output/{data_path}/pose_npy_in_cams_video")
    output_path2.mkdir(parents=True, exist_ok=True)
    video_out2 = cv2.VideoWriter(
        str(output_path2 / f"{output_idx}{uuid}_{pose_file}_pose_npy_in_cams_2x4.mp4"),
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

    pool = multiprocessing.Pool(processes=min(32, os.cpu_count()))
    args_list2 = [
        (i, pose_data, Ks, serials, [], orig_vertices, orig_mesh)
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
