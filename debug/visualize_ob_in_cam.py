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

def concat_frames_grid(frames, grid_shape=(2, 4)):
    assert len(frames) == grid_shape[0] * grid_shape[1]
    h, w = frames[0].shape[:2]
    rows = []
    for i in range(grid_shape[0]):
        row = np.concatenate(frames[i*grid_shape[1]:(i+1)*grid_shape[1]], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    return grid

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

def get_depth_img(frame_idx, serial_idx):
    if use_h5:
        depth = depths_h5[frame_idx, serial_idx]
        return depth.copy()
    else:
        serial = serials[serial_idx]
        depth_path = Path(color_roots[serial]) / f"depth_{frame_idx:06d}.png"
        if not depth_path.exists():
            return None
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        return depth

def draw_frame_number(frame, frame_idx):
    # Draw frame number at top-left corner
    text = f"Frame: {frame_idx:05d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    color = (255, 255, 255) if np.mean(frame[:40, :200]) < 128 else (0, 0, 0)  # auto white/black
    cv2.putText(frame, text, (20, 50), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def process_frame_cam(args):
    i, Ks, serials, ob_in_cam_root, orig_vertices, orig_mesh = args
    W, H = 640, 480
    frame_tiles = []
    for serial_idx, serial in enumerate(serials):
        pose_path = Path(ob_in_cam_root) / serial / f"{i:06d}.txt"
        color = get_color_img(i, serial_idx)
        sam_mask = get_mask_img(i, serial_idx)
        if color is None or sam_mask is None or not pose_path.exists():
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        sam_overlay = color.copy()
        sam_overlay[sam_mask > 0] = [0, 0, 255]
        mesh = orig_mesh.copy()
        mesh.vertices = orig_vertices.copy()
        ob_in_cam = load_pose(pose_path)
        mesh.apply_transform(ob_in_cam)
        pts = project_points(mesh.vertices, Ks[serial])
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
        pts = pts.astype(np.int32)[::200]
        vis = sam_overlay.copy()
        color_dot = (255, 0, 0)
        for x, y in pts:
            cv2.circle(vis, (x, y), 2, color_dot, -1)
        frame_tiles.append(vis)
    grid = concat_frames_grid(frame_tiles, (2, 4))
    grid = draw_frame_number(grid, i)
    return grid

def process_frame_masked_depth(args):
    i, serials = args
    W, H = 640, 480
    frame_tiles = []
    for serial_idx, serial in enumerate(serials):
        depth = get_depth_img(i, serial_idx)
        mask = get_mask_img(i, serial_idx)
        if depth is None or mask is None:
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        # Apply mask
        masked_depth = depth.copy()
        masked_depth[mask == 0] = 0
        # Normalize for visualization
        if np.max(masked_depth) > 0:
            norm_depth = (masked_depth.astype(np.float32) - masked_depth[masked_depth > 0].min())
            norm_depth = norm_depth / (masked_depth[masked_depth > 0].max() - masked_depth[masked_depth > 0].min() + 1e-6)
            norm_depth = (norm_depth * 255).astype(np.uint8)
        else:
            norm_depth = masked_depth.astype(np.uint8)
        vis = cv2.cvtColor(norm_depth, cv2.COLOR_GRAY2BGR)
        # Set background to black where mask==0
        vis[mask == 0] = [0, 0, 0]
        frame_tiles.append(vis)
    grid = concat_frames_grid(frame_tiles, (2, 4))
    grid = draw_frame_number(grid, i)
    return grid

def process_frame_raw_depth(args):
    i, serials = args
    W, H = 640, 480
    frame_tiles = []
    for serial_idx, serial in enumerate(serials):
        depth = get_depth_img(i, serial_idx)
        if depth is None:
            frame_tiles.append(np.ones((H, W, 3), dtype=np.uint8) * 255)
            continue
        # Normalize for visualization
        if np.max(depth) > 0:
            valid = (depth > 0)
            norm_depth = np.zeros_like(depth, dtype=np.float32)
            norm_depth[valid] = (depth[valid] - depth[valid].min()) / (depth[valid].max() - depth[valid].min() + 1e-6)
            norm_depth = (norm_depth * 255).astype(np.uint8)
        else:
            norm_depth = depth.astype(np.uint8)
        vis = cv2.cvtColor(norm_depth, cv2.COLOR_GRAY2BGR)
        # Set background to white where depth==0
        vis[depth == 0] = [255, 255, 255]
        frame_tiles.append(vis)
    grid = concat_frames_grid(frame_tiles, (2, 4))
    grid = draw_frame_number(grid, i)
    return grid

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="数据路径，如 test_1/20250701_012148")
    parser.add_argument("--tool_name", type=str, required=True, help="工具名，如 blue_scooper")
    parser.add_argument("--output_idx", type=str, default="0", help="输出编号")
    parser.add_argument("--uuid", type=str, default="", help="uuid")
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
    uuid = args.uuid
    ################

    base_path = f"/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/{data_path}"
    sam_base = f"{base_path}/processed/segmentation/sam2"
    color_roots = {s: f"{base_path}/{s}" for s in serials}
    sam_mask_roots = {s: f"{sam_base}/{s}/mask" for s in serials}
    ob_in_cam_root = f"{base_path}/processed/fd_pose_solver/{tool_name}/ob_in_cam"
    mesh_path = f"{base_path}/../../models/{tool_name}/cleaned_mesh_10000.obj"

    # H5 support
    h5_path = Path(base_path) / "all_data.h5"
    use_h5 = h5_path.exists()
    if use_h5:
        h5_file = h5py.File(h5_path, "r")
        colors_h5 = h5_file["colors"]  # (N, 8, H, W, 3)
        masks_h5 = h5_file["masks"]    # (N, 8, H, W)

    orig_mesh = trimesh.load(mesh_path, process=False)
    orig_vertices = orig_mesh.vertices.copy()
    W, H = 640, 480

    # 自动获取帧数
    if use_h5:
        num_frames = colors_h5.shape[0]
    else:
        color_dir = Path(color_roots[serials[0]])
        num_frames = len(list(color_dir.glob("color_*.jpg")))

    output_path = Path(f"debug_output/{data_path}/ob_in_cam_video/{uuid}")
    output_path.mkdir(parents=True, exist_ok=True)
    video_out = cv2.VideoWriter(
        str(output_path / f"{output_idx}_ob_in_cam_tracking_2x4.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (W * 4, H * 2)
    )
    pool = multiprocessing.Pool(processes=min(8, os.cpu_count()))
    args_list = [
        (i, Ks, serials, ob_in_cam_root, orig_vertices, orig_mesh)
        for i in range(num_frames)
    ]
    for frame in tqdm(pool.imap(process_frame_cam, args_list), total=num_frames):
        video_out.write(frame)
    
    pool.close()
    pool.join()
    video_out.release()
    print(f"[INFO] ob_in_cam_tracking_2x4.mp4 saved to {output_path}")
    if use_h5:
        h5_file.close()

    if use_h5:
        depths_h5 = h5_file["depths"]  # (N, 8, H, W)
    # Save masked depth video
    video_out_depth = cv2.VideoWriter(
        str(output_path / f"{output_idx}_masked_depth_2x4.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (W * 4, H * 2)
    )
    args_list_depth = [
        (i, serials)
        for i in range(num_frames)
    ]
    pool = multiprocessing.Pool(processes=min(8, os.cpu_count()))
    for frame in tqdm(pool.imap(process_frame_masked_depth, args_list_depth), total=num_frames):
        video_out_depth.write(frame)
    video_out_depth.release()
    print(f"[INFO] masked_depth_2x4.mp4 saved to {output_path}")

    # Save raw depth video
    video_out_raw_depth = cv2.VideoWriter(
        str(output_path / f"{output_idx}_raw_depth_2x4.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (W * 4, H * 2)
    )
    args_list_raw_depth = [
        (i, serials)
        for i in range(num_frames)
    ]
    for frame in tqdm(pool.imap(process_frame_raw_depth, args_list_raw_depth), total=num_frames):
        video_out_raw_depth.write(frame)
    video_out_raw_depth.release()
    print(f"[INFO] raw_depth_2x4.mp4 saved to {output_path}")

    pool.close()
    
