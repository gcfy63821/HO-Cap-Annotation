import os
import numpy as np
import random
from pathlib import Path
import trimesh
import yaml
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# --- Helper functions from visualize_ob_in_world.py ---
def load_pose(pose_txt):
    with open(pose_txt, 'r') as f:
        arr = np.array([float(x) for x in f.read().strip().split()])
        t = np.array(arr[4:7])
        q = np.array(arr[:4])  # xyzw
        flag = arr[7] if len(arr) > 7 else 0.0  # preserve flag
        R_mat = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        return T, flag

def project_points(vertices, K):
    pts = vertices @ K[:3, :3].T
    pts = pts[:, :2] / pts[:, 2:]
    return pts

def compute_iou(mask_pred, mask_gt):
    mask_pred = (mask_pred > 0)
    mask_gt = (mask_gt > 0)
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def load_extrinsics_yaml(yaml_path, serials):
    def create_mat(values):
        return np.array([values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]], dtype=np.float32)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    extr = data["extrinsics"]
    return {s: create_mat(extr[s]) for s in serials}

def read_K_from_yaml(calib_folder, serial, cam_type="color"):
    file_path = Path(calib_folder) / "intrinsics" / f"{serial}.yaml"
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)[cam_type]
    K = np.array([
        [data["fx"], 0.0, data["ppx"]],
        [0.0, data["fy"], data["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return K

# --- Main optimization script ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="数据路径，如 test_1/20250701_012148")
    parser.add_argument("--tool_name", type=str, required=True, help="工具名，如 blue_scooper")
    parser.add_argument("--M", type=int, default=10, help="优化用的帧数")
    parser.add_argument("--max_iter", type=int, default=50, help="SGD最大迭代次数")
    parser.add_argument("--step", type=float, default=0.01, help="SGD步长")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--object_idx", type=int, default=0, help="物体索引，默认为0")
    parser.add_argument("--camera_indices", type=str, default=None, help="逗号分隔的相机索引列表，如 '0,1,2'，默认全部相机")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    serials_all = [f"{i:02d}" for i in range(8)]
    if args.camera_indices is not None:
        camera_indices = [int(idx) for idx in args.camera_indices.split(',')]
        serials = [f"{i:02d}" for i in camera_indices]
    else:
        serials = serials_all

    # Load Ks from calibration files
    calib_folder = Path("/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration")
    Ks = {s: read_K_from_yaml(calib_folder, s) for s in serials_all}

    base_path = f"/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/{args.data_path}"
    ob_in_world_dir = Path(base_path) / "processed" / "fd_pose_solver" / args.tool_name / "ob_in_world"
    sam_base = f"{base_path}/processed/segmentation/sam2"
    color_roots = {s: f"{base_path}/{s}" for s in serials_all}
    sam_mask_roots = {s: f"{sam_base}/{s}/mask" for s in serials_all}
    extrinsics_yaml = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml"
    extrinsics_dict = load_extrinsics_yaml(extrinsics_yaml, serials_all)

    # 加载mesh
    mesh_path = f"{base_path}/../../models/{args.tool_name}/cleaned_mesh_10000.obj"
    orig_mesh = trimesh.load(mesh_path, process=False)
    orig_vertices = orig_mesh.vertices.copy()
    W, H = 640, 480

    # 获取所有可用帧
    pose_files = sorted(list(ob_in_world_dir.glob("*.txt")))
    num_frames = len(pose_files)
    if num_frames == 0:
        print(f"[ERROR] No pose files found in {ob_in_world_dir}")
        exit(1)
    frame_indices = [int(f.stem) for f in pose_files]
    # 随机选M帧
    if args.M > num_frames:
        args.M = num_frames
    sample_indices = random.sample(frame_indices, args.M)
    # make sure that the indices include frame 0
    if 0 not in sample_indices:
        sample_indices.append(0)
    selected_indices = sorted(sample_indices)

    # 预加载所有mask
    def get_mask_img(frame_idx, serial_idx):
        serial = serials[serial_idx]
        mask_path = Path(sam_mask_roots[serial]) / f"mask_{frame_idx:06d}.png"
        if not mask_path.exists():
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return (mask > 0).astype(np.uint8) * 255

    # 优化目标函数
    def evaluate_offset(offset_xyz):
        iou_list = []
        for frame_idx in selected_indices:
            pose_path = ob_in_world_dir / f"{frame_idx:06d}.txt"
            ob_in_world, flag = load_pose(pose_path)
            # Apply offset
            ob_in_world_offset = ob_in_world.copy()
            # print(f"[DEBUG] ob_in_world: {ob_in_world_offset}")
            ob_in_world_offset[:3, 3] += offset_xyz
            # print(f"[DEBUG] ob_in_world_offset: {ob_in_world_offset}")
            for serial_idx, serial in enumerate(serials):
                mask = get_mask_img(frame_idx, serial_idx)
                if mask is None:
                    continue
                mesh = orig_mesh.copy()
                mesh.vertices = orig_vertices.copy()
                mesh.apply_transform(ob_in_world_offset)
                world2cam = np.linalg.inv(extrinsics_dict[serial])
                mesh.apply_transform(world2cam)
                pts = project_points(mesh.vertices, Ks[serial])
                pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
                pts = pts.astype(np.int32)[::200]
                pred_mask = np.zeros((H, W), dtype=np.uint8)
                for x, y in pts:
                    if 0 <= y < H and 0 <= x < W:
                        pred_mask[y, x] = 255
                iou = compute_iou(pred_mask, mask)
                iou_list.append(iou)
        if len(iou_list) == 0:
            return 0.0
        return np.mean(iou_list)

    # --- 随机SGD优化 ---
    best_offset = np.zeros(3)
    best_iou = evaluate_offset(best_offset)
    print(f"[INIT] offset: {best_offset}, IoU: {best_iou:.4f}")
    step = args.step
    for it in tqdm(range(args.max_iter)):
        # 随机扰动
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        candidate = best_offset + step * direction
        candidate_iou = evaluate_offset(candidate)
        if candidate_iou > best_iou:
            best_offset = candidate
            best_iou = candidate_iou
            print(f"[UPDATE] iter {it}: offset: {best_offset}, IoU: {best_iou:.4f}")
        if it % 10 == 0:
            step *= 0.9
    print(f"[RESULT] Best offset: {best_offset}, IoU: {best_iou:.4f}")
    # 可选：保存结果
    np.savez("fd_pose_offset_result.npz", offset=best_offset, iou=best_iou)

    # --- 应用offset并保存npy ---
    pose_files_sorted = sorted(list(ob_in_world_dir.glob("*.txt")), key=lambda f: int(f.stem))
    all_poses = []
    for pose_path in pose_files_sorted:
        T, flag = load_pose(pose_path)
        T[:3, 3] += best_offset
        q = R.from_matrix(T[:3, :3]).as_quat()
        t = T[:3, 3]
        arr = np.concatenate([q, t, [flag]])
        all_poses.append(arr)
    all_poses = np.stack(all_poses, axis=0)
    np.save(ob_in_world_dir / "ob_in_world_offset.npy", all_poses)
    print(f"[INFO] Saved offset-applied poses to {ob_in_world_dir / 'ob_in_world_offset.npy'}")

    # --- Also save offset-applied result into ob_in_cam txt files for each camera ---
    # Load extrinsics for all serials
    for frame_idx, pose_path in enumerate(pose_files_sorted):
        T_w, flag = load_pose(pose_path)
        # print(f"[DEBUG] {frame_idx} T_w: {T_w}, flag: {flag}")
        T_w[:3, 3] += best_offset
        q = R.from_matrix(T_w[:3, :3]).as_quat()
        t = T_w[:3, 3]
        # print(f"[DEBUG] {frame_idx} q: {q}, t: {t}, flag: {flag}")
        arr = np.concatenate([q, t, [flag]])
        # Save to txt file (overwrite original)
        out_path = ob_in_world_dir / f"{frame_idx:06d}.txt"
        np.savetxt(out_path, arr, fmt="%.8f")
    print(f"[INFO] Saved offset-applied ob_in_cam txts for all cameras.")

