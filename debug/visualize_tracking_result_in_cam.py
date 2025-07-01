import os
import cv2
import numpy as np
from pathlib import Path
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def load_pose(pose_txt):
    with open(pose_txt, 'r') as f:
        arr = np.array([float(x) for x in f.read().strip().split()])
        t = np.array(arr[4:7])  # translation
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


def load_all_poses(pose_folder, num_frames):
    poses = []
    for i in range(num_frames):
        pose_path = pose_folder / f"{i:06d}.txt"
        if not pose_path.exists():
            poses.append(None)
            continue
        poses.append(load_pose(pose_path))
    return poses

def linear_interpolate_poses(poses, num_frames):
    """
    对于缺失的pose 使用线性插值填充。
    """
    interpolated_poses = []
    for i in range(num_frames):
        if poses[i] is not None:
            interpolated_poses.append(poses[i])
        else:
            # 找到前后非空的pose进行插值
            prev_pose = None
            next_pose = None
            for j in range(i - 1, -1, -1):
                if poses[j] is not None:
                    prev_pose = poses[j]
                    break
            for j in range(i + 1, num_frames):
                if poses[j] is not None:
                    next_pose = poses[j]
                    break
            if prev_pose is not None and next_pose is not None:
                interp_pose = (prev_pose + next_pose) / 2
                interpolated_poses.append(interp_pose)
            else:
                interpolated_poses.append(None)
    return interpolated_poses

def visualize_tracking(
    color_root, sam_mask_root,
    pose_root_pre, 
    mesh_path, K, output_path,
    serial="00", num_frames=300,
    outlier_idxs=[]
):
    color_root = Path(color_root)
    sam_mask_root = Path(sam_mask_root)
    pose_root_pre = Path(pose_root_pre)
    # pose_root_post = Path(pose_root_post)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    orig_mesh = trimesh.load(mesh_path, process=False)
    orig_mesh.vertices *= 0.001
    to_origin, extents = trimesh.bounds.oriented_bounds(orig_mesh)
    # orig_mesh.apply_transform(to_origin)
    orig_vertices = orig_mesh.vertices.copy()

    all_pre_optim_poses = load_all_poses(pose_root_pre, num_frames)
    all_post_optim_poses = linear_interpolate_poses(
        all_pre_optim_poses, num_frames
    )  # 如果没有后处理的pose，这里可以直接用前处理的

    video_out = None

    for i in tqdm(range(num_frames)):
        color_path = color_root / f"color_{i:06d}.jpg"
        mask_path = sam_mask_root / f"mask_{i:06d}.png"
        if not color_path.exists() or not mask_path.exists():
            print(f"[WARNING] Frame {i} missing: {color_path} or {mask_path}")
            continue

        color = cv2.imread(str(color_path))
        sam = color.copy()
        sam_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        sam[sam_mask > 0] = [0, 0, 255]

        H, W = color.shape[:2]
        if video_out is None:
            video_out = cv2.VideoWriter(
                str(output_path / "tracking_result.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'), 20, (W * 2, H * 2)
            )

        # ---------- 优化前重投影 ----------
        mesh = orig_mesh.copy()
        mesh.vertices = orig_vertices.copy()
        T_pre = all_pre_optim_poses[i]
        if T_pre is not None:
            mesh.apply_transform(T_pre)
            pts = project_points(mesh.vertices, K)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) &
                      (pts[:, 1] >= 0) & (pts[:, 1] < H)]
            pts = pts.astype(np.int32)[::200]
            outlier_frame = color.copy()
            for x, y in pts:
                c = (0, 0, 255) if i in outlier_idxs else (255, 0, 0)
                cv2.circle(outlier_frame, (x, y), 2, c, -1)
        else:
            outlier_frame = np.ones_like(color) * 255

        # ---------- 优化后重投影 ----------
        mesh = orig_mesh.copy()
        mesh.vertices = orig_vertices.copy()
        T_post = all_post_optim_poses[i]
        if T_post is not None:
            mesh.apply_transform(T_post)
            pts = project_points(mesh.vertices, K)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W) &
                      (pts[:, 1] >= 0) & (pts[:, 1] < H)]
            pts = pts.astype(np.int32)[::200]
            corr_outlier_frame = color.copy()
            for x, y in pts:
                c = (0, 0, 255) if i in outlier_idxs else (255, 0, 0)
                cv2.circle(corr_outlier_frame, (x, y), 2, c, -1)
        else:
            corr_outlier_frame = np.ones_like(color) * 255

        # 拼图（2x2）
        top = np.concatenate((color, sam), axis=1)
        bottom = np.concatenate((outlier_frame, corr_outlier_frame), axis=1)
        final = np.concatenate((top, bottom), axis=0)

        video_out.write(final)

    if video_out is not None:
        video_out.release()
        print(f"[INFO] Video saved to {output_path / 'tracking_result.mp4'}")


def concat_videos_grid(video_paths, output_path, grid_shape=(2, 4)):
    """
    将多个视频拼接成grid大视频，grid_shape如(2,4)
    video_paths: list of 8 paths
    output_path: 输出大视频路径
    """
    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    assert all([c.isOpened() for c in caps]), "有视频无法打开"
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
    for _ in range(min_frames):
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
    print(f"[INFO] 拼接大视频已保存到 {output_path}")


if __name__ == "__main__":
    serials = [f"{i:02d}" for i in range(8)]
    video_paths = []
    for serial in serials:
        out_dir = Path(f"debug_output/wooden_spoon_cam{serial}")
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / "tracking_result.mp4"
        video_paths.append(video_path)
        # if not video_path.exists():
        #     visualize_tracking(
        #         color_root=Path(f"/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250626_203130/{serial}"),
        #         sam_mask_root=Path(f"/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250626_203130/processed/segmentation/sam2/{serial}/mask"),
        #         pose_root_pre=Path(f"/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250626_203130/processed/fd_pose_solver/wooden_spoon/ob_in_cam/{serial}"),
        #         mesh_path="/home/wys/learning-compliant/crq_ws/data/decim_mesh_files_1/decim_mesh_files/wooden_spoon.obj",
        #         K=np.array([[607.4, 0.0, 320.0],
        #                     [0.0, 607.4, 240.0],
        #                     [0.0, 0.0, 1.0]]),
        #         output_path=out_dir,
        #         serial=serial,
        #         num_frames=246,
        #         outlier_idxs=[]
        #     )
    # 拼接2x4大视频
    concat_videos_grid(video_paths, output_path="debug_output/wooden_spoon_2x4.mp4", grid_shape=(2, 4))
