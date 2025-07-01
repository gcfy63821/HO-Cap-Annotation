import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import yaml
from tqdm import tqdm

def load_extrinsics(yaml_path, serials):
    with open(yaml_path, 'r') as f:
        extrinsics = yaml.safe_load(f)['extrinsics']

    def create_mat(values):
        return np.array([
            values[0:4],
            values[4:8],
            values[8:12],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    extrinsic_mats = {s: create_mat(extrinsics[s]) for s in serials}
    return extrinsic_mats

def depth_to_point_cloud(depth, K, extrinsic, depth_scale=1000.0):
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 生成像素坐标
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float32) / depth_scale
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    valid = (z > 0).reshape(-1)

    points = points[valid]

    # 加上齐次坐标再转换到世界坐标系
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points, ones))
    points_world = (extrinsic @ points_hom.T).T[:, :3]

    return points_world

def visualize_multi_camera_depth(
    depth_dirs,                  # {"00": Path(...), "01": Path(...), ...}
    extrinsics_yaml,
    Ks,                          # {"00": np.ndarray(3x3), ...}
    serials,                     # ["00", ..., "07"]
    frame_id=0,
    depth_scale=1000.0
):
    all_points = []

    # 加载相机外参
    extrinsics = load_extrinsics(extrinsics_yaml, serials)

    for serial in tqdm(serials):
        depth_path = depth_dirs[serial] / f"depth_{frame_id:06d}.png"
        if not depth_path.exists():
            print(f"[WARNING] missing: {depth_path}")
            continue

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"[ERROR] failed to read {depth_path}")
            continue

        K = Ks[serial]
        ext = extrinsics[serial]

        points = depth_to_point_cloud(depth, K, ext, depth_scale=depth_scale)
        all_points.append(points)

    all_points = np.concatenate(all_points, axis=0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
    pcd.paint_uniform_color([0.1, 0.7, 0.9])
    o3d.visualization.draw_geometries([pcd])

# ========== 示例调用 ==========
if __name__ == "__main__":
    base_dir = Path("/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250626_203130")
    serials = [f"{i:02d}" for i in range(8)]

    depth_dirs = {s: base_dir / s for s in serials}
    Ks = {
        s: np.array([[615.0, 0.0, 320.0],
                     [0.0, 615.0, 240.0],
                     [0.0, 0.0, 1.0]]) for s in serials
    }
    extrinsics_yaml = base_dir.parent / "../calibration/extrinsics/extrinsics.yaml"

    visualize_multi_camera_depth(
        depth_dirs=depth_dirs,
        extrinsics_yaml=extrinsics_yaml,
        Ks=Ks,
        serials=serials,
        frame_id=0,                # 选择你要查看的帧
        depth_scale=1000.0         # 若为 mm，设为 1000
    )
