"""
多视角投票：把各相机的 2D 预测点通过相机外参三角化为 3D 点，
用迭代外点剔除找到 consensus 3D 位置，再重投影回各相机得到修正后的 2D 坐标。

优势：
  - 若某相机误匹配到别的工具，重投影误差大 → 被排除为 outlier
  - 对没有独立预测（被 SAM2 验证拒绝）的相机，可以从 consensus 补充位置
  - 有效消除跨视角的系统性误差
"""
import yaml
import numpy as np
from pathlib import Path
from functools import lru_cache

CALIB_ROOT = Path("/data/robotool/calibrations")

# 视频日期前缀 → 标定文件目录映射（支持 videos_0105_1 → videos_0105）
# 用函数动态查找，不硬编码


def _find_calib_yaml(date_str: str) -> Path | None:
    """在 CALIB_ROOT/videos_{date_str}/realsense_calibrate_*/ 下找 global_aligned.yaml。"""
    search_dirs = sorted(CALIB_ROOT.glob(f"videos_{date_str}/realsense_calibrate_*/"))
    # 对于 videos_0105_1 → 先尝试 0105_1，再尝试 0105
    if not search_dirs:
        base = date_str.split("_")[0]
        search_dirs = sorted(CALIB_ROOT.glob(f"videos_{base}*/realsense_calibrate_*/"))
    for d in search_dirs:
        yamls = sorted(d.glob("*_global_aligned.yaml"))
        if yamls:
            return yamls[-1]  # 优先最新版本（如 _1_global_aligned > _global_aligned）
    return None


@lru_cache(maxsize=32)
def load_calibration(task: str) -> dict:
    """
    从 task（如 "videos_0106/spoon_scoop_nuts"）加载对应的相机标定。

    返回 {camera_id: {"K": (3,3) ndarray, "T_c2w": (4,4) ndarray}}
    其中 camera_id = 0..7，对应 cam0_rgb..cam7_rgb。
    """
    date_str = task.split("/")[0].replace("videos_", "")  # "0106"
    yaml_path = _find_calib_yaml(date_str)
    if yaml_path is None:
        return {}

    entries = yaml.safe_load(yaml_path.read_text())
    result = {}
    for e in entries:
        cid = int(e["camera_id"])
        K   = np.array(e["color_intrinsic_matrix"], dtype=np.float64)
        T   = np.array(e["transformation"],         dtype=np.float64)
        result[cid] = {"K": K, "T_c2w": T}
    return result


def cam_name_to_id(cam: str) -> int:
    """'cam3_rgb' → 3"""
    return int(cam.replace("cam", "").replace("_rgb", ""))


# ─── 三角化 ───────────────────────────────────────────────────────────────────

def _make_ray(px: float, py: float, K: np.ndarray, T_c2w: np.ndarray):
    """返回 (origin, direction) 世界坐标系中的射线。"""
    O = T_c2w[:3, 3]
    K_inv = np.linalg.inv(K)
    d_cam = K_inv @ np.array([px, py, 1.0])
    d_world = T_c2w[:3, :3] @ d_cam
    d_world /= np.linalg.norm(d_world)
    return O, d_world


def triangulate_rays(rays: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    最小二乘三角化：最小化各射线到 3D 点的垂直距离之和。
    rays: [(origin, direction), ...]，direction 已归一化
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for O, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ O
    return np.linalg.lstsq(A, b, rcond=None)[0]


def project_to_image(P_world: np.ndarray, K: np.ndarray,
                     T_c2w: np.ndarray, img_hw=(720, 1280)):
    """将 3D 世界点重投影到图像坐标，越界时 clamp。返回 None 如果在相机后方。"""
    T_w2c = np.linalg.inv(T_c2w)
    P_cam = T_w2c[:3, :3] @ P_world + T_w2c[:3, 3]
    if P_cam[2] <= 0.01:
        return None
    uv = K @ P_cam
    uv = uv[:2] / uv[2]
    H, W = img_hw
    return np.clip(uv, [0, 0], [W - 1, H - 1])


# ─── 多视角投票 ───────────────────────────────────────────────────────────────

def multiview_vote(
    cam_predictions: dict[int, np.ndarray],  # {cam_id: [px, py]}
    calibration: dict,
    reproj_thresh: float = 40.0,   # 重投影误差阈值（像素）
    min_inliers: int = 3,          # 最少需要多少相机一致才接受结果
    img_hw: tuple = (720, 1280),
    max_iter: int = 4,
) -> tuple[np.ndarray | None, dict[int, np.ndarray], set[int]]:
    """
    多视角三角化 + 迭代外点剔除。

    Returns:
        P_world    : (3,) consensus 3D 点，失败时为 None
        reprojected: {cam_id: [px, py]} 所有相机的重投影坐标
        inlier_ids : 参与最终三角化的相机 id 集合
    """
    valid_cids = [cid for cid in cam_predictions if cid in calibration]
    if len(valid_cids) < 2:
        return None, {}, set()

    rays = {}
    for cid in valid_cids:
        pt = cam_predictions[cid]
        calib = calibration[cid]
        O, d = _make_ray(float(pt[0]), float(pt[1]), calib["K"], calib["T_c2w"])
        rays[cid] = (O, d)

    active = set(valid_cids)
    P_world = None

    for _ in range(max_iter):
        active_rays = [rays[cid] for cid in active]
        P = triangulate_rays(active_rays)

        # 计算每个相机的重投影误差
        errors = {}
        for cid in valid_cids:
            proj = project_to_image(P, calibration[cid]["K"],
                                    calibration[cid]["T_c2w"], img_hw)
            if proj is None:
                errors[cid] = 9999.0
            else:
                errors[cid] = float(np.linalg.norm(proj - cam_predictions[cid]))

        new_active = {cid for cid in valid_cids if errors[cid] < reproj_thresh}
        if len(new_active) < min_inliers:
            return None, {}, set()
        if new_active == active:
            P_world = P
            break
        active = new_active

    if P_world is None:
        P_world = triangulate_rays([rays[cid] for cid in active])

    # 重投影回所有相机（包括 outlier 相机，给它们修正后的坐标）
    reprojected = {}
    for cid, calib in calibration.items():
        proj = project_to_image(P_world, calib["K"], calib["T_c2w"], img_hw)
        if proj is not None:
            reprojected[cid] = proj

    return P_world, reprojected, active


# ─── 跨帧多视角投票（annotate_exp 使用） ──────────────────────────────────────

def vote_frame_predictions(
    frame_cam_preds: dict[int, np.ndarray],  # {cam_id: center_xy}
    task: str,
    reproj_thresh: float = 40.0,
    min_inliers: int = 3,
    img_hw: tuple = (720, 1280),
) -> tuple[dict[int, np.ndarray], set[int]]:
    """
    对单帧单 role 的多视角预测做投票。

    返回 (voted_pts, inlier_ids)：
      - voted_pts : {cam_id: center_xy}，成功时包含所有相机的重投影坐标；
                    失败时只含原始有预测的相机
      - inlier_ids: 参与三角化的相机 id 集合（失败时为空 set）
    """
    calib = load_calibration(task)
    if not calib or len(frame_cam_preds) < 2:
        return dict(frame_cam_preds), set()

    P, reprojected, inliers = multiview_vote(
        frame_cam_preds, calib,
        reproj_thresh=reproj_thresh,
        min_inliers=min_inliers,
        img_hw=img_hw,
    )

    if P is None:
        return dict(frame_cam_preds), set()

    # 合并原始预测 + 重投影（含从未预测到的相机）
    result = dict(frame_cam_preds)
    for cid, proj in reprojected.items():
        result[cid] = proj
    return result, inliers
