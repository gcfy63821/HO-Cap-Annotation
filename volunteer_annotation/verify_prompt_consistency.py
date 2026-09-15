"""
验证自动标注点 prompt 的跨视角3D一致性。

对每个实验，把各相机的正样本点通过 depth 反投影到 3D，
再投影到其他相机，检查落点是否和其他相机的正样本点区域一致。

Usage:
  python verify_prompt_consistency.py \
    --task videos_0202/spoon_press_sponge \
    [--max-exps N] [--role primary_tool] [--suspect-threshold 0.3] \
    [--out-csv /path/to/out.csv]
"""
import sys, json, argparse, csv
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from template_multiview_vote import load_calibration, cam_name_to_id
from depth_utils import load_depth

BUNDLE   = Path("/data/robotool/_va_bundle_v2")
AP_ROOT  = Path("/data/robotool/_va_bundle_v2_auto_prompts")

POS_RADIUS_PX  = 80   # 正样本点周围的圆半径，用于构建"目标区域"
SUSPECT_THR    = 0.3  # 一致性低于此值 → suspect（绝对阈值）
SUSPECT_REL    = 0.4  # 若某cam分数 < 其他cam均值 × SUSPECT_REL → suspect（相对阈值）


# ── 几何辅助 ──────────────────────────────────────────────────────────────────

def unproject_point(x: float, y: float, depth_m: np.ndarray,
                    K: np.ndarray, T_c2w: np.ndarray):
    """把单个 2D 像素点反投影到世界坐标系。返回 (3,) 或 None（depth 无效）。"""
    yi, xi = int(round(y)), int(round(x))
    H, W = depth_m.shape
    if not (0 <= yi < H and 0 <= xi < W):
        return None
    d = depth_m[yi, xi]
    if d < 0.05:
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Xc = (x - cx) / fx * d
    Yc = (y - cy) / fy * d
    pt_cam = np.array([Xc, Yc, d, 1.0], dtype=np.float64)
    pt_world = T_c2w @ pt_cam
    return pt_world[:3]


def project_point(pt3d: np.ndarray, K: np.ndarray,
                  T_c2w: np.ndarray, H: int, W: int):
    """把世界坐标系 3D 点投影到 2D 像素。返回 (x, y) 或 None（在相机后/外）。"""
    T_wc = np.linalg.inv(T_c2w)
    pt_h = np.array([*pt3d, 1.0], dtype=np.float64)
    pt_cam = T_wc @ pt_h
    Z = pt_cam[2]
    if Z <= 0:
        return None
    x = pt_cam[0] / Z * K[0, 0] + K[0, 2]
    y = pt_cam[1] / Z * K[1, 1] + K[1, 2]
    if not (0 <= x < W and 0 <= y < H):
        return None
    return float(x), float(y)


def build_target_mask(pos_pts, H: int, W: int, radius: int = POS_RADIUS_PX):
    """把正样本点集合渲染成二值 mask（每点画圆）。"""
    canvas = np.zeros((H, W), dtype=np.uint8)
    for x, y in pos_pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(canvas, (xi, yi), radius, 1, -1)
    return canvas.astype(bool)


# ── 核心：计算一个实验里某 role 的各相机一致性分数 ────────────────────────────

def score_exp(exp_dir: Path, task: str, role: str, calib: dict) -> dict:
    """
    返回 {cam_id: {"score": float, "frames_scored": int, "n_suspect_frames": int}}
    score = 平均一致性（0-1）
    """
    prompt_dir = AP_ROOT / task / exp_dir.name / "tool_masks" / "prompts"
    if not prompt_dir.exists():
        return {}

    # 收集各相机的 prompt JSON
    cam_jsons = {}
    for jf in prompt_dir.glob("cam*_rgb.json"):
        cam_id = cam_name_to_id(jf.stem)
        if cam_id not in calib:
            continue
        cam_jsons[cam_id] = json.loads(jf.read_text())

    if len(cam_jsons) < 2:
        return {}

    # 按 frame_index 分组，收集正样本点
    # frame_pos[frame][cam_id] = [(x, y), ...]
    frame_pos = defaultdict(dict)
    frame_hw  = {}  # (H, W) from JSON
    for cam_id, data in cam_jsons.items():
        H, W = data["height"], data["width"]
        frame_hw[cam_id] = (H, W)
        for obj in data.get("objects", []):
            if obj.get("role") != role:
                continue
            fi = obj["frame_index"]
            pts = [(p[0], p[1]) for p, l in zip(obj["points"], obj["labels"]) if l == 1]
            if pts:
                frame_pos[fi][cam_id] = pts

    if not frame_pos:
        return {}

    # 对每个 frame，对每个 cam_A：把点投影到其他 cam，看落入率
    cam_scores = defaultdict(list)  # cam_id → [per-frame consistency]

    for frame_idx, cam_pts in frame_pos.items():
        if len(cam_pts) < 2:
            continue

        # 加载 depth
        depths = {}
        for cam_id in cam_pts:
            dp = BUNDLE / task / exp_dir.name / f"cam{cam_id}_depth.kf{frame_idx}.png"
            d = load_depth(dp)
            if d is not None:
                depths[cam_id] = d

        if len(depths) < 2:
            continue  # 没有足够的 depth 数据

        # 预建 target mask（cam → bool mask）
        target_masks = {}
        for cam_id, pts in cam_pts.items():
            H, W = frame_hw.get(cam_id, (720, 1280))
            target_masks[cam_id] = build_target_mask(pts, H, W)

        # 对每个 cam_A 评分：它的3D点投影到其他相机后的命中率
        for cam_A, pts_A in cam_pts.items():
            if cam_A not in depths:
                continue
            c_A = calib[cam_A]
            K_A, T_A = c_A["K"], c_A["T_c2w"]
            depth_A = depths[cam_A]

            # 反投影到 3D
            pts3d = []
            for x, y in pts_A:
                pt = unproject_point(x, y, depth_A, K_A, T_A)
                if pt is not None:
                    pts3d.append(pt)

            if not pts3d:
                continue

            # 投影到每个 cam_B，检查落入 target_mask
            per_cam_hits = []
            for cam_B, mask_B in target_masks.items():
                if cam_B == cam_A or cam_B not in calib:
                    continue
                c_B = calib[cam_B]
                K_B, T_B = c_B["K"], c_B["T_c2w"]
                H_B, W_B = mask_B.shape

                hits = 0
                for pt3d in pts3d:
                    uv = project_point(pt3d, K_B, T_B, H_B, W_B)
                    if uv is None:
                        continue
                    xi, yi = int(round(uv[0])), int(round(uv[1]))
                    if 0 <= xi < W_B and 0 <= yi < H_B and mask_B[yi, xi]:
                        hits += 1

                rate = hits / len(pts3d) if pts3d else 0.0
                per_cam_hits.append(rate)

            if per_cam_hits:
                cam_scores[cam_A].append(float(np.mean(per_cam_hits)))

    # 计算每个相机的平均分
    raw = {}
    for cam_id in cam_jsons:
        scores = cam_scores.get(cam_id, [])
        raw[cam_id] = float(np.mean(scores)) if scores else float("nan")

    # 计算有效分数的均值，用于相对阈值
    valid_scores = [v for v in raw.values() if not np.isnan(v)]
    group_mean = float(np.mean(valid_scores)) if valid_scores else float("nan")

    result = {}
    for cam_id in cam_jsons:
        s = raw[cam_id]
        scores = cam_scores.get(cam_id, [])
        # suspect 判断：绝对阈值 OR 相对阈值（分数远低于组均值）
        if np.isnan(s):
            is_suspect = False
        else:
            abs_suspect = s < SUSPECT_THR
            rel_suspect = (not np.isnan(group_mean)) and s < group_mean * SUSPECT_REL
            is_suspect = abs_suspect or rel_suspect
        result[cam_id] = {
            "score": s,
            "frames_scored": len(scores),
            "group_mean": group_mean,
            "is_suspect": is_suspect,
        }
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",             required=True)
    ap.add_argument("--role",             default="primary_tool")
    ap.add_argument("--max-exps",         type=int, default=0)
    ap.add_argument("--suspect-threshold",type=float, default=SUSPECT_THR)
    ap.add_argument("--out-csv",          default="")
    args = ap.parse_args()

    calib = load_calibration(args.task)
    if not calib:
        print(f"[WARN] No calibration found for {args.task}")
        return

    task_ap = AP_ROOT / args.task
    if not task_ap.exists():
        print(f"[ERROR] No auto_prompts dir: {task_ap}")
        return

    exps = sorted([d for d in task_ap.iterdir() if d.is_dir()])
    if args.max_exps > 0:
        exps = exps[:args.max_exps]

    rows = []
    suspect_total = 0
    for exp_dir in exps:
        scores = score_exp(exp_dir, args.task, args.role, calib)
        for cam_id, info in scores.items():
            s = info["score"]
            is_suspect = info.get("is_suspect", False)
            if is_suspect:
                suspect_total += 1
            gm = info.get("group_mean", float("nan"))
            rows.append({
                "exp":           exp_dir.name,
                "cam_id":        cam_id,
                "score":         f"{s:.3f}" if not np.isnan(s) else "nan",
                "group_mean":    f"{gm:.3f}" if not np.isnan(gm) else "nan",
                "frames_scored": info["frames_scored"],
                "suspect":       is_suspect,
            })
            mark = " ← SUSPECT" if is_suspect else ""
            gm_str = f"  group_mean={gm:.3f}" if not np.isnan(gm) else ""
            print(f"  cam{cam_id}  score={s:.3f}  frames={info['frames_scored']}{gm_str}{mark}")

        if scores:
            print(f"{exp_dir.name}")

    print(f"\n总计: {len(exps)} exps, {suspect_total} suspect 相机")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["exp", "cam_id", "score", "frames_scored", "suspect"])
            w.writeheader()
            w.writerows(rows)
        print(f"CSV 写出: {args.out_csv}")


if __name__ == "__main__":
    main()
