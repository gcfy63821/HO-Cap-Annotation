"""
修正跨视角3D一致性差的相机 prompt。

对 suspect 相机（一致性 < suspect_threshold）：
  1. 找 anchor 相机（一致性 > min_anchor_score）
  2. 把 anchor 相机的正样本点 unproject → 3D → project 到 suspect 相机
  3. 用投影点替换 suspect 相机的正样本点（保留负样本点不变）
  4. 在 method 字段追加 +depth_verify_corrected
  5. 原地更新 cam*.json

Usage:
  python correct_prompt_consistency.py \
    --task videos_0202/spoon_press_sponge \
    [--max-exps N] [--role primary_tool] [--dry-run] \
    [--suspect-threshold 0.3] [--min-anchor-score 0.5] \
    [--min-anchors 2]
"""
import sys, json, argparse, copy
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from template_multiview_vote import load_calibration, cam_name_to_id
from depth_utils import load_depth
from verify_prompt_consistency import (
    score_exp, unproject_point, project_point, SUSPECT_THR, POS_RADIUS_PX, SUSPECT_REL
)

BUNDLE   = Path("/data/robotool/_va_bundle_v2")
AP_ROOT  = Path("/data/robotool/_va_bundle_v2_auto_prompts")

MIN_ANCHOR_SCORE = 0.5
MIN_ANCHORS      = 2
N_POS_SAMPLE     = 5   # 从 anchor 投影点中最多采样多少个正样本


# ── 修正单个实验的一个 frame/role ─────────────────────────────────────────────

def correct_exp(exp_dir: Path, task: str, role: str, calib: dict,
                suspect_thr: float, min_anchor_score: float, min_anchors: int,
                dry_run: bool) -> int:
    """返回修正的 (cam, frame) 数量。"""
    prompt_dir = AP_ROOT / task / exp_dir.name / "tool_masks" / "prompts"
    if not prompt_dir.exists():
        return 0

    # 计算一致性分数
    scores = score_exp(exp_dir, task, role, calib)
    if not scores:
        return 0

    # suspect: 用 score_exp 计算的 is_suspect 字段（已含相对阈值逻辑）
    suspect_cams = {c for c, info in scores.items() if info.get("is_suspect", False)}

    # anchor: 有效分数中高于均值的相机（相对选取，不用固定阈值）
    valid = {c: info["score"] for c, info in scores.items()
             if not np.isnan(info["score"]) and not info.get("is_suspect", False)}
    anchor_cams = set(valid.keys())  # 非 suspect 且有有效分数的即为 anchor

    if not suspect_cams or len(anchor_cams) < min_anchors:
        return 0

    # 读取所有 cam JSON
    cam_jsons = {}
    cam_json_paths = {}
    for jf in prompt_dir.glob("cam*_rgb.json"):
        cam_id = cam_name_to_id(jf.stem)
        if cam_id not in calib:
            continue
        data = json.loads(jf.read_text())
        cam_jsons[cam_id] = data
        cam_json_paths[cam_id] = jf

    # 收集 anchor 正样本 (按 frame_index)
    # anchor_pts[frame][cam_id] = [(x,y), ...]
    anchor_pts = defaultdict(dict)
    for cam_id in anchor_cams:
        data = cam_jsons.get(cam_id)
        if data is None:
            continue
        for obj in data.get("objects", []):
            if obj.get("role") != role:
                continue
            fi = obj["frame_index"]
            pts = [(p[0], p[1]) for p, l in zip(obj["points"], obj["labels"]) if l == 1]
            if pts:
                anchor_pts[fi][cam_id] = pts

    n_corrected = 0

    for cam_S in suspect_cams:
        data_S = cam_jsons.get(cam_S)
        if data_S is None:
            continue
        H, W = data_S["height"], data_S["width"]
        c_S = calib[cam_S]
        K_S, T_S = c_S["K"], c_S["T_c2w"]

        modified = False
        new_objects = []
        for obj in data_S.get("objects", []):
            if obj.get("role") != role:
                new_objects.append(obj)
                continue

            fi = obj["frame_index"]
            anchors_at_frame = anchor_pts.get(fi, {})
            if len(anchors_at_frame) < min_anchors:
                new_objects.append(obj)
                continue

            # 从所有 anchor 相机收集 3D 点
            pts3d = []
            for cam_A, pts_A in anchors_at_frame.items():
                if cam_A not in calib:
                    continue
                c_A = calib[cam_A]
                K_A, T_A = c_A["K"], c_A["T_c2w"]
                dp = BUNDLE / task / exp_dir.name / f"cam{cam_A}_depth.kf{fi}.png"
                depth_A = load_depth(dp)
                if depth_A is None:
                    continue
                for x, y in pts_A:
                    pt = unproject_point(x, y, depth_A, K_A, T_A)
                    if pt is not None:
                        pts3d.append(pt)

            if len(pts3d) < 3:
                new_objects.append(obj)
                continue

            # 投影到 suspect 相机
            proj_uv = []
            for pt3d in pts3d:
                uv = project_point(pt3d, K_S, T_S, H, W)
                if uv is not None:
                    proj_uv.append(uv)

            if len(proj_uv) < 2:
                new_objects.append(obj)
                continue

            # 均匀采样 N_POS_SAMPLE 个投影点
            if len(proj_uv) > N_POS_SAMPLE:
                step = len(proj_uv) // N_POS_SAMPLE
                proj_uv = proj_uv[::step][:N_POS_SAMPLE]

            # 保留原负样本点
            neg_pts = [(p[0], p[1]) for p, l in zip(obj["points"], obj["labels"]) if l == 0]

            new_obj = copy.deepcopy(obj)
            new_obj["points"] = [[float(x), float(y)] for x, y in proj_uv] + \
                                 [[float(x), float(y)] for x, y in neg_pts]
            new_obj["labels"] = [1] * len(proj_uv) + [0] * len(neg_pts)
            old_method = new_obj.get("method", "")
            if "+depth_verify_corrected" not in old_method:
                new_obj["method"] = old_method + "+depth_verify_corrected"

            new_objects.append(new_obj)
            modified = True
            n_corrected += 1
            print(f"  [修正] {exp_dir.name} cam{cam_S} frame={fi}  "
                  f"{len(proj_uv)} pos pts from {len(anchors_at_frame)} anchors")

        if modified and not dry_run:
            updated = dict(data_S)
            updated["objects"] = new_objects
            cam_json_paths[cam_S].write_text(
                json.dumps(updated, ensure_ascii=False, indent=2)
            )

    return n_corrected


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",              required=True)
    ap.add_argument("--role",              default="primary_tool")
    ap.add_argument("--max-exps",          type=int, default=0)
    ap.add_argument("--suspect-threshold", type=float, default=SUSPECT_THR)
    ap.add_argument("--min-anchor-score",  type=float, default=MIN_ANCHOR_SCORE)
    ap.add_argument("--min-anchors",       type=int,   default=MIN_ANCHORS)
    ap.add_argument("--dry-run",           action="store_true")
    args = ap.parse_args()

    calib = load_calibration(args.task)
    if not calib:
        print(f"[WARN] No calibration for {args.task}, skipping.")
        return

    task_ap = AP_ROOT / args.task
    if not task_ap.exists():
        print(f"[ERROR] No auto_prompts dir: {task_ap}")
        return

    exps = sorted([d for d in task_ap.iterdir() if d.is_dir()])
    if args.max_exps > 0:
        exps = exps[:args.max_exps]

    total_corrected = 0
    for exp_dir in exps:
        n = correct_exp(exp_dir, args.task, args.role, calib,
                        args.suspect_threshold, args.min_anchor_score,
                        args.min_anchors, args.dry_run)
        total_corrected += n

    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{mode}总计修正: {total_corrected} 个 (cam, frame) in {len(exps)} exps")


if __name__ == "__main__":
    main()
