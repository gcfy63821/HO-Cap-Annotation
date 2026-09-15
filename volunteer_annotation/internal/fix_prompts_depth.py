#!/usr/bin/env python3
"""Per-experiment depth-based prompt correction.

For each camera whose positive prompt points project inconsistently across
views (suspect cameras), replace their positive points with reprojected points
from anchor cameras that do pass the 3D consistency check.

Reads prompt JSONs from --prompts_dir.
Writes corrected JSONs to --out_dir (copies unmodified if no depth available).
Does NOT modify the source prompts.

Usage (standalone):
    python fix_prompts_depth.py \\
        --prompts_dir  /path/to/auto_prompts/videos_0202/task/exp/tool_masks/prompts \\
        --out_dir      /path/to/corrected_prompts/videos_0202/task/exp/tool_masks/prompts \\
        --bundle       /viscam/projects/robotool/_va_bundle_v2 \\
        --calib_root   /viscam/projects/robotool/calibrations \\
        --task         videos_0202/spoon_press_sponge \\
        [--role primary_tool] [--dry_run] [--min_anchors 2] [--suspect_thr 0.3]

Exit codes:
    0 — success (possibly 0 corrections if all prompts already consistent)
    2 — no depth PNGs found for this exp; prompts copied unmodified
    3 — calibration not found; prompts copied unmodified
"""

import argparse
import copy
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── path resolution ────────────────────────────────────────────────────────────

def _add_volunteer_to_path():
    """Add HO-Cap-Annotation/volunteer_annotation to sys.path."""
    here = Path(__file__).resolve()
    for p in here.parents:
        candidate = p / "volunteer_annotation"
        if candidate.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    sys.exit("[fix_prompts_depth] Cannot find volunteer_annotation/ in parent dirs")

_add_volunteer_to_path()

try:
    import cv2
    from depth_utils import load_depth
    from template_multiview_vote import load_calibration as _load_calib_default
    from verify_prompt_consistency import (
        unproject_point, project_point, POS_RADIUS_PX, SUSPECT_THR, SUSPECT_REL,
    )
except ImportError as e:
    sys.exit(f"[fix_prompts_depth] Import error: {e}\n"
             "Make sure hocap-annotation conda env is active.")

DEPTH_SNAP_RADIUS = 80    # px search radius around projected centroid
DEPTH_SNAP_TOL    = 0.05  # metres — depth tolerance for matching (5 cm)
DEPTH_SNAP_MIN_PX = 5     # minimum matching pixels to accept snap


# ── configurable calibration loader ───────────────────────────────────────────

def load_calibration(task: str, calib_root: Path) -> dict:
    """Like template_multiview_vote.load_calibration but with configurable root."""
    import yaml

    date_str = task.split("/")[0].replace("videos_", "")

    def _find(root: Path, date: str):
        for pattern in [f"videos_{date}/realsense_calibrate_*/",
                        f"realsense_calibrate_{date}/"]:
            dirs = sorted(root.glob(pattern))
            for d in dirs:
                yamls = sorted(d.glob("*_global_aligned.yaml"))
                if yamls:
                    return yamls[-1]
        # Wider search
        yamls = sorted(root.glob(f"**/*{date}*global_aligned.yaml"))
        return yamls[-1] if yamls else None

    yaml_path = _find(calib_root, date_str)
    if yaml_path is None:
        return {}

    entries = yaml.safe_load(yaml_path.read_text())
    result = {}
    for e in entries:
        cid = int(e["camera_id"])
        K = np.array(e["color_intrinsic_matrix"], dtype=np.float64)
        T = np.array(e["transformation"], dtype=np.float64)
        result[cid] = {"K": K, "T_c2w": T}
    return result


def cam_name_to_id(name: str) -> int:
    return int(name.replace("cam", "").replace("_rgb", ""))


# ── consistency scoring (inline, no import of score_exp to keep deps minimal) ─

def score_cameras(
    prompt_jsons: dict,   # {cam_id: parsed JSON}
    calib: dict,
    bundle_exp_dir: Path,
    role: str,
    pos_radius: int = POS_RADIUS_PX,
    suspect_thr: float = SUSPECT_THR,
    suspect_rel: float = SUSPECT_REL,
) -> dict:
    """
    Returns {cam_id: {"score": float, "is_suspect": bool}}.
    Score = fraction of projected positive points that land near another cam's prompts.
    """
    # Collect positive points per cam per frame
    cam_pts = {}   # cam_id -> {frame_index: [(x,y),...]}
    cam_hw  = {}   # cam_id -> (H, W)
    for cam_id, data in prompt_jsons.items():
        cam_pts[cam_id] = defaultdict(list)
        cam_hw[cam_id]  = (data["height"], data["width"])
        for obj in data.get("objects", []):
            if obj.get("role") != role:
                continue
            fi = obj["frame_index"]
            for pt, lbl in zip(obj["points"], obj["labels"]):
                if lbl == 1:
                    cam_pts[cam_id][fi].append((pt[0], pt[1]))

    cam_ids = sorted(set(cam_pts) & set(calib))
    if len(cam_ids) < 2:
        return {}

    # Build target masks per (cam, frame) for radius-based hit check
    target_masks = {}  # (cam_id, fi) -> bool (H, W)
    for cam_id in cam_ids:
        H, W = cam_hw[cam_id]
        for fi, pts in cam_pts[cam_id].items():
            canvas = np.zeros((H, W), dtype=np.uint8)
            for x, y in pts:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < W and 0 <= yi < H:
                    cv2.circle(canvas, (xi, yi), pos_radius, 1, -1)
            target_masks[(cam_id, fi)] = canvas.astype(bool)

    per_cam_frame_scores = defaultdict(list)   # cam_id -> [0..1 scores]

    for src in cam_ids:
        K_s = calib[src]["K"]
        T_s = calib[src]["T_c2w"]
        for fi, pts in cam_pts[src].items():
            dp = bundle_exp_dir / f"cam{src}_depth.kf{fi}.png"
            depth = load_depth(dp)
            if depth is None:
                continue

            # Unproject source positive points to 3D
            pts3d = []
            for x, y in pts:
                pt = unproject_point(x, y, depth, K_s, T_s)
                if pt is not None:
                    pts3d.append(pt)
            if not pts3d:
                continue

            # For each target camera, compute hit rate
            per_tgt_rates = []
            for tgt in cam_ids:
                if tgt == src:
                    continue
                mask_t = target_masks.get((tgt, fi))
                if mask_t is None:
                    continue
                H_t, W_t = cam_hw[tgt]
                hits = 0
                for pt3d in pts3d:
                    uv = project_point(pt3d, calib[tgt]["K"], calib[tgt]["T_c2w"], H_t, W_t)
                    if uv is None:
                        continue
                    xi, yi = int(round(uv[0])), int(round(uv[1]))
                    if 0 <= xi < W_t and 0 <= yi < H_t and mask_t[yi, xi]:
                        hits += 1
                per_tgt_rates.append(hits / len(pts3d))

            if per_tgt_rates:
                per_cam_frame_scores[src].append(float(np.mean(per_tgt_rates)))

    scores = {}
    for src in cam_ids:
        frame_scores = per_cam_frame_scores.get(src, [])
        scores[src] = float(np.mean(frame_scores)) if frame_scores else float("nan")

    # Determine suspect cams (mirrors verify_prompt_consistency.score_exp logic)
    valid = {c: s for c, s in scores.items() if not np.isnan(s)}
    group_mean = float(np.mean(list(valid.values()))) if valid else float("nan")

    result = {}
    for cam_id in cam_ids:
        s = scores.get(cam_id, float("nan"))
        if np.isnan(s):
            # No depth data for this cam — cannot evaluate, treat as ok (not suspect)
            is_suspect = False
        else:
            abs_suspect = s < suspect_thr
            rel_suspect = (not np.isnan(group_mean)) and s < group_mean * suspect_rel
            is_suspect = abs_suspect or rel_suspect
        result[cam_id] = {"score": s, "is_suspect": is_suspect}
    return result


# ── per-experiment correction ──────────────────────────────────────────────────

def fix_one_exp(
    prompts_dir: Path,
    out_dir: Path,
    bundle: Path,
    calib: dict,
    task: str,
    role: str,
    min_anchors: int,
    n_pos_sample: int,
    dry_run: bool,
    verbose: bool,
) -> tuple[int, int]:
    """
    Returns (n_corrected, n_copied_unchanged).
    Writes corrected/copied JSONs to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all prompt JSONs
    cam_jsons = {}
    cam_json_paths = {}
    for jf in sorted(prompts_dir.glob("cam*_rgb.json")):
        cam_id = cam_name_to_id(jf.stem)
        if cam_id not in calib:
            if verbose:
                print(f"  [skip calib] cam{cam_id}", file=sys.stderr)
            shutil.copy2(jf, out_dir / jf.name)
            continue
        cam_jsons[cam_id] = json.loads(jf.read_text())
        cam_json_paths[cam_id] = jf

    if not cam_jsons:
        return 0, 0

    # Extract exp name from prompts_dir path (…/<task>/<exp>/tool_masks/prompts)
    exp_name = prompts_dir.parts[-3]  # e.g. 20260202_bigwoodenspoon_...
    bundle_exp_dir = bundle / task / exp_name

    # Check depth availability
    has_depth = any(bundle_exp_dir.glob("cam*_depth.kf*.png"))
    if not has_depth:
        if verbose:
            print(f"  [no depth] {exp_name} — copying prompts unchanged", file=sys.stderr)
        for jf in cam_json_paths.values():
            shutil.copy2(jf, out_dir / jf.name)
        return 0, len(cam_json_paths)

    # Score cameras
    scores = score_cameras(cam_jsons, calib, bundle_exp_dir, role)
    if not scores:
        for jf in cam_json_paths.values():
            shutil.copy2(jf, out_dir / jf.name)
        return 0, len(cam_json_paths)

    suspect_cams = {c for c, info in scores.items() if info.get("is_suspect", False)}
    anchor_cams  = {c for c, info in scores.items() if not info.get("is_suspect", False)
                    and not np.isnan(info["score"])}

    if verbose:
        for cid, info in sorted(scores.items()):
            flag = "SUSPECT" if info["is_suspect"] else "ok"
            print(f"  cam{cid} score={info['score']:.3f} [{flag}]", file=sys.stderr)

    # Collect anchor positive points
    anchor_pts = defaultdict(dict)   # frame_index -> {cam_id: [(x,y),...]}
    for cam_id in anchor_cams:
        for obj in cam_jsons[cam_id].get("objects", []):
            if obj.get("role") != role:
                continue
            fi = obj["frame_index"]
            pts = [(p[0], p[1]) for p, l in zip(obj["points"], obj["labels"]) if l == 1]
            if pts:
                anchor_pts[fi][cam_id] = pts

    n_corrected = 0
    for cam_id, data in cam_jsons.items():
        is_suspect = cam_id in suspect_cams and cam_id in scores
        if not is_suspect or len(anchor_cams) < min_anchors:
            shutil.copy2(cam_json_paths[cam_id], out_dir / cam_json_paths[cam_id].name)
            continue

        H, W = data["height"], data["width"]
        K_S  = calib[cam_id]["K"]
        T_S  = calib[cam_id]["T_c2w"]
        modified = False
        new_objects = []

        for obj in data.get("objects", []):
            if obj.get("role") != role:
                new_objects.append(obj)
                continue

            fi = obj["frame_index"]
            anchors_at_frame = anchor_pts.get(fi, {})
            if len(anchors_at_frame) < min_anchors:
                new_objects.append(obj)
                continue

            # Unproject anchor points → 3D
            pts3d = []
            for cam_A, pts_A in anchors_at_frame.items():
                K_A = calib[cam_A]["K"]
                T_A = calib[cam_A]["T_c2w"]
                dp  = bundle_exp_dir / f"cam{cam_A}_depth.kf{fi}.png"
                depth_A = load_depth(dp)
                if depth_A is None:
                    continue
                for x, y in pts_A:
                    pt = unproject_point(x, y, depth_A, K_A, T_A)
                    if pt is not None:
                        pts3d.append(pt)

            if len(pts3d) < 2:
                new_objects.append(obj)
                continue

            # Project to suspect camera
            proj_uv = []
            for pt3d in pts3d:
                uv = project_point(pt3d, K_S, T_S, H, W)
                if uv is not None:
                    proj_uv.append(uv)

            if len(proj_uv) < 2:
                new_objects.append(obj)
                continue

            # Subsample
            if len(proj_uv) > n_pos_sample:
                step = max(1, len(proj_uv) // n_pos_sample)
                proj_uv = proj_uv[::step][:n_pos_sample]

            # Depth snap: find pixels near projected centroid whose depth matches
            # the consensus 3D point — avoids jumping to same-colour nearby objects
            P_world = np.median(pts3d, axis=0)
            # P_world → suspect-camera frame depth
            R_S_mat = np.asarray(T_S)[:3, :3]
            t_S_vec = np.asarray(T_S)[:3, 3]
            d_exp = float((R_S_mat.T @ (P_world - t_S_vec))[2])
            dp_s = bundle_exp_dir / f"cam{cam_id}_depth.kf{fi}.png"
            depth_S = load_depth(dp_s)
            if depth_S is not None and d_exp > 0:
                centroid = np.mean(proj_uv, axis=0)
                cx, cy = float(centroid[0]), float(centroid[1])
                r = DEPTH_SNAP_RADIUS
                x1 = max(0, int(cx - r)); y1 = max(0, int(cy - r))
                x2 = min(W, int(cx + r)); y2 = min(H, int(cy + r))
                if x2 - x1 >= 10 and y2 - y1 >= 10:
                    crop = depth_S[y1:y2, x1:x2]
                    dmask = (np.abs(crop - d_exp) < DEPTH_SNAP_TOL) & (crop > 0)
                    if dmask.sum() >= DEPTH_SNAP_MIN_PX:
                        ys, xs = np.where(dmask)
                        snapped = np.array([x1 + xs.mean(), y1 + ys.mean()])
                        if verbose:
                            print(f"    depth_snap cam{cam_id} f{fi}: "
                                  f"({cx:.0f},{cy:.0f})"
                                  f" -> ({snapped[0]:.0f},{snapped[1]:.0f})"
                                  f" d_exp={d_exp:.3f}m px={dmask.sum()}",
                                  file=sys.stderr)
                        proj_uv = [snapped]

            # Keep original negative points
            neg_pts = [(p[0], p[1]) for p, l in zip(obj["points"], obj["labels"]) if l == 0]

            new_obj = copy.deepcopy(obj)
            new_obj["points"] = [[float(x), float(y)] for x, y in proj_uv] + \
                                 [[float(x), float(y)] for x, y in neg_pts]
            new_obj["labels"] = [1] * len(proj_uv) + [0] * len(neg_pts)
            old_method = new_obj.get("method", "")
            if "+depth_corrected" not in old_method:
                new_obj["method"] = old_method + "+depth_corrected"

            new_objects.append(new_obj)
            modified = True
            n_corrected += 1
            if verbose:
                print(f"  [fix] cam{cam_id} frame={fi} "
                      f"{len(proj_uv)} pos pts from {len(anchors_at_frame)} anchors",
                      file=sys.stderr)

        out_path = out_dir / cam_json_paths[cam_id].name
        out_data = dict(data)
        out_data["objects"] = new_objects
        if not dry_run:
            out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2))
        elif verbose:
            print(f"  [dry_run] would write {out_path}", file=sys.stderr)

    return n_corrected, len(cam_jsons) - n_corrected


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts_dir", required=True,
                    help="source prompts dir (cam*_rgb.json)")
    ap.add_argument("--out_dir", required=True,
                    help="output corrected prompts dir (cam*_rgb.json)")
    ap.add_argument("--bundle", required=True,
                    help="root of _va_bundle_v2 (contains depth PNGs)")
    ap.add_argument("--calib_root", required=True,
                    help="root containing calibration yamls")
    ap.add_argument("--task", required=True,
                    help="task path e.g. videos_0202/spoon_press_sponge")
    ap.add_argument("--role", default="primary_tool")
    ap.add_argument("--min_anchors", type=int, default=2)
    ap.add_argument("--n_pos_sample", type=int, default=5,
                    help="max positive points to use per corrected camera")
    ap.add_argument("--suspect_thr", type=float, default=SUSPECT_THR)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    prompts_dir = Path(args.prompts_dir)
    out_dir     = Path(args.out_dir)
    bundle      = Path(args.bundle)
    calib_root  = Path(args.calib_root)

    if not prompts_dir.is_dir():
        sys.exit(f"[fix_prompts_depth] prompts_dir not found: {prompts_dir}")
    if not bundle.is_dir():
        sys.exit(f"[fix_prompts_depth] bundle not found: {bundle}")

    calib = load_calibration(args.task, calib_root)
    if not calib:
        print(f"[fix_prompts_depth] WARNING: no calibration found for {args.task} "
              f"in {calib_root} — copying prompts unmodified", file=sys.stderr)
        out_dir.mkdir(parents=True, exist_ok=True)
        for jf in sorted(prompts_dir.glob("cam*_rgb.json")):
            shutil.copy2(jf, out_dir / jf.name)
        sys.exit(3)

    n_fix, n_copy = fix_one_exp(
        prompts_dir=prompts_dir,
        out_dir=out_dir,
        bundle=bundle,
        calib=calib,
        task=args.task,
        role=args.role,
        min_anchors=args.min_anchors,
        n_pos_sample=args.n_pos_sample,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # Check if any depth was found (score_cameras returns {} if none)
    exp_name = prompts_dir.parts[-3]
    bundle_exp_dir = bundle / args.task / exp_name
    has_depth = any(bundle_exp_dir.glob("cam*_depth.kf*.png"))
    if not has_depth:
        sys.exit(2)

    print(f"[fix_prompts] {exp_name}: corrected={n_fix} unchanged={n_copy}")
    sys.exit(0)


if __name__ == "__main__":
    main()
