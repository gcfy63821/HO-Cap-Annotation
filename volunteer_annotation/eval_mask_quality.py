#!/usr/bin/env python3
"""
Multi-view mask consistency evaluator.

For each auto-annotated experiment:
  1. Decode SAM2 masks from all cameras for each keyframe
  2. Triangulate tool centroid in 3D using DLT
  3. Project 3D centroid back to each camera
  4. Measure consistency: is the projected point inside each camera's mask?
  5. Also checks: area ratio outliers, SAM2 score variance, border color leakage

Usage:
  # Evaluate all experiments in a task:
  python eval_mask_quality.py --task videos_0218/rollingpin_roll_sand [--role primary_tool]

  # Evaluate one experiment:
  python eval_mask_quality.py --exp 20260219_rollingpin_roll_large_sand_in_cuttingboard_1 \
                               --task videos_0218/rollingpin_roll_sand

  # Compute IoU against human annotations (template experiments):
  python eval_mask_quality.py --task videos_0218/rollingpin_roll_sand --vs-human
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cloud"))

BUNDLE_DIR = Path("/data/robotool/_va_bundle_v2")
AUTO_DIR = Path("/data/robotool/_va_bundle_v2_auto_prompts")
HUMAN_DIR = Path("/data/robotool/_va_bundle_v2_prompts")
CALIB_DIR = Path("/data/robotool/calibrations")

CAMERAS = [f"cam{i}_rgb" for i in range(8)]


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def _find_calib_yaml(date_tag: str) -> Optional[Path]:
    """Find the global-aligned calibration YAML for a date tag like 'videos_0218'."""
    calib_root = CALIB_DIR / date_tag
    if not calib_root.exists():
        return None
    for sub in calib_root.iterdir():
        for f in sub.glob("*global_aligned*.yaml"):
            return f
        for f in sub.glob("*.yaml"):
            return f
    return None


def load_calibration(date_tag: str) -> Optional[dict]:
    """
    Returns dict: {cam_id (int) → {"K": (3,3), "T_cw": (4,4)}}
    T_cw = camera-to-world (extr2world convention from my_sequence_loader.py).
    T_wc = inv(T_cw) is what we use for projection.
    """
    f = _find_calib_yaml(date_tag)
    if f is None:
        return None
    with open(f) as fh:
        data = yaml.safe_load(fh)
    out = {}
    for cam in data:
        cid = int(cam["camera_id"])
        K = np.array(cam["color_intrinsic_matrix"], dtype=np.float64)
        T_cw = np.array(cam["transformation"], dtype=np.float64)  # cam-to-world
        out[cid] = {"K": K, "T_cw": T_cw, "T_wc": np.linalg.inv(T_cw)}
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3D geometry
# ──────────────────────────────────────────────────────────────────────────────

def triangulate_dlt(
    points_2d: list[tuple[float, float]],
    Ks: list[np.ndarray],
    T_wcs: list[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Linear triangulation (DLT) from N ≥ 2 2D observations.

    Each camera i gives two equations: cross(x_i, P_i @ X) = 0
    where P_i = K_i @ T_wc_i[:3, :]  (3×4 projection matrix).

    Returns the 3D point in world coordinates, or None if degenerate.
    """
    if len(points_2d) < 2:
        return None
    rows = []
    for (u, v), K, T_wc in zip(points_2d, Ks, T_wcs):
        P = K @ T_wc[:3, :]  # (3, 4)
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows, axis=0)  # (2N, 4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return None
    return (X[:3] / X[3]).astype(np.float64)


def project_point(
    P_world: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
) -> tuple[float, float]:
    """Project 3D world point to image coordinates using world-to-camera T_wc."""
    X = np.append(P_world, 1.0)
    x_cam = T_wc @ X            # (4,)
    x_img = K @ x_cam[:3]       # (3,)
    return float(x_img[0] / x_img[2]), float(x_img[1] / x_img[2])


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_frame(
    exp_dir: Path,
    frame_idx: int,
    prompts_by_cam: dict[int, dict],  # cam_id → annotation object (one role)
    calib: dict,
    decoder,
    role: str,
    color_matcher=None,
) -> dict:
    """
    Returns a quality-metrics dict for one frame across all cameras:
      - area_ratio_min: min_cam_area / median_cam_area  (< 0.5 = suspect)
      - proj_inside_frac: fraction of cameras where reprojected 3D centroid is inside mask
      - proj_dist_mean: mean pixel distance of reprojected centroid from mask centroid
      - sam2_score_min: min SAM2 confidence across cameras
      - border_tool_frac_max: max border-color-leakage across cameras (if color_matcher given)
    """
    cam_ids = sorted(prompts_by_cam.keys())
    if len(cam_ids) < 2:
        return {}

    masks = {}
    centroids = {}
    areas = {}
    scores = {}

    for cid in cam_ids:
        cam_name = f"cam{cid}_rgb"
        obj = prompts_by_cam[cid]
        embed = exp_dir / f"{cam_name}.kf{frame_idx}.embed.npz"
        if not embed.exists():
            continue
        pts = obj["points"]
        lbls = obj["labels"]
        mask, score = decoder.infer(embed, pts, lbls)
        masks[cid] = mask
        scores[cid] = score
        H, W = mask.shape
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        areas[cid] = int(mask.sum())
        centroids[cid] = (float(xs.mean()), float(ys.mean()))

    if len(centroids) < 2:
        return {}

    # ── Area consistency ──────────────────────────────────────────────────────
    area_vals = list(areas.values())
    med_area = float(np.median(area_vals))
    area_ratio_min = min(area_vals) / med_area if med_area > 0 else 0.0

    # ── Multi-view centroid triangulation & reprojection ─────────────────────
    proj_inside = []
    proj_dists = []

    valid_cids = [c for c in cam_ids if c in centroids and c in calib]
    if len(valid_cids) >= 2:
        pts2d = [centroids[c] for c in valid_cids]
        Ks    = [calib[c]["K"] for c in valid_cids]
        T_wcs = [calib[c]["T_wc"] for c in valid_cids]

        P3d = triangulate_dlt(pts2d, Ks, T_wcs)

        if P3d is not None:
            for cid in cam_ids:
                if cid not in masks or cid not in calib:
                    continue
                u, v = project_point(P3d, calib[cid]["K"], calib[cid]["T_wc"])
                mask = masks[cid]
                H, W = mask.shape
                ui, vi = int(round(u)), int(round(v))
                inside = (0 <= ui < W and 0 <= vi < H and mask[vi, ui])
                proj_inside.append(float(inside))
                if cid in centroids:
                    cx, cy = centroids[cid]
                    proj_dists.append(float(np.hypot(u - cx, v - cy)))

    # ── Border color leakage ─────────────────────────────────────────────────
    border_fracs = []
    if color_matcher is not None:
        for cid, mask in masks.items():
            img_path = exp_dir / f"cam{cid}_rgb.kf{frame_idx}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            color_sc = color_matcher.score_image(img)  # (H,W) lower=more tool-like
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            border = dilated & ~mask
            if border.sum() == 0:
                continue
            border_sc = color_sc[border]
            frac = float((border_sc < 1.05).mean())  # tool-colored border pixels
            border_fracs.append(frac)

    result = {
        "frame": frame_idx,
        "n_cams": len(cam_ids),
        "area_ratio_min": round(area_ratio_min, 3),
        "sam2_score_min": round(min(scores.values()), 3) if scores else 0.0,
        "sam2_score_mean": round(float(np.mean(list(scores.values()))), 3) if scores else 0.0,
    }
    if proj_inside:
        result["proj_inside_frac"] = round(float(np.mean(proj_inside)), 3)
        result["proj_dist_mean_px"] = round(float(np.mean(proj_dists)), 1) if proj_dists else None
    if border_fracs:
        result["border_tool_frac_max"] = round(max(border_fracs), 3)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Per-experiment evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_experiment(
    task: str,
    exp_name: str,
    role: str,
    calib: dict,
    decoder,
    color_matcher=None,
    use_human: bool = False,
) -> Optional[dict]:
    """
    Evaluate mask quality for one experiment across all keyframes.
    Returns summary dict or None if no annotations found.
    """
    prompt_root = HUMAN_DIR if use_human else AUTO_DIR
    exp_dir = BUNDLE_DIR / task / exp_name

    # Load prompt JSONs for each camera
    cam_data: dict[int, list[dict]] = {}
    for cid in range(8):
        cam_name = f"cam{cid}_rgb"
        pf = prompt_root / task / exp_name / "tool_masks" / "prompts" / f"{cam_name}.json"
        if not pf.exists():
            continue
        objs = [o for o in json.loads(pf.read_text()).get("objects", [])
                if o.get("role") == role]
        if objs:
            cam_data[cid] = objs

    if len(cam_data) < 2:
        return None

    # Collect all keyframe indices
    all_frames = set()
    for objs in cam_data.values():
        for obj in objs:
            all_frames.add(obj.get("frame_index", 0))

    frame_results = []
    for frame_idx in sorted(all_frames):
        prompts_by_cam = {}
        for cid, objs in cam_data.items():
            frame_objs = [o for o in objs if o.get("frame_index", 0) == frame_idx]
            if frame_objs:
                prompts_by_cam[cid] = frame_objs[0]

        if len(prompts_by_cam) < 2:
            continue

        m = eval_frame(exp_dir, frame_idx, prompts_by_cam, calib, decoder, role, color_matcher)
        if m:
            frame_results.append(m)

    if not frame_results:
        return None

    # Aggregate over frames
    def _mean(key):
        vals = [r[key] for r in frame_results if key in r]
        return round(float(np.mean(vals)), 3) if vals else None

    agg = {
        "exp": exp_name,
        "n_frames": len(frame_results),
        "area_ratio_min_mean": _mean("area_ratio_min"),
        "sam2_score_min_mean": _mean("sam2_score_min"),
        "sam2_score_mean": _mean("sam2_score_mean"),
        "proj_inside_frac_mean": _mean("proj_inside_frac"),
        "proj_dist_mean_px": _mean("proj_dist_mean_px"),
        "border_tool_frac_max_mean": _mean("border_tool_frac_max"),
        "frames": frame_results,
    }
    # Quality flag: suspect if any metric is bad
    suspect = False
    if agg["area_ratio_min_mean"] is not None and agg["area_ratio_min_mean"] < 0.45:
        suspect = True
    if agg["proj_inside_frac_mean"] is not None and agg["proj_inside_frac_mean"] < 0.7:
        suspect = True
    if agg["border_tool_frac_max_mean"] is not None and agg["border_tool_frac_max_mean"] > 0.3:
        suspect = True
    agg["suspect"] = suspect
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# IoU vs human annotations
# ──────────────────────────────────────────────────────────────────────────────

def compute_iou_vs_human(
    task: str,
    exp_name: str,
    role: str,
    decoder,
) -> Optional[dict]:
    """
    For experiments with both human and auto annotations, compute per-camera IoU.
    Returns None if no human annotation exists.
    """
    results = []
    for cid in range(8):
        cam = f"cam{cid}_rgb"
        human_pf = HUMAN_DIR / task / exp_name / "tool_masks" / "prompts" / f"{cam}.json"
        auto_pf  = AUTO_DIR  / task / exp_name / "tool_masks" / "prompts" / f"{cam}.json"
        if not human_pf.exists() or not auto_pf.exists():
            continue
        human_objs = [o for o in json.loads(human_pf.read_text()).get("objects", []) if o.get("role") == role]
        auto_objs  = [o for o in json.loads(auto_pf.read_text()).get("objects", []) if o.get("role") == role]
        if not human_objs or not auto_objs:
            continue
        # Match by frame_index
        human_by_frame = {o.get("frame_index", 0): o for o in human_objs}
        auto_by_frame  = {o.get("frame_index", 0): o for o in auto_objs}
        for fi in sorted(set(human_by_frame) & set(auto_by_frame)):
            embed = BUNDLE_DIR / task / exp_name / f"{cam}.kf{fi}.embed.npz"
            if not embed.exists():
                continue
            ho = human_by_frame[fi]
            ao = auto_by_frame[fi]
            mask_h, _ = decoder.infer(embed, ho["points"], ho["labels"])
            mask_a, _ = decoder.infer(embed, ao["points"], ao["labels"])
            inter = int((mask_h & mask_a).sum())
            union = int((mask_h | mask_a).sum())
            iou = inter / union if union > 0 else 0.0
            results.append({"cam": cid, "frame": fi, "iou": round(iou, 4)})

    if not results:
        return None
    ious = [r["iou"] for r in results]
    return {
        "exp": exp_name,
        "mean_iou": round(float(np.mean(ious)), 4),
        "min_iou": round(float(np.min(ious)), 4),
        "n_pairs": len(results),
        "details": results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="e.g. videos_0218/rollingpin_roll_sand")
    ap.add_argument("--exp", help="Single experiment name; if omitted, all auto-annotated")
    ap.add_argument("--role", default="primary_tool")
    ap.add_argument("--vs-human", action="store_true",
                    help="Compute IoU vs human annotations instead of multi-view check")
    ap.add_argument("--with-color", action="store_true",
                    help="Also run border-color-leakage check (needs color model)")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Show top-k suspect experiments")
    ap.add_argument("--out", help="Save full results to JSON file")
    args = ap.parse_args()

    from cloud.decoder import Sam2CpuDecoder
    decoder = Sam2CpuDecoder()

    date_tag = args.task.split("/")[0]  # e.g. videos_0218
    calib = load_calibration(date_tag)
    if calib is None:
        print(f"WARNING: no calibration found for {date_tag}, skipping 3D checks")
        calib = {}

    color_matcher = None
    if args.with_color:
        from template_auto_annotate import ColorMatcher, MODEL_DIR, exp_keyword
        task_slug = args.task.replace("/", "_")
        model_dir = MODEL_DIR / task_slug
        # Try to find a color model for this task
        meta_files = list(model_dir.glob("*.meta.json"))
        if meta_files:
            meta = json.loads(meta_files[0].read_text())
            role_meta = meta.get("roles", {}).get(args.role)
            if role_meta:
                npz_path = model_dir / role_meta["npz"]
                if npz_path.exists():
                    color_matcher = ColorMatcher(npz_path, role_meta)
                    print(f"Using color model: {meta_files[0].name}")

    # Collect experiments to evaluate
    if args.exp:
        exps = [args.exp]
    else:
        task_dir = AUTO_DIR / args.task
        if not task_dir.exists():
            print(f"No auto-annotations found for {args.task}")
            return
        exps = [d.name for d in sorted(task_dir.iterdir()) if d.is_dir()]

    print(f"Evaluating {len(exps)} experiments, role={args.role}, task={args.task}")

    all_results = []

    if args.vs_human:
        # IoU mode
        human_exps = {d.name for d in (HUMAN_DIR / args.task).iterdir()
                      if d.is_dir()} if (HUMAN_DIR / args.task).exists() else set()
        eval_exps = [e for e in exps if e in human_exps]
        print(f"  {len(eval_exps)} have human annotations for IoU comparison")
        for exp in eval_exps:
            r = compute_iou_vs_human(args.task, exp, args.role, decoder)
            if r:
                all_results.append(r)
                print(f"  {exp[-45:]:45s}  IoU={r['mean_iou']:.3f}  min={r['min_iou']:.3f}  n={r['n_pairs']}")
        if all_results:
            mean_iou = np.mean([r["mean_iou"] for r in all_results])
            print(f"\nOverall mean IoU: {mean_iou:.4f} across {len(all_results)} experiments")
    else:
        # Multi-view consistency mode
        suspect_list = []
        for i, exp in enumerate(exps):
            r = eval_experiment(args.task, exp, args.role, calib, decoder,
                                color_matcher=color_matcher)
            if r is None:
                continue
            all_results.append(r)
            if r.get("suspect"):
                suspect_list.append(r)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(exps)}] ...")

        # Summary
        print(f"\n{'─'*70}")
        print(f"Evaluated: {len(all_results)}  |  Suspect: {len(suspect_list)}")

        if all_results:
            for key in ("area_ratio_min_mean", "proj_inside_frac_mean",
                        "sam2_score_min_mean", "border_tool_frac_max_mean"):
                vals = [r[key] for r in all_results if r.get(key) is not None]
                if vals:
                    print(f"  {key:35s}: mean={np.mean(vals):.3f}  min={np.min(vals):.3f}")

        if suspect_list:
            print(f"\nTop-{args.top_k} suspect experiments (lowest proj_inside_frac):")
            suspect_list.sort(key=lambda r: r.get("proj_inside_frac_mean") or 1.0)
            for r in suspect_list[:args.top_k]:
                pif = r.get("proj_inside_frac_mean")
                arm = r.get("area_ratio_min_mean")
                print(f"  {r['exp'][-50:]:50s}  proj_inside={pif}  area_ratio_min={arm}")

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
