"""Depth-map utilities shared by template_auto_annotate and eval_mask_quality.

All depth PNGs are uint16 millimetres, produced by extract_depth_keyframes.py.
Read back:  depth_m = load_depth(path)   →  float32 metres, 0 = invalid
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_depth(png_path: Path) -> Optional[np.ndarray]:
    """Load depth PNG → float32 metres. Returns None if file missing."""
    if not png_path.is_file():
        return None
    d = cv2.imread(str(png_path), cv2.IMREAD_ANYDEPTH)
    if d is None:
        return None
    return (d.astype(np.float32)) / 1000.0  # mm → m, 0 = invalid


def depth_png_path(bundle: Path, task: str, exp: str, cam_idx: int, frame_idx: int) -> Path:
    return bundle / task / exp / f"cam{cam_idx}_depth.kf{frame_idx}.png"


# ── modal depth ───────────────────────────────────────────────────────────────

def modal_depth(depth_m: np.ndarray, mask: Optional[np.ndarray] = None,
                min_d: float = 0.1, max_d: float = 3.0,
                bin_size: float = 0.02) -> Optional[float]:
    """
    Return the modal depth (most common depth value) within a mask region.

    Uses histogram binning (bin_size metres).  Returns None if no valid pixels.
    """
    if mask is not None:
        d = depth_m[mask.astype(bool)]
    else:
        d = depth_m.ravel()
    d = d[(d >= min_d) & (d <= max_d)]
    if len(d) == 0:
        return None
    n_bins = int((max_d - min_d) / bin_size) + 1
    hist, edges = np.histogram(d, bins=n_bins, range=(min_d, max_d))
    peak_bin = int(np.argmax(hist))
    return float((edges[peak_bin] + edges[peak_bin + 1]) / 2)


# ── unproject / project ───────────────────────────────────────────────────────

def unproject_mask(depth_m: np.ndarray, mask: np.ndarray,
                   K: np.ndarray, T_cw: np.ndarray,
                   max_pts: int = 4000,
                   depth_tol: float = 0.05) -> Optional[np.ndarray]:
    """
    Unproject mask pixels to 3D world coordinates using depth.

    Args:
        depth_m : (H, W) float32 metres
        mask    : (H, W) bool
        K       : (3, 3) intrinsics
        T_cw    : (4, 4) camera-to-world (extr2world convention)
        max_pts : max points to return (random subsample)
        depth_tol : ignore pixels within depth_tol of the modal depth boundary

    Returns (N, 3) float32 world points, or None if insufficient valid pixels.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None

    d = depth_m[ys, xs]
    valid = d > 0.05
    ys, xs, d = ys[valid], xs[valid], d[valid]
    if len(ys) == 0:
        return None

    # Subsample for speed
    if len(ys) > max_pts:
        idx = np.random.choice(len(ys), max_pts, replace=False)
        ys, xs, d = ys[idx], xs[idx], d[idx]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Camera-frame 3D
    Xc = (xs - cx) / fx * d
    Yc = (ys - cy) / fy * d
    Zc = d
    pts_cam = np.stack([Xc, Yc, Zc, np.ones_like(d)], axis=1)  # (N, 4)

    # World-frame
    pts_world = (T_cw @ pts_cam.T).T[:, :3]  # (N, 3)
    return pts_world.astype(np.float32)


def project_points_to_image(pts_world: np.ndarray,
                             K: np.ndarray, T_wc: np.ndarray,
                             H: int, W: int) -> np.ndarray:
    """
    Project (N, 3) world points into image coords. Returns (N, 2) float32 (x, y).
    Points behind camera (Z ≤ 0) get coords (-1, -1).
    """
    ones = np.ones((len(pts_world), 1), dtype=np.float32)
    pts_h = np.concatenate([pts_world, ones], axis=1)       # (N, 4)
    pts_cam = (T_wc @ pts_h.T).T                            # (N, 4)
    Z = pts_cam[:, 2]
    uv = np.full((len(pts_world), 2), -1.0, dtype=np.float32)
    ok = Z > 0
    x = (pts_cam[ok, 0] / Z[ok]) * K[0, 0] + K[0, 2]
    y = (pts_cam[ok, 1] / Z[ok]) * K[1, 1] + K[1, 2]
    uv[ok, 0] = x
    uv[ok, 1] = y
    return uv


def mask_from_projected_points(uv: np.ndarray, H: int, W: int,
                                radius: int = 3) -> np.ndarray:
    """
    Rasterise projected 2D points into a binary mask (H, W) bool.
    Each point draws a small circle of given radius.
    """
    canvas = np.zeros((H, W), dtype=np.uint8)
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    pts = uv[valid].astype(np.int32)
    for x, y in pts:
        cv2.circle(canvas, (x, y), radius, 1, -1)
    return canvas.astype(bool)


# ── depth-guided point selection ──────────────────────────────────────────────

def depth_filter_candidates(candidates_xy: np.ndarray,
                             depth_m: np.ndarray,
                             target_depth: float,
                             depth_tol: float = 0.15) -> np.ndarray:
    """
    Filter candidate (x, y) points to those within depth_tol of target_depth.
    Returns boolean mask over candidates.
    """
    H, W = depth_m.shape
    keep = np.zeros(len(candidates_xy), dtype=bool)
    for i, (x, y) in enumerate(candidates_xy):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            d = depth_m[yi, xi]
            if d > 0.05 and abs(d - target_depth) <= depth_tol:
                keep[i] = True
    return keep


def depth_best_point(score_map: np.ndarray,
                     depth_m: np.ndarray,
                     mask: np.ndarray,
                     target_depth: float,
                     depth_tol: float = 0.15) -> Optional[tuple[float, float]]:
    """
    Among pixels in mask with score_map < threshold (low cost = good color match),
    pick the one whose depth is closest to target_depth.

    Returns (x, y) or None.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    d = depth_m[ys, xs]
    valid = (d > 0.05) & (np.abs(d - target_depth) <= depth_tol)
    if not valid.any():
        # Relax tolerance
        valid = d > 0.05
    if not valid.any():
        return None
    ys, xs, d = ys[valid], xs[valid], d[valid]
    scores = score_map[ys, xs]
    best = int(np.argmin(scores))
    return float(xs[best]), float(ys[best])


# ── cross-camera depth consistency ────────────────────────────────────────────

def depth_cross_camera_iou(
    masks: dict,        # {cam_idx: (H, W) bool}
    depths: dict,       # {cam_idx: (H, W) float32 m}
    Ks: dict,           # {cam_idx: (3,3)}
    T_cws: dict,        # {cam_idx: (4,4) cam-to-world}
    T_wcs: dict,        # {cam_idx: (4,4) world-to-cam}
    max_pts: int = 2000,
    proj_radius: int = 4,
) -> dict:
    """
    For each source camera, unproject its mask to 3D, project into all other
    cameras, and compute IoU with their masks.

    Returns {src_cam: {"mean_iou": float, "per_cam": {tgt_cam: iou}}}
    """
    cam_ids = sorted(set(masks) & set(depths) & set(Ks) & set(T_cws))
    results = {}

    for src in cam_ids:
        pts3d = unproject_mask(depths[src], masks[src], Ks[src], T_cws[src],
                               max_pts=max_pts)
        if pts3d is None or len(pts3d) < 10:
            results[src] = {"mean_iou": 0.0, "per_cam": {}}
            continue

        H, W = masks[src].shape
        per_cam = {}
        for tgt in cam_ids:
            if tgt == src:
                continue
            tgt_mask = masks.get(tgt)
            if tgt_mask is None:
                continue
            Ht, Wt = tgt_mask.shape
            uv = project_points_to_image(pts3d, Ks[tgt], T_wcs[tgt], Ht, Wt)
            proj_mask = mask_from_projected_points(uv, Ht, Wt, radius=proj_radius)
            inter = (proj_mask & tgt_mask).sum()
            union = (proj_mask | tgt_mask).sum()
            per_cam[tgt] = float(inter / union) if union > 0 else 0.0

        mean_iou = float(np.mean(list(per_cam.values()))) if per_cam else 0.0
        results[src] = {"mean_iou": mean_iou, "per_cam": per_cam}

    return results


# ── depth-based multi-view reprojection prompts ────────────────────────────────

def depth_reproject_prompts(
    anchor_masks: dict,   # {cam_id: (H, W) bool}
    anchor_depths: dict,  # {cam_id: (H, W) float32 m}
    anchor_Ks: dict,      # {cam_id: (3, 3)}
    anchor_T_cws: dict,   # {cam_id: (4, 4) cam-to-world}
    target_T_wc: np.ndarray,  # (4, 4) world-to-cam
    target_K: np.ndarray,     # (3, 3)
    target_H: int, target_W: int,
    n_pos: int = 8,
    neg_margin_px: int = 20,
    max_pts: int = 3000,
    proj_radius: int = 4,
) -> tuple:
    """
    Unproject anchor cameras' masks to 3D, project into a target camera,
    and return positive/negative SAM2 point prompts for that camera.

    Returns (pos_pts, neg_pts) as lists of (x, y) floats.
    pos_pts: points sampled from the dense interior of the projected mask.
    neg_pts: points from the ring just outside the projected mask boundary.
    """
    # Accumulate projected points from all anchors
    all_uv = []
    for cam_id in sorted(anchor_masks):
        mask = anchor_masks.get(cam_id)
        depth = anchor_depths.get(cam_id)
        K = anchor_Ks.get(cam_id)
        T_cw = anchor_T_cws.get(cam_id)
        if mask is None or depth is None or K is None or T_cw is None:
            continue
        pts3d = unproject_mask(depth, mask, K, T_cw, max_pts=max_pts // max(len(anchor_masks), 1))
        if pts3d is None or len(pts3d) < 5:
            continue
        uv = project_points_to_image(pts3d, target_K, target_T_wc, target_H, target_W)
        # Keep only on-screen points
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < target_W) & (uv[:, 1] >= 0) & (uv[:, 1] < target_H)
        all_uv.append(uv[ok])

    if not all_uv:
        return [], []

    uv_all = np.concatenate(all_uv, axis=0)
    if len(uv_all) < 5:
        return [], []

    # Build projected mask (dense coverage)
    proj_mask = mask_from_projected_points(uv_all, target_H, target_W, radius=proj_radius)
    if proj_mask.sum() == 0:
        return [], []

    # ── Positive points: sample from interior (erosion to stay away from boundary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eroded = cv2.erode(proj_mask.astype(np.uint8), kernel)
    interior = eroded.astype(bool)
    if interior.sum() < n_pos:
        interior = proj_mask  # fallback: no erosion

    ys_in, xs_in = np.where(interior)
    # Sample evenly spread points via k-means-like grid sampling
    n_sample = min(n_pos, len(ys_in))
    if n_sample > 0:
        step = max(1, len(ys_in) // n_sample)
        idx = np.arange(0, len(ys_in), step)[:n_sample]
        pos_pts = [(float(xs_in[i]), float(ys_in[i])) for i in idx]
    else:
        pos_pts = []

    # ── Negative points: ring just outside the projected mask
    dilated = cv2.dilate(proj_mask.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (neg_margin_px * 2, neg_margin_px * 2)))
    ring = dilated.astype(bool) & ~proj_mask
    ys_out, xs_out = np.where(ring)
    n_neg = max(2, n_pos // 2)
    if len(ys_out) >= n_neg:
        step = max(1, len(ys_out) // n_neg)
        idx = np.arange(0, len(ys_out), step)[:n_neg]
        neg_pts = [(float(xs_out[i]), float(ys_out[i])) for i in idx]
    else:
        neg_pts = []

    return pos_pts, neg_pts
