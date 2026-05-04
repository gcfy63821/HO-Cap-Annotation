#!/usr/bin/env python3
"""
Frame-0 debug montage generator.

For every DONE <task>/<exp> under a videos_root (i.e. has Stage 5's
processed/fd_pose_solver/fd_poses_merged_fixed.npy on the annotated side),
render a single PNG showing:
  - color + mask overlay   (left column, one row per camera)
  - depth as a colormap    (right column, same)

By default RGB is read from `data00000000.h5` (the tensor downstream tools
actually consume) — this lets you visually verify the mp4→h5 conversion is
correct. Pass --rgb_source mp4 to read the source mp4 instead.

Output:
  <videos_root>_annotated/<task>/<exp>/debug/frame0_debug.png

Goal: let you eyeball *every* completed experiment in seconds and spot the
obvious failure modes:
  - object not visible at frame 0      → can't register
  - empty mask (SAM2/DINO produced 0)  → fd_pose_solver fails
  - mask drifts to background          → poses drift later
  - depth all-black (sensor issue)     → depth-validation will fail
  - h5 vs mp4 color/order mismatch     → 00_convert_videos_to_h5.py bug
                                         (visible if you compare --rgb_source h5
                                          and --rgb_source mp4 outputs)

Reads as little data as possible:
  - RGB: data00000000.h5 imgs[0, cam_idx]    (default; bypass mp4 entirely)
         OR cam*_rgb.mp4 first frame         (with --rgb_source mp4)
  - Mask: tool_masks/masks.h5[0, cam_idx]    (if exists; else "no mask")
  - Depth: data00000000.h5[depths][0, cam_idx] (if exists; else "no h5")

Usage:
  conda activate hocap-annotation
  python scripts/generate_frame0_debug.py \\
    --videos_root /viscam/projects/robotool/data/videos_0102

Common flags:
  --jobs N             Parallelism (default 4). Each worker handles one exp.
  --force              Re-render even if PNG already exists.
  --task_filter A B    Only process these task subfolders.
  --include_unfinished Also render exps without fd_poses_merged_fixed.npy.
  --rgb_source h5|mp4  Where to read RGB from (default: h5 — verifies conversion).
  --max_width 2400     Clamp mosaic width (downscale per-cam if needed).
"""

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np


# ============================================================
# Paths
# ============================================================

def annotated_root_for_videos(videos_root: Path) -> Path:
    return videos_root.parent / f"{videos_root.name}_annotated"


def is_exp_done(ann_exp_dir: Path) -> bool:
    """Annotation finished sentinel: Stage 5 merger output exists."""
    return (ann_exp_dir / "processed" / "fd_pose_solver"
            / "fd_poses_merged_fixed.npy").exists()


def discover_exps(videos_root: Path, task_filter=None, only_done=True):
    """Walk <videos_root>/<task>/<exp>/, return [(task, exp, exp_dir, ann_exp_dir)].

    only_done=True (default) → keep exps where the annotated-side has
    `processed/fd_pose_solver/fd_poses_merged_fixed.npy` (Stage 5 done).
    Use only_done=False to include in-progress / never-tracked exps too
    (useful to spot exps that *failed early* before producing any output).
    """
    out = []
    if not videos_root.is_dir():
        return out
    skipped_unfinished = 0
    for task_dir in sorted(videos_root.iterdir()):
        if not task_dir.is_dir(): continue
        if task_dir.name.startswith("realsense_calibrate"): continue
        if task_filter and task_dir.name not in task_filter: continue
        ann_task = annotated_root_for_videos(videos_root) / task_dir.name
        for exp_dir in sorted(task_dir.iterdir()):
            if not exp_dir.is_dir(): continue
            if not list(exp_dir.glob("cam*_rgb.mp4")): continue
            ann_exp = ann_task / exp_dir.name
            if only_done and not is_exp_done(ann_exp):
                skipped_unfinished += 1
                continue
            out.append((task_dir.name, exp_dir.name, exp_dir, ann_exp))
    return out, skipped_unfinished


# ============================================================
# Frame readers
# ============================================================

def read_first_rgb_from_mp4(mp4_path: Path):
    """Returns RGB uint8 (H, W, 3) or None. Cheap reference (decoded from
    the source video, not h5)."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_first_rgb_from_h5(data_h5_path: Path, cam_idx: int):
    """Returns RGB uint8 (H, W, 3) or None — read directly from
    data00000000.h5's `imgs` dataset (the SAME tensor downstream tools
    consume). Use this to verify the mp4→h5 conversion: if the displayed
    frame looks wrong (color shifted, BGR/RGB swapped, off-by-one frame),
    the bug is in 00_convert_videos_to_h5.py, not in the tracking stack."""
    if not data_h5_path.exists():
        return None
    try:
        with h5py.File(data_h5_path, "r") as f:
            ds = f.get("imgs")
            if ds is None or ds.shape[0] == 0 or cam_idx >= ds.shape[1]:
                return None
            return np.asarray(ds[0, cam_idx])    # uint8 (H, W, 3) RGB by convention
    except Exception:
        return None


def load_mask_frame0(masks_h5_path: Path, cam_idx: int):
    """Returns (H, W) uint8 mask (binary) or None."""
    if not masks_h5_path.exists():
        return None
    try:
        with h5py.File(masks_h5_path, 'r') as f:
            ds = f.get('masks')
            if ds is None or ds.shape[0] == 0:
                return None
            n_cams = ds.shape[1]
            if cam_idx >= n_cams:
                return None
            return (np.asarray(ds[0, cam_idx]) > 0).astype(np.uint8)
    except Exception:
        return None


def load_depth_frame0(data_h5_path: Path, cam_idx: int):
    """Returns (H, W) float32 meters or None."""
    if not data_h5_path.exists():
        return None
    try:
        with h5py.File(data_h5_path, 'r') as f:
            ds = f.get('depths')
            if ds is None or ds.shape[0] == 0:
                return None
            n_cams = ds.shape[1]
            if cam_idx >= n_cams:
                return None
            d = np.asarray(ds[0, cam_idx], dtype=np.float32) * 0.001
            return d
    except Exception:
        return None


# ============================================================
# Rendering
# ============================================================

def colorize_depth(depth_m, lo=0.2, hi=2.5):
    """depth_m (H,W) float32 → (H,W,3) RGB uint8 jet, with invalid (≤0) black."""
    H, W = depth_m.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    valid = depth_m > 0
    if not valid.any():
        return out
    norm = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0, 1)
    norm_u8 = (norm * 255).astype(np.uint8)
    cm = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)   # BGR
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    out[valid] = cm[valid]
    return out


def overlay_mask(rgb, mask):
    """rgb (H,W,3) RGB + mask (H,W) bool → annotated RGB with green overlay
    + yellow contour. Mutates a copy."""
    if mask is None:
        return rgb.copy(), "no mask"
    if mask.sum() == 0:
        return rgb.copy(), "EMPTY mask"
    out = rgb.copy()
    overlay = rgb.copy(); overlay[mask > 0] = (0, 255, 0)
    out = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)
    cnt, _ = cv2.findContours((mask * 255).astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, (255, 255, 0), 2)
    return out, f"area={int(mask.sum())}"


def label_image(img, text, scale=0.7):
    """Draw text top-left with a black halo for readability. Mutates img."""
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def make_exp_mosaic(exp_dir: Path, ann_exp_dir: Path, max_width=2400,
                     rgb_source="h5"):
    """Render the n_cams × 2 mosaic for this exp.

    rgb_source:
      'h5'  (default) — read RGB from data00000000.h5 imgs[0]; this is what
                        downstream tools see, so what you see is what they see.
                        Falls back to mp4 if data h5 is missing.
      'mp4'           — read RGB direct from cam*_rgb.mp4 first frame.

    Returns RGB uint8 (Ht, Wt, 3) or None on failure.
    """
    rgb_videos = sorted(exp_dir.glob("cam*_rgb.mp4"))
    if not rgb_videos:
        return None
    masks_h5 = ann_exp_dir / "tool_masks" / "masks.h5"
    data_h5  = exp_dir / "data00000000.h5"

    rows = []
    for cam_idx, mp4 in enumerate(rgb_videos):
        rgb = None
        rgb_src_label = ""
        if rgb_source == "h5":
            rgb = read_first_rgb_from_h5(data_h5, cam_idx)
            if rgb is not None:
                rgb_src_label = "from h5"
            else:
                rgb = read_first_rgb_from_mp4(mp4)
                rgb_src_label = "from mp4 (h5 missing)"
        else:
            rgb = read_first_rgb_from_mp4(mp4)
            rgb_src_label = "from mp4"
        if rgb is None:
            continue
        H, W = rgb.shape[:2]
        # color + mask
        mask = load_mask_frame0(masks_h5, cam_idx)
        color_panel, mask_info = overlay_mask(rgb, mask)
        cam_name = mp4.stem  # "cam0_rgb"
        label_image(color_panel, f"{cam_name}  rgb {rgb_src_label}  ·  {mask_info}")
        # depth
        depth = load_depth_frame0(data_h5, cam_idx)
        if depth is None:
            depth_panel = np.zeros((H, W, 3), dtype=np.uint8)
            label_image(depth_panel, "no depth (data h5 missing)")
        else:
            depth_panel = colorize_depth(depth)
            valid_pct = float((depth > 0).mean()) * 100
            label_image(depth_panel,
                        f"depth from h5  (valid {valid_pct:.0f}%, "
                        f"range {depth[depth>0].min():.2f}-{depth[depth>0].max():.2f}m)"
                        if (depth > 0).any() else "depth from h5  (all zero!)")
        # concatenate side-by-side
        row = np.concatenate([color_panel, depth_panel], axis=1)
        rows.append(row)

    if not rows:
        return None
    mosaic = np.concatenate(rows, axis=0)

    # Optional downscale to max_width
    Ht, Wt = mosaic.shape[:2]
    if Wt > max_width:
        scale = max_width / Wt
        new_w = int(Wt * scale)
        new_h = int(Ht * scale)
        mosaic = cv2.resize(mosaic, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return mosaic


# ============================================================
# Per-exp worker
# ============================================================

def render_one(args_tuple):
    task, exp, exp_dir, ann_exp_dir, force, max_width, rgb_source = args_tuple
    out_dir = ann_exp_dir / "debug"
    out_path = out_dir / "frame0_debug.png"
    if out_path.exists() and not force:
        return (task, exp, "skip", str(out_path))
    try:
        mosaic = make_exp_mosaic(exp_dir, ann_exp_dir, max_width=max_width,
                                  rgb_source=rgb_source)
        if mosaic is None:
            return (task, exp, "fail:no-rgb", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        return (task, exp, "ok", str(out_path))
    except Exception as e:
        return (task, exp, f"fail:{type(e).__name__}: {e}", "")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_root", required=True,
                    help="Raw videos root (e.g. /viscam/projects/robotool/data/videos_0102). "
                         "Output goes to <videos_root>_annotated/<task>/<exp>/debug/frame0_debug.png.")
    ap.add_argument("--jobs", type=int, default=4,
                    help="Number of parallel exp workers. Each worker decodes 1-8 mp4 frame-0s "
                         "+ reads ~1 MB of mask/depth, so high --jobs is fine on NFS.")
    ap.add_argument("--force", action="store_true",
                    help="Re-render even if frame0_debug.png already exists.")
    ap.add_argument("--task_filter", nargs='+', default=None,
                    help="Only process these task subfolders (basename match).")
    ap.add_argument("--max_width", type=int, default=2400,
                    help="Clamp final mosaic width (PNGs > this are downscaled).")
    ap.add_argument("--include_unfinished", action="store_true",
                    help="Also render exps that don't yet have "
                         "fd_poses_merged_fixed.npy. Default OFF — only DONE "
                         "exps are rendered (matches the request 'only show "
                         "exps that finished annotation').")
    ap.add_argument("--rgb_source", choices=["h5", "mp4"], default="h5",
                    help="Where to pull the RGB frame 0. "
                         "'h5' (default) reads data00000000.h5 imgs[0] — same "
                         "tensor the tracking stack consumes, so the panel "
                         "shows you exactly what fd_pose_solver / DINO see; "
                         "useful for verifying the mp4→h5 conversion is "
                         "correct. 'mp4' reads from cam*_rgb.mp4 directly "
                         "(bypasses h5; useful only if you want a reference).")
    args = ap.parse_args()

    videos_root = Path(args.videos_root).resolve()
    if not videos_root.is_dir():
        print(f"Error: --videos_root not a directory: {videos_root}")
        sys.exit(1)

    exps, n_skip_unfinished = discover_exps(
        videos_root,
        task_filter=set(args.task_filter) if args.task_filter else None,
        only_done=(not args.include_unfinished),
    )
    if not exps:
        print(f"No DONE exps found under {videos_root}")
        if n_skip_unfinished:
            print(f"   ({n_skip_unfinished} exp(s) under videos_root have no "
                  f"fd_poses_merged_fixed.npy yet — run the pipeline first or "
                  f"pass --include_unfinished to render them anyway)")
        sys.exit(0)

    print(f"[INFO] videos_root      = {videos_root}")
    print(f"[INFO] annotated        = {annotated_root_for_videos(videos_root)}")
    print(f"[INFO] tasks            = {sorted({t for t,_,_,_ in exps})}")
    print(f"[INFO] DONE exps        = {len(exps)}   "
          f"(skipped {n_skip_unfinished} unfinished)" if not args.include_unfinished
          else f"[INFO] all exps         = {len(exps)}   (--include_unfinished)")
    print(f"[INFO] rgb source       = {args.rgb_source}")
    print(f"[INFO] jobs             = {args.jobs}   force={args.force}")
    print()

    work = [(t, e, ed, aed, args.force, args.max_width, args.rgb_source)
            for (t, e, ed, aed) in exps]
    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    if args.jobs <= 1:
        results = (render_one(w) for w in work)
    else:
        pool = mp.Pool(processes=args.jobs)
        results = pool.imap_unordered(render_one, work, chunksize=1)
    for i, (task, exp, status, path) in enumerate(results, 1):
        prefix = f"[{i:>4d}/{len(work)}]"
        if status == "ok":
            n_ok += 1
            print(f"{prefix} OK    {task}/{exp}  →  {path}")
        elif status == "skip":
            n_skip += 1
            print(f"{prefix} skip  {task}/{exp}  (already exists)")
        else:
            n_fail += 1
            print(f"{prefix} FAIL  {task}/{exp}  ({status})")
    if args.jobs > 1:
        pool.close(); pool.join()
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"Summary: ok={n_ok}  skip={n_skip}  fail={n_fail}   "
          f"elapsed={elapsed:.1f}s ({len(work)/max(elapsed,1e-6):.1f} exps/s)")
    print(f"Browse with: python scripts/browse_frame0_debug.py --videos_root {videos_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
