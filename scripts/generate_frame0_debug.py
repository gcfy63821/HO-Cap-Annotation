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
import atexit
import getpass
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np


# Resolve once at import time (used by per-exp temp h5 paths)
HOCAP_ROOT = Path(__file__).resolve().parent.parent
CONVERT_SCRIPT = HOCAP_ROOT / "tools" / "00_convert_videos_to_h5.py"


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


def build_temp_frame0_h5(exp_dir: Path, scratch_dir: Path) -> Path:
    """Run tools/00_convert_videos_to_h5.py for frame 0 only, output into
    scratch_dir (typically /dev/shm). Returns the resulting h5 path, or
    raises RuntimeError if conversion failed.

    This is exactly the same conversion code path the production pipeline
    uses (via run_auto_annotator → 00_convert_videos_to_h5.py), so the
    resulting h5 reflects how downstream stages would see frame 0. If
    something looks wrong in the debug image (BGR/RGB swap, off-by-one
    frame, depth all zero) the bug is in the conversion code, not in this
    debug tool.
    """
    if not CONVERT_SCRIPT.exists():
        raise RuntimeError(f"convert script not found: {CONVERT_SCRIPT}")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    # Unique filename per exp so concurrent workers don't collide.
    safe_name = f"{exp_dir.parent.name}__{exp_dir.name}__frame0.h5".replace("/", "_")
    out_path = scratch_dir / safe_name
    if out_path.exists():
        out_path.unlink()
    cmd = [
        sys.executable, "-u", str(CONVERT_SCRIPT),
        "--input_dir", str(exp_dir),
        "--output_file", str(out_path),
        "--start_frame", "0",
        "--end_frame", "0",  # inclusive — gives a 1-frame h5
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"00_convert_videos_to_h5.py failed for {exp_dir}\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr (tail):\n{proc.stderr[-500:] if proc.stderr else '<empty>'}"
        )
    return out_path


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


MASK_CANDIDATES_REL = [
    Path("masks") / "masks.h5",
    Path("tool_masks") / "masks.h5",
]


def inspect_mask_sources(ann_exp_dir: Path):
    """Return (chosen_path or None, ds_shape or None, candidates_report).

    candidates_report is a list of (path, exists, shape_or_reason) so the
    caller can print exactly which paths were tried and why each was skipped.
    The first candidate that exists, opens, and has a valid 'masks' dataset
    is chosen.
    """
    report = []
    chosen = None
    chosen_shape = None
    for rel in MASK_CANDIDATES_REL:
        p = ann_exp_dir / rel
        if not p.exists():
            report.append((p, False, "missing"))
            continue
        try:
            with h5py.File(p, 'r') as f:
                ds = f.get('masks')
                if ds is None:
                    report.append((p, True, "no 'masks' dataset"))
                    continue
                shape = tuple(ds.shape)
                report.append((p, True, f"shape={shape}"))
                if chosen is None and shape[0] > 0:
                    chosen = p
                    chosen_shape = shape
        except Exception as e:
            report.append((p, True, f"open failed: {type(e).__name__}: {e}"))
    return chosen, chosen_shape, report


def load_mask_frame0(masks_h5_path: Path, cam_idx: int):
    """Returns (H, W) uint8 binary mask, or None on miss.

    `masks_h5_path` should be the chosen path returned by inspect_mask_sources
    (so we don't repeat the candidate walk per cam)."""
    if masks_h5_path is None or not masks_h5_path.exists():
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

# OpenCV's TURBO is perceptually uniform (better than JET for depth) but only
# shipped since cv2 4.1. Fall back to JET if not available.
_DEPTH_COLORMAP = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)


def colorize_depth(depth_m, lo=None, hi=None, invalid_rgb=(80, 0, 80)):
    """depth_m (H,W) float32 → (H,W,3) RGB uint8 colored depth.

    lo / hi:
      None (default) → auto from 2nd–98th percentile of valid pixels.
                       This adapts to the actual scene depth and keeps the
                       colormap in the meaningful range (no wasted dynamic
                       range on a few far-away outliers).
      explicit values → fixed range (clip outside).

    Invalid (depth <= 0) pixels are painted in `invalid_rgb` instead of the
    usual black so they're visually obvious — black bleeds into TURBO's blue
    end and is hard to distinguish.

    Returns (H,W,3) RGB uint8 + (lo_used, hi_used) for the caller to label.
    """
    H, W = depth_m.shape
    out = np.full((H, W, 3), invalid_rgb, dtype=np.uint8)
    valid = depth_m > 0
    if not valid.any():
        return out, (0.0, 0.0)
    valid_d = depth_m[valid]
    if lo is None or hi is None:
        # Use percentiles so a few stray sky / sensor-noise pixels don't
        # squash the meaningful range.
        p_lo = float(np.percentile(valid_d, 2))
        p_hi = float(np.percentile(valid_d, 98))
        # Guarantee a minimum spread (3 cm) so the colormap doesn't degenerate
        # when the scene is essentially flat.
        if p_hi - p_lo < 0.03:
            p_hi = p_lo + 0.03
        lo = p_lo if lo is None else lo
        hi = p_hi if hi is None else hi
    norm = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0, 1)
    norm_u8 = (norm * 255).astype(np.uint8)
    cm = cv2.applyColorMap(norm_u8, _DEPTH_COLORMAP)        # BGR
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    out[valid] = cm[valid]
    return out, (float(lo), float(hi))


def overlay_mask(rgb, mask):
    """rgb (H,W,3) RGB + mask (H,W) bool → annotated RGB with green overlay
    + yellow contour. Mutates a copy.

    Returns (out_rgb, info_text) where info_text always includes the mask
    H×W (or 'no mask' / 'EMPTY mask <H>x<W>') so you can spot resolution
    mismatches at a glance.
    """
    if mask is None:
        return rgb.copy(), "no mask"
    h, w = mask.shape[:2]
    rgb_h, rgb_w = rgb.shape[:2]
    size_label = f"{h}x{w}"
    if (h, w) != (rgb_h, rgb_w):
        size_label += f" (RGB {rgb_h}x{rgb_w} — MISMATCH)"
    if mask.sum() == 0:
        return rgb.copy(), f"EMPTY mask  {size_label}"
    if (h, w) != (rgb_h, rgb_w):
        # Resize so we still get a meaningful overlay even when shapes differ.
        mask = cv2.resize(mask, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
    out = rgb.copy()
    overlay = rgb.copy(); overlay[mask > 0] = (0, 255, 0)
    out = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)
    cnt, _ = cv2.findContours((mask * 255).astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, (255, 255, 0), 2)
    return out, f"area={int(mask.sum())}  {size_label}"


def label_image(img, text, scale=0.7):
    """Draw text top-left with a black halo for readability. Mutates img."""
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def make_exp_mosaic(exp_dir: Path, ann_exp_dir: Path, max_width=2400,
                     rgb_source="h5", scratch_dir: Path = None):
    """Render the n_cams × 2 mosaic for this exp.

    rgb_source:
      'h5'  (default) — read RGB from a frame-0 h5; this is what downstream
                        tools see, so what you see is what they see.
      'mp4'           — read RGB direct from cam*_rgb.mp4 first frame.

    Frame-0 h5 source priority:
      1. <exp>/data00000000.h5 if it already exists (cheap)
      2. Build a fresh 1-frame h5 in scratch_dir (default /dev/shm/$USER/...)
         via tools/00_convert_videos_to_h5.py — this exercises the SAME
         conversion code the pipeline uses, so the panel reflects the real
         h5 the tracker sees. The temp h5 is removed after the exp.
      3. If conversion fails entirely, depth panel falls back to magenta
         "NO DEPTH" placeholder; RGB falls back to mp4.

    Returns RGB uint8 (Ht, Wt, 3) or None on failure.
    """
    rgb_videos = sorted(exp_dir.glob("cam*_rgb.mp4"))
    if not rgb_videos:
        return None

    # ---- mask source diagnostics (printed once per exp) ----
    chosen_mask_h5, mask_ds_shape, mask_report = inspect_mask_sources(ann_exp_dir)
    print(f"  [mask] {ann_exp_dir.name}:")
    for p, exists, info in mask_report:
        marker = "OK " if exists else "no "
        print(f"    [{marker}] {p}   ({info})")
    if chosen_mask_h5 is not None:
        print(f"    -> using {chosen_mask_h5}  shape={mask_ds_shape}")
    else:
        print(f"    -> no usable masks.h5 found")
    mask_src_short = (f"{chosen_mask_h5.parent.name}/{chosen_mask_h5.name}"
                      if chosen_mask_h5 is not None else "no mask")

    # Resolve the data h5 for this exp: prefer existing, else build fresh
    # in scratch (/dev/shm) and clean up at the end.
    permanent_h5 = exp_dir / "data00000000.h5"
    data_h5 = None
    temp_h5_built = False
    if permanent_h5.exists():
        data_h5 = permanent_h5
    elif scratch_dir is not None:
        try:
            data_h5 = build_temp_frame0_h5(exp_dir, scratch_dir)
            temp_h5_built = True
        except Exception as e:
            print(f"  [WARN] {exp_dir.name}: temp h5 build failed: {e}")
            data_h5 = None
    # data_h5 is now either: existing h5, freshly-built scratch h5, or None
    if data_h5 is None:
        # Sentinel: load_*_from_h5 helpers return None when this is missing.
        data_h5 = permanent_h5

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
        mask = load_mask_frame0(chosen_mask_h5, cam_idx)
        color_panel, mask_info = overlay_mask(rgb, mask)
        cam_name = mp4.stem  # "cam0_rgb"
        label_image(color_panel,
                    f"{cam_name}  rgb {rgb_src_label}  ·  mask src={mask_src_short}  ·  {mask_info}")
        # depth
        depth = load_depth_frame0(data_h5, cam_idx)
        if depth is None:
            # Distinct loud color so it's instantly obvious which panels
            # don't have data, instead of a black square that the user
            # might mistake for a depth bug.
            reason = ("data00000000.h5 missing"
                      if not data_h5.exists()
                      else "data h5 has no 'depths' key (or read failed)")
            depth_panel = np.full((H, W, 3), (60, 0, 60), dtype=np.uint8)
            # Two-line label
            label_image(depth_panel, f"NO DEPTH — {reason}", scale=0.7)
            cv2.putText(depth_panel,
                        f"({data_h5.name})",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        else:
            depth_panel, (lo_used, hi_used) = colorize_depth(depth)
            valid = depth > 0
            valid_pct = float(valid.mean()) * 100
            if valid.any():
                d_min = float(depth[valid].min())
                d_max = float(depth[valid].max())
                label_image(depth_panel,
                            f"depth h5  valid {valid_pct:.0f}%  "
                            f"data {d_min:.2f}-{d_max:.2f}m  "
                            f"colormap {lo_used:.2f}-{hi_used:.2f}m  "
                            f"(magenta=invalid)")
            else:
                # Even all-zero depth → fill with bright color, not black.
                depth_panel = np.full((H, W, 3), (60, 0, 60), dtype=np.uint8)
                label_image(depth_panel,
                            "depth h5 ALL ZERO — sensor/conversion issue?")
        # concatenate side-by-side
        row = np.concatenate([color_panel, depth_panel], axis=1)
        rows.append(row)

    # Clean up the temp h5 we built in /dev/shm (other path: existing
    # permanent h5 — leave it alone).
    if temp_h5_built and data_h5 is not None and data_h5.exists():
        try:
            data_h5.unlink()
        except Exception:
            pass

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
    task, exp, exp_dir, ann_exp_dir, force, max_width, rgb_source, scratch_dir = args_tuple
    out_dir = ann_exp_dir / "debug"
    out_path = out_dir / "frame0_debug.png"
    if out_path.exists() and not force:
        return (task, exp, "skip", str(out_path), False)
    try:
        mosaic = make_exp_mosaic(exp_dir, ann_exp_dir, max_width=max_width,
                                  rgb_source=rgb_source,
                                  scratch_dir=Path(scratch_dir) if scratch_dir else None)
        if mosaic is None:
            return (task, exp, "fail:no-rgb", "", False)
        out_dir.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        # Side-channel: did this exp have a permanent h5 (vs. needing scratch)?
        had_permanent_h5 = (exp_dir / "data00000000.h5").exists()
        return (task, exp, "ok", str(out_path), not had_permanent_h5)
    except Exception as e:
        return (task, exp, f"fail:{type(e).__name__}: {e}", "", False)


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
    ap.add_argument("--scratch_dir", type=str, default=None,
                    help="Directory for ephemeral 1-frame h5s built when an "
                         "exp doesn't have a permanent data00000000.h5. "
                         "Default: /dev/shm/$USER/frame0_debug_$$ (tmpfs; "
                         "fast and auto-freed at script exit). Pass "
                         "--no_temp_h5 to disable temp-h5 building entirely.")
    ap.add_argument("--no_temp_h5", action="store_true",
                    help="Don't auto-build a 1-frame h5 in scratch when the "
                         "exp lacks one. Then depth panels for those exps "
                         "show the magenta 'NO DEPTH' placeholder.")
    args = ap.parse_args()

    # ---- Resolve scratch dir + register cleanup ----
    if args.no_temp_h5:
        scratch_dir = None
    else:
        if args.scratch_dir:
            scratch_dir = Path(args.scratch_dir).resolve()
        else:
            user = getpass.getuser()
            scratch_dir = Path("/dev/shm") / user / f"frame0_debug_{os.getpid()}"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        def _cleanup_scratch():
            try:
                if scratch_dir and scratch_dir.exists():
                    shutil.rmtree(scratch_dir, ignore_errors=True)
            except Exception:
                pass
        atexit.register(_cleanup_scratch)

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
    print(f"[INFO] scratch_dir      = {scratch_dir if scratch_dir else '<disabled (--no_temp_h5)>'}")
    print(f"[INFO] jobs             = {args.jobs}   force={args.force}")
    print()

    sd_str = str(scratch_dir) if scratch_dir else None
    work = [(t, e, ed, aed, args.force, args.max_width, args.rgb_source, sd_str)
            for (t, e, ed, aed) in exps]
    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    no_depth_exps = []
    if args.jobs <= 1:
        results = (render_one(w) for w in work)
    else:
        pool = mp.Pool(processes=args.jobs)
        results = pool.imap_unordered(render_one, work, chunksize=1)
    for i, (task, exp, status, path, used_temp_h5) in enumerate(results, 1):
        prefix = f"[{i:>4d}/{len(work)}]"
        if status == "ok":
            n_ok += 1
            tag = ("  [built 1-frame h5 in scratch]" if (used_temp_h5 and scratch_dir is not None)
                   else "  [no data h5 → magenta depth panel]" if used_temp_h5
                   else "")
            print(f"{prefix} OK    {task}/{exp}  →  {path}{tag}")
            if used_temp_h5:
                no_depth_exps.append(f"{task}/{exp}")
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
    if no_depth_exps:
        if scratch_dir is not None:
            print(f"\n[!] {len(no_depth_exps)} exp(s) had no permanent data00000000.h5 — "
                  f"a 1-frame h5 was built in scratch ({scratch_dir}) for the "
                  f"depth panel and removed after rendering. The depth you see "
                  f"is exactly what tools/00_convert_videos_to_h5.py produces "
                  f"so any visual oddity = real conversion bug.")
        else:
            print(f"\n[!] {len(no_depth_exps)} exp(s) have NO data00000000.h5 and "
                  f"--no_temp_h5 was passed; their depth panel is magenta. "
                  f"Drop --no_temp_h5 to auto-build a frame-0 h5 in /dev/shm.")
        for e in no_depth_exps[:5]:
            print(f"      - {e}")
        if len(no_depth_exps) > 5:
            print(f"      ... and {len(no_depth_exps) - 5} more")
    print(f"\nBrowse with: python scripts/browse_frame0_debug.py --videos_root {videos_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
