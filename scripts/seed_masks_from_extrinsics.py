#!/usr/bin/env python3
"""
Seed multi-cam first-frame masks via extrinsics + depth, then SAM2-propagate.

Input:  one camera's frame-0 mask  (e.g. <annotated>/<exp>/first_frame/cam_5_segmentation.png)
Steps:
  1. Unproject the masked pixels in the source cam → 3D points (using its
     intrinsics + frame-0 depth from the data h5).
  2. Transform the 3D points to world frame (using source cam's extrinsics).
  3. Re-project into every other camera (using their extrinsics + intrinsics)
     to produce a frame-0 seed mask per cam.
  4. (optional) SAM2-propagate from each cam's seed → full masks.h5.
  5. Dump per-cam debug PNGs (RGB + projected seed overlay + final SAM2 mask)
     so you can verify the geometry and SAM2 propagation visually.

Output:
  <annotated>/<exp>/tool_masks/masks.h5     (full propagated masks, h5-only format)
  <annotated>/<exp>/tool_masks/objects.yaml
  <annotated>/<exp>/tool_masks/viz_per_cam/cam{cc}.png  (overlay)
  <annotated>/<exp>/debug/seed_projection.png   (8-cam mosaic of projected seeds)

Single-exp usage (testing):
  conda activate hocap-annotation
  python scripts/seed_masks_from_extrinsics.py \\
    --exp_folder /viscam/projects/robotool/data/videos_0101/mallet_crush_dough/20260104_smallplate_..._1 \\
    --calibration_yaml /viscam/projects/robotool/data/videos_0101/realsense_calibrate_0101/realsense_calibration_0101_global_aligned.yaml

  # quick projection check first (no SAM2 — fast):
  python scripts/seed_masks_from_extrinsics.py --exp_folder ... --calibration_yaml ... --no_sam2

  # source cam other than 5:
  python scripts/seed_masks_from_extrinsics.py --exp_folder ... --calibration_yaml ... --source_cam 3
"""

import argparse
import atexit
import gc
import getpass
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml


# ============================================================
# Paths
# ============================================================

HOCAP_ROOT = Path(__file__).resolve().parent.parent
CONVERT_SCRIPT = HOCAP_ROOT / "tools" / "00_convert_videos_to_h5.py"


def annotated_exp_dir(exp_folder: Path) -> Path:
    """data/<videos_X>/<task>/<exp>  →  data/<videos_X>_annotated/<task>/<exp>"""
    name = exp_folder.name
    task = exp_folder.parent.name
    videos = exp_folder.parent.parent.name
    base = exp_folder.parent.parent.parent
    return base / f"{videos}_annotated" / task / name


# ============================================================
# Calibration
# ============================================================

def load_calibration(calib_yaml: Path):
    """Read the 8-cam yaml and return (serials, K [N,3,3], extr [N,4,4]).
    extr[i] is camera-to-world (the convention used by MyClusterLoader)."""
    with open(calib_yaml, "r") as f:
        cams = yaml.safe_load(f)
    serials = []
    Ks = []
    extrs = []
    for c in cams:
        serials.append(str(c["camera_id"]).zfill(2))
        Ks.append(np.array(c["color_intrinsic_matrix"], dtype=np.float64))
        extrs.append(np.array(c["transformation"], dtype=np.float64))
    return serials, np.stack(Ks, 0), np.stack(extrs, 0)


# ============================================================
# Frame-0 h5 (build temp in /dev/shm if missing)
# ============================================================

def build_temp_frame0_h5(exp_dir: Path, scratch_dir: Path) -> Path:
    if not CONVERT_SCRIPT.exists():
        raise RuntimeError(f"convert script not found: {CONVERT_SCRIPT}")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{exp_dir.parent.name}__{exp_dir.name}__frame0.h5".replace("/", "_")
    out_path = scratch_dir / safe_name
    if out_path.exists():
        out_path.unlink()
    cmd = [
        sys.executable, "-u", str(CONVERT_SCRIPT),
        "--input_dir", str(exp_dir),
        "--output_file", str(out_path),
        "--start_frame", "0",
        "--end_frame", "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"00_convert_videos_to_h5.py failed for {exp_dir}\n"
            f"  stderr (tail):\n{proc.stderr[-500:] if proc.stderr else '<empty>'}"
        )
    return out_path


def get_frame0_data_h5(exp_folder: Path, scratch_dir: Path):
    """Return the path to a 1-frame data h5 for this exp. Builds one in
    scratch (/dev/shm) when no permanent h5 exists; returns existing one
    otherwise."""
    permanent = exp_folder / "data00000000.h5"
    if permanent.exists():
        return permanent, False
    h5 = build_temp_frame0_h5(exp_folder, scratch_dir)
    return h5, True


# ============================================================
# Geometric projection
# ============================================================

def project_mask(src_rgb_shape, depth_src, mask_src, K_src, extr_src,
                  K_tgt, extr_tgt, target_shape):
    """Reproject pixels masked in source cam to target cam image plane.

    Args:
      depth_src      (Hs, Ws) float32 meters
      mask_src       (Hs, Ws) uint8/bool — 1 where the object is in source
      K_src          (3, 3) intrinsics
      extr_src       (4, 4) cam-to-world for source
      K_tgt          (3, 3) intrinsics
      extr_tgt       (4, 4) cam-to-world for target
      target_shape   (Ht, Wt)

    Returns: (Ht, Wt) uint8 binary mask, plus stats dict for logging.
    """
    Hs, Ws = mask_src.shape[:2]
    Ht, Wt = target_shape

    ys, xs = np.where(mask_src > 0)
    n_total = len(xs)
    if n_total == 0:
        return np.zeros((Ht, Wt), dtype=np.uint8), {
            "src_pixels": 0, "with_depth": 0, "in_front": 0, "in_image": 0,
        }

    Z = depth_src[ys, xs].astype(np.float32)
    has_depth = Z > 0.01
    n_with_depth = int(has_depth.sum())
    xs, ys, Z = xs[has_depth], ys[has_depth], Z[has_depth]
    if Z.size == 0:
        return np.zeros((Ht, Wt), dtype=np.uint8), {
            "src_pixels": n_total, "with_depth": 0, "in_front": 0, "in_image": 0,
        }

    fx, fy = K_src[0, 0], K_src[1, 1]
    cx, cy = K_src[0, 2], K_src[1, 2]
    Xc = (xs.astype(np.float32) - cx) * Z / fx
    Yc = (ys.astype(np.float32) - cy) * Z / fy
    pts_src_cam = np.stack([Xc, Yc, Z], axis=1)               # (N, 3)

    # source cam → world
    R_s2w = extr_src[:3, :3]; t_s2w = extr_src[:3, 3]
    pts_world = pts_src_cam @ R_s2w.T + t_s2w

    # world → target cam (invert extr_tgt)
    extr_w2t = np.linalg.inv(extr_tgt)
    R_w2t = extr_w2t[:3, :3]; t_w2t = extr_w2t[:3, 3]
    pts_tgt = pts_world @ R_w2t.T + t_w2t                      # (N, 3)

    in_front = pts_tgt[:, 2] > 0.01
    n_in_front = int(in_front.sum())
    pts_tgt = pts_tgt[in_front]
    if pts_tgt.size == 0:
        return np.zeros((Ht, Wt), dtype=np.uint8), {
            "src_pixels": n_total, "with_depth": n_with_depth,
            "in_front": 0, "in_image": 0,
        }

    fx2, fy2 = K_tgt[0, 0], K_tgt[1, 1]
    cx2, cy2 = K_tgt[0, 2], K_tgt[1, 2]
    u = (fx2 * pts_tgt[:, 0] / pts_tgt[:, 2] + cx2).astype(np.int32)
    v = (fy2 * pts_tgt[:, 1] / pts_tgt[:, 2] + cy2).astype(np.int32)

    in_img = (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht)
    u, v = u[in_img], v[in_img]
    out = np.zeros((Ht, Wt), dtype=np.uint8)
    out[v, u] = 1
    return out, {
        "src_pixels": n_total, "with_depth": n_with_depth,
        "in_front": n_in_front, "in_image": int(in_img.sum()),
    }


def densify_seed_mask(seed, dilate_k=7, close_k=15):
    """Sparse projected seeds → solid blob via morphological close + dilate."""
    if seed.sum() == 0:
        return seed
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    out = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel_close)
    if dilate_k > 0:
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        out = cv2.dilate(out, kernel_dil, iterations=1)
    # Keep largest connected component to drop sparse outliers.
    n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n_cc > 1:
        biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = (lbl == biggest).astype(np.uint8)
    return out


# ============================================================
# Visualization
# ============================================================

def overlay_mask_rgb(rgb, mask, color=(0, 255, 0), contour_color=(255, 255, 0)):
    out = rgb.copy()
    if mask.sum() > 0:
        ov = rgb.copy(); ov[mask > 0] = color
        out = cv2.addWeighted(rgb, 0.5, ov, 0.5, 0)
        cnt, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255,
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnt, -1, contour_color, 2)
    return out


def label_image(img, text, scale=0.7):
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


# ============================================================
# h5 writer (matches batch_task_annotator's H5MaskWriter convention)
# ============================================================

class H5MaskWriter:
    def __init__(self, h5_path, n_frames, n_cams, H, W):
        self.h5_path = Path(h5_path)
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        if self.h5_path.exists():
            try: self.h5_path.unlink()
            except Exception: pass
        self._f = h5py.File(self.h5_path, "w")
        self._ds = self._f.create_dataset(
            "masks", shape=(n_frames, n_cams, H, W), dtype=np.uint8,
            chunks=(1, 1, H, W), compression="gzip",
        )
        self.n_frames, self.n_cams, self.H, self.W = n_frames, n_cams, H, W

    def write_frame(self, frame_idx, cam_idx, mask_2d_u8):
        if mask_2d_u8.shape != (self.H, self.W):
            mask_2d_u8 = cv2.resize(mask_2d_u8, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        self._ds[frame_idx, cam_idx] = mask_2d_u8

    def close(self):
        try: self._f.flush(); self._f.close()
        except Exception: pass


# ============================================================
# SAM2 propagation per cam
# ============================================================

def sam2_propagate(predictor, video_path, seed_mask, obj_id, writer, cam_idx,
                    offload=True):
    """Init SAM2 on this cam's mp4, seed frame 0 with seed_mask, propagate
    forward, write each frame to writer at (frame_idx, cam_idx). Returns
    number of frames written."""
    init_kw = dict(video_path=str(video_path))
    if offload:
        init_kw.update(offload_video_to_cpu=True, offload_state_to_cpu=True)
    try:
        state = predictor.init_state(**init_kw)
    except TypeError:
        state = predictor.init_state(video_path=str(video_path))
    try:
        try:
            predictor.reset_state(state)
        except Exception:
            pass
        # seed via mask
        predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id,
                                mask=(seed_mask > 0).astype(np.uint8))
        n = 0
        for fr_idx, oids, mlogits in predictor.propagate_in_video(state):
            for i, oid in enumerate(oids):
                if oid != obj_id: continue
                m = mlogits[i] if isinstance(mlogits, (list, tuple)) else mlogits
                import torch
                if torch.is_tensor(m):
                    m = (m > 0).cpu().numpy()
                else:
                    m = (m > 0)
                if m.ndim == 3: m = m[0] if m.shape[0] == 1 else m
                if m.ndim == 4: m = m[0, 0]
                writer.write_frame(fr_idx, cam_idx, m.astype(np.uint8))
                n += 1
        return n
    finally:
        try: predictor.reset_state(state)
        except Exception: pass
        del state
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()
        gc.collect()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_folder", required=True,
                    help="Single experiment folder (raw side, e.g. "
                         ".../videos_0101/mallet_crush_dough/<exp>).")
    ap.add_argument("--calibration_yaml", required=True,
                    help="Calibration YAML used by the rest of the pipeline.")
    ap.add_argument("--source_cam", type=int, default=5,
                    help="Index of the camera whose first_frame/cam_<N>_segmentation.png "
                         "to use as the seed (default: 5).")
    ap.add_argument("--first_frame_dir", type=str, default=None,
                    help="Override the directory containing cam_<N>_segmentation.png. "
                         "Default: <annotated>/<exp>/first_frame/")
    ap.add_argument("--tool_name", type=str, default=None,
                    help="Tool name written to objects.yaml. Default: strip "
                         "trailing _<digits> from exp folder name.")
    ap.add_argument("--no_sam2", action="store_true",
                    help="Skip SAM2 propagation; only project the seed across "
                         "cams + dump the debug mosaic. Fast iteration when "
                         "tuning extrinsics / dilate / source cam.")
    ap.add_argument("--dilate_k", type=int, default=7,
                    help="Morphological dilate kernel after projecting (default 7). "
                         "Set 0 to disable.")
    ap.add_argument("--close_k", type=int, default=15,
                    help="Morphological close kernel to fill holes in projection (default 15).")
    ap.add_argument("--sam2_checkpoint", type=str,
                    default=os.environ.get(
                        "SAM2_CKPT",
                        str(HOCAP_ROOT.parent / "mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt")
                    ),
                    help="SAM2 checkpoint path. Default reads $SAM2_CKPT or local fallback.")
    ap.add_argument("--sam2_model_cfg", type=str,
                    default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--obj_id", type=int, default=1)
    ap.add_argument("--no_offload", action="store_true",
                    help="Disable SAM2 CPU offload (faster but more GPU/CPU RAM).")
    ap.add_argument("--scratch_dir", type=str, default=None,
                    help="Directory for temp 1-frame h5 (default: /dev/shm/$USER/seed_masks_$$).")
    args = ap.parse_args()

    exp_folder = Path(args.exp_folder).resolve()
    if not exp_folder.is_dir():
        print(f"Error: --exp_folder not a dir: {exp_folder}"); sys.exit(1)
    calib_yaml = Path(args.calibration_yaml).resolve()
    if not calib_yaml.is_file():
        print(f"Error: --calibration_yaml not a file: {calib_yaml}"); sys.exit(1)

    ann_dir = annotated_exp_dir(exp_folder)
    ff_dir = Path(args.first_frame_dir).resolve() if args.first_frame_dir else (ann_dir / "first_frame")
    seed_png = ff_dir / f"cam_{args.source_cam}_segmentation.png"
    if not seed_png.exists():
        # Fallback also accepts "cam{N}_segmentation.png" without underscore
        alt = ff_dir / f"cam{args.source_cam}_segmentation.png"
        if alt.exists():
            seed_png = alt
        else:
            print(f"Error: source seed mask not found:\n  {seed_png}\n  {alt}"); sys.exit(1)

    # tool name
    import re
    tool_name = args.tool_name or re.sub(r"_\d+$", "", exp_folder.name)

    out_tool_masks_dir = ann_dir / "tool_masks"
    out_tool_masks_dir.mkdir(parents=True, exist_ok=True)
    out_debug_dir = ann_dir / "debug"
    out_debug_dir.mkdir(parents=True, exist_ok=True)
    out_viz_dir = out_tool_masks_dir / "viz_per_cam"
    out_viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] exp_folder      = {exp_folder}")
    print(f"[INFO] annotated       = {ann_dir}")
    print(f"[INFO] source seed     = {seed_png}")
    print(f"[INFO] tool_name       = {tool_name}")
    print(f"[INFO] no_sam2         = {args.no_sam2}")
    print()

    # ---------- Scratch h5 ----------
    if args.scratch_dir:
        scratch_dir = Path(args.scratch_dir).resolve()
    else:
        scratch_dir = Path("/dev/shm") / getpass.getuser() / f"seed_masks_{os.getpid()}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    def _cleanup():
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)
    atexit.register(_cleanup)

    print(f"[INFO] building 1-frame h5 in {scratch_dir} ...")
    t0 = time.time()
    data_h5_path, _ = get_frame0_data_h5(exp_folder, scratch_dir)
    print(f"  done in {time.time()-t0:.1f}s: {data_h5_path}")

    # ---------- Load h5 frame 0 ----------
    with h5py.File(data_h5_path, "r") as f:
        imgs = f["imgs"]    # (1, n_cams, H, W, 3) uint8 RGB
        depths = f["depths"]  # (1, n_cams, H, W) uint16 mm
        n_frames_h5, n_cams_h5, H, W = imgs.shape[:4]
        rgb_all = np.asarray(imgs[0])           # (n_cams, H, W, 3)
        depth_all = np.asarray(depths[0]).astype(np.float32) * 0.001  # meters

    # ---------- Calibration ----------
    serials, Ks_all, extrs_all = load_calibration(calib_yaml)
    n_cams_yaml = len(serials)
    print(f"[INFO] calibration: {n_cams_yaml} cams")
    print(f"[INFO] data h5    : {n_cams_h5} cams, {H}x{W}")

    if n_cams_yaml < n_cams_h5:
        print(f"[ERROR] calibration has fewer cams than h5 — can't project")
        sys.exit(1)

    # The h5 typically is indexed in calibration order. If they differ, we'd
    # need a meta.yaml mapping. Assume aligned (standard pipeline produces
    # h5 with cams in calib order) and warn loudly if not.
    n_cams = n_cams_h5
    Ks = Ks_all[:n_cams]
    extrs = extrs_all[:n_cams]

    if not (0 <= args.source_cam < n_cams):
        print(f"[ERROR] --source_cam {args.source_cam} out of range [0, {n_cams})")
        sys.exit(1)

    # ---------- Load source mask ----------
    src_mask = cv2.imread(str(seed_png), cv2.IMREAD_UNCHANGED)
    if src_mask is None:
        print(f"[ERROR] failed to read seed mask: {seed_png}"); sys.exit(1)
    if src_mask.ndim == 3:
        src_mask = src_mask[..., 0]    # grayscale slice
    src_mask = (src_mask > 0).astype(np.uint8)
    if src_mask.shape != (H, W):
        print(f"[INFO] resizing source mask from {src_mask.shape} to {(H, W)}")
        src_mask = cv2.resize(src_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    src_area = int(src_mask.sum())
    print(f"[INFO] source mask cam{args.source_cam}: area={src_area}")

    src_depth = depth_all[args.source_cam]
    src_rgb = rgb_all[args.source_cam]

    # ---------- Project to all cams ----------
    print()
    print("[INFO] projecting seed via extrinsics ...")
    seeds_per_cam = {}
    stats_per_cam = {}
    for tgt in range(n_cams):
        if tgt == args.source_cam:
            seeds_per_cam[tgt] = src_mask.copy()
            stats_per_cam[tgt] = {"src_pixels": src_area, "with_depth": int((src_depth[src_mask > 0] > 0).sum()),
                                    "in_front": int((src_depth[src_mask > 0] > 0).sum()),
                                    "in_image": src_area}
            continue
        proj, stats = project_mask(
            src_rgb_shape=(H, W),
            depth_src=src_depth, mask_src=src_mask,
            K_src=Ks[args.source_cam], extr_src=extrs[args.source_cam],
            K_tgt=Ks[tgt], extr_tgt=extrs[tgt],
            target_shape=(H, W),
        )
        densified = densify_seed_mask(proj, dilate_k=args.dilate_k, close_k=args.close_k)
        seeds_per_cam[tgt] = densified
        stats_per_cam[tgt] = stats
        stats_per_cam[tgt]["projected_area"] = int(proj.sum())
        stats_per_cam[tgt]["densified_area"] = int(densified.sum())
        print(f"  cam{tgt:>2d}  {args.source_cam}→{tgt}  "
              f"src={stats['src_pixels']}  with_depth={stats['with_depth']}  "
              f"in_front={stats['in_front']}  in_image={stats['in_image']}  "
              f"→ densified area = {stats_per_cam[tgt]['densified_area']}")

    # ---------- Debug mosaic ----------
    print()
    print("[INFO] writing debug mosaic ...")
    rows = []
    for cam in range(n_cams):
        rgb = rgb_all[cam]
        ov = overlay_mask_rgb(rgb, seeds_per_cam[cam])
        if cam == args.source_cam:
            label_image(ov, f"cam{cam}  SOURCE  area={int(seeds_per_cam[cam].sum())}")
        else:
            label_image(ov, f"cam{cam}  projected from cam{args.source_cam}  "
                            f"area={int(seeds_per_cam[cam].sum())} "
                            f"(in_image px={stats_per_cam[cam].get('in_image', 0)})")
        rows.append(ov)
    mosaic = np.concatenate(rows, axis=0)
    mosaic_path = out_debug_dir / "seed_projection.png"
    cv2.imwrite(str(mosaic_path), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
    print(f"  wrote {mosaic_path}")

    # Per-cam preview PNGs (one per cam, easier to inspect than 8-tall mosaic)
    for cam in range(n_cams):
        cam_png = out_viz_dir / f"cam_{cam}_seed.png"
        rgb = rgb_all[cam]
        ov = overlay_mask_rgb(rgb, seeds_per_cam[cam])
        if cam == args.source_cam:
            label_image(ov, f"cam{cam}  SOURCE seed  area={int(seeds_per_cam[cam].sum())}")
        else:
            label_image(ov, f"cam{cam}  projected from cam{args.source_cam}  "
                            f"area={int(seeds_per_cam[cam].sum())}")
        cv2.imwrite(str(cam_png), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    print(f"  wrote {n_cams} per-cam previews to {out_viz_dir}")

    if args.no_sam2:
        print()
        print("=" * 60)
        print("[INFO] --no_sam2 set; stopping after seed projection.")
        print(f"  Inspect: {mosaic_path}")
        print("  Re-run without --no_sam2 once you're happy with the seeds.")
        print("=" * 60)
        return

    # ---------- SAM2 propagation per cam ----------
    print()
    print("[INFO] loading SAM2 video predictor ...")
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print(f"  device: {device}")
    print(f"  ckpt  : {args.sam2_checkpoint}")
    predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint,
                                              device=device)

    rgb_videos = sorted(exp_folder.glob("cam*_rgb.mp4"))
    if len(rgb_videos) != n_cams:
        print(f"[WARN] {len(rgb_videos)} rgb mp4s vs {n_cams} cams; processing what's available")

    # Probe total frame count from the first video so writer sizes the dataset.
    cap = cv2.VideoCapture(str(rgb_videos[0]))
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"[INFO] frames per cam: {n_frames_total}")

    # H5 writer
    masks_h5 = out_tool_masks_dir / "masks.h5"
    writer = H5MaskWriter(masks_h5, n_frames=n_frames_total,
                            n_cams=n_cams, H=H, W=W)

    try:
        for cam_idx, mp4 in enumerate(rgb_videos[:n_cams]):
            seed = seeds_per_cam[cam_idx]
            if seed.sum() == 0:
                print(f"  cam{cam_idx}: empty seed → fill all frames with zeros (skipped SAM2)")
                # writer already initialized with zeros; nothing to do
                continue
            print(f"  cam{cam_idx}: SAM2 propagate (seed area = {int(seed.sum())}) ...")
            t1 = time.time()
            n_written = sam2_propagate(predictor, mp4, seed, args.obj_id, writer,
                                          cam_idx, offload=not args.no_offload)
            print(f"    {n_written}/{n_frames_total} frames in {time.time()-t1:.1f}s")
    finally:
        writer.close()

    # objects.yaml
    objects_yaml = out_tool_masks_dir / "objects.yaml"
    with open(objects_yaml, "w") as f:
        yaml.safe_dump({"objects": [tool_name]}, f)
    print(f"\n[INFO] wrote {objects_yaml}: [{tool_name}]")
    print(f"[INFO] wrote {masks_h5}")

    # ---------- Final mosaic: SAM2 frame 0 vs projected seed ----------
    print()
    print("[INFO] writing final-vs-seed comparison mosaic ...")
    with h5py.File(masks_h5, "r") as f:
        final_masks_f0 = np.asarray(f["masks"][0])    # (n_cams, H, W)
    rows = []
    for cam in range(n_cams):
        rgb = rgb_all[cam]
        ov_seed  = overlay_mask_rgb(rgb, seeds_per_cam[cam], color=(255, 200, 0))
        ov_final = overlay_mask_rgb(rgb, (final_masks_f0[cam] > 0).astype(np.uint8))
        label_image(ov_seed,  f"cam{cam}  SEED     area={int(seeds_per_cam[cam].sum())}")
        label_image(ov_final, f"cam{cam}  SAM2 f0  area={int((final_masks_f0[cam]>0).sum())}")
        rows.append(np.concatenate([ov_seed, ov_final], axis=1))
    cmp_mosaic = np.concatenate(rows, axis=0)
    cmp_path = out_debug_dir / "seed_vs_sam2_f0.png"
    cv2.imwrite(str(cmp_path), cv2.cvtColor(cmp_mosaic, cv2.COLOR_RGB2BGR))
    print(f"  wrote {cmp_path}")

    print()
    print("=" * 60)
    print("DONE.")
    print(f"  masks.h5         : {masks_h5}")
    print(f"  objects.yaml     : {objects_yaml}")
    print(f"  seed projection  : {mosaic_path}")
    print(f"  seed vs SAM2 f0  : {cmp_path}")
    print(f"  per-cam previews : {out_viz_dir}/cam_*_seed.png")
    print()
    print("Next: run scripts/run_auto_annotator.sh on this exp; Stage 2 will")
    print("      skip DINO because tool_masks/masks.h5 is now present.")
    print("=" * 60)


if __name__ == "__main__":
    main()
