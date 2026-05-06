#!/usr/bin/env python3
"""SAM2 click-and-propagate annotator for DEFORMABLE objects (dough,
fabric, food, etc.) — sister tool to mesh_reconstruction/sam2/notebooks/
batch_task_annotator.py, but writes to a separate annotated subdir so the
rigid-tool tool_masks/masks.h5 isn't overwritten.

Output layout:
    <annotated>/<task>/<exp>/<output_subdir>/masks.h5     (n_frames, n_cams, H, W) uint8
    <annotated>/<task>/<exp>/<output_subdir>/objects.yaml ({objects: [name]})

Default <output_subdir> = "deformable_masks". Use --output_subdir to write
multiple deformable layers (e.g. one masks.h5 per dough piece) by running
this script multiple times with different subdirs and --object_name.

Usage:
    # All cams of one exp, default subdir:
    python scripts/batch_deformable_annotator.py \\
        --exp_folder /viscam/.../videos_0102/<task>/<exp> \\
        --object_name dough

    # Walk every exp under one task:
    python scripts/batch_deformable_annotator.py \\
        --task_folder /viscam/.../videos_0102/<task> \\
        --object_name dough

    # Custom subdir (e.g. second deformable layer):
    python scripts/batch_deformable_annotator.py \\
        --exp_folder ... --object_name fabric \\
        --output_subdir deformable_fabric_masks

Workflow per cam (interactive):
    1. Frame 0 opens in matplotlib. Left-click positive points, right-click
       negative points (use shift+click for negative on macOS / Linux).
    2. Press Enter (or just close the figure) when done.
    3. SAM2 image-predictor preview pops up. Close to confirm + start propagation.
    4. SAM2 video predictor streams masks frame-by-frame into masks.h5.
    5. Move on to next cam.

Resume: cams whose masks.h5 group already contains a non-empty entry are
auto-skipped. Pass --force to redo.

Display the produced masks afterwards via --show_deformable in
debug/visualize_hand_object_video.py and
debug/visualize_hand_object_surface_video.py.
"""

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


# ============================================================
# Helpers
# ============================================================

def video_metadata(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return n, h, w


def extract_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read first frame: {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_to_2d(m) -> np.ndarray:
    if torch.is_tensor(m):
        m = m.cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[0] if m.shape[0] == 1 else m
    if m.ndim == 4:
        m = m[0, 0]
    return m


def cam_idx_of(name: str) -> int:
    """cam0_rgb.mp4 → 0; cam12_rgb.mp4 → 12. Fallback to 0 if unparseable."""
    import re
    m = re.match(r"cam(\d+)", Path(name).stem)
    return int(m.group(1)) if m else 0


# ============================================================
# H5 writer
# ============================================================

class H5MaskWriter:
    """(n_frames, n_cams, H, W) uint8, gzip-compressed, chunk per (1,1,H,W)."""

    def __init__(self, h5_path: Path, n_frames: int, n_cams: int, H: int, W: int):
        self.path = Path(h5_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_frames = n_frames
        self.n_cams = n_cams
        self.H = H
        self.W = W
        # Open or create
        if self.path.exists():
            self._f = h5py.File(self.path, "a")
            if "masks" not in self._f:
                self._ds = self._f.create_dataset(
                    "masks", shape=(n_frames, n_cams, H, W), dtype=np.uint8,
                    chunks=(1, 1, H, W), compression="gzip",
                )
            else:
                self._ds = self._f["masks"]
                # Sanity check shape
                if self._ds.shape != (n_frames, n_cams, H, W):
                    print(f"[WARN] {self.path}: existing masks shape "
                          f"{self._ds.shape} != requested ({n_frames},{n_cams},{H},{W})")
        else:
            self._f = h5py.File(self.path, "w")
            self._ds = self._f.create_dataset(
                "masks", shape=(n_frames, n_cams, H, W), dtype=np.uint8,
                chunks=(1, 1, H, W), compression="gzip",
            )

    def write_frame(self, frame_idx: int, cam_idx: int, mask_2d: np.ndarray):
        if mask_2d.shape != (self.H, self.W):
            mask_2d = cv2.resize(mask_2d, (self.W, self.H),
                                  interpolation=cv2.INTER_NEAREST)
        self._ds[frame_idx, cam_idx] = mask_2d.astype(np.uint8)

    def cam_already_done(self, cam_idx: int) -> bool:
        # Sample frames 0, n//2, n-1 and see if any has nonzero pixels.
        if cam_idx >= self._ds.shape[1]:
            return False
        for fi in (0, self.n_frames // 2, max(self.n_frames - 1, 0)):
            try:
                if int(self._ds[fi, cam_idx].sum()) > 0:
                    return True
            except Exception:
                pass
        return False

    def flush(self):
        try: self._f.flush()
        except Exception: pass

    def close(self):
        try: self._f.close()
        except Exception: pass


# ============================================================
# UI helpers
# ============================================================

def show_mask_overlay(rgb, mask, color=(0, 255, 0)):
    out = rgb.copy()
    if mask is None or mask.sum() == 0: return out
    out[mask > 0] = (0.5 * out[mask > 0] + 0.5 * np.array(color)).astype(np.uint8)
    cnt, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, (255, 255, 0), 2)
    return out


def click_points(rgb, title="click positive (left), negative (right); close to continue"):
    """Returns (points (N,2) float32, labels (N,) int32). Empty = skip cam."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("on")

    pts, lbls = [], []
    def onclick(event):
        if event.xdata is None or event.ydata is None: return
        is_pos = event.button == 1
        pts.append((float(event.xdata), float(event.ydata)))
        lbls.append(1 if is_pos else 0)
        col = "g" if is_pos else "r"
        marker = "*" if is_pos else "x"
        ax.plot(event.xdata, event.ydata, marker=marker, color=col,
                markersize=14, markeredgecolor="white", markeredgewidth=1.5)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show(block=True)
    if not pts:
        return None, None
    return np.array(pts, dtype=np.float32), np.array(lbls, dtype=np.int32)


# ============================================================
# Per-cam annotation
# ============================================================

def annotate_cam(video_path: Path, predictor, obj_id: int,
                   cam_name: str, cam_idx: int,
                   exp_label: str, writer: H5MaskWriter,
                   force: bool = False):
    print(f"\n[{exp_label}] cam={cam_name}  cam_idx={cam_idx}")

    if not force and writer.cam_already_done(cam_idx):
        print(f"  [resume] cam {cam_name} already has masks — skip (pass --force to redo)")
        return 0

    n_frames, H, W = video_metadata(video_path)
    print(f"  video: {n_frames} frames @ {H}x{W}")

    t0 = time.time()
    try:
        state = predictor.init_state(
            video_path=str(video_path),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
    except TypeError:
        state = predictor.init_state(video_path=str(video_path))
    try:
        predictor.reset_state(state)
    except Exception:
        pass
    print(f"  SAM2 init_state: {time.time()-t0:.1f}s")

    try:
        # 1. Click points on first frame.
        plt.close("all")
        rgb0 = extract_first_frame(video_path)
        pts, lbls = click_points(
            rgb0,
            title=f"[{exp_label}/{cam_name}] click positive (left) / negative (right) → close",
        )
        if pts is None:
            print(f"  no clicks → skipping this cam")
            return 0
        print(f"  {len(pts)} clicks  (+{int((lbls==1).sum())} / -{int((lbls==0).sum())})")

        # 2. Add points → preview.
        _, oids, mlogits = predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=obj_id,
            points=pts, labels=lbls,
        )
        m0 = mlogits[0] if isinstance(mlogits, (list, tuple)) else mlogits
        m0_2d = mask_to_2d((m0 > 0).cpu().numpy() if torch.is_tensor(m0) else (m0 > 0)).astype(np.uint8)
        if m0_2d.shape != (H, W):
            m0_2d = cv2.resize(m0_2d, (W, H), interpolation=cv2.INTER_NEAREST)

        # Preview window — close to start propagation.
        preview = show_mask_overlay(rgb0, m0_2d, color=(220, 50, 200))
        plt.figure(figsize=(10, 10))
        plt.imshow(preview)
        plt.title(f"[{exp_label}/{cam_name}] preview — close to propagate, "
                   "Ctrl-C in terminal to abort")
        plt.axis("off")
        plt.show()

        # 3. Propagate, write to disk per frame.
        print("  propagating masks ...")
        t1 = time.time()
        n_written = 0
        last_log = time.time()
        for fr_idx, oids_p, mlogits_p in predictor.propagate_in_video(state):
            for i, oid in enumerate(oids_p):
                if oid != obj_id: continue
                m = mlogits_p[i] if isinstance(mlogits_p, (list, tuple)) else mlogits_p
                mb = (m > 0).cpu().numpy() if torch.is_tensor(m) else (m > 0)
                m2d = mask_to_2d(mb).astype(np.uint8)
                writer.write_frame(fr_idx, cam_idx, m2d)
                n_written += 1
                del m2d, mb
            if time.time() - last_log > 5:
                elapsed = time.time() - t1
                fps = n_written / max(elapsed, 1e-6)
                print(f"    {n_written}/{n_frames} frames  ({fps:.1f} fps)")
                last_log = time.time()
        writer.flush()
        print(f"  wrote {n_written} masks for cam {cam_name}  ({time.time()-t1:.1f}s)")
        return n_written
    finally:
        try: predictor.reset_state(state)
        except Exception: pass
        del state
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()


# ============================================================
# Per-exp dispatch
# ============================================================

def annotate_exp(exp_folder: Path, ann_dir: Path, predictor,
                   object_name: str, output_subdir: str = "deformable_masks",
                   force: bool = False):
    rgb_videos = sorted(exp_folder.glob("cam*_rgb.mp4"))
    if not rgb_videos:
        print(f"[skip] {exp_folder.name}: no cam*_rgb.mp4")
        return
    n_frames, H, W = video_metadata(rgb_videos[0])
    n_cams = len(rgb_videos)

    out_dir = ann_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "masks.h5"
    writer = H5MaskWriter(h5_path, n_frames, n_cams, H, W)

    # Save objects.yaml so downstream tools can name the layer.
    yaml_path = out_dir / "objects.yaml"
    if not yaml_path.exists() or force:
        yaml_path.write_text(yaml.safe_dump({"objects": [object_name]}))
    else:
        try:
            existing = yaml.safe_load(yaml_path.read_text()) or {}
            if (existing.get("objects") or [None])[0] != object_name:
                print(f"  [WARN] existing objects.yaml has different name: "
                      f"{existing.get('objects')}; keeping it (pass --force to overwrite)")
        except Exception:
            pass

    obj_id = 1
    try:
        for cam_idx, video_path in enumerate(rgb_videos):
            cam_name = video_path.stem    # cam0_rgb, cam1_rgb, ...
            try:
                annotate_cam(video_path, predictor, obj_id, cam_name, cam_idx,
                              exp_label=exp_folder.name, writer=writer, force=force)
            except KeyboardInterrupt:
                print("[INTERRUPTED]")
                raise
            except Exception as e:
                print(f"[ERR] cam {cam_name}: {e}")
                traceback.print_exc()
                continue
    finally:
        writer.close()
    print(f"[OK] {exp_folder.name} → {h5_path}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp_folder", type=Path, default=None,
                    help="Path to one <videos_root>/<task>/<exp>.")
    g.add_argument("--task_folder", type=Path, default=None,
                    help="Path to <videos_root>/<task>; iterates over all exps.")
    ap.add_argument("--object_name", type=str, default="deformable",
                    help="Name written to <output_subdir>/objects.yaml. "
                         "Default 'deformable'. Use 'dough' / 'fabric' / etc.")
    ap.add_argument("--output_subdir", type=str, default="deformable_masks",
                    help="Subdir under <annotated>/<task>/<exp>/ where masks.h5 "
                         "+ objects.yaml are written. Default 'deformable_masks'. "
                         "Use a different value to keep multiple deformable "
                         "layers separate (e.g. 'deformable_dough_masks').")
    ap.add_argument("--annotated_root", type=Path, default=None,
                    help="Override <videos_root>_annotated location. Default: "
                         "auto-derive from --exp_folder / --task_folder parent.")
    ap.add_argument("--exps", nargs="+", default=None,
                    help="(--task_folder mode) only process these exp names.")
    ap.add_argument("--force", action="store_true",
                    help="Re-do cams that already have non-empty masks.")
    ap.add_argument("--sam2_checkpoint", type=str,
                    default=os.environ.get(
                        "SAM2_CKPT",
                        os.environ.get(
                            "SAM2_ROOT",
                            "/viscam/u/chenrq/crq_ws/robotool/sam2",
                        ) + "/checkpoints/sam2.1_hiera_large.pt",
                    ))
    ap.add_argument("--sam2_model_cfg", type=str,
                    default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = ap.parse_args()

    # Resolve task / exp / annotated paths.
    if args.exp_folder is not None:
        exp_folder = args.exp_folder.resolve()
        task_folder = exp_folder.parent
        forced_exp_names = [exp_folder.name]
    else:
        task_folder = args.task_folder.resolve()
        forced_exp_names = None

    videos_root = task_folder.parent
    if args.annotated_root is not None:
        annotated_root = args.annotated_root.resolve()
    else:
        annotated_root = videos_root.parent / f"{videos_root.name}_annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)

    # Build SAM2 video predictor (lazy import so --help works without torch).
    print("[sam2] loading video predictor ...", flush=True)
    from sam2.build_sam import build_sam2_video_predictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint,
                                              device=device)
    print(f"[sam2] device={device}")

    # Pick exps.
    if forced_exp_names is not None:
        exp_names = forced_exp_names
    else:
        exp_names = sorted(d.name for d in task_folder.iterdir() if d.is_dir())
        if args.exps:
            keep = set(args.exps)
            exp_names = [n for n in exp_names if n in keep]

    if not exp_names:
        print("no exps to process"); sys.exit(0)
    print(f"task_folder    = {task_folder}")
    print(f"annotated_root = {annotated_root}")
    print(f"output_subdir  = {args.output_subdir}")
    print(f"object_name    = {args.object_name}")
    print(f"exps to do     = {len(exp_names)}")
    print()

    for i, exp_name in enumerate(exp_names, 1):
        exp_folder = task_folder / exp_name
        if not exp_folder.is_dir(): continue
        if not list(exp_folder.glob("cam*_rgb.mp4")): continue
        ann_dir = annotated_root / task_folder.name / exp_name
        print("=" * 60)
        print(f"[{i}/{len(exp_names)}] {task_folder.name}/{exp_name}")
        print("=" * 60)
        try:
            annotate_exp(exp_folder, ann_dir, predictor,
                          object_name=args.object_name,
                          output_subdir=args.output_subdir,
                          force=args.force)
        except KeyboardInterrupt:
            print("[INTERRUPTED]"); sys.exit(130)
        except Exception as e:
            print(f"[ERR] {exp_name}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
