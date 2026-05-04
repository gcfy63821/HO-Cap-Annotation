#!/usr/bin/env python3
"""Re-generate tool_masks/masks.h5 for every exp queued by simple_mask_annotator.py.

Reads <annotated_root>/_mask_redo_log.csv (or scans for mask_prompts.json
files) and, for every row with redo_status='pending', does:

  1. Load frame 0 of every cam{i}_rgb.mp4 in <videos_root>/<task>/<exp>/.
  2. For each cam that has clicks in mask_prompts.json, run SAM2 image
     predictor with those clicks → per-cam frame-0 seed.
  3. Run SAM2 video predictor (propagate_in_video) per cam, write the full
     per-frame mask stream to a scratch h5.
  4. Copy scratch h5 → <annotated>/<task>/<exp>/tool_masks/masks.h5 and
     write objects.yaml. Update the CSV row to redo_status='done' (or
     'failed' on error).

This is the offline counterpart to scripts/simple_mask_annotator.py — the
interactive tool only saves clicks; this script burns the GPU time. Submit
under sbatch with --mem=128G --gres=gpu:1 for long videos (SAM2's video
predictor preloads the decoded mp4 into CPU RAM, ~6 MB × n_frames per cam).

Usage:
    python scripts/batch_redo_masks.py --videos_root /viscam/.../videos_0102 \\
        [--scratch_dir $HOME/.cache/batch_redo] \\
        [--only_pending]                  (default: process pending+failed; "kept" never)
        [--dry_run]
        [--sam2_checkpoint /abs/path.pt]  (env: SAM2_CKPT)
        [--sam2_model_cfg cfg]
"""

import argparse
import csv
import datetime
import gc
import getpass
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml

HOCAP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HOCAP_ROOT / "scripts"))

# Reuse the chunked gzip uint8 writer the rest of the pipeline uses.
from seed_masks_from_extrinsics import H5MaskWriter    # noqa: E402

REDO_LOG_FIELDS = [
    "task", "exp", "tool_name", "n_cams_with_clicks", "n_clicks_total",
    "prompts_path", "saved_at", "redo_status", "redo_done_at",
    "redo_h5_path", "redo_log",
]


# ============================================================
# CSV helpers (mirror simple_mask_annotator.py)
# ============================================================

def _read_redo_log(annotated_root: Path):
    p = annotated_root / "_mask_redo_log.csv"
    if not p.exists():
        return []
    with open(p, "r", newline="") as f:
        return list(csv.DictReader(f))


def _write_redo_log(annotated_root: Path, rows):
    p = annotated_root / "_mask_redo_log.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REDO_LOG_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in REDO_LOG_FIELDS})


# ============================================================
# Frame-0 readers
# ============================================================

def load_frame0_per_cam(exp_folder: Path):
    rgb_videos = sorted(exp_folder.glob("cam*_rgb.mp4"))
    if not rgb_videos:
        raise FileNotFoundError(f"no cam*_rgb.mp4 under {exp_folder}")
    rgbs, n_frames = [], None
    for p in rgb_videos:
        cap = cv2.VideoCapture(str(p))
        if n_frames is None:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"frame 0 read failed: {p}")
        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(rgbs, axis=0), rgb_videos, n_frames


# ============================================================
# Per-exp processing
# ============================================================

def process_one_exp(row: dict, videos_root: Path, annotated_root: Path,
                      sam2_image_pred, sam2_video_pred, scratch_dir: Path,
                      dry_run: bool = False):
    """Returns the updated row (status filled in)."""
    task = row["task"]; exp = row["exp"]
    exp_folder = videos_root / task / exp
    ann_dir = annotated_root / task / exp
    prompts_path = Path(row.get("prompts_path") or (ann_dir / "mask_prompts.json"))

    log_lines = []
    def log(s):
        print(f"  [{task}/{exp}] {s}", flush=True)
        log_lines.append(s)

    if not prompts_path.exists():
        row["redo_status"] = "failed"
        row["redo_log"] = "prompts file missing"
        row["redo_done_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        return row
    try:
        prompts = json.loads(prompts_path.read_text())
    except Exception as e:
        row["redo_status"] = "failed"
        row["redo_log"] = f"bad JSON: {e}"
        row["redo_done_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        return row

    # Frame 0 for every cam.
    rgbs, rgb_videos, n_frames = load_frame0_per_cam(exp_folder)
    n_cams, H, W = rgbs.shape[:3]
    log(f"loaded frame 0 for {n_cams} cams, n_frames={n_frames}, H×W={H}×{W}")

    # Build per-cam seed via SAM2 image predictor on the stored clicks.
    clicks_by_cam = prompts.get("clicks") or {}
    seeds = [None] * n_cams
    for k, clicks in clicks_by_cam.items():
        cam = int(k)
        if not clicks or cam >= n_cams: continue
        sam2_image_pred.set_image(rgbs[cam])
        pts = np.array([[c["x"], c["y"]] for c in clicks], dtype=np.float32)
        lbls = np.array([c["label"] for c in clicks], dtype=np.int32)
        masks, _, _ = sam2_image_pred.predict(point_coords=pts, point_labels=lbls,
                                                 multimask_output=False)
        seeds[cam] = masks[0].astype(np.uint8)
        log(f"  cam{cam}: {len(clicks)} clicks → seed area={int(seeds[cam].sum())}")
    n_with = sum(1 for s in seeds if s is not None and s.sum() > 0)
    if n_with == 0:
        row["redo_status"] = "failed"
        row["redo_log"] = "no non-empty seeds"
        row["redo_done_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        return row

    if dry_run:
        log(f"[dry_run] would propagate {n_with} cams")
        row["redo_status"] = "dry_run"
        row["redo_done_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        return row

    # Propagate per cam to scratch h5.
    scratch_h5 = scratch_dir / f"{task}__{exp}__propagated_masks.h5"
    if scratch_h5.exists(): scratch_h5.unlink()
    writer = H5MaskWriter(scratch_h5, n_frames=n_frames, n_cams=n_cams, H=H, W=W)
    import torch    # noqa: E402
    try:
        for cam_idx, mp4 in enumerate(rgb_videos[:n_cams]):
            seed = seeds[cam_idx]
            if seed is None or seed.sum() == 0:
                log(f"  cam{cam_idx}: skip (empty seed)")
                continue
            try:
                state = sam2_video_pred.init_state(video_path=str(mp4),
                                                       offload_video_to_cpu=True,
                                                       offload_state_to_cpu=True)
            except TypeError:
                state = sam2_video_pred.init_state(video_path=str(mp4))
            n_written = 0
            try:
                try: sam2_video_pred.reset_state(state)
                except Exception: pass
                sam2_video_pred.add_new_mask(state, frame_idx=0, obj_id=1,
                                                 mask=(seed > 0).astype(np.uint8))
                for fr_idx, oids, mlogits in sam2_video_pred.propagate_in_video(state):
                    for i, oid in enumerate(oids):
                        if oid != 1: continue
                        m = mlogits[i] if isinstance(mlogits, (list, tuple)) else mlogits
                        if torch.is_tensor(m):
                            m = (m > 0).cpu().numpy()
                        else:
                            m = (m > 0)
                        if m.ndim == 3: m = m[0] if m.shape[0] == 1 else m
                        if m.ndim == 4: m = m[0, 0]
                        writer.write_frame(fr_idx, cam_idx, m.astype(np.uint8))
                        n_written += 1
                log(f"  cam{cam_idx}: propagated {n_written} frames")
            except Exception as e:
                log(f"  cam{cam_idx}: PROPAGATE ERROR: {e}")
                raise
            finally:
                try: sam2_video_pred.reset_state(state)
                except Exception: pass
                del state
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
    finally:
        writer.close()

    # Copy scratch → final.
    out_dir = ann_dir / "tool_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_h5 = out_dir / "masks.h5"
    if final_h5.exists(): final_h5.unlink()
    shutil.copy2(scratch_h5, final_h5)
    try: scratch_h5.unlink()
    except Exception: pass

    tool_name = (prompts.get("tool_name") or row.get("tool_name") or "").strip()
    if not tool_name:
        # Fallback: strip trailing _<digits> from exp name (matches the
        # interactive annotator's heuristic).
        import re as _re
        tool_name = _re.sub(r"_\d+$", "", exp)
    with open(out_dir / "objects.yaml", "w") as f:
        yaml.safe_dump({"objects": [tool_name]}, f)
    log(f"wrote {final_h5}  tool_name={tool_name}")

    row["redo_status"] = "done"
    row["redo_done_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    row["redo_h5_path"] = str(final_h5)
    row["redo_log"] = " | ".join(log_lines[-10:])
    if not row.get("tool_name"):
        row["tool_name"] = tool_name
    return row


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_root", required=True)
    ap.add_argument("--scratch_dir", default="",
                    help="Where to stage per-exp scratch h5. Default: "
                         "$HOME/.cache/batch_redo_masks.")
    ap.add_argument("--only_pending", action="store_true",
                    help="Only process rows with redo_status=='pending'. "
                         "By default we also retry 'failed' rows.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Build seeds but don't propagate or write masks.h5.")
    ap.add_argument("--sam2_checkpoint", type=str,
                    default=os.environ.get(
                        "SAM2_CKPT",
                        os.environ.get(
                            "SAM2_ROOT",
                            "/viscam/u/chenrq/crq_ws/robotool/sam2",
                        ) + "/checkpoints/sam2.1_hiera_large.pt"
                    ))
    ap.add_argument("--sam2_model_cfg", type=str,
                    default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = ap.parse_args()

    videos_root = Path(args.videos_root).expanduser().resolve()
    annotated_root = videos_root.parent / f"{videos_root.name}_annotated"
    if not annotated_root.is_dir():
        print(f"[ERR] annotated root missing: {annotated_root}")
        sys.exit(1)

    rows = _read_redo_log(annotated_root)
    if not rows:
        print(f"[INFO] no entries in {annotated_root}/_mask_redo_log.csv")
        return

    # Pick the rows we should process.
    eligible = []
    for r in rows:
        st = (r.get("redo_status") or "").strip()
        if st == "kept" or st == "done":
            continue
        if args.only_pending and st != "pending":
            continue
        eligible.append(r)
    print(f"[INFO] {len(eligible)} / {len(rows)} rows to process "
          f"({'pending only' if args.only_pending else 'pending + failed'})")
    if not eligible:
        return

    scratch_dir = (Path(args.scratch_dir).expanduser().resolve()
                    if args.scratch_dir else
                    Path.home() / ".cache" / "batch_redo_masks")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Lazy SAM2 build.
    print("[sam2] building image predictor ...", flush=True)
    import torch
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_model = build_sam2(args.sam2_model_cfg, args.sam2_checkpoint, device=device)
    img_pred = SAM2ImagePredictor(img_model)
    print("[sam2] building video predictor ...", flush=True)
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    vid_pred = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint,
                                            device=device)
    print(f"[sam2] device={device}")

    n_done = n_fail = 0
    for i, row in enumerate(eligible, 1):
        print(f"\n[{i}/{len(eligible)}] {row['task']}/{row['exp']}  "
              f"(status={row.get('redo_status')})", flush=True)
        try:
            updated = process_one_exp(row, videos_root, annotated_root,
                                         img_pred, vid_pred, scratch_dir,
                                         dry_run=args.dry_run)
        except Exception as e:
            traceback.print_exc()
            updated = {**row,
                          "redo_status": "failed",
                          "redo_done_at": datetime.datetime.now().isoformat(timespec="seconds"),
                          "redo_log": f"exception: {e}"}
        # Splice updated row back into the global list (find by task+exp).
        for j, r in enumerate(rows):
            if (r.get("task"), r.get("exp")) == (updated["task"], updated["exp"]):
                rows[j] = {**r, **updated}
                break
        # Persist after every exp (resumable on crash / ctrl-c).
        _write_redo_log(annotated_root, rows)
        if updated.get("redo_status") == "done": n_done += 1
        elif updated.get("redo_status") == "failed": n_fail += 1

    print(f"\n[INFO] done. ok={n_done}  fail={n_fail}  "
          f"log={annotated_root}/_mask_redo_log.csv")


if __name__ == "__main__":
    main()
