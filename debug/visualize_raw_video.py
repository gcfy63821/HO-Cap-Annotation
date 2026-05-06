"""Render the RAW 2x4 tiled camera grid — the un-annotated counterpart of
debug/visualize_hand_object_surface_video.py. Same frame range, frame_step,
tile_w/tile_h, and per-tile labels, so the two mp4s line up frame-for-frame
and you can play them side-by-side or cross-fade between them in an editor.

Output (to mirror the surface script's path):
    <annotated>/<exp>/debug/hand_object_video/raw_tiled.mp4

Two modes:
  (a) Workspace:
      python debug/visualize_raw_video.py \\
          --videos_root /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/videos_dataset
  (b) Single-exp:
      python debug/visualize_raw_video.py \\
          --annotated_sequence /viscam/.../<task>/<exp>
"""

import argparse
import contextlib
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np

HOCAP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HOCAP_ROOT))
sys.path.insert(0, str(HOCAP_ROOT / "debug"))

# Reuse the same helpers the surface script uses, so paths / discovery /
# tile layout stay byte-identical.
from visualize_hand_object_video import (    # noqa: E402
    derive_original_sequence,
    find_h5_with_rgbd,
    VideoRGBSource,
    find_dataset,
    discover_done_exps,
    open_writer,
    concat_frames_grid,
)

import yaml    # noqa: E402


# ============================================================
# Per-exp render
# ============================================================

def render_one_exp(annotated_sequence: Path,
                     start_frame: int = 0, end_frame: int = -1,
                     frame_step: int = 1,
                     fps: int = 20,
                     tile_w: int = 640, tile_h: int = 480,
                     output_path: Path = None,
                     force: bool = False) -> dict:
    annotated_sequence = annotated_sequence.resolve()
    original_sequence = derive_original_sequence(annotated_sequence)
    meta_path = original_sequence / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta.yaml: {meta_path}")
    meta = yaml.safe_load(meta_path.read_text())
    calib_path = Path(meta["calibration_yaml_path"])
    calib = yaml.safe_load(calib_path.read_text())
    serials = [str(c["camera_id"]).zfill(2) for c in calib]

    out_path = output_path if output_path is not None else (
        annotated_sequence / "debug" / "hand_object_video" / "raw_tiled.mp4"
    )
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists (pass --force)")
        return {"output": str(out_path), "skipped": True}

    h5_path = None; video_source = None
    try:
        h5_path = find_h5_with_rgbd(original_sequence)
    except FileNotFoundError:
        video_source = VideoRGBSource(original_sequence, len(serials))

    data_ctx = (contextlib.nullcontext(None) if h5_path is None
                  else h5py.File(h5_path, "r"))
    with data_ctx as data_f:
        imgs_ds = find_dataset(data_f, "imgs") if data_f else None
        n_frames_total = (int(imgs_ds.shape[0]) if imgs_ds is not None
                            else int(video_source.frame_count))
        if end_frame < 0 or end_frame >= n_frames_total:
            end_frame = n_frames_total - 1
        start_frame = max(0, min(start_frame, end_frame))
        chosen = list(range(start_frame, end_frame + 1, max(1, frame_step)))

        n_cams = len(serials)
        grid_w = 4 * tile_w; grid_h = 2 * tile_h
        writer = open_writer(out_path, grid_w, grid_h, fps=fps)

        n_written = 0; t0 = time.time()
        try:
            for fi, frame_idx in enumerate(chosen):
                tiles = []
                for cam_idx in range(n_cams):
                    if imgs_ds is not None:
                        rgb = np.asarray(imgs_ds[frame_idx, cam_idx],
                                          dtype=np.uint8)
                    else:
                        rgb = video_source.get_color(frame_idx, cam_idx)
                    cv2.putText(rgb, f"f{frame_idx:06d} cam{serials[cam_idx]}",
                                  (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                  (0, 0, 0), 4)
                    cv2.putText(rgb, f"f{frame_idx:06d} cam{serials[cam_idx]}",
                                  (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                  (255, 255, 255), 1)
                    if (rgb.shape[1], rgb.shape[0]) != (tile_w, tile_h):
                        rgb = cv2.resize(rgb, (tile_w, tile_h),
                                          interpolation=cv2.INTER_AREA)
                    tiles.append(rgb)
                grid = concat_frames_grid(tiles, grid=(2, 4))
                writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                n_written += 1
                if (fi + 1) % 50 == 0:
                    fps_now = (fi + 1) / max(time.time() - t0, 1e-6)
                    print(f"  [{fi+1}/{len(chosen)}] frame {frame_idx}  "
                          f"{fps_now:.2f} fps")
        finally:
            writer.release()
            if video_source is not None: video_source.close()

    print(f"[OK] wrote {out_path}  ({n_written} frames)")
    return {
        "annotated_sequence": str(annotated_sequence),
        "output": str(out_path),
        "kind": "raw",
        "n_frames_written": n_written,
        "frame_range": [start_frame, end_frame, frame_step],
        "tile_size": [tile_w, tile_h],
        "fps": fps,
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_root", type=Path, default=None)
    ap.add_argument("--annotated_sequence", type=Path, default=None)
    ap.add_argument("--task_filter", nargs="+", default=None)
    ap.add_argument("--only_remasked", action="store_true",
                    help="Workspace mode: only render exps with mask_prompts.json.")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1)
    ap.add_argument("--frame_step", type=int, default=1)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--tile_w", type=int, default=640)
    ap.add_argument("--tile_h", type=int, default=480)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true", default=True)
    args = ap.parse_args()

    if args.videos_root is None and args.annotated_sequence is None:
        ap.error("provide either --videos_root or --annotated_sequence")
    if args.videos_root is not None and args.annotated_sequence is not None:
        ap.error("--videos_root and --annotated_sequence are mutually exclusive")

    common = dict(
        start_frame=args.start_frame, end_frame=args.end_frame,
        frame_step=args.frame_step, fps=args.fps,
        tile_w=args.tile_w, tile_h=args.tile_h,
        force=args.force,
    )

    if args.annotated_sequence is not None:
        try:
            render_one_exp(annotated_sequence=args.annotated_sequence,
                            output_path=args.output_dir, **common)
        except Exception as e:
            print(f"[ERR] {e}", file=sys.stderr)
            traceback.print_exc(); sys.exit(1)
        return

    videos_root = args.videos_root.resolve()
    if not videos_root.is_dir():
        ap.error(f"--videos_root not a directory: {videos_root}")

    exps, n_skip = discover_done_exps(
        videos_root,
        task_filter=set(args.task_filter) if args.task_filter else None,
        only_remasked=args.only_remasked)
    if not exps:
        print(f"No DONE exps under {videos_root} (skipped {n_skip} unfinished)")
        sys.exit(0)
    print(f"[INFO] {len(exps)} done exps  (skipped {n_skip} unfinished)\n")

    n_ok = n_fail = n_skip2 = 0
    failures = []
    t0 = time.time()
    for i, (task, exp, _exp_dir, ann, reason) in enumerate(exps, 1):
        prefix = f"[{i:>4d}/{len(exps)}] {task}/{exp}"
        per_exp_out = (args.output_dir / task / exp / "raw_tiled.mp4"
                        if args.output_dir is not None else None)
        existing = (per_exp_out if per_exp_out is not None else
                     ann / "debug" / "hand_object_video" / "raw_tiled.mp4")
        if existing.exists() and not args.force:
            print(f"{prefix}  skip  (mp4 exists; pass --force)")
            n_skip2 += 1; continue
        try:
            t1 = time.time()
            summary = render_one_exp(annotated_sequence=ann,
                                          output_path=per_exp_out, **common)
            n_ok += 1
            print(f"{prefix}  OK [{time.time()-t1:.1f}s]  "
                  f"{summary.get('n_frames_written', '?')} frames  ({reason})")
        except KeyboardInterrupt:
            print("[INTERRUPTED]"); sys.exit(130)
        except Exception as e:
            n_fail += 1; failures.append((task, exp, str(e)))
            print(f"{prefix}  FAIL  ({type(e).__name__}: {e})")
            if not args.continue_on_error: raise

    print()
    print("=" * 60)
    print(f"summary: ok={n_ok}  skip={n_skip2}  fail={n_fail}  "
          f"elapsed={time.time()-t0:.1f}s")
    if failures:
        print("\nFailures:")
        for task, exp, msg in failures[:10]:
            print(f"  {task}/{exp}: {msg}")
        if len(failures) > 10:
            print(f"  ... and {len(failures)-10} more")
    print("=" * 60)


if __name__ == "__main__":
    main()
