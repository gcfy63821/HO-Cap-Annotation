#!/usr/bin/env python3
"""Minimal DINO-once + SAM2-propagate tool segmentation.

This is the simplified counterpart to ``tools/dino_tool_segment.py``. It
reuses that file's helper functions but cuts the pipeline down to two
stages:

  Stage 1 — DINO seed at one frame per camera
    * Render the textured mesh from a few angles, mean DINOv2 features
      give a reference vector ``mesh_ref``.
    * For each camera, scan the candidate frames given by
      ``--seed_frames`` (default just frame 0). Pick the patch with the
      highest cosine similarity to ``mesh_ref``, click SAM2 there to get
      a candidate mask, validate the mask's mean DINO feature against
      ``mesh_ref``, keep the best candidate per camera that passes
      ``--seed_min_mesh_sim``.
    * Cameras that never produce a passing seed are left empty.

  Stage 2 — SAM2 video propagation
    * For every seeded camera, the SAM2 video predictor propagates the
      seed mask forward + backward. ``propagate_camera`` is called with
      ``drift_threshold=0`` so it never tries to re-seed via DINO — that
      is the part that makes the full ``dino_tool_segment.py`` slow.
    * Progress is shown as a single tqdm bar across all cameras × all
      frames written.

Outputs (same layout as the full pipeline so downstream tools see no
difference):
  <output_dir>/masks.h5
  <output_dir>/seed_info.json
  <pipeline_tool_masks_dir>/cam{serial}_rgb/{frame:04d}.npz   (+ objects.yaml)

Usage:
  python tools/dino_simple_segment.py \
    --data_h5      .../data00000000.h5 \
    --calib_yaml   .../realsense_calibration_*.yaml \
    --tool_mesh    .../models/<tool>/textured_mesh.obj \
    --output_dir   .../<exp>/dino_auto \
    --pipeline_tool_masks_dir .../<exp>/tool_masks \
    --tool_name    rubber_mallet \
    [--seed_frames "0 100 300 700"] \
    [--seed_min_mesh_sim 0.40] \
    [--chunk_size 200] \
    [--no_viz]
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

# EGL backend for headless mesh rendering — must be set before pyrender import.
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Make sibling modules importable so we can pull helpers from
# dino_tool_segment.py without duplicating their bodies.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dino_tool_segment import (  # noqa: E402
    load_dino,
    build_mesh_reference,
    mesh_dino_seed,
    extract_jpegs,
    propagate_camera,
    export_pipeline_masks,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_h5', required=True)
    p.add_argument('--calib_yaml', required=True,
                   help='Used only to discover camera count; geometry not used here.')
    p.add_argument('--tool_mesh', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--sam2_ckpt',
                   default=os.environ.get('SAM2_CKPT',
                       '/home/ruoqu/crq_ws/robotool/mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt'))
    p.add_argument('--sam2_image_cfg',
                   default=os.environ.get('SAM2_IMAGE_CFG', 'configs/sam2.1/sam2.1_hiera_l.yaml'))
    p.add_argument('--sam2_video_cfg',
                   default=os.environ.get('SAM2_VIDEO_CFG',
                       '/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/config/sam2_config/sam2.1_hiera_l.yaml'))
    p.add_argument('--n_mesh_views', type=int, default=12,
                   help='Mesh angles rendered to build the DINO reference vector.')

    p.add_argument('--seed_frames', type=int, nargs='+', default=None,
                   help='Candidate frame indices for the per-camera seed search. '
                        'Best mesh_sim wins. Default = [0].')
    p.add_argument('--seed_min_area', type=int, default=100)
    p.add_argument('--seed_max_area', type=int, default=15000)
    p.add_argument('--seed_min_sim', type=float, default=0.20,
                   help='Minimum patch-to-mesh DINOv2 similarity to bother clicking SAM2 at.')
    p.add_argument('--seed_min_mesh_sim', type=float, default=None,
                   help='Hard floor on the validated mesh_sim of an accepted seed. '
                        'Defaults to --seed_min_sim. Tighten (e.g. 0.40) to refuse '
                        'wrong-object seeds outright — empty mask is better than wrong mask.')

    p.add_argument('--chunk_size', type=int, default=100,
                   help='SAM2 video predictor chunk size. Larger = fewer init_state '
                        'calls (faster) but more CPU RAM per chunk. 200-300 is fine on 96GB nodes.')

    p.add_argument('--pipeline_tool_masks_dir', type=str, default=None,
                   help='If set, also write per-frame .npz label-masks + objects.yaml '
                        'in the layout generate_meta.py expects.')
    p.add_argument('--tool_name', type=str, default='tool')
    p.add_argument('--cam_serials', type=str, nargs='+', default=None,
                   help='Camera serials in order. Used to name pipeline output folders.')
    p.add_argument('--pipeline_mask_format', choices=['npz', 'npy'], default='npz')

    p.add_argument('--no_viz', action='store_true', default=True,
                   help='Skip writing snapshot PNGs and overlay video. ON by default '
                        '(this is the simple/cluster path); pass --with_viz to opt in.')
    p.add_argument('--with_viz', action='store_true',
                   help='Override --no_viz default: write snapshot PNGs (no MP4).')
    p.add_argument('--cameras', type=int, nargs='+', default=None,
                   help='Restrict processing to these cameras only. Default = all.')
    args = p.parse_args()

    # --no_viz default + --with_viz override
    do_viz = bool(args.with_viz)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / 'tmp_jpegs'

    t_start = time.time()
    def _t(): return f'[{time.time() - t_start:.0f}s]'

    # ── load h5 metadata only ──────────────────────────────────────
    data_h5 = h5py.File(args.data_h5, 'r')
    N = int(data_h5['imgs'].shape[0])
    n_cams = int(data_h5['imgs'].shape[1])
    H = int(data_h5['imgs'].shape[2])
    W = int(data_h5['imgs'].shape[3])
    print(f'[INFO] {N} frames, {n_cams} cameras, image {H}x{W}')

    cams_to_process = list(args.cameras) if args.cameras else list(range(n_cams))

    # ── Stage 1: mesh reference + per-camera DINO seed ─────────────
    print(f'\n{_t()} Loading DINOv2 ...')
    dino = load_dino()

    print(f'\n{_t()} Building mesh DINOv2 reference ({args.n_mesh_views} views) ...')
    mesh_ref = build_mesh_reference(dino, args.tool_mesh, n_views=args.n_mesh_views)
    if mesh_ref is None:
        print('[ERROR] Failed to build mesh reference.')
        data_h5.close(); return

    print(f'\n{_t()} Loading SAM2 image predictor ...')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_img = build_sam2(args.sam2_image_cfg, args.sam2_ckpt, device='cuda:0')
    image_predictor = SAM2ImagePredictor(sam2_img)

    candidate_frames = sorted({fr for fr in (args.seed_frames or [0]) if 0 <= fr < N})
    if not candidate_frames:
        candidate_frames = [0]
    mesh_sim_floor = (args.seed_min_mesh_sim
                      if args.seed_min_mesh_sim is not None else args.seed_min_sim)

    print(f'\n{_t()} === Seeding (candidates={candidate_frames}, '
          f'mesh_sim_floor={mesh_sim_floor:.2f}) ===')
    seeds = {}
    for cam in cams_to_process:
        best = None
        for fr in candidate_frames:
            r = mesh_dino_seed(
                dino, data_h5, cam, mesh_ref, image_predictor,
                frames=[fr],
                min_area=args.seed_min_area,
                max_area=args.seed_max_area,
                min_sim=args.seed_min_sim,
                first_hit=True)
            if r is None:
                continue
            if r['mesh_sim'] < mesh_sim_floor:
                print(f'  cam{cam} frame{fr}: rejected '
                      f'(mesh_sim={r["mesh_sim"]:.3f} < floor {mesh_sim_floor:.2f})')
                continue
            if best is None or r['mesh_sim'] > best['mesh_sim']:
                best = r
        if best:
            seeds[cam] = best
            print(f'  cam{cam}: seed OK at frame {best["frame"]} '
                  f'(area={best["area"]}, mesh_sim={best["mesh_sim"]:.3f})')
        else:
            print(f'  cam{cam}: seed FAILED on all candidates (mask left empty)')

    if not seeds:
        print('[ERROR] No camera produced a valid seed.')
        data_h5.close(); return

    # ── Stage 2: SAM2 video propagation, no re-seeding ─────────────
    print(f'\n{_t()} Loading SAM2 video predictor ...')
    from hocap_annotation.wrappers.sam2 import build_sam2_video_predictor
    video_predictor = build_sam2_video_predictor(
        config_file=args.sam2_video_cfg, ckpt_path=args.sam2_ckpt, device='cuda:0')

    masks_path = output_dir / 'masks.h5'
    seeded_cams = sorted(seeds.keys())
    extract_jpegs(data_h5, tmp_dir, seeded_cams)

    masks_h5 = h5py.File(masks_path, 'w')
    masks_ds = masks_h5.create_dataset(
        'masks', shape=(N, n_cams, H, W),
        dtype=np.uint8, chunks=(1, 1, H, W), compression='gzip')

    print(f'\n{_t()} === Propagating ===')
    from tqdm import tqdm
    total_budget = sum(
        (N - s['frame']) + max(0, s['frame']) for s in seeds.values()
    )
    with tqdm(total=total_budget, desc='SAM2 propagate',
              unit='fr', dynamic_ncols=True, mininterval=1.0) as pbar:
        for cam in seeded_cams:
            s = seeds[cam]
            cam_dir = tmp_dir / f'cam{cam:02d}'
            pbar.set_description(f'SAM2 propagate cam{cam}')
            written, _ = propagate_camera(
                video_predictor, cam_dir, s['mask'], s['frame'], N,
                masks_ds, cam,
                dino=None, image_predictor=None, global_ref=None, data_h5=None,
                ref_area=float(s['mask'].sum()),
                drift_threshold=0.0,        # disable DINO re-seeding entirely
                chunk_size=args.chunk_size,
                pbar=pbar)
            masks_h5.flush()
            pbar.write(f'  cam{cam}: wrote {written} frames')

    # ── tear down models ───────────────────────────────────────────
    del dino, image_predictor, video_predictor
    torch.cuda.empty_cache(); gc.collect()

    # ── seed_info.json (mask removed for size) ─────────────────────
    meta = {f'cam{c}': {k: v for k, v in s.items() if k != 'mask'}
            for c, s in seeds.items()}
    (output_dir / 'seed_info.json').write_text(
        json.dumps(meta, indent=2, default=str)
    )
    masks_h5.close()

    # ── pipeline-format export ─────────────────────────────────────
    if args.pipeline_tool_masks_dir is not None:
        print(f'\n{_t()} === Exporting to pipeline format ===')
        export_pipeline_masks(
            masks_path=masks_path,
            out_dir=Path(args.pipeline_tool_masks_dir),
            tool_name=args.tool_name,
            cam_serials=args.cam_serials,
            n_cams=n_cams,
            n_frames=N,
            fmt=args.pipeline_mask_format,
        )

    # ── viz (off by default) ───────────────────────────────────────
    if do_viz:
        from dino_tool_segment import visualize
        viz_dir = output_dir / 'viz'
        try:
            visualize(data_h5, masks_path, viz_dir, make_video=False)
        except Exception as e:
            print(f'  [WARN] visualize failed: {e}')

    data_h5.close()
    elapsed = time.time() - t_start
    print(f'\n[DONE] dino_simple_segment: {elapsed/60:.1f} min ({elapsed:.0f}s)')


if __name__ == '__main__':
    main()
