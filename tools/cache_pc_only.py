#!/usr/bin/env python
"""
Cache per-camera point clouds to <out>/cached_pc/cam{i}_uncropped.ply.

This is a minimal, memory-frugal subset of tools/00-0_align_cameras.py that
ONLY does the caching step — it does NOT run colored ICP, does NOT estimate
normals, and does NOT write the large prealign_full.ply / posts/ / ref_pc_stages/
intermediates. Use it when you just want cached PCs for manual alignment
(scripts/manual_align_viser.py) without blowing up RAM.

Each PLY contains xyz + rgb, already transformed to the world frame using the
given extrinsic. Points with invalid depth (0 or > --depth_trunc meters) are
dropped. Points outside --x/y/z_threshold are dropped (defaults are very wide
so nothing is cropped — override if you want tighter bounds).
"""
import argparse
import math
import os
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
import yaml
from tqdm import tqdm


def load_cams(extrinsic_file):
    with open(extrinsic_file, "r") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "extrinsics" in data:
        return [v for k, v in data["extrinsics"].items() if not k.startswith("tag_")]
    if isinstance(data, list):
        return sorted(data, key=lambda c: c.get("camera_id", 0))
    raise ValueError(f"Unsupported YAML format in {extrinsic_file}")


def build_pc(rgb, depth, intrinsic, extrinsic, depth_scale, depth_trunc,
             x_range, y_range, z_range):
    """Return o3d.geometry.PointCloud in world frame (xyz+rgb, no normals)."""
    H, W = depth.shape
    depth_m = depth.astype(np.float32) / depth_scale

    # Validity mask on depth
    valid = (depth_m > 0.01) & (depth_m < depth_trunc)
    if not valid.any():
        return o3d.geometry.PointCloud()

    # Use the same FOV math as 00-0 (vertical fov from fy)
    fy = float(intrinsic[1][1])
    fov_v = 2.0 * math.atan(H / (2.0 * fy))
    f = H / (2.0 * math.tan(fov_v / 2.0))

    # Generate pixel grid only for valid pixels (saves memory)
    vy, vx = np.nonzero(valid)
    d = depth_m[vy, vx]
    x = (2.0 * (vx.astype(np.float32) + 0.5) - W) / f * d / 2.0
    y = (2.0 * (vy.astype(np.float32) + 0.5) - H) / f * d / 2.0
    xyz_cam = np.stack([x, y, d], axis=1)  # (N, 3)

    # Transform to world
    ext = np.asarray(extrinsic, dtype=np.float32).reshape(4, 4)
    xyz_h = np.concatenate([xyz_cam, np.ones((xyz_cam.shape[0], 1), dtype=np.float32)], axis=1)
    xyz_world = (ext @ xyz_h.T).T[:, :3]

    # World-bounds crop
    m = (
        (xyz_world[:, 0] > x_range[0]) & (xyz_world[:, 0] < x_range[1])
        & (xyz_world[:, 1] > y_range[0]) & (xyz_world[:, 1] < y_range[1])
        & (xyz_world[:, 2] > z_range[0]) & (xyz_world[:, 2] < z_range[1])
    )
    xyz_world = xyz_world[m]
    rgb_valid = rgb[vy, vx][m].astype(np.float32) / 255.0

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz_world.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(rgb_valid.astype(np.float64))
    return pc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5_file", required=True)
    ap.add_argument("--extrinsic_file", required=True)
    ap.add_argument("--out_path", default=None,
                    help="Output dir (default: extrinsic_file's parent). Caches go in <out>/cached_pc/")
    ap.add_argument("--frame_idx", type=int, default=0)
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--depth_trunc", type=float, default=2.0)
    ap.add_argument("--x_threshold", type=float, nargs=2, default=[-5.0, 5.0])
    ap.add_argument("--y_threshold", type=float, nargs=2, default=[-5.0, 5.0])
    ap.add_argument("--z_threshold", type=float, nargs=2, default=[-5.0, 5.0])
    args = ap.parse_args()

    extrinsic_file = Path(args.extrinsic_file)
    out_path = Path(args.out_path) if args.out_path else extrinsic_file.parent
    cache_dir = out_path / "cached_pc"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cams = load_cams(extrinsic_file)
    n_cams = len(cams)
    print(f"[INFO] {n_cams} cams, writing to {cache_dir}")

    with h5py.File(args.h5_file, "r") as f:
        h5_frames = f["imgs"].shape[0]
        rgbs = f["imgs"][args.frame_idx]     # (N, H, W, 3) uint8
        depths = f["depths"][args.frame_idx]  # (N, H, W) uint16

    print(f"[INFO] h5 has {h5_frames} frames; reading index {args.frame_idx}")
    print(f"[INFO] depth stats across all cams: "
          f"min={depths.min()} max={depths.max()} mean={depths.mean():.1f} "
          f"nonzero_frac={(depths > 0).mean():.3f}")

    if rgbs.shape[0] < n_cams:
        print(f"[WARN] h5 has {rgbs.shape[0]} cams but yaml has {n_cams}; using min")
        n_cams = min(n_cams, rgbs.shape[0])

    n_empty = 0
    for i in tqdm(range(n_cams), desc="caching"):
        rgb = rgbs[i]
        depth = depths[i]
        per_cam_valid = ((depth > 10) & (depth < args.depth_trunc * args.depth_scale)).mean()
        if per_cam_valid < 0.001:
            # <0.1% pixels in [10mm, depth_trunc] — almost certainly warm-up / bad frame
            print(f"  cam{i}: valid depth frac={per_cam_valid*100:.2f}% (mostly zeros)")
        intrinsic = cams[i]["color_intrinsic_matrix"]
        extrinsic = cams[i]["transformation"]
        pc = build_pc(rgb, depth, intrinsic, extrinsic,
                      args.depth_scale, args.depth_trunc,
                      tuple(args.x_threshold), tuple(args.y_threshold),
                      tuple(args.z_threshold))
        n = len(pc.points)
        out_file = cache_dir / f"cam{i}_uncropped.ply"
        if n == 0:
            print(f"  cam{i}: 0 points (bounds too tight or depth empty) — skipping write")
            n_empty += 1
            continue
        o3d.io.write_point_cloud(str(out_file), pc)
        print(f"  cam{i}: {n} pts -> {out_file.name}")

    if n_empty == n_cams:
        print(f"[FAIL] all {n_cams} cams produced 0 points. Most likely the frame_idx "
              f"({args.frame_idx}) in this h5 corresponds to RealSense warm-up where "
              f"depth is still zero. Try building the tiny h5 from a LATER video frame "
              f"(e.g. --cal_frame_idx 200 or 500 in batch_cache_pc.sh).")
        raise SystemExit(2)
    print(f"[DONE] cached {n_cams-n_empty}/{n_cams} cams in {cache_dir}")


if __name__ == "__main__":
    main()
