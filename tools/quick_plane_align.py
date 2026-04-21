#!/usr/bin/env python
"""
Minimal one-shot table-plane alignment.

Very lightweight alternative to tools/00-3_align_cameras_global.py:
  - loads the cached per-camera PLYs (in original world frame),
  - merges them and voxel-downsamples,
  - fits ONE dominant plane via RANSAC,
  - computes the single rigid transform T that rotates the plane to z=0 and
    drops the table to z=0,
  - applies that SAME T to every camera's extrinsic,
  - writes <stem>_global_aligned.yaml and postalign_global.ply next to the
    input.

This does NOT do any pairwise ICP or pose-graph optimization. It assumes the
initial extrinsics are already roughly consistent (which is the usual case
from RealSense marker-board calibration) and only fixes the world orientation
so the table is flat at z=0. Runs in a few seconds with low memory.

Usage:
    python tools/quick_plane_align.py \
        --cached_pc /.../realsense_calibrate_XXXX/cached_pc \
        --extrinsic_file /.../realsense_calibration_XXXX.yaml \
        [--out_path /.../realsense_calibrate_XXXX]
        [--voxel 0.005] [--plane_dist_thresh 0.01]
        [--crop_xy 1.0]   # ignore points farther than |xy|>crop for plane fit
        [--z_min -0.3 --z_max 0.6]   # z range to consider for plane fit
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


def load_extrinsics(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "extrinsics" in data:
        return [v for k, v in data["extrinsics"].items() if not k.startswith("tag_")]
    if isinstance(data, list):
        return sorted(data, key=lambda x: x.get("camera_id", 0))
    raise ValueError(f"Unsupported YAML format: {path}")


def plane_to_z0_transform(plane_model):
    """Rigid transform that maps points on plane_model to z=0 with +z up."""
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
        d = -d
    target = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, target)
    s = np.linalg.norm(v)
    c_ = float(np.dot(n, target))
    if s < 1e-8:
        R = np.eye(3) if c_ > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c_) / (s * s))
    T = np.eye(4)
    T[:3, :3] = R
    T[2, 3] = d      # after rotating, plane sits at z=d; shift it to z=0
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cached_pc", "--cached_pc_dir", dest="cached_pc", required=True)
    ap.add_argument("--extrinsic_file", required=True)
    ap.add_argument("--out_path", default=None,
                    help="output directory (default: extrinsic_file's parent)")
    ap.add_argument("--voxel", type=float, default=0.005,
                    help="voxel size (m) for downsampling merged cloud before plane fit")
    ap.add_argument("--plane_dist_thresh", type=float, default=0.01,
                    help="RANSAC inlier distance threshold (m) for plane segmentation")
    ap.add_argument("--num_iterations", type=int, default=2000)
    ap.add_argument("--crop_xy", type=float, default=1.0,
                    help="drop points with |x|>crop_xy or |y|>crop_xy before plane fit")
    ap.add_argument("--z_min", type=float, default=-0.3)
    ap.add_argument("--z_max", type=float, default=0.6)
    args = ap.parse_args()

    cached_dir = Path(args.cached_pc)
    extrinsic_file = Path(args.extrinsic_file)
    out_path = Path(args.out_path) if args.out_path else extrinsic_file.parent
    out_path.mkdir(parents=True, exist_ok=True)

    cams = load_extrinsics(extrinsic_file)
    n = len(cams)
    print(f"[INFO] {n} cams from {extrinsic_file.name}")

    # Load all cached PCs and merge
    merged = o3d.geometry.PointCloud()
    raw_pcs = []
    for i in range(n):
        p = cached_dir / f"cam{i}_uncropped.ply"
        if not p.exists():
            print(f"  [WARN] missing {p.name}")
            raw_pcs.append(o3d.geometry.PointCloud())
            continue
        pc = o3d.io.read_point_cloud(str(p))
        raw_pcs.append(pc)
        merged += pc
    n_total = len(merged.points)
    if n_total == 0:
        raise RuntimeError(f"merged point cloud is empty (no cached PLYs under {cached_dir})")
    print(f"[INFO] merged {n_total} points across cams")

    # Crop to a bbox that isolates the table region (the plane fit is biased by
    # walls/ceiling otherwise)
    merged_ds = merged.voxel_down_sample(args.voxel)
    pts = np.asarray(merged_ds.points)
    mask = (
        (np.abs(pts[:, 0]) < args.crop_xy)
        & (np.abs(pts[:, 1]) < args.crop_xy)
        & (pts[:, 2] > args.z_min)
        & (pts[:, 2] < args.z_max)
    )
    idx = np.where(mask)[0]
    if len(idx) < 100:
        raise RuntimeError(
            f"only {len(idx)} points remain after crop "
            f"(xy<±{args.crop_xy}, z in [{args.z_min}, {args.z_max}]). "
            "Initial extrinsics may be way off — try manual_align_viser first."
        )
    crop = merged_ds.select_by_index(idx)
    print(f"[INFO] after voxel={args.voxel*1000:.1f}mm + crop: {len(crop.points)} pts -> plane fit")

    # Fit dominant plane
    plane_model, inliers = crop.segment_plane(
        distance_threshold=args.plane_dist_thresh,
        ransac_n=3,
        num_iterations=args.num_iterations,
    )
    a, b, c, d = plane_model
    n_vec = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    if n_vec[2] < 0:
        n_vec = -n_vec
    tilt_deg = float(np.degrees(np.arccos(np.clip(n_vec[2], -1, 1))))
    inlier_ratio = len(inliers) / max(1, len(crop.points))
    print(f"[INFO] plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"[INFO] tilt vs +z = {tilt_deg:.2f}°, inlier ratio = {inlier_ratio*100:.1f}%")

    T_plane = plane_to_z0_transform(plane_model)

    # Apply T_plane to every cam extrinsic
    updated = []
    for i, cam in enumerate(cams):
        ext = np.array(cam["transformation"]).reshape(4, 4)
        new_ext = T_plane @ ext
        updated.append({
            "camera_id": cam.get("camera_id", i),
            "serial_number": cam["serial_number"],
            "transformation": new_ext.tolist(),
            "color_intrinsic_matrix": cam["color_intrinsic_matrix"],
            "depth_intrinsic_matrix": cam["depth_intrinsic_matrix"],
        })
    stem = extrinsic_file.stem
    out_yaml = extrinsic_file.parent / f"{stem}_global_aligned.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(updated, f, default_flow_style=False, sort_keys=False)

    # Write merged PLY in the new frame
    aligned = o3d.geometry.PointCloud()
    for pc in raw_pcs:
        p2 = copy.deepcopy(pc)
        p2.transform(T_plane)
        aligned += p2
    out_ply = out_path / "postalign_global.ply"
    o3d.io.write_point_cloud(str(out_ply), aligned)

    print(f"[DONE]")
    print(f"  yaml : {out_yaml}")
    print(f"  ply  : {out_ply}")


if __name__ == "__main__":
    main()
