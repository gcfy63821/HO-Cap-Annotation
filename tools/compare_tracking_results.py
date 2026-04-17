#!/usr/bin/env python3
"""
Compare two tracking runs by computing depth reprojection error on both results.

This script loads the per-camera poses from two runs (baseline vs improved),
reprojects the object mesh into each camera's depth map, and reports:
  - Mean/median depth error per run
  - Inlier ratio (% of projected points within 2cm of observed depth)
  - Pose temporal smoothness (frame-to-frame translation/rotation jitter)
  - Per-frame comparison

Usage:
    python tools/compare_tracking_results.py \
        --sequence_folder /path/to/sequence \
        --baseline_suffix fd_pose_solver_baseline \
        --improved_suffix fd_pose_solver_depthval \
        --object_idx 1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from hocap_annotation.loaders import MyClusterLoader as HOCapLoader


def compute_depth_error(pose_4x4, mesh_verts, depth_map, K, mask=None):
    """Compute depth reprojection error."""
    if np.all(pose_4x4 == -1) or np.all(pose_4x4[:3, 3] == 0):
        return np.nan, np.nan

    H, W = depth_map.shape[:2]
    verts_cam = (pose_4x4[:3, :3] @ mesh_verts.T).T + pose_4x4[:3, 3]
    valid_z = verts_cam[:, 2] > 0.01
    if valid_z.sum() < 10:
        return np.nan, np.nan
    verts_cam = verts_cam[valid_z]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = (fx * verts_cam[:, 0] / verts_cam[:, 2] + cx).astype(np.int32)
    v = (fy * verts_cam[:, 1] / verts_cam[:, 2] + cy).astype(np.int32)
    rendered_z = verts_cam[:, 2]

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, rendered_z = u[in_bounds], v[in_bounds], rendered_z[in_bounds]
    if len(u) < 10:
        return np.nan, np.nan

    observed_z = depth_map[v, u]
    valid = observed_z > 0.01
    if mask is not None:
        valid &= mask[v, u] > 0
    if valid.sum() < 5:
        return np.nan, np.nan

    diff = np.abs(rendered_z[valid] - observed_z[valid])
    return float(np.mean(diff)), float(np.mean(diff < 0.02))


def compute_smoothness(poses_world):
    """Compute frame-to-frame jitter in translation and rotation."""
    trans_diffs = []
    rot_diffs = []
    for i in range(1, len(poses_world)):
        p0, p1 = poses_world[i - 1], poses_world[i]
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
            continue
        if np.all(p0[:3] == 0) or np.all(p1[:3] == 0):
            continue
        # Translation diff
        t0, t1 = p0[4:7], p1[4:7]
        trans_diffs.append(np.linalg.norm(t1 - t0))
        # Rotation diff
        q0, q1 = p0[:4], p1[:4]
        try:
            r0 = R.from_quat(q0)
            r1 = R.from_quat(q1)
            angle = (r0.inv() * r1).magnitude()
            rot_diffs.append(np.degrees(angle))
        except Exception:
            pass

    return {
        "mean_trans_jitter_mm": float(np.mean(trans_diffs) * 1000) if trans_diffs else -1,
        "median_trans_jitter_mm": float(np.median(trans_diffs) * 1000) if trans_diffs else -1,
        "mean_rot_jitter_deg": float(np.mean(rot_diffs)) if rot_diffs else -1,
        "median_rot_jitter_deg": float(np.median(rot_diffs)) if rot_diffs else -1,
    }


def load_world_poses(pose_folder, object_id, num_frames, start_frame):
    """Load world poses from txt files."""
    poses = []
    for frame_id in range(start_frame, start_frame + num_frames):
        p = pose_folder / object_id / "ob_in_world" / f"{frame_id:06d}.txt"
        if p.exists():
            poses.append(np.loadtxt(p))
        else:
            poses.append(np.full(7, np.nan))
    return np.array(poses)


def load_cam_poses(pose_folder, object_id, serial, num_frames, start_frame):
    """Load per-camera poses from txt files."""
    poses = []
    for frame_id in range(start_frame, start_frame + num_frames):
        p = pose_folder / object_id / "ob_in_cam" / serial / f"{frame_id:06d}.txt"
        if p.exists():
            q = np.loadtxt(p)
            if np.all(q == -1):
                poses.append(None)
            else:
                mat = np.eye(4)
                r = R.from_quat(q[:4])
                mat[:3, :3] = r.as_matrix()
                mat[:3, 3] = q[4:7]
                poses.append(mat)
        else:
            poses.append(None)
    return poses


def main():
    parser = argparse.ArgumentParser(description="Compare tracking results")
    parser.add_argument("--sequence_folder", type=str, required=True)
    parser.add_argument("--baseline_suffix", type=str, default="fd_pose_solver_baseline")
    parser.add_argument("--improved_suffix", type=str, default="fd_pose_solver_depthval")
    parser.add_argument("--object_idx", type=int, default=1)
    parser.add_argument("--num_eval_frames", type=int, default=-1,
                        help="Number of frames to evaluate (-1 for all)")
    args = parser.parse_args()

    # Load data
    loader = HOCapLoader(args.sequence_folder)
    object_id = loader.object_ids[args.object_idx - 1]
    num_frames = loader.num_frames
    start_frame = loader._start_frame if hasattr(loader, "_start_frame") else 0
    valid_serials = loader.get_valid_seg_serials()
    valid_indices = [loader.rs_serials.index(s) for s in valid_serials]
    valid_Ks = loader.rs_Ks[valid_indices]

    # Load mesh
    mesh = trimesh.load(loader.object_cleaned_files[args.object_idx - 1], force="mesh")
    verts = np.array(mesh.vertices, dtype=np.float32)
    if len(verts) > 2000:
        verts = verts[np.random.RandomState(42).choice(len(verts), 2000, replace=False)]

    # Paths
    annotated_base = (
        loader._data_folder.parent.parent.parent
        / f"{loader._folder_name}_annotated"
        / loader._task_name
        / loader._sequence_name
        / "processed"
    )
    baseline_folder = annotated_base / args.baseline_suffix
    improved_folder = annotated_base / args.improved_suffix

    if not baseline_folder.exists():
        print(f"[ERROR] Baseline folder not found: {baseline_folder}")
        return
    if not improved_folder.exists():
        print(f"[ERROR] Improved folder not found: {improved_folder}")
        return

    eval_frames = num_frames if args.num_eval_frames < 0 else min(args.num_eval_frames, num_frames)

    # Evaluate depth errors per camera
    results = {}
    for label, folder in [("baseline", baseline_folder), ("depth_val", improved_folder)]:
        all_errors = []
        all_inliers = []

        for cam_i, serial in enumerate(valid_serials):
            cam_poses = load_cam_poses(folder, object_id, serial, eval_frames, start_frame)
            K = valid_Ks[cam_i]

            for frame_idx in range(eval_frames):
                pose = cam_poses[frame_idx]
                if pose is None:
                    continue
                depth = loader.get_depth(serial, start_frame + frame_idx)
                mask = loader.get_mask(serial, start_frame + frame_idx, args.object_idx - 1)
                err, inl = compute_depth_error(pose, verts, depth, K, mask)
                if not np.isnan(err):
                    all_errors.append(err)
                    all_inliers.append(inl)

        # World pose smoothness
        world_poses = load_world_poses(folder, object_id, eval_frames, start_frame)
        smoothness = compute_smoothness(world_poses)

        results[label] = {
            "mean_depth_error_mm": float(np.mean(all_errors) * 1000) if all_errors else -1,
            "median_depth_error_mm": float(np.median(all_errors) * 1000) if all_errors else -1,
            "mean_inlier_ratio": float(np.mean(all_inliers)) if all_inliers else -1,
            "num_valid_frames": len(all_errors),
            **smoothness,
        }

    # Print comparison
    print("\n" + "=" * 70)
    print("TRACKING COMPARISON REPORT")
    print("=" * 70)
    print(f"Sequence: {args.sequence_folder}")
    print(f"Object:   {object_id}")
    print(f"Frames:   {eval_frames}")
    print(f"Cameras:  {len(valid_serials)}")
    print()

    header = f"{'Metric':<30} {'Baseline':>15} {'Depth Val':>15} {'Change':>10}"
    print(header)
    print("-" * 70)

    metrics = [
        ("Depth Error (mm)", "mean_depth_error_mm"),
        ("Median Depth Error (mm)", "median_depth_error_mm"),
        ("Inlier Ratio (<2cm)", "mean_inlier_ratio"),
        ("Trans Jitter (mm/frame)", "mean_trans_jitter_mm"),
        ("Rot Jitter (deg/frame)", "mean_rot_jitter_deg"),
    ]

    for name, key in metrics:
        b = results["baseline"][key]
        d = results["depth_val"][key]
        if b > 0 and d > 0:
            if "inlier" in key.lower():
                change = f"{(d - b) / b * 100:+.1f}%"
            else:
                change = f"{(d - b) / b * 100:+.1f}%"
        else:
            change = "N/A"
        print(f"{name:<30} {b:>15.3f} {d:>15.3f} {change:>10}")

    print()

    # Save report
    report_path = annotated_base / "depth_validation_comparison.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
