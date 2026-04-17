#!/usr/bin/env python3
"""
Convert PLY point clouds to cleaned OBJ meshes with robust outer-shell extraction.

This script is designed for cases where the point cloud contains many interior points
(for example, a sphere with points inside). It first extracts likely outer-shell points
by radius percentile, then reconstructs a surface mesh.

Pipeline:
  1) Load point cloud
  2) Statistical outlier removal
  3) Extract outer shell points using distance-to-center percentile
  4) Estimate normals and orient outward
  5) Poisson reconstruction + density crop
  6) Keep largest component, simplify, center, export OBJ

Usage:
  python preprocess/ply_to_obj_shell.py --input data/new_models/ball.ply --scale 0.1
  python preprocess/ply_to_obj_shell.py --input_dir data/new_models --output_dir data/models
  python preprocess/ply_to_obj_shell.py --input data/new_models/ball.ply --shell_q 75 --poisson_depth 9
"""

import argparse
import os
import numpy as np
import open3d as o3d
import trimesh
import trimesh.repair


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    pcd_clean, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    removed = len(pcd.points) - len(inlier_idx)
    return pcd_clean, removed


def extract_outer_shell_points(pcd, shell_q=70.0):
    """
    Keep only points whose radius is in [shell_q percentile, max].

    For a sphere containing interior points, interior radii are smaller, so this
    step keeps mostly the outer shell.
    """
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 100:
        raise ValueError("Too few points for shell extraction")

    center = np.median(pts, axis=0)
    radii = np.linalg.norm(pts - center[None, :], axis=1)
    r_min = np.percentile(radii, shell_q)

    keep = radii >= r_min
    shell_pts = pts[keep]
    if shell_pts.shape[0] < 200:
        raise ValueError(
            f"Shell extraction kept too few points ({shell_pts.shape[0]}). "
            f"Try smaller --shell_q."
        )

    shell_pcd = o3d.geometry.PointCloud()
    shell_pcd.points = o3d.utility.Vector3dVector(shell_pts)
    return shell_pcd, center, r_min, shell_pts.shape[0], pts.shape[0]


def estimate_orient_normals(shell_pcd, center, knn=30):
    shell_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    )
    normals = np.asarray(shell_pcd.normals)
    pts = np.asarray(shell_pcd.points)
    dirs = pts - center[None, :]
    sign = np.sum(normals * dirs, axis=1)
    normals[sign < 0] *= -1.0
    shell_pcd.normals = o3d.utility.Vector3dVector(normals)


def reconstruct_poisson(shell_pcd, poisson_depth=10, density_q=1.0):
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        shell_pcd, depth=poisson_depth
    )

    densities = np.asarray(densities)
    d_thr = np.percentile(densities, density_q)
    remove_mask = densities < d_thr
    mesh_o3d.remove_vertices_by_mask(remove_mask)

    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_unreferenced_vertices()

    return mesh_o3d


def keep_largest_component(mesh_o3d):
    cluster_ids, cluster_counts, _ = mesh_o3d.cluster_connected_triangles()
    cluster_ids = np.asarray(cluster_ids)
    cluster_counts = np.asarray(cluster_counts)
    if cluster_counts.size <= 1:
        return mesh_o3d, 0

    largest = int(np.argmax(cluster_counts))
    remove_mask = cluster_ids != largest
    mesh_o3d.remove_triangles_by_mask(remove_mask)
    mesh_o3d.remove_unreferenced_vertices()
    return mesh_o3d, int(cluster_counts.size - 1)


def cleanup_trimesh(tm):
    """
    Cleanup faces with API compatibility across trimesh versions.
    """
    if hasattr(tm, "remove_degenerate_faces"):
        tm.remove_degenerate_faces()
    elif hasattr(tm, "nondegenerate_faces"):
        mask = tm.nondegenerate_faces()
        tm.update_faces(mask)

    if hasattr(tm, "remove_duplicate_faces"):
        tm.remove_duplicate_faces()
    elif hasattr(tm, "unique_faces"):
        mask = tm.unique_faces()
        tm.update_faces(mask)

    tm.remove_unreferenced_vertices()
    return tm


def fill_mesh_holes(tm, max_iters=3):
    """
    Fill boundary holes iteratively for better watertightness.
    """
    for _ in range(max_iters):
        before = int(tm.faces.shape[0])
        trimesh.repair.fill_holes(tm)
        tm.remove_unreferenced_vertices()
        after = int(tm.faces.shape[0])
        if after == before:
            break
    return tm


def build_mesh_from_ply(
    ply_path, target_faces=10000, shell_q=70.0, poisson_depth=10, density_q=1.0
):
    pcd = o3d.io.read_point_cloud(ply_path)
    n_points = len(pcd.points)
    print(f"  Loaded {n_points} points")
    if n_points == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")

    pcd, removed = remove_outliers(pcd)
    if removed > 0:
        print(f"  Outlier removal: removed {removed} points")

    shell_pcd, center, r_min, n_shell, n_total = extract_outer_shell_points(pcd, shell_q=shell_q)
    print(
        f"  Shell extraction: kept {n_shell}/{n_total} points "
        f"(q={shell_q:.1f}, radius >= {r_min:.5f})"
    )

    estimate_orient_normals(shell_pcd, center)

    mesh_o3d = reconstruct_poisson(shell_pcd, poisson_depth=poisson_depth, density_q=density_q)
    print(f"  Poisson mesh: {len(mesh_o3d.vertices)} verts, {len(mesh_o3d.triangles)} faces")
    if len(mesh_o3d.triangles) < 10:
        raise ValueError("Poisson reconstruction produced too few faces")

    mesh_o3d, removed_components = keep_largest_component(mesh_o3d)
    if removed_components > 0:
        print(f"  Removed {removed_components} small components")

    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
        process=True,
    )

    if tm.faces.shape[0] > target_faces:
        tm = tm.simplify_quadric_decimation(face_count=target_faces)
        print(f"  Simplified to {tm.faces.shape[0]} faces")

    tm = cleanup_trimesh(tm)
    tm = fill_mesh_holes(tm, max_iters=3)

    print(f"  Watertight: {tm.is_watertight}")
    print(f"  Final mesh: {tm.vertices.shape[0]} verts, {tm.faces.shape[0]} faces")
    return tm


def process_one_ply(
    ply_path,
    output_dir,
    scale=None,
    target_faces=10000,
    shell_q=70.0,
    poisson_depth=10,
    density_q=1.0,
):
    name = os.path.splitext(os.path.basename(ply_path))[0]
    print(f"\n=== Processing: {name} ===")
    print(f"  Input: {ply_path}")

    mesh = build_mesh_from_ply(
        ply_path,
        target_faces=target_faces,
        shell_q=shell_q,
        poisson_depth=poisson_depth,
        density_q=density_q,
    )

    if scale is not None:
        mesh.apply_scale(scale)
        print(f"  Applied scale: {scale}")

    ext = mesh.bounds[1] - mesh.bounds[0]
    print(f"  Mesh size (m): x={ext[0]:.4f}, y={ext[1]:.4f}, z={ext[2]:.4f}")
    print(f"  Mesh size (mm): x={ext[0]*1000:.1f}, y={ext[1]*1000:.1f}, z={ext[2]*1000:.1f}")

    to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
    mesh.apply_transform(to_origin)
    ext_final = mesh.bounds[1] - mesh.bounds[0]

    obj_dir = os.path.join(output_dir, name)
    os.makedirs(obj_dir, exist_ok=True)
    obj_path = os.path.join(obj_dir, "cleaned_mesh_10000.obj")
    mesh.export(obj_path, file_type="obj")
    print(f"  Saved: {obj_path}")

    return name, ext_final


def main():
    parser = argparse.ArgumentParser(
        description="Convert PLY point clouds to OBJ with robust outer-shell extraction"
    )
    parser.add_argument("--input", type=str, default=None, help="Single PLY file")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory with PLY files")
    parser.add_argument("--output_dir", type=str, default="data/models", help="Output models dir")
    parser.add_argument("--scale", type=float, default=None, help="Optional global scale factor")
    parser.add_argument("--target_faces", type=int, default=10000, help="Target face count")
    parser.add_argument(
        "--shell_q",
        type=float,
        default=70.0,
        help="Radius percentile for shell extraction. Larger => thinner shell (default: 70)",
    )
    parser.add_argument(
        "--poisson_depth",
        type=int,
        default=10,
        help="Poisson reconstruction depth (default: 10)",
    )
    parser.add_argument(
        "--density_q",
        type=float,
        default=1.0,
        help="Drop lowest-density vertices below this percentile (default: 1)",
    )
    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input_dir must be specified")

    ply_files = []
    if args.input:
        ply_files.append(args.input)
    if args.input_dir:
        for f in sorted(os.listdir(args.input_dir)):
            if f.endswith(".ply"):
                ply_files.append(os.path.join(args.input_dir, f))

    if not ply_files:
        print("[ERROR] No PLY files found")
        return

    print(f"Found {len(ply_files)} PLY file(s)")
    results = []
    for ply_path in ply_files:
        name, ext = process_one_ply(
            ply_path,
            args.output_dir,
            scale=args.scale,
            target_faces=args.target_faces,
            shell_q=args.shell_q,
            poisson_depth=args.poisson_depth,
            density_q=args.density_q,
        )
        results.append((name, ext))

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'Name':<25} {'X (mm)':>8} {'Y (mm)':>8} {'Z (mm)':>8}")
    print("-" * 60)
    for name, ext in results:
        print(f"{name:<25} {ext[0]*1000:>8.1f} {ext[1]*1000:>8.1f} {ext[2]*1000:>8.1f}")


if __name__ == "__main__":
    main()
