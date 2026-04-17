#!/usr/bin/env python3
"""
Convert SAM3D PLY point clouds to cleaned_mesh_10000.obj for HO-Cap pipeline.

SAM3D point clouds typically have:
  - Points distributed both inside and on the surface of the object
  - Zero or unreliable normals

This script uses alpha shape reconstruction (which doesn't require normals)
to extract the outer surface, then cleans and simplifies the mesh.

Steps:
  1. Load PLY point cloud
  2. Statistical outlier removal
  3. Alpha shape surface reconstruction (auto-tunes alpha parameter)
  4. Clean up (remove degenerate faces, fill holes)
  5. Simplify to target face count
  6. Scale to real-world size (user-specified or keep original)
  7. Center with oriented bounds
  8. Export as cleaned_mesh_10000.obj

Usage:
    python preprocess/ply_to_obj.py --input data/new_models/bottle.ply --scale 0.15
    python preprocess/ply_to_obj.py --input_dir data/new_models --output_dir data/models
    python preprocess/ply_to_obj.py --input data/new_models/cup.ply --alpha 0.04 --scale 0.1
"""

import argparse
import os
import numpy as np
import open3d as o3d
import trimesh


def find_best_alpha(pcd, alphas=None):
    """
    Try multiple alpha values and pick the one that produces
    the most watertight mesh (highest volume / surface ratio).
    """
    if alphas is None:
        alphas = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]

    pts = np.asarray(pcd.points)
    extent = np.linalg.norm(pts.max(0) - pts.min(0))

    best_alpha = alphas[len(alphas) // 2]
    best_score = -1

    for alpha in alphas:
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            n_verts = len(mesh.vertices)
            n_faces = len(mesh.triangles)
            if n_faces < 100:
                continue

            # Convert to trimesh for watertight check
            tm = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
            )

            # Score: prefer meshes that are watertight, have reasonable face count,
            # and use most of the input points
            point_usage = n_verts / len(pts)
            watertight_bonus = 2.0 if tm.is_watertight else 1.0
            # Penalize too many or too few faces relative to points
            face_ratio = n_faces / len(pts)
            face_penalty = 1.0 if 0.5 < face_ratio < 3.0 else 0.5

            score = point_usage * watertight_bonus * face_penalty

            print(f"    alpha={alpha:.3f}: {n_verts} verts, {n_faces} faces, "
                  f"watertight={tm.is_watertight}, score={score:.3f}")

            if score > best_score:
                best_score = score
                best_alpha = alpha
        except Exception as e:
            print(f"    alpha={alpha:.3f}: failed ({e})")

    return best_alpha


def ply_to_mesh(ply_path, target_faces=10000, alpha=None):
    """Convert point cloud PLY to trimesh mesh via alpha shape reconstruction."""
    pcd = o3d.io.read_point_cloud(ply_path)
    n_points = len(pcd.points)
    print(f"  Loaded {n_points} points")

    if n_points == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")

    # Step 1: Statistical outlier removal
    pcd_clean, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    n_removed = n_points - len(inlier_idx)
    if n_removed > 0:
        print(f"  Outlier removal: removed {n_removed} points ({n_removed/n_points*100:.1f}%)")
    pcd = pcd_clean

    # Step 2: Alpha shape reconstruction
    if alpha is None:
        print("  Auto-tuning alpha parameter...")
        alpha = find_best_alpha(pcd)
    print(f"  Using alpha={alpha:.3f}")

    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    print(f"  Alpha shape: {len(mesh_o3d.vertices)} verts, {len(mesh_o3d.triangles)} faces")

    if len(mesh_o3d.triangles) < 10:
        raise ValueError(f"Alpha shape produced too few faces. Try a larger --alpha value.")

    # Step 3: Clean up
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_unreferenced_vertices()

    # Keep only the largest connected component (removes floating fragments)
    cluster_ids, cluster_counts, _ = mesh_o3d.cluster_connected_triangles()
    cluster_ids = np.asarray(cluster_ids)
    cluster_counts = np.asarray(cluster_counts)
    if len(cluster_counts) > 1:
        largest = np.argmax(cluster_counts)
        remove_mask = cluster_ids != largest
        mesh_o3d.remove_triangles_by_mask(remove_mask)
        mesh_o3d.remove_unreferenced_vertices()
        print(f"  Kept largest component: {len(mesh_o3d.triangles)} faces "
              f"(removed {len(cluster_counts)-1} fragments)")

    mesh_o3d.compute_vertex_normals()

    # Convert to trimesh
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print(f"  Watertight: {mesh.is_watertight}")

    # Step 4: Simplify if too many faces
    if mesh.faces.shape[0] > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        print(f"  Simplified to {mesh.faces.shape[0]} faces")

    print(f"  Final mesh: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")
    return mesh


def process_ply(ply_path, output_dir, scale=None, target_faces=10000, alpha=None):
    """Process a single PLY file into cleaned_mesh_10000.obj."""
    name = os.path.splitext(os.path.basename(ply_path))[0]
    print(f"\n=== Processing: {name} ===")
    print(f"  Input: {ply_path}")

    # Report raw point cloud size
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print(f"  [ERROR] Empty point cloud, skipping")
        return None, None
    raw_ext = pts.max(axis=0) - pts.min(axis=0)
    print(f"  Raw point cloud size: x={raw_ext[0]:.4f}, y={raw_ext[1]:.4f}, z={raw_ext[2]:.4f}")

    # Reconstruct mesh
    mesh = ply_to_mesh(ply_path, target_faces=target_faces, alpha=alpha)

    # Scale
    if scale is not None:
        mesh.apply_scale(scale)
        print(f"  Applied scale: {scale}")

    # Report size before centering
    ext = mesh.bounds[1] - mesh.bounds[0]
    print(f"  Mesh size (m): x={ext[0]:.4f}, y={ext[1]:.4f}, z={ext[2]:.4f}")
    print(f"  Mesh size (mm): x={ext[0]*1000:.1f}, y={ext[1]*1000:.1f}, z={ext[2]*1000:.1f}")

    # Center with oriented bounds (same as preprocess_mesh.py)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    mesh.apply_transform(to_origin)

    ext_final = mesh.bounds[1] - mesh.bounds[0]
    print(f"  Final size (m): x={ext_final[0]:.4f}, y={ext_final[1]:.4f}, z={ext_final[2]:.4f}")

    # Create output directory and save
    obj_dir = os.path.join(output_dir, name)
    os.makedirs(obj_dir, exist_ok=True)
    obj_path = os.path.join(obj_dir, "cleaned_mesh_10000.obj")
    mesh.export(obj_path, file_type='obj')
    print(f"  Saved: {obj_path}")

    return name, ext_final


def main():
    parser = argparse.ArgumentParser(description="Convert SAM3D PLY point clouds to OBJ meshes")
    parser.add_argument("--input", type=str, default=None, help="Single PLY file path")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing PLY files")
    parser.add_argument("--output_dir", type=str, default="data/models", help="Output models directory")
    parser.add_argument("--scale", type=float, default=None,
                        help="Scale factor to apply (e.g. 0.15 to make a 1m-normalized cup into 15cm)")
    parser.add_argument("--target_faces", type=int, default=10000, help="Target face count for simplification")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Alpha shape parameter (auto-tuned if not specified). "
                             "Smaller = tighter surface, larger = smoother but may include interior")
    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input_dir must be specified")

    ply_files = []
    if args.input:
        ply_files.append(args.input)
    if args.input_dir:
        for f in sorted(os.listdir(args.input_dir)):
            if f.endswith('.ply'):
                ply_files.append(os.path.join(args.input_dir, f))

    if not ply_files:
        print("[ERROR] No PLY files found")
        return

    print(f"Found {len(ply_files)} PLY file(s)")
    results = []
    for ply_path in ply_files:
        name, ext = process_ply(ply_path, args.output_dir, scale=args.scale,
                                target_faces=args.target_faces, alpha=args.alpha)
        if name is not None:
            results.append((name, ext))

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'Name':<25} {'X (mm)':>8} {'Y (mm)':>8} {'Z (mm)':>8}")
    print("-" * 60)
    for name, ext in results:
        print(f"{name:<25} {ext[0]*1000:>8.1f} {ext[1]*1000:>8.1f} {ext[2]*1000:>8.1f}")


if __name__ == "__main__":
    main()
