#!/usr/bin/env python3
"""
Convert SAM3D PLY point clouds to cleaned_mesh_10000.obj for HO-Cap pipeline.

Steps:
  1. Load PLY point cloud
  2. Estimate normals
  3. Poisson surface reconstruction
  4. Clean up (remove low-density vertices)
  5. Optionally simplify to target face count
  6. Scale to real-world size (user-specified or keep original)
  7. Center with oriented bounds
  8. Export as cleaned_mesh_10000.obj

Usage:
    python preprocess/ply_to_obj.py --input data/new_models/bottle.ply --scale 0.15
    python preprocess/ply_to_obj.py --input_dir data/new_models --output_dir data/models
"""

import argparse
import os
import numpy as np
import open3d as o3d
import trimesh


def ply_to_mesh(ply_path, poisson_depth=9, density_threshold=0.01, target_faces=10000):
    """Convert point cloud PLY to trimesh mesh via Poisson reconstruction."""
    pcd = o3d.io.read_point_cloud(ply_path)
    n_points = len(pcd.points)
    print(f"  Loaded {n_points} points")

    # Re-estimate normals and orient toward center for reliable Poisson reconstruction
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    # Poisson surface reconstruction
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, linear_fit=True
    )

    # Remove low-density vertices (cleans up spurious faces)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_threshold)
    vertices_to_remove = densities < threshold
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
    mesh_o3d.compute_vertex_normals()

    # Convert to trimesh
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    vertex_colors = None
    if mesh_o3d.has_vertex_colors():
        vertex_colors = (np.asarray(mesh_o3d.vertex_colors) * 255).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if vertex_colors is not None:
        mesh.visual.vertex_colors = vertex_colors

    # Simplify if too many faces
    if mesh.faces.shape[0] > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        print(f"  Simplified to {mesh.faces.shape[0]} faces")

    print(f"  Mesh: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")
    return mesh


def process_ply(ply_path, output_dir, scale=None, target_faces=10000):
    """Process a single PLY file into cleaned_mesh_10000.obj."""
    name = os.path.splitext(os.path.basename(ply_path))[0]
    print(f"\n=== Processing: {name} ===")
    print(f"  Input: {ply_path}")

    # Report raw point cloud size
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    raw_ext = pts.max(axis=0) - pts.min(axis=0)
    print(f"  Raw point cloud size: x={raw_ext[0]:.4f}, y={raw_ext[1]:.4f}, z={raw_ext[2]:.4f}")

    # Reconstruct mesh
    mesh = ply_to_mesh(ply_path, target_faces=target_faces)

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
    parser.add_argument("--poisson_depth", type=int, default=9, help="Poisson reconstruction depth")
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
        name, ext = process_ply(ply_path, args.output_dir, scale=args.scale, target_faces=args.target_faces)
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
