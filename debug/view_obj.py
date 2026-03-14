#!/usr/bin/env python3
"""
Visualize OBJ files using trimesh.

Usage:
    python debug/view_obj.py model.obj
    python debug/view_obj.py model1.obj model2.obj --wireframe
    python debug/view_obj.py model.obj --axis --scale 0.001
"""

import argparse
import trimesh
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize OBJ files with trimesh")
    parser.add_argument("files", nargs="+", help="OBJ file path(s)")
    parser.add_argument("--wireframe", action="store_true", help="Show wireframe")
    parser.add_argument("--axis", action="store_true", help="Show coordinate axes")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor (e.g. 0.001 for mm->m)")
    args = parser.parse_args()

    scene = trimesh.Scene()

    if args.axis:
        axis = trimesh.creation.axis(origin_size=0.01, axis_length=0.1)
        scene.add_geometry(axis)

    colors = [
        [180, 180, 180, 255],
        [100, 149, 237, 255],
        [144, 238, 144, 255],
        [255, 182, 193, 255],
        [255, 218, 185, 255],
    ]

    for i, filepath in enumerate(args.files):
        mesh = trimesh.load(filepath, force="mesh")

        if args.scale != 1.0:
            mesh.apply_scale(args.scale)

        if not mesh.visual.defined or args.wireframe:
            mesh.visual.face_colors = colors[i % len(colors)]

        print(f"[{i}] {filepath}: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces, "
              f"bounds: {mesh.bounds[0].round(4)} ~ {mesh.bounds[1].round(4)}")

        scene.add_geometry(mesh)

    if args.wireframe:
        for g in scene.geometry.values():
            if hasattr(g, "faces"):
                edges = g.face_adjacency_edges
                path = trimesh.load_path(g.vertices[edges].reshape(-1, 2, 3))
                scene.add_geometry(path)

    scene.show()


if __name__ == "__main__":
    main()
