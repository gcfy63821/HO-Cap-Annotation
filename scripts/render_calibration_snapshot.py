#!/usr/bin/env python
"""
Save a headless snapshot PNG of a globally-aligned point cloud.

Renders three views (top-down XY, side XZ, side YZ) of `postalign_global.ply`
as matplotlib scatter plots colored by height (z). No display needed.

Usage:
    python scripts/render_calibration_snapshot.py \
        --ply /path/to/postalign_global.ply \
        --out /path/to/calibration_snapshot.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_points", type=int, default=200000,
                    help="Random-subsample to this many points for the scatter")
    ap.add_argument("--xlim", type=float, nargs=2, default=[-0.6, 0.6])
    ap.add_argument("--ylim", type=float, nargs=2, default=[-0.6, 0.6])
    ap.add_argument("--zlim", type=float, nargs=2, default=[-0.1, 0.5])
    args = ap.parse_args()

    ply_path = Path(args.ply)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    # Lazy import open3d (heavy)
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        raise RuntimeError(f"empty point cloud in {ply_path}")

    colors = np.asarray(pc.colors) if pc.has_colors() else None

    if len(pts) > args.max_points:
        idx = np.random.choice(len(pts), args.max_points, replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def _scatter(ax, xs, ys, title, xl, yl, xlim, ylim):
        if colors is not None:
            c = np.clip(colors, 0.0, 1.0)
            ax.scatter(xs, ys, c=c, s=0.5, marker=".", linewidths=0)
        else:
            ax.scatter(xs, ys, c=pts[:, 2], cmap="viridis", s=0.5,
                       marker=".", linewidths=0)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    _scatter(axes[0], pts[:, 0], pts[:, 1], "Top-down (XY)", "x [m]", "y [m]",
             args.xlim, args.ylim)
    _scatter(axes[1], pts[:, 0], pts[:, 2], "Front (XZ)", "x [m]", "z [m]",
             args.xlim, args.zlim)
    _scatter(axes[2], pts[:, 1], pts[:, 2], "Side (YZ)", "y [m]", "z [m]",
             args.ylim, args.zlim)

    fig.suptitle(f"{ply_path.name}  |  {len(pts)} points shown")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[snapshot] saved {out_path}")


if __name__ == "__main__":
    main()
