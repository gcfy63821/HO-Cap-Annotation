#!/usr/bin/env python3
"""
Visualize masked depth for frame 0 of a given sequence, across all cameras and objects.

Usage:
    python debug/visualize_masked_depth.py --sequence_folder /path/to/sequence

    # Specify frame and object:
    python debug/visualize_masked_depth.py --sequence_folder /path/to/sequence --frame_idx 10 --object_idx 1

    # Save to file instead of showing:
    python debug/visualize_masked_depth.py --sequence_folder /path/to/sequence --save output.png
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hocap_annotation.loaders.my_cluster_loader import MyClusterLoader


def depth_to_colormap(depth, valid_mask=None, percentile=(5, 95)):
    """Convert depth to jet colormap. Invalid pixels shown as black."""
    if valid_mask is None:
        valid_mask = depth > 0
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid_mask):
        return out
    vmin, vmax = np.percentile(depth[valid_mask], percentile)
    norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colored = (plt.cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)
    out[valid_mask] = colored[valid_mask]
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize masked depth for a sequence")
    parser.add_argument("--sequence_folder", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to visualize")
    parser.add_argument("--object_idx", type=int, default=None,
                        help="Object index (1-based). Default: show all objects.")
    parser.add_argument("--save", type=str, default=None, help="Save to file instead of showing")
    args = parser.parse_args()

    loader = MyClusterLoader(args.sequence_folder)
    serials = loader._rs_serials
    num_cams = len(serials)
    object_ids = loader._object_ids
    num_objects = len(object_ids)
    frame_idx = args.frame_idx

    print(f"[INFO] Sequence: {loader._sequence_name}")
    print(f"[INFO] {num_cams} cameras, {num_objects} objects: {object_ids}")
    print(f"[INFO] Visualizing frame {frame_idx}")

    # Determine which objects to show
    if args.object_idx is not None:
        obj_indices = [args.object_idx - 1]  # convert to 0-based
    else:
        obj_indices = list(range(num_objects))

    # Layout: rows = (raw_depth, obj1_masked, obj2_masked, ...), cols = cameras
    num_rows = 1 + len(obj_indices)  # raw depth + one row per object
    fig, axes = plt.subplots(num_rows, num_cams, figsize=(3.5 * num_cams, 3.5 * num_rows),
                             squeeze=False)

    for cam_idx, serial in enumerate(serials):
        depth = loader.get_depth(serial, frame_idx)  # (H, W) in meters
        color = loader.get_color(serial, frame_idx)   # (H, W, 3) uint8

        # Row 0: raw depth
        depth_vis = depth_to_colormap(depth)
        axes[0][cam_idx].imshow(depth_vis)
        axes[0][cam_idx].set_title(f"cam {serial}" if cam_idx < 8 else f"c{serial}", fontsize=9)
        axes[0][cam_idx].axis('off')

        # Row 1+: masked depth per object
        for row_offset, obj_idx in enumerate(obj_indices):
            mask = loader.get_mask(serial, frame_idx, obj_idx)  # (H, W) binary
            masked_depth = depth.copy()
            masked_depth[mask == 0] = 0

            masked_vis = depth_to_colormap(masked_depth)
            # Overlay mask contour on the visualization
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(masked_vis, contours, -1, (0, 255, 0), 1)

            ax = axes[1 + row_offset][cam_idx]
            ax.imshow(masked_vis)
            if cam_idx == 0:
                ax.set_ylabel(f"obj {obj_idx+1}: {object_ids[obj_idx]}", fontsize=9)
            ax.axis('off')

    # Row labels
    axes[0][0].set_ylabel("raw depth", fontsize=9)

    fig.suptitle(f"{loader._sequence_name}  frame={frame_idx}", fontsize=12, y=0.99)
    plt.tight_layout()

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved to {args.save}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
