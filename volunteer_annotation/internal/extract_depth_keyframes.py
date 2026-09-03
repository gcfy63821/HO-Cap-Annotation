#!/usr/bin/env python3
"""Extract depth keyframes from H5 files — no GPU needed.

For each experiment, reads the 'depths' dataset (uint16, mm) and saves
a compact set of keyframe depth maps alongside the existing bundle.

Output per keyframe per camera:
  <bundle>/<task>/<exp>/cam{c}_depth.kf{idx}.png   — uint16 PNG, millimetres
                                                      (0 = invalid pixel)
  <bundle>/<task>/<exp>/_depth_manifest.json        — done marker

The 16-bit PNG format is:
  - Lossless, widely supported (cv2, PIL, imageio, MATLAB, etc.)
  - ~0.5-1 MB/frame/camera after PNG compression (vs 1.8 MB raw)
  - Read back: depth_m = cv2.imread(f, cv2.IMREAD_ANYDEPTH).astype(float32) / 1000

Usage:
  # Single experiment:
  python extract_depth_keyframes.py --exp /path/to/exp --bundle /path/to/bundle

  # All exps under a root (auto-discover):
  python extract_depth_keyframes.py --all /viscam/projects/robotool/data/videos_0218 \
                                    --bundle /viscam/projects/robotool/_va_bundle_v2

  # Parallel shard k/N (for SLURM array):
  python extract_depth_keyframes.py --exp_list /tmp/exp_list.txt \
                                    --bundle /viscam/... --shard 2/8

  # Match the keyframe fractions used during embedding precompute:
  python extract_depth_keyframes.py --all ... --bundle ... --keyframe_fracs 0,0.1,0.2
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

H5_NAME = "data00000000.h5"
DEPTH_FRAGMENT = "_depth_manifest.json"


def task_exp_from_path(exp_dir: Path):
    """Mirror of precompute_embeddings.task_exp_from_path."""
    exp_dir = exp_dir.resolve()
    for anc in exp_dir.parents:
        if anc.name.startswith("videos_"):
            rel = exp_dir.relative_to(anc.parent)
            return str(rel.parent), rel.name
    return exp_dir.parent.name, exp_dir.name


def fracs_to_idx(fracs, n):
    return sorted({max(0, min(int(round(f * (n - 1))), n - 1)) for f in fracs})


def discover_exps(root: Path):
    exps = {p.parent for p in root.rglob(H5_NAME)}
    return sorted(exps)


def process_exp(exp_dir: Path, bundle_root: Path, kf_fracs: list[float],
                skip_existing: bool = True) -> str:
    """
    Extract depth keyframes for one experiment.
    Returns 'skipped', 'no_depth', or 'done'.
    """
    exp_dir = exp_dir.resolve()
    h5_path = exp_dir / H5_NAME
    if not h5_path.is_file():
        return "no_h5"

    task, exp = task_exp_from_path(exp_dir)
    out_dir = bundle_root / task / exp
    frag = out_dir / DEPTH_FRAGMENT

    with h5py.File(h5_path, "r") as f:
        if "depths" not in f:
            return "no_depth"
        shape = f["depths"].shape   # (N_frames, N_cams, H, W)

    n_frames, n_cams, H, W = shape
    kf_idxs = fracs_to_idx(kf_fracs, n_frames)

    def out_path(c, i):
        return out_dir / f"cam{c}_depth.kf{i}.png"

    # Skip if all files already exist
    if skip_existing and frag.is_file() and all(
            out_path(c, i).is_file() for c in range(n_cams) for i in kf_idxs):
        return "skipped"

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Read all needed frames in one H5 open (minimise seek overhead)
    with h5py.File(h5_path, "r") as f:
        depths_ds = f["depths"]
        # Read requested frames for all cameras at once
        # depths_ds[idx] gives shape (N_cams, H, W)
        frames_data = {i: np.ascontiguousarray(depths_ds[i]) for i in kf_idxs}

    n_written = 0
    for c in range(n_cams):
        for i in kf_idxs:
            p = out_path(c, i)
            if skip_existing and p.is_file():
                continue
            depth_mm = frames_data[i][c]   # uint16, mm
            cv2.imwrite(str(p), depth_mm)  # 16-bit PNG, lossless
            n_written += 1

    # Write done marker
    frag.write_text(json.dumps({
        "n_frames": n_frames, "n_cams": n_cams, "H": H, "W": W,
        "keyframes": kf_idxs, "written": n_written,
    }, indent=2))

    elapsed = time.time() - t0
    print(f"  {task}/{exp}: {n_cams}cam x {len(kf_idxs)}kf = {n_written} written "
          f"({W}x{H}, {n_frames}f) in {elapsed:.1f}s")
    return "done"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--exp", help="single experiment directory")
    src.add_argument("--all", metavar="ROOT",
                     help="root directory; auto-discover all H5 experiments")
    src.add_argument("--exp_list", metavar="FILE",
                     help="text file with one exp dir per line (for SLURM array)")
    ap.add_argument("--bundle", required=True,
                    help="bundle root — depth PNGs go alongside the embeddings")
    ap.add_argument("--keyframe_fracs", default="0,0.1,0.2",
                    help="frame positions to extract (default: 0,0.1,0.2 — match embedding precompute)")
    ap.add_argument("--shard", default=None, metavar="k/N",
                    help="process only exps where index%%N==k (for parallel cluster runs)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing PNG files")
    ap.add_argument("--dry_run", action="store_true",
                    help="print what would be done without writing anything")
    args = ap.parse_args()

    bundle_root = Path(args.bundle)
    kf_fracs = [float(x) for x in args.keyframe_fracs.split(",") if x.strip()]
    skip_existing = not args.force

    if args.exp:
        exps = [Path(args.exp)]
    elif args.all:
        print(f"[scan] discovering H5 experiments under {args.all} ...")
        exps = discover_exps(Path(args.all))
        print(f"[scan] {len(exps)} experiment(s) found")
    else:
        exps = [Path(l.strip()) for l in Path(args.exp_list).read_text().splitlines() if l.strip()]

    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        exps = [e for i, e in enumerate(exps) if i % n == k]
        print(f"[shard] {k}/{n}: {len(exps)} exp(s) for this shard")

    print(f"[init] {len(exps)} experiment(s) | keyframe_fracs={kf_fracs} | bundle={bundle_root}")

    if args.dry_run:
        for e in exps[:20]:
            task, exp = task_exp_from_path(e)
            print(f"  would process: {task}/{exp}")
        if len(exps) > 20:
            print(f"  ... and {len(exps)-20} more")
        return

    t_start = time.time()
    counts = {"done": 0, "skipped": 0, "no_depth": 0, "no_h5": 0}
    for exp_dir in exps:
        result = process_exp(exp_dir, bundle_root, kf_fracs, skip_existing)
        counts[result] = counts.get(result, 0) + 1

    total = time.time() - t_start
    print(f"[done] {counts['done']} extracted, {counts['skipped']} skipped, "
          f"{counts['no_depth']} no-depth-dataset, {counts['no_h5']} no-h5 "
          f"({total:.1f}s total)")


if __name__ == "__main__":
    main()
