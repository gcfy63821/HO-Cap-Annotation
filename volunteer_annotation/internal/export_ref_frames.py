#!/usr/bin/env python3
"""Add a mid-sequence reference JPEG per camera into an existing bundle.

The annotation UI shows this mid-frame in a side panel so the volunteer can see
the tool actually being used (frame 0 often shows it idle on the table). CPU-only
— reads raw video, no SAM2, no GPU, safe to run alongside other GPU jobs.

Writes <bundle>/<task>/<exp>/cam{c}_rgb.mid.jpg for every exp present in --src.

Usage:
    python export_ref_frames.py --src /data/robotool/videos_0102 --bundle /tmp/va_real
    python export_ref_frames.py --src <root> --bundle <b> --frac 0.5
"""
import argparse
import glob
from pathlib import Path

import cv2
import h5py
import numpy as np

H5_NAME = "data00000000.h5"


def task_exp(exp_dir):
    exp_dir = Path(exp_dir).resolve()
    return exp_dir.parent.name, exp_dir.name


def discover_exps(root):
    root = Path(root)
    exps = {p.parent for p in root.rglob(H5_NAME)}
    exps |= {p.parent for p in root.rglob("cam0_rgb.mp4")}
    return sorted(exps)


def export_exp(exp_dir, bundle_root, frac):
    exp_dir = Path(exp_dir)
    task, exp = task_exp(exp_dir)
    out_dir = bundle_root / task / exp
    if not out_dir.is_dir():
        print(f"  [skip] {task}/{exp} not in bundle")
        return 0
    n = 0
    if (exp_dir / H5_NAME).is_file():
        with h5py.File(exp_dir / H5_NAME, "r") as f:
            imgs = f["imgs"]
            mid = int(frac * imgs.shape[0])
            for c in range(imgs.shape[1]):
                cv2.imwrite(str(out_dir / f"cam{c}_rgb.mid.jpg"),
                            cv2.cvtColor(np.ascontiguousarray(imgs[mid, c]), cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                n += 1
    else:
        for p in sorted(glob.glob(str(exp_dir / "cam*_rgb.mp4"))):
            c = int("".join(ch for ch in Path(p).stem.replace("_rgb", "") if ch.isdigit()) or 0)
            cap = cv2.VideoCapture(p)
            mid = int(frac * cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, frame = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(str(out_dir / f"cam{c}_rgb.mid.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                n += 1
    print(f"  {task}/{exp}: {n} mid-frames (frac={frac})")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source root with raw exps (mp4/h5)")
    ap.add_argument("--bundle", required=True, help="existing bundle root")
    ap.add_argument("--frac", type=float, default=0.5, help="reference frame position (0..1)")
    args = ap.parse_args()

    bundle_root = Path(args.bundle)
    total = sum(export_exp(e, bundle_root, args.frac) for e in discover_exps(args.src))
    print(f"[done] {total} mid-frame(s) written")


if __name__ == "__main__":
    main()
