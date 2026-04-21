#!/usr/bin/env python
"""
Copy the *_global_aligned.yaml from one session to another.

Each output entry uses:
  - serial_number / color_intrinsic_matrix / depth_intrinsic_matrix  from DST's
    own original calibration yaml (so per-sensor intrinsics stay correct),
  - transformation                                                   from SRC's
    *_global_aligned.yaml (cameras matched by serial_number).

If DST has a camera whose serial doesn't appear in SRC, its original
transformation is preserved and a warning is printed. Use this when the rig
wasn't moved between two recording days so yesterday's hand-tuned alignment
can be reused verbatim today.

Usage:
    python tools/copy_aligned_yaml.py \
        --src_yaml        .../videos_0101/realsense_calibrate_0101/realsense_calibration_0101_global_aligned.yaml \
        --dst_orig_yaml   .../videos_0102/realsense_calibrate_0102/realsense_calibration_0102.yaml \
        --dst_aligned_yaml .../videos_0102/realsense_calibrate_0102/realsense_calibration_0102_global_aligned.yaml
"""
import argparse
from pathlib import Path

import yaml


def load_extrinsics(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "extrinsics" in data:
        return [v for k, v in data["extrinsics"].items() if not k.startswith("tag_")]
    if isinstance(data, list):
        return sorted(data, key=lambda c: c.get("camera_id", 0))
    raise ValueError(f"Unsupported YAML format: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_yaml", required=True,
                    help="source *_global_aligned.yaml (the one to copy transforms from)")
    ap.add_argument("--dst_orig_yaml", required=True,
                    help="destination session's ORIGINAL calibration yaml (NOT aligned)")
    ap.add_argument("--dst_aligned_yaml", required=True,
                    help="output path for destination's *_global_aligned.yaml")
    args = ap.parse_args()

    src_path = Path(args.src_yaml)
    dst_orig_path = Path(args.dst_orig_yaml)
    dst_out_path = Path(args.dst_aligned_yaml)

    for p in [src_path, dst_orig_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    src_cams = load_extrinsics(src_path)
    dst_cams = load_extrinsics(dst_orig_path)

    # Match by serial number
    src_by_sn = {str(c["serial_number"]): c for c in src_cams}

    merged = []
    matched = 0
    missing = []
    for cam in dst_cams:
        sn = str(cam["serial_number"])
        if sn in src_by_sn:
            new_ext = src_by_sn[sn]["transformation"]
            matched += 1
        else:
            new_ext = cam["transformation"]
            missing.append(sn)
        merged.append({
            "camera_id": cam.get("camera_id", 0),
            "serial_number": cam["serial_number"],
            "transformation": new_ext,
            "color_intrinsic_matrix": cam["color_intrinsic_matrix"],
            "depth_intrinsic_matrix": cam["depth_intrinsic_matrix"],
        })

    dst_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_out_path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

    print(f"[INFO] src_yaml      = {src_path}")
    print(f"[INFO] dst_orig_yaml = {dst_orig_path}")
    print(f"[INFO] out           = {dst_out_path}")
    print(f"[INFO] matched {matched}/{len(dst_cams)} cams by serial_number")
    if missing:
        print(f"[WARN] {len(missing)} dst cam(s) had no matching serial in src "
              f"-> kept their ORIGINAL extrinsic: {missing}")
    print("[DONE]")


if __name__ == "__main__":
    main()
