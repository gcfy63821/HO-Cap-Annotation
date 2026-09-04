#!/usr/bin/env python3
"""Remove the 0.1-fraction keyframe files from every experiment in the bundle.

For each experiment:
  - reads _manifest.json to find n_frames and the kf index at frac 0.1
  - deletes cam*_rgb.kf{idx}.jpg and cam*_rgb.kf{idx}.embed.npz (if present)
  - rewrites _manifest.json with that index removed from every camera's keyframes

Safety:
  - dry_run mode (default) only prints what would be deleted
  - never touches kf0 / kf at frac >= 0.2

Usage:
  # Dry run (safe default):
  python purge_kf01.py --bundle /data/robotool/_va_bundle_v2

  # Actually delete:
  python purge_kf01.py --bundle /data/robotool/_va_bundle_v2 --execute

  # Limit to specific months:
  python purge_kf01.py --bundle /data/robotool/_va_bundle_v2 --months videos_0204 videos_0209 --execute
"""

import argparse
import json
import os
import sys
from pathlib import Path


def kf01_idx(n_frames: int) -> int:
    return max(0, min(round(0.1 * (n_frames - 1)), n_frames - 1))


def purge_exp(manifest_path: Path, execute: bool) -> dict:
    data = json.loads(manifest_path.read_text())
    cams = data.get("cameras", [])
    if not cams:
        return {"status": "skip_no_cams"}

    n_frames = cams[0].get("n_frames", 1)
    idx = kf01_idx(n_frames)

    # Safety: never remove frame 0 (and guard against 2-frame edge cases)
    if idx == 0:
        return {"status": "skip_idx0"}

    # Check: is this index actually in the keyframes list?
    kf_list = cams[0].get("keyframes", [])
    if idx not in kf_list:
        return {"status": "skip_not_present", "idx": idx}

    exp_dir = manifest_path.parent
    n_cams = len(cams)

    # Collect files to delete
    to_delete = []
    for c in range(n_cams):
        for suffix in [".jpg", ".embed.npz"]:
            p = exp_dir / f"cam{c}_rgb.kf{idx}{suffix}"
            if p.exists():
                to_delete.append(p)

    if not execute:
        print(f"  [dry] {exp_dir.parent.name}/{exp_dir.name}: "
              f"n={n_frames} kf01={idx}, would delete {len(to_delete)} files")
        return {"status": "dry", "idx": idx, "n_delete": len(to_delete)}

    # Delete files
    for p in to_delete:
        p.unlink()

    # Update manifest: remove idx from each camera's keyframes
    for cam in data["cameras"]:
        cam["keyframes"] = sorted(k for k in cam.get("keyframes", []) if k != idx)
    manifest_path.write_text(json.dumps(data, indent=2))

    return {"status": "done", "idx": idx, "deleted": len(to_delete)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="bundle root directory")
    ap.add_argument("--months", nargs="*", default=None,
                    help="limit to these month folders (e.g. videos_0204 videos_0209)")
    ap.add_argument("--execute", action="store_true",
                    help="actually delete files; default is dry-run (print only)")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    if not bundle.is_dir():
        sys.exit(f"Bundle not found: {bundle}")

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"[purge_kf01] mode={mode}  bundle={bundle}")

    # Collect manifests
    if args.months:
        manifests = []
        for m in args.months:
            manifests += sorted((bundle / m).rglob("_manifest.json"))
    else:
        manifests = sorted(bundle.rglob("_manifest.json"))

    print(f"[purge_kf01] {len(manifests)} manifests found")
    if not args.execute:
        print("[purge_kf01] (dry-run — pass --execute to actually delete)\n")

    counts = {"done": 0, "dry": 0, "skip_not_present": 0,
              "skip_idx0": 0, "skip_no_cams": 0}
    total_deleted = 0
    total_would_delete = 0

    for mp in manifests:
        r = purge_exp(mp, args.execute)
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] == "done":
            total_deleted += r.get("deleted", 0)
        elif r["status"] == "dry":
            total_would_delete += r.get("n_delete", 0)

    print()
    print(f"[purge_kf01] Summary ({mode}):")
    if args.execute:
        print(f"  Processed:     {counts['done']}")
        print(f"  Files deleted: {total_deleted}")
    else:
        print(f"  Would process: {counts['dry']}")
        print(f"  Files to delete: {total_would_delete}")
    print(f"  Skipped (kf0.1 not present): {counts.get('skip_not_present', 0)}")
    print(f"  Skipped (idx==0 edge case):  {counts.get('skip_idx0', 0)}")


if __name__ == "__main__":
    main()
