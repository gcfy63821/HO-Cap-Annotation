#!/usr/bin/env python3
"""Scan a data tree and list every exp that does NOT have a hand-reconstruction
result. Designed for the layout produced by DataCollection/run_auto_annotator:

    <data_root>/                     e.g. /viscam/projects/robotool/data/
        videos_0101/                 ← one "videos_<id>" capture session
            realsense_calibrate_*/   ← REQUIRED for the videos_root to count;
            <task>/                     <task>/<exp> is only scanned if at
                <exp>/                  least one calibration folder exists.
                    cam0_rgb.mp4 ...
                ...
        videos_0102/
            ...
        videos_0102_annotated/       ← per-exp pipeline outputs live here:
            <task>/<exp>/result_hand_optimized.pkl  ← what we look for

For every (videos_root, task, exp) we report:
  - whether the videos_root has a realsense_calibrate_* folder (skipped otherwise)
  - whether the annotated side exists at all
  - whether result_hand_optimized.pkl is present
  - a couple of useful related markers (mp / mano_pose_solver / fd merge)

Output is a CSV. The default location is
<data_root>/_missing_hand_recon.csv when --data_root is used, or
./missing_hand_recon.csv otherwise. A pretty summary is also printed to
stdout.

Usage:
    # Scan one or more explicit videos_root folders.
    python scripts/scan_missing_hand_recon.py \\
        --videos_root /viscam/projects/robotool/data/videos_0101 \\
                       /viscam/projects/robotool/data/videos_0102

    # Scan all videos_* under a data root.
    python scripts/scan_missing_hand_recon.py \\
        --data_root /viscam/projects/robotool/data

    # Custom CSV path.
    python scripts/scan_missing_hand_recon.py \\
        --data_root /viscam/projects/robotool/data \\
        --output /tmp/hand_audit.csv

    # Also include exps that DO have a hand result in the CSV (status=ok).
    python scripts/scan_missing_hand_recon.py --data_root ... --include_done

    # Don't require a calibration folder (treat every videos_* as scannable).
    python scripts/scan_missing_hand_recon.py --data_root ... --no_require_calib
"""

import argparse
import csv
import datetime
import sys
from collections import Counter
from pathlib import Path


SKIP_PREFIXES = ("realsense_calibrate", "ref_pc", "posts")
SKIP_EXACT = {"first_frame", "tool_meshes", "models"}


def _is_task_dir(p: Path) -> bool:
    if not p.is_dir(): return False
    nm = p.name
    if nm in SKIP_EXACT: return False
    if any(nm.startswith(pre) for pre in SKIP_PREFIXES): return False
    return True


def _has_calibration(videos_root: Path) -> tuple:
    """Returns (has_calib, calib_paths_str). A videos_root counts as having
    calibration if either:
      - a realsense_calibrate_* subfolder exists, OR
      - a top-level realsense_calibration_*.yaml exists."""
    cal_folders = [p for p in videos_root.iterdir()
                    if p.is_dir() and p.name.startswith("realsense_calibrate")]
    cal_yamls = list(videos_root.glob("realsense_calibration_*.yaml"))
    paths = []
    for p in cal_folders + cal_yamls:
        paths.append(p.name)
    return (len(paths) > 0), "; ".join(paths) if paths else ""


def _annotated_root_for(videos_root: Path) -> Path:
    return videos_root.parent / f"{videos_root.name}_annotated"


def discover_videos_roots(data_root: Path):
    """Find every videos_<id> subdir (NOT *_annotated)."""
    out = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir(): continue
        if not p.name.startswith("videos_"): continue
        if p.name.endswith("_annotated"): continue
        out.append(p)
    return out


def scan_videos_root(videos_root: Path, require_calib: bool = True):
    """Yield dicts, one per exp."""
    videos_root = videos_root.resolve()
    has_calib, calib_paths = _has_calibration(videos_root)
    annotated_root = _annotated_root_for(videos_root)

    if require_calib and not has_calib:
        # Still emit one summary row so the user sees the videos_root was
        # consciously skipped (rather than silently absent).
        yield {
            "videos_root": str(videos_root),
            "annotated_root": str(annotated_root),
            "task": "<all>", "exp": "<all>",
            "exp_folder": "", "ann_folder": "",
            "status": "skipped_no_calib",
            "has_calib": False, "calib": calib_paths,
            "has_annotated_dir": annotated_root.is_dir(),
            "has_hand_recon": False,
            "has_mp_processed": False,
            "has_mano_pose_solver": False,
            "has_fd_merged": False,
            "n_cams": 0,
            "note": "no realsense_calibrate_* folder or realsense_calibration_*.yaml",
        }
        return

    for task_dir in sorted(videos_root.iterdir()):
        if not _is_task_dir(task_dir): continue
        for exp_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            cams = sorted(exp_dir.glob("cam*_rgb.mp4"))
            if not cams:
                # Not a real exp (no captured video). Skip silently.
                continue

            ann = annotated_root / task_dir.name / exp_dir.name
            hand_pkl     = ann / "result_hand_optimized.pkl"
            mp_dir       = ann / "processed" / "mp"
            mano_dir     = ann / "processed" / "mano_pose_solver"
            fd_merged    = ann / "processed" / "fd_pose_solver" / "fd_poses_merged_fixed.npy"

            row = {
                "videos_root": str(videos_root),
                "annotated_root": str(annotated_root),
                "task": task_dir.name,
                "exp": exp_dir.name,
                "exp_folder": str(exp_dir),
                "ann_folder": str(ann),
                "has_calib": has_calib,
                "calib": calib_paths,
                "has_annotated_dir": ann.is_dir(),
                "has_hand_recon": hand_pkl.exists(),
                "has_mp_processed": mp_dir.is_dir() and any(mp_dir.iterdir()),
                "has_mano_pose_solver": mano_dir.is_dir() and any(mano_dir.iterdir()),
                "has_fd_merged": fd_merged.exists(),
                "n_cams": len(cams),
                "note": "",
            }
            row["status"] = "ok" if row["has_hand_recon"] else (
                "no_annotated_dir" if not row["has_annotated_dir"]
                else "missing_hand_recon"
            )
            yield row


def write_csv(rows, out_path: Path):
    if not rows:
        out_path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--videos_root", nargs="+", type=Path, default=None,
                    help="One or more videos_<id> folders to scan directly.")
    g.add_argument("--data_root", type=Path, default=None,
                    help="Parent dir containing videos_<id>/ subfolders. "
                         "All videos_* (excluding *_annotated) under here are scanned.")
    ap.add_argument("--output", type=Path, default=None,
                    help="CSV path. Default: <data_root>/_missing_hand_recon.csv "
                         "or ./missing_hand_recon.csv when no --data_root.")
    ap.add_argument("--include_done", action="store_true",
                    help="Also list exps that DO have hand recon. By default "
                         "the CSV only contains rows with status != 'ok' "
                         "(plus skipped-no-calib summary rows).")
    ap.add_argument("--no_require_calib", action="store_true",
                    help="Don't require a realsense_calibrate_* folder; scan "
                         "every videos_* regardless. Default: skip videos_root "
                         "with no calibration and emit a single 'skipped_no_calib' row.")
    args = ap.parse_args()

    if args.data_root is not None:
        data_root = args.data_root.expanduser().resolve()
        if not data_root.is_dir():
            print(f"[ERR] not a dir: {data_root}", file=sys.stderr); sys.exit(1)
        videos_roots = discover_videos_roots(data_root)
        if not videos_roots:
            print(f"[ERR] no videos_* subfolders under {data_root}",
                  file=sys.stderr); sys.exit(1)
        out_path = args.output or (data_root / "_missing_hand_recon.csv")
    else:
        videos_roots = [Path(p).expanduser().resolve() for p in args.videos_root]
        for p in videos_roots:
            if not p.is_dir():
                print(f"[ERR] not a dir: {p}", file=sys.stderr); sys.exit(1)
        out_path = args.output or Path("./missing_hand_recon.csv").resolve()

    print(f"[scan] {len(videos_roots)} videos_root(s):")
    for p in videos_roots:
        print(f"  - {p}")
    print(f"[scan] require_calib = {not args.no_require_calib}")
    print(f"[scan] output csv    = {out_path}")
    print()

    all_rows = []
    per_root_counts = []
    for vroot in videos_roots:
        rows = list(scan_videos_root(vroot,
                                       require_calib=not args.no_require_calib))
        c = Counter(r["status"] for r in rows)
        # Per-root summary line
        total_exps = sum(1 for r in rows if r.get("status") != "skipped_no_calib")
        n_missing  = sum(1 for r in rows if r["status"] == "missing_hand_recon")
        n_no_ann   = sum(1 for r in rows if r["status"] == "no_annotated_dir")
        n_ok       = sum(1 for r in rows if r["status"] == "ok")
        print(f"[{vroot.name}]  exps={total_exps}  ok={n_ok}  "
              f"missing={n_missing}  no_annotated_dir={n_no_ann}  "
              f"calib={'yes' if any(r.get('has_calib') for r in rows) else 'no'}")
        if "skipped_no_calib" in c:
            print(f"  ↳ SKIPPED (no calibration)")
        all_rows.extend(rows)
        per_root_counts.append((vroot, c))

    # Filter for output (and pretty list) unless --include_done.
    if args.include_done:
        rows_for_csv = all_rows
    else:
        rows_for_csv = [r for r in all_rows if r["status"] != "ok"]

    write_csv(rows_for_csv, out_path)
    print()
    print(f"[OK] wrote {len(rows_for_csv)} rows to {out_path}  "
          f"(of {len(all_rows)} scanned)")

    # Quick top-of-list dump so the user sees the missing exps immediately.
    missing = [r for r in all_rows if r["status"] == "missing_hand_recon"]
    print()
    print(f"=== {len(missing)} exps with NO hand recon ===")
    for r in missing[:50]:
        flags = []
        if not r["has_mp_processed"]:    flags.append("no_mp")
        if not r["has_mano_pose_solver"]: flags.append("no_mano")
        if not r["has_fd_merged"]:       flags.append("no_fd_merge")
        if r["has_annotated_dir"]:       flags.append("ann_dir_present")
        flag_s = (" [" + ",".join(flags) + "]") if flags else ""
        print(f"  {Path(r['videos_root']).name} / {r['task']} / {r['exp']}{flag_s}")
    if len(missing) > 50:
        print(f"  ... and {len(missing) - 50} more (see CSV)")
    skipped = [r for r in all_rows if r["status"] == "skipped_no_calib"]
    if skipped:
        print()
        print(f"=== {len(skipped)} videos_root SKIPPED (no calibration) ===")
        for r in skipped:
            print(f"  {Path(r['videos_root']).name}")
    print()
    print(f"finished: {datetime.datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
