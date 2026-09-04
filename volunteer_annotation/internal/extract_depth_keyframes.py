#!/usr/bin/env python3
"""Extract depth keyframes — supports both H5 and raw cam*_depth.mkv sources.

For each experiment, reads depth at the requested keyframe fractions and saves
16-bit PNG files alongside the existing bundle.

Source priority (per experiment):
  1. data00000000.h5  — direct array read, fastest
  2. cam*_depth.mkv   — ffmpeg gray16le decode, specific frames only (no full decode)

Output:
  <bundle>/<task>/<exp>/cam{c}_depth.kf{idx}.png   — uint16 PNG, millimetres
  <bundle>/<task>/<exp>/_depth_manifest.json        — done marker

Read back: depth_m = cv2.imread(f, cv2.IMREAD_ANYDEPTH).astype(float32) / 1000

Usage:
  # Single experiment:
  python extract_depth_keyframes.py --exp /path/to/exp --bundle /path/to/bundle

  # All exps under a root:
  python extract_depth_keyframes.py --all /viscam/.../videos_0212 \
                                    --bundle /viscam/.../_va_bundle_v2 \
                                    --keyframe_fracs 0,0.5,1.0

  # SLURM shard:
  python extract_depth_keyframes.py --exp_list /tmp/list.txt \
                                    --bundle /viscam/... --shard 2/8
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

H5_NAME       = "data00000000.h5"
DEPTH_FRAGMENT = "_depth_manifest.json"


# ── path helpers ─────────────────────────────────────────────────────────────

def task_exp_from_path(exp_dir: Path):
    exp_dir = exp_dir.resolve()
    for anc in exp_dir.parents:
        if anc.name.startswith("videos_"):
            rel = exp_dir.relative_to(anc.parent)
            return str(rel.parent), rel.name
    return exp_dir.parent.name, exp_dir.name


def fracs_to_idx(fracs, n):
    return sorted({max(0, min(int(round(f * (n - 1))), n - 1)) for f in fracs})


# ── source detection ─────────────────────────────────────────────────────────

def detect_source(exp_dir: Path):
    """Returns ('h5', h5_path) or ('mkv', sorted_mkv_list) or (None, None)."""
    h5 = exp_dir / H5_NAME
    if _HAS_H5PY and h5.is_file():
        return "h5", h5
    mkvs = sorted(exp_dir.glob("cam*_depth.mkv"))
    if not mkvs:
        mkvs = sorted(exp_dir.glob("cam*_depth.mp4"))
    if mkvs:
        return "mkv", mkvs
    return None, None


# ── H5 backend ───────────────────────────────────────────────────────────────

def read_from_h5(h5_path: Path, kf_fracs):
    """Returns (n_frames, n_cams, H, W, {frame_idx: (n_cams, H, W) uint16})."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        if "depths" not in f:
            return None
        shape = f["depths"].shape   # (N, C, H, W)
        n_frames, n_cams, H, W = shape
        kf_idxs = fracs_to_idx(kf_fracs, n_frames)
        frames = {i: np.ascontiguousarray(f["depths"][i]) for i in kf_idxs}
    return n_frames, n_cams, H, W, kf_idxs, frames


# ── MKV backend ──────────────────────────────────────────────────────────────

def _probe_video(mkv: Path):
    """Return (n_frames, fps, W, H) via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate,width,height",
        "-of", "default=noprint_wrappers=1",
        str(mkv),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    info = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    W = int(info.get("width", 0))
    H = int(info.get("height", 0))
    # fps as fraction string e.g. "30000/1001"
    fps_str = info.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)
    try:
        n = int(info.get("nb_frames", 0))
    except (ValueError, TypeError):
        n = 0
    if n == 0:
        # fallback: count packets (works for mkv where nb_frames=N/A)
        cmd2 = ["ffprobe", "-v", "error", "-count_packets",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_packets",
                "-of", "default=noprint_wrappers=1", str(mkv)]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        for line in r2.stdout.splitlines():
            if "nb_read_packets" in line:
                try:
                    n = int(line.split("=")[1])
                except ValueError:
                    pass
    return n, fps, W, H


def _decode_frame(mkv: Path, frame_idx: int, fps: float, W: int, H: int) -> np.ndarray:
    """Decode one frame from mkv as uint16 gray16le (mm)."""
    ts = frame_idx / fps
    cmd = [
        "ffmpeg", "-ss", f"{ts:.6f}", "-i", str(mkv),
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = r.stdout
    expected = W * H * 2
    if len(raw) < expected:
        return np.zeros((H, W), dtype=np.uint16)
    return np.frombuffer(raw[:expected], dtype=np.uint16).reshape(H, W).copy()


def read_from_mkv(mkvs: list, kf_fracs):
    """Returns (n_frames, n_cams, H, W, kf_idxs, frames_dict) or None."""
    # Probe the first mkv for metadata
    n_frames, fps, W, H = _probe_video(mkvs[0])
    if n_frames == 0 or W == 0:
        return None

    n_cams = len(mkvs)
    kf_idxs = fracs_to_idx(kf_fracs, n_frames)

    # Decode each (cam, frame) independently — only the requested frames
    frames = {}   # frame_idx -> (n_cams, H, W) uint16
    for i in kf_idxs:
        arr = np.zeros((n_cams, H, W), dtype=np.uint16)
        for c, mkv in enumerate(mkvs):
            arr[c] = _decode_frame(mkv, i, fps, W, H)
        frames[i] = arr

    return n_frames, n_cams, H, W, kf_idxs, frames


# ── per-experiment processor ──────────────────────────────────────────────────

def process_exp(exp_dir: Path, bundle_root: Path, kf_fracs: list,
                skip_existing: bool = True) -> str:
    exp_dir = exp_dir.resolve()
    src_kind, src = detect_source(exp_dir)
    if src_kind is None:
        return "no_source"

    task, exp = task_exp_from_path(exp_dir)
    out_dir = bundle_root / task / exp
    frag = out_dir / DEPTH_FRAGMENT

    # Quick skip: manifest exists and all PNGs present
    if skip_existing and frag.is_file():
        existing = json.loads(frag.read_text())
        kf_idxs_check = fracs_to_idx(kf_fracs, existing.get("n_frames", 1))
        n_cams_check = existing.get("n_cams", 0)
        if all((out_dir / f"cam{c}_depth.kf{i}.png").is_file()
               for c in range(n_cams_check) for i in kf_idxs_check):
            return "skipped"

    t0 = time.time()
    if src_kind == "h5":
        result = read_from_h5(src, kf_fracs)
    else:
        result = read_from_mkv(src, kf_fracs)

    if result is None:
        return "no_depth"

    n_frames, n_cams, H, W, kf_idxs, frames = result
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for c in range(n_cams):
        for i in kf_idxs:
            p = out_dir / f"cam{c}_depth.kf{i}.png"
            if skip_existing and p.is_file():
                continue
            cv2.imwrite(str(p), frames[i][c])
            n_written += 1

    frag.write_text(json.dumps({
        "source": src_kind,
        "n_frames": n_frames, "n_cams": n_cams, "H": H, "W": W,
        "keyframes": kf_idxs, "written": n_written,
    }, indent=2))

    elapsed = time.time() - t0
    print(f"  [{src_kind}] {task}/{exp}: {n_cams}cam x {len(kf_idxs)}kf = "
          f"{n_written} written ({W}x{H}, {n_frames}f) in {elapsed:.1f}s")
    return "done"


# ── discovery ─────────────────────────────────────────────────────────────────

def discover_exps(root: Path):
    exps = {p.parent for p in root.rglob(H5_NAME)}
    exps |= {p.parent for p in root.rglob("cam0_depth.mkv")}
    exps |= {p.parent for p in root.rglob("cam0_depth.mp4")}
    return sorted(exps)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--exp", help="single experiment directory")
    src.add_argument("--all", metavar="ROOT",
                     help="root directory; auto-discover all experiments")
    src.add_argument("--exp_list", metavar="FILE",
                     help="text file with one exp dir per line (for SLURM array)")
    ap.add_argument("--bundle", required=True,
                    help="bundle root — depth PNGs go alongside the embeddings")
    ap.add_argument("--keyframe_fracs", default="0,0.5,1.0",
                    help="frame positions to extract (default: 0,0.5,1.0)")
    ap.add_argument("--shard", default=None, metavar="k/N",
                    help="process only exps where index%%N==k (SLURM array)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing PNG files")
    ap.add_argument("--dry_run", action="store_true",
                    help="print what would be done without writing")
    args = ap.parse_args()

    bundle_root = Path(args.bundle)
    kf_fracs = [float(x) for x in args.keyframe_fracs.split(",") if x.strip()]
    skip_existing = not args.force

    if args.exp:
        exps = [Path(args.exp)]
    elif args.all:
        print(f"[scan] discovering experiments under {args.all} ...")
        exps = discover_exps(Path(args.all))
        print(f"[scan] {len(exps)} experiment(s) found")
    else:
        exps = [Path(l.strip()) for l in Path(args.exp_list).read_text().splitlines() if l.strip()]

    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        exps = [e for i, e in enumerate(exps) if i % n == k]
        print(f"[shard] {k}/{n}: {len(exps)} exp(s)")

    print(f"[init] {len(exps)} exp(s) | fracs={kf_fracs} | bundle={bundle_root}")

    if args.dry_run:
        for e in exps[:20]:
            kind, _ = detect_source(e)
            task, exp = task_exp_from_path(e)
            print(f"  [{kind or 'none'}] {task}/{exp}")
        if len(exps) > 20:
            print(f"  ... and {len(exps)-20} more")
        return

    t_start = time.time()
    counts: dict = {}
    for exp_dir in exps:
        r = process_exp(exp_dir, bundle_root, kf_fracs, skip_existing)
        counts[r] = counts.get(r, 0) + 1

    total = time.time() - t_start
    print(f"[done] {counts.get('done',0)} extracted, {counts.get('skipped',0)} skipped, "
          f"{counts.get('no_source',0)} no-source, {counts.get('no_depth',0)} no-depth "
          f"({total:.1f}s total)")


if __name__ == "__main__":
    main()
