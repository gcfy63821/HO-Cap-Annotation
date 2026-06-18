#!/usr/bin/env python3
"""Code-driven orchestrator for the precompute pipeline.

Keeps up to --max_parallel videos folders precomputing AT ONCE and advances as
each finishes (submit -> poll squeue -> when a folder's array is gone, start the
next). Explicit concurrency control in code, instead of relying on the cluster's
implicit MaxJobs limit. Reuses sbatch_precompute_array.sh to submit each folder's
array (with --no_merge_dep), then runs ONE merge at the very end.

RUN ON THE CLUSTER (needs sbatch + squeue on PATH). It's long-running, so either:
    nohup python orchestrate.py > orch.log 2>&1 &
or submit the CPU wrapper:    sbatch sbatch_orchestrate.sh [args...]

Examples:
    python orchestrate.py                                  # all videos_*, 4 at a time
    python orchestrate.py --max_parallel 4 --shards 4      # 4 folders x 4 shards = 16 tasks
    python orchestrate.py --videos_root videos_0106 videos_0107
    python orchestrate.py --add_frames 0.5,1.0             # 50%+last-frame pass
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRONTEND = HERE / "sbatch_precompute_array.sh"
MERGE = HERE / "merge_manifests.py"


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def discover(data_root):
    dr = Path(data_root)
    if not dr.is_dir():
        sys.exit(f"[orch] data_root not found: {dr}")
    return [str(p) for p in sorted(dr.iterdir())
            if p.is_dir() and p.name.startswith("videos_") and not p.name.endswith("_annotated")]


def submit_folder(folder, a):
    """Submit one folder's array via the frontend; return its array jobid (or None
    if nothing pending = folder already done)."""
    cmd = ["bash", str(FRONTEND), "--videos_root", folder, "--bundle", a.bundle,
           "--shards", str(a.shards), "--no_merge_dep"]
    if a.add_frames:
        cmd += ["--add_frames", a.add_frames]
    if a.force:
        cmd += ["--force"]
    if a.refmask:
        cmd += ["--refmask"]
    r = sh(cmd)
    for line in (r.stdout + r.stderr).splitlines():
        print("   " + line)
    m = re.search(r"array job:\s*(\d+)", r.stdout)
    return m.group(1) if m else None


def alive(jids):
    """Subset of jobids still in the queue (running or pending)."""
    if not jids:
        return set()
    r = sh(["squeue", "-h", "-j", ",".join(jids), "-o", "%A"])
    return {x.strip() for x in r.stdout.split() if x.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="/viscam/projects/robotool/data")
    ap.add_argument("--bundle", default="/viscam/projects/robotool/_va_bundle_v2")
    ap.add_argument("--videos_root", nargs="*", default=None, help="folder names or abs paths")
    ap.add_argument("--max_parallel", type=int, default=4, help="folders running at once")
    ap.add_argument("--shards", type=int, default=4, help="shards (tasks) per folder")
    ap.add_argument("--add_frames", default=None, help="e.g. 0.5,1.0 for the 50%%+last-frame pass")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--refmask", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()

    if not FRONTEND.is_file():
        sys.exit(f"[orch] missing {FRONTEND}")
    if sh(["bash", "-c", "command -v squeue"]).returncode != 0:
        print("[orch] WARNING: squeue not on PATH — run this on the cluster (interactive env)")

    folders = a.videos_root or discover(a.data_root)
    folders = [f if f.startswith("/") else str(Path(a.data_root) / f) for f in folders]
    print(f"[orch] {len(folders)} folder(s), max_parallel={a.max_parallel}, shards={a.shards}/folder "
          f"(~{a.max_parallel * a.shards} tasks), bundle={a.bundle}"
          + (f", add_frames={a.add_frames}" if a.add_frames else ""), flush=True)

    pending = list(folders)
    inflight = {}   # folder -> array jobid
    while pending or inflight:
        while pending and len(inflight) < a.max_parallel:
            f = pending.pop(0)
            print(f"[orch] submit {Path(f).name}", flush=True)
            jid = submit_folder(f, a)
            if jid:
                inflight[f] = jid
                print(f"[orch] START {Path(f).name} -> {jid}  (inflight={len(inflight)}, queued={len(pending)})", flush=True)
            else:
                print(f"[orch] {Path(f).name}: nothing pending (done) — skip", flush=True)
        if inflight:
            live = alive(list(inflight.values()))
            for f in [f for f, j in inflight.items() if j not in live]:
                print(f"[orch] DONE  {Path(f).name}", flush=True)
                del inflight[f]
            if pending or inflight:
                time.sleep(a.poll)

    print("[orch] all arrays finished — merging manifest…", flush=True)
    r = sh(["python3", str(MERGE), "--bundle", a.bundle])
    print(r.stdout.rstrip() or r.stderr.rstrip())
    print("[orch] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
