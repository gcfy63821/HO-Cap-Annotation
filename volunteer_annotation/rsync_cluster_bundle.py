"""
Rsync cluster SAM2 bundle to local /data, skipping the 10% keyframe (kf1) embed files
to save ~313G of disk space.

Phase 1: rsync all _manifest.json files (fast)
Phase 2: for each experiment, rsync with --exclude for kf1 frame embed files

Usage:
  python rsync_cluster_bundle.py [--dry-run] [--workers N]
"""
import json, subprocess, argparse, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

REMOTE_HOST = "chenrq@scdt.stanford.edu"
REMOTE_BASE = "/viscam/projects/robotool/_va_bundle_v2"
LOCAL_BASE = Path("/data/robotool/_va_bundle_v2")
# Folders to rsync (Feb 2026 onwards)
FOLDERS = [
    "videos_0202", "videos_0204", "videos_0209", "videos_0210",
    "videos_0211", "videos_0212", "videos_0213", "videos_0216",
    "videos_0218", "videos_0220", "videos_0222", "videos_0227",
    "videos_0304", "videos_0309",
]

print_lock = Lock()

def log(msg):
    with print_lock:
        print(msg, flush=True)


def rsync_manifests():
    """Phase 1: pull all _manifest.json files from cluster."""
    log("=== Phase 1: Syncing _manifest.json files ===")
    for folder in FOLDERS:
        local_dir = LOCAL_BASE / folder
        local_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync", "-az", "--info=progress2",
            "--include=*/",
            "--include=*/_manifest.json",
            "--exclude=*",
            f"{REMOTE_HOST}:{REMOTE_BASE}/{folder}/",
            str(local_dir) + "/"
        ]
        log(f"  Syncing manifests: {folder}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"  WARN: {folder} manifest sync failed: {result.stderr[:200]}")
    log("Phase 1 done.")


def collect_exp_dirs():
    """Return list of (folder, task, exp) for all local experiments."""
    exps = []
    for folder in FOLDERS:
        folder_dir = LOCAL_BASE / folder
        if not folder_dir.exists():
            continue
        for task_dir in sorted(folder_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for exp_dir in sorted(task_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                exps.append((folder, task_dir.name, exp_dir.name, exp_dir))
    return exps


def get_kf1_frame(exp_dir: Path) -> str | None:
    """Read _manifest.json and return the 10% keyframe frame number string."""
    manifest_path = exp_dir / "_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
        # Manifest has per-camera entries; find keyframes array
        # Structure: {"cameras": [{"camera": "cam0", "keyframes": [0, 102, 205, 512, 1024], ...}]}
        cameras = data.get("cameras", [])
        if not cameras:
            return None
        kfs = cameras[0].get("keyframes", [])
        if len(kfs) < 2:
            return None
        return str(kfs[1])  # index 1 = 10% frame
    except Exception:
        return None


def rsync_exp(args):
    folder, task, exp, exp_dir, dry_run = args
    remote_path = f"{REMOTE_HOST}:{REMOTE_BASE}/{folder}/{task}/{exp}/"
    local_path = str(exp_dir) + "/"

    kf1 = get_kf1_frame(exp_dir)
    exclude_args = []
    if kf1:
        # Exclude only the embed.npz for the 10% frame (keep the .jpg)
        exclude_args = [f"--exclude=*.kf{kf1}.embed.npz"]

    cmd = ["rsync", "-az"]
    if dry_run:
        cmd.append("--dry-run")
    cmd += ["--info=stats2"] + exclude_args + [remote_path, local_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return (exp, False, result.stderr[:300])
    return (exp, True, "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-phase1", action="store_true", help="Skip manifest sync (already done)")
    args = parser.parse_args()

    LOCAL_BASE.mkdir(parents=True, exist_ok=True)

    if not args.skip_phase1:
        rsync_manifests()

    log("\n=== Phase 2: Collecting experiments ===")
    exps = collect_exp_dirs()
    log(f"Found {len(exps)} experiments to sync")

    log(f"\n=== Phase 2: Rsyncing experiments (workers={args.workers}) ===")
    if args.dry_run:
        log("DRY RUN mode")

    ok = 0
    fail = 0
    total = len(exps)

    tasks = [(folder, task, exp, exp_dir, args.dry_run) for folder, task, exp, exp_dir in exps]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(rsync_exp, t): t[2] for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            exp_name = futures[future]
            try:
                name, success, err = future.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    log(f"  FAIL [{i+1}/{total}] {name}: {err}")
            except Exception as e:
                fail += 1
                log(f"  ERROR [{i+1}/{total}] {exp_name}: {e}")
            if (i + 1) % 100 == 0:
                log(f"  Progress: {i+1}/{total} done, {ok} ok, {fail} fail")

    log(f"\nDone: {ok} OK, {fail} failed out of {total} experiments")


if __name__ == "__main__":
    main()
