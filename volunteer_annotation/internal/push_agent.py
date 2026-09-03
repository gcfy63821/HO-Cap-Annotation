#!/usr/bin/env python3
"""Local push agent: syncs embedding folders to the cloud annotation server on demand.

Runs on the LOCAL machine (where the full bundle lives). Polls the cloud API for
tasks that need embeddings, rsyncs them over SSH, and cleans up submitted tasks
to keep cloud disk usage bounded.

Usage:
    python push_agent.py [--bundle /data/robotool/_va_bundle_v2] \
                         [--cloud xulab-review] \
                         [--cloud_dir ~/volunteer_annotation/data] \
                         [--api http://localhost:8077] \
                         [--poll 30] [--prefetch 5] [--max_disk_gb 25]

Run in tmux so it survives disconnects:
    tmux new -s va_push
    python volunteer_annotation/internal/push_agent.py
    # Ctrl-b d to detach
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests


def rsync_exp(bundle: Path, task: str, exp: str, cloud: str, cloud_dir: str) -> bool:
    """rsync one exp's embedding folder to cloud. Returns True on success."""
    src = bundle / task / exp
    if not src.is_dir():
        print(f"  [push] WARNING: local exp not found: {src}", flush=True)
        return False
    # ensure parent dir exists on cloud (use bash -c so ~ expands)
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", cloud,
         f"bash -c 'mkdir -p {cloud_dir}/{task}/{exp}'"],
        capture_output=True)
    dst = f"{cloud}:{cloud_dir}/{task}/{exp}"
    cmd = ["rsync", "-a", "--partial", "-e", "ssh -o BatchMode=yes",
           str(src) + "/", dst + "/"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [push] rsync FAILED {task}/{exp}: {r.stderr[:200]}", flush=True)
        return False
    return True


def delete_cloud_exp(api: str, task: str, exp: str):
    """Ask cloud to delete an exp's embedding folder (frees disk)."""
    try:
        requests.delete(f"{api}/api/bundle/{task}/{exp}", timeout=10)
    except Exception as e:
        print(f"  [push] delete {task}/{exp} failed: {e}", flush=True)


def cloud_disk_free_gb(cloud: str, cloud_dir: str) -> float:
    """Check free disk on cloud via ssh df."""
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", cloud,
         f"df -BG --output=avail {cloud_dir} | tail -1"],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip().rstrip("G"))
    except Exception:
        return 999.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="/data/robotool/_va_bundle_v2")
    ap.add_argument("--cloud", default="xulab-review",
                    help="SSH host alias for the cloud server")
    ap.add_argument("--cloud_dir", default="~/volunteer_annotation/data",
                    help="Bundle root on the cloud server")
    ap.add_argument("--api", default="http://101.6.90.121:8077",
                    help="Cloud FastAPI base URL (reachable from local)")
    ap.add_argument("--poll", type=int, default=30, help="Poll interval (s)")
    ap.add_argument("--prefetch", type=int, default=5,
                    help="How many upcoming tasks to pre-push")
    ap.add_argument("--max_disk_gb", type=float, default=25,
                    help="When cloud free disk < this, clean up submitted exps first")
    a = ap.parse_args()

    bundle = Path(a.bundle)
    pushed = set()     # (task, exp) already pushed this session
    cleaned = set()    # (task, exp) already cleaned up

    print(f"[push] start  bundle={bundle}  cloud={a.cloud}  api={a.api}", flush=True)

    while True:
        try:
            # 1. Check cloud disk; clean up submitted exps if low
            free = cloud_disk_free_gb(a.cloud, a.cloud_dir)
            if free < a.max_disk_gb:
                print(f"[push] cloud disk low ({free:.1f}G free) — cleaning submitted exps",
                      flush=True)
                try:
                    resp = requests.get(f"{a.api}/api/stats", timeout=10)
                    # get submitted list via a simple query approach
                    r2 = requests.get(f"{a.api}/api/needs_embeddings?limit=0", timeout=10)
                except Exception:
                    pass
                # Ask cloud to list submitted tasks and delete their bundles
                # (we use a direct ssh du to find exp dirs on cloud that are done)
                r = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", a.cloud,
                     f"python3 ~/volunteer_annotation/cloud/tasks_db.py --list_submitted "
                     f"--db ~/volunteer_annotation/data/tasks.db 2>/dev/null || echo ''"],
                    capture_output=True, text=True)
                for line in r.stdout.strip().splitlines():
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        t, e = parts
                        if (t, e) not in cleaned:
                            delete_cloud_exp(a.api, t, e)
                            cleaned.add((t, e))
                            print(f"  [push] cleaned {t}/{e}", flush=True)

            # 2. Ask cloud which tasks need embeddings
            resp = requests.get(
                f"{a.api}/api/needs_embeddings?limit={a.prefetch + 5}", timeout=15)
            missing = resp.json().get("missing", [])

            if missing:
                print(f"[push] {len(missing)} task(s) need embeddings", flush=True)
                for item in missing[:a.prefetch + 2]:
                    key = (item["task"], item["exp"])
                    if key in pushed:
                        continue
                    status = item.get("status", "?")
                    print(f"  [push] -> {item['task']}/{item['exp']} ({status})", flush=True)
                    ok = rsync_exp(bundle, item["task"], item["exp"], a.cloud, a.cloud_dir)
                    if ok:
                        pushed.add(key)
                        print(f"  [push] OK  {item['task']}/{item['exp']}", flush=True)
            else:
                print(f"[push] all good (free={free:.1f}G)", flush=True)

        except requests.exceptions.ConnectionError:
            print(f"[push] cloud API not reachable ({a.api}) — retrying in {a.poll}s",
                  flush=True)
        except Exception as e:
            print(f"[push] error: {e}", flush=True)

        time.sleep(a.poll)


if __name__ == "__main__":
    main()
