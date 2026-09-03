#!/usr/bin/env python3
"""Scan a bundle directory and compare two scans (local vs cluster).

MODE 1 — scan (run on each machine):
  python scan_bundle.py --scan --bundle /path/to/bundle > scan_local.json
  python scan_bundle.py --scan --bundle /viscam/.../bundle > scan_cluster.json

MODE 2 — diff (run locally after rsyncing both scan files):
  python scan_bundle.py --diff scan_local.json scan_cluster.json

Output of --scan: compact JSON, one line per experiment:
  { "task/exp": [kf_fracs...], ... }
  where kf_fracs are normalised to 1 decimal (0.0, 0.1, 0.2, 0.5, 1.0)

Output of --diff: human-readable table of discrepancies.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def scan_bundle(bundle_root: Path) -> dict:
    """Returns {task/exp: sorted_kf_fracs_list} for every _manifest.json found."""
    result = {}
    for frag in sorted(bundle_root.rglob("_manifest.json")):
        data = json.loads(frag.read_text())
        cams = data.get("cameras", [])
        if not cams:
            continue
        cam0 = cams[0]
        task = cam0.get("task", "")
        exp  = cam0.get("exp", "")
        kfs  = sorted(cam0.get("keyframes", []))
        n    = cam0.get("n_frames", 1)
        fracs = sorted({round(k / max(n - 1, 1), 1) for k in kfs}) if n > 1 else [0.0]
        result[f"{task}/{exp}"] = fracs
    return result


def classify(fracs):
    has_start = 0.0 in fracs
    has_mid   = any(0.4 <= f <= 0.6 for f in fracs)
    has_end   = any(f >= 0.9 for f in fracs)
    if has_start and has_mid and has_end:
        return "complete"
    elif not has_end and not has_mid:
        return "early-only"
    elif not has_start and not has_mid:
        return "late-only"
    else:
        return "partial"


def diff(local: dict, cluster: dict):
    all_keys = sorted(set(local) | set(cluster))

    # Categorise
    only_local   = []   # in local but not cluster
    only_cluster = []   # in cluster but not local
    same         = []
    local_better = []   # local has more keyframes
    cluster_better = [] # cluster has more keyframes

    for key in all_keys:
        lf = local.get(key)
        cf = cluster.get(key)
        if lf is None:
            only_cluster.append((key, cf))
        elif cf is None:
            only_local.append((key, lf))
        elif lf == cf:
            same.append(key)
        elif len(lf) >= len(cf):
            local_better.append((key, lf, cf))
        else:
            cluster_better.append((key, lf, cf))

    # Summary
    print(f"Scanned: {len(local)} local  /  {len(cluster)} cluster  /  {len(all_keys)} total unique\n")

    if only_cluster:
        print(f"═══ Only on CLUSTER ({len(only_cluster)}) — not copied to local yet ═══")
        by_month = defaultdict(list)
        for key, cf in only_cluster:
            month = key.split("/")[0]
            by_month[month].append((key, classify(cf), cf))
        for month in sorted(by_month):
            items = by_month[month]
            cls_counts = defaultdict(int)
            for _, cls, _ in items:
                cls_counts[cls] += 1
            print(f"  {month:<20} {len(items):>4} exps  [{', '.join(f'{c}:{n}' for c,n in sorted(cls_counts.items()))}]")
        print()

    if only_local:
        print(f"═══ Only LOCAL ({len(only_local)}) — deleted from cluster or wrong path ═══")
        for key, lf in only_local[:20]:
            print(f"  {key}  kfs={lf}")
        if len(only_local) > 20:
            print(f"  ... and {len(only_local)-20} more")
        print()

    if cluster_better:
        print(f"═══ CLUSTER has MORE keyframes ({len(cluster_better)}) — not fully synced ═══")
        by_month = defaultdict(list)
        for key, lf, cf in cluster_better:
            month = key.split("/")[0]
            by_month[month].append((key, lf, cf))
        for month in sorted(by_month):
            items = by_month[month]
            print(f"  {month:<20} {len(items):>4} exps (local has {set(items[0][1])}, cluster has {set(items[0][2])})")
            for key, lf, cf in items[:3]:
                print(f"    {key[-60:]}")
                print(f"      local={lf}  cluster={cf}")
            if len(items) > 3:
                print(f"    ... and {len(items)-3} more")
        print()

    if local_better:
        print(f"═══ LOCAL has MORE keyframes ({len(local_better)}) — cluster is behind ═══")
        for key, lf, cf in local_better[:10]:
            print(f"  {key[-60:]}  local={lf}  cluster={cf}")
        if len(local_better) > 10:
            print(f"  ... and {len(local_better)-10} more")
        print()

    print(f"═══ Summary ═══")
    print(f"  Identical:        {len(same):>5}")
    print(f"  Only cluster:     {len(only_cluster):>5}  ← should rsync these")
    print(f"  Only local:       {len(only_local):>5}")
    print(f"  Cluster has more: {len(cluster_better):>5}  ← need re-rsync")
    print(f"  Local has more:   {len(local_better):>5}")

    # Suggest rsync filter if there are cluster-only or cluster-better exps
    if only_cluster or cluster_better:
        print(f"\n→ To sync missing/incomplete exps from cluster:")
        print(f"  rsync -avz --include='*/' --include='*.npz' --include='*.jpg' --include='*_manifest.json' \\")
        print(f"        --exclude='*' \\")
        print(f"        <cluster>:/viscam/projects/robotool/_va_bundle_v2/ \\")
        print(f"        /data/robotool/_va_bundle_v2/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan", action="store_true",
                      help="scan a bundle and print JSON to stdout")
    mode.add_argument("--diff", nargs=2, metavar=("LOCAL", "CLUSTER"),
                      help="compare two scan JSON files")
    ap.add_argument("--bundle", help="bundle root (required with --scan)")
    args = ap.parse_args()

    if args.scan:
        if not args.bundle:
            ap.error("--bundle required with --scan")
        data = scan_bundle(Path(args.bundle))
        print(json.dumps(data, separators=(",", ":")))
    else:
        local   = json.loads(Path(args.diff[0]).read_text())
        cluster = json.loads(Path(args.diff[1]).read_text())
        diff(local, cluster)


if __name__ == "__main__":
    main()
