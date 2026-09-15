#!/usr/bin/env python3
"""Build the worklist TSV for sbatch_masks_array.sh.

Scans prompts_root for experiments that have complete volunteer point prompts
(cam*_rgb.json under .../tool_masks/prompts/) and emits one line per experiment
that still needs masks.h5 generated.

Output columns (TSV):
    <exp_dir>  <prompts_dir>

exp_dir maps prompts path to data path:
    prompts_root/<videos_X>/<task>/<exp>  ->  data_root/<videos_X>/<task>/<exp>
    (e.g. _va_bundle_v2_prompts/videos_0102/task/exp -> data/videos_0102/task/exp)

Skip conditions:
    - exp_dir not found in data_root (video data not synced yet)
    - masks.h5 already exists in the annotated output dir with correct shape
    - prompts_dir flagged with BAD.json
    - fewer than --min_cams prompt files (default 6 of 8)

Usage:
    python build_masks_worklist.py \\
        --data_root   /viscam/projects/robotool/data \\
        --prompts_root /viscam/projects/robotool/_va_bundle_v2_prompts \\
        --out /viscam/u/chenrq/crq_ws/masks_worklist.tsv

    # Then check sizing:
    wc -l /viscam/u/chenrq/crq_ws/masks_worklist.tsv
    # Last task index for exps_per_task=8:
    awk 'END{print "last task idx:", int((NR+7)/8)-1}' masks_worklist.tsv
"""

import argparse
import sys
from pathlib import Path

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def check_masks_done(exp_dir: Path, annotated_root: Path) -> bool:
    """Return True if this exp already has a complete masks.h5."""
    if not HAS_H5PY:
        return False
    # Derive the annotated output path (mirrors sbatch_masks_array.sh logic)
    videos_anc = None
    cur = exp_dir
    while cur != cur.parent:
        if cur.name.startswith("videos_"):
            videos_anc = cur
            break
        cur = cur.parent
    if videos_anc is None:
        return False
    rel = exp_dir.relative_to(videos_anc)
    masks_h5 = annotated_root / (videos_anc.name + "_annotated") / rel / "tool_masks" / "masks.h5"
    if not masks_h5.is_file():
        return False
    # Quick shape sanity: must have a "masks" dataset with 4 dims and 8 cams
    try:
        with h5py.File(masks_h5, "r") as f:
            if "masks" not in f:
                return False
            shape = f["masks"].shape
            return len(shape) == 4 and shape[1] == 8
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="/viscam/projects/robotool/data",
                    help="root that contains videos_XXXX/<task>/<exp> subdirs")
    ap.add_argument("--prompts_root", default="/viscam/projects/robotool/_va_bundle_v2_prompts",
                    help="root of volunteer prompts (same layout as data_root)")
    ap.add_argument("--annotated_root", default=None,
                    help="where masks.h5 will be written (default: same as data_root, "
                         "outputs go to videos_XXXX_annotated/...)")
    ap.add_argument("--out", default="-", help="output TSV path (default: stdout)")
    ap.add_argument("--videos_filter", default=None,
                    help="only scan this videos_XXXX folder name (e.g. videos_0202)")
    ap.add_argument("--min_cams", type=int, default=6,
                    help="minimum cam*.json files required (default 6)")
    ap.add_argument("--skip_done", action="store_true", default=True,
                    help="skip exps whose masks.h5 already has correct shape (default True)")
    ap.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    prompts_root = Path(args.prompts_root)
    data_root = Path(args.data_root)
    annotated_root = Path(args.annotated_root) if args.annotated_root else data_root

    if not prompts_root.is_dir():
        sys.exit(f"[ERR] prompts_root not found: {prompts_root}")

    n_total = n_skip_bad = n_skip_few_cams = n_skip_no_data = n_skip_done = 0
    rows = []

    # Enumerate: prompts_root/<videos_X>/<task>/<exp>/tool_masks/prompts/
    for videos_dir in sorted(prompts_root.iterdir()):
        if not videos_dir.is_dir() or not videos_dir.name.startswith("videos_"):
            continue
        if args.videos_filter and videos_dir.name != args.videos_filter:
            continue
        for task_dir in sorted(videos_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for exp_dir_prompts in sorted(task_dir.iterdir()):
                if not exp_dir_prompts.is_dir():
                    continue
                prompts_dir = exp_dir_prompts / "tool_masks" / "prompts"
                if not prompts_dir.is_dir():
                    continue

                n_total += 1
                exp_name = exp_dir_prompts.name

                # Check BAD flag
                if (exp_dir_prompts / "tool_masks" / "BAD.json").is_file():
                    n_skip_bad += 1
                    if args.verbose:
                        print(f"[skip bad] {exp_name}", file=sys.stderr)
                    continue

                # Check prompt file count
                prompt_files = sorted(prompts_dir.glob("cam*_rgb.json"))
                if len(prompt_files) < args.min_cams:
                    n_skip_few_cams += 1
                    if args.verbose:
                        print(f"[skip cams={len(prompt_files)}] {exp_name}", file=sys.stderr)
                    continue

                # Derive data path
                rel = exp_dir_prompts.relative_to(prompts_root)
                exp_dir_data = data_root / rel
                if not exp_dir_data.is_dir():
                    n_skip_no_data += 1
                    if args.verbose:
                        print(f"[skip no_data] {exp_dir_data}", file=sys.stderr)
                    continue

                # Skip if already done
                if args.skip_done and check_masks_done(exp_dir_data, annotated_root):
                    n_skip_done += 1
                    if args.verbose:
                        print(f"[skip done] {exp_name}", file=sys.stderr)
                    continue

                rows.append((str(exp_dir_data), str(prompts_dir)))

    # Write output
    out_fh = open(args.out, "w") if args.out != "-" else sys.stdout
    for exp, prompts in rows:
        out_fh.write(f"{exp}\t{prompts}\n")
    if args.out != "-":
        out_fh.close()

    print(f"[worklist] scanned {n_total} experiments", file=sys.stderr)
    print(f"  skip bad={n_skip_bad}  few_cams={n_skip_few_cams}  "
          f"no_data={n_skip_no_data}  already_done={n_skip_done}", file=sys.stderr)
    print(f"  -> {len(rows)} experiments written to {args.out}", file=sys.stderr)
    if rows:
        exps_per_task = 8
        n_tasks = (len(rows) + exps_per_task - 1) // exps_per_task
        print(f"  -> with exps_per_task={exps_per_task}: {n_tasks} array tasks  "
              f"(submit with --array=0-{n_tasks-1}%32)", file=sys.stderr)


if __name__ == "__main__":
    main()
