#!/usr/bin/env python
"""
Index the experiments that volunteers have annotated, and resolve everything the
object-tracking pipeline needs for each one.

The volunteer pipeline (see volunteer_annotation/STATUS.md) drops its output in a
*separate* tree from the raw videos:

    <prompts_root>/<videos_XXXX>/<task>/<exp>/tool_masks/
        prompts/cam{i}_rgb.json      <- point prompts, one file per camera
        BAD.json                     <- present iff the volunteer flagged the clip

    <data_root>/<videos_XXXX>/<task>/<exp>/cam{i}_rgb.mp4     (cluster layout)
    <data_root>/<videos_XXXX>/<exp>/cam{i}_rgb.mp4            (flat local layout)

This script joins the two, and resolves the tool name + mesh per exp. Unlike the
DINO/auto path, the tool name does NOT have to be guessed from the folder name:
the volunteer picked the primary_tool name in the UI, so it comes straight out of
the prompt JSON. Folder-name matching is only a fallback for the older prompts
where the UI defaulted the name to the exp name.

Tool -> mesh resolution order:
  1. <models_folder>/<tool_name>/{textured_mesh.obj, cleaned_mesh_10000.obj, mesh.obj}
  2. tool_name -> mesh path derived by joining scripts/mesh_name_mapping.json
     (key -> tool_name) with scripts/mapping.json (same key -> mesh path)
  3. substring match of "<task>/<exp>" against mesh_name_mapping.json keys
     (the legacy path used by sbatch_run_auto_videos.sh)

Usage:
  # worklist for the whole prompts tree (TSV: seq_folder <tab> tool_name <tab> mesh <tab> prompts_dir)
  python scripts/volunteer_exp_index.py \
      --prompts_root /data/robotool/_va_bundle_v2_prompts \
      --data_root    /data/robotool \
      --models_folder /home/ruoqu/models \
      --out worklist.tsv

  # one exp, verbose (what would run)
  python scripts/volunteer_exp_index.py --prompts_root ... --data_root ... \
      --exp 20260104_smallplate_mallet_flatten_crush_smalldough_34 --verbose

  # only exps whose sequence folder actually exists locally, JSON output
  python scripts/volunteer_exp_index.py ... --require_sequence --format json
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MESH_CANDIDATES = ("textured_mesh.obj", "cleaned_mesh_10000.obj", "mesh.obj")


def load_tool_mesh_map(mesh_name_json, mesh_path_json):
    """tool_name -> absolute mesh path, by joining the two legacy mapping files
    on their shared keys. Returns {} if either file is missing."""
    try:
        names = json.loads(Path(mesh_name_json).read_text())
        paths = json.loads(Path(mesh_path_json).read_text())
    except (OSError, ValueError):
        return {}, {}
    tool_to_mesh = {}
    for key, tool in names.items():
        if key in paths:
            tool_to_mesh.setdefault(tool, paths[key])
    return tool_to_mesh, names


def resolve_mesh(tool_name, models_folder, tool_to_mesh):
    """Mesh path for a tool name, or None. Prefers models_folder (portable across
    machines) over the baked-in /viscam paths in mapping.json."""
    if models_folder:
        d = Path(models_folder).resolve() / tool_name
        for cand in MESH_CANDIDATES:
            if (d / cand).is_file():
                return str(d / cand)
    mesh = tool_to_mesh.get(tool_name)
    if mesh and Path(mesh).is_file():
        return mesh
    # Report the un-verifiable mapping anyway when nothing is on this machine —
    # the path may well exist on the node that ends up running the job.
    return mesh


def primary_tool_from_prompts(prompt_files):
    """Most common primary_tool name across a exp's camera prompt files."""
    votes = collections.Counter()
    for pf in prompt_files:
        try:
            data = json.loads(pf.read_text())
        except (OSError, ValueError):
            continue
        for o in data.get("objects", []):
            if o.get("role") == "primary_tool" and o.get("name"):
                votes[o["name"]] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def name_from_folder(task, exp, key_to_tool):
    """Longest case-insensitive substring hit of a mapping key in '<task>/<exp>'."""
    hay = f"{task}/{exp}".lower()
    best = None
    for key, tool in key_to_tool.items():
        if key.lower() in hay and (best is None or len(key) > len(best[0])):
            best = (key, tool)
    return best[1] if best else None


def find_sequence_folder(data_root, videos_folder, task, exp):
    """The raw video dir. Cluster layout nests <task>/, the local mirror is flat."""
    for cand in (Path(data_root) / videos_folder / task / exp,
                 Path(data_root) / videos_folder / exp,
                 Path(data_root) / task / exp,
                 Path(data_root) / exp):
        if cand.is_dir():
            return cand
    # Nothing on disk: return the canonical cluster-layout guess so the caller
    # can still emit a worklist for a machine that does have the data.
    return Path(data_root) / videos_folder / task / exp


def iter_exps(prompts_root):
    """Yield (videos_folder, task, exp, tool_masks_dir) for every annotated exp."""
    prompts_root = Path(prompts_root)
    for prompts_dir in sorted(prompts_root.glob("*/*/*/tool_masks/prompts")):
        tool_masks = prompts_dir.parent
        exp = tool_masks.parent.name
        task = tool_masks.parent.parent.name
        videos_folder = tool_masks.parent.parent.parent.name
        yield videos_folder, task, exp, tool_masks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts_root", required=True,
                    help="root of the volunteer prompt tree (e.g. /data/robotool/_va_bundle_v2_prompts)")
    ap.add_argument("--data_root", required=True,
                    help="root holding the videos_XXXX/ folders with the raw mp4s")
    ap.add_argument("--models_folder", default=os.environ.get("MODELS_FOLDER", ""),
                    help="folder of <tool_name>/ mesh dirs (default $MODELS_FOLDER)")
    ap.add_argument("--mesh_name_json", default=str(SCRIPT_DIR / "mesh_name_mapping.json"))
    ap.add_argument("--mesh_path_json", default=str(SCRIPT_DIR / "mapping.json"))
    ap.add_argument("--exp", default=None,
                    help="only this exp name (or '<task>/<exp>')")
    ap.add_argument("--videos_folder", default=None,
                    help="restrict to one videos_XXXX folder")
    ap.add_argument("--require_sequence", action="store_true",
                    help="drop exps whose raw video folder isn't on this machine")
    ap.add_argument("--require_mesh", action="store_true",
                    help="drop exps with no resolvable mesh")
    ap.add_argument("--min_cameras", type=int, default=1,
                    help="drop exps with fewer than N camera prompt files (default 1)")
    ap.add_argument("--format", choices=("tsv", "json"), default="tsv")
    ap.add_argument("--out", default=None, help="write here instead of stdout")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tool_to_mesh, key_to_tool = load_tool_mesh_map(args.mesh_name_json, args.mesh_path_json)
    if args.verbose:
        print(f"[map] {len(tool_to_mesh)} tool_name -> mesh entries, "
              f"{len(key_to_tool)} folder-name keys", file=sys.stderr)

    rows, stats = [], collections.Counter()
    for videos_folder, task, exp, tool_masks in iter_exps(args.prompts_root):
        stats["total"] += 1
        if args.videos_folder and videos_folder != args.videos_folder:
            continue
        if args.exp and args.exp not in (exp, f"{task}/{exp}"):
            continue
        if (tool_masks / "BAD.json").is_file():
            stats["bad"] += 1
            continue
        prompt_files = sorted((tool_masks / "prompts").glob("cam*.json"))
        if len(prompt_files) < args.min_cameras:
            stats["too_few_cameras"] += 1
            continue

        tool_name = primary_tool_from_prompts(prompt_files)
        source = "prompt"
        # Older prompts defaulted the name to the exp/manifest name, which is not
        # a tool at all — detect that and fall back to folder-name matching.
        if not tool_name or (tool_name not in tool_to_mesh and tool_name == exp) \
                or (tool_name and tool_name.startswith("2026")):
            fallback = name_from_folder(task, exp, key_to_tool)
            if fallback:
                tool_name, source = fallback, "folder"
        if not tool_name:
            stats["no_tool"] += 1
            continue

        mesh = resolve_mesh(tool_name, args.models_folder, tool_to_mesh)
        if args.require_mesh and not (mesh and Path(mesh).is_file()):
            stats["no_mesh"] += 1
            continue

        seq = find_sequence_folder(args.data_root, videos_folder, task, exp)
        if args.require_sequence and not seq.is_dir():
            stats["no_sequence"] += 1
            continue

        stats["kept"] += 1
        stats[f"tool_source:{source}"] += 1
        rows.append({
            "videos_folder": videos_folder,
            "task": task,
            "exp": exp,
            "sequence_folder": str(seq),
            "prompts_dir": str(tool_masks / "prompts"),
            "tool_name": tool_name,
            "tool_mesh": mesh or "",
            "tool_source": source,
            "n_cameras": len(prompt_files),
        })

    if args.format == "json":
        text = json.dumps(rows, indent=2, ensure_ascii=False)
    else:
        text = "".join(
            f"{r['sequence_folder']}\t{r['tool_name']}\t{r['tool_mesh']}\t{r['prompts_dir']}\n"
            for r in rows)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[out] {len(rows)} exp(s) -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)

    if args.verbose or args.out:
        print("[stats] " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())),
              file=sys.stderr)


if __name__ == "__main__":
    main()
