#!/usr/bin/env python3
"""
Viser-based browser inspector for a videos_XXXX/ folder.

What it does
------------
Given a date-level folder like videos_0101/, it presents a web GUI that lets
you:
  1. Pick any task folder in that date, then any experiment inside it.
  2. See the frame-0 mosaic (all cam*_rgb.mp4 stacked as one image).
  3. Build a keyword -> tool_name mapping by typing an exp-name substring
     and choosing a model from your models/ folder. Mappings are written
     (auto-saved) to <videos_root>/tool_keyword_mapping.yaml .

Downstream (batch_auto_annotator.sh / match_tool_name.py) can consult this
mapping file first and only fall back to the fuzzy automatic matcher when no
keyword matches.

Usage
-----
    python scripts/data_inspector_viser.py \
        --videos_root /abs/path/to/data/videos_0101 \
        [--models_folder /abs/path/to/models] \
        [--port 8080]

Then SSH-forward the port and open http://localhost:<port> in a browser.
"""
import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import viser
import yaml


MAPPING_FILENAME = "tool_keyword_mapping.yaml"


def _has_experiment(task_folder: Path) -> bool:
    """A task folder must have >=1 sub-dir containing cam*_rgb.mp4."""
    for sub in task_folder.iterdir():
        if not sub.is_dir():
            continue
        if sub.name == "board_reference" or sub.name.endswith(".tar.gz"):
            continue
        if any(sub.glob("cam*_rgb.mp4")):
            return True
    return False


def discover_tasks(videos_root: Path):
    """Task folders: direct subdirs that contain at least one experiment
    (sub-dir with cam*_rgb.mp4). This filters out calibration folders,
    cached_pc/, posts/, flat_pc/, etc."""
    tasks = []
    for p in sorted(videos_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("realsense_calibrate_"):
            continue
        if p.name.endswith("_annotated"):
            continue
        # Also skip common non-task artifacts
        if p.name in {"cached_pc", "flat_pc", "posts", "posts_global",
                      "ref_pc_stages", "manual_aligned"}:
            continue
        if not _has_experiment(p):
            continue
        tasks.append(p)
    return tasks


def discover_exps(task_folder: Path):
    exps = []
    for p in sorted(task_folder.iterdir()):
        if not p.is_dir():
            continue
        if p.name == "board_reference" or p.name.endswith(".tar.gz"):
            continue
        if list(p.glob("cam*_rgb.mp4")):
            exps.append(p)
    return exps


def load_frame0_mosaic(exp_folder: Path, max_width: int = 1600):
    """Decode frame 0 of each cam*_rgb.mp4 and return (mosaic_rgb, n_cams, info)."""
    vids = sorted(exp_folder.glob("cam*_rgb.mp4"))
    if not vids:
        return None, 0, "no cam*_rgb.mp4 found"

    frames = []
    for vid in vids:
        cap = cv2.VideoCapture(str(vid))
        ok, f = cap.read()
        cap.release()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if not frames:
        return None, 0, "frame 0 could not be decoded"

    H, W = frames[0].shape[:2]
    n = len(frames)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    mosaic = np.zeros((H * rows, W * cols, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = i // cols, i % cols
        mosaic[r * H:(r + 1) * H, c * W:(c + 1) * W] = f
        cv2.putText(mosaic, f"cam{i}", (c * W + 10, r * H + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Downsample for GUI
    if mosaic.shape[1] > max_width:
        scale = max_width / mosaic.shape[1]
        mosaic = cv2.resize(mosaic, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
    return mosaic, n, f"{n} cams, {W}x{H}"


def normalize_exp(s: str) -> str:
    """Lowercase + drop underscores/hyphens/digits — matches the matcher."""
    return re.sub(r"[_\-\d]+", "", s.lower())


def load_mapping(yaml_path: Path):
    if not yaml_path.exists():
        return {}
    try:
        data = yaml.safe_load(yaml_path.read_text()) or {}
        return dict(data.get("mappings", {}) or {})
    except Exception as e:
        print(f"[WARN] could not read {yaml_path}: {e}")
        return {}


def save_mapping(yaml_path: Path, mapping: dict):
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_note": ("Map keywords (substring of exp name, matched against the "
                   "normalized lowercase exp name with underscores/digits "
                   "dropped) to model folder names. First-match wins."),
        "mappings": dict(mapping),
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))


def list_models(models_folder: Path):
    if not models_folder.is_dir():
        return []
    return sorted(
        p.name for p in models_folder.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "*"
    )


def guess_keyword_from_exp(exp_name: str) -> str:
    """Pull the middle alpha-run that's most likely the tool descriptor.
    Heuristic: take the longest alphabetic token when split by digits / _ / -.
    This is just a starting suggestion the user can edit."""
    tokens = [t for t in re.split(r"[_\-\d]+", exp_name.lower()) if t and not t.isdigit()]
    if not tokens:
        return ""
    # Prefer tokens longer than ~6 chars — they're usually the tool compound
    long_tokens = [t for t in tokens if len(t) >= 6]
    return max(long_tokens or tokens, key=len)


def match_keyword(exp_name: str, mapping: dict):
    """Check whether any keyword in `mapping` appears in the normalized exp name."""
    norm = normalize_exp(exp_name)
    for kw, tool in mapping.items():
        if kw and normalize_exp(kw) in norm:
            return kw, tool
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_root", required=True,
                    help="Path to a date-level folder (e.g. .../data/videos_0101)")
    ap.add_argument("--models_folder", default=None,
                    help="Path to models/ — default <HO-Cap-Annotation>/data/models")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max_width", type=int, default=1600,
                    help="Max mosaic width in pixels; larger = more detail")
    args = ap.parse_args()

    videos_root = Path(args.videos_root).resolve()
    if not videos_root.is_dir():
        raise SystemExit(f"videos_root not a dir: {videos_root}")

    # Resolution order for models folder:
    #   1. --models_folder CLI
    #   2. $MODELS_FOLDER env var (matches sbatch wrappers)
    #   3. <HO-Cap-Annotation>/data/models  (local dev default)
    hocap_root = Path(__file__).resolve().parent.parent
    if args.models_folder:
        models_folder = Path(args.models_folder)
    elif os.environ.get("MODELS_FOLDER"):
        models_folder = Path(os.environ["MODELS_FOLDER"])
    else:
        models_folder = hocap_root / "data" / "models"
    if not models_folder.is_dir():
        print(f"[WARN] models folder not found: {models_folder} — dropdown will be empty")

    mapping_path = videos_root / MAPPING_FILENAME
    mapping = load_mapping(mapping_path)
    print(f"[info] videos_root = {videos_root}")
    print(f"[info] mapping file = {mapping_path}  ({len(mapping)} existing entries)")
    print(f"[info] models folder = {models_folder}")

    # Discover content
    tasks = discover_tasks(videos_root)
    if not tasks:
        raise SystemExit(f"no task folders under {videos_root}")
    task_names = [t.name for t in tasks]
    print(f"[info] {len(tasks)} task(s): {task_names}")

    all_models = list_models(models_folder)
    if all_models:
        print(f"[info] {len(all_models)} model(s)")
    else:
        all_models = ["<no models found>"]

    # State
    state = {"task_idx": 0, "exp_idx": 0, "exps": []}

    def refresh_exps():
        state["exps"] = discover_exps(tasks[state["task_idx"]])

    refresh_exps()
    if not state["exps"]:
        print(f"[WARN] first task has no usable experiments")

    # ---------- Viser ----------
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"[info] viser at http://{args.host}:{args.port}")

    # ---- Navigation ----
    server.gui.add_markdown(f"### videos_root\n`{videos_root.name}`")
    task_dd = server.gui.add_dropdown("Task", options=task_names, initial_value=task_names[0])
    with server.gui.add_folder("Task nav", expand_by_default=False):
        btn_prev_task = server.gui.add_button("< Prev task")
        btn_next_task = server.gui.add_button("Next task >")

    exp_options_initial = [e.name for e in state["exps"]] or ["<none>"]
    exp_dd = server.gui.add_dropdown("Experiment", options=exp_options_initial,
                                       initial_value=exp_options_initial[0])
    with server.gui.add_folder("Exp nav", expand_by_default=True):
        btn_prev_exp = server.gui.add_button("< Prev exp")
        btn_next_exp = server.gui.add_button("Next exp >")
    exp_info = server.gui.add_markdown("*loading...*")

    server.gui.add_markdown("---")
    # Frame-0 mosaic
    placeholder = np.full((200, 400, 3), 40, dtype=np.uint8)
    frame_img = server.gui.add_image(placeholder, label="frame 0 mosaic")

    server.gui.add_markdown("---")
    # ---- Keyword mapping editor ----
    server.gui.add_markdown("### Keyword → tool mapping")
    match_status = server.gui.add_markdown("*auto-match hint will appear here*")
    keyword_input = server.gui.add_text(
        "Keyword (case-insensitive, matched against exp name)",
        initial_value="",
        hint="e.g. 'redrubberspatula' — the exp's distinctive substring",
    )
    btn_suggest_kw = server.gui.add_button("Suggest keyword from current exp")

    tool_dd = server.gui.add_dropdown("Tool (model folder)",
                                        options=all_models,
                                        initial_value=all_models[0])
    btn_add = server.gui.add_button("+ Add / update mapping")
    btn_remove = server.gui.add_button("- Remove mapping for this keyword")

    server.gui.add_markdown("---")
    server.gui.add_markdown(f"**Mapping file** (auto-saved):\n`{mapping_path}`")
    mapping_display = server.gui.add_markdown("*no mappings yet*")

    # ---------- helpers ----------
    def render_mapping():
        if not mapping:
            mapping_display.content = "*no mappings yet*"
            return
        lines = [f"- `{kw}` → `{tool}`" for kw, tool in mapping.items()]
        mapping_display.content = "\n".join(lines)

    def persist():
        save_mapping(mapping_path, mapping)
        render_mapping()

    def load_current_exp():
        exps = state["exps"]
        if not exps:
            exp_info.content = f"*task `{tasks[state['task_idx']].name}` has no experiments*"
            frame_img.image = placeholder
            match_status.content = "*no experiments*"
            return
        state["exp_idx"] = max(0, min(state["exp_idx"], len(exps) - 1))
        exp = exps[state["exp_idx"]]
        task_name = tasks[state["task_idx"]].name
        idx_str = f"{state['exp_idx'] + 1} / {len(exps)}"
        exp_info.content = f"**{idx_str}** · `{task_name}` / `{exp.name}`"

        # Reload mosaic
        mosaic, n_cams, info = load_frame0_mosaic(exp, max_width=args.max_width)
        if mosaic is not None:
            frame_img.image = mosaic
            exp_info.content += f"  \n_{info}_"
        else:
            frame_img.image = placeholder
            exp_info.content += f"  \n*{info}*"

        # Auto-match hint
        kw, tool = match_keyword(exp.name, mapping)
        if kw is not None:
            match_status.content = f"**Mapping hit**: keyword `{kw}` → tool `{tool}`"
        else:
            suggested = guess_keyword_from_exp(exp.name)
            match_status.content = (f"No keyword matches — suggested keyword: "
                                      f"`{suggested}` (edit the box below)")

        # Pre-fill keyword input with a guess if empty
        if not keyword_input.value:
            keyword_input.value = guess_keyword_from_exp(exp.name)

    def update_exp_dd():
        names = [e.name for e in state["exps"]] or ["<none>"]
        exp_dd.options = names
        if names:
            exp_dd.value = names[min(state["exp_idx"], len(names) - 1)]

    def switch_task_to(task_name):
        try:
            state["task_idx"] = task_names.index(task_name)
        except ValueError:
            return
        state["exp_idx"] = 0
        refresh_exps()
        update_exp_dd()
        load_current_exp()

    # ---------- initial render ----------
    load_current_exp()
    render_mapping()

    # ---------- event handlers ----------
    @task_dd.on_update
    def _(_evt):
        switch_task_to(task_dd.value)

    @btn_prev_task.on_click
    def _(_evt):
        i = state["task_idx"]
        if i > 0:
            task_dd.value = task_names[i - 1]
            switch_task_to(task_names[i - 1])

    @btn_next_task.on_click
    def _(_evt):
        i = state["task_idx"]
        if i < len(task_names) - 1:
            task_dd.value = task_names[i + 1]
            switch_task_to(task_names[i + 1])

    @exp_dd.on_update
    def _(_evt):
        names = [e.name for e in state["exps"]]
        try:
            state["exp_idx"] = names.index(exp_dd.value)
        except ValueError:
            return
        load_current_exp()

    @btn_prev_exp.on_click
    def _(_evt):
        if state["exp_idx"] > 0:
            state["exp_idx"] -= 1
            update_exp_dd()
            load_current_exp()

    @btn_next_exp.on_click
    def _(_evt):
        if state["exp_idx"] < len(state["exps"]) - 1:
            state["exp_idx"] += 1
            update_exp_dd()
            load_current_exp()

    @btn_suggest_kw.on_click
    def _(_evt):
        if not state["exps"]:
            return
        exp = state["exps"][state["exp_idx"]]
        keyword_input.value = guess_keyword_from_exp(exp.name)

    @btn_add.on_click
    def _(_evt):
        kw = (keyword_input.value or "").strip().lower()
        tool = tool_dd.value
        if not kw:
            match_status.content = "Error: keyword cannot be empty"
            return
        if tool == "<no models found>":
            match_status.content = "Error: no models available"
            return
        mapping[kw] = tool
        persist()
        match_status.content = f"Saved: `{kw}` → `{tool}`"
        # Re-evaluate current exp to refresh hint
        if state["exps"]:
            exp = state["exps"][state["exp_idx"]]
            kw_hit, tool_hit = match_keyword(exp.name, mapping)
            if kw_hit:
                match_status.content += f"  \n(current exp now matches `{kw_hit}` → `{tool_hit}`)"

    @btn_remove.on_click
    def _(_evt):
        kw = (keyword_input.value or "").strip().lower()
        if kw in mapping:
            del mapping[kw]
            persist()
            match_status.content = f"Removed: `{kw}`"
        else:
            match_status.content = f"No such keyword: `{kw}`"

    # block forever
    import time
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[info] stopped")


if __name__ == "__main__":
    main()
