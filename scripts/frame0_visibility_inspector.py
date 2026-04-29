#!/usr/bin/env python3
"""
Viser-based frame-0 visibility annotator for a videos_XXXX/ folder.

What it does
------------
Given a date-level folder like videos_0101/, you pick a task, then for each
experiment in that task you see the frame-0 mosaic (all cam*_rgb.mp4 stacked)
and click one of three buttons:

    [Y] visible       — the operating tool is visible in (most) cameras
    [N] not visible   — frame 0 doesn't show the tool clearly enough
    [?] unsure        — keep for later review

The annotation is auto-saved to:

    <videos_root>/<task>/frame0_visibility.yaml

so each task carries its own annotation file. Downstream
(batch_auto_annotator.sh / run_full_auto_annotator.sh) can read this to
decide whether to run --frame0_only or to skip / re-record an exp.

Usage
-----
    python scripts/frame0_visibility_inspector.py \
        --videos_root /abs/path/to/data/videos_0101 \
        [--port 8080]

Then SSH-forward the port and open http://localhost:<port> in a browser.
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import viser
import yaml


ANNOTATION_FILENAME = "frame0_visibility.yaml"
LABELS = ("visible", "not_visible", "unsure")


# ──────────────────────────────────────────────────────────────────────
#  Discovery (mirrors data_inspector_viser.py for consistent semantics)
# ──────────────────────────────────────────────────────────────────────

def _has_experiment(task_folder: Path) -> bool:
    for sub in task_folder.iterdir():
        if not sub.is_dir():
            continue
        if sub.name == "board_reference" or sub.name.endswith(".tar.gz"):
            continue
        if any(sub.glob("cam*_rgb.mp4")):
            return True
    return False


def discover_tasks(videos_root: Path):
    tasks = []
    for p in sorted(videos_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("realsense_calibrate_"):
            continue
        if p.name.endswith("_annotated"):
            continue
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


# ──────────────────────────────────────────────────────────────────────
#  Frame-0 decoding
# ──────────────────────────────────────────────────────────────────────

def load_frame0_mosaic(exp_folder: Path, max_width: int = 1600):
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

    if mosaic.shape[1] > max_width:
        scale = max_width / mosaic.shape[1]
        mosaic = cv2.resize(mosaic, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
    return mosaic, n, f"{n} cams, {W}x{H}"


# ──────────────────────────────────────────────────────────────────────
#  Annotation file IO
# ──────────────────────────────────────────────────────────────────────

def annotation_path(task_folder: Path) -> Path:
    return task_folder / ANNOTATION_FILENAME


def load_annotations(task_folder: Path) -> dict:
    p = annotation_path(task_folder)
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text()) or {}
        ann = data.get("annotations", {}) or {}
        # Filter to known labels only.
        return {k: v for k, v in ann.items() if v in LABELS}
    except Exception as e:
        print(f"[WARN] could not read {p}: {e}")
        return {}


def save_annotations(task_folder: Path, ann: dict):
    p = annotation_path(task_folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_note": (
            "Per-exp answer to: 'is the operating tool clearly visible in frame 0?'. "
            "Set by scripts/frame0_visibility_inspector.py. Downstream pipelines "
            "(batch_auto_annotator.sh) can use 'visible' to gate --frame0_only "
            "and skip / requeue exps marked 'not_visible'."
        ),
        "labels": list(LABELS),
        "annotations": dict(sorted(ann.items())),
    }
    p.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))


# ──────────────────────────────────────────────────────────────────────
#  Main GUI
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_root", required=True,
                    help="Path to a date-level folder (e.g. .../data/videos_0101)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max_width", type=int, default=1600)
    args = ap.parse_args()

    videos_root = Path(args.videos_root).resolve()
    if not videos_root.is_dir():
        raise SystemExit(f"videos_root not a dir: {videos_root}")

    tasks = discover_tasks(videos_root)
    if not tasks:
        raise SystemExit(f"no task folders under {videos_root}")
    task_names = [t.name for t in tasks]

    print(f"[info] videos_root = {videos_root}")
    print(f"[info] {len(tasks)} task(s): {task_names}")

    # Per-task annotation cache (loaded lazily).
    ann_cache: dict[Path, dict] = {}

    def get_ann(task: Path) -> dict:
        if task not in ann_cache:
            ann_cache[task] = load_annotations(task)
        return ann_cache[task]

    state = {
        "task_idx": 0,
        "exp_idx": 0,
        "exps": [],
        "filter_unannotated": False,
    }

    def visible_exps():
        """Return the (filtered) exp list for the current task."""
        all_exps = state["exps"]
        if not state["filter_unannotated"]:
            return all_exps
        ann = get_ann(tasks[state["task_idx"]])
        return [e for e in all_exps if e.name not in ann]

    def refresh_exps():
        state["exps"] = discover_exps(tasks[state["task_idx"]])

    refresh_exps()

    # ──────────────────────── Viser ────────────────────────
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"[info] viser at http://{args.host}:{args.port}")

    server.gui.add_markdown(f"### videos_root\n`{videos_root.name}`")
    server.gui.add_markdown(
        "Annotate whether the **operating tool is visible at frame 0** "
        "for each experiment. Saved per-task to "
        f"`<task>/{ANNOTATION_FILENAME}`."
    )

    # Task selector
    task_dd = server.gui.add_dropdown("Task", options=task_names,
                                      initial_value=task_names[0])
    with server.gui.add_folder("Task nav", expand_by_default=False):
        btn_prev_task = server.gui.add_button("< Prev task")
        btn_next_task = server.gui.add_button("Next task >")

    task_summary = server.gui.add_markdown("")

    # Filter
    filter_cb = server.gui.add_checkbox("Show unannotated only", initial_value=False)

    # Exp selector
    exp_options_initial = [e.name for e in state["exps"]] or ["<none>"]
    exp_dd = server.gui.add_dropdown("Experiment",
                                     options=exp_options_initial,
                                     initial_value=exp_options_initial[0])
    with server.gui.add_folder("Exp nav", expand_by_default=True):
        btn_prev_exp = server.gui.add_button("< Prev exp")
        btn_next_exp = server.gui.add_button("Next exp >")

    exp_info = server.gui.add_markdown("*loading…*")

    server.gui.add_markdown("---")
    placeholder = np.full((200, 400, 3), 40, dtype=np.uint8)
    frame_img = server.gui.add_image(placeholder, label="frame 0 mosaic")

    server.gui.add_markdown("---")
    current_label_md = server.gui.add_markdown("**Current label**: —")
    with server.gui.add_folder("Annotate (auto-advances)"):
        btn_visible     = server.gui.add_button("[Y] Visible")
        btn_notvisible  = server.gui.add_button("[N] Not visible")
        btn_unsure      = server.gui.add_button("[?] Unsure")
        btn_clear       = server.gui.add_button("Clear annotation")

    server.gui.add_markdown("---")
    save_path_md = server.gui.add_markdown("")

    # ─────────────────────── helpers ───────────────────────

    def update_save_path_md():
        task = tasks[state["task_idx"]]
        save_path_md.content = (
            f"**Annotation file** (auto-saved):\n`{annotation_path(task)}`"
        )

    def update_task_summary():
        task = tasks[state["task_idx"]]
        all_exps = state["exps"]
        ann = get_ann(task)
        v = sum(1 for e in all_exps if ann.get(e.name) == "visible")
        nv = sum(1 for e in all_exps if ann.get(e.name) == "not_visible")
        un = sum(1 for e in all_exps if ann.get(e.name) == "unsure")
        none = len(all_exps) - v - nv - un
        task_summary.content = (
            f"**Task `{task.name}`** — {len(all_exps)} exp(s)  ·  "
            f"visible: **{v}**  ·  not_visible: **{nv}**  ·  "
            f"unsure: **{un}**  ·  unannotated: **{none}**"
        )

    def update_exp_dd():
        names = [e.name for e in visible_exps()] or ["<none>"]
        exp_dd.options = names
        # Clamp idx into visible list, then reflect its name back to dropdown.
        cur_exp = (visible_exps() or [None])[
            min(state["exp_idx"], max(0, len(visible_exps()) - 1))
        ] if visible_exps() else None
        if cur_exp is not None:
            exp_dd.value = cur_exp.name
        else:
            exp_dd.value = names[0]

    def load_current_exp():
        update_save_path_md()
        update_task_summary()

        exps = visible_exps()
        if not exps:
            exp_info.content = (
                f"*task `{tasks[state['task_idx']].name}` has no experiments to show*"
            )
            frame_img.image = placeholder
            current_label_md.content = "**Current label**: —"
            return

        state["exp_idx"] = max(0, min(state["exp_idx"], len(exps) - 1))
        exp = exps[state["exp_idx"]]
        task = tasks[state["task_idx"]]
        idx_str = f"{state['exp_idx'] + 1} / {len(exps)}"
        exp_info.content = f"**{idx_str}** · `{task.name}` / `{exp.name}`"

        mosaic, n_cams, info = load_frame0_mosaic(exp, max_width=args.max_width)
        if mosaic is not None:
            frame_img.image = mosaic
            exp_info.content += f"  \n_{info}_"
        else:
            frame_img.image = placeholder
            exp_info.content += f"  \n*{info}*"

        cur = get_ann(task).get(exp.name, None)
        current_label_md.content = (
            f"**Current label**: `{cur}`" if cur else "**Current label**: —"
        )

    def switch_task_to(task_name):
        try:
            state["task_idx"] = task_names.index(task_name)
        except ValueError:
            return
        state["exp_idx"] = 0
        refresh_exps()
        update_exp_dd()
        load_current_exp()

    def set_label(label: str):
        exps = visible_exps()
        if not exps:
            return
        task = tasks[state["task_idx"]]
        exp = exps[state["exp_idx"]]
        ann = get_ann(task)
        if label is None:
            ann.pop(exp.name, None)
        else:
            ann[exp.name] = label
        save_annotations(task, ann)
        # Auto-advance forward to next available exp (in current filtered view).
        if label is not None:
            # If filter is on, the just-labeled exp may have left the view —
            # don't increment, just refresh and clamp.
            if state["filter_unannotated"]:
                update_exp_dd()
                load_current_exp()
                return
            if state["exp_idx"] < len(exps) - 1:
                state["exp_idx"] += 1
        update_exp_dd()
        load_current_exp()

    # ─────────────────────── handlers ───────────────────────
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

    @filter_cb.on_update
    def _(_evt):
        state["filter_unannotated"] = bool(filter_cb.value)
        state["exp_idx"] = 0
        update_exp_dd()
        load_current_exp()

    @exp_dd.on_update
    def _(_evt):
        names = [e.name for e in visible_exps()]
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
        if state["exp_idx"] < len(visible_exps()) - 1:
            state["exp_idx"] += 1
            update_exp_dd()
            load_current_exp()

    @btn_visible.on_click
    def _(_evt):    set_label("visible")

    @btn_notvisible.on_click
    def _(_evt):    set_label("not_visible")

    @btn_unsure.on_click
    def _(_evt):    set_label("unsure")

    @btn_clear.on_click
    def _(_evt):    set_label(None)

    # ─────────────────────── initial render ───────────────────────
    update_exp_dd()
    load_current_exp()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[info] stopped")


if __name__ == "__main__":
    main()
