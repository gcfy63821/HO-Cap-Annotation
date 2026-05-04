#!/usr/bin/env python3
"""Simpler per-cam frame-0 mask annotator + prompts recorder.

Workflow (web UI, single user):
  1. Enter a videos_root (e.g. /viscam/.../videos_0102) and click "Load".
  2. The page lists every exp whose <task>/frame0_visibility.yaml marks it
     "visible". Each row shows 8 cam tiles with frame 0 (+ existing
     tool_masks/masks.h5 frame-0 mask overlay if present).
  3. Per exp, choose:
       a. "Keep existing"  → marks the exp as OK, does nothing else.
       b. Click on each cam tile (left = +, right = -) until the SAM2
          image-predictor mask looks right. Click "Save prompts" to
          persist the clicks to <annotated>/<task>/<exp>/mask_prompts.json
          and append a row to <annotated_root>/_mask_redo_log.csv.
  4. Run scripts/batch_redo_masks.py later to read each pending row,
     re-run SAM2 image+video predictor with the stored prompts, and
     overwrite tool_masks/masks.h5 + objects.yaml. The CSV row gets
     updated to status=done so you have a paper trail of which exps
     were re-annotated.

Cluster usage (recommended — SAM2 model loads on GPU):
  ssh -L 8765:<node>:8765 cluster
  conda activate hocap-annotation
  python scripts/simple_mask_annotator.py --videos_root /viscam/.../videos_0102

This script does NOT run video propagation — that's batch_redo_masks.py's
job, so the interactive loop stays cheap and you can flag many exps in one
sitting before kicking off the batch run."""

import argparse
import atexit
import csv
import datetime
import getpass
import io
import json
import os
import shutil
import sys
import threading
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from flask import Flask, jsonify, request, send_file

HOCAP_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# Globals
# ============================================================

STATE = {
    "videos_root": None,        # Path
    "annotated_root": None,     # Path
    # Currently selected exp:
    "task": None, "exp": None,
    "exp_folder": None, "ann_dir": None,
    "n_cams": 0, "H": 0, "W": 0, "n_frames": 0,
    "rgbs": None,           # (n_cams, H, W, 3) uint8 RGB at frame 0
    "existing_masks": None, # (n_cams, H, W) uint8 or None
    "clicks": None,         # list[ list[{"x":int,"y":int,"label":int}] ] per cam
}
LOCK = threading.Lock()

REDO_LOG_FIELDS = [
    "task", "exp", "tool_name", "n_cams_with_clicks", "n_clicks_total",
    "prompts_path", "saved_at", "redo_status", "redo_done_at",
    "redo_h5_path", "redo_log",
]

# ============================================================
# Helpers
# ============================================================

def _jpg(arr_rgb, q=82):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, q])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


def overlay(rgb, mask, color=(0, 255, 0), alpha=0.45, contour=(255, 255, 0)):
    out = rgb.copy()
    if mask is not None and mask.sum() > 0:
        layer = rgb.copy(); layer[mask > 0] = color
        out = cv2.addWeighted(rgb, 1 - alpha, layer, alpha, 0)
        cnt, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255,
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnt, -1, contour, 2)
    return out


def draw_clicks(img, clicks):
    if not clicks:
        return
    H, W = img.shape[:2]
    # Scale markers to the source resolution so they're still visible after
    # the browser shrinks the tile to a few hundred pixels wide. ~1.5% of
    # the longest side gives ~15 px on a 1920x1080 image.
    r = max(8, int(round(max(H, W) * 0.012)))
    border = max(2, r // 4)
    for idx, c in enumerate(clicks):
        x, y = int(c["x"]), int(c["y"])
        col = (0, 220, 0) if c["label"] == 1 else (230, 30, 30)
        # White border ring → makes the dot pop on any background.
        cv2.circle(img, (x, y), r + border, (255, 255, 255), -1)
        cv2.circle(img, (x, y), r, col, -1)
        cv2.circle(img, (x, y), r, (0, 0, 0), 1)
        # Order label (1-indexed). White text with black outline.
        text = str(idx + 1)
        scale = max(0.4, r / 22.0)
        thick = max(1, r // 8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        tx, ty = x - tw // 2, y + th // 2
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                     (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                     (255, 255, 255), thick, cv2.LINE_AA)


def load_frame0_from_videos(exp_folder: Path, n_cams: int = None) -> np.ndarray:
    """Read frame 0 of every cam{i}_rgb.mp4. Returns (n_cams, H, W, 3) uint8 RGB."""
    rgb_videos = sorted(exp_folder.glob("cam*_rgb.mp4"))
    if not rgb_videos:
        raise FileNotFoundError(f"no cam*_rgb.mp4 under {exp_folder}")
    if n_cams is not None:
        rgb_videos = rgb_videos[:n_cams]
    rgbs = []
    H = W = None
    for p in rgb_videos:
        cap = cv2.VideoCapture(str(p))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"frame 0 read failed: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if H is None: H, W = rgb.shape[:2]
        rgbs.append(rgb)
    return np.stack(rgbs, axis=0)


def load_existing_frame0_masks(ann_dir: Path, n_cams: int) -> np.ndarray:
    """Try tool_masks/masks.h5 (frame 0) first, then masks/masks.h5.
    Returns (n_cams, H, W) uint8, or None if nothing usable."""
    for rel in ("tool_masks/masks.h5", "masks/masks.h5"):
        p = ann_dir / rel
        if not p.exists():
            continue
        try:
            with h5py.File(p, "r") as f:
                ds = f.get("masks")
                if ds is None or ds.shape[0] == 0: continue
                arr = np.asarray(ds[0])    # (n_cams, H, W) or (n_cams, H, W, 1)
                if arr.ndim == 4 and arr.shape[-1] == 1:
                    arr = arr[..., 0]
                if arr.shape[0] >= n_cams:
                    arr = arr[:n_cams]
                return (arr > 0).astype(np.uint8)
        except Exception:
            continue
    return None


def discover_visible_exps(videos_root: Path):
    """Walk <videos_root>/<task>/<exp>/ and return the list of (task, exp,
    exp_folder, ann_dir, status_flags) where the task's frame0_visibility.yaml
    has labeled the exp 'visible'. Exps in tasks without a yaml are ALL
    listed (to mirror sbatch_run_auto_videos.sh's auto behavior)."""
    out = []
    annotated_root = videos_root.parent / f"{videos_root.name}_annotated"
    for task_dir in sorted(p for p in videos_root.iterdir() if p.is_dir()):
        nm = task_dir.name
        if nm.startswith("realsense_calibrate"): continue
        if nm.startswith("ref_pc"):              continue
        if nm.startswith("posts"):               continue
        if nm == "first_frame":                  continue
        f0_yaml = task_dir / "frame0_visibility.yaml"
        ann = None
        if f0_yaml.exists():
            try:
                ann = (yaml.safe_load(f0_yaml.read_text()) or {}).get("annotations") or {}
            except Exception:
                ann = {}
        for exp_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            if not list(exp_dir.glob("cam*_rgb.mp4")):
                continue
            if ann is not None and ann.get(exp_dir.name) != "visible":
                continue
            ann_dir = annotated_root / nm / exp_dir.name
            tool_h5 = ann_dir / "tool_masks" / "masks.h5"
            prompts = ann_dir / "mask_prompts.json"
            out.append({
                "task": nm,
                "exp": exp_dir.name,
                "exp_folder": str(exp_dir),
                "ann_dir": str(ann_dir),
                "has_tool_masks_h5": tool_h5.exists(),
                "has_prompts": prompts.exists(),
            })
    return annotated_root, out


# ============================================================
# Redo log CSV helpers
# ============================================================

def _redo_log_path(annotated_root: Path) -> Path:
    return annotated_root / "_mask_redo_log.csv"


def _read_redo_log(annotated_root: Path):
    p = _redo_log_path(annotated_root)
    if not p.exists():
        return []
    rows = []
    try:
        with open(p, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    except Exception:
        return []
    return rows


def _write_redo_log(annotated_root: Path, rows):
    p = _redo_log_path(annotated_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REDO_LOG_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in REDO_LOG_FIELDS})


def _upsert_redo_log(annotated_root: Path, entry: dict):
    rows = _read_redo_log(annotated_root)
    key = (entry["task"], entry["exp"])
    found = False
    for i, r in enumerate(rows):
        if (r.get("task"), r.get("exp")) == key:
            rows[i] = {**r, **entry}
            found = True
            break
    if not found:
        rows.append(entry)
    _write_redo_log(annotated_root, rows)
    return rows


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/load_root", methods=["POST"])
def api_load_root():
    payload = request.get_json(silent=True) or {}
    vr = (payload.get("videos_root") or "").strip()
    if not vr:
        return jsonify({"error": "videos_root required"}), 400
    vroot = Path(vr).expanduser().resolve()
    if not vroot.is_dir():
        return jsonify({"error": f"not a dir: {vroot}"}), 400
    annotated_root, exps = discover_visible_exps(vroot)
    redo_log = _read_redo_log(annotated_root)
    redo_status_by_key = {(r.get("task"), r.get("exp")): r for r in redo_log}
    for e in exps:
        r = redo_status_by_key.get((e["task"], e["exp"]))
        e["redo_status"] = (r or {}).get("redo_status", "")
    with LOCK:
        STATE["videos_root"] = vroot
        STATE["annotated_root"] = annotated_root
    return jsonify({
        "videos_root": str(vroot),
        "annotated_root": str(annotated_root),
        "exps": exps,
        "redo_log": redo_log,
    })


@app.route("/api/select_exp", methods=["POST"])
def api_select_exp():
    payload = request.get_json(silent=True) or {}
    task = payload.get("task")
    exp_name = payload.get("exp")
    with LOCK:
        vroot = STATE["videos_root"]
        annotated_root = STATE["annotated_root"]
    if vroot is None:
        return jsonify({"error": "load_root first"}), 400
    if not task or not exp_name:
        return jsonify({"error": "task + exp required"}), 400
    exp_folder = vroot / task / exp_name
    ann_dir = annotated_root / task / exp_name
    if not exp_folder.is_dir():
        return jsonify({"error": f"not a dir: {exp_folder}"}), 400

    rgbs = load_frame0_from_videos(exp_folder)
    n_cams, H, W = rgbs.shape[:3]

    # n_frames from cam0 (assume aligned across cams).
    cap = cv2.VideoCapture(str(exp_folder / "cam0_rgb.mp4"))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    existing = load_existing_frame0_masks(ann_dir, n_cams)

    # Try to load any previously-saved prompts for this exp so the user
    # can iterate without losing prior clicks.
    prev_clicks = [[] for _ in range(n_cams)]
    prompts_path = ann_dir / "mask_prompts.json"
    if prompts_path.exists():
        try:
            data = json.loads(prompts_path.read_text())
            for k, v in (data.get("clicks") or {}).items():
                ki = int(k)
                if 0 <= ki < n_cams:
                    prev_clicks[ki] = list(v)
        except Exception as e:
            print(f"[WARN] could not parse {prompts_path}: {e}")

    with LOCK:
        STATE.update(
            task=task, exp=exp_name,
            exp_folder=exp_folder, ann_dir=ann_dir,
            n_cams=n_cams, H=H, W=W, n_frames=n_frames,
            rgbs=rgbs, existing_masks=existing,
            clicks=prev_clicks,
        )

    return jsonify({
        "task": task, "exp": exp_name,
        "n_cams": n_cams, "H": H, "W": W, "n_frames": n_frames,
        "has_existing_masks": existing is not None,
        "has_prompts": prompts_path.exists(),
        "n_clicks_per_cam": [len(c) for c in prev_clicks],
    })


@app.route("/api/cam/<int:cam>")
def api_cam(cam):
    """Render cam's frame 0 with the existing tool_masks/masks.h5 frame-0
    mask (if any) + the user's click markers. NO SAM2 inference happens
    here — the interactive tool only records clicks. SAM2 reruns these
    later in scripts/batch_redo_masks.py."""
    with LOCK:
        if STATE["rgbs"] is None or cam < 0 or cam >= STATE["n_cams"]:
            return "no exp loaded", 400
        rgb = STATE["rgbs"][cam].copy()
        clicks = list(STATE["clicks"][cam])
        existing = STATE["existing_masks"]
    base_mask = existing[cam] if existing is not None else None
    img = overlay(rgb, base_mask)
    draw_clicks(img, clicks)
    area = int(base_mask.sum()) if base_mask is not None else 0
    cv2.putText(img, f"cam{cam}  existing_area={area}  clicks={len(clicks)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, f"cam{cam}  existing_area={area}  clicks={len(clicks)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return _jpg(img)


@app.route("/api/click", methods=["POST"])
def api_click():
    """Record a click. No SAM2 — just store the (x, y, label) tuple."""
    payload = request.get_json(silent=True) or {}
    cam = int(payload.get("cam", -1))
    x = int(payload.get("x", -1))
    y = int(payload.get("y", -1))
    label = int(payload.get("label", 1))    # 1 = +, 0 = -
    with LOCK:
        if STATE["rgbs"] is None or cam < 0 or cam >= STATE["n_cams"]:
            return jsonify({"error": "bad cam"}), 400
        STATE["clicks"][cam].append({"x": x, "y": y, "label": label})
        n = len(STATE["clicks"][cam])
    return jsonify({"ok": True, "n_clicks": n})


@app.route("/api/reset_clicks", methods=["POST"])
def api_reset():
    payload = request.get_json(silent=True) or {}
    cam = int(payload.get("cam", -1))
    with LOCK:
        if cam < 0 or cam >= STATE["n_cams"]:
            return jsonify({"error": "bad cam"}), 400
        STATE["clicks"][cam] = []
    return jsonify({"ok": True})


@app.route("/api/save_prompts", methods=["POST"])
def api_save_prompts():
    payload = request.get_json(silent=True) or {}
    tool_name = (payload.get("tool_name") or "").strip()
    with LOCK:
        if STATE["task"] is None:
            return jsonify({"error": "no exp selected"}), 400
        task = STATE["task"]; exp = STATE["exp"]
        ann_dir = STATE["ann_dir"]; annotated_root = STATE["annotated_root"]
        clicks = STATE["clicks"]
        n_cams = STATE["n_cams"]
        H, W = STATE["H"], STATE["W"]
    if all(len(c) == 0 for c in clicks):
        return jsonify({"error": "no clicks to save — use 'Keep existing' instead"}), 400

    ann_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = ann_dir / "mask_prompts.json"
    saved_at = datetime.datetime.now().isoformat(timespec="seconds")
    payload_to_save = {
        "task": task, "exp": exp,
        "tool_name": tool_name or None,
        "n_cams": n_cams, "frame_shape": [int(H), int(W)],
        "saved_at": saved_at,
        "clicks": {str(i): clicks[i] for i in range(n_cams) if clicks[i]},
    }
    prompts_path.write_text(json.dumps(payload_to_save, indent=2))

    n_with = sum(1 for c in clicks if c)
    n_total = sum(len(c) for c in clicks)
    rows = _upsert_redo_log(annotated_root, {
        "task": task, "exp": exp,
        "tool_name": tool_name,
        "n_cams_with_clicks": n_with,
        "n_clicks_total": n_total,
        "prompts_path": str(prompts_path),
        "saved_at": saved_at,
        "redo_status": "pending",
        "redo_done_at": "", "redo_h5_path": "", "redo_log": "",
    })
    return jsonify({"ok": True, "prompts_path": str(prompts_path),
                      "redo_log_path": str(_redo_log_path(annotated_root)),
                      "redo_log_rows": rows})


@app.route("/api/keep_existing", methods=["POST"])
def api_keep_existing():
    """User vouched for the existing tool_masks/masks.h5 — log it as 'kept'
    so you can audit later, but do nothing else."""
    with LOCK:
        if STATE["task"] is None:
            return jsonify({"error": "no exp selected"}), 400
        task = STATE["task"]; exp = STATE["exp"]
        annotated_root = STATE["annotated_root"]
    rows = _upsert_redo_log(annotated_root, {
        "task": task, "exp": exp,
        "tool_name": "",
        "n_cams_with_clicks": 0, "n_clicks_total": 0,
        "prompts_path": "",
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "redo_status": "kept",
        "redo_done_at": "", "redo_h5_path": "", "redo_log": "",
    })
    return jsonify({"ok": True, "redo_log_rows": rows})


@app.route("/api/state")
def api_state():
    with LOCK:
        return jsonify({
            "videos_root": str(STATE["videos_root"]) if STATE["videos_root"] else None,
            "annotated_root": str(STATE["annotated_root"]) if STATE["annotated_root"] else None,
            "task": STATE["task"], "exp": STATE["exp"],
            "n_cams": STATE["n_cams"], "H": STATE["H"], "W": STATE["W"],
            "n_clicks_per_cam": [len(c) for c in (STATE["clicks"] or [])],
        })


# ============================================================
# HTML
# ============================================================

INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>simple mask annotator</title>
<style>
  body { background:#1a1a1a;color:#e0e0e0;font-family:-apple-system,sans-serif;margin:0;padding:12px; }
  h1 { font-size:16px;margin:0 0 8px; }
  input,button,select { background:#2a2a2a;color:#e0e0e0;border:1px solid #444;border-radius:4px;padding:5px 9px; }
  button { cursor:pointer; }
  button:hover { background:#3a3a3a; }
  .row { display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px; }
  .exps { max-height:240px;overflow:auto;border:1px solid #333;border-radius:4px;margin-top:6px; }
  table { width:100%;border-collapse:collapse;font-size:12px; }
  th,td { padding:4px 8px;text-align:left;border-bottom:1px solid #333; }
  tr.selected { background:#2a4d6a; }
  tr:hover { background:#222; cursor:pointer; }
  .badge { display:inline-block;font-size:10px;padding:1px 6px;border-radius:3px;margin-right:4px;background:#333;color:#888; }
  .badge.ok { background:#1e4d2b; color:#8f8; }
  .badge.warn { background:#4d4d1e; color:#ff8; }
  .badge.miss { background:#4d1e1e; color:#f88; }
  .grid { display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:8px; }
  .tile { position:relative;background:#000;border:1px solid #222; }
  .tile img { width:100%;height:auto;display:block;cursor:crosshair; }
  .tile .lbl { position:absolute;top:2px;left:4px;color:#fc6;font-size:11px;
               font-family:monospace;text-shadow:0 0 2px #000,0 0 2px #000; }
  .tile .reset { position:absolute;top:2px;right:4px;font-size:10px;
                  background:#333;color:#aaa;border:1px solid #555;cursor:pointer;padding:1px 5px; }
  .ctrls { margin-top:8px;padding:8px;background:#222;border-radius:4px; }
  #status { font-family:monospace;font-size:11px;color:#aaa;margin-top:4px;white-space:pre-wrap; }
  .log { margin-top:12px;font-size:11px;font-family:monospace;color:#aaa; }
</style></head><body>

<h1>simple mask annotator</h1>
<div class="row">
  <input id="vroot" type="text" placeholder="/viscam/.../videos_0102" style="width:480px">
  <button id="btnload">Load videos_root</button>
  <span style="color:#888;font-size:11px">click left=+ / right=−. tiles show existing tool_masks/masks.h5 frame-0 mask + your click dots; SAM2 will re-run later in batch_redo_masks.py.</span>
</div>
<div id="status"></div>

<div class="exps"><table id="exptbl">
<thead><tr><th>task</th><th>exp</th><th>masks.h5</th><th>prompts</th><th>redo</th></tr></thead>
<tbody></tbody>
</table></div>

<div id="annotator" style="display:none">
  <div class="ctrls row">
    <span id="cur_label"></span>
    <input id="tool_name" type="text" placeholder="tool_name (optional)" style="width:220px">
    <button id="btnkeep">Keep existing (no redo)</button>
    <button id="btnsave">Save prompts (queue redo)</button>
  </div>
  <div class="grid" id="grid"></div>
</div>

<div class="log">
  redo log (CSV at <span id="logpath"></span>):
  <pre id="logdump"></pre>
</div>

<script>
let STATE = { exps: [], cur: null, ncams: 0 };

function setStatus(msg) { document.getElementById('status').textContent = msg || ''; }

function badge(ok, label) {
  const cls = ok ? 'ok' : 'miss';
  return `<span class="badge ${cls}">${label}</span>`;
}

function renderExps() {
  const tb = document.querySelector('#exptbl tbody');
  tb.innerHTML = '';
  STATE.exps.forEach((e, idx) => {
    const tr = document.createElement('tr');
    if (STATE.cur && STATE.cur.task === e.task && STATE.cur.exp === e.exp) tr.classList.add('selected');
    tr.innerHTML = `<td>${e.task}</td><td>${e.exp}</td>
                    <td>${badge(e.has_tool_masks_h5, e.has_tool_masks_h5 ? 'h5' : 'no h5')}</td>
                    <td>${badge(e.has_prompts, e.has_prompts ? 'json' : 'no json')}</td>
                    <td>${e.redo_status || ''}</td>`;
    tr.onclick = () => selectExp(e.task, e.exp);
    tb.appendChild(tr);
  });
}

async function loadRoot() {
  const vr = document.getElementById('vroot').value.trim();
  if (!vr) return;
  setStatus('loading...');
  const r = await fetch('/api/load_root', { method:'POST', headers:{'Content-Type':'application/json'},
                                              body: JSON.stringify({videos_root: vr}) });
  const j = await r.json();
  if (j.error) { setStatus('ERROR: ' + j.error); return; }
  STATE.exps = j.exps;
  document.getElementById('logpath').textContent = j.annotated_root + '/_mask_redo_log.csv';
  document.getElementById('logdump').textContent =
      j.redo_log.map(r => `${r.redo_status||'-'}  ${r.task}/${r.exp}  ${r.saved_at||''}  clicks=${r.n_clicks_total||0}  ${r.tool_name||''}`).join('\n');
  setStatus(`loaded ${j.exps.length} visible exps`);
  renderExps();
}

async function selectExp(task, exp) {
  setStatus(`selecting ${task}/${exp} ...`);
  const r = await fetch('/api/select_exp', { method:'POST', headers:{'Content-Type':'application/json'},
                                                body: JSON.stringify({task, exp}) });
  const j = await r.json();
  if (j.error) { setStatus('ERROR: ' + j.error); return; }
  STATE.cur = {task, exp};
  STATE.ncams = j.n_cams;
  document.getElementById('annotator').style.display = '';
  document.getElementById('cur_label').textContent =
      `${task} / ${exp}   (n_cams=${j.n_cams}, n_frames=${j.n_frames}, ` +
      `existing_h5=${j.has_existing_masks}, prompts=${j.has_prompts})`;
  buildGrid();
  refreshAllTiles();
  setStatus(`loaded ${task}/${exp}`);
  renderExps();
}

function buildGrid() {
  const g = document.getElementById('grid');
  g.innerHTML = '';
  for (let i = 0; i < STATE.ncams; i++) {
    const tile = document.createElement('div');
    tile.className = 'tile';
    tile.dataset.cam = i;
    tile.innerHTML = `<span class="lbl">cam ${i}</span>
                       <span class="reset" onclick="resetCam(${i});event.stopPropagation();">↺</span>
                       <img id="tile_${i}" src="" alt="cam ${i}">`;
    const img = tile.querySelector('img');
    img.addEventListener('contextmenu', e => e.preventDefault());
    img.addEventListener('mousedown', e => onTileClick(e, i));
    g.appendChild(tile);
  }
}

function refreshTile(cam) {
  const ts = Date.now();
  document.getElementById(`tile_${cam}`).src = `/api/cam/${cam}?t=${ts}`;
}

function refreshAllTiles() {
  for (let i = 0; i < STATE.ncams; i++) refreshTile(i);
}

async function onTileClick(e, cam) {
  if (e.button !== 0 && e.button !== 2) return;
  const img = e.target;
  const r = img.getBoundingClientRect();
  const x = Math.round((e.clientX - r.left) * (img.naturalWidth / r.width));
  const y = Math.round((e.clientY - r.top)  * (img.naturalHeight / r.height));
  const label = (e.button === 0) ? 1 : 0;
  const resp = await fetch('/api/click', { method:'POST', headers:{'Content-Type':'application/json'},
                                              body: JSON.stringify({cam, x, y, label}) });
  const j = await resp.json();
  if (j.error) { setStatus('ERROR: ' + j.error); return; }
  refreshTile(cam);
}

async function resetCam(cam) {
  await fetch('/api/reset_clicks', { method:'POST', headers:{'Content-Type':'application/json'},
                                       body: JSON.stringify({cam}) });
  refreshTile(cam);
}

document.getElementById('btnload').onclick = loadRoot;

document.getElementById('btnsave').onclick = async () => {
  if (!STATE.cur) { setStatus('select an exp first'); return; }
  const tn = document.getElementById('tool_name').value.trim();
  setStatus('saving prompts ...');
  const r = await fetch('/api/save_prompts', { method:'POST', headers:{'Content-Type':'application/json'},
                                                  body: JSON.stringify({tool_name: tn}) });
  const j = await r.json();
  if (j.error) { setStatus('ERROR: ' + j.error); return; }
  setStatus(`saved prompts → ${j.prompts_path}`);
  document.getElementById('logdump').textContent =
      (j.redo_log_rows||[]).map(r => `${r.redo_status||'-'}  ${r.task}/${r.exp}  ${r.saved_at||''}  clicks=${r.n_clicks_total||0}  ${r.tool_name||''}`).join('\n');
  // Also flip the table flags so the user sees this exp now has prompts.
  STATE.exps.forEach(e => {
    if (e.task === STATE.cur.task && e.exp === STATE.cur.exp) {
      e.has_prompts = true; e.redo_status = 'pending';
    }
  });
  renderExps();
};

document.getElementById('btnkeep').onclick = async () => {
  if (!STATE.cur) { setStatus('select an exp first'); return; }
  setStatus('marking as kept ...');
  const r = await fetch('/api/keep_existing', { method:'POST', headers:{'Content-Type':'application/json'},
                                                  body: JSON.stringify({}) });
  const j = await r.json();
  if (j.error) { setStatus('ERROR: ' + j.error); return; }
  setStatus(`marked kept`);
  document.getElementById('logdump').textContent =
      (j.redo_log_rows||[]).map(r => `${r.redo_status||'-'}  ${r.task}/${r.exp}  ${r.saved_at||''}  clicks=${r.n_clicks_total||0}  ${r.tool_name||''}`).join('\n');
  STATE.exps.forEach(e => {
    if (e.task === STATE.cur.task && e.exp === STATE.cur.exp) {
      e.redo_status = 'kept';
    }
  });
  renderExps();
};

window.resetCam = resetCam;
</script>
</body></html>
"""


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind host. 127.0.0.1 = SSH-tunnel only. 0.0.0.0 = LAN.")
    ap.add_argument("--videos_root", default="",
                    help="Optional pre-fill in the form.")
    args = ap.parse_args()

    if args.videos_root:
        global INDEX_HTML
        INDEX_HTML = INDEX_HTML.replace(
            'id="vroot" type="text"',
            f'id="vroot" type="text" value="{args.videos_root}"', 1)

    print(f"[simple_mask_annotator] http://{args.host}:{args.port}/")
    print(f"  ssh tunnel: ssh -L {args.port}:localhost:{args.port} <cluster>")
    print(f"  NOTE: this tool ONLY records click prompts. No SAM2 runs here. "
          f"To actually regenerate masks.h5, run "
          f"scripts/batch_redo_masks.py --videos_root <root> afterwards.")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
