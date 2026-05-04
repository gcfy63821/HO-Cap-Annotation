#!/usr/bin/env python3
"""Interactive web-based mask annotator (cluster-friendly).

Workflow (every step is reviewable in the browser before committing):
  1. Init exp → loads source first_frame mask (e.g. cam5) + calibration + h5.
  2. Press "Project" → projects source mask to all 8 cams via depth+extrinsics.
     Each cam's tile shows the densified projected seed in green.
  3. Inspect every cam tile. If the seed is wrong:
       - Left-click on the tile to add a POSITIVE SAM2 click (+).
       - Right-click on the tile to add a NEGATIVE click (−).
       - Click ↺ on the cam header to discard clicks and restore the
         projection-only seed.
     Seed updates live via SAM2's image predictor.
  4. Press "Propagate" → for every cam with a non-empty seed, run SAM2 video
     propagation. The result is staged in /dev/shm — nothing on disk yet.
  5. Switch to "Browse propagated" + drag the frame slider to scrub through
     the propagated masks. If something looks bad you can flip back to "Edit
     seed", fix the seed, then re-propagate.
  6. Press "Save" → copies the staged h5 to
        <annotated>/<exp>/tool_masks/masks.h5
     and writes <annotated>/<exp>/tool_masks/objects.yaml.

Cluster usage:
  ssh -L 8765:localhost:8765 cluster
  conda activate hocap-annotation
  python scripts/interactive_mask_annotator.py --port 8765
  # then open http://localhost:8765 on your laptop and fill in the paths.

You can also pre-fill paths via CLI:
  python scripts/interactive_mask_annotator.py \\
    --exp_folder /viscam/.../<task>/<exp> \\
    --calibration_yaml /viscam/.../realsense_calibration_*.yaml \\
    --source_cam 5 --port 8765
"""

import argparse
import atexit
import gc
import getpass
import io
import os
import re
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
sys.path.insert(0, str(HOCAP_ROOT / "scripts"))

# Re-use exactly the same projection / densify / writer logic that the batch
# script uses, so the interactive output is byte-identical to a non-interactive
# `seed_masks_from_extrinsics` run when the user just hits Project + Propagate
# with no manual edits.
from seed_masks_from_extrinsics import (  # noqa: E402
    H5MaskWriter,
    annotated_exp_dir,
    build_temp_frame0_h5,
    densify_seed_mask,
    load_calibration,
    project_mask,
)


# ============================================================
# Global state (single-user web app)
# ============================================================

STATE = {
    "ready": False,
    "videos_root": None,        # workspace root (.../videos_0102), used for browsing
    "exp_folder": None,
    "ann_dir": None,
    "calibration_yaml": None,
    "source_cam": 5,
    "first_frame_dir": None,
    "n_cams": 0, "H": 0, "W": 0, "n_frames": 0,
    "rgbs": None,           # (n_cams, H, W, 3) uint8 RGB at frame 0
    "depths": None,         # (n_cams, H, W) float32 meters at frame 0
    "Ks": None,             # (n_cams, 3, 3)
    "extrs": None,          # (n_cams, 4, 4) cam-to-world
    "rgb_videos": [],       # list of cam{i}_rgb.mp4 paths
    "source_mask": None,    # (H, W) uint8 — source cam first_frame mask
    "projected": [],        # list of (H,W) uint8 — raw projected seed per cam
    "seeds": [],            # list of (H,W) uint8 — current seed per cam
                             #   = densified projection, OR SAM2 image-predictor
                             #     output if user added clicks
    "clicks": [],           # list of [{x,y,label}, ...] per cam
    "image_predictor": None,
    "video_predictor": None,
    "image_predictor_cam": None,    # which cam's features are currently set
    "propagated_h5": None,  # Path to scratch h5 (post-propagation, pre-save)
    "propagated_path": None,        # Path to final masks.h5 (post-save)
    "scratch_dir": None,
    "data_h5_path": None,
    "sam2_checkpoint": None,
    "sam2_model_cfg": None,
    "device": None,
}
LOCK = threading.Lock()


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
    for c in clicks:
        x, y, lbl = int(c["x"]), int(c["y"]), int(c["label"])
        color = (0, 255, 0) if lbl == 1 else (255, 60, 60)
        cv2.circle(img, (x, y), 9, color, -1)
        cv2.circle(img, (x, y), 9, (255, 255, 255), 2)
    return img


def label_top(img, text, scale=0.7):
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
    return img


# ============================================================
# SAM2 lazy build
# ============================================================

def _ensure_image_predictor():
    with LOCK:
        if STATE["image_predictor"] is not None:
            return STATE["image_predictor"]
    print("[sam2] building image predictor ...", flush=True)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sam2(STATE["sam2_model_cfg"], STATE["sam2_checkpoint"],
                        device=device)
    pred = SAM2ImagePredictor(model)
    with LOCK:
        STATE["image_predictor"] = pred
        STATE["device"] = device
    return pred


def _ensure_video_predictor():
    with LOCK:
        if STATE["video_predictor"] is not None:
            return STATE["video_predictor"]
    print("[sam2] building video predictor ...", flush=True)
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    pred = build_sam2_video_predictor(STATE["sam2_model_cfg"],
                                        STATE["sam2_checkpoint"],
                                        device=device)
    with LOCK:
        STATE["video_predictor"] = pred
    return pred


def _refine_with_clicks(cam):
    """Re-run SAM2 image predictor on cam using all current clicks. Updates
    STATE['seeds'][cam]. Returns the new mask."""
    pred = _ensure_image_predictor()
    with LOCK:
        rgb = STATE["rgbs"][cam].copy()
        clicks = list(STATE["clicks"][cam])
        cur = STATE["image_predictor_cam"]
    if cur != cam:
        pred.set_image(rgb)
        with LOCK:
            STATE["image_predictor_cam"] = cam
    if not clicks:
        return None
    pts = np.array([[c["x"], c["y"]] for c in clicks], dtype=np.float32)
    lbls = np.array([c["label"] for c in clicks], dtype=np.int32)
    masks, scores, _ = pred.predict(point_coords=pts, point_labels=lbls,
                                      multimask_output=False)
    m = masks[0].astype(np.uint8)
    with LOCK:
        STATE["seeds"][cam] = m
    return m


# ============================================================
# Flask routes
# ============================================================

app = Flask(__name__)


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/state")
def api_state():
    with LOCK:
        return jsonify({
            "ready": STATE["ready"],
            "exp_folder": str(STATE["exp_folder"]) if STATE["exp_folder"] else None,
            "ann_dir": str(STATE["ann_dir"]) if STATE["ann_dir"] else None,
            "calibration_yaml": str(STATE["calibration_yaml"]) if STATE["calibration_yaml"] else None,
            "source_cam": STATE["source_cam"],
            "n_cams": STATE["n_cams"],
            "H": STATE["H"], "W": STATE["W"],
            "n_frames": STATE["n_frames"],
            "has_source": STATE["source_mask"] is not None and STATE["source_mask"].sum() > 0,
            "has_projection": any((s is not None and s.sum() > 0) for s in STATE["projected"]),
            "has_propagation": STATE["propagated_h5"] is not None,
            "propagated_path": str(STATE["propagated_path"]) if STATE["propagated_path"] else None,
            "seed_areas": [int(s.sum()) for s in (STATE["seeds"] or [])],
            "click_counts": [len(c) for c in (STATE["clicks"] or [])],
        })


@app.route("/api/browse", methods=["GET"])
def api_browse():
    """List tasks/exps under a videos_root, plus per-exp status flags so the
    UI can show which exps already have a tool_masks/masks.h5 (already
    annotated) and which have a first_frame source PNG ready to project."""
    videos_root_arg = request.args.get("videos_root", "").strip()
    source_cam = int(request.args.get("source_cam", 5))
    if not videos_root_arg:
        return jsonify({"error": "videos_root required"}), 400
    vroot = Path(videos_root_arg).expanduser().resolve()
    if not vroot.is_dir():
        return jsonify({"error": f"videos_root not a dir: {vroot}"}), 400
    annotated_root = vroot.parent / f"{vroot.name}_annotated"
    out_tasks = []
    for task_dir in sorted(p for p in vroot.iterdir() if p.is_dir()):
        # Filter out calib / ply / debug dirs (they aren't task folders).
        if task_dir.name.startswith("realsense_calibrate"): continue
        if task_dir.name.startswith("ref_pc"): continue
        if task_dir.name.startswith("posts"): continue
        if task_dir.name == "first_frame": continue
        # Only count directories that look like a task (have at least one
        # exp subfolder with cam*_rgb.mp4 inside).
        exps = []
        for exp_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            if not list(exp_dir.glob("cam*_rgb.mp4")):
                continue
            ann_exp = annotated_root / task_dir.name / exp_dir.name
            ff_dir = ann_exp / "first_frame"
            seed_png = ff_dir / f"cam_{source_cam}_segmentation.png"
            if not seed_png.exists():
                seed_png_alt = ff_dir / f"cam{source_cam}_segmentation.png"
                seed_png = seed_png_alt if seed_png_alt.exists() else seed_png
            tool_h5 = ann_exp / "tool_masks" / "masks.h5"
            legacy_h5 = ann_exp / "masks" / "masks.h5"
            exps.append({
                "name": exp_dir.name,
                "exp_folder": str(exp_dir),
                "ann_dir": str(ann_exp),
                "has_source_mask": seed_png.exists(),
                "source_png": str(seed_png) if seed_png.exists() else None,
                "has_tool_masks_h5": tool_h5.exists(),
                "has_legacy_masks_h5": legacy_h5.exists(),
            })
        if exps:
            out_tasks.append({"name": task_dir.name, "exps": exps})
    with LOCK:
        STATE["videos_root"] = vroot
    return jsonify({
        "videos_root": str(vroot),
        "annotated_root": str(annotated_root),
        "tasks": out_tasks,
    })


@app.route("/api/calibs", methods=["GET"])
def api_calibs():
    """List candidate calibration YAMLs under videos_root + a few common
    parents. Lets the user pick from a dropdown rather than typing."""
    videos_root_arg = request.args.get("videos_root", "").strip()
    if not videos_root_arg:
        return jsonify({"error": "videos_root required"}), 400
    vroot = Path(videos_root_arg).expanduser().resolve()
    if not vroot.is_dir():
        return jsonify({"error": f"not a dir: {vroot}"}), 400
    cands = []
    for pat in ("realsense_calibration*.yaml", "calibration*.yaml"):
        cands.extend(sorted(vroot.glob(pat)))
        # also one level deeper (e.g. realsense_calibrate_xxx/realsense_calibration_*.yaml)
        for sub in sorted(p for p in vroot.iterdir() if p.is_dir()):
            cands.extend(sorted(sub.glob(pat)))
    # de-dup, keep order
    seen = set(); uniq = []
    for c in cands:
        s = str(c)
        if s in seen: continue
        seen.add(s); uniq.append(s)
    # Heuristic: prefer global_aligned > aligned > base
    def pri(s):
        if "global_aligned" in s: return 0
        if "manual_aligned" in s: return 1
        if "_aligned" in s: return 2
        return 3
    uniq.sort(key=pri)
    return jsonify({"calibs": uniq})


@app.route("/api/reset_session", methods=["POST"])
def api_reset_session():
    """Clear current exp's state so the user can pick a different exp.
    Keeps SAM2 predictors loaded (avoids re-loading model). Removes the
    scratch propagated h5."""
    with LOCK:
        prev = STATE.get("propagated_h5")
        STATE.update({
            "ready": False,
            "exp_folder": None, "ann_dir": None,
            "calibration_yaml": None,
            "first_frame_dir": None,
            "n_cams": 0, "H": 0, "W": 0, "n_frames": 0,
            "rgbs": None, "depths": None, "Ks": None, "extrs": None,
            "rgb_videos": [],
            "source_mask": None,
            "projected": [], "seeds": [], "clicks": [],
            "image_predictor_cam": None,
            "propagated_h5": None, "propagated_path": None,
            "data_h5_path": None,
        })
    if prev is not None and Path(prev).exists():
        try: Path(prev).unlink()
        except Exception: pass
    return jsonify({"ok": True})


@app.route("/api/init", methods=["POST"])
def api_init():
    data = request.get_json(force=True)
    # Reset any prior session first (keeps SAM2 predictors loaded though).
    with LOCK:
        prev = STATE.get("propagated_h5")
    if prev is not None and Path(prev).exists():
        try: Path(prev).unlink()
        except Exception: pass

    # Two ways to specify which exp to load:
    #   a) explicit exp_folder (legacy power-user mode)
    #   b) videos_root + task + exp_name (workspace browser mode)
    if "exp_folder" in data and data["exp_folder"]:
        exp_folder = Path(data["exp_folder"]).expanduser().resolve()
    elif data.get("videos_root") and data.get("task") and data.get("exp_name"):
        exp_folder = (Path(data["videos_root"]).expanduser().resolve()
                       / data["task"] / data["exp_name"])
    else:
        return jsonify({"error": "give exp_folder OR (videos_root + task + exp_name)"}), 400
    calib = Path(data["calibration_yaml"]).expanduser().resolve()
    source_cam = int(data.get("source_cam", 5))
    first_frame_dir = data.get("first_frame_dir") or ""

    if not exp_folder.is_dir():
        return jsonify({"error": f"exp_folder not found: {exp_folder}"}), 400
    if not calib.is_file():
        return jsonify({"error": f"calibration_yaml not found: {calib}"}), 400

    ann_dir = annotated_exp_dir(exp_folder)
    ann_dir.mkdir(parents=True, exist_ok=True)
    ff_dir = (Path(first_frame_dir).resolve() if first_frame_dir
                else ann_dir / "first_frame")

    # Build (or reuse) a 1-frame h5 so we can read frame-0 RGB+depth quickly.
    scratch_dir = (STATE["scratch_dir"]
                    or Path("/dev/shm") / getpass.getuser()
                       / f"interactive_mask_{os.getpid()}")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    permanent = exp_folder / "data00000000.h5"
    if permanent.exists():
        data_h5_path = permanent
    else:
        try:
            data_h5_path = build_temp_frame0_h5(exp_folder, scratch_dir)
        except Exception as e:
            return jsonify({"error": f"build h5 failed: {e}"}), 500

    with h5py.File(data_h5_path, "r") as f:
        n_cams = int(f["imgs"].shape[1])
        H, W = int(f["imgs"].shape[2]), int(f["imgs"].shape[3])
        rgbs = np.asarray(f["imgs"][0])
        depths = np.asarray(f["depths"][0]).astype(np.float32) * 0.001

    serials, Ks_all, extrs_all = load_calibration(calib)
    if len(serials) < n_cams:
        return jsonify({"error": f"calib has {len(serials)} cams, h5 has {n_cams}"}), 400
    Ks = Ks_all[:n_cams]
    extrs = extrs_all[:n_cams]

    rgb_videos = sorted(exp_folder.glob("cam*_rgb.mp4"))
    if rgb_videos:
        cap = cv2.VideoCapture(str(rgb_videos[0]))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    else:
        n_frames = 0

    seed_png = ff_dir / f"cam_{source_cam}_segmentation.png"
    if not seed_png.exists():
        alt = ff_dir / f"cam{source_cam}_segmentation.png"
        seed_png = alt if alt.exists() else None
    src_mask = None
    if seed_png is not None:
        m = cv2.imread(str(seed_png), cv2.IMREAD_UNCHANGED)
        if m is not None:
            if m.ndim == 3: m = m[..., 0]
            m = (m > 0).astype(np.uint8)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            src_mask = m

    with LOCK:
        STATE.update(
            ready=True,
            exp_folder=exp_folder, ann_dir=ann_dir,
            calibration_yaml=calib, source_cam=source_cam,
            first_frame_dir=ff_dir,
            data_h5_path=data_h5_path, scratch_dir=scratch_dir,
            n_cams=n_cams, H=H, W=W, n_frames=n_frames,
            rgbs=rgbs, depths=depths, Ks=Ks, extrs=extrs,
            rgb_videos=rgb_videos,
            source_mask=src_mask,
            projected=[np.zeros((H, W), np.uint8) for _ in range(n_cams)],
            seeds=[np.zeros((H, W), np.uint8) for _ in range(n_cams)],
            clicks=[[] for _ in range(n_cams)],
            image_predictor_cam=None,
            propagated_h5=None, propagated_path=None,
        )
        if src_mask is not None and 0 <= source_cam < n_cams:
            STATE["seeds"][source_cam] = src_mask.copy()
            STATE["projected"][source_cam] = src_mask.copy()

    return jsonify({
        "ok": True, "n_cams": n_cams, "n_frames": n_frames,
        "H": H, "W": W,
        "has_source_mask": src_mask is not None,
        "source_area": int(src_mask.sum()) if src_mask is not None else 0,
        "seed_png": str(seed_png) if seed_png else None,
        "ann_dir": str(ann_dir),
    })


@app.route("/api/project", methods=["POST"])
def api_project():
    payload = request.get_json(silent=True) or {}
    dilate_k = int(payload.get("dilate_k", 7))
    close_k = int(payload.get("close_k", 15))
    with LOCK:
        if not STATE["ready"]:
            return jsonify({"error": "not initialized"}), 400
        if STATE["source_mask"] is None or STATE["source_mask"].sum() == 0:
            return jsonify({"error": "no source mask loaded for this exp"}), 400
        sc = STATE["source_cam"]
        H, W = STATE["H"], STATE["W"]
        n_cams = STATE["n_cams"]
        Ks, extrs = STATE["Ks"], STATE["extrs"]
        depths = STATE["depths"]
        src_mask = STATE["source_mask"].copy()
        src_depth = depths[sc].copy()
        K_src, ex_src = Ks[sc], extrs[sc]
    proj_results = []
    seeds = []
    stats = []
    for tgt in range(n_cams):
        if tgt == sc:
            proj_results.append(src_mask.copy())
            seeds.append(src_mask.copy())
            stats.append({"src_pixels": int(src_mask.sum()),
                            "in_image": int(src_mask.sum()),
                            "is_source": True})
            continue
        proj, st = project_mask(
            src_rgb_shape=(H, W),
            depth_src=src_depth, mask_src=src_mask,
            K_src=K_src, extr_src=ex_src,
            K_tgt=Ks[tgt], extr_tgt=extrs[tgt],
            target_shape=(H, W),
        )
        proj_results.append(proj)
        seeds.append(densify_seed_mask(proj, dilate_k=dilate_k, close_k=close_k))
        st["is_source"] = False
        stats.append(st)
    with LOCK:
        STATE["projected"] = proj_results
        STATE["seeds"] = seeds
        STATE["clicks"] = [[] for _ in range(n_cams)]
        STATE["image_predictor_cam"] = None
    return jsonify({"ok": True,
                     "areas": [int(s.sum()) for s in seeds],
                     "stats": stats})


@app.route("/api/click/<int:cam>", methods=["POST"])
def api_click(cam):
    data = request.get_json(force=True)
    action = data.get("action", "add")
    with LOCK:
        if not STATE["ready"] or cam < 0 or cam >= STATE["n_cams"]:
            return jsonify({"error": "invalid"}), 400

    if action == "clear":
        with LOCK:
            STATE["clicks"][cam] = []
            # Restore seed = densified projection (or source mask if cam ==
            # source_cam).
            sc = STATE["source_cam"]
            if cam == sc and STATE["source_mask"] is not None:
                STATE["seeds"][cam] = STATE["source_mask"].copy()
            else:
                proj = STATE["projected"][cam]
                STATE["seeds"][cam] = (
                    densify_seed_mask(proj) if proj.sum() > 0
                    else np.zeros((STATE["H"], STATE["W"]), np.uint8)
                )
            area = int(STATE["seeds"][cam].sum())
        return jsonify({"ok": True, "area": area, "n_clicks": 0})

    if action == "undo":
        with LOCK:
            if STATE["clicks"][cam]:
                STATE["clicks"][cam].pop()
            n = len(STATE["clicks"][cam])
        if n == 0:
            return api_click_clear_internal(cam)
        m = _refine_with_clicks(cam)
        return jsonify({"ok": True, "area": int(m.sum() if m is not None else 0),
                         "n_clicks": n})

    # add
    x, y, label = float(data["x"]), float(data["y"]), int(data["label"])
    with LOCK:
        H, W = STATE["H"], STATE["W"]
        if x < 0 or x >= W or y < 0 or y >= H:
            return jsonify({"error": "click out of image"}), 400
        STATE["clicks"][cam].append({"x": x, "y": y, "label": label})
        n = len(STATE["clicks"][cam])
    try:
        m = _refine_with_clicks(cam)
    except Exception as e:
        return jsonify({"error": f"sam2 image predictor failed: {e}"}), 500
    return jsonify({"ok": True,
                     "area": int(m.sum()) if m is not None else 0,
                     "n_clicks": n})


def api_click_clear_internal(cam):
    with LOCK:
        STATE["clicks"][cam] = []
        sc = STATE["source_cam"]
        if cam == sc and STATE["source_mask"] is not None:
            STATE["seeds"][cam] = STATE["source_mask"].copy()
        else:
            proj = STATE["projected"][cam]
            STATE["seeds"][cam] = (densify_seed_mask(proj) if proj.sum() > 0
                                     else np.zeros((STATE["H"], STATE["W"]), np.uint8))
        area = int(STATE["seeds"][cam].sum())
    return jsonify({"ok": True, "area": area, "n_clicks": 0})


@app.route("/api/seed/<int:cam>")
def api_seed(cam):
    with LOCK:
        if not STATE["ready"] or cam < 0 or cam >= STATE["n_cams"]:
            return "bad cam", 400
        rgb = STATE["rgbs"][cam].copy()
        seed = STATE["seeds"][cam].copy() if STATE["seeds"] else None
        clicks = list(STATE["clicks"][cam]) if STATE["clicks"] else []
        sc = STATE["source_cam"]
        proj_area = int(STATE["projected"][cam].sum()) if STATE["projected"] else 0
    img = overlay(rgb, seed)
    img = draw_clicks(img, clicks)
    tag = " [SOURCE]" if cam == sc else ""
    label_top(img, f"cam{cam}{tag}  seed={int(seed.sum()) if seed is not None else 0}"
                    f"  proj={proj_area}  clicks={len(clicks)}")
    return _jpg(img)


@app.route("/api/frame/<int:cam>/<int:frame>")
def api_frame(cam, frame):
    with LOCK:
        if not STATE["ready"] or cam < 0 or cam >= STATE["n_cams"]:
            return "bad cam", 400
        rgb_videos = list(STATE["rgb_videos"])
        prop = STATE["propagated_h5"]
    if cam >= len(rgb_videos):
        return "no mp4 for cam", 400
    cap = cv2.VideoCapture(str(rgb_videos[cam]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return "frame read failed", 500
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = None
    src = "no h5"
    if prop is not None and Path(prop).exists():
        try:
            with h5py.File(prop, "r") as f:
                ds = f["masks"]
                if frame < ds.shape[0] and cam < ds.shape[1]:
                    mask = (np.asarray(ds[frame, cam]) > 0).astype(np.uint8)
                    src = "scratch h5"
        except Exception as e:
            src = f"h5 err: {e}"
    img = overlay(rgb, mask)
    label_top(img, f"cam{cam}  f{frame}  area={int(mask.sum()) if mask is not None else 0}  ({src})")
    return _jpg(img)


@app.route("/api/propagate", methods=["POST"])
def api_propagate():
    with LOCK:
        if not STATE["ready"]:
            return jsonify({"error": "not init"}), 400
        seeds = [s.copy() if s is not None else None for s in STATE["seeds"]]
        rgb_videos = list(STATE["rgb_videos"])
        n_cams = STATE["n_cams"]
        H, W = STATE["H"], STATE["W"]
        n_frames = STATE["n_frames"]
        scratch_dir = STATE["scratch_dir"]
    if n_frames <= 0:
        return jsonify({"error": "no rgb mp4 frames detected"}), 400
    if all(s is None or s.sum() == 0 for s in seeds):
        return jsonify({"error": "all seeds empty — nothing to propagate"}), 400

    vp = _ensure_video_predictor()
    scratch_h5 = scratch_dir / "propagated_masks.h5"
    if scratch_h5.exists():
        scratch_h5.unlink()
    writer = H5MaskWriter(scratch_h5, n_frames=n_frames, n_cams=n_cams, H=H, W=W)

    import torch
    cam_results = []
    try:
        for cam_idx, mp4 in enumerate(rgb_videos[:n_cams]):
            seed = seeds[cam_idx]
            if seed is None or seed.sum() == 0:
                cam_results.append({"cam": cam_idx, "status": "empty_seed", "n_frames_written": 0})
                continue
            try:
                state = vp.init_state(video_path=str(mp4),
                                       offload_video_to_cpu=True,
                                       offload_state_to_cpu=True)
            except TypeError:
                state = vp.init_state(video_path=str(mp4))
            n_written = 0
            try:
                try: vp.reset_state(state)
                except Exception: pass
                vp.add_new_mask(state, frame_idx=0, obj_id=1,
                                 mask=(seed > 0).astype(np.uint8))
                for fr_idx, oids, mlogits in vp.propagate_in_video(state):
                    for i, oid in enumerate(oids):
                        if oid != 1: continue
                        m = mlogits[i] if isinstance(mlogits, (list, tuple)) else mlogits
                        if torch.is_tensor(m):
                            m = (m > 0).cpu().numpy()
                        else:
                            m = (m > 0)
                        if m.ndim == 3: m = m[0] if m.shape[0] == 1 else m
                        if m.ndim == 4: m = m[0, 0]
                        writer.write_frame(fr_idx, cam_idx, m.astype(np.uint8))
                        n_written += 1
            except Exception as e:
                cam_results.append({"cam": cam_idx, "status": f"err: {e}",
                                      "n_frames_written": n_written})
                continue
            finally:
                try: vp.reset_state(state)
                except Exception: pass
                del state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            cam_results.append({"cam": cam_idx, "status": "ok",
                                  "n_frames_written": n_written,
                                  "seed_area": int(seed.sum())})
    finally:
        writer.close()

    with LOCK:
        STATE["propagated_h5"] = scratch_h5
        # Force a reset of any cached video predictor-state tracking.
    return jsonify({"ok": True, "scratch_h5": str(scratch_h5),
                     "cam_results": cam_results})


@app.route("/api/save", methods=["POST"])
def api_save():
    payload = request.get_json(silent=True) or {}
    tool_name = payload.get("tool_name")
    with LOCK:
        if STATE["propagated_h5"] is None or not STATE["propagated_h5"].exists():
            return jsonify({"error": "no propagation in scratch — run Propagate first"}), 400
        ann = STATE["ann_dir"]
        exp = STATE["exp_folder"]
        scratch_h5 = STATE["propagated_h5"]
    out_dir = ann / "tool_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "masks.h5"
    if final.exists():
        final.unlink()
    shutil.copy2(scratch_h5, final)
    if not tool_name:
        tool_name = re.sub(r"_\d+$", "", exp.name)
    with open(out_dir / "objects.yaml", "w") as f:
        yaml.safe_dump({"objects": [tool_name]}, f)
    with LOCK:
        STATE["propagated_path"] = final
    return jsonify({"ok": True, "path": str(final),
                     "objects_yaml": str(out_dir / "objects.yaml"),
                     "tool_name": tool_name})


@app.route("/api/discard_propagation", methods=["POST"])
def api_discard():
    """Drop the staged propagation so user can edit seeds and re-propagate."""
    with LOCK:
        p = STATE["propagated_h5"]
        STATE["propagated_h5"] = None
        STATE["propagated_path"] = None
    if p is not None and Path(p).exists():
        try: Path(p).unlink()
        except Exception: pass
    return jsonify({"ok": True})


# ============================================================
# HTML page
# ============================================================

INDEX_HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Interactive Mask Annotator</title>
<style>
  body { font-family: ui-monospace, monospace; margin: 0; background: #1a1a1a; color: #ddd; }
  #topbar { padding: 8px 10px; background: #222; border-bottom: 1px solid #444;
            display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  #topbar input[type=text] { width: 380px; padding: 4px; background:#111; color:#ddd; border:1px solid #444; }
  #topbar input[type=number] { width: 50px; padding: 4px; background:#111; color:#ddd; border:1px solid #444; }
  button { padding: 5px 10px; cursor: pointer; background: #333; color: #ddd; border: 1px solid #555; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-strong { background: #2a6; color: white; border: 0; padding: 7px 14px; font-weight: bold; }
  .btn-strong:disabled { background: #555; }
  .btn-warn { background: #b53; color: white; border: 0; }
  .controls { padding: 6px 10px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
              background: #1d1d1d; border-bottom: 1px solid #444; }
  .controls label { padding: 4px 8px; background: #2a2a2a; border: 1px solid #444;
                     border-radius: 3px; cursor: pointer; user-select: none; }
  .controls label.active { background: #2a6; color: white; border-color: #2a6; }
  .status { color: #fc6; padding: 4px 8px; }
  #grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; padding: 8px; }
  .cam { background: #111; border: 1px solid #333; }
  .cam .header { padding: 4px 8px; background: #1f1f1f; font-size: 11px;
                  display: flex; justify-content: space-between; align-items: center; }
  .cam canvas { width: 100%; height: auto; cursor: crosshair; display: block; background: #000; }
  #frame_slider { width: 380px; }
  .legend { font-size: 11px; color: #aaa; padding: 0 10px; }
  kbd { background: #333; border: 1px solid #555; padding: 1px 4px; border-radius: 3px; font-size: 10px; }
</style>
</head><body>

<div id="topbar">
  <span>videos_root:</span>
  <input id="videos_root" type="text" placeholder="/viscam/projects/robotool/data/videos_0102">
  <span>src cam:</span>
  <input id="source_cam" type="number" value="5" min="0" max="15">
  <button onclick="loadWorkspace()">Browse</button>
  <span style="border-left:1px solid #555;height:20px;margin:0 6px;"></span>
  <span>task:</span>
  <select id="task_select" onchange="onTaskSelected()" style="min-width:240px;background:#111;color:#ddd;border:1px solid #444;padding:4px"></select>
  <span>exp:</span>
  <select id="exp_select" onchange="onExpSelected()" style="min-width:280px;background:#111;color:#ddd;border:1px solid #444;padding:4px"></select>
  <span>calib:</span>
  <select id="calib_select" style="min-width:280px;background:#111;color:#ddd;border:1px solid #444;padding:4px"></select>
  <button class="btn-strong" onclick="initExp()">Init</button>
  <button onclick="resetSession()" title="Clear state to pick another exp">Reset</button>
  <span class="status" id="status">idle</span>
</div>

<details style="background:#181818;border-bottom:1px solid #333;padding:4px 10px">
  <summary style="cursor:pointer;color:#aaa;font-size:12px">advanced: enter paths directly</summary>
  <div style="padding:6px 0;display:flex;gap:6px;flex-wrap:wrap;align-items:center">
    <span>exp_folder:</span>
    <input id="exp_folder" type="text" style="width:480px;padding:4px;background:#111;color:#ddd;border:1px solid #444"
           placeholder="/viscam/.../<task>/<exp> (overrides task/exp dropdowns when filled)">
    <span>calib path:</span>
    <input id="calib" type="text" style="width:480px;padding:4px;background:#111;color:#ddd;border:1px solid #444"
           placeholder="/viscam/.../realsense_calibration_*.yaml (overrides calib dropdown)">
  </div>
</details>

<div class="controls">
  <button class="btn-strong" id="btn_project" onclick="projectMask()" disabled>1) Project</button>
  <button class="btn-strong" id="btn_propagate" onclick="propagate()" disabled>2) Propagate (SAM2)</button>
  <button class="btn-strong" id="btn_save" onclick="saveH5()" disabled>3) Save masks.h5</button>
  <button class="btn-warn" id="btn_discard" onclick="discardProp()" disabled>Discard staged</button>
  <span style="border-left:1px solid #555;height:24px;margin:0 6px;"></span>
  <span>view:</span>
  <label class="active" id="mode_seed_lbl">
    <input type="radio" name="vm" value="seed" checked onchange="setMode('seed')"> edit seed
  </label>
  <label id="mode_prop_lbl">
    <input type="radio" name="vm" value="prop" onchange="setMode('prop')"> browse propagated
  </label>
  <span style="border-left:1px solid #555;height:24px;margin:0 6px;"></span>
  <span>frame:</span>
  <input type="range" id="frame_slider" min="0" max="0" value="0" oninput="onFrameChange()">
  <span id="frame_label">0 / 0</span>
</div>

<div class="legend">
  <kbd>Left-click</kbd> = positive +&nbsp;&nbsp;
  <kbd>Right-click</kbd> = negative −&nbsp;&nbsp;
  <kbd>↺</kbd> per cam = clear clicks &amp; restore projected seed
</div>

<div id="grid"></div>

<script>
let nCams = 0;
let nFrames = 0;
let curFrame = 0;
let mode = "seed";    // 'seed' (edit) or 'prop' (browse propagated)

function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}

let WORKSPACE = {videos_root: "", tasks: []};   // populated by /api/browse
let CALIBS = [];

async function loadWorkspace() {
  const vroot = document.getElementById("videos_root").value.trim();
  const sc = parseInt(document.getElementById("source_cam").value);
  if (!vroot) { setStatus("ERR: fill videos_root"); return; }
  setStatus("scanning workspace...");
  const r = await fetch(`/api/browse?videos_root=${encodeURIComponent(vroot)}&source_cam=${sc}`);
  const j = await r.json();
  if (j.error) { setStatus("ERR: " + j.error); return; }
  WORKSPACE = j;
  // populate task dropdown
  const ts = document.getElementById("task_select");
  ts.innerHTML = "";
  if (!j.tasks.length) { ts.innerHTML = "<option>(no tasks)</option>"; setStatus("no tasks under " + vroot); return; }
  j.tasks.forEach(t => {
    const o = document.createElement("option");
    o.value = t.name; o.textContent = `${t.name} (${t.exps.length} exps)`;
    ts.appendChild(o);
  });
  await loadCalibs(vroot);
  onTaskSelected();
  setStatus(`workspace loaded: ${j.tasks.length} tasks, ${j.tasks.reduce((s,t)=>s+t.exps.length,0)} exps total`);
}

async function loadCalibs(vroot) {
  const r = await fetch(`/api/calibs?videos_root=${encodeURIComponent(vroot)}`);
  const j = await r.json();
  const sel = document.getElementById("calib_select");
  sel.innerHTML = "";
  CALIBS = j.calibs || [];
  if (!CALIBS.length) {
    sel.innerHTML = "<option value=''>(none found — use 'enter paths directly')</option>";
    return;
  }
  CALIBS.forEach((c, i) => {
    const o = document.createElement("option");
    o.value = c;
    // show last 2 path components for readability
    const parts = c.split("/");
    o.textContent = parts.slice(-2).join("/") + (i === 0 ? "  ★" : "");
    sel.appendChild(o);
  });
}

function onTaskSelected() {
  const taskName = document.getElementById("task_select").value;
  const task = WORKSPACE.tasks.find(t => t.name === taskName);
  const sel = document.getElementById("exp_select");
  sel.innerHTML = "";
  if (!task) return;
  task.exps.forEach(e => {
    const o = document.createElement("option");
    o.value = e.name;
    let badges = "";
    badges += e.has_source_mask ? "" : " [no src]";
    badges += e.has_tool_masks_h5 ? " ✓DONE" : "";
    badges += e.has_legacy_masks_h5 && !e.has_tool_masks_h5 ? " (legacy)" : "";
    o.textContent = e.name + badges;
    o.dataset.expFolder = e.exp_folder;
    o.dataset.hasSource = e.has_source_mask ? "1" : "0";
    o.dataset.hasTool = e.has_tool_masks_h5 ? "1" : "0";
    sel.appendChild(o);
  });
  onExpSelected();
}

function onExpSelected() {
  const sel = document.getElementById("exp_select");
  const opt = sel.options[sel.selectedIndex];
  if (!opt) return;
  setStatus(`selected: ${opt.value}` +
            (opt.dataset.hasSource === "1" ? "" : "   [WARN no first_frame source mask]") +
            (opt.dataset.hasTool === "1" ? "   [already has tool_masks/masks.h5]" : ""));
}

async function initExp() {
  const sc = parseInt(document.getElementById("source_cam").value);
  // allow direct paths to override the dropdowns
  const expDirect = document.getElementById("exp_folder").value.trim();
  const calibDirect = document.getElementById("calib").value.trim();
  let body;
  if (expDirect) {
    if (!calibDirect) { setStatus("ERR: when using direct exp_folder, also fill calib path"); return; }
    body = {exp_folder: expDirect, calibration_yaml: calibDirect, source_cam: sc};
  } else {
    const taskSel = document.getElementById("task_select");
    const expSel = document.getElementById("exp_select");
    const calibSel = document.getElementById("calib_select");
    if (!taskSel.value || !expSel.value) { setStatus("ERR: pick task + exp first (Browse)"); return; }
    const calib = calibDirect || calibSel.value;
    if (!calib) { setStatus("ERR: pick a calibration yaml"); return; }
    body = {videos_root: WORKSPACE.videos_root, task: taskSel.value,
             exp_name: expSel.value, calibration_yaml: calib, source_cam: sc};
  }
  setStatus("init...");
  const r = await fetch("/api/init", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if (j.error) { setStatus("ERR: " + j.error); return; }
  nCams = j.n_cams; nFrames = j.n_frames;
  document.getElementById("frame_slider").max = Math.max(0, nFrames - 1);
  document.getElementById("frame_slider").value = 0;
  document.getElementById("frame_label").textContent = `0 / ${nFrames}`;
  document.getElementById("btn_project").disabled = !j.has_source_mask;
  document.getElementById("btn_propagate").disabled = true;
  document.getElementById("btn_save").disabled = true;
  document.getElementById("btn_discard").disabled = true;
  buildGrid();
  setMode("seed");
  setStatus(`Ready. ann=${j.ann_dir}  n_cams=${j.n_cams}  n_frames=${j.n_frames}  source_area=${j.source_area}  png=${j.seed_png || 'NONE'}`);
  refreshAll();
}

async function resetSession() {
  await fetch("/api/reset_session", {method: "POST"});
  document.getElementById("grid").innerHTML = "";
  nCams = 0; nFrames = 0; curFrame = 0;
  document.getElementById("frame_slider").max = 0;
  document.getElementById("frame_slider").value = 0;
  document.getElementById("frame_label").textContent = "0 / 0";
  document.getElementById("btn_project").disabled = true;
  document.getElementById("btn_propagate").disabled = true;
  document.getElementById("btn_save").disabled = true;
  document.getElementById("btn_discard").disabled = true;
  setStatus("session reset — pick another exp + Init");
}

function buildGrid() {
  const g = document.getElementById("grid");
  g.innerHTML = "";
  for (let c = 0; c < nCams; c++) {
    const div = document.createElement("div");
    div.className = "cam";
    div.innerHTML = `
      <div class="header">
        <span>cam${c}</span>
        <span>
          <button onclick="resetCam(${c})">↺ reset</button>
          <button onclick="undoCam(${c})">↶ undo</button>
        </span>
      </div>
      <canvas id="canvas_${c}"></canvas>`;
    g.appendChild(div);
    const cv = document.getElementById("canvas_" + c);
    cv.addEventListener("mousedown", (e) => onCanvasClick(c, e));
    cv.addEventListener("contextmenu", (e) => e.preventDefault());
  }
}

function loadCamImage(cam) {
  const url = (mode === "seed")
      ? `/api/seed/${cam}?t=` + Date.now()
      : `/api/frame/${cam}/${curFrame}?t=` + Date.now();
  const img = new Image();
  img.onload = () => {
    const cv = document.getElementById("canvas_" + cam);
    if (!cv) return;
    cv.width = img.naturalWidth;
    cv.height = img.naturalHeight;
    cv.getContext("2d").drawImage(img, 0, 0);
  };
  img.onerror = () => console.warn("img err cam", cam);
  img.src = url;
}

function refreshAll() {
  for (let c = 0; c < nCams; c++) loadCamImage(c);
}

function setMode(m) {
  mode = m;
  document.getElementById("mode_seed_lbl").classList.toggle("active", m === "seed");
  document.getElementById("mode_prop_lbl").classList.toggle("active", m === "prop");
  document.querySelector('input[name=vm][value=' + m + ']').checked = true;
  refreshAll();
}

function onFrameChange() {
  curFrame = parseInt(document.getElementById("frame_slider").value);
  document.getElementById("frame_label").textContent = `${curFrame} / ${nFrames}`;
  if (mode !== "prop") setMode("prop");
  else refreshAll();
}

async function onCanvasClick(cam, e) {
  if (mode !== "seed") {
    setStatus("Switch to 'edit seed' mode to click.");
    return;
  }
  const cv = document.getElementById("canvas_" + cam);
  const rect = cv.getBoundingClientRect();
  const sx = cv.width / rect.width;
  const sy = cv.height / rect.height;
  const x = (e.clientX - rect.left) * sx;
  const y = (e.clientY - rect.top) * sy;
  // button: 0=left=pos, 2=right=neg
  const lbl = (e.button === 2) ? 0 : 1;
  setStatus(`cam${cam}: click (${x.toFixed(0)},${y.toFixed(0)}) ${lbl ? '+' : '−'} ...`);
  const r = await fetch(`/api/click/${cam}`, {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({action: "add", x, y, label: lbl})
  });
  const j = await r.json();
  if (j.error) setStatus("ERR: " + j.error);
  else setStatus(`cam${cam}: area=${j.area}, clicks=${j.n_clicks}`);
  loadCamImage(cam);
}

async function resetCam(cam) {
  await fetch(`/api/click/${cam}`, {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({action: "clear"})
  });
  setStatus(`cam${cam}: reset to projection`);
  loadCamImage(cam);
}

async function undoCam(cam) {
  await fetch(`/api/click/${cam}`, {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({action: "undo"})
  });
  setStatus(`cam${cam}: undo`);
  loadCamImage(cam);
}

async function projectMask() {
  setStatus("projecting...");
  const r = await fetch("/api/project", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: "{}"
  });
  const j = await r.json();
  if (j.error) { setStatus("ERR: " + j.error); return; }
  document.getElementById("btn_propagate").disabled = false;
  setStatus("projected. areas=[" + j.areas.join(",") + "]");
  setMode("seed");
}

async function propagate() {
  if (!confirm("Run SAM2 propagation across all cams? May take minutes.")) return;
  setStatus("propagating SAM2...");
  document.getElementById("btn_propagate").disabled = true;
  const r = await fetch("/api/propagate", {method: "POST"});
  const j = await r.json();
  document.getElementById("btn_propagate").disabled = false;
  if (j.error) { setStatus("ERR: " + j.error); return; }
  document.getElementById("btn_save").disabled = false;
  document.getElementById("btn_discard").disabled = false;
  const lines = j.cam_results.map(r => `cam${r.cam}=${r.status}(${r.n_frames_written})`).join(" ");
  setStatus("propagated -> " + j.scratch_h5 + "  " + lines);
  setMode("prop");
}

async function saveH5() {
  setStatus("saving masks.h5...");
  const r = await fetch("/api/save", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
  const j = await r.json();
  if (j.error) { setStatus("ERR: " + j.error); return; }
  setStatus(`SAVED: ${j.path}  (objects=[${j.tool_name}])`);
}

async function discardProp() {
  if (!confirm("Discard the staged propagation? You'll need to re-Propagate.")) return;
  await fetch("/api/discard_propagation", {method:"POST"});
  document.getElementById("btn_save").disabled = true;
  document.getElementById("btn_discard").disabled = true;
  setStatus("staged propagation discarded");
  setMode("seed");
}
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
                    help="Bind host. Default 127.0.0.1 (use SSH tunnel from laptop). "
                         "Use 0.0.0.0 to expose on the LAN (only on a trusted network).")
    ap.add_argument("--videos_root", default="",
                    help="Optional pre-fill for the workspace browser "
                         "(e.g. /viscam/projects/robotool/data/videos_0102). "
                         "After loading you pick task+exp from dropdowns and "
                         "you can switch between exps without restarting.")
    ap.add_argument("--exp_folder", default="",
                    help="Optional direct pre-fill (legacy single-exp mode).")
    ap.add_argument("--calibration_yaml", default="",
                    help="Optional direct pre-fill (legacy single-exp mode).")
    ap.add_argument("--source_cam", type=int, default=5)
    ap.add_argument("--sam2_checkpoint", type=str,
                    default=os.environ.get(
                        "SAM2_CKPT",
                        str(HOCAP_ROOT.parent / "mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt")
                    ))
    ap.add_argument("--sam2_model_cfg", type=str,
                    default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = ap.parse_args()

    STATE["sam2_checkpoint"] = args.sam2_checkpoint
    STATE["sam2_model_cfg"] = args.sam2_model_cfg
    STATE["source_cam"] = args.source_cam

    scratch_dir = Path("/dev/shm") / getpass.getuser() / f"interactive_mask_{os.getpid()}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    STATE["scratch_dir"] = scratch_dir

    def _cleanup():
        try:
            if scratch_dir.exists():
                shutil.rmtree(scratch_dir, ignore_errors=True)
        except Exception:
            pass
    atexit.register(_cleanup)

    # Pre-fill the form via a dynamic HTML mutation: replace placeholders.
    global INDEX_HTML
    if args.videos_root:
        INDEX_HTML = INDEX_HTML.replace(
            'id="videos_root" type="text"',
            f'id="videos_root" type="text" value="{args.videos_root}"', 1)
    if args.exp_folder:
        INDEX_HTML = INDEX_HTML.replace(
            'id="exp_folder" type="text"',
            f'id="exp_folder" type="text" value="{args.exp_folder}"', 1)
    if args.calibration_yaml:
        INDEX_HTML = INDEX_HTML.replace(
            'id="calib" type="text"',
            f'id="calib" type="text" value="{args.calibration_yaml}"', 1)
    if args.source_cam is not None:
        INDEX_HTML = INDEX_HTML.replace(
            'id="source_cam" type="number" value="5"',
            f'id="source_cam" type="number" value="{args.source_cam}"', 1)

    print(f"[interactive_mask_annotator] starting on http://{args.host}:{args.port}/")
    print(f"  ssh tunnel from your laptop:")
    print(f"    ssh -L {args.port}:localhost:{args.port} <cluster>")
    print(f"  then open  http://localhost:{args.port}/")
    print(f"  scratch:   {scratch_dir}")
    print(f"  sam2 ckpt: {args.sam2_checkpoint}")
    print(f"  sam2 cfg:  {args.sam2_model_cfg}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
