#!/usr/bin/env python3
"""Volunteer SAM2 annotation web server (cloud side).

One task = one EXPERIMENT (8 synchronized camera views). The volunteer sees all
views at once, a mid-sequence reference frame to identify the tool, and annotates
each role (color-coded) across every view, then submits once. SAM2 mask decode
runs on CPU (no GPU) against embeddings precomputed on a GPU box.

Run (local dev):
    BUNDLE_DIR=/tmp/va_real PROMPTS_DIR=/tmp/va_real_prompts DB_PATH=/tmp/va_real_tasks.db \
        uvicorn app:app --host 127.0.0.1 --port 8077
    # seed: python seed_tasks.py --bundle /tmp/va_real --db /tmp/va_real_tasks.db

Endpoints:
    GET  /                                 annotation UI
    GET  /api/task/next                    lock and return the next exp (all cameras)
    GET  /api/image/{task_id}/{camera}     frame-0 JPEG for a camera
    GET  /api/refframe/{task_id}/{camera}  mid-sequence reference JPEG for a camera
    POST /api/preview                      decode one camera's points for one role -> mask
    POST /api/submit                       persist per-camera prompt JSON for the whole exp
    GET  /api/stats                        task counts by status
"""

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

import tasks_db
from decoder import Sam2CpuDecoder

BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", "/tmp/va_real"))
PROMPTS_DIR = Path(os.environ.get("PROMPTS_DIR", "/tmp/va_real_prompts"))
DB_PATH = os.environ.get("DB_PATH", "/tmp/va_real_tasks.db")
FRONTEND = Path(__file__).parent / "frontend" / "index.html"

# role -> overlay/point color index (kept in sync with the frontend palette)
ROLE_COLOR_IDX = {"primary_tool": 0, "auxiliary_tool": 1, "manipulated_object": 2}

app = FastAPI(title="Volunteer SAM2 Annotation")
_decoder = None
_manifest_sha = None  # (task,exp,camera) -> image_sha1


def decoder():
    global _decoder
    if _decoder is None:
        _decoder = Sam2CpuDecoder()
    return _decoder


def manifest_sha():
    global _manifest_sha
    if _manifest_sha is None:
        m = json.loads((BUNDLE_DIR / "manifest.json").read_text())
        model = m.get("sam2_model")
        _manifest_sha = {(c["task"], c["exp"], c["camera"]): (c.get("image_sha1"), model)
                         for c in m["cameras"]}
    return _manifest_sha


def db():
    return tasks_db.connect(DB_PATH)


def stem_for(row, camera):
    return BUNDLE_DIR / row["task"] / row["exp"] / camera


class ObjectPrompt(BaseModel):
    role: str = "primary_tool"
    name: str = "object"
    frame_index: int = 0          # which keyframe this role was annotated on
    points: list[list[float]] = []
    labels: list[int] = []
    box: list[float] | None = None
    preview_iou: float | None = None


class PreviewReq(BaseModel):
    task_id: int
    camera: str
    frame_index: int = 0
    role: str = "primary_tool"
    points: list[list[float]] = []
    labels: list[int] = []
    box: list[float] | None = None


class SubmitReq(BaseModel):
    task_id: int
    annotator_id: str = "anon"
    # camera -> list of role objects present in that view
    cameras: dict[str, list[ObjectPrompt]]


class FlagReq(BaseModel):
    task_id: int
    annotator_id: str = "anon"
    reason: str = ""


@app.get("/", response_class=HTMLResponse)
def index():
    return FRONTEND.read_text()


def _task_payload(row):
    cams = json.loads(row["cameras_json"])
    for c in cams:
        kfs = c.get("keyframes", [0])
        c["keyframe_urls"] = {kf: f"/api/image/{row['id']}/{c['camera']}/{kf}" for kf in kfs}
        c["thumb_urls"] = {th: f"/api/thumb/{row['id']}/{c['camera']}/{th}"
                           for th in c.get("thumbs", [])}
    return {"id": row["id"], "task": row["task"], "exp": row["exp"],
            "status": row["status"], "roles": json.loads(row["roles_json"]),
            "cameras": cams}


@app.get("/api/task/next")
def api_next(annotator_id: str = "anon"):
    row = tasks_db.next_task(db(), annotator_id)
    if row is None:
        return JSONResponse({"task": None, "message": "no tasks available"})
    return {"task": _task_payload(row)}


@app.get("/api/task/{task_id}")
def api_task(task_id: int, annotator_id: str = "anon"):
    row = tasks_db.open_task(db(), task_id, annotator_id)
    if row is None:
        raise HTTPException(404, "task not found")
    return {"task": _task_payload(row)}


@app.get("/api/tasks")
def api_tasks(annotator_id: str = "anon"):
    return {"tasks": tasks_db.list_tasks(db(), annotator_id)}


@app.get("/api/progress")
def api_progress(annotator_id: str = "anon"):
    return tasks_db.progress(db(), annotator_id)


def _serve(row, camera, suffix):
    if row is None:
        raise HTTPException(404, "task not found")
    p = Path(f"{stem_for(row, camera)}{suffix}")
    if not p.is_file():
        raise HTTPException(404, f"not in bundle: {p.name}")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/image/{task_id}/{camera}/{frame_index}")
def api_image(task_id: int, camera: str, frame_index: int):
    return _serve(tasks_db.get_task(db(), task_id), camera, f".kf{frame_index}.jpg")


@app.get("/api/thumb/{task_id}/{camera}/{frame_index}")
def api_thumb(task_id: int, camera: str, frame_index: int):
    return _serve(tasks_db.get_task(db(), task_id), camera, f".th{frame_index}.jpg")


def _overlay_png(mask, color_idx, h, w):
    palette = [(255, 80, 80), (80, 200, 80), (80, 140, 255)]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    r, g, b = palette[color_idx % len(palette)]
    rgba[mask, 0], rgba[mask, 1], rgba[mask, 2], rgba[mask, 3] = b, g, r, 150
    ok, buf = cv2.imencode(".png", rgba)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@app.post("/api/preview")
def api_preview(req: PreviewReq):
    row = tasks_db.get_task(db(), req.task_id)
    if row is None:
        raise HTTPException(404, "task not found")
    embed = Path(f"{stem_for(row, req.camera)}.kf{req.frame_index}.embed.npz")
    if not embed.is_file():
        raise HTTPException(404, f"embedding not in bundle: {embed.name}")
    if not req.points and not req.box:
        raise HTTPException(400, "no points or box provided")
    mask, score = decoder().infer(embed, req.points, req.labels, req.box)
    h, w = mask.shape
    return {"overlay": _overlay_png(mask, ROLE_COLOR_IDX.get(req.role, 0), h, w),
            "iou": round(score, 4)}


@app.post("/api/submit")
def api_submit(req: SubmitReq):
    conn = db()
    row = tasks_db.get_task(conn, req.task_id)
    if row is None:
        raise HTTPException(404, "task not found")
    cam_meta = {c["camera"]: c for c in json.loads(row["cameras_json"])}
    sha = manifest_sha()
    out_root = PROMPTS_DIR / row["task"] / row["exp"] / "tool_masks" / "prompts"
    out_root.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    written, ious = [], []

    for camera, objs in req.cameras.items():
        objs = [o for o in objs if o.points or o.box]
        if not objs:
            continue
        meta = cam_meta.get(camera, {})
        image_sha1, sam2_model = sha.get((row["task"], row["exp"], camera), (None, None))
        ious += [o.preview_iou for o in objs if o.preview_iou is not None]
        out = {
            "schema_version": 1,
            "task": row["task"], "exp": row["exp"],
            "camera": camera, "cam_index": meta.get("cam_index"), "frame_index": 0,
            "width": meta.get("width"), "height": meta.get("height"),
            "image_sha1": image_sha1, "sam2_model": sam2_model,
            "annotator_id": req.annotator_id, "submitted_at": now,
            "review_status": "submitted",
            "objects": [o.model_dump() for o in objs],
        }
        (out_root / f"{camera}.json").write_text(json.dumps(out, indent=2))
        written.append(camera)

    if not written:
        raise HTTPException(400, "no annotated cameras in submission")
    mean_iou = round(float(np.mean(ious)), 4) if ious else None
    tasks_db.submit_task(conn, req.task_id, req.annotator_id, mean_iou)
    return {"ok": True, "cameras_saved": written, "dir": str(out_root)}


@app.post("/api/flag_bad")
def api_flag_bad(req: FlagReq):
    conn = db()
    row = tasks_db.get_task(conn, req.task_id)
    if row is None:
        raise HTTPException(404, "task not found")
    tasks_db.flag_bad(conn, req.task_id, req.annotator_id, req.reason)
    # leave a marker that travels with the prompt sync to the GPU side
    out_dir = PROMPTS_DIR / row["task"] / row["exp"] / "tool_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "BAD.json").write_text(json.dumps({
        "task": row["task"], "exp": row["exp"], "status": "bad",
        "reason": req.reason, "annotator_id": req.annotator_id,
        "flagged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, ensure_ascii=False, indent=2))
    return {"ok": True, "exp": row["exp"]}


@app.get("/api/stats")
def api_stats():
    return tasks_db.stats(db())
