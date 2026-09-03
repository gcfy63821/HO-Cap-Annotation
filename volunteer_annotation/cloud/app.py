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

import asyncio
import concurrent.futures

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

import tasks_db
from decoder import DecoderPool

BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", "/tmp/va_real"))
PROMPTS_DIR = Path(os.environ.get("PROMPTS_DIR", "/tmp/va_real_prompts"))
AUTO_PROMPTS_DIR = Path(os.environ.get("AUTO_PROMPTS_DIR", "/tmp/va_real_auto_prompts"))
DB_PATH = os.environ.get("DB_PATH", "/tmp/va_real_tasks.db")
FRONTEND = Path(__file__).parent / "frontend" / "index.html"

# role -> overlay/point color index (kept in sync with the frontend palette)
ROLE_COLOR_IDX = {"primary_tool": 0, "auxiliary_tool": 1, "manipulated_object": 2}

# Trial task: fixed exp used for onboarding. Override via env var (format: "task_path|exp_name").
_TRIAL_DEFAULT = "videos_0101/mallet_crush_dough|20260104_largeplate_mallet_flatten_crush_largedough_1"
_TRIAL_KEY = os.environ.get("TRIAL_TASK", _TRIAL_DEFAULT).split("|")
TRIAL_TASK_PATH, TRIAL_EXP = _TRIAL_KEY[0], _TRIAL_KEY[1]

app = FastAPI(title="Volunteer SAM2 Annotation")
_decoder: DecoderPool | None = None
_manifest_sha = None  # (task,exp,camera) -> image_sha1
# Bounded executor: max concurrent SAM2 inferences = pool size, leaving threads free for other endpoints
_POOL_SIZE = int(os.environ.get("DECODER_POOL_SIZE", "4"))
_infer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_POOL_SIZE)
# Separate low-priority executor for background prewarm — never competes with inference slots
_prewarm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="prewarm")


def decoder() -> DecoderPool:
    global _decoder
    if _decoder is None:
        _decoder = DecoderPool(size=_POOL_SIZE)
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


def _load_prompt_file(path: Path) -> dict:
    """Load a prompt JSON file; return {} on missing/error."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _merged_objects(task: str, exp: str, cam: str) -> list[dict]:
    """
    Merge human + auto prompt objects for one camera.
    Human annotations take priority: if the same (role, frame_index) exists in both,
    the human entry wins. Auto entries fill gaps where humans have not annotated.
    """
    human_pf = PROMPTS_DIR / task / exp / "tool_masks" / "prompts" / f"{cam}.json"
    auto_pf  = AUTO_PROMPTS_DIR / task / exp / "tool_masks" / "prompts" / f"{cam}.json"

    human_data = _load_prompt_file(human_pf)
    auto_data  = _load_prompt_file(auto_pf)

    human_objs = human_data.get("objects", [])
    auto_objs  = auto_data.get("objects", [])

    # Build set of (role, frame_index) covered by humans
    human_rf = {(o["role"], o.get("frame_index", 0)) for o in human_objs if o.get("points")}

    merged = list(human_objs)
    for obj in auto_objs:
        rf = (obj["role"], obj.get("frame_index", 0))
        if rf not in human_rf:
            merged.append(obj)
    return merged


def _task_quality(row) -> dict:
    """Check required-frame coverage for one task (supplement check for existing data).

    Primary tool: camera flagged only if BOTH frame0 AND ~50% (kf[3]) are missing
                  (lenient rule for existing data annotated with old UI).
    Manipulated object: at least one camera must have frame0;
                        if frame0 exists but no camera has ~50% or end frame, require end frame.
    Auxiliary tool / end frame: not required.
    """
    cams = json.loads(row["cameras_json"])
    prompt_dir = PROMPTS_DIR / row["task"] / row["exp"] / "tool_masks" / "prompts"
    errors: list[str] = []
    pt_50_any = False
    pt_last_any = False
    mo_f0_any = False
    mo_50_any = False
    mo_last_any = False

    for cam in cams:
        cam_name = cam["camera"]
        kf = cam.get("keyframes", [0])
        kf0 = kf[0]
        kf_last = kf[-1]
        kf50 = kf[-2] if len(kf) > 2 else kf_last
        pf = prompt_dir / f"{cam_name}.json"
        if not pf.is_file():
            errors.append(f"{cam_name}:no_file")
            continue
        try:
            data = json.loads(pf.read_text())
        except Exception:
            continue
        by_role: dict[str, set] = {}
        for o in data.get("objects", []):
            if o.get("points"):
                by_role.setdefault(o["role"], set()).add(o.get("frame_index", 0))

        pt = by_role.get("primary_tool", set())
        if kf50 in pt:
            pt_50_any = True
        if kf_last in pt:
            pt_last_any = True

        mo = by_role.get("manipulated_object", set())
        if kf0 in mo:
            mo_f0_any = True
        if kf50 in mo:
            mo_50_any = True
        if kf_last in mo:
            mo_last_any = True

    # Primary tool: flag only if NO camera across all views has 50% OR 100%
    if not pt_50_any and not pt_last_any:
        errors.append("pt_50and100:all_cameras")

    # Manipulated object: frame0 required; end frame required only if 50% is also absent
    if not mo_f0_any:
        errors.append("mo_f0:no_camera")
    if not mo_50_any and not mo_last_any:
        errors.append("mo_last:no_camera")

    return {"ok": len(errors) == 0, "errors": errors, "soft_warns": []}


class ObjectPrompt(BaseModel):
    role: str = "primary_tool"
    name: str = "object"
    frame_index: int = 0          # which keyframe this role was annotated on
    points: list[list[float]] = []
    labels: list[int] = []
    box: list[float] | None = None
    preview_iou: float | None = None
    supplemented_at: str | None = None  # set by server when this object was added/changed via supplement


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
    first_point_at: str | None = None   # ISO timestamp of the first annotation click
    is_supplement: bool = False          # true when submitting via supplement/review mode
    # camera -> list of role objects present in that view
    cameras: dict[str, list[ObjectPrompt]]


class FlagReq(BaseModel):
    task_id: int
    annotator_id: str = "anon"
    reason: str = ""


ADMIN_FRONTEND = Path(__file__).parent / "frontend" / "admin.html"

@app.get("/", response_class=HTMLResponse)
def index():
    return FRONTEND.read_text()


@app.get("/admin", response_class=HTMLResponse)
def admin():
    return ADMIN_FRONTEND.read_text()


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


def _prewarm_embeds(row):
    """Background: load all embed files for this task into each decoder's LRU cache."""
    cams = json.loads(row["cameras_json"])
    pool = decoder()
    for cam in cams:
        for kf in cam.get("keyframes", [0]):
            embed = Path(f"{stem_for(row, cam['camera'])}.kf{kf}.embed.npz")
            if embed.is_file():
                try:
                    dec = pool._pool.get(timeout=30)
                    try:
                        dec._load_into_predictor(embed)
                    finally:
                        pool._pool.put(dec)
                except Exception:
                    pass


@app.get("/api/task/next")
def api_next(annotator_id: str = "anon"):
    row = tasks_db.next_task(db(), annotator_id)
    if row is None:
        return JSONResponse({"task": None, "message": "no tasks available"})
    _prewarm_executor.submit(_prewarm_embeds, row)
    return {"task": _task_payload(row)}


@app.get("/api/task/next_sibling/{task_id}")
def api_next_sibling(task_id: int, annotator_id: str = "anon"):
    """Return the next unsubmitted task in the same exp family (same prefix, next number).
    Falls back to None if no sibling exists."""
    import re as _re2
    conn = db()
    cur = tasks_db.get_task(conn, task_id)
    if cur is None:
        return {"task": None}

    m = _re2.search(r'^(.*?)_(\d+)$', cur["exp"])
    if not m:
        return {"task": None}
    prefix, cur_num = m.group(1), int(m.group(2))

    siblings = conn.execute(
        "SELECT * FROM tasks WHERE task=? AND exp LIKE ? ORDER BY exp",
        (cur["task"], prefix + "_%"),
    ).fetchall()

    for s in siblings:
        s = dict(s)
        sm = _re2.search(r'_(\d+)$', s["exp"])
        if sm and int(sm.group(1)) > cur_num and s["status"] != "submitted":
            row = tasks_db.open_task(conn, s["id"], annotator_id)
            if row:
                _prewarm_executor.submit(_prewarm_embeds, row)
                return {"task": _task_payload(row)}

    return {"task": None}


@app.get("/api/task/{task_id}")
def api_task(task_id: int, annotator_id: str = "anon"):
    row = tasks_db.open_task(db(), task_id, annotator_id)
    if row is None:
        raise HTTPException(404, "task not found")
    _prewarm_executor.submit(_prewarm_embeds, row)
    return {"task": _task_payload(row)}


@app.get("/api/tasks")
def api_tasks(annotator_id: str = "anon", near_id: int = 0, window: int = 400):
    """Return tasks near a given id (window/2 before, window/2 after).
    If near_id=0, returns the first `window` tasks."""
    return {"tasks": tasks_db.list_tasks_window(db(), annotator_id, near_id, window)}


@app.get("/api/progress")
def api_progress(annotator_id: str = "anon"):
    return tasks_db.progress(db(), annotator_id)


def _serve(row, camera, suffix):
    if row is None:
        raise HTTPException(404, "task not found")
    p = Path(f"{stem_for(row, camera)}{suffix}")
    if not p.is_file():
        raise HTTPException(404, f"not in bundle: {p.name}")
    return FileResponse(p, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


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
async def api_preview(req: PreviewReq):
    row = tasks_db.get_task(db(), req.task_id)
    if row is None:
        raise HTTPException(404, "task not found")
    embed = Path(f"{stem_for(row, req.camera)}.kf{req.frame_index}.embed.npz")
    if not embed.is_file():
        raise HTTPException(404, f"embedding not in bundle: {embed.name}")
    if not req.points and not req.box:
        raise HTTPException(400, "no points or box provided")
    loop = asyncio.get_event_loop()
    mask, score = await loop.run_in_executor(
        _infer_executor,
        lambda: decoder().infer(embed, req.points, req.labels, req.box)
    )
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

        # Load existing file to detect supplement changes
        existing_by_key: dict[tuple, dict] = {}
        existing_top: dict = {}
        pf = out_root / f"{camera}.json"
        if req.is_supplement and pf.is_file():
            try:
                existing_top = json.loads(pf.read_text())
                for eo in existing_top.get("objects", []):
                    existing_by_key[(eo["role"], eo["frame_index"])] = eo
            except Exception:
                pass

        obj_dicts = []
        for o in objs:
            d = o.model_dump()
            if req.is_supplement:
                key = (o.role, o.frame_index)
                prev = existing_by_key.get(key)
                if prev is None or prev.get("points") != o.points or prev.get("labels") != o.labels:
                    d["supplemented_at"] = now
                else:
                    # unchanged — preserve original supplemented_at if any
                    d["supplemented_at"] = prev.get("supplemented_at")
            obj_dicts.append(d)

        out = {
            "schema_version": 1,
            "task": row["task"], "exp": row["exp"],
            "camera": camera, "cam_index": meta.get("cam_index"), "frame_index": 0,
            "width": meta.get("width"), "height": meta.get("height"),
            "image_sha1": image_sha1, "sam2_model": sam2_model,
            "annotator_id": req.annotator_id, "submitted_at": now,
            "review_status": "submitted",
            "objects": obj_dicts,
        }
        if req.is_supplement:
            out["supplement_submitted_at"] = now
            # preserve original submission time if available
            if existing_top.get("submitted_at"):
                out["original_submitted_at"] = existing_top.get("original_submitted_at") or existing_top["submitted_at"]
        (out_root / f"{camera}.json").write_text(json.dumps(out, indent=2))
        written.append(camera)

    if not written:
        raise HTTPException(400, "no annotated cameras in submission")
    mean_iou = round(float(np.mean(ious)), 4) if ious else None
    tasks_db.submit_task(conn, req.task_id, req.annotator_id, mean_iou, req.first_point_at)
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


@app.get("/api/bulk_quality")
def api_bulk_quality(annotator_id: str = "anon"):
    """Quality check for all submitted tasks of one annotator (for supplement mode)."""
    conn = db()
    rows = tasks_db.list_submitted_by_annotator(conn, annotator_id, limit=2000)
    quality = {}
    for row in rows:
        task_row = tasks_db.get_task(conn, row["id"])
        if task_row:
            quality[row["id"]] = _task_quality(task_row)
    return {"quality": quality}


@app.get("/api/annotators")
def api_annotators():
    """Return all annotators with submission counts (for admin view)."""
    rows = tasks_db.list_annotators(db())
    return {"annotators": rows}


@app.get("/api/annotator_stats/{annotator_id}")
def api_annotator_stats(annotator_id: str):
    return tasks_db.annotator_stats(db(), annotator_id)


@app.get("/api/my_submissions")
def api_my_submissions(annotator_id: str = "anon"):
    """Return tasks submitted by this annotator, newest first."""
    rows = tasks_db.list_submitted_by_annotator(db(), annotator_id)
    return {"submissions": rows}


@app.get("/api/prompts/{task_id}")
def api_prompts(task_id: int):
    """Return annotation points for a task (human + auto merged, human wins)."""
    row = tasks_db.get_task(db(), task_id)
    if row is None:
        raise HTTPException(404, "task not found")
    task, exp = row["task"], row["exp"]

    # Collect camera names from both dirs
    human_dir = PROMPTS_DIR / task / exp / "tool_masks" / "prompts"
    auto_dir  = AUTO_PROMPTS_DIR / task / exp / "tool_masks" / "prompts"
    cam_names = set()
    for d in (human_dir, auto_dir):
        if d.is_dir():
            cam_names.update(f.stem for f in d.glob("*.json"))

    if not cam_names:
        return {"cameras": {}}

    cameras = {}
    for cam in sorted(cam_names):
        objs = _merged_objects(task, exp, cam)
        if objs:
            cameras[cam] = objs
    return {"cameras": cameras}


# ---------------------------------------------------------------------------
# Embedding readiness (used by local push_agent.py)
# ---------------------------------------------------------------------------

def _exp_has_embeddings(row) -> bool:
    """True if ALL expected .embed.npz files exist for this exp."""
    cams = json.loads(row["cameras_json"])
    for c in cams:
        for kf in c.get("keyframes", [0]):
            p = Path(f"{stem_for(row, c['camera'])}.kf{kf}.embed.npz")
            if not p.is_file():
                return False
    return True


@app.get("/api/needs_embeddings")
def api_needs_embeddings(limit: int = 20):
    """Return tasks that are assigned/unassigned but missing embeddings.
    Push agent polls this and rsync the needed exp folders."""
    conn = db()
    # prioritise assigned (in-progress) first, then upcoming unassigned
    rows = tasks_db.list_tasks_for_push(conn, limit=limit)
    missing = []
    for row in rows:
        if not _exp_has_embeddings(row):
            missing.append({"task": row["task"], "exp": row["exp"],
                            "status": row["status"]})
    return {"missing": missing}


@app.get("/api/embedding_ready/{task_id}")
def api_embedding_ready(task_id: int):
    row = tasks_db.get_task(db(), task_id)
    if row is None:
        raise HTTPException(404, "task not found")
    return {"ready": _exp_has_embeddings(row)}


@app.delete("/api/bundle/{task}/{exp}")
def api_bundle_delete(task: str, exp: str):
    """Delete a completed exp's embedding folder from cloud to free disk space.
    Called by push_agent after confirming exp is submitted."""
    exp_dir = BUNDLE_DIR / task / exp
    if exp_dir.is_dir():
        import shutil
        shutil.rmtree(exp_dir)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Trial mode (onboarding)
# ---------------------------------------------------------------------------

@app.get("/api/trial/task")
def api_trial_task():
    """Return the fixed trial task without locking it."""
    row = db().execute(
        "SELECT * FROM tasks WHERE task=? AND exp=?",
        (TRIAL_TASK_PATH, TRIAL_EXP),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "trial task not configured — check TRIAL_TASK env var")
    return {"task": _task_payload(dict(row))}


class TrialRefReq(BaseModel):
    camera: str


@app.post("/api/trial/reference_overlay")
def api_trial_reference_overlay(req: TrialRefReq):
    """Decode ALL reference prompts for one camera and return per-object overlays.
    Each object carries its own frame_index so the frontend can show it on the
    correct frame image. Returns ready=False if no reference annotation exists."""
    row = db().execute(
        "SELECT * FROM tasks WHERE task=? AND exp=?",
        (TRIAL_TASK_PATH, TRIAL_EXP),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "trial task not found")
    row = dict(row)

    ref_path = (PROMPTS_DIR / row["task"] / row["exp"]
                / "tool_masks" / "prompts" / f"{req.camera}.json")
    if not ref_path.is_file():
        return {"objects": [], "ready": False,
                "message": "参考标注尚未完成，请联系管理员"}

    ref = json.loads(ref_path.read_text())
    objects = []
    for obj in ref.get("objects", []):
        pts, labs = obj.get("points", []), obj.get("labels", [])
        if not pts:
            continue
        fi = obj.get("frame_index", 0)
        embed_path = Path(f"{stem_for(row, req.camera)}.kf{fi}.embed.npz")
        if not embed_path.is_file():
            continue  # skip objects whose embedding isn't in the bundle
        mask, _ = decoder().infer(embed_path, pts, labs, obj.get("box"))
        h, w = mask.shape
        objects.append({
            "role": obj["role"],
            "frame_index": fi,
            "overlay": _overlay_png(mask, ROLE_COLOR_IDX.get(obj["role"], 0), h, w),
        })

    return {"objects": objects, "ready": True}


# ── Browse API ────────────────────────────────────────────────────────────────
import re as _re

_VERBS = ['_push_','_flip_','_dig_','_scoop_','_shape_',
          '_spread_','_press_','_open_','_roll_','_flatten_','_crush_']

def _parse_exp(exp: str) -> dict:
    core = _re.sub(r'^\d{8}_', '', exp)
    core = _re.sub(r'_\d+$', '', core)
    for verb in _VERBS:
        if verb in core:
            tool, rest = core.split(verb, 1)
            obj = _re.split(r'_in_|_from_', rest)[0]
            obj = _re.sub(r'^(large|small)(amount_|_amount_|amout_)?', '', obj)
            container = ''
            for sep in ('_in_', '_from_'):
                if sep in rest:
                    container = rest.split(sep, 1)[1]
                    break
            return {'primary_tool': tool, 'action': verb.strip('_'),
                    'manipulated_object': obj, 'container': container}
    return {'primary_tool': core, 'action': '', 'manipulated_object': '', 'container': ''}


def _exp_annotation_status(task: str, exp: str) -> dict:
    """Return {human_cams, auto_cams} for an exp."""
    human_dir = PROMPTS_DIR / task / exp / "tool_masks" / "prompts"
    auto_dir  = AUTO_PROMPTS_DIR / task / exp / "tool_masks" / "prompts"
    human_cams = len(list(human_dir.glob("*.json"))) if human_dir.is_dir() else 0
    auto_cams  = len(list(auto_dir.glob("*.json")))  if auto_dir.is_dir()  else 0
    return {"human_cams": human_cams, "auto_cams": auto_cams}


BROWSE_FRONTEND = Path(__file__).parent / "frontend" / "browse.html"

@app.get("/browse", response_class=HTMLResponse)
def browse():
    return BROWSE_FRONTEND.read_text()


@app.get("/api/browse/groups")
def api_browse_groups(by: str = "primary_tool", task_prefix: str = ""):
    """Return groups of exps by category. by = primary_tool | task | manipulated_object | container"""
    conn = db()
    q = "SELECT id, task, exp, status FROM tasks WHERE 1=1"
    params = []
    if task_prefix:
        q += " AND task LIKE ?"
        params.append(task_prefix + "%")
    rows = conn.execute(q, params).fetchall()

    from collections import defaultdict
    groups: dict[str, dict] = {}

    for row in rows:
        task, exp, status = row["task"], row["exp"], row["status"]
        parsed = _parse_exp(exp)
        if by == "primary_tool":
            key = parsed["primary_tool"]
        elif by == "task":
            key = task.split("/")[-1]
        elif by == "manipulated_object":
            key = parsed["manipulated_object"] or "(unknown)"
        elif by == "container":
            key = parsed["container"] or "(none)"
        else:
            key = parsed["primary_tool"]

        if key not in groups:
            groups[key] = {"name": key, "total": 0, "submitted": 0,
                           "locked": 0, "unassigned": 0}
        g = groups[key]
        g["total"] += 1
        if status == "submitted":
            g["submitted"] += 1
        elif status == "locked":
            g["locked"] += 1
        else:
            g["unassigned"] += 1

    result = sorted(groups.values(), key=lambda x: -x["total"])
    return {"groups": result}


@app.get("/api/browse/exps")
def api_browse_exps(by: str = "primary_tool", name: str = "", task_prefix: str = ""):
    """Return exps in one group with annotation status."""
    conn = db()
    q = "SELECT id, task, exp, status, submitted_at FROM tasks WHERE 1=1"
    params = []
    if task_prefix:
        q += " AND task LIKE ?"
        params.append(task_prefix + "%")
    rows = conn.execute(q, params).fetchall()

    result = []
    for row in rows:
        task, exp = row["task"], row["exp"]
        parsed = _parse_exp(exp)
        if by == "primary_tool":
            key = parsed["primary_tool"]
        elif by == "task":
            key = task.split("/")[-1]
        elif by == "manipulated_object":
            key = parsed["manipulated_object"] or "(unknown)"
        elif by == "container":
            key = parsed["container"] or "(none)"
        else:
            key = parsed["primary_tool"]

        if key != name:
            continue

        ann = _exp_annotation_status(task, exp)
        result.append({
            "id": row["id"],
            "task": task,
            "exp": exp,
            "status": row["status"],
            "submitted_at": row["submitted_at"],
            "human_cams": ann["human_cams"],
            "auto_cams":  ann["auto_cams"],
        })

    result.sort(key=lambda x: x["exp"])
    return {"exps": result}


# ── Template annotation tester (mounted at /tester) ───────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from template_test_server import app as _tester_app
    app.mount("/tester", _tester_app)
except Exception as _e:
    import logging
    logging.warning(f"Template tester not loaded: {_e}")
