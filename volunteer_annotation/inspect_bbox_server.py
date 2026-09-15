"""
bbox 质量检查服务器 (port 8089)

启动：
  conda run -n hocap-annotation python volunteer_annotation/inspect_bbox_server.py

访问：
  http://localhost:8089
  或公网 http://49.233.81.226:8089
"""
import json, cv2, base64
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

AP_ROOT    = Path("/data/robotool/_va_bundle_v2_auto_prompts")
EMBED_ROOT = Path("/data/robotool/_va_bundle_v2")

app = FastAPI(title="BBox Inspector")

ROLE_COLORS = {
    "primary_tool":       (0,   200,  80),
    "manipulated_object": (30,  160, 255),
    "auxiliary_tool":     (220, 100, 220),
}


def draw_overlay(img: np.ndarray, objects: list) -> np.ndarray:
    out = img.copy()
    for obj in objects:
        if obj.get("frame_index") != 0:
            continue
        role  = obj.get("role", "primary_tool")
        color = ROLE_COLORS.get(role, (200, 200, 200))
        bgr   = (color[2], color[1], color[0])

        bbox = obj.get("bbox")
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)

        pts  = obj.get("points", [])
        lbls = obj.get("labels", [])
        for (px, py), lbl in zip(pts, lbls):
            c = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(out, (int(px), int(py)), 5, c, -1)
            cv2.circle(out, (int(px), int(py)), 5, (255,255,255), 1)

    return out


def img_to_b64(img: np.ndarray, max_w: int = 640) -> str:
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


# ── API ──────────────────────────────────────────────────────────────────────

@app.get("/api/tasks")
def list_tasks():
    tasks = []
    for p in sorted(AP_ROOT.iterdir()):
        if p.is_dir() and p.name.startswith("videos_"):
            for sub in sorted(p.iterdir()):
                if sub.is_dir():
                    tasks.append(f"{p.name}/{sub.name}")
    return tasks


@app.get("/api/exps")
def list_exps(task: str = Query(...), keyword: str = Query("")):
    task_dir = AP_ROOT / task
    if not task_dir.exists():
        return []
    exps = []
    for exp_dir in sorted(task_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if keyword and keyword not in exp_dir.name:
            continue
        prompt_dir = exp_dir / "tool_masks" / "prompts"
        if not prompt_dir.exists():
            continue
        # check at least one json has bbox
        has_bbox = False
        cam_list = []
        for jf in sorted(prompt_dir.glob("cam*_rgb.json")):
            data = json.loads(jf.read_text())
            for obj in data.get("objects", []):
                if obj.get("frame_index") == 0 and "bbox" in obj:
                    has_bbox = True
                    break
            cam_list.append(jf.stem)
        exps.append({"exp": exp_dir.name, "cams": cam_list, "has_bbox": has_bbox})
    return exps


@app.get("/api/image")
def get_image(task: str = Query(...), exp: str = Query(...), cam: str = Query(...)):
    # load json
    json_path = AP_ROOT / task / exp / "tool_masks" / "prompts" / f"{cam}.json"
    if not json_path.exists():
        raise HTTPException(404, f"JSON not found: {json_path}")
    data = json.loads(json_path.read_text())
    objects = data.get("objects", [])

    # load kf0 image
    img_path = EMBED_ROOT / task / exp / f"{cam}.kf0.jpg"
    if not img_path.exists():
        # try alternate
        candidates = list((EMBED_ROOT / task / exp).glob(f"{cam}.kf*.jpg"))
        if not candidates:
            raise HTTPException(404, f"Image not found for {task}/{exp}/{cam}")
        img_path = sorted(candidates)[0]

    img = cv2.imread(str(img_path))
    if img is None:
        raise HTTPException(500, "Failed to read image")

    overlaid = draw_overlay(img, objects)
    b64 = img_to_b64(overlaid)

    # summarize
    frame0 = [o for o in objects if o.get("frame_index") == 0]
    bbox_count = sum(1 for o in frame0 if "bbox" in o)
    pt_count   = sum(len([p for p,l in zip(o.get("points",[]),o.get("labels",[])) if l==1])
                     for o in frame0)

    return JSONResponse({
        "image": f"data:image/jpeg;base64,{b64}",
        "bbox_count": bbox_count,
        "point_count": pt_count,
        "roles": [o.get("role") for o in frame0],
    })


@app.get("/api/stats")
def get_stats(task: str = Query(...), keyword: str = Query("")):
    task_dir = AP_ROOT / task
    if not task_dir.exists():
        return {"total": 0, "with_bbox": 0, "no_bbox": 0}
    total = with_bbox = no_bbox = 0
    for exp_dir in task_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if keyword and keyword not in exp_dir.name:
            continue
        prompt_dir = exp_dir / "tool_masks" / "prompts"
        if not prompt_dir.exists():
            continue
        for jf in prompt_dir.glob("cam*_rgb.json"):
            total += 1
            data = json.loads(jf.read_text())
            has = any("bbox" in o for o in data.get("objects", []) if o.get("frame_index")==0)
            if has:
                with_bbox += 1
            else:
                no_bbox += 1
    return {"total": total, "with_bbox": with_bbox, "no_bbox": no_bbox}


# ── HTML ─────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BBox Inspector</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #111; color: #eee; height: 100vh; display: flex; }
#sidebar { width: 320px; min-width: 260px; background: #1a1a1a; display: flex; flex-direction: column; border-right: 1px solid #333; }
#sidebar h2 { padding: 12px 16px; font-size: 14px; color: #aaa; border-bottom: 1px solid #333; }
#controls { padding: 12px; display: flex; flex-direction: column; gap: 8px; border-bottom: 1px solid #333; }
#controls select, #controls input { background: #2a2a2a; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 6px 8px; font-size: 13px; width: 100%; }
#stats { padding: 8px 16px; font-size: 12px; color: #888; border-bottom: 1px solid #333; }
#stats span { color: #4ade80; }
#exp-list { flex: 1; overflow-y: auto; }
.exp-item { padding: 8px 16px; cursor: pointer; font-size: 12px; border-bottom: 1px solid #222; display: flex; align-items: center; gap: 6px; }
.exp-item:hover { background: #2a2a2a; }
.exp-item.active { background: #1e3a5f; }
.dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot.ok { background: #4ade80; }
.dot.miss { background: #f87171; }
#main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
#toolbar { padding: 10px 16px; background: #1a1a1a; border-bottom: 1px solid #333; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
#toolbar select { background: #2a2a2a; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 4px 8px; font-size: 13px; }
#info { font-size: 12px; color: #888; margin-left: auto; }
#canvas-wrap { flex: 1; overflow: auto; display: flex; align-items: center; justify-content: center; padding: 16px; }
#canvas-wrap img { max-width: 100%; border-radius: 6px; border: 1px solid #333; }
#status { padding: 8px 16px; font-size: 12px; color: #888; background: #1a1a1a; border-top: 1px solid #333; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin-right: 4px; }
.badge.green { background: #166534; color: #4ade80; }
.badge.red   { background: #7f1d1d; color: #f87171; }
.badge.blue  { background: #1e3a5f; color: #60a5fa; }
#nav { display: flex; gap: 6px; align-items: center; }
#nav button { background: #333; color: #eee; border: none; border-radius: 4px; padding: 4px 10px; cursor: pointer; font-size: 12px; }
#nav button:hover { background: #444; }
#nav span { font-size: 12px; color: #888; }
</style>
</head>
<body>
<div id="sidebar">
  <h2>BBox Inspector</h2>
  <div id="controls">
    <select id="task-sel" onchange="onTaskChange()"><option value="">-- 选择 task --</option></select>
    <input id="kw-input" placeholder="keyword 过滤（如 redrubberspatula）" oninput="onKwChange()">
  </div>
  <div id="stats">加载中…</div>
  <div id="exp-list"></div>
</div>
<div id="main">
  <div id="toolbar">
    <div id="nav">
      <button onclick="prevExp()">◀</button>
      <span id="nav-pos">-/-</span>
      <button onclick="nextExp()">▶</button>
    </div>
    <select id="cam-sel" onchange="loadImage()"></select>
    <div id="info"></div>
  </div>
  <div id="canvas-wrap"><img id="img" src="" alt="选择一个实验"></div>
  <div id="status">就绪</div>
</div>

<script>
let exps = [], curExpIdx = -1, curTask = "", curKw = "";

async function init() {
  const r = await fetch("/api/tasks");
  const tasks = await r.json();
  const sel = document.getElementById("task-sel");
  tasks.forEach(t => { const o = document.createElement("option"); o.value = o.text = t; sel.appendChild(o); });
}

async function onTaskChange() {
  curTask = document.getElementById("task-sel").value;
  curKw   = document.getElementById("kw-input").value.trim();
  if (!curTask) return;
  await loadExps();
}

async function onKwChange() {
  curKw = document.getElementById("kw-input").value.trim();
  if (!curTask) return;
  await loadExps();
}

async function loadExps() {
  document.getElementById("exp-list").innerHTML = "<div style='padding:12px;color:#666'>加载中…</div>";
  document.getElementById("stats").innerHTML = "加载中…";

  const [expR, statR] = await Promise.all([
    fetch(`/api/exps?task=${encodeURIComponent(curTask)}&keyword=${encodeURIComponent(curKw)}`),
    fetch(`/api/stats?task=${encodeURIComponent(curTask)}&keyword=${encodeURIComponent(curKw)}`)
  ]);
  exps = await expR.json();
  const stats = await statR.json();

  document.getElementById("stats").innerHTML =
    `共 ${exps.length} 个实验 &nbsp;|&nbsp; ` +
    `<span>${stats.with_bbox}</span> 有bbox / ${stats.no_bbox} 无bbox (cam维度)`;

  const list = document.getElementById("exp-list");
  list.innerHTML = "";
  exps.forEach((e, i) => {
    const div = document.createElement("div");
    div.className = "exp-item";
    div.innerHTML = `<div class="dot ${e.has_bbox ? 'ok' : 'miss'}"></div><span>${e.exp}</span>`;
    div.onclick = () => selectExp(i);
    list.appendChild(div);
  });

  if (exps.length > 0) selectExp(0);
}

function selectExp(idx) {
  curExpIdx = idx;
  document.querySelectorAll(".exp-item").forEach((el, i) => el.classList.toggle("active", i === idx));
  document.querySelectorAll(".exp-item")[idx]?.scrollIntoView({block:"nearest"});

  const e = exps[idx];
  document.getElementById("nav-pos").textContent = `${idx+1}/${exps.length}`;

  const camSel = document.getElementById("cam-sel");
  camSel.innerHTML = "";
  e.cams.forEach(c => { const o = document.createElement("option"); o.value = o.text = c; camSel.appendChild(o); });

  loadImage();
}

async function loadImage() {
  if (curExpIdx < 0) return;
  const e    = exps[curExpIdx];
  const cam  = document.getElementById("cam-sel").value;
  document.getElementById("status").textContent = `加载 ${e.exp} / ${cam} …`;
  const r = await fetch(`/api/image?task=${encodeURIComponent(curTask)}&exp=${encodeURIComponent(e.exp)}&cam=${encodeURIComponent(cam)}`);
  if (!r.ok) { document.getElementById("status").textContent = `错误: ${r.status}`; return; }
  const d = await r.json();
  document.getElementById("img").src = d.image;
  document.getElementById("info").innerHTML =
    `<span class="badge green">bbox: ${d.bbox_count}</span>` +
    `<span class="badge blue">正点: ${d.point_count}</span>` +
    d.roles.map(r => `<span class="badge">${r}</span>`).join("");
  document.getElementById("status").textContent = `${e.exp} / ${cam}  — 绿框=bbox  绿点=正点  红点=负点`;
}

function prevExp() { if (curExpIdx > 0) selectExp(curExpIdx - 1); }
function nextExp() { if (curExpIdx < exps.length - 1) selectExp(curExpIdx + 1); }

document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft")  prevExp();
  if (e.key === "ArrowRight") nextExp();
});

init();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089, log_level="warning")
