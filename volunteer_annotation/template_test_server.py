"""
模板自动标注可视化测试服务器 (port 8088)

启动：
  conda run -n hocap-annotation python volunteer_annotation/template_test_server.py

访问：
  http://localhost:8088
"""
import sys, json, cv2, base64, re
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

ROOT      = Path(__file__).resolve().parent
CLOUD_DIR = ROOT / "cloud"
SAM2_DIR  = ROOT.parents[2] / "mesh_reconstruction" / "sam2"
sys.path.insert(0, str(CLOUD_DIR))
sys.path.insert(0, str(SAM2_DIR))
sys.path.insert(0, str(ROOT))

from decoder import Sam2CpuDecoder
from template_feature_match import TemplateFeatureStore, predict_from_store, load_embed
from template_auto_annotate import ColorMatcher, MODEL_DIR, sam2_verify_refine as _sam2_vr, EXCLUSIVE_ROLES

BUNDLE       = Path("/data/robotool/_va_bundle_v2")
PROMPTS      = Path("/data/robotool/_va_bundle_v2_prompts")
AUTO_PROMPTS = Path("/data/robotool/_va_bundle_v2_auto_prompts")

# 全局缓存
_decoder: Optional[Sam2CpuDecoder] = None
_stores:  dict[str, TemplateFeatureStore] = {}
_matchers: dict[str, dict[str, ColorMatcher]] = {}

ROLE_COLORS = {
    "primary_tool":    (0,   200,  80),   # 绿
    "manipulated_object": (30, 160, 255), # 橙
    "auxiliary_tool":  (220, 100, 220),   # 紫
}

app = FastAPI(title="Template Annotation Tester")


def get_decoder() -> Sam2CpuDecoder:
    global _decoder
    if _decoder is None:
        _decoder = Sam2CpuDecoder()
    return _decoder


def get_store(task: str, template_exp: str, keyword: str,
              template_task: str | None = None) -> TemplateFeatureStore:
    key = f"{task}|{keyword}"
    if key not in _stores:
        store = TemplateFeatureStore()
        # Cross-task models store the template in a different task directory
        store.add_from_prompts(template_task or task, template_exp)
        _stores[key] = store
    return _stores[key]


def get_matchers(task: str, keyword: str) -> dict[str, ColorMatcher]:
    key = f"{task}|{keyword}"
    if key not in _matchers:
        task_slug = task.replace("/", "_")
        model_dir = MODEL_DIR / task_slug
        meta_path = model_dir / f"{keyword}.meta.json"
        if not meta_path.exists():
            return {}
        meta = json.loads(meta_path.read_text())
        result = {}
        for role, rm in meta.get("roles", {}).items():
            npz = model_dir / rm["npz"]
            if npz.exists():
                result[role] = ColorMatcher(npz, rm)
        _matchers[key] = result
    return _matchers[key]


def exp_keyword(exp_name: str) -> str:
    s = re.sub(r"^\d{8}_", "", exp_name)
    s = re.sub(r"_\d+$", "", s)
    s = re.sub(r"_from_.+", "", s)
    return s


def img_to_b64(img: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def draw_points(img: np.ndarray, points: list, labels: list,
                role: str, scale: float = 1.0) -> np.ndarray:
    out = img.copy()
    color_pos = ROLE_COLORS.get(role, (0, 200, 80))
    color_neg = (80, 80, 220)
    r = max(6, int(10 * scale))
    thick = max(2, int(3 * scale))
    for (x, y), lb in zip(points, labels):
        ix, iy = int(x), int(y)
        c = color_pos if lb == 1 else color_neg
        cv2.circle(out, (ix, iy), r, c, -1)
        cv2.circle(out, (ix, iy), r + 2, (255, 255, 255), thick)
    return out


def draw_mask_overlay(img: np.ndarray, mask: np.ndarray,
                      role: str, alpha: float = 0.45) -> np.ndarray:
    out = img.copy().astype(np.float32)
    color = np.array(ROLE_COLORS.get(role, (0, 200, 80)), dtype=np.float32)
    m = mask.astype(bool)
    out[m] = out[m] * (1 - alpha) + color[::-1] * alpha  # BGR
    return out.astype(np.uint8)


def load_prompts_for_cam(task: str, exp: str, cam: str) -> list[dict]:
    """Load human + auto prompts merged; human annotations take priority."""
    def _read(pf: Path) -> list[dict]:
        if not pf.exists():
            return []
        try:
            return json.loads(pf.read_text()).get("objects", [])
        except Exception:
            return []

    human_objs = _read(PROMPTS / task / exp / "tool_masks" / "prompts" / f"{cam}.json")
    auto_objs  = _read(AUTO_PROMPTS / task / exp / "tool_masks" / "prompts" / f"{cam}.json")

    human_rf = {(o["role"], o.get("frame_index", 0)) for o in human_objs if o.get("points")}
    merged = list(human_objs)
    for obj in auto_objs:
        if (obj["role"], obj.get("frame_index", 0)) not in human_rf:
            merged.append(obj)
    return merged


def get_manifest(task: str, exp: str) -> dict:
    mf = BUNDLE / task / exp / "_manifest.json"
    if not mf.exists():
        return {}
    return json.loads(mf.read_text())


# ─── API ──────────────────────────────────────────────────────────────────────

@app.get("/api/models")
def api_models():
    """返回已有颜色模型列表。"""
    result = []
    if not MODEL_DIR.exists():
        return []
    for task_dir in sorted(MODEL_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name.replace("_", "/", 1)  # videos_0106_spoon → videos_0106/spoon
        # 更精确：找第一个 /
        # task_dir.name 格式：videos_0106_spoon_scoop_nuts
        # task 格式：videos_0106/spoon_scoop_nuts
        parts = task_dir.name.split("_")
        # videos + date = 2 parts, rest is task name
        task = parts[0] + "_" + parts[1] + "/" + "_".join(parts[2:])
        for meta_file in sorted(task_dir.glob("*.meta.json")):
            meta = json.loads(meta_file.read_text())
            result.append({
                "task": task,
                "keyword": meta["keyword"],
                "template_exp": meta["template_exp"],
                "template_task": meta.get("template_task"),
                "roles": list(meta.get("roles", {}).keys()),
            })
    return result


@app.get("/api/exps/{task:path}")
def api_exps(task: str, keyword: str = ""):
    """返回 task 下有 embed 的实验列表（可按 keyword 过滤）。"""
    task_dir = BUNDLE / task
    if not task_dir.exists():
        return []
    exps = []
    for e in sorted(task_dir.iterdir()):
        if not e.is_dir():
            continue
        if not (e / "_manifest.json").exists():
            continue
        if keyword and keyword not in exp_keyword(e.name):
            continue
        exps.append(e.name)
    return exps


@app.get("/api/cameras/{task:path}")
def api_cameras(task: str, exp: str):
    """返回实验的相机列表和关键帧。"""
    mf = get_manifest(task, exp)
    if not mf:
        return []
    return [{"camera": c["camera"], "keyframes": c["keyframes"]}
            for c in mf.get("cameras", [])]


@app.get("/api/template_image")
def api_template_image(task: str, exp: str, camera: str, frame: int,
                        show_mask: bool = True):
    """返回 template 图像（带人工标注点和 SAM2 mask）。"""
    img_path = BUNDLE / task / exp / f"{camera}.kf{frame}.jpg"
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    img = cv2.imread(str(img_path))

    objects = load_prompts_for_cam(task, exp, camera)
    frame_objs = [o for o in objects
                  if o.get("frame_index") == frame and o.get("points")]

    # 逐角色画 mask + 点
    for obj in frame_objs:
        role = obj["role"]
        pts = obj["points"]
        labels = obj.get("labels", [1] * len(pts))
        if show_mask:
            embed_path = BUNDLE / task / exp / f"{camera}.kf{frame}.embed.npz"
            if embed_path.exists():
                try:
                    mask, score = get_decoder().infer(embed_path, pts, labels)
                    img = draw_mask_overlay(img, mask, role)
                except Exception:
                    pass
        img = draw_points(img, pts, labels, role)

    # 标注来源标签
    h, w = img.shape[:2]
    if frame_objs:
        n_human = sum(1 for o in frame_objs if not o.get("auto_generated"))
        n_auto  = sum(1 for o in frame_objs if o.get("auto_generated"))
        label = f"Human:{n_human}  Auto:{n_auto}"
        cv2.putText(img, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "No annotations", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 220), 2)

    return {"image": img_to_b64(img), "annotations": frame_objs}


@app.get("/api/predict")
def api_predict(task: str, exp: str, camera: str, frame: int,
                keyword: str, template_exp: str,
                template_task: str = "", min_sim: float = 0.65, show_mask: bool = True):
    """对 target exp 运行自动预测，返回带预测点和 mask 的图像。"""
    img_path = BUNDLE / task / exp / f"{camera}.kf{frame}.jpg"
    embed_path = BUNDLE / task / exp / f"{camera}.kf{frame}.embed.npz"
    if not img_path.exists():
        raise HTTPException(404, "Image not found")

    img = cv2.imread(str(img_path))
    embed = load_embed(embed_path) if embed_path.exists() else None
    store = get_store(task, template_exp, keyword, template_task=template_task or None)
    matchers = get_matchers(task, keyword)

    # ── 第一轮：预测所有 role 位置 ──
    role_results: dict[str, dict | None] = {}
    for role in store._data:
        matcher = matchers.get(role)
        used_center, conf, method, verify_status = None, 0.0, "", ""

        def _vr(cx, cy, _m=matcher):
            return _sam2_vr(embed_path, img, cx, cy, _m, get_decoder())

        if embed is not None:
            raw_pt, feat_conf = predict_from_store(embed, store, role, min_sim=min_sim, cam=camera)
            if raw_pt is not None:
                refined, status, mask_negs, _ = _vr(*raw_pt)
                if refined is not None:
                    used_center, conf, method, verify_status = refined, feat_conf, "feature", status

        if used_center is None and matcher is not None:
            raw_color, color_conf, _ = matcher.find_object(img, 0.85)
            if raw_color is not None and color_conf >= 0.45:
                refined_c, status_c, mask_negs_c, _ = _vr(*raw_color)
                if refined_c is not None:
                    used_center, conf, method, verify_status = refined_c, color_conf, "color", status_c

        # 若 live 预测失败，从已存的 auto prompts 中读取作为 fallback
        if used_center is None:
            existing = load_prompts_for_cam(task, exp, camera)
            for obj in existing:
                if obj.get("role") == role and obj.get("frame_index") == frame:
                    pts_e = obj.get("points", [])
                    if pts_e and obj.get("labels", [1])[0] == 1:
                        used_center = (float(pts_e[0][0]), float(pts_e[0][1]))
                        conf = obj.get("confidence", 0.5)
                        method = obj.get("method", "existing") + "(file)"
                        verify_status = "from_file"
                        break

        role_results[role] = {"center": used_center, "conf": conf,
                              "method": method, "verify": verify_status} if used_center is not None else None

    # ── 第二轮：生成带互斥负样本的 prompt 并可视化 ──
    predictions = []
    for role, res in role_results.items():
        if res is None:
            predictions.append({"role": role, "found": False})
            continue

        cx, cy = res["center"]
        matcher = matchers.get(role)
        sam2_score = 0.0

        if matcher is not None:
            neg_pts = matcher.background_points(img, cx, cy, 2)
        else:
            h, w = img.shape[:2]
            neg_pts = [(w * 0.05, h * 0.05), (w * 0.95, h * 0.05)]

        # 互斥 role 的位置作为额外负样本
        for excl_role in EXCLUSIVE_ROLES.get(role, set()):
            excl_res = role_results.get(excl_role)
            if excl_res is not None:
                ecx, ecy = excl_res["center"]
                neg_pts.append((float(ecx), float(ecy)))

        pts  = [[round(cx, 1), round(cy, 1)]] + [[round(x,1),round(y,1)] for x,y in neg_pts]
        labs = [1] + [0] * len(neg_pts)

        if show_mask and embed_path.exists():
            try:
                mask_full, score_full = get_decoder().infer(embed_path, pts, labs)
                img = draw_mask_overlay(img, mask_full, role)
                sam2_score = score_full
            except Exception:
                pass

        img = draw_points(img, pts, labs, role)
        predictions.append({
            "role": role, "found": True,
            "cx": round(float(cx), 1), "cy": round(float(cy), 1),
            "confidence": round(float(res["conf"]), 3),
            "sam2_score": round(float(sam2_score), 3),
            "method": res["method"],
            "verify": res["verify"],
        })

    h, w = img.shape[:2]
    label = "  ".join(
        f"{p['role'].split('_')[0]}:{p.get('confidence',0):.2f}({p.get('method','?')[0]})"
        if p["found"] else f"{p['role'].split('_')[0]}:NOT FOUND"
        for p in predictions
    )
    cv2.putText(img, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return {"image": img_to_b64(img), "predictions": predictions}


# ─── 前端 ─────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Template Auto-Annotation Tester</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#111;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
header{padding:10px 16px;background:#1a1a2e;border-bottom:1px solid #333;display:flex;align-items:center;gap:16px;flex-shrink:0}
header h1{font-size:1rem;font-weight:600;color:#7eb8f7}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
select,button{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:4px 8px;font-size:0.82rem}
select:focus,button:focus{outline:2px solid #7eb8f7;outline-offset:1px}
button{cursor:pointer;background:#2a3a5e;border-color:#4a6a9e}
button:hover{background:#3a4a6e}
button.active{background:#1a5a2e;border-color:#3a9a5e}
label{font-size:0.78rem;color:#aaa}
.main{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#333;overflow:hidden}
.panel{background:#111;display:flex;flex-direction:column;overflow:hidden}
.panel-header{padding:6px 12px;background:#181828;border-bottom:1px solid #2a2a4a;display:flex;gap:8px;align-items:center;flex-shrink:0}
.panel-title{font-size:0.8rem;font-weight:600;color:#aac}
.panel-body{flex:1;overflow:hidden;display:flex;flex-direction:column;align-items:center;justify-content:center;position:relative}
.img-wrap{width:100%;height:100%;display:flex;align-items:center;justify-content:center;overflow:hidden}
img.preview{max-width:100%;max-height:100%;object-fit:contain;display:block}
.info-bar{padding:5px 12px;background:#141420;border-top:1px solid #2a2a4a;font-size:0.75rem;color:#8a8aaa;flex-shrink:0;min-height:28px}
.badge{display:inline-block;padding:2px 7px;border-radius:10px;font-size:0.72rem;font-weight:600;margin-right:4px}
.badge.feature{background:#1a4a2e;color:#6adf8e}
.badge.color{background:#3a2a10;color:#f0a830}
.badge.notfound{background:#3a1a1a;color:#f08080}
.badge.human{background:#1a2a4a;color:#80b0f0}
.spinner{width:36px;height:36px;border:4px solid #333;border-top:4px solid #7eb8f7;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.cam-grid{display:flex;gap:4px;flex-wrap:wrap;align-items:center}
.cam-btn{padding:3px 7px;font-size:0.75rem;border-radius:3px;cursor:pointer;background:#1a1a2e;border:1px solid #333;color:#aaa}
.cam-btn.sel{background:#2a3a6e;border-color:#6a8abe;color:#dde}
.row{display:flex;gap:8px;align-items:center}
</style>
</head>
<body>

<header>
  <h1>🎯 Template Auto-Annotation Tester</h1>
  <div class="controls">
    <label>模型</label>
    <select id="modelSel" onchange="onModelChange()"><option value="">— 加载中 —</option></select>
    <label>Template</label>
    <select id="tplExpSel" onchange="onTplExpChange()"><option value="">—</option></select>
    <label>min_sim</label>
    <select id="minSim">
      <option value="0.5">0.50</option>
      <option value="0.6">0.60</option>
      <option value="0.65" selected>0.65</option>
      <option value="0.7">0.70</option>
      <option value="0.75">0.75</option>
      <option value="0.8">0.80</option>
    </select>
    <label><input type="checkbox" id="showMask" checked> SAM2 mask</label>
  </div>
</header>

<div class="main">
  <!-- 左：Template -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">📌 Template（人工标注）</span>
      <div class="cam-grid" id="tplCamGrid"></div>
      <select id="tplFrameSel" onchange="loadTemplate()" style="margin-left:8px"></select>
    </div>
    <div class="panel-body" id="tplBody">
      <div class="spinner"></div>
    </div>
    <div class="info-bar" id="tplInfo">选择模型后显示 template</div>
  </div>

  <!-- 右：Target -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">🤖 Target（自动预测）</span>
      <div class="row">
        <label>实验</label>
        <select id="tgtExpSel" onchange="onTgtExpChange()"><option value="">— 选择实验 —</option></select>
      </div>
      <div class="cam-grid" id="tgtCamGrid"></div>
      <select id="tgtFrameSel" onchange="runPredict()" style="margin-left:8px"></select>
    </div>
    <div class="panel-body" id="tgtBody">
      <div style="color:#555;font-size:0.9rem">选择 target 实验后自动预测</div>
    </div>
    <div class="info-bar" id="tgtInfo">—</div>
  </div>
</div>

<script>
let models = [], curModel = null;
let tplCams = [], tgtCams = [];
let selTplCam = '', selTgtCam = '';

// 自动检测 base path（独立运行 vs 挂载到 /tester）
// 路径以 / 结尾 → 当前目录即 BASE；否则去掉最后一段
const _p = location.pathname;
const BASE = _p.endsWith('/') ? _p.slice(0, -1) : _p.replace(/\/[^/]*$/, '');
const API = (p) => `${BASE}${p}`;

function fmtExp(exp){
  const m=exp.match(/^(\d{8}_)(.+?)(_\d+)$/);
  if(!m) return exp.slice(9,50);
  const num=m[3].slice(1);
  let name=m[2].replace(/_in_.+$/,'').replace(/_from_.+$/,'');
  if(name.length>26) name=name.slice(0,24)+'..';
  return `#${num} ${name}`;
}

// ── 初始化 ──
async function init() {
  const r = await fetch(API('/api/models'));
  models = await r.json();
  const sel = document.getElementById('modelSel');
  sel.innerHTML = '<option value="">— 选择模型 —</option>' +
    models.map((m,i) => `<option value="${i}">${m.task.split('/')[1]} › ${m.keyword}</option>`).join('');
}

// ── 模型选择 ──
async function onModelChange() {
  const idx = document.getElementById('modelSel').value;
  if (idx === '') return;
  curModel = models[parseInt(idx)];

  // 填充 template exp（固定）
  const tplSel = document.getElementById('tplExpSel');
  tplSel.innerHTML = `<option value="${curModel.template_exp}">${fmtExp(curModel.template_exp)}</option>`;

  // 加载 template 相机
  await loadTplCameras();

  // 加载 target exps
  const r = await fetch(API(`/api/exps/${curModel.task}?keyword=${curModel.keyword}`));
  const exps = await r.json();
  const tgtSel = document.getElementById('tgtExpSel');
  tgtSel.innerHTML = '<option value="">— 选择实验 —</option>' +
    exps.filter(e => e !== curModel.template_exp)
        .map(e => `<option value="${e}">${fmtExp(e)}</option>`).join('');
}

async function loadTplCameras() {
  const exp = curModel.template_exp;
  const r = await fetch(API(`/api/cameras/${curModel.task}?exp=${encodeURIComponent(exp)}`));
  tplCams = await r.json();
  if (!tplCams.length) return;
  selTplCam = tplCams[0].camera;
  renderCamGrid('tplCamGrid', tplCams, selTplCam, (cam) => {
    selTplCam = cam; loadTemplate();
  });
  populateFrames('tplFrameSel', tplCams.find(c=>c.camera===selTplCam)?.keyframes || [], loadTemplate);
  await loadTemplate();
}

async function onTplExpChange() {
  await loadTplCameras();
}

async function onTgtExpChange() {
  const exp = document.getElementById('tgtExpSel').value;
  if (!exp) return;
  const r = await fetch(API(`/api/cameras/${curModel.task}?exp=${encodeURIComponent(exp)}`));
  tgtCams = await r.json();
  if (!tgtCams.length) return;
  selTgtCam = tgtCams[0].camera;
  renderCamGrid('tgtCamGrid', tgtCams, selTgtCam, (cam) => {
    selTgtCam = cam; runPredict();
  });
  populateFrames('tgtFrameSel', tgtCams.find(c=>c.camera===selTgtCam)?.keyframes || [], runPredict);
  await runPredict();
}

// ── 相机按钮 ──
function renderCamGrid(containerId, cams, selected, onClick) {
  const el = document.getElementById(containerId);
  el.innerHTML = cams.map(c =>
    `<button class="cam-btn${c.camera===selected?' sel':''}" data-cam="${c.camera}"
     >${c.camera.replace('_rgb','')}</button>`
  ).join('');
  el.querySelectorAll('.cam-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      el.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('sel'));
      btn.classList.add('sel');
      onClick(btn.dataset.cam);
    });
  });
}

function populateFrames(selId, keyframes, onChange) {
  const sel = document.getElementById(selId);
  sel.innerHTML = keyframes.map(f => `<option value="${f}">f${f}</option>`).join('');
  sel.onchange = onChange;
}

// ── Template 加载 ──
async function loadTemplate() {
  if (!curModel) return;
  const cam = selTplCam;
  const frame = document.getElementById('tplFrameSel').value;
  const mask = document.getElementById('showMask').checked;
  if (!cam || !frame) return;

  setLoading('tplBody');
  document.getElementById('tplInfo').textContent = '加载中…';

  const url = API(`/api/template_image?task=${encodeURIComponent(curModel.task)}`+
    `&exp=${encodeURIComponent(curModel.template_exp)}&camera=${cam}&frame=${frame}&show_mask=${mask}`);
  const r = await fetch(url);
  const data = await r.json();

  setImage('tplBody', data.image);

  const anns = (data.annotations||[]).filter(o=>o.frame_index==frame);
  const parts = anns.map(o => {
    const isAuto = o.auto_generated;
    const tag = isAuto ? `<span class="badge color">auto</span>` : `<span class="badge human">human</span>`;
    return `${tag} <b>${o.role.split('_')[0]}</b> ${(o.points||[]).length}pts`;
  });
  document.getElementById('tplInfo').innerHTML = parts.join(' &nbsp;|&nbsp; ') || '该帧无标注';
}

// ── 预测 ──
async function runPredict() {
  if (!curModel) return;
  const exp = document.getElementById('tgtExpSel').value;
  if (!exp) return;
  const cam = selTgtCam;
  const frame = document.getElementById('tgtFrameSel').value;
  const mask = document.getElementById('showMask').checked;
  const minSim = document.getElementById('minSim').value;
  if (!cam || !frame) return;

  setLoading('tgtBody');
  document.getElementById('tgtInfo').textContent = '预测中…';

  const tmplTask = curModel.template_task ? `&template_task=${encodeURIComponent(curModel.template_task)}` : '';
  const url = API(`/api/predict?task=${encodeURIComponent(curModel.task)}`+
    `&exp=${encodeURIComponent(exp)}&camera=${cam}&frame=${frame}`+
    `&keyword=${encodeURIComponent(curModel.keyword)}&template_exp=${encodeURIComponent(curModel.template_exp)}`+
    `${tmplTask}&min_sim=${minSim}&show_mask=${mask}`);
  const r = await fetch(url);
  const data = await r.json();

  setImage('tgtBody', data.image);

  const parts = (data.predictions||[]).map(p => {
    if (!p.found) return `<span class="badge notfound">✗ ${p.role.split('_')[0]}</span>`;
    const cls = p.method === 'feature' ? 'feature' : 'color';
    const icon = p.method === 'feature' ? '⚡' : '🎨';
    const vfx = p.verify && p.verify.startsWith('shape_mismatch')
      ? `<span style="color:#f88">⬡${p.verify}</span>` : '';
    return `<span class="badge ${cls}">${icon} ${p.role.split('_')[0]}</span>`+
           ` conf:<b>${p.confidence}</b> sam2:<b>${p.sam2_score}</b>${vfx}`;
  });
  document.getElementById('tgtInfo').innerHTML = parts.join(' &nbsp;|&nbsp; ') || '无预测结果';
}

// ── 辅助 ──
function setLoading(id) {
  document.getElementById(id).innerHTML = '<div class="spinner"></div>';
}
function setImage(id, b64) {
  document.getElementById(id).innerHTML =
    `<div class="img-wrap"><img class="preview" src="data:image/jpeg;base64,${b64}"></div>`;
}

// 切换 checkbox 时刷新
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('showMask').onchange = () => { loadTemplate(); runPredict(); };
  document.getElementById('minSim').onchange = () => runPredict();
  init();
});
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def root():
    return HTML


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--host", default="127.0.0.1")
    a = parser.parse_args()
    print(f"Starting template annotation test server on http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
