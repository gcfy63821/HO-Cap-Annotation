"""
Step 2: 用颜色模型自动生成其他实验的点 prompt

对与 template 关键词一致的其他实验（同 task，exp 名包含相同 tool+object 关键词）：
  1. 加载颜色模型（由 template_color_extract.py 生成）
  2. 对每个相机 × 关键帧：颜色匹配 → 找物体质心
  3. 生成点 prompt（正样本：质心；负样本：背景区域）
  4. 写入 prompt JSON（仅在原 JSON 缺失对应 role+frame 时才补充）

Usage:
  python template_auto_annotate.py \
    --task videos_0106/spoon_scoop_nuts \
    --keyword plasticspoon_scoop_almond_nuts \
    [--overwrite]       # 覆盖已有标注
    [--dry-run]         # 只打印不写入
    [--confidence 0.4]  # 颜色匹配置信度阈值

颜色模型由 template_color_extract.py 生成，放在:
  volunteer_annotation/color_models/{task_slug}/{keyword}.{role}.npz
"""
import sys, json, argparse, cv2, re, sqlite3, datetime
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[0]
CLOUD_DIR = ROOT / "cloud"
SAM2_DIR = ROOT.parents[2] / "mesh_reconstruction" / "sam2"
sys.path.insert(0, str(CLOUD_DIR))
sys.path.insert(0, str(SAM2_DIR))

BUNDLE       = Path("/data/robotool/_va_bundle_v2")
PROMPTS      = Path("/data/robotool/_va_bundle_v2_prompts")       # human only (read-only)
AUTO_PROMPTS = Path("/data/robotool/_va_bundle_v2_auto_prompts")  # auto write target
DB           = Path("/data/robotool/_va_bundle_v2/tasks.db")
MODEL_DIR = ROOT / "color_models"

from template_feature_match import (
    TemplateFeatureStore, predict_from_store, load_embed
)
from template_multiview_vote import (
    load_calibration, cam_name_to_id, vote_frame_predictions
)

# 匹配参数
MAX_CLUSTER_RATIO = 0.12   # 候选簇面积 < 图像 12%（否则是背景）
MIN_CLUSTER_PX = 30
N_NEG_POINTS = 2           # 每个预测结果附加的背景负样本点数

# 互斥 role 对：key role 的标注点会作为 value 集合里每个 role 的负样本
# auxiliary_tool 和 {primary_tool, manipulated_object} 互斥，反之亦然
EXCLUSIVE_ROLES: dict[str, set[str]] = {
    "auxiliary_tool":    {"primary_tool", "manipulated_object"},
    "primary_tool":      {"auxiliary_tool"},
    "manipulated_object": {"auxiliary_tool"},
}

# SAM2 验证参数
SAM2_SCORE_MIN   = 0.70    # mask 质量分阈值
SAM2_MASK_MIN_PX  = 80     # mask 最少像素
SAM2_MASK_MAX_RT  = 0.15   # mask 最大面积比（>15%=误匹配背景）
COLOR_MASK_MAX    = 1.05   # mask 区域平均颜色得分上限（超过=颜色不匹配）
MASK_BAD_COLOR_THRESH = 0.8   # mask 内颜色得分高于此值的像素视为"错误区域"
MASK_BAD_MIN_PX   = 200    # 错误区域至少多少像素才加负样本点


# ─── 关键词 ────────────────────────────────────────────────────────────────────

def exp_keyword(exp_name: str) -> str:
    s = re.sub(r"^\d{8}_", "", exp_name)
    s = re.sub(r"_\d+$", "", s)
    s = re.sub(r"_from_.+", "", s)
    return s


# ─── 颜色模型加载 ──────────────────────────────────────────────────────────────

class ColorMatcher:
    """
    从 NPZ 文件加载前景/背景颜色分布 + 形状统计，用于在新图像中定位和验证目标。
    """
    def __init__(self, npz_path: Path, meta: dict):
        z = np.load(str(npz_path))
        self.fg_lab = z["fg_lab"].astype(np.float32)   # (N, 3)
        self.fg_hsv = z["fg_hsv"].astype(np.float32)   # (N, 3)
        self.bg_lab = z["bg_lab"].astype(np.float32) if len(z["bg_lab"]) > 0 \
                      else np.zeros((0, 3), np.float32)
        self.mean_cx = meta.get("mean_cx", 0.5)
        self.mean_cy = meta.get("mean_cy", 0.5)

        # 形状统计（可选，旧版 NPZ 没有则禁用形状验证）
        self.shape_stats = None
        if "shape_solidity_mean" in z:
            self.shape_stats = {
                "solidity_mean":    float(z["shape_solidity_mean"][0]),
                "solidity_std":     float(z["shape_solidity_std"][0]),
                "compactness_mean": float(z["shape_compactness_mean"][0]),
                "compactness_std":  float(z["shape_compactness_std"][0]),
                "aspect_mean":      float(z["shape_aspect_mean"][0]),
                "aspect_std":       float(z["shape_aspect_std"][0]),
                "extent_mean":      float(z["shape_extent_mean"][0]),
                "extent_std":       float(z["shape_extent_std"][0]),
                "log_hu_mean":      z["shape_log_hu_mean"].astype(np.float32),
                "log_hu_std":       z["shape_log_hu_std"].astype(np.float32),
            }
        self._build_stats()

    def _build_stats(self):
        """计算颜色统计量（HSV + LAB 双通道）。"""
        # ── HSV 统计（色相循环处理）
        H = self.fg_hsv[:, 0]
        theta = H / 180.0 * np.pi * 2
        sin_m, cos_m = np.sin(theta).mean(), np.cos(theta).mean()
        r = max(np.sqrt(sin_m**2 + cos_m**2), 1e-6)
        self.h_mean = float(np.arctan2(sin_m, cos_m) / (np.pi * 2) * 180 % 180)
        self.h_std = float(max(10, np.sqrt(-2 * np.log(r)) / (np.pi * 2) * 180))
        SV = self.fg_hsv[:, 1:]
        self.sv_mean = SV.mean(axis=0)
        self.sv_std  = np.maximum(SV.std(axis=0), 10)

        # ── LAB 统计（补充通道 a,b 的精度）
        self.lab_mean = self.fg_lab.mean(axis=0)
        self.lab_std  = np.maximum(self.fg_lab.std(axis=0), [8, 5, 5])

        # ── 背景颜色（用于排除）
        if len(self.bg_lab) > 0:
            self.bg_lab_mean = self.bg_lab.mean(axis=0)
            self.bg_lab_std  = np.maximum(self.bg_lab.std(axis=0), [8, 5, 5])
        else:
            self.bg_lab_mean = None

    def score_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        返回 score 图 (H×W)，值越小 = 颜色越匹配。
        """
        H, W = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # HSV score
        dh = np.abs(hsv[:, :, 0] - self.h_mean)
        dh = np.minimum(dh, 180 - dh)
        h_sc = dh / (self.h_std * 1.3)
        s_sc = np.abs(hsv[:, :, 1] - self.sv_mean[0]) / (self.sv_std[0] * 1.3)
        v_sc = np.abs(hsv[:, :, 2] - self.sv_mean[1]) / (self.sv_std[1] * 1.3)

        # 低饱和度时 hue 不可靠
        low_sat = hsv[:, :, 1] < 35
        hsv_sc = np.where(low_sat,
                          np.maximum(s_sc, v_sc),
                          np.maximum(h_sc, np.maximum(s_sc, v_sc)))

        # LAB a,b score（弥补 HSV 对白/灰物体的不足）
        ab_sc = np.maximum(
            np.abs(lab[:, :, 1] - self.lab_mean[1]) / (self.lab_std[1] * 1.3),
            np.abs(lab[:, :, 2] - self.lab_mean[2]) / (self.lab_std[2] * 1.3),
        )

        # 综合：取 HSV 和 LAB_ab 的最小分（并集匹配，任一通道匹配则得分低）
        score = np.minimum(hsv_sc, ab_sc)

        # 背景惩罚：与背景颜色越相似则惩罚越重（提高 score）
        if self.bg_lab_mean is not None:
            bg_d = np.maximum(
                np.abs(lab[:, :, 0] - self.bg_lab_mean[0]) / self.bg_lab_std[0],
                np.maximum(
                    np.abs(lab[:, :, 1] - self.bg_lab_mean[1]) / self.bg_lab_std[1],
                    np.abs(lab[:, :, 2] - self.bg_lab_mean[2]) / self.bg_lab_std[2],
                )
            )
            # 像背景的地方加分（变难匹配）
            bg_penalty = np.clip(1.5 - bg_d, 0, 0.8)
            score = score + bg_penalty

        return score.astype(np.float32)

    def find_object(self, img_bgr: np.ndarray,
                    threshold: float = 1.0) -> tuple[np.ndarray | None, float, np.ndarray]:
        """
        在图像中找目标，返回 (质心[cx,cy], 置信度, debug_mask)。
        置信度 = 最佳簇的平均得分（越低越好，转为 confidence=1-score）
        """
        H, W = img_bgr.shape[:2]
        score = self.score_image(img_bgr)
        mask = (score < threshold).astype(np.uint8) * 255

        # 形态学去噪
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.dilate(mask, k, iterations=1)

        n, labels_img, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return None, 0.0, mask

        max_area = H * W * MAX_CLUSTER_RATIO
        candidates = []
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if MIN_CLUSTER_PX <= a <= max_area:
                # 空间先验：靠近已知均值位置得分更高
                cx_n, cy_n = centroids[i][0] / W, centroids[i][1] / H
                pos_dist = np.sqrt((cx_n - self.mean_cx)**2 + (cy_n - self.mean_cy)**2)
                pos_score = 1.0 - min(pos_dist * 1.5, 0.5)   # 越近 = 分越高
                # 颜色得分：该簇内像素的平均 score（越低越好）
                cluster_pixels = score[labels_img == i]
                color_conf = 1.0 - float(cluster_pixels.mean())
                combined = (color_conf + pos_score) / 2.0
                candidates.append((combined, a, i))

        if not candidates:
            return None, 0.0, mask

        # 取综合分最高的候选
        best_conf, best_area, best_i = max(candidates, key=lambda x: x[0])
        cx, cy = centroids[best_i]
        return np.array([cx, cy]), float(best_conf), mask

    def background_points(self, img_bgr: np.ndarray,
                          obj_cx: float, obj_cy: float,
                          n: int = 2) -> list[tuple[float, float]]:
        """
        在距目标较远的区域找 n 个背景负样本点（图像边缘 / 距目标 >1/4 图像宽度）。
        """
        H, W = img_bgr.shape[:2]
        score = self.score_image(img_bgr)
        # 生成候选背景点：远离目标的区域
        pts = []
        min_dist = W * 0.25

        # 在图像边缘 1/4 区域内随机采样，找最佳背景点
        candidates_bg = []
        for y in range(0, H, 40):
            for x in range(0, W, 40):
                dist = np.sqrt((x - obj_cx)**2 + (y - obj_cy)**2)
                if dist < min_dist:
                    continue
                sc = float(score[y, x])
                candidates_bg.append((sc, x, y))

        # score 越高 = 越不像目标 = 越好的负样本
        candidates_bg.sort(reverse=True)
        for sc, x, y in candidates_bg[:n]:
            pts.append((float(x), float(y)))
        return pts


# ─── SAM2 验证 ────────────────────────────────────────────────────────────────

SHAPE_SIGMA_MAX  = 2.5   # mask 形状特征最大容忍偏差（单位：模板分布的 sigma）
SHAPE_HU_W       = 0.5   # Hu moments 在综合距离中的权重（降低，因为跨视角变化较大）


def _shape_distance(mask: np.ndarray, stats: dict) -> tuple[float, str]:
    """
    计算 mask 形状与模板形状统计的归一化距离。
    返回 (max_sigma_distance, 最差特征名)。
    """
    m = mask.astype(np.uint8)
    area = int(m.sum())
    if area < 50:
        return 99.0, "too_small"

    ys, xs = np.where(m)
    h_bb = int(ys.max() - ys.min() + 1)
    w_bb = int(xs.max() - xs.min() + 1)
    aspect = float(max(h_bb, w_bb) / max(min(h_bb, w_bb), 1))
    extent = float(area / max(h_bb * w_bb, 1))

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 99.0, "no_contour"
    cnt = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    compactness = float(perimeter ** 2 / (4 * np.pi * area)) if area > 0 else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    solidity = float(area / max(hull_area, 1))

    moments = cv2.moments(m)
    hu = cv2.HuMoments(moments).flatten()
    log_hu = (-np.sign(hu) * np.log10(np.abs(hu) + 1e-10)).astype(np.float32)

    # 各特征归一化距离（sigma 倍数）
    dists = {}
    for feat, val in [("solidity", solidity), ("compactness", compactness),
                      ("aspect", aspect), ("extent", extent)]:
        d = abs(val - stats[f"{feat}_mean"]) / max(stats[f"{feat}_std"], 1e-4)
        dists[feat] = d

    hu_d = np.abs(log_hu[:4] - stats["log_hu_mean"]) / np.maximum(stats["log_hu_std"], 1e-4)
    dists["log_hu"] = float(hu_d.mean()) * SHAPE_HU_W

    worst = max(dists, key=dists.get)
    return float(dists[worst]), worst


def sam2_verify_refine(
    embed_path: Path,
    img: np.ndarray | None,
    cx: float, cy: float,
    matcher: "ColorMatcher | None",
    decoder,
) -> tuple[np.ndarray | None, str, list[tuple[float, float]], list[tuple[float, float]]]:
    """
    用 SAM2 decode 验证候选点并精炼坐标。
    返回 (精炼后坐标, 状态说明, 额外负样本点列表, 额外正样本点列表)；
    验证失败返回 (None, 原因, [], [])。

    验证顺序：
      1. SAM2 score ≥ SAM2_SCORE_MIN
      2. mask 面积在合理范围
      3. mask 形状与模板形状匹配（sigma 距离）
      4. mask 区域平均颜色得分匹配
    通过后：
      - 用颜色加权质心替代原始坐标（偏向颜色最匹配的区域）
      - 在 mask 内找颜色不匹配的连通区域，作为额外负样本点
      - 在 mask 内找颜色最好的第二个位置（距质心≥40px），作为额外正样本点
    """
    if decoder is None:
        return np.array([cx, cy]), "unverified", [], []
    try:
        mask, score = decoder.infer(embed_path, [[cx, cy]], [1])
    except Exception:
        return np.array([cx, cy]), "decode_error", [], []

    if score < SAM2_SCORE_MIN:
        return None, f"low_score={score:.2f}", [], []

    H, W = mask.shape
    mask_px = int(mask.sum())
    if mask_px < SAM2_MASK_MIN_PX:
        return None, f"mask_too_small={mask_px}px", [], []
    if mask_px > H * W * SAM2_MASK_MAX_RT:
        return None, f"mask_too_large={mask_px}px", [], []

    # 形状验证（有形状统计时才做）
    if matcher is not None and matcher.shape_stats is not None:
        sigma_d, worst_feat = _shape_distance(mask, matcher.shape_stats)
        if sigma_d > SHAPE_SIGMA_MAX:
            return None, f"shape_mismatch={worst_feat}({sigma_d:.1f}σ)", [], []

    ys, xs = np.where(mask)
    extra_neg_pts: list[tuple[float, float]] = []
    extra_pos_pts: list[tuple[float, float]] = []

    # 颜色验证 + 颜色加权质心 + 错误颜色区域负样本 + 第二正样本点
    if img is not None and matcher is not None:
        color_sc = matcher.score_image(img)
        mask_color = float(color_sc[mask].mean())
        if mask_color > COLOR_MASK_MAX:
            return None, f"color_mismatch={mask_color:.2f}", [], []

        scores_in_mask = color_sc[ys, xs]

        # 颜色加权质心（颜色越匹配权重越高）
        weights = np.maximum(0.0, COLOR_MASK_MAX - scores_in_mask)
        w_sum = weights.sum()
        if w_sum > 1e-6:
            weights /= w_sum
            cx_ref = float((xs * weights).sum())
            cy_ref = float((ys * weights).sum())
        else:
            cx_ref, cy_ref = float(xs.mean()), float(ys.mean())

        # 找 mask 内颜色不匹配的区域（可能是误入的邻近工具）→ 加负样本点
        bad_region = mask & (color_sc > MASK_BAD_COLOR_THRESH)
        if bad_region.sum() >= MASK_BAD_MIN_PX:
            # 连通分量分析，每个显著区域放一个负样本点
            n_labels, labels = cv2.connectedComponents(bad_region.astype(np.uint8))
            for lbl in range(1, n_labels):
                comp = labels == lbl
                if comp.sum() >= MASK_BAD_MIN_PX:
                    comp_ys, comp_xs = np.where(comp)
                    extra_neg_pts.append((float(comp_xs.mean()), float(comp_ys.mean())))

        # 在 mask 内找颜色最好的第二个正样本点（距主质心 ≥ 40px，mask 足够大时）
        if mask_px >= 400:
            cost_map = np.full((H, W), 999.0, dtype=np.float32)
            cost_map[mask] = color_sc[mask]
            # 抑制主质心周边 40px
            suppressed = cost_map.copy()
            cv2.circle(suppressed, (int(cx_ref), int(cy_ref)), 40, 999.0, -1)
            min_val = suppressed.min()
            if min_val < COLOR_MASK_MAX:
                min_idx = int(np.argmin(suppressed))
                ey, ex = divmod(min_idx, W)
                extra_pos_pts.append((float(ex), float(ey)))
    else:
        cx_ref, cy_ref = float(xs.mean()), float(ys.mean())

    return np.array([cx_ref, cy_ref]), f"ok_score={score:.2f}", extra_neg_pts, extra_pos_pts


# ─── 现有 prompt 读写 ──────────────────────────────────────────────────────────

def load_existing_prompts(pf: Path) -> dict:
    if not pf.exists():
        return {}
    try:
        return json.loads(pf.read_text())
    except Exception:
        return {}


def existing_role_frames(prompt_data: dict) -> set[tuple]:
    """返回已有标注的 (role, frame_index) 集合（有 points 的）。"""
    result = set()
    for obj in prompt_data.get("objects", []):
        if obj.get("points"):
            result.add((obj["role"], obj.get("frame_index", 0)))
    return result


def make_prompt_entry(role: str, frame: int, cx: float, cy: float,
                      neg_pts: list[tuple],
                      extra_pos_pts: list[tuple] | None = None) -> dict:
    points = [[round(cx, 1), round(cy, 1)]]
    labels = [1]
    for px, py in (extra_pos_pts or []):
        points.append([round(px, 1), round(py, 1)])
        labels.append(1)
    for nx, ny in neg_pts:
        points.append([round(nx, 1), round(ny, 1)])
        labels.append(0)
    return {
        "role": role,
        "frame_index": frame,
        "points": points,
        "labels": labels,
        "auto_generated": True,
        "generated_at": datetime.datetime.now().isoformat(),
    }


def write_prompts(pf: Path, cam: str, new_objects: list[dict],
                  template_data: dict, overwrite: bool):
    """写入或合并 prompt JSON。"""
    existing = load_existing_prompts(pf) if not overwrite else {}
    existing_rf = existing_role_frames(existing)

    if not existing:
        # 复制 template 的头部字段（schema、camera 信息等）
        data = {k: v for k, v in template_data.items() if k not in ("objects",)}
        data["camera"] = cam
        data["objects"] = []
    else:
        data = existing

    added = 0
    for obj in new_objects:
        rf = (obj["role"], obj["frame_index"])
        if rf in existing_rf and not overwrite:
            continue
        # 移除旧的同 role+frame 对象（仅在 overwrite 模式）
        if overwrite:
            data["objects"] = [o for o in data["objects"]
                                if not (o["role"] == obj["role"] and
                                        o.get("frame_index") == obj["frame_index"])]
        data["objects"].append(obj)
        added += 1

    if added == 0:
        return 0

    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    return added


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def get_exp_keyframes(task: str, exp: str) -> dict[str, list[int]]:
    """返回 {camera: [keyframe_list]} from _manifest.json。"""
    mf = BUNDLE / task / exp / "_manifest.json"
    if not mf.exists():
        return {}
    try:
        d = json.loads(mf.read_text())
        return {c["camera"]: c["keyframes"] for c in d.get("cameras", [])}
    except Exception:
        return {}


def _predict_single_cam_frame(
    task, exp, cam, frame, store, matchers, min_feat_sim, color_threshold,
    embed_path, img_path, decoder
) -> dict[str, tuple | None]:
    """
    对单个相机 × 关键帧预测所有 role 的位置。
    返回 {role: (center_xy, conf, method, extra_neg_pts) | None}
    """
    embed = load_embed(embed_path)
    img   = cv2.imread(str(img_path)) if img_path.exists() else None

    frame_preds: dict[str, tuple | None] = {}
    for role in store._data:
        matcher = matchers.get(role)
        center, conf, method, extra_negs, extra_pos = None, 0.0, "", [], []

        if embed is not None:
            raw_pt, feat_conf = predict_from_store(
                embed, store, role, min_sim=min_feat_sim, cam=cam)
            if raw_pt is not None and embed_path.exists():
                refined, _, negs, pos = sam2_verify_refine(
                    embed_path, img, *raw_pt, matcher, decoder)
                if refined is not None:
                    center, conf, method, extra_negs, extra_pos = refined, feat_conf, "feature", negs, pos

        if center is None and img is not None and matcher is not None:
            raw_color, color_conf, _ = matcher.find_object(img, color_threshold)
            if raw_color is not None and color_conf >= 0.45:
                if embed_path.exists():
                    refined_c, _, negs_c, pos_c = sam2_verify_refine(
                        embed_path, img, *raw_color, matcher, decoder)
                    if refined_c is not None:
                        center, conf, method, extra_negs, extra_pos = refined_c, color_conf, "color", negs_c, pos_c
                else:
                    center, conf, method = raw_color, color_conf, "color"

        frame_preds[role] = (center, conf, method, extra_negs, extra_pos) if center is not None else None

    return frame_preds


def annotate_exp(task: str, exp: str,
                 store: TemplateFeatureStore,
                 matchers: dict[str, "ColorMatcher"],
                 min_feat_sim: float,
                 color_threshold: float,
                 dry_run: bool, overwrite: bool,
                 template_header: dict,
                 decoder=None) -> dict:
    """
    对 exp 的所有相机×关键帧：
      阶段 1 — 每个相机独立预测（特征匹配 + SAM2 验证 + 颜色 fallback）
      阶段 2 — 多视角投票：把各相机预测三角化为 3D 点，剔除外点，重投影修正坐标
      阶段 3 — 生成 prompt（含互斥 role 负样本），写入文件
    """
    cam_kfs = get_exp_keyframes(task, exp)
    if not cam_kfs:
        return {}

    prompt_dir = AUTO_PROMPTS / task / exp / "tool_masks" / "prompts"
    roles = list(store._data.keys())

    # 早期退出：若所有相机的 json 已存在且 overwrite=False，直接跳过
    if not overwrite and all((prompt_dir / f"{cam}.json").exists() for cam in cam_kfs):
        n = sum(len(kfs) for kfs in cam_kfs.values())
        return {r: {"added": 0, "skipped": n, "not_found": 0, "voted": 0} for r in roles}

    calib_available = bool(load_calibration(task))
    stats = defaultdict(lambda: {"added": 0, "skipped": 0, "not_found": 0,
                                  "voted": 0})

    # ── 阶段 1：收集所有相机 × 帧的预测 ──────────────────────────────────────
    # all_preds[frame][cam] = {role: (center, conf, method) | None}
    all_frames = sorted({f for kfs in cam_kfs.values() for f in kfs})
    all_preds: dict[int, dict[str, dict]] = {f: {} for f in all_frames}

    for cam, keyframes in cam_kfs.items():
        cam_id = cam_name_to_id(cam)
        for frame in keyframes:
            embed_path = BUNDLE / task / exp / f"{cam}.kf{frame}.embed.npz"
            img_path   = BUNDLE / task / exp / f"{cam}.kf{frame}.jpg"
            all_preds[frame][cam] = _predict_single_cam_frame(
                task, exp, cam, frame, store, matchers,
                min_feat_sim, color_threshold,
                embed_path, img_path, decoder
            )

    # ── 阶段 2：多视角投票，修正各相机坐标 ────────────────────────────────────
    # voted_centers[frame][role][cam] = center_xy（投票后）
    voted_centers: dict[int, dict[str, dict[str, np.ndarray]]] = {}

    for frame in all_frames:
        voted_centers[frame] = {}
        for role in roles:
            # 收集这一帧里所有有预测的相机
            cam_pts: dict[int, np.ndarray] = {}
            cam_meta: dict[str, tuple] = {}  # cam → (conf, method, extra_negs, extra_pos)
            for cam, preds in all_preds[frame].items():
                pred = preds.get(role)
                if pred is not None:
                    cam_id = cam_name_to_id(cam)
                    cam_pts[cam_id] = pred[0]   # center_xy
                    cam_meta[cam]   = (pred[1], pred[2], pred[3], pred[4])  # conf, method, extra_negs, extra_pos

            if calib_available and len(cam_pts) >= 2:
                voted_pts, inlier_ids = vote_frame_predictions(cam_pts, task)

                voted_centers[frame][role] = {}
                for cam in cam_kfs:
                    cid = cam_name_to_id(cam)
                    if cid in voted_pts:
                        conf, method, extra_negs, extra_pos = cam_meta.get(cam, (0.5, "voted", [], []))
                        if cam not in cam_meta:
                            method = "voted"
                        voted_centers[frame][role][cam] = (
                            voted_pts[cid], conf,
                            method + ("+voted" if cid in inlier_ids else "+reproj"),
                            extra_negs, extra_pos,
                        )
            else:
                # 标定不可用或相机数不足，直接用原始预测
                voted_centers[frame][role] = {}
                for cam, preds in all_preds[frame].items():
                    pred = preds.get(role)
                    if pred is not None:
                        voted_centers[frame][role][cam] = pred

    # ── 阶段 3：生成 prompt 并写入文件 ────────────────────────────────────────
    for cam, keyframes in cam_kfs.items():
        pf = prompt_dir / f"{cam}.json"
        existing = load_existing_prompts(pf)
        existing_rf = existing_role_frames(existing)
        new_objects = []

        for frame in keyframes:
            img_path = BUNDLE / task / exp / f"{cam}.kf{frame}.jpg"
            img = cv2.imread(str(img_path)) if img_path.exists() else None

            # 当前帧、当前相机的所有 role 投票结果
            frame_voted = {
                role: voted_centers[frame].get(role, {}).get(cam)
                for role in roles
            }

            for role in roles:
                if (role, frame) in existing_rf and not overwrite:
                    stats[role]["skipped"] += 1
                    continue

                pred = frame_voted.get(role)
                if pred is None:
                    stats[role]["not_found"] += 1
                    continue

                center, conf, method, mask_neg_pts, extra_pos_pts = pred
                cx, cy = float(center[0]), float(center[1])
                matcher = matchers.get(role)

                # 背景负样本
                if img is not None and matcher is not None:
                    neg_pts = matcher.background_points(img, cx, cy, N_NEG_POINTS)
                else:
                    H, W = (720, 1280) if img is None else img.shape[:2]
                    neg_pts = [(W * 0.05, H * 0.05), (W * 0.95, H * 0.05)]

                # mask 内颜色不匹配的区域（误入的邻近工具等）
                neg_pts.extend(mask_neg_pts)

                # 互斥 role 的位置作为额外负样本
                for excl_role in EXCLUSIVE_ROLES.get(role, set()):
                    excl = frame_voted.get(excl_role)
                    if excl is not None:
                        neg_pts.append((float(excl[0][0]), float(excl[0][1])))

                obj_entry = make_prompt_entry(role, frame, cx, cy, neg_pts, extra_pos_pts)
                obj_entry["confidence"] = round(conf, 3)
                obj_entry["method"] = method
                new_objects.append(obj_entry)
                stats[role]["added"] += 1
                if "voted" in method:
                    stats[role]["voted"] += 1

        if not new_objects:
            continue
        if not dry_run:
            write_prompts(pf, cam, new_objects, template_header, overwrite)

    return dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--keyword", required=True,
                        help="如 plasticspoon_scoop_almond_nuts（由 extract 脚本输出）")
    parser.add_argument("--min-sim", type=float, default=0.65,
                        help="特征匹配最低余弦相似度（0.65=推荐）")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="颜色匹配阈值（fallback，越低越严格）")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已有 auto_generated 标注")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-exps", type=int, default=0,
                        help="最多处理 N 个 exp（0=全部）")
    args = parser.parse_args()

    task_slug = args.task.replace("/", "_")
    model_dir = MODEL_DIR / task_slug

    # 加载 meta
    meta_path = model_dir / f"{args.keyword}.meta.json"
    if not meta_path.exists():
        print(f"[ERROR] Color model not found: {meta_path}")
        print("  Run template_color_extract.py first.")
        return
    meta = json.loads(meta_path.read_text())
    template_exp = meta["template_exp"]
    print(f"Task    : {args.task}")
    print(f"Keyword : {args.keyword}")
    print(f"Template: {template_exp}")

    # 加载颜色匹配器
    matchers: dict[str, ColorMatcher] = {}
    for role, role_meta in meta.get("roles", {}).items():
        npz_path = model_dir / role_meta["npz"]
        if not npz_path.exists():
            print(f"  [WARN] missing npz: {npz_path}")
            continue
        matchers[role] = ColorMatcher(npz_path, role_meta)
        print(f"  Loaded {role}: {role_meta['n_fg_px']} fg px, "
              f"mean_pos=({role_meta['mean_cx']:.2f},{role_meta['mean_cy']:.2f})")

    if not matchers:
        print("[ERROR] No color models loaded.")
        return

    # template_exp may be pipe-separated (from cross-task model builder)
    template_exps = [t.strip() for t in template_exp.split("|")]
    # template_task: the task where the template exp actually lives (may differ from args.task
    # when the model was built cross-task)
    template_task = meta.get("template_task") or args.task

    # 获取 template 的头部信息（用于新建 JSON 时的元数据）
    template_header = {}
    for _tpl in template_exps:
        tpl_pfs = list((PROMPTS / template_task / _tpl / "tool_masks" / "prompts").glob("*.json"))
        if tpl_pfs:
            template_header = json.loads(tpl_pfs[0].read_text())
            break

    # 加载特征 store（主定位器）
    print(f"\nBuilding feature store from template (task={template_task})...")
    store = TemplateFeatureStore()
    for _tpl in template_exps:
        store.add_from_prompts(template_task, _tpl)

    # 初始化 SAM2 decoder（用于验证候选点）
    decoder = None
    try:
        from decoder import Sam2CpuDecoder
        print("Loading SAM2 decoder for verification...")
        decoder = Sam2CpuDecoder()
        print("  SAM2 decoder ready.")
    except Exception as e:
        print(f"  [WARN] SAM2 decoder unavailable, running without verification: {e}")

    # 检查标定数据可用性
    calib = load_calibration(args.task)
    if calib:
        print(f"Calibration loaded: {len(calib)} cameras for {args.task}")
    else:
        print(f"[WARN] No calibration found for {args.task}, multiview voting disabled")

    # 找匹配关键词的实验列表（从 DB）
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT exp FROM tasks WHERE task=? ORDER BY exp", (args.task,)
    ).fetchall()
    all_exps = [r["exp"] for r in rows if args.keyword in exp_keyword(r["exp"])]
    # 排除 template 自身，并过滤掉没有 embed 的 exp（bundle 已删除的 OK 实验）
    target_exps = [e for e in all_exps
                   if e not in template_exps
                   and (BUNDLE / args.task / e / "_manifest.json").exists()]

    if args.max_exps > 0:
        target_exps = target_exps[:args.max_exps]

    print(f"\nTarget exps (keyword match): {len(target_exps)}")
    if args.dry_run:
        print("[DRY RUN]")

    total_added = defaultdict(int)
    total_not_found = defaultdict(int)
    total_skipped = defaultdict(int)
    total_voted = defaultdict(int)

    for i, exp in enumerate(target_exps):
        stats = annotate_exp(args.task, exp, store, matchers,
                             getattr(args, "min_sim", 0.65),
                             args.threshold,
                             args.dry_run, args.overwrite, template_header,
                             decoder=decoder)
        for role, s in stats.items():
            total_added[role]     += s.get("added", 0)
            total_not_found[role] += s.get("not_found", 0)
            total_skipped[role]   += s.get("skipped", 0)
            total_voted[role]     += s.get("voted", 0)

        if (i + 1) % 20 == 0 or (i + 1) == len(target_exps):
            print(f"  [{i+1}/{len(target_exps)}] {exp[:50]}")

    print("\n─── 完成 ────────────────────────────────────────")
    for role in sorted(set(list(total_added) + list(total_not_found))):
        voted = total_voted[role]
        print(f"  {role}:")
        print(f"    生成: {total_added[role]}  (其中多视角修正: {voted})"
              f"  未检测到: {total_not_found[role]}  跳过: {total_skipped[role]}")


if __name__ == "__main__":
    main()
