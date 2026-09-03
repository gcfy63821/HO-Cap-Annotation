"""
SAM2 特征图匹配：利用已有 embed.npz 定位目标

思路：
  1. template 已有人工标注点 (x, y)
  2. 在 template 的 embed feature map 中提取该位置的特征向量
  3. 在 target embed feature map 中做余弦相似度搜索
  4. 最相似位置 → 预测坐标

优势：使用 SAM2 的视觉特征（而非原始颜色），对光照/视角变化鲁棒得多。
分辨率：image_embed 64×64 对应约 20px×11px/格，spatial precision ~15px。

独立使用：
  python template_feature_match.py --task ... --template-exp ...

或作为模块由 template_auto_annotate.py 调用。
"""
import sys, json, cv2
import numpy as np
from pathlib import Path
from typing import Optional

BUNDLE       = Path("/data/robotool/_va_bundle_v2")
PROMPTS      = Path("/data/robotool/_va_bundle_v2_prompts")
AUTO_PROMPTS = Path("/data/robotool/_va_bundle_v2_auto_prompts")
ROOT         = Path(__file__).resolve().parent


# ─── 特征提取工具 ──────────────────────────────────────────────────────────────

def load_embed(embed_path: Path) -> dict | None:
    if not embed_path.exists():
        return None
    try:
        z = np.load(str(embed_path))
        return {
            "image_embed":    z["image_embed"].astype(np.float32),   # (256,64,64)
            "high_res_feat1": z["high_res_feat_1"].astype(np.float32),# (64,128,128)
            "orig_hw":        tuple(int(x) for x in z["orig_hw"]),   # (H, W)
        }
    except Exception:
        return None


def pt_to_feat_idx(px: float, py: float, feat_h: int, feat_w: int,
                   img_h: int, img_w: int) -> tuple[int, int]:
    """图像坐标 → feature map 下标（四舍五入）。"""
    fy = int(round(py / img_h * feat_h))
    fx = int(round(px / img_w * feat_w))
    fy = max(0, min(feat_h - 1, fy))
    fx = max(0, min(feat_w - 1, fx))
    return fy, fx


def feat_idx_to_pt(fy: int, fx: int, feat_h: int, feat_w: int,
                   img_h: int, img_w: int) -> tuple[float, float]:
    """feature map 下标 → 图像中心坐标（整格中心）。"""
    py = (fy + 0.5) / feat_h * img_h
    px = (fx + 0.5) / feat_w * img_w
    return px, py


def extract_feat_vec(embed: dict, px: float, py: float,
                     use_high_res: bool = True) -> np.ndarray:
    """
    提取标注点对应的特征向量。
    use_high_res=True: 将 image_embed 和 high_res_feat1 都用上（concat 后归一化）
    """
    H, W = embed["orig_hw"]

    # image_embed  (256,64,64)
    ie = embed["image_embed"]          # C×H×W
    fy1, fx1 = pt_to_feat_idx(px, py, ie.shape[1], ie.shape[2], H, W)
    vec_ie = ie[:, fy1, fx1]           # (256,)

    if use_high_res:
        hr = embed["high_res_feat1"]   # (64,128,128)
        fy2, fx2 = pt_to_feat_idx(px, py, hr.shape[1], hr.shape[2], H, W)
        vec_hr = hr[:, fy2, fx2]       # (64,)
        vec = np.concatenate([vec_ie, vec_hr])  # (320,)
    else:
        vec = vec_ie

    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def cosine_similarity_map(embed: dict, query_vec: np.ndarray,
                           use_high_res: bool = True) -> np.ndarray:
    """
    在整个 feature map 上计算余弦相似度，返回 (feat_H, feat_W) 相似度图。
    相似度越高 = 越可能是目标位置。
    """
    ie = embed["image_embed"]          # (256,64,64)
    C_ie, fH, fW = ie.shape

    if use_high_res:
        hr = embed["high_res_feat1"]   # (64,128,128)
        # 上采样 high_res_feat1 到 64×64（与 image_embed 对齐）
        hr_resized = np.zeros((hr.shape[0], fH, fW), dtype=np.float32)
        for c in range(hr.shape[0]):
            hr_resized[c] = cv2.resize(hr[c], (fW, fH), interpolation=cv2.INTER_LINEAR)
        feat_map = np.concatenate([ie, hr_resized], axis=0)  # (320,64,64)
    else:
        feat_map = ie

    # 展平空间维度：(C, fH*fW)
    flat = feat_map.reshape(feat_map.shape[0], -1)           # (C, fH*fW)

    # 归一化每个空间位置
    norms = np.linalg.norm(flat, axis=0, keepdims=True)      # (1, fH*fW)
    flat_norm = flat / (norms + 1e-8)                        # (C, fH*fW)

    # 余弦相似度
    sim = (query_vec[:, None] * flat_norm).sum(axis=0)       # (fH*fW,)
    return sim.reshape(fH, fW)


# ─── 模板特征库 ────────────────────────────────────────────────────────────────

class TemplateFeatureStore:
    """
    存储 template 实验中每个 role 的特征向量集合。
    支持多相机、多帧的平均特征查询。
    """

    def __init__(self):
        # role → list of (feat_vec, cam, frame, px, py)
        self._data: dict[str, list] = {}

    def add_from_prompts(self, task: str, template_exp: str,
                         roles: tuple[str, ...] = ("primary_tool", "manipulated_object")):
        prompt_dir = PROMPTS / task / template_exp / "tool_masks" / "prompts"
        auto_dir   = AUTO_PROMPTS / task / template_exp / "tool_masks" / "prompts"
        # collect per-cam: human wins, auto fills gaps
        cam_files: dict[str, Path] = {}
        for pf in sorted(auto_dir.glob("*.json")):
            cam_files[pf.name] = pf
        for pf in sorted(prompt_dir.glob("*.json")):
            cam_files[pf.name] = pf  # human overwrites auto
        if not cam_files:
            return
        for pf in sorted(cam_files.values()):
            try:
                d = json.loads(pf.read_text())
            except Exception:
                continue
            cam = d["camera"]
            for obj in d.get("objects", []):
                role = obj.get("role")
                if role not in roles:
                    continue
                pts = obj.get("points", [])
                labels = obj.get("labels", [1] * len(pts))
                frame = obj.get("frame_index", 0)
                pos_pts = [(x, y) for (x, y), lb in zip(pts, labels) if lb == 1]
                if not pos_pts:
                    continue

                embed_path = BUNDLE / task / template_exp / f"{cam}.kf{frame}.embed.npz"
                embed = load_embed(embed_path)
                if embed is None:
                    continue

                for px, py in pos_pts:
                    vec = extract_feat_vec(embed, px, py)
                    self._data.setdefault(role, []).append(
                        (vec, cam, frame, px, py)
                    )

        for role, items in self._data.items():
            print(f"  [FeatureStore] {role}: {len(items)} template vectors")

    def query_vectors(self, role: str, cam: str | None = None) -> list[np.ndarray]:
        """Return feature vectors for role. If cam given, same-camera vectors come first."""
        items = self._data.get(role, [])
        if cam is None:
            return [item[0] for item in items]
        same = [item[0] for item in items if item[1] == cam]
        other = [item[0] for item in items if item[1] != cam]
        return same + other

    def mean_vector(self, role: str, cam: str | None = None) -> np.ndarray | None:
        """Mean vector; if cam given, prefer same-camera vectors (weighted 2×)."""
        items = self._data.get(role, [])
        if not items:
            return None
        if cam is None:
            vecs = [item[0] for item in items]
        else:
            same = [item[0] for item in items if item[1] == cam]
            other = [item[0] for item in items if item[1] != cam]
            vecs = same * 2 + other  # same-cam weighted 2×
        m = np.stack(vecs).mean(axis=0)
        return m / (np.linalg.norm(m) + 1e-8)


# ─── 特征匹配预测 ─────────────────────────────────────────────────────────────

def predict_point_by_feature(embed: dict, query_vec: np.ndarray,
                              topk_avg: int = 3,
                              min_sim: float = 0.7) -> tuple[Optional[np.ndarray], float]:
    """
    给定 target embed 和 query 特征向量，预测目标位置。
    返回 (predicted_xy, confidence)，confidence < 0 表示未找到。

    topk_avg: 取相似度 top-k 位置做加权平均（更稳定）
    """
    sim_map = cosine_similarity_map(embed, query_vec)        # (fH, fW)
    H, W = embed["orig_hw"]
    fH, fW = sim_map.shape

    max_sim = float(sim_map.max())
    if max_sim < min_sim:
        return None, max_sim

    # Top-K 位置（加权平均）
    flat_sim = sim_map.ravel()
    if topk_avg >= flat_sim.size:
        topk_avg = flat_sim.size
    top_idxs = np.argpartition(flat_sim, -topk_avg)[-topk_avg:]
    top_sims = flat_sim[top_idxs]
    top_ys, top_xs = np.unravel_index(top_idxs, sim_map.shape)

    # 按相似度加权平均
    w = top_sims - top_sims.min() + 1e-6
    w /= w.sum()
    cx = float((top_xs * w).sum())
    cy = float((top_ys * w).sum())

    # 转回图像坐标
    px = (cx + 0.5) / fW * W
    py = (cy + 0.5) / fH * H

    return np.array([px, py]), max_sim


def predict_from_store(embed: dict, store: TemplateFeatureStore, role: str,
                       n_best_templates: int = 3,
                       min_sim: float = 0.65,
                       cam: str | None = None) -> tuple[Optional[np.ndarray], float]:
    """
    用 store 中的多个 template 向量逐一匹配，取置信度最高的结果。
    cam: 若指定，同视角向量排在前面优先匹配（same-camera priority）。
    n_best_templates: 只用相似度前 N 的 template 向量（太多会引入噪声）
    """
    vecs = store.query_vectors(role, cam=cam)
    if not vecs:
        return None, 0.0

    # 也试一下（同视角加权的）平均向量
    mean_vec = store.mean_vector(role, cam=cam)
    candidate_vecs = list(vecs[:n_best_templates]) + ([mean_vec] if mean_vec is not None else [])

    best_pt = None
    best_conf = 0.0
    for vec in candidate_vecs:
        pt, conf = predict_point_by_feature(embed, vec, min_sim=min_sim)
        if conf > best_conf:
            best_conf = conf
            best_pt = pt

    return best_pt, best_conf


# ─── 验证 ─────────────────────────────────────────────────────────────────────

def run_validation(task: str, template_exp: str, max_test_exps: int = 30,
                   hit_radius: int = 80, min_sim: float = 0.65):
    import re
    from collections import defaultdict

    def exp_keyword(name):
        s = re.sub(r"^\d{8}_", "", name)
        s = re.sub(r"_\d+$", "", s)
        s = re.sub(r"_from_.+", "", s)
        return s

    keyword = exp_keyword(template_exp)
    print(f"Template: {template_exp[:60]}")
    print(f"Keyword : {keyword}")

    print("\nBuilding feature store...")
    store = TemplateFeatureStore()
    store.add_from_prompts(task, template_exp)

    # 找 target exps
    task_dir = PROMPTS / task
    target_exps = [e.name for e in sorted(task_dir.iterdir())
                   if e.is_dir() and e.name != template_exp and keyword in exp_keyword(e.name)]
    target_exps = target_exps[:max_test_exps]
    print(f"Test exps: {len(target_exps)}")

    results = defaultdict(list)

    for exp in target_exps:
        pdir = PROMPTS / task / exp / "tool_masks" / "prompts"
        if not pdir.exists():
            continue
        for pf in sorted(pdir.glob("*.json")):
            try:
                d = json.loads(pf.read_text())
            except Exception:
                continue
            cam = d["camera"]
            for obj in d.get("objects", []):
                role = obj.get("role")
                if role not in store._data:
                    continue
                pts = obj.get("points", [])
                labels = obj.get("labels", [1] * len(pts))
                frame = obj.get("frame_index", 0)
                gt_pts = [(x, y) for (x, y), lb in zip(pts, labels) if lb == 1]
                if not gt_pts:
                    continue

                embed_path = BUNDLE / task / exp / f"{cam}.kf{frame}.embed.npz"
                embed = load_embed(embed_path)
                if embed is None:
                    continue

                pred, conf = predict_from_store(embed, store, role, min_sim=min_sim)
                gt_arr = np.array(gt_pts)

                if pred is None:
                    results[role].append({"hit": False, "dist": None, "detected": False, "conf": conf})
                    continue

                dists = np.linalg.norm(gt_arr - pred, axis=1)
                dist = float(dists.min())
                results[role].append({"hit": dist < hit_radius, "dist": dist,
                                      "detected": True, "conf": conf})

    print(f"\n── 验证结果 (hit_radius={hit_radius}px) ──")
    for role in sorted(results):
        rs = results[role]
        det = [r for r in rs if r["detected"]]
        hit = [r for r in det if r["hit"]]
        dists = [r["dist"] for r in det if r["dist"]]
        confs = [r["conf"] for r in rs]
        print(f"  {role}:")
        print(f"    n={len(rs)}  detected={len(det)/len(rs)*100:.0f}%  "
              f"hit={len(hit)/len(det)*100:.0f}%  "
              f"median_dist={np.median(dists):.0f}px  "
              f"mean_conf={np.mean(confs):.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="videos_0106/spoon_scoop_nuts")
    parser.add_argument("--template-exp",
                        default="20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_10")
    parser.add_argument("--max-exps", type=int, default=30)
    parser.add_argument("--hit-radius", type=int, default=80)
    parser.add_argument("--min-sim", type=float, default=0.65)
    args = parser.parse_args()
    run_validation(args.task, args.template_exp, args.max_exps,
                   args.hit_radius, args.min_sim)
