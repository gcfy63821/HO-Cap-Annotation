"""
Step 1: 从 template 实验提取颜色模型

对 template exp 的每个相机 × 关键帧 × 角色：
  1. 加载人工标注的点 prompt
  2. 用 SAM2 decoder 推断精确 mask
  3. 从 mask 内像素提取颜色分布（LAB + HSV）
  4. 同时记录 mask 在图像中的位置分布

输出：volunteer_annotation/color_models/{task}/{keyword}.npz
         + keyword.meta.json（可读信息）

Usage:
  python template_color_extract.py \
    --task videos_0106/spoon_scoop_nuts \
    --template-exp 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_10
"""
import sys, json, argparse, cv2, re
import numpy as np
from pathlib import Path

# --- 路径 ---
ROOT = Path(__file__).resolve().parents[0]
CLOUD_DIR = ROOT / "cloud"
SAM2_DIR = ROOT.parents[2] / "mesh_reconstruction" / "sam2"
sys.path.insert(0, str(CLOUD_DIR))
sys.path.insert(0, str(SAM2_DIR))

BUNDLE = Path("/data/robotool/_va_bundle_v2")
PROMPTS = Path("/data/robotool/_va_bundle_v2_prompts")
MODEL_DIR = ROOT / "color_models"

from decoder import Sam2CpuDecoder


# ─── 关键词提取 ────────────────────────────────────────────────────────────────

def exp_keyword(exp_name: str) -> str:
    """从 exp 名提取工具+操作对象关键词（去掉日期和序号）。

    例: 20260105_plasticspoon_scoop_almond_nuts_from_deepbowl_22
    → plasticspoon_scoop_almond_nuts
    """
    # 去掉开头日期 (8位数字_)
    s = re.sub(r"^\d{8}_", "", exp_name)
    # 去掉结尾 _数字（实验序号）
    s = re.sub(r"_\d+$", "", s)
    # 去掉 _from_xxx（容器描述，与物体颜色无关）
    s = re.sub(r"_from_.+", "", s)
    return s


# ─── 颜色模型（从精确 mask 建立） ─────────────────────────────────────────────

def extract_mask_colors(img_bgr: np.ndarray, mask: np.ndarray,
                        dilate_px: int = 8) -> dict:
    """
    从 SAM2 mask 提取颜色分布。
    返回：
      fg_lab: (N, 3) 前景像素 LAB
      fg_hsv: (N, 3) 前景像素 HSV
      bg_lab: (M, 3) 背景边框像素 LAB（用于对比）
      mask_cx, mask_cy: 质心（0-1 归一化）
    """
    H, W = img_bgr.shape[:2]
    m = mask.astype(np.uint8)

    # 前景
    fg_bgr = img_bgr[m == 1]
    if len(fg_bgr) == 0:
        return None
    fg_lab = cv2.cvtColor(fg_bgr[:, None, :], cv2.COLOR_BGR2LAB)[:, 0, :].astype(np.float32)
    fg_hsv = cv2.cvtColor(fg_bgr[:, None, :], cv2.COLOR_BGR2HSV)[:, 0, :].astype(np.float32)

    # 背景边框（mask 膨胀 - mask）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
    dilated = cv2.dilate(m, kernel)
    border = (dilated - m).astype(bool)
    bg_bgr = img_bgr[border]
    bg_lab = cv2.cvtColor(bg_bgr[:, None, :], cv2.COLOR_BGR2LAB)[:, 0, :].astype(np.float32) \
        if len(bg_bgr) > 0 else np.zeros((0, 3), np.float32)

    # 质心
    ys, xs = np.where(m)
    cx, cy = xs.mean() / W, ys.mean() / H

    return dict(fg_lab=fg_lab, fg_hsv=fg_hsv, bg_lab=bg_lab, mask_cx=cx, mask_cy=cy,
                mask_area=int(m.sum()), img_hw=(H, W))


# ─── 形状特征（从 SAM2 mask 提取） ───────────────────────────────────────────

def extract_mask_shape(mask: np.ndarray) -> dict | None:
    """
    从二值 mask 提取形状特征向量。
    返回的特征对旋转/平移不变（solidity、compactness、aspect 对缩放鲁棒）。

    特征说明：
      solidity    = area / convex_hull_area  （叉子有间隙 ≈0.6，勺 ≈0.9）
      compactness = perimeter² / (4π·area)  （越复杂越高，圆 = 1.0）
      aspect      = long_side / short_side   （长宽比）
      extent      = area / bbox_area         （填充率）
      log_hu[:4]  = log10(|Hu moments|)     （整体形状描述符，旋转不变）
    """
    m = mask.astype(np.uint8)
    area = int(m.sum())
    if area < 50:
        return None

    ys, xs = np.where(m)
    h_bb = int(ys.max() - ys.min() + 1)
    w_bb = int(xs.max() - xs.min() + 1)
    aspect = max(h_bb, w_bb) / max(min(h_bb, w_bb), 1)
    extent = area / max(h_bb * w_bb, 1)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(cnt, True)
    compactness = float(perimeter ** 2 / (4 * np.pi * area)) if area > 0 else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / max(hull_area, 1))

    moments = cv2.moments(m)
    hu = cv2.HuMoments(moments).flatten()
    log_hu = (-np.sign(hu) * np.log10(np.abs(hu) + 1e-10)).astype(np.float32)

    return {
        "solidity":    float(solidity),
        "compactness": float(compactness),
        "aspect":      float(aspect),
        "extent":      float(extent),
        "log_hu":      log_hu[:4],
    }


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def detect_roles(task: str, template_exp: str) -> list[str]:
    """从 template prompt 中自动检测所有出现的 role 名称。"""
    prompt_dir = PROMPTS / task / template_exp / "tool_masks" / "prompts"
    roles = set()
    for pf in prompt_dir.glob("*.json"):
        try:
            d = json.loads(pf.read_text())
            for obj in d.get("objects", []):
                if obj.get("role"):
                    roles.add(obj["role"])
        except Exception:
            pass
    # 固定顺序：主工具 → 操作对象 → 辅助工具
    order = ["primary_tool", "manipulated_object", "auxiliary_tool"]
    return sorted(roles, key=lambda r: order.index(r) if r in order else 99)


def build_model_for_template(task: str, template_exp: str, dec: Sam2CpuDecoder,
                              roles: list[str] | None = None):
    """
    对 template_exp 每个相机×帧×角色运行 SAM2，收集颜色和形状样本。
    roles=None 时自动从 prompt 检测所有 role（包括 auxiliary_tool）。
    返回 {role: [sample_dict, ...]}
    """
    prompt_dir = PROMPTS / task / template_exp / "tool_masks" / "prompts"
    if not prompt_dir.exists():
        print(f"  [WARN] no prompts: {prompt_dir}")
        return {}

    if roles is None:
        roles = detect_roles(task, template_exp)
        print(f"  Auto-detected roles: {roles}")

    # role → list of color dicts across all cameras & frames
    role_colors: dict[str, list[dict]] = {r: [] for r in roles}
    role_frame_map: dict[str, set] = {r: set() for r in roles}  # 记录已处理的帧

    for pf in sorted(prompt_dir.glob("*.json")):
        try:
            data = json.loads(pf.read_text())
        except Exception:
            continue
        cam = data["camera"]

        for obj in data.get("objects", []):
            role = obj.get("role")
            if role not in roles:
                continue
            pts = obj.get("points", [])
            labels = obj.get("labels", [1] * len(pts))
            frame = obj.get("frame_index", 0)
            if not pts:
                continue

            embed_path = BUNDLE / task / template_exp / f"{cam}.kf{frame}.embed.npz"
            img_path   = BUNDLE / task / template_exp / f"{cam}.kf{frame}.jpg"
            if not embed_path.exists() or not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            try:
                mask, score = dec.infer(embed_path, pts, labels)
            except Exception as e:
                print(f"  [WARN] SAM2 fail {cam} frame={frame}: {e}")
                continue

            if score < 0.5 or mask.sum() < 50:
                print(f"  [WARN] low quality mask: {cam} {role} frame={frame} score={score:.2f}")
                continue

            colors = extract_mask_colors(img, mask)
            if colors is None:
                continue

            shape = extract_mask_shape(mask)
            colors["cam"] = cam
            colors["frame"] = frame
            colors["score"] = float(score)
            colors["shape"] = shape
            role_colors[role].append(colors)
            shape_str = (f"sol={shape['solidity']:.2f} cmp={shape['compactness']:.2f}"
                         if shape else "shape=N/A")
            print(f"  OK  {cam}  {role}  frame={frame}  "
                  f"area={colors['mask_area']}px  score={score:.2f}  {shape_str}")

    return role_colors


def save_model(model_dir: Path, keyword: str, role_colors: dict, template_exp: str,
               template_task: str | None = None):
    """将颜色模型保存为 NPZ + JSON。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"template_exp": template_exp, "keyword": keyword, "roles": {}}
    if template_task is not None:
        meta["template_task"] = template_task

    for role, samples in role_colors.items():
        if not samples:
            continue
        fg_lab = np.concatenate([s["fg_lab"] for s in samples], axis=0)
        fg_hsv = np.concatenate([s["fg_hsv"] for s in samples], axis=0)
        bg_lab = np.concatenate([s["bg_lab"] for s in samples if len(s["bg_lab"]) > 0], axis=0)
        cx_list = [s["mask_cx"] for s in samples]
        cy_list = [s["mask_cy"] for s in samples]

        # ── 形状统计 ──
        shape_samples = [s["shape"] for s in samples if s.get("shape") is not None]
        shape_npz = {}
        shape_meta = {}
        if shape_samples:
            for key in ("solidity", "compactness", "aspect", "extent"):
                vals = np.array([sh[key] for sh in shape_samples], np.float32)
                shape_npz[f"shape_{key}_mean"] = np.array([vals.mean()], np.float32)
                shape_npz[f"shape_{key}_std"]  = np.array([max(vals.std(), 1e-4)], np.float32)
                shape_meta[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
            hu_arr = np.stack([sh["log_hu"] for sh in shape_samples])  # (N, 4)
            shape_npz["shape_log_hu_mean"] = hu_arr.mean(axis=0).astype(np.float32)
            shape_npz["shape_log_hu_std"]  = np.maximum(hu_arr.std(axis=0), 1e-4).astype(np.float32)

        npz_path = model_dir / f"{keyword}.{role}.npz"
        np.savez_compressed(str(npz_path),
                            fg_lab=fg_lab.astype(np.float16),
                            fg_hsv=fg_hsv.astype(np.float16),
                            bg_lab=bg_lab.astype(np.float16) if len(bg_lab) else np.zeros((0, 3), np.float16),
                            **shape_npz)
        meta["roles"][role] = {
            "npz": npz_path.name,
            "n_fg_px": int(len(fg_lab)),
            "n_bg_px": int(len(bg_lab)),
            "n_samples": len(samples),
            "mean_cx": float(np.mean(cx_list)),
            "mean_cy": float(np.mean(cy_list)),
            "std_cx": float(np.std(cx_list)),
            "std_cy": float(np.std(cy_list)),
            "shape": shape_meta,
        }
        if shape_meta:
            print(f"  Saved {role}: {len(fg_lab)} fg px  "
                  f"sol={shape_meta['solidity']['mean']:.2f}±{shape_meta['solidity']['std']:.2f}  "
                  f"cmp={shape_meta['compactness']['mean']:.2f}±{shape_meta['compactness']['std']:.2f}"
                  f" → {npz_path.name}")
        else:
            print(f"  Saved {role}: {len(fg_lab)} fg px, {len(bg_lab)} bg px → {npz_path.name}")

    meta_path = model_dir / f"{keyword}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  Meta → {meta_path}")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="如 videos_0106/spoon_scoop_nuts")
    parser.add_argument("--template-exp", required=True, help="template 实验名（全名）")
    parser.add_argument("--roles", nargs="+", default=None,
                        help="指定 role 列表，默认从 prompt 自动检测所有 role")
    parser.add_argument("--keyword", default=None,
                        help="覆盖自动派生的 keyword（默认从 template_exp 提取）")
    args = parser.parse_args()

    keyword = args.keyword if args.keyword else exp_keyword(args.template_exp)
    print(f"Task    : {args.task}")
    print(f"Template: {args.template_exp}")
    print(f"Keyword : {keyword}")

    model_dir = MODEL_DIR / args.task.replace("/", "_")

    print("\nLoading SAM2 decoder...")
    dec = Sam2CpuDecoder()

    print("\nRunning SAM2 on template...")
    role_colors = build_model_for_template(args.task, args.template_exp, dec, args.roles)

    print("\nSaving color models...")
    meta = save_model(model_dir, keyword, role_colors, args.template_exp)

    print("\nDone.")
    for role, info in meta.get("roles", {}).items():
        print(f"  {role}: {info['n_fg_px']} fg px from {info['n_samples']} samples")


if __name__ == "__main__":
    main()
