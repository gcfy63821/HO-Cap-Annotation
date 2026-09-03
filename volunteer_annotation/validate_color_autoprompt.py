"""
验证：基于颜色的自动点 prompt 生成

核心用例：同一 exp 内，某帧在相机 A 已有人工标注 → 自动预测其他相机同帧的点位
跨 exp（同 task 同工具）：作为扩展验证

运行：
  python validate_color_autoprompt.py [--task videos_0106/spoon_scoop_nuts] [--cross-exp]
"""
import json, argparse, sqlite3, cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

DB = Path("/data/robotool/_va_bundle_v2/tasks.db")
BUNDLE = Path("/data/robotool/_va_bundle_v2")
PROMPTS = Path("/data/robotool/_va_bundle_v2_prompts")

PATCH_R = 12             # 颜色采样半径（像素）
HIT_RADIUS = 80          # 命中判定：预测质心与真实点距离 < 80px
MAX_CLUSTER_RATIO = 0.15 # 最大簇不超过图像面积的 15%（否则是背景）
MIN_CLUSTER_PX = 20      # 最小簇像素数


# ─── 颜色模型（单角色，基于 HSV） ──────────────────────────────────────────────

@dataclass
class ColorModel:
    role: str
    hue_samples: list[np.ndarray] = field(default_factory=list)
    sv_samples: list[np.ndarray] = field(default_factory=list)

    def add_patch(self, img_bgr: np.ndarray, cx: float, cy: float):
        h, w = img_bgr.shape[:2]
        x1, x2 = max(0, int(cx)-PATCH_R), min(w, int(cx)+PATCH_R)
        y1, y2 = max(0, int(cy)-PATCH_R), min(h, int(cy)+PATCH_R)
        patch = img_bgr[y1:y2, x1:x2]
        if patch.size < 9:
            return
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
        self.hue_samples.append(hsv[:, 0])   # H: 0-180
        self.sv_samples.append(hsv[:, 1:])   # S,V

    @property
    def n_samples(self):
        return sum(len(s) for s in self.hue_samples)

    def _hue_mean_std(self) -> tuple[float, float]:
        """循环均值/标准差（Hue 是环形量）"""
        all_h = np.concatenate(self.hue_samples) if self.hue_samples else np.array([])
        if len(all_h) == 0:
            return 0, 30
        # 转到 [-180, 180] 以 0 为中心的角度域做统计
        theta = all_h / 180.0 * np.pi * 2  # → [0, 4π]
        sin_m, cos_m = np.sin(theta).mean(), np.cos(theta).mean()
        mean_h = np.arctan2(sin_m, cos_m) / (np.pi * 2) * 180 % 180
        # 标准差：用最大似然圆周标准差近似
        r = np.sqrt(sin_m**2 + cos_m**2)
        std_h = max(8, np.sqrt(-2 * np.log(max(r, 1e-6))) / (np.pi*2) * 180)
        return float(mean_h), float(std_h)

    def _sv_stats(self) -> tuple[np.ndarray, np.ndarray]:
        all_sv = np.concatenate(self.sv_samples) if self.sv_samples else np.zeros((1, 2))
        return all_sv.mean(axis=0), np.maximum(all_sv.std(axis=0), 10)

    def match_image(self, img_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        """返回 (质心 [cx,cy] 或 None, debug_mask)"""
        if not self.hue_samples:
            return None, np.zeros(img_bgr.shape[:2], np.uint8)
        H, W = img_bgr.shape[:2]

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        # ── Hue 距离（环形）
        mean_h, std_h = self._hue_mean_std()
        dh = np.abs(h_ch - mean_h)
        dh = np.minimum(dh, 180 - dh)           # wrap-around
        h_score = dh / (std_h * 1.5)            # < 1 → 合格

        # ── Saturation / Value
        sv_mean, sv_std = self._sv_stats()
        s_score = np.abs(s_ch - sv_mean[0]) / (sv_std[0] * 1.5)
        v_score = np.abs(v_ch - sv_mean[1]) / (sv_std[1] * 1.5)

        # 综合分（取 max，即最差通道决定）
        score = np.maximum(np.maximum(h_score, s_score), v_score)

        # 低饱和度时放宽 Hue 要求（灰/白物体 hue 不稳定）
        low_sat = s_ch < 40
        score[low_sat] = np.maximum(s_score[low_sat], v_score[low_sat])

        mask = (score < 1.0).astype(np.uint8) * 255

        # 形态学去噪
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.dilate(mask, k)

        # 连通域分析
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return None, mask

        # 过滤：忽略背景大区域
        max_area = H * W * MAX_CLUSTER_RATIO
        candidates = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if MIN_CLUSTER_PX <= area <= max_area:
                candidates.append((area, i))

        if not candidates:
            return None, mask

        # 取面积最大的候选
        best_i = max(candidates, key=lambda x: x[0])[1]
        cx, cy = centroids[best_i]
        return np.array([cx, cy]), mask


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def load_prompts_for_exp(task: str, exp: str) -> dict[str, list[dict]]:
    """返回 {camera: [object_dict]}，只保留有 points 的"""
    result: dict[str, list[dict]] = {}
    d = PROMPTS / task / exp / "tool_masks" / "prompts"
    if not d.exists():
        return result
    for pf in d.glob("*.json"):
        try:
            data = json.loads(pf.read_text())
        except Exception:
            continue
        objs = [o for o in data.get("objects", [])
                if o.get("points") and o["role"] != "auxiliary_tool"]
        if objs:
            result[data["camera"]] = objs
    return result


def load_img(task: str, exp: str, cam: str, frame: int) -> np.ndarray | None:
    p = BUNDLE / task / exp / f"{cam}.kf{frame}.jpg"
    return cv2.imread(str(p)) if p.exists() else None


def positive_points(obj: dict) -> list[tuple[float, float]]:
    pts = obj.get("points", [])
    labels = obj.get("labels", [1] * len(pts))
    return [(x, y) for (x, y), lb in zip(pts, labels) if lb == 1]


# ─── 同一 exp 内跨相机验证 ─────────────────────────────────────────────────────

def validate_within_exp(task: str, exp: str) -> list[dict]:
    """
    对每个角色、每帧：
    - 用 N-1 个相机的标注建颜色模型
    - 预测第 N 个相机的点位（留一法）
    """
    cam_prompts = load_prompts_for_exp(task, exp)
    if len(cam_prompts) < 2:
        return []

    results = []

    # 收集所有 (role, frame) 组合
    role_frame_cams: dict[tuple, dict[str, list[tuple]]] = defaultdict(dict)
    for cam, objs in cam_prompts.items():
        for obj in objs:
            key = (obj["role"], obj["frame_index"])
            pts = positive_points(obj)
            if pts:
                role_frame_cams[key][cam] = pts

    for (role, frame), cam_pts in role_frame_cams.items():
        if len(cam_pts) < 2:
            continue   # 至少需要 2 个相机才能做留一

        cams = list(cam_pts.keys())
        for test_cam in cams:
            # 建颜色模型：用其余相机
            model = ColorModel(role=role)
            for src_cam, pts in cam_pts.items():
                if src_cam == test_cam:
                    continue
                img = load_img(task, exp, src_cam, frame)
                if img is None:
                    continue
                for x, y in pts:
                    model.add_patch(img, x, y)

            if model.n_samples == 0:
                continue

            # 预测 test_cam
            img_test = load_img(task, exp, test_cam, frame)
            if img_test is None:
                continue

            pred, _ = model.match_image(img_test)
            gt_pts = np.array(cam_pts[test_cam])

            if pred is None:
                results.append(dict(exp=exp, cam=test_cam, frame=frame, role=role,
                                    detected=False, hit=False, dist=None))
                continue

            dists = np.linalg.norm(gt_pts - pred, axis=1)
            min_dist = float(dists.min())
            hit = min_dist < HIT_RADIUS
            results.append(dict(exp=exp, cam=test_cam, frame=frame, role=role,
                                detected=True, hit=hit, dist=min_dist,
                                pred=pred.tolist(), gt=gt_pts[int(dists.argmin())].tolist()))
    return results


# ─── 跨 exp（同 task）验证 ────────────────────────────────────────────────────

def validate_cross_exp(task: str, train_exps: list[str], test_exps: list[str]) -> list[dict]:
    """
    用 train_exps 建颜色模型，预测 test_exps 各相机各帧
    注意：需要 task 内工具类型一致
    """
    # 建颜色模型
    models: dict[str, ColorModel] = {}
    for exp in train_exps:
        for cam, objs in load_prompts_for_exp(task, exp).items():
            for obj in objs:
                role = obj["role"]
                pts = positive_points(obj)
                if not pts:
                    continue
                img = load_img(task, exp, cam, obj["frame_index"])
                if img is None:
                    continue
                if role not in models:
                    models[role] = ColorModel(role=role)
                for x, y in pts:
                    models[role].add_patch(img, x, y)

    results = []
    for exp in test_exps:
        for cam, objs in load_prompts_for_exp(task, exp).items():
            for obj in objs:
                role = obj["role"]
                if role not in models:
                    continue
                gt_pts = positive_points(obj)
                if not gt_pts:
                    continue
                img = load_img(task, exp, cam, obj["frame_index"])
                if img is None:
                    continue

                pred, _ = models[role].match_image(img)
                gt_arr = np.array(gt_pts)

                if pred is None:
                    results.append(dict(exp=exp, cam=cam, frame=obj["frame_index"],
                                        role=role, detected=False, hit=False, dist=None))
                    continue

                dists = np.linalg.norm(gt_arr - pred, axis=1)
                min_dist = float(dists.min())
                results.append(dict(exp=exp, cam=cam, frame=obj["frame_index"], role=role,
                                    detected=True, hit=min_dist < HIT_RADIUS,
                                    dist=min_dist))
    return results


# ─── 报告 ─────────────────────────────────────────────────────────────────────

def report(results: list[dict], title: str):
    if not results:
        print(f"  [{title}] 无结果")
        return
    by_role: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_role[r["role"]].append(r)
    print(f"\n  ── {title} ──")
    for role in sorted(by_role):
        rlist = by_role[role]
        det = [r for r in rlist if r["detected"]]
        hit = [r for r in det if r["hit"]]
        dists = [r["dist"] for r in det if r["dist"] is not None]
        det_r = len(det)/len(rlist)*100
        hit_r = len(hit)/len(det)*100 if det else 0
        med_d = np.median(dists) if dists else float("nan")
        print(f"    {role:25s}  n={len(rlist):4d}  "
              f"检测率={det_r:5.1f}%  命中率={hit_r:5.1f}%  "
              f"中位距离={med_d:6.1f}px")


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None)
    parser.add_argument("--cross-exp", action="store_true", help="额外做跨 exp 验证")
    parser.add_argument("--max-exps", type=int, default=60)
    parser.add_argument("--all-tasks", action="store_true")
    args = parser.parse_args()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT task, exp FROM tasks WHERE status='submitted' ORDER BY task, exp"
    ).fetchall()

    by_task: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        task, exp = r["task"], r["exp"]
        exp_dir = BUNDLE / task / exp
        if exp_dir.exists() and any(f.suffix == ".jpg" for f in exp_dir.iterdir()):
            by_task[task].append(exp)

    if args.task:
        task_list = [args.task]
    elif args.all_tasks:
        task_list = sorted(by_task.keys())
    else:
        task_list = [t for t, _ in sorted(by_task.items(), key=lambda x: -len(x[1]))[:4]]

    all_within: list[dict] = []

    for task in task_list:
        exps = by_task.get(task, [])[:args.max_exps]
        if len(exps) < 2:
            continue
        print(f"\n{'='*62}")
        print(f"Task: {task}  ({len(exps)} exps with images)")

        # 1. 同 exp 跨相机
        within_results = []
        for exp in exps:
            within_results.extend(validate_within_exp(task, exp))
        report(within_results, "同 exp 跨相机（留一法）")
        all_within.extend(within_results)

        # 2. 跨 exp（可选）
        if args.cross_exp and len(exps) >= 4:
            n_train = max(2, int(len(exps) * 0.6))
            cross_results = validate_cross_exp(task, exps[:n_train], exps[n_train:])
            report(cross_results, f"跨 exp（train={n_train}, test={len(exps)-n_train}）")

    if len(task_list) > 1 and all_within:
        print(f"\n{'='*62}")
        report(all_within, "全局汇总（同 exp 跨相机）")


if __name__ == "__main__":
    main()
