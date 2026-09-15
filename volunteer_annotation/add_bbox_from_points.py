"""
对已有 auto_prompts JSON（有 points 但无 bbox）的 frame_index=0 条目，
利用 ColorMatcher.bbox_estimate() 补写 bbox 字段。

用法:
  python add_bbox_from_points.py \
      --task videos_0204/spatula_flip_egg \
      --keyword redrubberspatula \
      [--ap-root /data/robotool/_va_bundle_v2_auto_prompts] \
      [--embed-root /data/robotool/_va_bundle_v2] \
      [--dry-run]

  # 批量跑所有 Feb+ task/keyword:
  python add_bbox_from_points.py --all
"""
import sys, json, argparse, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "volunteer_annotation"))

from template_auto_annotate import ColorMatcher

AP_ROOT    = Path("/data/robotool/_va_bundle_v2_auto_prompts")
EMBED_ROOT = Path("/data/robotool/_va_bundle_v2")
MODEL_DIR  = Path(__file__).parent / "color_models"

# Feb+ tasks with their keywords (from run_reannot_overwrite_all.sh)
FEB_JOBS = [
    "videos_0202/spoon_press_sponge|smallwoodenspoon",
    "videos_0202/spoon_press_sponge|bigwoodenspoon_press_sponge_in_largeshallowcontainer",
    "videos_0202/spoon_press_sponge|smallwoodenspoon_press_sponge_in_largeshallowcontainer",
    "videos_0202/spoon_press_sponge|bigwoodenspoon",
    "videos_0204/fork_flip_egg|bigwoodenspatula",
    "videos_0204/fork_flip_egg|redrubberspatula",
    "videos_0204/knife_spread_tomatosauce|orangeknife",
    "videos_0204/knife_spread_tomatosauce|greenknife",
    "videos_0204/spatula_flip_egg|bigwoodenspatula",
    "videos_0204/spatula_flip_egg|redrubberspatula",
    "videos_0204/spatula_flip_egg|curvedwoodenspatula",
    "videos_0204/spatula_spread_tomatosauce|curvedwoodenspatula",
    "videos_0209/knife_spread_tomatosauce|orangeknife",
    "videos_0209/knife_spread_tomatosauce|greenknife",
    "videos_0209/spatula_spread_tomatosauce|redrubberspatula",
    "videos_0209/spoon_spread_tomatosauce|greenspoon",
    "videos_0209/spoon_spread_tomatosauce|bigwoodenspoon1",
    "videos_0210/fork_flip_naan|greenfork",
    "videos_0210/fork_flip_naan|bigwoodenfork",
    "videos_0211/spatula_flip_naan|bigwoodenspatula",
    "videos_0211/spatula_flip_naan|redrubberspatula",
    "videos_0212/fork_spread_tomatosauce|bigwoodenfork",
    "videos_0213/fork_flip_egg|bigwoodenfork",
    "videos_0213/fork_flip_egg|smallwoodenfork",
    "videos_0213/spatula_flip_naan|redrubberspatula",
    "videos_0213/spatula_press_sponge|redrubberspatula",
    "videos_0213/spatula_press_sponge|curvedwoodenspatula",
    "videos_0216/fork_scoop_nuts|greenfork",
    "videos_0216/fork_scoop_nuts|largeodenfork",
    "videos_0216/fork_scoop_nuts|smallwoodenfork",
    "videos_0216/ladle_scoop_nuts|ladle",
    "videos_0216/ladle_scoop_nuts|ladle_scoop_smallamount_peanuts_nuts_in_mediumshallowcontainer",
    "videos_0218/fork_dig_toysand|greenfork",
    "videos_0218/fork_dig_toysand|bigwoodenfork2",
    "videos_0218/fork_dig_toysand|bigwoodenfork_dig_toy",
    "videos_0218/fork_dig_toysand|bigwoodenfork",
    "videos_0218/rollingpin_roll_sand|rollingpin_small",
    "videos_0218/rollingpin_roll_sand|rollingpin_roll_small_sand_in_cuttingboard",
    "videos_0218/rollingpin_roll_sand|rollingpin_roll_large_sand_in_cuttingboard",
    "videos_0218/rollingpin_roll_sand|rollingpin_large",
    "videos_0218/rollingpin_roll_sand|rollingpin2",
    "videos_0218/rollingpin_roll_sand|rollingpin",
    "videos_0218/scooper_dig_toysand|icecreamscooper2",
    "videos_0218/scooper_dig_toysand|openscooper",
    "videos_0218/scooper_dig_toysand|icecreamscooper",
    "videos_0218/scooper_dig_toysand|icecreamscooper_dig_toy",
    "videos_0218/scooper_dig_toysand|openscooper_dig_toy",
    "videos_0218/scooper_dig_toysand|openscooper2",
    "videos_0218/spoon_dig_toysand|greenspoon",
    "videos_0218/spoon_dig_toysand|bigwoodenspoon",
    "videos_0218/spoon_dig_toysand|greenspoon_dig_toy",
    "videos_0220/fork_spread_tomatosauce|greenfork",
    "videos_0220/knife_open_box|greyplasticknife_open_cardboard_box_bluetape",
    "videos_0220/knife_open_box|pinkknife_open_cardboard_box_bluetape",
    "videos_0220/knife_open_box|greyplasticknife2",
    "videos_0220/knife_open_box|pinkknife",
    "videos_0220/knife_open_box|orangeknife",
    "videos_0220/knife_open_box|greenknife",
    "videos_0220/knife_open_box|greyplasticknife",
    "videos_0222/fork_shape_kineticsand|greenfork",
    "videos_0222/fork_shape_kineticsand|bigwoodenfork",
    "videos_0222/knifeblade_open_box|whiteknifeblade",
    "videos_0222/scooper_shape_kineticsand|icecreamscooper",
    "videos_0222/scooper_shape_kineticsand|measuringscooper",
    "videos_0227/book_push_toy|book",
    "videos_0227/book_push_toy|book_leftup",
    "videos_0227/book_push_toy|book_push_toy_rightup_faraway",
    "videos_0227/book_push_toy|book_rightup",
    "videos_0227/book_push_toy|book_push_toy_leftup_faraway",
    "videos_0227/knifeblade_open_box|greenknifeblade",
    "videos_0227/knifeblade_open_box|whiteknifeblade",
    "videos_0227/mallet_push_toy|mallet_rightup",
    "videos_0227/mallet_push_toy|mallet_push_toy_leftup_faraway",
    "videos_0227/mallet_push_toy|mallet",
    "videos_0227/mallet_push_toy|mallet_leftup",
    "videos_0227/mallet_push_toy|mallet_push_toy_rightup_faraway",
    "videos_0227/pan_push_toy|whitean_push_toy_leftup_faraway",
    "videos_0227/pan_push_toy|blackpan",
    "videos_0227/pan_push_toy|blackpan_push_toy_leftup_faraway",
    "videos_0227/pan_push_toy|whitepan_push_toy_rightup_faraway",
    "videos_0227/pan_push_toy|whitepan_rightup",
    "videos_0227/pan_push_toy|blackpan_leftup",
    "videos_0227/pan_push_toy|whitepan",
    "videos_0227/pan_push_toy|whitean",
    "videos_0304/malletpestle_crush_nuts|pestle",
    "videos_0304/scooper_scoop_icecream|icecreamscooper",
    "videos_0304/scooper_scoop_icecream|measuringscooper",
    "videos_0304/spatula_lift_place_egg|redrubberspatula",
    "videos_0304/spatula_lift_place_egg|bigwoodenspatula_lift_egg",
    "videos_0304/spoon_lift_place_egg|greenspoon",
    "videos_0304/spoon_lift_place_egg|bigwoodenspoon",
    "videos_0309/pestle_crush_nuts|pestle",
    "videos_0309/spoon_dig_plantingpot|greenspoon",
    "videos_0309/spoon_dig_plantingpot|smallwoodenspoon",
    "videos_0309/spoon_dig_plantingpot|yellowplasticspoon",
    "videos_0309/spoon_dig_plantingpot|bigwoodenspoon",
]


def load_matcher(task: str, keyword: str) -> dict[str, ColorMatcher] | None:
    slug = task.replace("/", "_")
    meta_path = MODEL_DIR / slug / f"{keyword}.meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    matchers = {}
    for role, role_meta in meta.get("roles", {}).items():
        npz_path = MODEL_DIR / slug / role_meta["npz"]
        if npz_path.exists():
            matchers[role] = ColorMatcher(npz_path, role_meta)
    return matchers if matchers else None


def get_image_size(task: str, exp: str, cam: str) -> tuple[int, int] | None:
    """从 embed.npz 推断图像尺寸，或从 jpg 读。"""
    # 先找 kf0 的 embed
    embed_glob = list((EMBED_ROOT / task / exp).glob(f"{cam}.kf0.embed.npz"))
    if not embed_glob:
        embed_glob = list((EMBED_ROOT / task / exp).glob(f"{cam}.kf*.embed.npz"))
    if embed_glob:
        z = np.load(str(embed_glob[0]))
        # embed 的 high_res_feats 是 (C, fH, fW)，需从 jpg 读原始尺寸
        jpg = embed_glob[0].with_suffix("").with_suffix(".jpg")
        jpg_path = EMBED_ROOT / task / exp / f"{cam}.kf0.jpg"
        if not jpg_path.exists():
            # 找任意 jpg
            jpgs = list((EMBED_ROOT / task / exp).glob(f"{cam}.kf*.jpg"))
            if jpgs:
                jpg_path = jpgs[0]
        if jpg_path.exists():
            import cv2
            img = cv2.imread(str(jpg_path))
            if img is not None:
                return img.shape[0], img.shape[1]  # H, W
    return None


def process_json(json_path: Path, matchers: dict[str, ColorMatcher],
                 task: str, exp: str, dry_run: bool) -> int:
    """为该 JSON 里 frame_index=0 且无 bbox 的 entry 补写 bbox。返回写入数量。"""
    cam = json_path.stem  # e.g. cam0_rgb
    data = json.loads(json_path.read_text())
    changed = 0

    hw = get_image_size(task, exp, cam)
    if hw is None:
        return 0
    H, W = hw

    for obj in data.get("objects", []):
        if obj.get("frame_index") != 0:
            continue
        if "bbox" in obj:
            continue
        role = obj.get("role", "primary_tool")
        matcher = matchers.get(role)
        if matcher is None:
            continue
        pts = obj.get("points", [])
        lbls = obj.get("labels", [])
        pos_pts = [p for p, l in zip(pts, lbls) if l == 1]
        if not pos_pts:
            continue
        cx = float(np.mean([p[0] for p in pos_pts]))
        cy = float(np.mean([p[1] for p in pos_pts]))
        bbox = matcher.bbox_estimate(cx, cy, H, W)
        if bbox is None:
            continue
        obj["bbox"] = bbox
        changed += 1

    if changed > 0 and not dry_run:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return changed


def run_one(task: str, keyword: str, dry_run: bool) -> tuple[int, int]:
    matchers = load_matcher(task, keyword)
    if not matchers:
        print(f"  [SKIP] no color model: {task}|{keyword}")
        return 0, 0

    ap_task = AP_ROOT / task
    if not ap_task.exists():
        return 0, 0

    total_exp = 0
    total_written = 0
    for exp_dir in sorted(ap_task.iterdir()):
        if not exp_dir.is_dir():
            continue
        if keyword not in exp_dir.name:
            continue
        prompt_dir = exp_dir / "tool_masks" / "prompts"
        if not prompt_dir.exists():
            continue
        for jf in sorted(prompt_dir.glob("cam*_rgb.json")):
            n = process_json(jf, matchers, task, exp_dir.name, dry_run)
            total_written += n
        total_exp += 1

    return total_exp, total_written


def main():
    global AP_ROOT, EMBED_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="")
    ap.add_argument("--keyword", default="")
    ap.add_argument("--all", action="store_true", help="处理所有 Feb+ task/keyword")
    ap.add_argument("--ap-root", default=str(AP_ROOT))
    ap.add_argument("--embed-root", default=str(EMBED_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    AP_ROOT = Path(args.ap_root)
    EMBED_ROOT = Path(args.embed_root)

    jobs = []
    if args.all:
        jobs = [(j.split("|")[0], j.split("|")[1]) for j in FEB_JOBS]
    elif args.task and args.keyword:
        jobs = [(args.task, args.keyword)]
    else:
        ap.print_help()
        return

    total_exp = total_written = 0
    for task, keyword in jobs:
        n_exp, n_written = run_one(task, keyword, args.dry_run)
        if n_written > 0 or n_exp > 0:
            print(f"[{task}|{keyword}] exps={n_exp}  bbox_added={n_written}"
                  + (" [DRY]" if args.dry_run else ""))
        total_exp += n_exp
        total_written += n_written

    print(f"\n完成: {len(jobs)} pairs, {total_exp} exps, {total_written} bbox 写入")


if __name__ == "__main__":
    main()
