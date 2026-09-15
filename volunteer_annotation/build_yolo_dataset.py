#!/usr/bin/env python3
"""
Convert SAM2 point-prompt annotations → YOLO segmentation dataset.

For each (exp, cam, frame): loads prompt JSON, decodes via SAM2, saves
  images/<split>/<task>_<exp>_<cam>_<frame>.jpg
  labels/<split>/<task>_<exp>_<cam>_<frame>.txt   (YOLO polygon format)

Usage:
  python build_yolo_dataset.py \
    --tasks videos_0204/fork_flip_egg videos_0213/fork_flip_egg \
    --role primary_tool \
    --out /tmp/yolo_fork_flip_egg \
    --max-exps 100 \
    --val-ratio 0.2
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

BUNDLE      = Path("/data/robotool/_va_bundle_v2")
AUTO_PROMPTS = Path("/data/robotool/_va_bundle_v2_auto_prompts")
HUM_PROMPTS  = Path("/data/robotool/_va_bundle_v2_prompts")

SAM2_SCORE_MIN = 0.75
MASK_MIN_PX    = 200


# ── mask → YOLO polygon ────────────────────────────────────────────────────────

def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.002) -> list[list[float]] | None:
    """
    Binary mask → list of YOLO polygon segments (each a flat [x,y,x,y,...] normalised list).
    Returns None if no valid contour found.
    """
    H, W = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep only contours with area > 50px
    contours = [c for c in contours if cv2.contourArea(c) > 50]
    if not contours:
        return None

    segments = []
    for c in contours:
        eps = epsilon_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        pts = approx.astype(float)
        pts[:, 0] /= W
        pts[:, 1] /= H
        segments.append(pts.flatten().tolist())
    return segments if segments else None


# ── dataset builder ────────────────────────────────────────────────────────────

def collect_samples(tasks: list[str], role: str, max_exps: int | None,
                    prompts_root: Path) -> list[dict]:
    """
    Scan prompt JSONs and return list of sample dicts:
      {task, exp, cam, frame, embed_path, img_path, points, labels}
    """
    samples = []
    for task in tasks:
        prompt_dir = prompts_root / task
        if not prompt_dir.exists():
            print(f"  [skip] no prompts: {task}")
            continue

        exps = sorted(prompt_dir.iterdir())
        if max_exps:
            exps = exps[:max_exps]

        for exp_dir in exps:
            exp = exp_dir.name
            prompt_sub = exp_dir / "tool_masks" / "prompts"
            if not prompt_sub.exists():
                continue

            for pf in sorted(prompt_sub.glob("*.json")):
                cam = pf.stem  # e.g. cam0_rgb
                try:
                    data = json.loads(pf.read_text())
                except Exception:
                    continue

                for obj in data.get("objects", []):
                    if obj.get("role") != role:
                        continue
                    frame = obj.get("frame_index")
                    pts   = obj.get("points")
                    lbls  = obj.get("labels")
                    if frame is None or not pts or not lbls:
                        continue

                    embed_path = BUNDLE / task / exp / f"{cam}.kf{frame}.embed.npz"
                    img_path   = BUNDLE / task / exp / f"{cam}.kf{frame}.jpg"
                    if not embed_path.exists() or not img_path.exists():
                        continue

                    samples.append({
                        "task": task, "exp": exp, "cam": cam, "frame": frame,
                        "embed_path": embed_path, "img_path": img_path,
                        "points": pts, "labels": lbls,
                    })
    return samples


def build(args):
    from cloud.decoder import Sam2CpuDecoder

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    tasks = args.tasks

    # Prefer human annotations; supplement with auto
    print(f"Scanning prompts for {len(tasks)} task(s), role={args.role} ...")
    samples = collect_samples(tasks, args.role, args.max_exps, HUM_PROMPTS)
    print(f"  Human annotations: {len(samples)}")
    auto = collect_samples(tasks, args.role, args.max_exps, AUTO_PROMPTS)
    # De-duplicate by (task, exp, cam, frame)
    existing = {(s["task"], s["exp"], s["cam"], s["frame"]) for s in samples}
    for s in auto:
        if (s["task"], s["exp"], s["cam"], s["frame"]) not in existing:
            samples.append(s)
    print(f"  Total after auto supplement: {len(samples)}")

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    splits = {"val": samples[:n_val], "train": samples[n_val:]}

    print("Loading SAM2 decoder ...")
    decoder = Sam2CpuDecoder()

    class_name = args.role.replace("_", "-")  # e.g. primary-tool
    class_id   = 0

    stats = {"ok": 0, "bad_score": 0, "bad_mask": 0, "no_contour": 0, "error": 0}

    for split, split_samples in splits.items():
        print(f"\n[{split}] {len(split_samples)} samples ...")
        for i, s in enumerate(split_samples):
            stem = f"{s['task'].replace('/', '_')}_{s['exp']}_{s['cam']}_f{s['frame']}"
            img_dst   = out / "images" / split / f"{stem}.jpg"
            label_dst = out / "labels" / split / f"{stem}.txt"

            if img_dst.exists() and label_dst.exists() and not args.force:
                stats["ok"] += 1
                continue

            # Decode mask
            try:
                mask, score = decoder.infer(s["embed_path"], s["points"], s["labels"])
            except Exception as e:
                stats["error"] += 1
                continue

            if score < SAM2_SCORE_MIN:
                stats["bad_score"] += 1
                continue
            if int(mask.sum()) < MASK_MIN_PX:
                stats["bad_mask"] += 1
                continue

            segs = mask_to_polygon(mask)
            if not segs:
                stats["no_contour"] += 1
                continue

            # Copy image
            shutil.copy2(s["img_path"], img_dst)

            # Write YOLO label: class_id x1 y1 x2 y2 ...
            with open(label_dst, "w") as f:
                for seg in segs:
                    coords = " ".join(f"{v:.6f}" for v in seg)
                    f.write(f"{class_id} {coords}\n")

            stats["ok"] += 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(split_samples)}  ok={stats['ok']}")

    print(f"\nDone: {stats}")

    # Write dataset YAML
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(yaml.dump({
        "path": str(out.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": [class_name],
    }))
    print(f"Dataset YAML: {yaml_path}")
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="Task paths, e.g. videos_0204/fork_flip_egg")
    ap.add_argument("--role", default="primary_tool",
                    help="Annotation role to extract (default: primary_tool)")
    ap.add_argument("--out", required=True, help="Output dataset directory")
    ap.add_argument("--max-exps", type=int, default=None,
                    help="Max experiments per task (for quick tests)")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
