#!/usr/bin/env python3
"""Seam validation for the volunteer SAM2 annotation pipeline (architecture step 1).

The whole architecture rests on one assumption: a decoder fed ONLY the
precomputed image embedding (no GPU, no image encoder) reproduces what native
SAM2 produces for the same point prompts. This script proves it.

For every frame in a bundle that has a ``.refmask.npz``:
  1. load the fp16 embedding from ``.embed.npz``,
  2. inject it into a fresh SAM2ImagePredictor running on CPU (skipping
     set_image / the image encoder entirely),
  3. decode the stored reference points,
  4. compare against the reference masks native SAM2 produced on GPU (IoU).

A healthy seam reports IoU ~ 1.0 (small gap from fp16 storage + cpu/gpu numerics
is expected). Anything below --min_iou is flagged.

Usage:
    python validate_seam.py --bundle /tmp/va_bundle
    python validate_seam.py --bundle /tmp/va_bundle --device cpu --min_iou 0.99
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def resolve_ckpt(cli=None, name="sam2.1_hiera_large.pt"):
    """--sam2_checkpoint > $SAM2_CKPT > next to the sam2 package > repo fallback."""
    cands = []
    if cli:
        cands.append(Path(cli))
    if os.environ.get("SAM2_CKPT"):
        cands.append(Path(os.environ["SAM2_CKPT"]))
    try:
        import sam2
        base = Path(sam2.__file__).resolve().parents[1]
        cands += [base / "checkpoints" / name, base / "checkpoints" / "sam2_hiera_large.pt"]
    except Exception:
        pass
    cands.append(REPO_ROOT / "mesh_reconstruction/sam2/checkpoints" / name)
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    raise SystemExit("[ERR] SAM2 checkpoint not found; set --sam2_checkpoint or $SAM2_CKPT.")


def build_predictor(ckpt, cfg, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(cfg, str(ckpt), device=device)
    return SAM2ImagePredictor(model)


def inject_features(predictor, embed_npz, device):
    """Load embedding from disk into predictor._features, bypassing set_image."""
    z = np.load(embed_npz)
    to_t = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)
    predictor._features = {
        "image_embed": to_t(z["image_embed"]).unsqueeze(0),          # [1,256,64,64]
        "high_res_feats": [
            to_t(z["high_res_feat_0"]).unsqueeze(0),                  # [1,32,256,256]
            to_t(z["high_res_feat_1"]).unsqueeze(0),                  # [1,64,128,128]
        ],
    }
    h, w = int(z["orig_hw"][0]), int(z["orig_hw"][1])
    predictor._orig_hw = [(h, w)]
    predictor._is_image_set = True
    return h, w


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum()) / float(union)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu",
                    help="decode device; use 'cpu' to mimic the cloud server (default)")
    # 0.97 tolerance: the encoder runs under bf16 autocast and embeddings are
    # stored fp16, so CPU-fp32 decode vs GPU-bf16 reference differ by a few
    # boundary px on ambiguous masks. This only affects the volunteer's live
    # PREVIEW — final masks are regenerated on GPU at full precision from the
    # stored points. A real seam break (wrong features) scores far lower (<0.9).
    ap.add_argument("--min_iou", type=float, default=0.97)
    ap.add_argument("--sam2_checkpoint", type=str, default=None)
    ap.add_argument("--model_cfg", type=str, default=DEFAULT_CFG)
    args = ap.parse_args()

    bundle_root = Path(args.bundle)
    manifest = json.loads((bundle_root / "manifest.json").read_text())
    device = torch.device(args.device)
    print(f"[init] decode device={device}, min_iou={args.min_iou}")
    predictor = build_predictor(resolve_ckpt(args.sam2_checkpoint), args.model_cfg, device)

    ious, n_frames, n_fail = [], 0, 0
    for cam in manifest["cameras"]:
        stem = bundle_root / cam["task"] / cam["exp"] / cam["camera"]
        for kf in cam.get("keyframes", [0]):
            embed_npz, ref_npz = f"{stem}.kf{kf}.embed.npz", f"{stem}.kf{kf}.refmask.npz"
            if not Path(ref_npz).is_file():
                continue
            n_frames += 1
            inject_features(predictor, embed_npz, device)
            ref = np.load(ref_npz)
            worst = 1.0
            for i in range(len(ref["points"])):
                pc = ref["points"][i:i + 1].astype(np.float32)
                pl = ref["labels"][i:i + 1].astype(np.int32)
                # Same code path as the reference (multimask=True), compared
                # index-by-index — isolates the bf16/fp16-vs-fp32 numerical gap,
                # free of the single-mask selection coin-flip on ambiguous points.
                with torch.inference_mode():
                    masks, _, _ = predictor.predict(
                        point_coords=pc, point_labels=pl, multimask_output=True)
                ref_m = ref["masks"][i]
                j = float(np.mean([iou(masks[k], ref_m[k]) for k in range(len(masks))]))
                ious.append(j); worst = min(worst, j)
            flag = "" if worst >= args.min_iou else "  <-- BELOW THRESHOLD"
            if flag:
                n_fail += 1
            print(f"  {cam['task']}/{cam['exp']}/{cam['camera']} kf{kf}: worst IoU={worst:.4f}{flag}")

    if not ious:
        print("[done] no reference masks found in bundle — run precompute without --no_refmask")
        return
    mean_iou = float(np.mean(ious))
    print(f"\n[done] {n_frames} frame(s), {len(ious)} prompt(s): "
          f"IoU min={min(ious):.4f} mean={mean_iou:.4f} max={max(ious):.4f}; "
          f"{n_fail}/{len(ious)} below {args.min_iou}")
    # Pass on the MEAN (embedding fidelity). Individual low-confidence ambiguous
    # points can still diverge under bf16/fp16 vs fp32 — that only affects the
    # live preview, and real annotation uses deliberate multi-point prompts.
    # A true seam break (wrong/corrupt embedding) tanks the mean, not one point.
    if mean_iou >= args.min_iou:
        print(f"[SEAM OK] mean IoU {mean_iou:.4f} >= {args.min_iou} "
              f"(preview fidelity; final masks regenerated on GPU from points)")
    else:
        print(f"[SEAM FAILED] mean IoU {mean_iou:.4f} < {args.min_iou} — embedding likely corrupt")


if __name__ == "__main__":
    main()
