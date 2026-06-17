#!/usr/bin/env python3
"""Stage (1) of the volunteer SAM2 annotation pipeline — runs on a GPU box.

For each experiment, per camera, export a LIGHT set of frames the cloud serves
without a GPU:
  * a few ANNOTATABLE keyframes (default frame 0 + a couple early frames):
      cam{c}_rgb.kf{idx}.embed.npz   fp16 image_embed + high-res feats + orig_hw
      cam{c}_rgb.kf{idx}.jpg         full-res frame for the annotation canvas
      cam{c}_rgb.kf{idx}.refmask.npz native SAM2 multimask (seam validation)
  * a couple of view-only BROWSE thumbnails (mid sequence, to identify the tool):
      cam{c}_rgb.th{idx}.jpg         downscaled jpg, NO embedding
plus a top-level manifest.json indexing every camera's keyframe/thumb indices.

Keyframes are early on purpose: the tool is usually picked up near the start, so
a volunteer can annotate it on the first frame where it's in hand (the prompt's
frame_index), and prompts_to_masks propagates both directions from there.

Usage:
    python precompute_embeddings.py --exp <dir> --bundle <b>
    python precompute_embeddings.py --all /data/robotool/videos_0102 --bundle <b>
    python precompute_embeddings.py --all <root> --bundle <b> \
        --keyframe_fracs 0,0.1,0.2 --thumb_fracs 0.4,0.6
"""

import argparse
import glob
import hashlib
import json
import os
import re
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_MESH_MAP = REPO_ROOT / "HO-Cap-Annotation/scripts/mesh_name_mapping.json"


def resolve_ckpt(cli=None, name="sam2.1_hiera_large.pt"):
    """Find the SAM2 checkpoint. Priority: --sam2_checkpoint > $SAM2_CKPT >
    next to the importable sam2 package (<sam2_repo>/checkpoints/) > this repo's
    mesh_reconstruction/sam2/checkpoints/. Avoids hardcoding a machine path."""
    cands = []
    if cli:
        cands.append(Path(cli))
    if os.environ.get("SAM2_CKPT"):
        cands.append(Path(os.environ["SAM2_CKPT"]))
    try:
        import sam2
        base = Path(sam2.__file__).resolve().parents[1]   # <sam2_repo>/sam2/__init__.py -> <sam2_repo>
        cands += [base / "checkpoints" / name, base / "checkpoints" / "sam2_hiera_large.pt"]
    except Exception:
        pass
    cands.append(REPO_ROOT / "mesh_reconstruction/sam2/checkpoints" / name)
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    raise SystemExit("[ERR] SAM2 checkpoint not found. Set --sam2_checkpoint or "
                     "$SAM2_CKPT. Tried:\n  " + "\n  ".join(str(c) for c in cands))

H5_NAME = "data00000000.h5"
SCHEMA_VERSION = 2
THUMB_W = 360  # browse thumbnail width
FRAGMENT = "_manifest.json"   # per-exp manifest fragment + done marker

REF_POINT_FRACS = [(0.50, 0.50), (0.35, 0.45), (0.62, 0.55)]


def load_mesh_map(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else {}


def derive_primary_name(exp_name, mesh_map):
    hay = exp_name.lower()
    hits = [(k, v) for k, v in mesh_map.items() if k.lower() in hay]
    if hits:
        hits.sort(key=lambda kv: -len(kv[0]))
        return hits[0][1]
    return re.sub(r"_\d+$", "", exp_name)


def build_predictor(ckpt, cfg, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2(cfg, str(ckpt), device=device))


def cam_name(c):
    return f"cam{c}_rgb"


def task_exp_from_path(exp_dir):
    exp_dir = Path(exp_dir).resolve()
    return exp_dir.parent.name, exp_dir.name


def fracs_to_idx(fracs, n):
    return sorted({max(0, min(int(round(f * (n - 1))), n - 1)) for f in fracs})


def video_info(exp_dir):
    """(n_frames, n_cams, kind) from data00000000.h5 or raw cam*_rgb.mp4."""
    exp_dir = Path(exp_dir)
    if (exp_dir / H5_NAME).is_file():
        with h5py.File(exp_dir / H5_NAME, "r") as f:
            s = f["imgs"].shape
        return s[0], s[1], "h5"
    mp4s = sorted(glob.glob(str(exp_dir / "cam*_rgb.mp4")))
    cap = cv2.VideoCapture(mp4s[0])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n, len(mp4s), "mp4"


def read_frames(exp_dir, kind, cam_index, idxs):
    """{idx: rgb HxWx3 uint8} for the requested frame indices of one camera."""
    exp_dir = Path(exp_dir)
    out = {}
    if kind == "h5":
        with h5py.File(exp_dir / H5_NAME, "r") as f:
            imgs = f["imgs"]
            for i in idxs:
                out[i] = np.ascontiguousarray(imgs[i, cam_index])
    else:
        cap = cv2.VideoCapture(str(exp_dir / f"cam{cam_index}_rgb.mp4"))
        for i in sorted(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {i} from cam{cam_index}")
            out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        cap.release()
    return out


def dump_embedding(predictor, out_path):
    feats = predictor._features
    image_embed = feats["image_embed"][0].float().cpu().numpy().astype(np.float16)
    hrf = [f[0].float().cpu().numpy().astype(np.float16) for f in feats["high_res_feats"]]
    h, w = predictor._orig_hw[0]
    np.savez_compressed(out_path, image_embed=image_embed,
                        high_res_feat_0=hrf[0], high_res_feat_1=hrf[1],
                        orig_hw=np.array([h, w], dtype=np.int32))


def dump_reference_masks(predictor, h, w, out_path):
    """Native SAM2 multimask (3 candidates) on fixed points; for seam validation."""
    pts, labels, masks, scores = [], [], [], []
    for fx, fy in REF_POINT_FRACS:
        pc = np.array([[fx * w, fy * h]], dtype=np.float32)
        pl = np.array([1], dtype=np.int32)
        m, s, _ = predictor.predict(point_coords=pc, point_labels=pl, multimask_output=True)
        pts.append(pc[0]); labels.append(pl[0])
        masks.append(m.astype(np.uint8)); scores.append(s.astype(np.float32))
    np.savez_compressed(out_path,
                        points=np.array(pts, dtype=np.float32),
                        labels=np.array(labels, dtype=np.int32),
                        masks=np.stack(masks, 0), scores=np.stack(scores, 0))


def process_exp(predictor, exp_dir, bundle_root, mesh_map, kf_fracs, th_fracs,
                want_refmask, device, sam2_model, skip_existing=False, force=False):
    exp_dir = Path(exp_dir)
    task, exp = task_exp_from_path(exp_dir)
    out_dir = bundle_root / task / exp
    frag_path = out_dir / FRAGMENT
    if skip_existing and not force and frag_path.is_file():
        print(f"  [skip] {task}/{exp} (done: {FRAGMENT} present)")
        return "skipped"
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = derive_primary_name(exp, mesh_map)
    n_frames, n_cams, kind = video_info(exp_dir)
    kf_idxs = fracs_to_idx(kf_fracs, n_frames)
    th_idxs = fracs_to_idx(th_fracs, n_frames)
    print(f"  {task}/{exp} [{kind}]: {n_frames}f x {n_cams}cam | primary={primary} "
          f"| keyframes={kf_idxs} thumbs={th_idxs}")
    entries = []

    for c in range(n_cams):
        t0 = time.time()
        frames = read_frames(exp_dir, kind, c, set(kf_idxs) | set(th_idxs))
        stem = out_dir / cam_name(c)
        H = W = None
        for idx in kf_idxs:
            img = frames[idx]; H, W = img.shape[:2]
            with torch.inference_mode(), torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                predictor.set_image(img)
                dump_embedding(predictor, f"{stem}.kf{idx}.embed.npz")
                if want_refmask:
                    dump_reference_masks(predictor, H, W, f"{stem}.kf{idx}.refmask.npz")
            cv2.imwrite(f"{stem}.kf{idx}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        for idx in th_idxs:
            img = frames[idx]
            th = cv2.resize(img, (THUMB_W, int(THUMB_W * img.shape[0] / img.shape[1])))
            cv2.imwrite(f"{stem}.th{idx}.jpg", cv2.cvtColor(th, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
        entries.append({
            "task": task, "exp": exp, "camera": cam_name(c), "cam_index": c,
            "width": int(W), "height": int(H), "n_frames": int(n_frames),
            "primary_name": primary, "keyframes": kf_idxs, "thumbs": th_idxs,
            "image_sha1": hashlib.sha1(frames[kf_idxs[0]].tobytes()).hexdigest(),
        })
        print(f"    {cam_name(c)}: {len(kf_idxs)} kf + {len(th_idxs)} thumb in {time.time()-t0:.2f}s")
    # per-exp manifest fragment = this exp's cameras + a "done" marker (written
    # LAST so an interrupted exp has no fragment and is re-done on resume).
    frag_path.write_text(json.dumps(
        {"sam2_model": sam2_model, "embed_dtype": "float16", "cameras": entries}, indent=2))
    return entries


def discover_exps(root):
    root = Path(root)
    exps = {p.parent for p in root.rglob(H5_NAME)}
    exps |= {p.parent for p in root.rglob("cam0_rgb.mp4")}
    return sorted(exps)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--exp", type=str)
    src.add_argument("--all", type=str)
    ap.add_argument("--bundle", type=str, required=True)
    ap.add_argument("--keyframe_fracs", type=str, default="0,0.1,0.2",
                    help="comma frac positions for annotatable keyframes (embed+jpg)")
    ap.add_argument("--thumb_fracs", type=str, default="0.4,0.6",
                    help="comma frac positions for view-only browse thumbnails (jpg)")
    ap.add_argument("--mesh_name_map", type=str, default=str(DEFAULT_MESH_MAP))
    ap.add_argument("--no_refmask", action="store_true")
    ap.add_argument("--shard", type=str, default=None,
                    help="k/N: with --all, process only exps where index%%N==k (parallel sharding)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="resume: skip exps whose <exp>/_manifest.json already exists")
    ap.add_argument("--no_merge", action="store_true",
                    help="don't rebuild top-level manifest.json (parallel workers; merge once at end)")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if <exp>/_manifest.json exists (overrides --skip_existing)")
    ap.add_argument("--sam2_checkpoint", type=str, default=None,
                    help="SAM2 .pt; default auto-resolves ($SAM2_CKPT or next to the sam2 package)")
    ap.add_argument("--model_cfg", type=str, default=DEFAULT_CFG)
    args = ap.parse_args()

    bundle_root = Path(args.bundle); bundle_root.mkdir(parents=True, exist_ok=True)
    kf_fracs = [float(x) for x in args.keyframe_fracs.split(",") if x.strip() != ""]
    th_fracs = [float(x) for x in args.thumb_fracs.split(",") if x.strip() != ""]
    mesh_map = load_mesh_map(args.mesh_name_map)
    want_refmask = not args.no_refmask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ckpt = resolve_ckpt(args.sam2_checkpoint)
    print(f"[init] device={device}, ckpt={ckpt}, keyframe_fracs={kf_fracs}, "
          f"thumb_fracs={th_fracs}, refmask={want_refmask}, mesh map={len(mesh_map)} entries")
    predictor = build_predictor(ckpt, args.model_cfg, device)

    exps = [Path(args.exp)] if args.exp else discover_exps(args.all)
    if args.shard and args.all:
        k, n = (int(x) for x in args.shard.split("/"))
        exps = [e for i, e in enumerate(exps) if i % n == k]
        print(f"[shard] {k}/{n}: {len(exps)} of this shard")
    print(f"[init] {len(exps)} experiment(s)")
    sam2_model = Path(ckpt).name
    t_start = time.time()
    n_done = n_skip = n_cams = 0
    for exp_dir in exps:
        r = process_exp(predictor, exp_dir, bundle_root, mesh_map, kf_fracs, th_fracs,
                        want_refmask, device, sam2_model, args.skip_existing, args.force)
        if r == "skipped":
            n_skip += 1
        else:
            n_done += 1; n_cams += len(r)
    msg = f"[done] {n_done} exp(s) processed, {n_skip} skipped, {n_cams} cameras ({time.time()-t_start:.1f}s)"
    if not args.no_merge:
        from merge_manifests import build_manifest
        nf, nc, ne = build_manifest(bundle_root)
        msg += f" | manifest.json: {nc} cameras / {ne} exps"
    print(msg)


if __name__ == "__main__":
    main()
