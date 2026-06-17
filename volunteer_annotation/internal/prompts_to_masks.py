#!/usr/bin/env python3
"""Stage (4) of the volunteer SAM2 annotation pipeline — runs on a GPU box.

Reads the volunteer point prompts (``tool_masks/prompts/<camera>.json``) for one
experiment, replays them through the SAM2 *video* predictor on the full
``data00000000.h5`` sequence, propagates to every frame, and writes the result
as a downstream-compatible label ``masks.h5`` plus ``objects.yaml``.

Output is byte-compatible with the existing pipeline (generate_meta.py /
my_cluster_loader.py): ``masks.h5`` dataset ``"masks"`` of shape
``(N_frames, N_cams, H, W)`` uint8, where pixel value 0 = background and k =
object k (1-indexed); a per-object binary mask is ``mask == (object_idx + 1)``.

This replaces the interactive ``batch_task_annotator_multi.py`` step.

Usage:
    python prompts_to_masks.py --exp ../../../DataCollection/data/0320_1/cube_small_1
    python prompts_to_masks.py --exp <dir> --max_frames 30   # quick test
"""

import argparse
import gc
import glob
import json
import tempfile
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT = REPO_ROOT / "mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt"
# hocap's build_sam2_video_predictor expects a real config FILE path (it uses
# the parent dir as the hydra config dir + the stem as config name).
DEFAULT_CFG = REPO_ROOT / "HO-Cap-Annotation/config/sam2_config/sam2.1_hiera_l.yaml"

H5_NAME = "data00000000.h5"

# Fixed semantic role order -> contiguous masks.h5 object_id. Primary, when
# present, is always object_id 1 (the object the hand-object joint optimization
# uses). object_id is assigned per-exp from the UNION of roles present across
# all cameras, so the same physical object keeps one id in every view.
ROLE_ORDER = ["primary_tool", "auxiliary_tool", "manipulated_object"]


def build_video_predictor(ckpt, cfg, device):
    from hocap_annotation.wrappers.sam2 import build_sam2_video_predictor
    return build_sam2_video_predictor(cfg, str(ckpt), device=device)


def cam_idx_from_name(cam):
    """'cam0_rgb' -> 0."""
    return int("".join(ch for ch in cam.replace("cam", "").replace("_rgb", "") if ch.isdigit()))


def video_meta(exp_dir):
    """(n_frames, n_cams, H, W, kind) from data00000000.h5 or raw cam*_rgb.mp4."""
    exp_dir = Path(exp_dir)
    if (exp_dir / H5_NAME).is_file():
        with h5py.File(exp_dir / H5_NAME, "r") as f:
            s = f["imgs"].shape                       # (N, C, H, W, 3)
        return s[0], s[1], s[2], s[3], "h5"
    mp4s = sorted(glob.glob(str(exp_dir / "cam*_rgb.mp4")))
    cap = cv2.VideoCapture(mp4s[0])
    n, W, H = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
               int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return n, len(mp4s), H, W, "mp4"


def extract_camera_frames(exp_dir, cam_index, out_dir, kind, max_frames=None):
    """Dump a camera's frames to out_dir as jpgs; return sorted path list.
    JPEGs are written in correct colors so SAM2's loader reads RGB back."""
    exp_dir = Path(exp_dir)
    paths = []
    if kind == "h5":
        with h5py.File(exp_dir / H5_NAME, "r") as f:
            imgs = f["imgs"]                          # RGB
            n = imgs.shape[0] if max_frames is None else min(max_frames, imgs.shape[0])
            for i in range(n):
                p = out_dir / f"{i:05d}.jpg"
                cv2.imwrite(str(p), cv2.cvtColor(np.ascontiguousarray(imgs[i, cam_index]),
                                                 cv2.COLOR_RGB2BGR))
                paths.append(str(p))
    else:
        cap = cv2.VideoCapture(str(exp_dir / f"cam{cam_index}_rgb.mp4"))
        i = 0
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()                    # BGR
            if not ok:
                break
            p = out_dir / f"{i:05d}.jpg"
            cv2.imwrite(str(p), frame)                # already BGR -> correct colors
            paths.append(str(p))
            i += 1
        cap.release()
    return paths


def open_mask_writer(h5_path, n_frames, n_cams, H, W):
    """Lazily allocate masks.h5 matching the existing pipeline layout."""
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(h5_path, "a")
    if "masks" in f:
        if f["masks"].shape != (n_frames, n_cams, H, W):
            del f["masks"]
    if "masks" not in f:
        f.create_dataset("masks", shape=(n_frames, n_cams, H, W), dtype=np.uint8,
                         chunks=(1, 1, H, W), compression="gzip")
    return f


def _logit_to_mask(logit, H, W):
    m = (logit[0] > 0.0).cpu().numpy()
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def process_camera(predictor, exp_dir, kind, H, W, cam, objects, role_to_id,
                   mask_ds, device, max_frames):
    """Propagate one camera's prompts and write its label masks into mask_ds.

    Each object may be prompted on its OWN frame (obj['frame_index'], default 0)
    — e.g. the tool annotated on a later frame where it's actually in hand. To
    cover the whole clip we propagate FORWARD (from the earliest prompt) and
    BACKWARD (from the latest prompt); the forward label wins, the reverse pass
    only fills frames an object hadn't reached yet."""
    cam_index = cam_idx_from_name(cam)
    with tempfile.TemporaryDirectory() as td:
        img_paths = extract_camera_frames(exp_dir, cam_index, Path(td), kind, max_frames)
        n = len(img_paths)
        state = predictor.init_state(img_paths=img_paths,
                                     offload_video_to_cpu=True, offload_state_to_cpu=True)
        cond_frames = []
        for obj in objects:
            fi = max(0, min(int(obj.get("frame_index", 0)), n - 1))
            cond_frames.append(fi)
            pts = np.array(obj["points"], dtype=np.float32)
            lbl = np.array(obj["labels"], dtype=np.int32)
            box = np.array(obj["box"], dtype=np.float32) if obj.get("box") else None
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=fi, obj_id=role_to_id[obj["role"]],
                points=(pts if len(pts) else None),
                labels=(lbl if len(lbl) else None),
                box=box, normalize_coords=True,
            )
        single_frame = len(set(cond_frames)) == 1 and cond_frames[0] == 0

        written = set()
        # forward pass (from earliest prompt to end): authoritative label
        for frame_idx, obj_ids, logits in predictor.propagate_in_video(state, reverse=False):
            label = np.zeros((H, W), dtype=np.uint8)
            for oid, lg in sorted(zip(obj_ids, logits), key=lambda t: t[0]):
                label[_logit_to_mask(lg, H, W)] = int(oid)
            mask_ds[frame_idx, cam_index] = label
            written.add(int(frame_idx))
        # backward pass (from latest prompt to 0): fill frames forward didn't cover
        if not single_frame:
            for frame_idx, obj_ids, logits in predictor.propagate_in_video(
                    state, reverse=True, start_frame_idx=max(cond_frames)):
                cur = mask_ds[frame_idx, cam_index] if int(frame_idx) in written \
                    else np.zeros((H, W), dtype=np.uint8)
                for oid, lg in sorted(zip(obj_ids, logits), key=lambda t: t[0]):
                    m = _logit_to_mask(lg, H, W)
                    cur[m & (cur == 0)] = int(oid)   # don't overwrite forward result
                mask_ds[frame_idx, cam_index] = cur
                written.add(int(frame_idx))
        del state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return len(written)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=str, required=True,
                    help="experiment dir (data00000000.h5 or cam*_rgb.mp4, + tool_masks/prompts/)")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="propagate only the first N frames (quick test)")
    ap.add_argument("--sam2_checkpoint", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument("--model_cfg", type=str, default=str(DEFAULT_CFG))
    args = ap.parse_args()

    exp_dir = Path(args.exp)
    prompts_dir = exp_dir / "tool_masks" / "prompts"
    prompt_files = sorted(prompts_dir.glob("*.json"))
    if not prompt_files:
        raise SystemExit(f"no prompt JSON in {prompts_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}, {len(prompt_files)} camera prompt file(s)")
    predictor = build_video_predictor(args.sam2_checkpoint, args.model_cfg, device)

    # Consolidate roles across all cameras: a role present in ANY view is part
    # of the exp roster. Assign contiguous object_ids in fixed role order so the
    # same physical object keeps one id in every camera.
    role_name = {}
    for pf in prompt_files:
        for o in json.loads(pf.read_text())["objects"]:
            role_name.setdefault(o["role"], o.get("name", o["role"]))
    present_roles = [r for r in ROLE_ORDER if r in role_name]
    role_to_id = {r: i + 1 for i, r in enumerate(present_roles)}
    objects_list = [role_name[r] for r in present_roles]
    n_objects = len(present_roles)
    print(f"  roles -> object_id: " +
          ", ".join(f"{r}={role_to_id[r]}({role_name[r]})" for r in present_roles))

    t_start = time.time()
    n_frames, n_cams, H, W, kind = video_meta(exp_dir)
    out_n = n_frames if args.max_frames is None else min(args.max_frames, n_frames)
    print(f"  sequence [{kind}]: {n_frames} frames, {n_cams} cams, {H}x{W}; "
          f"{n_objects} object(s); propagating {out_n} frame(s)")
    mask_h5 = open_mask_writer(exp_dir / "tool_masks" / "masks.h5",
                               n_frames, n_cams, H, W)
    mask_ds = mask_h5["masks"]
    for pf in prompt_files:
        data = json.loads(pf.read_text())
        cam = data["camera"]
        t0 = time.time()
        n = process_camera(predictor, exp_dir, kind, H, W, cam, data["objects"],
                           role_to_id, mask_ds, device, args.max_frames)
        print(f"    {cam}: {n} frame(s) propagated in {time.time()-t0:.1f}s")
    mask_h5.close()

    # objects.yaml lists ONLY tracked objects (primary tool) — generate_meta uses
    # it verbatim for FoundationPose. Auxiliary/manipulated-object masks stay in
    # masks.h5 (labels 2,3) but are NOT tracked; roles.yaml documents the mapping.
    tracked = [role_name[r] for r in present_roles if r == "primary_tool"] \
        or objects_list[:1]
    (exp_dir / "tool_masks" / "objects.yaml").write_text(
        yaml.safe_dump({"objects": tracked}, default_flow_style=False))
    roles_doc = [{"role": r, "object_id": role_to_id[r], "name": role_name[r],
                  "tracked": (r == "primary_tool")} for r in present_roles]
    (exp_dir / "tool_masks" / "roles.yaml").write_text(
        yaml.safe_dump({"roles": roles_doc}, default_flow_style=False, allow_unicode=True))
    print(f"[done] masks.h5 (labels for {objects_list}) + objects.yaml (tracked={tracked}) "
          f"+ roles.yaml in {time.time()-t_start:.1f}s -> {exp_dir / 'tool_masks'}")


if __name__ == "__main__":
    main()
