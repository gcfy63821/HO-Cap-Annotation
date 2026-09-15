"""
在 cluster 上对所有实验做 SAM2 bbox 双向传播，输出每帧 mask。

用法:
  python run_dense_masks_cluster.py \
    --data-root /viscam/projects/robotool/data \
    --ap-root   /viscam/projects/robotool/_va_bundle_v2_auto_prompts \
    --out-root  /viscam/projects/robotool/_va_dense_masks \
    --task      videos_0204/spatula_flip_egg \
    [--keyword  redrubberspatula]   # 省略则处理该task所有实验
    [--cams     cam0_rgb cam1_rgb]  # 默认全部 cam*_rgb
    [--role     primary_tool]
    [--workers  4]                  # 并行实验数
    [--max-exps N]                  # 调试用：只跑前N个实验
    [--overwrite]

输出:
  <out-root>/<task>/<exp>/cam{N}_rgb.masks.npz
    keys: "masks" shape (T, H, W) uint8 (0=背景 1=工具)
          "frame_ids" shape (T,) int  对应 mp4 帧索引
"""
import sys, json, os, argparse, time
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_cams(ap_dir: Path) -> list[str]:
    return sorted(p.stem for p in ap_dir.glob("cam*_rgb.json"))

def load_prompts(json_path: Path, role: str):
    """返回 {frame_idx: {"bbox": [...], "pos_pts": [...]}}"""
    data = json.loads(json_path.read_text())
    result = {}
    for obj in data.get("objects", []):
        if obj.get("role") != role:
            continue
        fi = obj["frame_index"]
        entry = result.setdefault(fi, {"bbox": None, "pos_pts": []})
        if "bbox" in obj:
            entry["bbox"] = obj["bbox"]
        pos = [(p[0], p[1]) for p, l in zip(obj["points"], obj["labels"]) if l == 1]
        entry["pos_pts"].extend(pos)
    return result

def extract_frames(mp4: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    existing = list(out_dir.glob("*.jpg"))
    if len(existing) == n:
        cap.release()
        return n
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(out_dir / f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    return idx

def run_sam2_on_video(frames_dir: Path, prompt_frame: int,
                      bbox, pos_pts, predictor) -> dict[int, np.ndarray]:
    """返回 {frame_idx: binary_mask}"""
    import torch
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(frames_dir))
        predictor.reset_state(state)

        box_arr = np.array(bbox, dtype=np.float32)
        if pos_pts:
            pts_arr = np.array(pos_pts, dtype=np.float32)
            lbls    = np.ones(len(pos_pts), dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=prompt_frame, obj_id=1,
                box=box_arr, points=pts_arr, labels=lbls)
        else:
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=prompt_frame, obj_id=1,
                box=box_arr)

        fwd = {}
        for fi, oids, logits in predictor.propagate_in_video(state):
            fwd[fi] = (logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8)

        bwd = {}
        for fi, oids, logits in predictor.propagate_in_video(state, reverse=True):
            bwd[fi] = (logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8)

    # merge: forward priority
    all_masks = {}
    for fi in set(list(fwd.keys()) + list(bwd.keys())):
        all_masks[fi] = fwd[fi] if fi in fwd else bwd[fi]
    return all_masks

def infer_one_exp(exp_dir: Path, task: str, cams: list[str], role: str,
                  data_root: Path, ap_root: Path, out_root: Path,
                  overwrite: bool, sam2_root: Path, ckpt: Path, cfg: str):
    import sys, torch
    sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(cfg, str(ckpt), device=device)

    ap_exp = ap_root / task / exp_dir.name / "tool_masks" / "prompts"
    out_exp = out_root / task / exp_dir.name

    results = []
    for cam in cams:
        out_npz = out_exp / f"{cam}.masks.npz"
        if out_npz.exists() and not overwrite:
            results.append(f"  [{cam}] skip (exists)")
            continue

        json_path = ap_exp / f"{cam}.json"
        if not json_path.exists():
            results.append(f"  [{cam}] skip (no prompt json)")
            continue

        prompts = load_prompts(json_path, role)
        bbox_frames = {fi: p for fi, p in prompts.items() if p["bbox"] is not None}
        if not bbox_frames:
            results.append(f"  [{cam}] skip (no bbox)")
            continue

        # find mp4
        mp4 = data_root / task / exp_dir.name / f"{cam.replace('_rgb','')}_rgb.mp4"
        if not mp4.exists():
            # try alternate naming
            mp4 = data_root / task / exp_dir.name / f"{cam}.mp4"
        if not mp4.exists():
            results.append(f"  [{cam}] skip (no mp4 at {mp4})")
            continue

        # extract frames
        frames_dir = out_exp / f"{cam}_frames"
        cap = cv2.VideoCapture(str(mp4))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        extract_frames(mp4, frames_dir)

        # use first bbox frame as prompt
        prompt_fi = sorted(bbox_frames.keys())[0]
        bbox     = bbox_frames[prompt_fi]["bbox"]
        pos_pts  = bbox_frames[prompt_fi]["pos_pts"]

        t0 = time.time()
        all_masks = run_sam2_on_video(frames_dir, prompt_fi, bbox, pos_pts, predictor)
        elapsed = time.time() - t0

        # save as NPZ
        out_exp.mkdir(parents=True, exist_ok=True)
        frame_ids = sorted(all_masks.keys())
        masks_arr = np.stack([all_masks[fi] for fi in frame_ids], axis=0)  # (T, H, W)
        np.savez_compressed(str(out_npz),
                            masks=masks_arr,
                            frame_ids=np.array(frame_ids),
                            prompt_frame=prompt_fi,
                            fps=fps)

        nonzero = int((masks_arr.sum(axis=(1,2)) > 0).sum())
        results.append(f"  [{cam}] {nonzero}/{n_frames} frames w/ mask  "
                       f"prompt=f{prompt_fi}  {elapsed:.1f}s  → {out_npz.name}")

    return exp_dir.name, results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",  required=True)
    ap.add_argument("--ap-root",    required=True)
    ap.add_argument("--out-root",   required=True)
    ap.add_argument("--task",       required=True)
    ap.add_argument("--keyword",    default="")
    ap.add_argument("--cams",       nargs="+", default=None)
    ap.add_argument("--role",       default="primary_tool")
    ap.add_argument("--workers",    type=int, default=1)
    ap.add_argument("--max-exps",   type=int, default=0)
    ap.add_argument("--overwrite",  action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    ap_root   = Path(args.ap_root)
    out_root  = Path(args.out_root)
    task_ap   = ap_root / args.task

    # SAM2 paths (relative to this script's location or absolute)
    sam2_root = Path(__file__).resolve().parents[2] / "mesh_reconstruction/sam2"
    if not sam2_root.exists():
        sam2_root = Path("/viscam/projects/robotool/src/HO-Cap-Annotation/../mesh_reconstruction/sam2")
    ckpt = sam2_root / "checkpoints/sam2.1_hiera_large.pt"
    cfg  = "configs/sam2.1/sam2.1_hiera_l.yaml"

    if not ckpt.exists():
        print(f"[ERROR] SAM2 checkpoint not found: {ckpt}")
        sys.exit(1)

    # discover experiments
    exps = sorted([d for d in task_ap.iterdir() if d.is_dir()
                   and (not args.keyword or args.keyword in d.name)])
    if args.max_exps > 0:
        exps = exps[:args.max_exps]

    # discover cams
    if args.cams:
        cams = args.cams
    else:
        # find cams from first exp
        sample_ap = task_ap / exps[0].name / "tool_masks" / "prompts"
        cams = get_cams(sample_ap) if sample_ap.exists() else ["cam0_rgb"]

    print(f"Task: {args.task}")
    print(f"Exps: {len(exps)}  Cams: {cams}  Workers: {args.workers}")
    print(f"Data: {data_root}")
    print(f"AP:   {ap_root}")
    print(f"Out:  {out_root}")
    print(f"SAM2: {sam2_root}")

    import os; os.chdir(sam2_root)  # SAM2 needs CWD for config

    # run (workers=1 recommended since SAM2 uses GPU internally)
    from functools import partial
    fn = partial(infer_one_exp,
                 task=args.task, cams=cams, role=args.role,
                 data_root=data_root, ap_root=ap_root, out_root=out_root,
                 overwrite=args.overwrite,
                 sam2_root=sam2_root, ckpt=ckpt, cfg=cfg)

    if args.workers <= 1:
        for exp_dir in exps:
            exp_name, results = fn(exp_dir)
            print(f"[{exp_name}]")
            for r in results:
                print(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fn, exp_dir): exp_dir for exp_dir in exps}
            for fut in as_completed(futures):
                exp_name, results = fut.result()
                print(f"[{exp_name}]")
                for r in results:
                    print(r)

    print("\n完成。")

if __name__ == "__main__":
    main()
