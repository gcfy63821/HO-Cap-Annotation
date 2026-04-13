#!/usr/bin/env python3
"""Fully automatic tool segmentation: DINOv2 cross-view seeding + SAM2 video propagation.

Self-correcting pipeline that iterates until all cameras have good masks:
  Phase 1: Find anchor cameras via spatial prior + AMG fallback
  Phase 2: Propagate anchors with SAM2 video predictor
  Phase 3: Build multi-frame DINOv2 reference from anchor masks
  Phase 4: Seed remaining cameras via DINOv2 dense scan + SAM2 click
  Phase 5: Propagate all cameras (incremental H5 writes)
  Phase 6: Validate all cameras (DINOv2 feature consistency + area checks)
  Phase 7: Re-seed and re-propagate failed cameras (up to MAX_ITERS)

Usage:
  python tools/dino_tool_segmentation.py \\
    --hocap_dir /path/to/hocap \\
    --calib_yaml /path/to/calibration.yaml \\
    --output_dir /path/to/output
"""
import argparse, gc, json, os, sys, time, cv2, h5py, numpy as np, yaml, torch
from pathlib import Path

# Force unbuffered stdout so progress appears in log files
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

H, W = 480, 640
CHUNK_SIZE = 50  # Small chunks for 8GB SLURM memory limit
MAX_ITERS = 3

# ═══════════════════════════════════════════════════════════════════════
#  DINOv2 utilities
# ═══════════════════════════════════════════════════════════════════════

def load_dino():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    return model.cuda().eval()


def get_dino_features(model, img_rgb, patch_size=14):
    """Dense DINOv2 patch features. Returns (ph, pw, dim), ph, pw, fh, fw."""
    h, w = img_rgb.shape[:2]
    fh = (h // patch_size) * patch_size
    fw = (w // patch_size) * patch_size
    img = cv2.resize(img_rgb, (fw, fh))
    t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).cuda()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    t = (t - mean) / std
    with torch.no_grad():
        tokens = model.forward_features(t)['x_norm_patchtokens']
    ph, pw = fh // patch_size, fw // patch_size
    return tokens[0].cpu().numpy().reshape(ph, pw, -1), ph, pw, fh, fw


def build_rich_reference(dino, data_h5, masks_ds, ref_cam, sample_every=10):
    """Build per-frame DINOv2 reference features from a single best camera.

    Returns:
        ref_feats: list of (frame, normalized_avg_feature) tuples
        global_ref: single averaged reference vector (dim,)
    Returns (None, None) if no valid frames found.
    """
    N = data_h5['imgs'].shape[0]
    ref_feats = []
    all_feats = []

    for fr in range(0, N, sample_every):
        mask = masks_ds[fr, ref_cam]
        if mask.sum() < 100:
            continue
        rgb = data_h5['imgs'][fr, ref_cam]
        fm, ph, pw, _, _ = get_dino_features(dino, rgb)
        mr = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
        if mr.sum() < 2:
            continue
        avg = fm[mr].mean(axis=0)
        avg /= (np.linalg.norm(avg) + 1e-8)
        ref_feats.append((fr, avg))
        all_feats.append(avg)

    if not all_feats:
        return None, None
    global_ref = np.mean(np.stack(all_feats), axis=0)
    global_ref /= (np.linalg.norm(global_ref) + 1e-8)
    print(f'  Built rich reference from cam{ref_cam}: {len(ref_feats)} frames, dim={global_ref.shape[0]}')
    return ref_feats, global_ref


# ═══════════════════════════════════════════════════════════════════════
#  Seed generation
# ═══════════════════════════════════════════════════════════════════════

def spatial_seed(data_h5, cam, K, T, image_predictor,
                 xy_center, xy_radius, z_min, z_max, scan_every=10):
    """Try spatial-prior seeding: backproject depth, click top-Z pestle point."""
    N = data_h5['imgs'].shape[0]
    best_frame, best_count = 0, 0
    for fr in range(0, N, scan_every):
        depth_m = data_h5['depths'][fr, cam].astype(np.float32) / 1000.0
        v, u = np.mgrid[:H, :W]
        pts = np.stack([(u - K[0,2]) * depth_m / K[0,0],
                        (v - K[1,2]) * depth_m / K[1,1], depth_m], axis=-1).reshape(-1, 3)
        pw = pts @ T[:3,:3].T + T[:3,3]
        valid = depth_m.flatten() > 0.01
        dx = pw[:,0] - xy_center[0]; dy = pw[:,1] - xy_center[1]
        spatial = valid & (np.sqrt(dx**2 + dy**2) < xy_radius) & \
                  (pw[:,2] > z_min) & (pw[:,2] < z_max)
        c = int(spatial.sum())
        if c > best_count:
            best_count = c; best_frame = fr
    if best_count < 20:
        return None

    # Generate click prompt
    depth_m = data_h5['depths'][best_frame, cam].astype(np.float32) / 1000.0
    v, u = np.mgrid[:H, :W]
    pts = np.stack([(u - K[0,2]) * depth_m / K[0,0],
                    (v - K[1,2]) * depth_m / K[1,1], depth_m], axis=-1).reshape(-1, 3)
    pw = pts @ T[:3,:3].T + T[:3,3]
    valid = depth_m.flatten() > 0.01
    dx = pw[:,0] - xy_center[0]; dy = pw[:,1] - xy_center[1]
    spatial = valid & (np.sqrt(dx**2 + dy**2) < xy_radius) & \
              (pw[:,2] > z_min) & (pw[:,2] < z_max)

    z_vals = pw[:,2].copy(); z_vals[~spatial] = -999
    z_thresh = np.percentile(z_vals[spatial], 95)
    top_z = (z_vals.reshape(H,W) > z_thresh)
    if top_z.sum() < 3:
        return None
    ys, xs = np.where(top_z)
    pos = np.array([[int(xs.mean()), int(ys.mean())]])
    labels = np.array([1])

    # Negative: mortar center
    mortar = valid & (np.sqrt(dx**2 + dy**2) < xy_radius) & \
             (pw[:,2] > 0.04) & (pw[:,2] < 0.09)
    if mortar.sum() > 50:
        mys, mxs = np.where(mortar.reshape(H,W))
        pos = np.vstack([pos, [[int(mxs.mean()), int(mys.mean())]]])
        labels = np.array([1, 0])

    rgb = data_h5['imgs'][best_frame, cam]
    image_predictor.set_image(rgb)
    masks, scores, _ = image_predictor.predict(
        point_coords=pos, point_labels=labels, multimask_output=True)

    # Pick best valid mask
    best_mask = _pick_best_mask(masks, rgb)
    if best_mask is None:
        return None
    return {'frame': best_frame, 'mask': best_mask, 'area': int(best_mask.sum()),
            'source': 'spatial'}


def _pick_best_mask(masks, rgb):
    """Score SAM2 multimask outputs, return best valid mask or None."""
    candidates = []
    for mi in range(masks.shape[0]):
        m = masks[mi].astype(bool)
        a = int(m.sum())
        if a < 100 or a > 6000:
            continue
        ys, xs = np.where(m)
        bw = xs.max() - xs.min(); bh = ys.max() - ys.min()
        asp = max(bw, bh) / max(1, min(bw, bh))
        if asp < 1.5:
            continue
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        dark = float(((hsv[...,2][m] < 140) & (hsv[...,1][m] < 130)).mean())
        if dark < 0.25:
            continue
        skin = float(((hsv[...,1][m] > 40) & (hsv[...,2][m] > 90) &
                       ((hsv[...,0][m] < 25) | (hsv[...,0][m] > 165))).mean())
        if skin > 0.30:
            continue
        # Centered size bonus: peaks at 2500px, drops off for too-small or too-large
        size_bonus = max(0, 2.0 - abs(a - 2500) / 1500)
        score = dark * asp * (1 + size_bonus)
        candidates.append((score, m.astype(np.uint8), a, asp, dark))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def dense_dino_seed(dino, data_h5, cam, global_ref, ref_feats, image_predictor,
                    scan_every=5, top_k=3):
    """Seed a camera using dense DINOv2 cross-view matching + temporal refs.

    For each scan frame:
      1. Compute DINOv2 patch features
      2. Score patches with combined global + temporal similarity
      3. Click SAM2 multimask at best patches
      4. Validate each mask with DINOv2 feature similarity (not just area/darkness)
    Returns best seed dict or None.
    """
    N = data_h5['imgs'].shape[0]
    scan_frames = list(range(0, N, scan_every))
    best = None

    # Build temporal lookup
    ref_frame_ids = np.array([f for f, _ in ref_feats])

    for fr in scan_frames:
        rgb = data_h5['imgs'][fr, cam]
        fm, ph, pw, fh, fw = get_dino_features(dino, rgb)
        flat = fm.reshape(-1, fm.shape[-1])
        flat_norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)

        # Global similarity
        global_sims = flat_norm @ global_ref

        # Temporal similarity from closest reference frame
        closest_ref_idx = np.argmin(np.abs(ref_frame_ids - fr))
        temporal_ref = ref_feats[closest_ref_idx][1]
        temporal_sims = flat_norm @ temporal_ref

        # Combined: weight temporal match higher
        combined_sims = 0.4 * global_sims + 0.6 * temporal_sims

        top_idx = np.argsort(combined_sims)[::-1][:top_k]
        for idx in top_idx:
            sim = float(combined_sims[idx])
            if sim < 0.40:
                break

            py_p, px_p = idx // pw, idx % pw
            cu = float((px_p * 14 + 7) * (rgb.shape[1] / fw))
            cv_p = float((py_p * 14 + 7) * (rgb.shape[0] / fh))

            image_predictor.set_image(rgb)
            masks, scores, _ = image_predictor.predict(
                point_coords=np.array([[cu, cv_p]]),
                point_labels=np.array([1]),
                multimask_output=True)

            for mi in range(masks.shape[0]):
                m = masks[mi].astype(bool)
                a = int(m.sum())
                if a < 150 or a > 6000:
                    continue

                # Shape check: pestle is elongated
                ys, xs = np.where(m)
                bw = xs.max() - xs.min(); bh = ys.max() - ys.min()
                asp = max(bw, bh) / max(1, min(bw, bh))
                if asp < 1.3:  # relaxed from 1.5
                    continue

                # DINOv2 mask-level similarity (strongest validation signal)
                mr = cv2.resize(m.astype(np.uint8), (pw, ph),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                if mr.sum() < 1:
                    continue
                mask_feat = fm[mr].mean(axis=0)
                mask_feat /= (np.linalg.norm(mask_feat) + 1e-8)
                mask_sim_global = float(mask_feat @ global_ref)
                mask_sim_temporal = float(mask_feat @ temporal_ref)

                # Reject if the mask's DINOv2 features don't match pestle
                if mask_sim_global < 0.50:
                    continue

                # Darkness check (pestle is dark gray stone)
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                dark = float(((hsv[..., 2][m] < 140) & (hsv[..., 1][m] < 130)).mean())
                if dark < 0.20:
                    continue

                # Skin rejection
                skin = float(((hsv[..., 1][m] > 40) & (hsv[..., 2][m] > 90) &
                               ((hsv[..., 0][m] < 25) | (hsv[..., 0][m] > 165))).mean())
                if skin > 0.35:
                    continue

                # Score: DINOv2 similarity is the primary signal
                size_bonus = max(0, 2.0 - abs(a - 2500) / 1500)
                score = mask_sim_global * mask_sim_temporal * asp * (1 + size_bonus) * (1 + dark)

                result = {
                    'frame': fr, 'mask': m.astype(np.uint8), 'area': a,
                    'asp': asp, 'dark': dark, 'skin': skin,
                    'sim_global': mask_sim_global, 'sim_temporal': mask_sim_temporal,
                    'click_sim': sim, 'score': score,
                    'click_u': cu, 'click_v': cv_p, 'source': 'dino'
                }
                if best is None or score > best['score']:
                    best = result
    return best


# ═══════════════════════════════════════════════════════════════════════
#  SAM2 video propagation
# ═══════════════════════════════════════════════════════════════════════

def extract_jpegs(data_h5, tmp_dir, cams):
    """Extract H5 RGB frames to per-camera JPEG folders (skips if done)."""
    N = data_h5['imgs'].shape[0]
    for cam in cams:
        cam_dir = Path(tmp_dir) / f'cam{cam:02d}'
        cam_dir.mkdir(parents=True, exist_ok=True)
        if (cam_dir / f'color_{N-1:06d}.jpg').exists():
            continue
        for fr in range(N):
            bgr = cv2.cvtColor(data_h5['imgs'][fr, cam], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(cam_dir / f'color_{fr:06d}.jpg'), bgr)
        print(f'  cam{cam}: extracted {N} frames')


def propagate_camera(video_predictor, cam_dir, seed_mask, seed_frame, N, masks_ds, cam):
    """Propagate seed mask forward+backward, write directly to H5 dataset."""
    written = 0

    def _run_chunk(img_paths, seed, base, forward):
        nonlocal written
        if not img_paths:
            return seed
        state = video_predictor.init_state(
            img_paths=img_paths, offload_video_to_cpu=True,
            offload_state_to_cpu=True, async_loading_frames=False)
        video_predictor.reset_state(state)
        video_predictor.add_new_mask(
            inference_state=state, frame_idx=0, obj_id=1,
            mask=seed.astype(np.uint8))
        last = seed
        for fi, _, logits in video_predictor.propagate_in_video(state):
            m = (logits[0] > 0.0).cpu().numpy().astype(np.uint8)
            if m.ndim == 3: m = m[0]
            actual = base + fi if forward else base - fi
            masks_ds[actual, cam] = m
            last = m
            written += 1
        del state; torch.cuda.empty_cache(); gc.collect()
        return last

    # Forward
    cur = seed_mask; pos = seed_frame
    while pos < N:
        end = min(pos + CHUNK_SIZE, N)
        paths = [str(cam_dir / f'color_{i:06d}.jpg') for i in range(pos, end)]
        cur = _run_chunk(paths, cur, pos, True)
        pos = end

    # Backward
    if seed_frame > 0:
        cur = seed_mask; pos = seed_frame
        while pos > 0:
            start = max(pos - CHUNK_SIZE, 0)
            paths = [str(cam_dir / f'color_{i:06d}.jpg') for i in range(pos, start-1, -1)]
            cur = _run_chunk(paths, cur, pos, False)
            pos = start

    return written


# ═══════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_camera(dino, data_h5, masks_ds, cam, ref_vec, N,
                    min_area=200, max_area=8000, min_sim=0.45):
    """Check propagated masks for a camera. Returns (pass, reason)."""
    sample_frames = list(range(50, N - 50, 50))  # skip endpoints (often drifted)
    if not sample_frames:
        sample_frames = [N // 2]

    areas = []
    sims = []
    for fr in sample_frames:
        mask = masks_ds[fr, cam]
        a = int(mask.sum())
        areas.append(a)
        if a < 50:
            continue
        fm, ph, pw, _, _ = get_dino_features(dino, data_h5['imgs'][fr, cam])
        mr = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
        if mr.sum() < 1:
            continue
        avg = fm[mr].mean(axis=0)
        avg /= (np.linalg.norm(avg) + 1e-8)
        sims.append(float(avg @ ref_vec))

    med_area = float(np.median(areas)) if areas else 0
    med_sim = float(np.median(sims)) if sims else 0
    area_std = float(np.std(areas)) if len(areas) > 1 else 0
    area_cv = area_std / max(med_area, 1)  # coefficient of variation

    if med_area < min_area:
        return False, f'under-segmented (median_area={med_area:.0f} < {min_area})'
    if med_area > max_area:
        return False, f'over-segmented (median_area={med_area:.0f} > {max_area})'
    if area_cv > 0.6:
        return False, f'unstable masks (area_cv={area_cv:.2f} > 0.6, med={med_area:.0f}, std={area_std:.0f})'
    if med_sim < min_sim:
        return False, f'wrong object (median_sim={med_sim:.3f} < {min_sim})'
    return True, f'OK (median_area={med_area:.0f}, median_sim={med_sim:.3f}, cv={area_cv:.2f})'


# ═══════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════

def visualize(data_h5, masks_path, output_dir, make_video=True):
    """Generate snapshot PNGs + MP4 video of mask overlay for all 8 cameras."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    mf = h5py.File(masks_path, 'r')
    N = data_h5['imgs'].shape[0]
    snap_frames = [0, 50, 100, 200, 350, 500, 700, min(N-1, 756)]

    def _render(fr):
        canvas = np.zeros((H*2, W*4, 3), dtype=np.uint8)
        for c in range(8):
            rgb = data_h5['imgs'][fr, c].copy()
            mask = mf['masks'][fr, c]
            if mask.sum() > 0:
                overlay = rgb.copy(); overlay[mask > 0] = (0, 255, 0)
                rgb = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)
                cnt, _ = cv2.findContours(mask*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, cnt, -1, (255,255,0), 2)
                info = f"a={int(mask.sum())}"
            else:
                info = "empty"
            r, col = c // 4, c % 4
            canvas[r*H:(r+1)*H, col*W:(col+1)*W] = rgb
            cv2.putText(canvas, f"cam{c} {info}", (col*W+5, r*H+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        cv2.putText(canvas, f"frame {fr}/{N-1}", (W*4-200, H*2-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return canvas

    for fr in snap_frames:
        if fr >= N: continue
        bgr = cv2.cvtColor(_render(fr), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / f'snapshot_f{fr:04d}.png'), bgr)

    if make_video:
        vid = out / 'masks_video.mp4'
        wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*'mp4v'), 30, (W*4, H*2))
        for fr in range(N):
            wr.write(cv2.cvtColor(_render(fr), cv2.COLOR_RGB2BGR))
            if fr % 100 == 0: print(f'  video: frame {fr}/{N}')
        wr.release()
        print(f'  Saved {vid}')

    mf.close()


# ═══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Automatic tool segmentation')
    parser.add_argument('--hocap_dir', required=True)
    parser.add_argument('--calib_yaml', required=True)
    parser.add_argument('--output_dir', required=True)
    _root = Path(__file__).resolve().parent.parent
    parser.add_argument('--sam2_ckpt',
        default=str(_root / 'config/checkpoints/sam2/sam2.1_hiera_large.pt'))
    parser.add_argument('--sam2_image_cfg',
        default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--sam2_video_cfg',
        default=str(_root / 'config/sam2_config/sam2.1_hiera_l.yaml'))
    parser.add_argument('--xy_center', type=float, nargs=2, default=[-0.07, -0.01])
    parser.add_argument('--xy_radius', type=float, default=0.12)
    parser.add_argument('--z_min', type=float, default=0.10)
    parser.add_argument('--z_max', type=float, default=0.30)
    parser.add_argument('--anchor_cams', type=int, nargs='+', default=None,
        help='Force these cameras as anchors (skip spatial search)')
    parser.add_argument('--min_area', type=int, default=200,
        help='Minimum median mask area to pass validation')
    parser.add_argument('--max_area', type=int, default=6000,
        help='Maximum median mask area to pass validation')
    parser.add_argument('--min_sim', type=float, default=0.45,
        help='Minimum DINOv2 similarity to pass validation')
    parser.add_argument('--no_video', action='store_true')
    args = parser.parse_args()

    t_start = time.time()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / 'tmp_jpegs'

    # Load calibration
    with open(args.calib_yaml) as fp:
        calib = yaml.safe_load(fp)
    Ks = np.array([np.array(c['color_intrinsic_matrix'], dtype=np.float32) for c in calib])
    Ts = np.array([np.array(c['transformation'], dtype=np.float32) for c in calib])
    sx, sy = 640/1280, 480/720
    for i in range(len(Ks)):
        Ks[i][0,:] *= sx; Ks[i][1,:] *= sy

    data_h5 = h5py.File(Path(args.hocap_dir) / 'data00000000.h5', 'r')
    N = data_h5['imgs'].shape[0]
    n_cams = data_h5['imgs'].shape[1]
    print(f'[INFO] {N} frames, {n_cams} cameras')

    def _elapsed():
        return f'[{time.time() - t_start:.0f}s]'

    # ── Load models ──────────────────────────────────────────────────
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(f'{_elapsed()} Loading SAM2 image predictor...')
    sam2_img = build_sam2(args.sam2_image_cfg, args.sam2_ckpt, device='cuda:0')
    image_predictor = SAM2ImagePredictor(sam2_img)

    print(f'{_elapsed()} Loading DINOv2...')
    dino = load_dino()

    # ── Phase 1: Find anchor cameras ─────────────────────────────────
    print(f'\n{_elapsed()} ═══ Phase 1: Finding anchor cameras ═══')
    seeds = {}  # cam → {frame, mask, area, source, ...}

    if args.anchor_cams:
        anchor_cams = args.anchor_cams
        print(f'  Using forced anchors: {anchor_cams}')
        for cam in anchor_cams:
            result = spatial_seed(data_h5, cam, Ks[cam], Ts[cam], image_predictor,
                                  tuple(args.xy_center), args.xy_radius, args.z_min, args.z_max)
            if result:
                seeds[cam] = result
                print(f'  cam{cam}: spatial seed OK (frame={result["frame"]}, area={result["area"]})')
            else:
                print(f'  cam{cam}: spatial seed FAILED, will try DINOv2 later')
    else:
        # Try spatial prior on all cameras, pick best 2
        spatial_results = {}
        for cam in range(n_cams):
            result = spatial_seed(data_h5, cam, Ks[cam], Ts[cam], image_predictor,
                                  tuple(args.xy_center), args.xy_radius, args.z_min, args.z_max)
            if result:
                spatial_results[cam] = result
                print(f'  cam{cam}: spatial seed area={result["area"]}')
            else:
                print(f'  cam{cam}: no spatial seed')

        if spatial_results:
            # Pick top 2 by area (bigger = more of pestle visible)
            ranked = sorted(spatial_results.items(), key=lambda x: x[1]['area'], reverse=True)
            anchor_cams = [c for c, _ in ranked[:2]]
            for cam in anchor_cams:
                seeds[cam] = spatial_results[cam]
            print(f'  Selected anchors: {anchor_cams}')
        else:
            print('  ERROR: No spatial seeds found. Provide --anchor_cams manually.')
            data_h5.close()
            return

    if not seeds:
        print('  ERROR: No anchor seeds generated.')
        data_h5.close()
        return

    # Unload SAM2 image predictor before loading video predictor
    del image_predictor, sam2_img
    torch.cuda.empty_cache(); gc.collect()
    print('  Unloaded SAM2 image predictor')

    # ── Phase 2: Propagate anchor cameras ────────────────────────────
    print(f'\n{_elapsed()} ═══ Phase 2: Propagating anchor cameras ═══')
    anchor_cams = list(seeds.keys())
    extract_jpegs(data_h5, tmp_dir, anchor_cams)

    # Close data_h5 + unload DINOv2 to free ALL memory for video predictor
    data_h5.close()
    del dino
    torch.cuda.empty_cache(); gc.collect()
    print('  Closed data H5 + unloaded DINOv2 to free memory')

    from hocap_annotation.wrappers.sam2 import build_sam2_video_predictor
    print(f'{_elapsed()}   Loading SAM2 video predictor...')
    video_predictor = build_sam2_video_predictor(
        config_file=args.sam2_video_cfg, ckpt_path=args.sam2_ckpt, device='cuda:0')

    # Create initial masks.h5
    masks_path = output_dir / 'masks.h5'
    masks_h5 = h5py.File(masks_path, 'w')
    masks_ds = masks_h5.create_dataset('masks', shape=(N, n_cams, H, W),
                                        dtype=np.uint8, chunks=(1, 1, H, W),
                                        compression='gzip')
    for cam in anchor_cams:
        s = seeds[cam]
        cam_dir = tmp_dir / f'cam{cam:02d}'
        print(f'  cam{cam}: propagating from frame {s["frame"]}...')
        written = propagate_camera(video_predictor, cam_dir, s['mask'],
                                    s['frame'], N, masks_ds, cam)
        masks_h5.flush()
        print(f'  cam{cam}: wrote {written} frames')

    # Unload video predictor before DINOv2 reference building
    del video_predictor
    torch.cuda.empty_cache(); gc.collect()
    print('  Unloaded video predictor')

    # ── Phase 3: Build rich DINOv2 reference ───────────────────────────
    print(f'\n{_elapsed()} ═══ Phase 3: Building rich DINOv2 reference ═══')
    # Reopen data_h5 (was closed for Phase 2 memory)
    data_h5 = h5py.File(Path(args.hocap_dir) / 'data00000000.h5', 'r')
    dino = load_dino()

    # Pick best anchor camera (lowest area CV) as reference
    best_ref_cam, best_cv = anchor_cams[0], 999.0
    for cam in anchor_cams:
        areas = [int(masks_ds[fr, cam].sum()) for fr in range(50, N - 50, 50)]
        if areas:
            med = float(np.median(areas))
            cv_val = float(np.std(areas)) / max(med, 1) if len(areas) > 1 else 0
            print(f'  anchor cam{cam}: med_area={med:.0f}, cv={cv_val:.2f}')
            if cv_val < best_cv:
                best_cv = cv_val; best_ref_cam = cam
    print(f'  Selected cam{best_ref_cam} as reference (cv={best_cv:.2f})')

    ref_feats, global_ref = build_rich_reference(dino, data_h5, masks_ds, best_ref_cam, sample_every=10)
    if global_ref is None:
        print('  ERROR: Could not build reference.')
        masks_h5.close(); data_h5.close()
        return

    # ── Phase 4-7: Seed, propagate, validate, iterate ────────────────
    remaining_cams = [c for c in range(n_cams) if c not in anchor_cams]
    extract_jpegs(data_h5, tmp_dir, remaining_cams)

    for iteration in range(MAX_ITERS):
        cams_to_process = remaining_cams if iteration == 0 else failed_cams

        if not cams_to_process:
            print(f'\n  All cameras passed validation!')
            break

        print(f'\n{_elapsed()} ═══ Iteration {iteration + 1}/{MAX_ITERS}: '
              f'Processing cameras {cams_to_process} ═══')

        # Phase 4: Seed with DINOv2 cross-view + SAM2 image predictor
        print(f'\n  {_elapsed()} [Phase 4] Loading SAM2 image predictor for seeding...')
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        # Hydra global state must be re-initialized for second build_sam2 call
        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        if not GlobalHydra.instance().is_initialized():
            initialize_config_module("sam2", version_base="1.2")
        sam2_img = build_sam2(args.sam2_image_cfg, args.sam2_ckpt, device='cuda:0')
        img_pred = SAM2ImagePredictor(sam2_img)

        print(f'  {_elapsed()} Seeding {len(cams_to_process)} cameras via DINOv2 cross-view...')
        scan_every = max(1, 5 - iteration)  # denser scan on retries
        for cam in cams_to_process:
            print(f'    {_elapsed()} cam{cam}: scanning...')
            result = dense_dino_seed(dino, data_h5, cam, global_ref, ref_feats,
                                     img_pred, scan_every=scan_every)
            if result:
                seeds[cam] = result
                print(f'    {_elapsed()} cam{cam}: seed at f{result["frame"]} '
                      f'area={result["area"]} sim_g={result["sim_global"]:.3f} '
                      f'sim_t={result["sim_temporal"]:.3f} score={result["score"]:.3f}')
            else:
                print(f'    {_elapsed()} cam{cam}: NO SEED FOUND')

        # Unload SAM2 image predictor + DINOv2 + close data_h5 before propagation
        del img_pred, sam2_img, dino
        data_h5.close()
        torch.cuda.empty_cache(); gc.collect()
        print('  Unloaded DINOv2 + SAM2 image predictor + closed data H5')

        # Phase 5: Propagate with SAM2 video predictor
        print(f'\n  {_elapsed()} [Phase 5] Loading SAM2 video predictor...')
        from hocap_annotation.wrappers.sam2 import build_sam2_video_predictor
        vid_pred = build_sam2_video_predictor(
            config_file=args.sam2_video_cfg, ckpt_path=args.sam2_ckpt, device='cuda:0')

        for cam in cams_to_process:
            if cam not in seeds:
                continue
            s = seeds[cam]
            cam_dir = tmp_dir / f'cam{cam:02d}'
            print(f'    {_elapsed()} cam{cam}: propagating from frame {s["frame"]}...')
            # Clear old masks for this camera
            for fr in range(N):
                masks_ds[fr, cam] = np.zeros((H, W), dtype=np.uint8)
            written = propagate_camera(vid_pred, cam_dir, s['mask'],
                                        s['frame'], N, masks_ds, cam)
            masks_h5.flush()
            print(f'    cam{cam}: wrote {written} frames')

        # Unload video predictor, reload DINOv2 + data_h5 for validation
        del vid_pred
        torch.cuda.empty_cache(); gc.collect()
        print('  Unloaded video predictor')

        # Phase 6: Validate ALL cameras with DINOv2
        print(f'\n  {_elapsed()} [Phase 6] Loading DINOv2 for validation...')
        data_h5 = h5py.File(Path(args.hocap_dir) / 'data00000000.h5', 'r')
        dino = load_dino()

        failed_cams = []
        for cam in range(n_cams):
            if cam not in seeds:
                failed_cams.append(cam)
                print(f'    cam{cam}: FAIL (no seed)')
                continue
            passed, reason = validate_camera(
                dino, data_h5, masks_ds, cam, global_ref, N,
                min_area=args.min_area, max_area=args.max_area,
                min_sim=args.min_sim)
            status = 'PASS' if passed else 'FAIL'
            print(f'    cam{cam}: {status} — {reason}')
            if not passed:
                failed_cams.append(cam)

        # Update reference from best passing camera for next iteration
        passing_cams = [c for c in range(n_cams) if c not in failed_cams]
        if passing_cams and failed_cams:
            # Pick best passing camera by lowest area CV
            best_pass_cam, best_pass_cv = passing_cams[0], 999.0
            for cam in passing_cams:
                areas = [int(masks_ds[fr, cam].sum()) for fr in range(50, N - 50, 50)]
                if areas:
                    med = float(np.median(areas))
                    cv_val = float(np.std(areas)) / max(med, 1) if len(areas) > 1 else 0
                    if cv_val < best_pass_cv:
                        best_pass_cv = cv_val; best_pass_cam = cam
            print(f'\n  Rebuilding rich reference from cam{best_pass_cam} (cv={best_pass_cv:.2f})')
            new_feats, new_ref = build_rich_reference(dino, data_h5, masks_ds, best_pass_cam, sample_every=10)
            if new_ref is not None:
                ref_feats, global_ref = new_feats, new_ref

        # DINOv2 stays loaded for next iteration's seeding (or final report)

    # ── Cleanup models ───────────────────────────────────────────────
    if 'dino' in dir():
        del dino
    torch.cuda.empty_cache(); gc.collect()

    # ── Save metadata ────────────────────────────────────────────────
    meta = {}
    for cam, s in seeds.items():
        meta[f'cam{cam}'] = {k: v for k, v in s.items() if k != 'mask'}
    with open(output_dir / 'seed_info.json', 'w') as fp:
        json.dump(meta, fp, indent=2, default=str)

    masks_h5.close()

    # ── Final validation report ──────────────────────────────────────
    print(f'\n{_elapsed()} ═══ Final Results ═══')
    dino2 = load_dino()
    masks_h5 = h5py.File(masks_path, 'r')
    masks_ds2 = masks_h5['masks']
    all_pass = True
    for cam in range(n_cams):
        passed, reason = validate_camera(
            dino2, data_h5, masks_ds2, cam, global_ref, N,
            min_area=args.min_area, max_area=args.max_area,
            min_sim=args.min_sim)
        status = 'PASS' if passed else 'FAIL'
        print(f'  cam{cam}: {status} — {reason}')
        if not passed:
            all_pass = False
    masks_h5.close()
    del dino2; torch.cuda.empty_cache()

    if all_pass:
        print('\n  ALL CAMERAS PASSED!')
    else:
        print(f'\n  Some cameras failed after {MAX_ITERS} iterations.')
        print('  Check visualizations and consider adjusting --min_area/--max_area/--min_sim')

    # ── Visualize ────────────────────────────────────────────────────
    print(f'\n{_elapsed()} ═══ Generating visualizations ═══')
    viz_dir = output_dir / 'viz'
    visualize(data_h5, masks_path, viz_dir, make_video=not args.no_video)

    data_h5.close()
    elapsed = time.time() - t_start
    print(f'\n[DONE] Total time: {elapsed/60:.1f} min ({elapsed:.0f} sec)')


if __name__ == '__main__':
    main()
