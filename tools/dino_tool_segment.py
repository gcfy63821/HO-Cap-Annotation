#!/usr/bin/env python3
"""Fully automatic tool segmentation: mesh DINOv2 reference + SAM2 video propagation.

Self-correcting pipeline that iterates until all cameras have good masks:
  Phase 1: Build DINOv2 reference from rendered mesh views, find anchor cameras
  Phase 2: Propagate anchors with SAM2 video predictor
  Phase 3: Build multi-frame DINOv2 reference from anchor masks (real camera)
  Phase 4: Seed remaining cameras via DINOv2 dense scan + SAM2 click
  Phase 5: Propagate all cameras (incremental H5 writes)
  Phase 6: Validate all cameras (DINOv2 feature consistency + area checks)
  Phase 7: Re-seed and re-propagate failed cameras (up to MAX_ITERS)

No spatial priors or object-specific heuristics — uses the tool's textured mesh
as the only reference for what to segment.

Usage:
  python -m robotool_flow.dino_tool_segmentation \\
    --data_h5 /path/to/data00000000.h5 \\
    --calib_yaml /path/to/calibration.yaml \\
    --tool_mesh /path/to/textured_mesh.obj \\
    --output_dir /path/to/output
"""
import argparse, gc, json, os, sys, time, cv2, h5py, numpy as np, yaml, torch
from pathlib import Path

# EGL backend for headless rendering (must be set before pyrender import)
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Force unbuffered stdout so progress appears in log files
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Add ho-cap root to path for hocap_annotation imports (tools/ lives one level below the root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

H, W = 480, 640  # default fallback; overridden from H5 imgs shape at runtime
CHUNK_SIZE = 100  # frames per SAM2 video chunk
MAX_ITERS = 3
DRIFT_THRESHOLD = 0.35  # re-seed if area drops below 35% of rolling reference
RESEED_SIM_THRESHOLD = 0.30  # acceptance threshold for re-seed masks (lower = more lenient)

# ═══════════════════════════════════════════════════════════════════════
#  Mesh rendering for DINOv2 reference
# ═══════════════════════════════════════════════════════════════════════

def render_mesh_views(mesh_path, n_views=12, img_size=480):
    """Render textured mesh from multiple viewpoints on a hemisphere.

    Returns list of (rgb_image, object_mask) tuples.
    """
    import pyrender, trimesh

    mesh = trimesh.load(mesh_path, process=True)
    center = mesh.bounding_box.centroid
    radius = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) / 2
    cam_dist = radius * 3.0

    # Generate viewpoints: 3 elevation rings around the object
    views = []
    for i in range(n_views):
        elev = np.radians(15 + 45 * (i % 3) / 2)  # 15, ~37.5, 60 deg
        azim = np.radians(360 * i / n_views)
        eye = center + cam_dist * np.array([
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev)])
        forward = center - eye
        forward /= np.linalg.norm(forward)
        up = np.array([0, 0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0, 1, 0.0])
            right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = -up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = eye
        views.append(cam_pose)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    cam_node = scene.add(camera, pose=views[0])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light_node = scene.add(light, pose=views[0])
    renderer = pyrender.OffscreenRenderer(img_size, img_size)

    results = []
    for vp in views:
        scene.set_pose(cam_node, vp)
        scene.set_pose(light_node, vp)
        color, depth = renderer.render(scene)
        obj_mask = depth > 0
        if obj_mask.sum() > 100:
            results.append((color, obj_mask))

    renderer.delete()
    return results


def build_mesh_reference(dino, mesh_path, n_views=12, patch_size=14):
    """Extract DINOv2 reference features from rendered mesh views.

    Returns global_ref: averaged normalized feature vector (dim,).
    """
    rendered = render_mesh_views(mesh_path, n_views=n_views)
    if not rendered:
        return None

    all_feats = []
    for color, obj_mask in rendered:
        h, w = color.shape[:2]
        fh = (h // patch_size) * patch_size
        fw = (w // patch_size) * patch_size
        img_r = cv2.resize(color, (fw, fh))
        t = torch.from_numpy(img_r).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).cuda()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        t = (t - mean) / std
        with torch.no_grad():
            tokens = dino.forward_features(t)['x_norm_patchtokens']
        ph, pw = fh // patch_size, fw // patch_size
        fm = tokens[0].cpu().numpy().reshape(ph, pw, -1)
        msk_r = cv2.resize(obj_mask.astype(np.uint8), (pw, ph),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        if msk_r.sum() > 0:
            feat = fm[msk_r].mean(axis=0)
            feat /= (np.linalg.norm(feat) + 1e-8)
            all_feats.append(feat)

    if not all_feats:
        return None
    global_ref = np.mean(np.stack(all_feats), axis=0)
    global_ref /= (np.linalg.norm(global_ref) + 1e-8)
    print(f'  Built mesh reference from {len(all_feats)} rendered views, dim={global_ref.shape[0]}')
    return global_ref


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

def mesh_dino_seed(dino, data_h5, cam, mesh_ref, image_predictor,
                   scan_every=10, top_k=5,
                   min_area=100, max_area=15000, min_sim=0.20,
                   first_hit=False):
    """Seed a camera using mesh DINOv2 reference.

    Scans frames, clicks SAM2 at top-matching DINOv2 patches, validates
    masks by comparing their DINOv2 features back to the mesh reference.
    No spatial prior or object-specific heuristics.

    Args:
        scan_every: stride over frames. Smaller = denser = more likely to
            catch a frame where the object is well-visible, at a linear
            compute cost.
        min_area: reject SAM2 masks with fewer pixels than this. Lower this
            (e.g. 20) to accept small / partially-visible objects.
        max_area: upper bound; masks bigger than this are rejected as
            likely over-segmenting (tool + arm + table).
        min_sim: minimum DINOv2 patch-to-mesh similarity to bother clicking.
            Lower this (e.g. 0.12) if partial views are getting rejected.
        first_hit: if True, return as soon as ONE frame produces a valid seed
            (frame 0, then frame `scan_every`, then `2*scan_every`, ...).
            This is much faster than the default exhaustive search which
            scores every scanned frame and keeps the best. Use this when the
            tool is visible for most of the sequence and you don't need the
            absolute best-score seed.

    Returns best (or first-hit) seed dict, or None.
    """
    N = data_h5['imgs'].shape[0]
    best = None

    for fr in range(0, N, scan_every):
        rgb = data_h5['imgs'][fr, cam]
        fm, ph, pw, fh, fw = get_dino_features(dino, rgb)
        flat = fm.reshape(-1, fm.shape[-1])
        flat_norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        sims = flat_norm @ mesh_ref

        top_idx = np.argsort(sims)[::-1][:top_k]
        for idx in top_idx:
            sim = float(sims[idx])
            if sim < min_sim:
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
                if a < min_area or a > max_area:
                    continue

                # DINOv2 mask-level validation against mesh reference
                mr = cv2.resize(m.astype(np.uint8), (pw, ph),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                if mr.sum() < 1:
                    continue
                mask_feat = fm[mr].mean(axis=0)
                mask_feat /= (np.linalg.norm(mask_feat) + 1e-8)
                mask_sim = float(mask_feat @ mesh_ref)

                if mask_sim < min_sim:
                    continue

                # Score: higher DINOv2 similarity = better
                score = mask_sim * sim

                result = {
                    'frame': fr, 'mask': m.astype(np.uint8), 'area': a,
                    'mesh_sim': mask_sim, 'click_sim': sim, 'score': score,
                    'click_u': cu, 'click_v': cv_p, 'source': 'mesh_dino'
                }
                if best is None or score > best['score']:
                    best = result
                    if first_hit:
                        return best

    return best


def dense_dino_seed(dino, data_h5, cam, global_ref, ref_feats, image_predictor,
                    scan_every=5, top_k=3, ref_median_area=None,
                    first_hit=False):
    """Seed a camera using dense DINOv2 cross-view matching + temporal refs.

    For each scan frame:
      1. Compute DINOv2 patch features
      2. Score patches with combined global + temporal similarity
      3. Click SAM2 multimask at best patches
      4. Validate each mask with DINOv2 feature similarity
      5. Reject masks too small/large vs ref_median_area

    Args:
        ref_median_area: median mask area from anchor camera. If provided,
            seeds with area < 30% or > 300% of this are rejected.
        first_hit: if True, return as soon as ONE valid seed is found (fast).

    Returns best (or first-hit) seed dict, or None.
    """
    N = data_h5['imgs'].shape[0]
    scan_frames = list(range(0, N, scan_every))
    best = None

    # Area bounds from anchor reference
    area_lo = int(ref_median_area * 0.25) if ref_median_area else 100
    area_hi = int(ref_median_area * 4.0) if ref_median_area else 15000

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
            if sim < 0.35:
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
                if a < area_lo or a > area_hi:
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

                if mask_sim_global < 0.45:
                    continue

                # Score: DINOv2 similarity is the primary signal
                score = mask_sim_global * mask_sim_temporal * (1 + sim)

                result = {
                    'frame': fr, 'mask': m.astype(np.uint8), 'area': a,
                    'sim_global': mask_sim_global, 'sim_temporal': mask_sim_temporal,
                    'click_sim': sim, 'score': score,
                    'click_u': cu, 'click_v': cv_p, 'source': 'dino'
                }
                if best is None or score > best['score']:
                    best = result
                    if first_hit:
                        return best
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


def reseed_at_frame(dino, data_h5, cam, fr, global_ref, image_predictor, ref_area=None):
    """Generate a fresh SAM2 seed mask at a specific frame using DINOv2.

    Finds the tool in the frame by matching DINOv2 patches to the reference,
    then clicks SAM2 to generate a mask. Returns the best mask or None.
    """
    rgb = data_h5['imgs'][fr, cam]
    fm, ph, pw, fh, fw = get_dino_features(dino, rgb)
    flat = fm.reshape(-1, fm.shape[-1])
    flat_norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
    sims = flat_norm @ global_ref

    top_idx = np.argsort(sims)[::-1][:5]
    image_predictor.set_image(rgb)

    best_mask, best_score = None, 0
    for idx in top_idx:
        if float(sims[idx]) < 0.30:
            break
        py_p, px_p = idx // pw, idx % pw
        cu = float((px_p * 14 + 7) * (rgb.shape[1] / fw))
        cv_p = float((py_p * 14 + 7) * (rgb.shape[0] / fh))

        masks, scores, _ = image_predictor.predict(
            point_coords=np.array([[cu, cv_p]]),
            point_labels=np.array([1]),
            multimask_output=True)

        for mi in range(masks.shape[0]):
            m = masks[mi].astype(bool)
            a = int(m.sum())
            if a < 100:
                continue
            if ref_area and (a < ref_area * 0.2 or a > ref_area * 5.0):
                continue

            mr = cv2.resize(m.astype(np.uint8), (pw, ph),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            if mr.sum() < 1:
                continue
            feat = fm[mr].mean(axis=0)
            feat /= (np.linalg.norm(feat) + 1e-8)
            sim = float(feat @ global_ref)
            if sim > best_score:
                best_score = sim
                best_mask = m.astype(np.uint8)

    if best_mask is not None and best_score > RESEED_SIM_THRESHOLD:
        return best_mask
    return None


def propagate_camera(video_predictor, cam_dir, seed_mask, seed_frame, N, masks_ds, cam,
                     dino=None, image_predictor=None, global_ref=None, data_h5=None,
                     ref_area=None, drift_threshold=None, chunk_size=None):
    """Propagate seed mask forward+backward with optional periodic re-seeding.

    When dino/image_predictor/global_ref are provided, detects drift at chunk
    boundaries and re-generates fresh seeds using DINOv2+SAM2 to prevent
    partial segmentation from accumulating.
    """
    written = 0
    reseed_count = 0
    can_reseed = all(x is not None for x in [dino, image_predictor, global_ref, data_h5])
    thr = drift_threshold if drift_threshold is not None else DRIFT_THRESHOLD
    chunk = chunk_size if chunk_size is not None else CHUNK_SIZE

    def _run_chunk(img_paths, seed, base, forward):
        nonlocal written
        if not img_paths:
            return seed
        try:
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
        except Exception as e:
            print(f'    WARNING: chunk propagation failed at base={base}: {e}')
            torch.cuda.empty_cache(); gc.collect()
            return seed

    def _check_and_reseed(cur_mask, frame_idx, rolling_ref_area):
        """Check for drift, re-seed if needed. Returns (mask, updated_ref_area, reseeded)."""
        nonlocal reseed_count
        cur_area = float(cur_mask.sum())

        if cur_area >= thr * rolling_ref_area or not can_reseed:
            # No drift or can't reseed — update rolling reference
            new_ref = 0.8 * rolling_ref_area + 0.2 * cur_area if cur_area > 50 else rolling_ref_area
            return cur_mask, new_ref, False

        # Drift detected — try to re-seed
        new_seed = reseed_at_frame(dino, data_h5, cam, frame_idx, global_ref,
                                   image_predictor, ref_area=rolling_ref_area)
        if new_seed is not None:
            masks_ds[frame_idx, cam] = new_seed
            reseed_count += 1
            return new_seed, rolling_ref_area, True
        return cur_mask, rolling_ref_area, False

    # Forward propagation with re-seeding
    cur = seed_mask; pos = seed_frame
    rolling_ref = float(seed_mask.sum()) if ref_area is None else ref_area

    while pos < N:
        end = min(pos + chunk, N)
        paths = [str(cam_dir / f'color_{i:06d}.jpg') for i in range(pos, end)]
        cur = _run_chunk(paths, cur, pos, True)
        # Check drift at chunk boundary
        if end < N:
            cur, rolling_ref, reseeded = _check_and_reseed(cur, end - 1, rolling_ref)
        pos = end

    # Backward propagation with re-seeding
    if seed_frame > 0:
        cur = seed_mask; pos = seed_frame
        rolling_ref = float(seed_mask.sum()) if ref_area is None else ref_area

        while pos > 0:
            start = max(pos - chunk, 0)
            paths = [str(cam_dir / f'color_{i:06d}.jpg') for i in range(pos, start-1, -1)]
            cur = _run_chunk(paths, cur, pos, False)
            if start > 0:
                cur, rolling_ref, reseeded = _check_and_reseed(cur, start, rolling_ref)
            pos = start

    return written, reseed_count


# ═══════════════════════════════════════════════════════════════════════
#  Post-propagation mask cleanup
# ═══════════════════════════════════════════════════════════════════════

def clean_masks_cc(dino, data_h5, masks_ds, cam, ref_vec, N, sample_every=10):
    """Clean propagated masks via connected component filtering.

    For each frame, if the mask has multiple disconnected components,
    keep only the component whose DINOv2 features best match the reference.
    This removes extra objects that SAM2 leaked into.

    Returns number of frames modified.
    """
    modified = 0
    # First, compute reference median area from sampled frames
    sample_areas = []
    for fr in range(0, N, sample_every):
        a = int(masks_ds[fr, cam].sum())
        if a > 50:
            sample_areas.append(a)
    if not sample_areas:
        return 0
    ref_median_area = float(np.median(sample_areas))

    for fr in range(N):
        mask = masks_ds[fr, cam]
        if mask.sum() < 50:
            continue

        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels <= 2:  # 0=bg + 1 component = single blob, no cleanup needed
            continue

        # Multiple components — score each by DINOv2 similarity
        rgb = data_h5['imgs'][fr, cam]
        fm, ph, pw, _, _ = get_dino_features(dino, rgb)

        best_label, best_score = -1, -1.0
        for lbl in range(1, num_labels):
            comp_mask = (labels == lbl).astype(np.uint8)
            comp_area = int(comp_mask.sum())
            if comp_area < 50:
                continue

            mr = cv2.resize(comp_mask, (pw, ph),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            if mr.sum() < 1:
                continue
            feat = fm[mr].mean(axis=0)
            feat /= (np.linalg.norm(feat) + 1e-8)
            sim = float(feat @ ref_vec)

            # Score: DINOv2 similarity, penalize components far from expected area
            area_ratio = comp_area / max(ref_median_area, 1)
            area_penalty = 1.0 if 0.3 < area_ratio < 3.0 else 0.5
            score = sim * area_penalty

            if score > best_score:
                best_score = score
                best_label = lbl

        if best_label > 0:
            cleaned = (labels == best_label).astype(np.uint8)
            if int(cleaned.sum()) != int(mask.sum()):
                masks_ds[fr, cam] = cleaned
                modified += 1

    return modified


# ═══════════════════════════════════════════════════════════════════════
#  Cross-view completeness refinement (object-agnostic)
# ═══════════════════════════════════════════════════════════════════════

def pick_completeness_ref_cam(masks_ds, N, n_cams, validation_results=None):
    """Pick the camera with the most 'complete' masks across time.

    Score = median_area * median_sim / (1 + area_cv).
    Favors large + stable + high-DINO-similarity masks.
    """
    best_cam, best_score = 0, -1.0
    for cam in range(n_cams):
        areas = [int(masks_ds[fr, cam].sum()) for fr in range(0, N, 20)]
        areas = [a for a in areas if a > 50]
        if len(areas) < 5:
            continue
        med = float(np.median(areas))
        cv = float(np.std(areas)) / max(med, 1)
        med_sim = 1.0
        if validation_results and cam in validation_results:
            med_sim = validation_results[cam].get('median_sim', 1.0)
        score = med * med_sim / (1.0 + cv)
        if score > best_score:
            best_score = score; best_cam = cam
    return best_cam, best_score


def build_completeness_bank(dino, data_h5, masks_ds, ref_cam, N,
                            top_k=30, max_patches=3000):
    """Collect DINOv2 patch features from the highest-area frames of ref_cam.

    Each patch = a piece of the object. Bank spans multiple poses/occlusions.
    Returns (N, dim) L2-normalized array, or None if too few frames.
    """
    frame_areas = [(fr, int(masks_ds[fr, ref_cam].sum())) for fr in range(N)]
    frame_areas = [x for x in frame_areas if x[1] > 100]
    frame_areas.sort(key=lambda x: x[1], reverse=True)
    sample = [fr for fr, _ in frame_areas[:top_k]]

    patches = []
    for fr in sample:
        rgb = data_h5['imgs'][fr, ref_cam]
        mask = masks_ds[fr, ref_cam]
        fm, ph, pw, _, _ = get_dino_features(dino, rgb)
        mr = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
        if mr.sum() < 1:
            continue
        patches.append(fm[mr])

    if not patches:
        return None
    bank = np.concatenate(patches, axis=0)
    bank = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
    if len(bank) > max_patches:
        idx = np.random.RandomState(0).choice(len(bank), max_patches, replace=False)
        bank = bank[idx]
    return bank.astype(np.float32)


def _spread_points(xs, ys, probs, k=3, min_dist=30):
    """Greedy: pick up to k high-prob points at least min_dist apart."""
    order = np.argsort(-probs)
    picked = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if all((y - py) ** 2 + (x - px) ** 2 > min_dist ** 2 for py, px in picked):
            picked.append((y, x))
            if len(picked) >= k:
                break
    return picked


def cross_view_refine_camera(dino, image_predictor, data_h5, masks_ds, cam, bank, N,
                              sim_thresh=0.55, coverage_thresh=0.70, adjacency_px=40,
                              grow_shrink_ratio=(1.0, 4.0), min_mask_iou=0.8,
                              mesh_ref=None, mesh_gate_thresh=0.35,
                              mesh_gate_frac=0.70):
    """Grow partial masks on this camera using DINO patch bank + SAM2 re-click.

    For each frame:
      1. Compute DINOv2 patch heatmap (max sim to bank, per patch).
      2. If coverage (high-sim mass inside mask / total high-sim mass) >= threshold, skip.
      3. Find high-sim patches outside mask but adjacent (within adjacency_px).
      4. Click SAM2 at top spread-out patches + retain mask. Accept if:
         - new_area in [old_area * lo, old_area * hi]
         - intersection(new, old) / area(old) >= min_mask_iou
         - new mask's bank similarity >= old's
    Returns number of frames refined.
    """
    refined = 0
    bank_t = torch.from_numpy(bank).cuda()  # (B, dim)

    for fr in range(N):
        mask = masks_ds[fr, cam]
        old_area = int(mask.sum())
        if old_area < 100:
            continue

        rgb = data_h5['imgs'][fr, cam]
        fm, ph, pw, fh, fw = get_dino_features(dino, rgb)
        flat = fm.reshape(-1, fm.shape[-1])
        flat_n = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        # max-sim per patch via GPU matmul (chunks avoid OOM for tiny inputs, ok here)
        with torch.no_grad():
            f_t = torch.from_numpy(flat_n).cuda()
            sims = (f_t @ bank_t.T).max(dim=1).values.cpu().numpy()
        sim_map_patch = sims.reshape(ph, pw)
        sim_map = cv2.resize(sim_map_patch.astype(np.float32), (W, H),
                             interpolation=cv2.INTER_LINEAR)

        high = (sim_map > sim_thresh).astype(np.uint8)
        if high.sum() < 50:
            continue

        inside_high = int(((mask > 0) & (high > 0)).sum())
        total_high = int(high.sum())
        coverage = inside_high / max(total_high, 1)
        if coverage >= coverage_thresh:
            continue

        # Adjacent high-sim region outside mask
        kernel = np.ones((adjacency_px, adjacency_px), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel)
        add_region = (high > 0) & (mask == 0) & (mask_dilated > 0)
        if add_region.sum() < 80:
            continue

        ys, xs = np.where(add_region)
        probs = sim_map[ys, xs]
        new_pts = _spread_points(xs, ys, probs, k=3, min_dist=30)
        if not new_pts:
            continue

        # Also seed current mask centroid as a positive anchor
        m_ys, m_xs = np.where(mask > 0)
        cy, cx = int(np.median(m_ys)), int(np.median(m_xs))
        all_pts = [(cy, cx)] + new_pts
        pts_xy = np.array([[x, y] for (y, x) in all_pts], dtype=np.float32)
        lbls = np.ones(len(all_pts), dtype=np.int32)

        image_predictor.set_image(rgb)
        new_masks, scores, _ = image_predictor.predict(
            point_coords=pts_xy, point_labels=lbls, multimask_output=True)

        # Pick the best candidate meeting constraints
        best_new, best_bank_sim = None, -1.0
        lo, hi = grow_shrink_ratio
        # Old mask's bank similarity
        mr_old = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
        old_bank_sim = 0.0
        if mr_old.sum() > 0:
            old_feat = fm[mr_old].mean(axis=0)
            old_feat /= (np.linalg.norm(old_feat) + 1e-8)
            with torch.no_grad():
                of = torch.from_numpy(old_feat).cuda()
                old_bank_sim = float((bank_t @ of).max().item())

        for mi in range(new_masks.shape[0]):
            cand = new_masks[mi].astype(np.uint8)
            new_area = int(cand.sum())
            if new_area < old_area * lo or new_area > old_area * hi:
                continue
            inter = int(((cand > 0) & (mask > 0)).sum())
            if inter / max(old_area, 1) < min_mask_iou:
                continue
            mr = cv2.resize(cand, (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
            if mr.sum() < 1:
                continue
            new_feat = fm[mr].mean(axis=0)
            new_feat /= (np.linalg.norm(new_feat) + 1e-8)
            with torch.no_grad():
                nf = torch.from_numpy(new_feat).cuda()
                new_bank_sim = float((bank_t @ nf).max().item())
            if new_bank_sim < old_bank_sim - 0.02:
                continue

            # Mesh-ref gate: require ADDED patches to match mesh reference
            # This blocks growth into hand/table/background, which DINOv2 bank
            # (built from SAM2 masks) can't reliably reject.
            if mesh_ref is not None:
                added_patch_mask = cv2.resize((cand > 0).astype(np.uint8) & (mask == 0).astype(np.uint8),
                                              (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
                if added_patch_mask.sum() >= 2:
                    added_feats = fm[added_patch_mask]
                    added_feats = added_feats / (np.linalg.norm(added_feats, axis=1, keepdims=True) + 1e-8)
                    mesh_sims_added = added_feats @ mesh_ref  # (n_added,)
                    frac_good = float((mesh_sims_added > mesh_gate_thresh).mean())
                    if frac_good < mesh_gate_frac:
                        continue

            if new_bank_sim > best_bank_sim:
                best_bank_sim = new_bank_sim; best_new = cand

        if best_new is not None and int(best_new.sum()) > old_area:
            masks_ds[fr, cam] = best_new
            refined += 1

    del bank_t; torch.cuda.empty_cache()
    return refined


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
    """Generate snapshot PNGs + MP4 video of mask overlay for all cameras.
    Grid is auto-computed from the h5 (works for 1..N cameras)."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    mf = h5py.File(masks_path, 'r')
    N = data_h5['imgs'].shape[0]
    n_cams = data_h5['imgs'].shape[1]
    # Use the H5's actual image resolution rather than module-level H,W
    # constants — those default to 480x640 and only get overridden inside main().
    H = int(data_h5['imgs'].shape[2])
    W = int(data_h5['imgs'].shape[3])
    # canvas grid: up to 4 cols, rows as needed
    cols = min(n_cams, 4)
    rows = max(1, (n_cams + cols - 1) // cols)
    snap_frames = [0, 50, 100, 200, 350, 500, 700, min(N-1, 756)]

    def _render(fr):
        canvas = np.zeros((H*rows, W*cols, 3), dtype=np.uint8)
        for c in range(n_cams):
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
            r, col = c // cols, c % cols
            canvas[r*H:(r+1)*H, col*W:(col+1)*W] = rgb
            cv2.putText(canvas, f"cam{c} {info}", (col*W+5, r*H+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        cv2.putText(canvas, f"frame {fr}/{N-1}", (W*cols-200, H*rows-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return canvas

    for fr in snap_frames:
        if fr >= N: continue
        bgr = cv2.cvtColor(_render(fr), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / f'snapshot_f{fr:04d}.png'), bgr)

    if make_video:
        vid = out / 'masks_video.mp4'
        wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*'mp4v'), 30, (W*cols, H*rows))
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
    parser.add_argument('--data_h5', required=True,
        help='Path to H5 file with imgs (N,C,H,W,3) and depths (N,C,H,W) datasets')
    parser.add_argument('--calib_yaml', required=True)
    parser.add_argument('--tool_mesh', required=False, default=None,
        help='Path to textured OBJ mesh of the tool (not required for --phase7_only)')
    parser.add_argument('--output_dir', required=True)
    # Defaults match the actual robotool layout (previously pointed at a
    # non-existent 'ho-cap' subfolder). Override via flags if needed.
    _hocap_root = Path(__file__).resolve().parent.parent      # HO-Cap-Annotation/
    _robotool_root = _hocap_root.parent                       # robotool/
    parser.add_argument('--sam2_ckpt',
        default=str(_robotool_root / 'mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt'))
    parser.add_argument('--sam2_image_cfg',
        default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--sam2_video_cfg',
        default=str(_hocap_root / 'config/sam2_config/sam2.1_hiera_l.yaml'))
    parser.add_argument('--n_mesh_views', type=int, default=12,
        help='Number of viewpoints for mesh rendering')
    parser.add_argument('--anchor_cams', type=int, nargs='+', default=None,
        help='Force these cameras as anchors (skip mesh-based search)')
    # ---- Seed-finding controls (for difficult first-frame / partial-view cases) ----
    parser.add_argument('--mesh_scan_every', type=int, default=10,
        help='Frame stride when scanning each cam for the mesh-DINO anchor '
             'seed. Smaller = denser scan = better chance of catching a frame '
             'where the object is clearly visible (at linear compute cost). '
             'Try 5 or 3 for videos where frame 0 of many cams misses the tool.')
    parser.add_argument('--dense_scan_every', type=int, default=5,
        help='Frame stride for the dense cross-view seeding pass (non-anchor '
             'cameras). Smaller = denser.')
    parser.add_argument('--seed_min_area', type=int, default=100,
        help='Minimum SAM2 mask area (pixels) during seeding. Lower this '
             '(e.g. 30) to accept small / partially-visible objects at the '
             'start of the sequence.')
    parser.add_argument('--seed_max_area', type=int, default=15000,
        help='Maximum SAM2 mask area during seeding. Raise for large tools.')
    parser.add_argument('--seed_min_sim', type=float, default=0.20,
        help='Minimum DINOv2 patch-to-mesh similarity to consider a seed. '
             'Lower (e.g. 0.12) to accept partial-view seeds.')
    parser.add_argument('--seed_fast', action='store_true',
        help='Fast seeding mode: for each camera try frame 0 first, then '
             'frame `mesh_scan_every` (default 10 or whatever you set), then '
             '2*stride, ... and STOP at the first frame that yields any valid '
             'seed. Skips the default "score all frames, pick best" sweep. '
             'Makes the anchor-search phase roughly `scan_every`x faster '
             'when the object is visible in most frames. Pair with '
             '--mesh_scan_every 20 for the "frame 0, else frame 20, 40, ..." '
             'behaviour.')
    parser.add_argument('--min_area', type=int, default=200,
        help='Minimum median mask area to pass validation')
    parser.add_argument('--max_area', type=int, default=8000,
        help='Maximum median mask area to pass validation')
    parser.add_argument('--drift_threshold', type=float, default=0.35,
        help='Re-seed if area drops below this fraction of rolling reference')
    parser.add_argument('--chunk_size', type=int, default=100,
        help='Frames per SAM2 video propagation chunk (drift check cadence)')
    parser.add_argument('--min_sim', type=float, default=0.45,
        help='Minimum DINOv2 similarity to pass validation')
    parser.add_argument('--cross_view_refine', action='store_true',
        help='Run Phase 7: cross-view completeness pass to grow partial masks')
    parser.add_argument('--cvr_sim_thresh', type=float, default=0.55,
        help='DINOv2 patch similarity threshold for "object-like" patches in Phase 7')
    parser.add_argument('--cvr_coverage', type=float, default=0.70,
        help='Skip refinement if mask already covers this fraction of high-sim mass')
    parser.add_argument('--cvr_adjacency_px', type=int, default=40,
        help='Only consider high-sim patches within this many px of current mask')
    parser.add_argument('--cvr_bank_frames', type=int, default=30,
        help='Number of top-area frames from reference cam to build patch bank')
    parser.add_argument('--cvr_mesh_gate_thresh', type=float, default=0.35,
        help='Per-patch mesh-ref similarity threshold for added-region gate')
    parser.add_argument('--cvr_mesh_gate_frac', type=float, default=0.70,
        help='Require this fraction of added patches above mesh-gate threshold')
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--frame0_only', action='store_true',
        help='Simple/fast mode: for every camera, run DINOv2+SAM2 seeding ONLY '
             'on frame 0, then SAM2-propagate forward/backward with no drift '
             're-seeding and no iterative validation. Skips Phase 3/3b/4/5/5b/'
             '6/7. Use this when DINO on the cluster is too slow and you are '
             'OK with per-camera mask quality being determined by frame 0. '
             'Cameras whose frame-0 seed fails are left with empty masks.')
    parser.add_argument('--cameras', type=int, nargs='+', default=None,
        help='Only process these cameras (reuses existing masks.h5 for others)')
    parser.add_argument('--existing_masks', type=str, default=None,
        help='Path to existing masks.h5 to copy good cameras from')
    parser.add_argument('--phase7_only', type=str, default=None,
        help='Standalone mode: path to existing masks.h5. Skip all propagation, '
             'only run Phase 7 cross-view refinement, save to output_dir/masks.h5.')
    # ---- pipeline-format export (for run_mydata/run_task_folder integration) ----
    parser.add_argument('--pipeline_tool_masks_dir', type=str, default=None,
        help='If set, also export per-frame label-mask .npy files and '
             'objects.yaml into this dir in the format the HO-Cap-Annotation '
             'pipeline expects (tool_masks/cam{i}_rgb/{frame:04d}.npy). '
             'generate_meta.py reads masks from here.')
    parser.add_argument('--tool_name', type=str, default='tool',
        help='Tool/object name, written into objects.yaml (single-object mode).')
    parser.add_argument('--cam_serials', type=str, nargs='+', default=None,
        help='Ordered list of camera serials (e.g. "00 01 02 03 04 05 06 07"). '
             'Used to name camera subfolders as cam{serial}_rgb/ . If omitted, '
             'falls back to integer index "cam{i}_rgb".')
    parser.add_argument('--pipeline_mask_format', choices=['npz', 'npy'], default='npz',
        help='Format for per-frame pipeline masks. Default npz (compressed, '
             '~20-100x smaller than npy for binary masks). generate_meta.py '
             'reads both — switch to npy only for legacy compatibility.')
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

    data_h5 = h5py.File(args.data_h5, 'r')
    N = data_h5['imgs'].shape[0]
    n_cams = data_h5['imgs'].shape[1]
    # Override module-level H,W to match the actual H5 image resolution so
    # masks_ds has the right shape (older defaults assumed 480x640, but cluster
    # recordings are typically 720x1280).
    global H, W
    H = int(data_h5['imgs'].shape[2])
    W = int(data_h5['imgs'].shape[3])
    print(f'[INFO] {N} frames, {n_cams} cameras, image {H}x{W}')

    def _elapsed():
        return f'[{time.time() - t_start:.0f}s]'

    # ── Standalone Phase 7 mode ──────────────────────────────────────
    if args.phase7_only is not None:
        src_masks_path = Path(args.phase7_only)
        if not src_masks_path.exists():
            print(f'  ERROR: {src_masks_path} does not exist')
            data_h5.close(); return
        masks_path = output_dir / 'masks.h5'
        print(f'{_elapsed()} === Standalone Phase 7 ===')
        print(f'  Copying {src_masks_path} -> {masks_path}')
        import shutil
        shutil.copy(str(src_masks_path), str(masks_path))

        print(f'{_elapsed()} Loading DINOv2 + SAM2 image predictor...')
        dino = load_dino()
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_img = build_sam2(args.sam2_image_cfg, args.sam2_ckpt, device='cuda:0')
        image_predictor = SAM2ImagePredictor(sam2_img)

        # Build mesh reference for gate (blocks growth into hand/table)
        mesh_ref_gate = None
        if args.tool_mesh is not None:
            print(f'{_elapsed()} Building mesh DINOv2 reference for gate...')
            mesh_ref_gate = build_mesh_reference(dino, args.tool_mesh, n_views=args.n_mesh_views)
        else:
            print('  WARNING: --tool_mesh not provided; skipping mesh gate (growth may leak)')

        masks_h5 = h5py.File(masks_path, 'r+')
        masks_ds = masks_h5['masks']

        ref_cam, ref_score = pick_completeness_ref_cam(masks_ds, N, n_cams, None)
        print(f'  Reference cam for completeness: cam{ref_cam} (score={ref_score:.1f})')

        print(f'  Building completeness bank from top-{args.cvr_bank_frames} frames...')
        bank = build_completeness_bank(dino, data_h5, masks_ds, ref_cam, N,
                                        top_k=args.cvr_bank_frames)
        if bank is None:
            print('  ERROR: could not build bank')
            masks_h5.close(); data_h5.close(); return
        print(f'  Bank: {bank.shape[0]} patches, dim={bank.shape[1]}')

        for cam in range(n_cams):
            if cam == ref_cam:
                continue
            print(f'  {_elapsed()} cam{cam}: refining...')
            n_ref = cross_view_refine_camera(
                dino, image_predictor, data_h5, masks_ds, cam, bank, N,
                sim_thresh=args.cvr_sim_thresh,
                coverage_thresh=args.cvr_coverage,
                adjacency_px=args.cvr_adjacency_px,
                mesh_ref=mesh_ref_gate,
                mesh_gate_thresh=args.cvr_mesh_gate_thresh,
                mesh_gate_frac=args.cvr_mesh_gate_frac)
            masks_h5.flush()
            print(f'    cam{cam}: refined {n_ref} frames')

        masks_h5.close()
        del dino, image_predictor; torch.cuda.empty_cache(); gc.collect()

        print(f'\n{_elapsed()} === Generating visualizations ===')
        viz_dir = output_dir / 'viz'
        visualize(data_h5, masks_path, viz_dir, make_video=not args.no_video)

        data_h5.close()
        elapsed = time.time() - t_start
        print(f'\n[DONE] Phase 7 standalone: {elapsed/60:.1f} min ({elapsed:.0f} sec)')
        return

    # ── Load models ──────────────────────────────────────────────────
    if args.tool_mesh is None:
        print('  ERROR: --tool_mesh required unless --phase7_only is set')
        data_h5.close(); return
    print(f'{_elapsed()} Loading DINOv2...')
    dino = load_dino()

    # ── Phase 1: Build mesh reference + find anchor cameras ──────────
    print(f'\n{_elapsed()} === Phase 1: Building mesh DINOv2 reference ===')
    mesh_ref = build_mesh_reference(dino, args.tool_mesh, n_views=args.n_mesh_views)
    if mesh_ref is None:
        print('  ERROR: Failed to build mesh reference.')
        data_h5.close()
        return

    print(f'\n{_elapsed()} === Phase 1b: Finding anchor cameras via mesh reference ===')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_img = build_sam2(args.sam2_image_cfg, args.sam2_ckpt, device='cuda:0')
    image_predictor = SAM2ImagePredictor(sam2_img)

    seeds = {}

    # ── Frame-0-only fast mode ───────────────────────────────────────
    # Seed every camera using only frame 0 via the mesh DINOv2 reference,
    # then SAM2-propagate with re-seeding disabled. Skips Phase 3–7.
    if args.frame0_only:
        print(f'\n{_elapsed()} === frame0_only: seeding every camera at frame 0 ===')
        for cam in range(n_cams):
            # scan_every=N+1 guarantees only frame 0 is tested; first_hit stops
            # immediately on success.
            result = mesh_dino_seed(
                dino, data_h5, cam, mesh_ref, image_predictor,
                scan_every=N + 1,
                min_area=args.seed_min_area,
                max_area=args.seed_max_area,
                min_sim=args.seed_min_sim,
                first_hit=True)
            if result:
                seeds[cam] = result
                print(f'  cam{cam}: frame0 seed OK (area={result["area"]}, '
                      f'mesh_sim={result["mesh_sim"]:.3f})')
            else:
                print(f'  cam{cam}: frame0 seed FAILED (mask left empty)')

        if not seeds:
            print('  ERROR: No camera produced a valid frame-0 seed.')
            data_h5.close()
            return

        print(f'\n{_elapsed()} Loading SAM2 video predictor ...')
        from hocap_annotation.wrappers.sam2 import build_sam2_video_predictor
        video_predictor = build_sam2_video_predictor(
            config_file=args.sam2_video_cfg, ckpt_path=args.sam2_ckpt, device='cuda:0')

        masks_path = output_dir / 'masks.h5'
        seeded_cams = sorted(seeds.keys())
        extract_jpegs(data_h5, tmp_dir, seeded_cams)

        masks_h5 = h5py.File(masks_path, 'w')
        masks_ds = masks_h5.create_dataset(
            'masks', shape=(N, n_cams, H, W),
            dtype=np.uint8, chunks=(1, 1, H, W), compression='gzip')

        print(f'\n{_elapsed()} === frame0_only: propagating (no re-seeding) ===')
        for cam in seeded_cams:
            s = seeds[cam]
            cam_dir = tmp_dir / f'cam{cam:02d}'
            print(f'  cam{cam}: propagating from frame {s["frame"]} ...')
            # drift_threshold=0.0 makes the drift check `area >= 0 * ref` always
            # true, so propagate_camera never triggers a DINO re-seed.
            written, reseeds = propagate_camera(
                video_predictor, cam_dir, s['mask'], s['frame'], N,
                masks_ds, cam,
                dino=None, image_predictor=None, global_ref=None, data_h5=None,
                ref_area=float(s['mask'].sum()),
                drift_threshold=0.0, chunk_size=args.chunk_size)
            masks_h5.flush()
            print(f'  cam{cam}: wrote {written} frames')

        del dino, image_predictor, video_predictor
        torch.cuda.empty_cache(); gc.collect()

        meta = {f'cam{c}': {k: v for k, v in s.items() if k != 'mask'}
                for c, s in seeds.items()}
        with open(output_dir / 'seed_info.json', 'w') as fp:
            json.dump(meta, fp, indent=2, default=str)
        masks_h5.close()

        if args.pipeline_tool_masks_dir is not None:
            print(f'\n{_elapsed()} === Exporting to pipeline format ===')
            export_pipeline_masks(
                masks_path=masks_path,
                out_dir=Path(args.pipeline_tool_masks_dir),
                tool_name=args.tool_name,
                cam_serials=args.cam_serials,
                n_cams=n_cams,
                n_frames=N,
                fmt=args.pipeline_mask_format,
            )

        print(f'\n{_elapsed()} === Generating visualizations ===')
        viz_dir = output_dir / 'viz'
        try:
            visualize(data_h5, masks_path, viz_dir, make_video=not args.no_video)
        except Exception as e:
            print(f'  [WARN] visualize failed: {e}')
            print('         masks.h5 and pipeline-format masks are still valid.')

        data_h5.close()
        elapsed = time.time() - t_start
        print(f'\n[DONE] frame0_only: {elapsed/60:.1f} min ({elapsed:.0f} sec)')
        return

    if args.anchor_cams:
        print(f'  Forced anchors: {args.anchor_cams}')
        for cam in args.anchor_cams:
            result = mesh_dino_seed(dino, data_h5, cam, mesh_ref, image_predictor,
                                      scan_every=args.mesh_scan_every,
                                      min_area=args.seed_min_area,
                                      max_area=args.seed_max_area,
                                      min_sim=args.seed_min_sim,
                                      first_hit=args.seed_fast)
            if result:
                seeds[cam] = result
                print(f'  cam{cam}: seed OK (frame={result["frame"]}, area={result["area"]}, '
                      f'sim={result["score"]:.3f})')
            else:
                print(f'  cam{cam}: seed FAILED')
    else:
        # Scan all cameras, pick best 2 as anchors
        all_results = {}
        for cam in range(n_cams):
            print(f'  {_elapsed()} cam{cam}: scanning frames...')
            result = mesh_dino_seed(dino, data_h5, cam, mesh_ref, image_predictor,
                                      scan_every=args.mesh_scan_every,
                                      min_area=args.seed_min_area,
                                      max_area=args.seed_max_area,
                                      min_sim=args.seed_min_sim,
                                      first_hit=args.seed_fast)
            if result:
                all_results[cam] = result
                print(f'    cam{cam}: score={result["score"]:.4f} frame={result["frame"]} '
                      f'area={result["area"]}')
            else:
                print(f'    cam{cam}: no match')

        if all_results:
            ranked = sorted(all_results.items(), key=lambda x: x[1]['score'], reverse=True)
            anchor_cams = [c for c, _ in ranked[:2]]
            for cam in anchor_cams:
                seeds[cam] = all_results[cam]
            scores_str = [f'{all_results[c]["score"]:.4f}' for c in anchor_cams]
            print(f'  Selected anchors: {anchor_cams} (scores: {scores_str})')
        else:
            print('  ERROR: No cameras matched the mesh reference.')
            data_h5.close()
            return

    if not seeds:
        print('  ERROR: No anchor seeds generated.')
        data_h5.close()
        return

    # ── Load all models (46GB L40S can hold all simultaneously) ──────
    print(f'\n{_elapsed()} Loading SAM2 video predictor (keeping all models loaded)...')
    from hocap_annotation.wrappers.sam2 import build_sam2_video_predictor
    video_predictor = build_sam2_video_predictor(
        config_file=args.sam2_video_cfg, ckpt_path=args.sam2_ckpt, device='cuda:0')

    masks_path = output_dir / 'masks.h5'

    # ── --cameras mode: reuse existing masks, only redo specified cameras ──
    if args.cameras is not None:
        existing = args.existing_masks or str(masks_path)
        if not Path(existing).exists():
            print(f'  ERROR: No existing masks at {existing}')
            data_h5.close(); return

        print(f'\n{_elapsed()} === Camera-specific mode: redoing {args.cameras} ===')
        print(f'  Copying existing masks from {existing}')

        # Open existing and create new masks.h5
        src_h5 = h5py.File(existing, 'r')
        tmp_masks_path = output_dir / 'masks_new.h5'
        masks_h5 = h5py.File(tmp_masks_path, 'w')
        masks_ds = masks_h5.create_dataset('masks', shape=(N, n_cams, H, W),
                                            dtype=np.uint8, chunks=(1, 1, H, W),
                                            compression='gzip')
        # Copy good cameras from existing file
        good_cams = [c for c in range(n_cams) if c not in args.cameras]
        for cam in good_cams:
            print(f'  Copying cam{cam} from existing masks...')
            for fr in range(N):
                try:
                    masks_ds[fr, cam] = src_h5['masks'][fr, cam]
                except Exception:
                    masks_ds[fr, cam] = np.zeros((H, W), dtype=np.uint8)
        masks_h5.flush()
        src_h5.close()

        # Find best good camera for reference
        best_ref_cam, best_cv = good_cams[0], 999.0
        for cam in good_cams:
            areas = [int(masks_ds[fr, cam].sum()) for fr in range(50, N - 50, 50)]
            if areas:
                med = float(np.median(areas))
                cv_val = float(np.std(areas)) / max(med, 1) if len(areas) > 1 else 0
                if cv_val < best_cv:
                    best_cv = cv_val; best_ref_cam = cam
        print(f'  Using cam{best_ref_cam} as reference (cv={best_cv:.2f})')

        ref_feats, global_ref = build_rich_reference(dino, data_h5, masks_ds, best_ref_cam, sample_every=10)
        if global_ref is None:
            print('  ERROR: Could not build reference.')
            masks_h5.close(); data_h5.close(); return

        ref_areas = [int(masks_ds[fr, best_ref_cam].sum())
                     for fr in range(0, N, 10) if masks_ds[fr, best_ref_cam].sum() > 50]
        ref_median_area = float(np.median(ref_areas)) if ref_areas else None
        print(f'  Reference median area: {ref_median_area:.0f}')

        extract_jpegs(data_h5, tmp_dir, args.cameras)
        remaining_cams = args.cameras

    else:
        # ── Full pipeline: Phase 2-3 ────────────────────────────────────
        # Phase 2: Propagate anchor cameras (with re-seeding)
        print(f'\n{_elapsed()} === Phase 2: Propagating anchor cameras ===')
        anchor_cams = list(seeds.keys())
        extract_jpegs(data_h5, tmp_dir, anchor_cams)

        masks_h5 = h5py.File(masks_path, 'w')
        masks_ds = masks_h5.create_dataset('masks', shape=(N, n_cams, H, W),
                                            dtype=np.uint8, chunks=(1, 1, H, W),
                                            compression='gzip')
        for cam in anchor_cams:
            s = seeds[cam]
            cam_dir = tmp_dir / f'cam{cam:02d}'
            print(f'  cam{cam}: propagating from frame {s["frame"]} (with re-seeding)...')
            written, reseeds = propagate_camera(
                video_predictor, cam_dir, s['mask'], s['frame'], N, masks_ds, cam,
                dino=dino, image_predictor=image_predictor, global_ref=mesh_ref,
                data_h5=data_h5, ref_area=float(s['mask'].sum()),
                drift_threshold=args.drift_threshold, chunk_size=args.chunk_size)
            masks_h5.flush()
            print(f'  cam{cam}: wrote {written} frames, {reseeds} re-seeds')

        # Phase 3: Build rich DINOv2 reference from real camera
        print(f'\n{_elapsed()} === Phase 3: Building rich DINOv2 reference ===')

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
            masks_h5.close(); data_h5.close(); return

        ref_areas = [int(masks_ds[fr, best_ref_cam].sum())
                     for fr in range(0, N, 10) if masks_ds[fr, best_ref_cam].sum() > 50]
        ref_median_area = float(np.median(ref_areas)) if ref_areas else None
        print(f'  Reference median area: {ref_median_area:.0f}')

        # Phase 3b: Clean anchor masks (remove leaked components)
        print(f'\n{_elapsed()} === Phase 3b: Cleaning anchor masks (CC filtering) ===')
        for cam in anchor_cams:
            mod = clean_masks_cc(dino, data_h5, masks_ds, cam, global_ref, N)
            masks_h5.flush()
            if mod > 0:
                print(f'  cam{cam}: cleaned {mod} frames')
            else:
                print(f'  cam{cam}: no cleanup needed')

        remaining_cams = [c for c in range(n_cams) if c not in anchor_cams]
        extract_jpegs(data_h5, tmp_dir, remaining_cams)

    for iteration in range(MAX_ITERS):
        cams_to_process = remaining_cams if iteration == 0 else failed_cams

        if not cams_to_process:
            print(f'\n  All cameras passed validation!')
            break

        print(f'\n{_elapsed()} === Iteration {iteration + 1}/{MAX_ITERS}: '
              f'Processing cameras {cams_to_process} ===')

        # Phase 4: Dense DINOv2 cross-view seeding for remaining cameras
        print(f'\n  {_elapsed()} [Phase 4] Seeding {len(cams_to_process)} cameras via dense DINOv2 scan...')
        for cam in cams_to_process:
            result = dense_dino_seed(dino, data_h5, cam, global_ref, ref_feats,
                                      image_predictor, ref_median_area=ref_median_area,
                                      scan_every=args.dense_scan_every,
                                      first_hit=args.seed_fast)
            if result:
                seeds[cam] = result
                print(f'    cam{cam}: seed at frame {result["frame"]} '
                      f'area={result["area"]} score={result["score"]:.3f}')
            else:
                print(f'    cam{cam}: seed FAILED')

        # Phase 5: Propagate with re-seeding
        print(f'\n  {_elapsed()} [Phase 5] Propagating with re-seeding...')
        for cam in cams_to_process:
            if cam not in seeds:
                continue
            s = seeds[cam]
            cam_dir = tmp_dir / f'cam{cam:02d}'
            print(f'    {_elapsed()} cam{cam}: propagating from frame {s["frame"]}...')
            # Clear old masks for this camera
            for fr in range(N):
                masks_ds[fr, cam] = np.zeros((H, W), dtype=np.uint8)
            written, reseeds = propagate_camera(
                video_predictor, cam_dir, s['mask'], s['frame'], N, masks_ds, cam,
                dino=dino, image_predictor=image_predictor, global_ref=global_ref,
                data_h5=data_h5, ref_area=ref_median_area,
                drift_threshold=args.drift_threshold, chunk_size=args.chunk_size)
            masks_h5.flush()
            print(f'    cam{cam}: wrote {written} frames, {reseeds} re-seeds')

        # Phase 5b: Clean masks via connected component filtering
        print(f'\n  {_elapsed()} [Phase 5b] Cleaning masks (CC filtering)...')
        for cam in cams_to_process:
            if cam not in seeds:
                continue
            mod = clean_masks_cc(dino, data_h5, masks_ds, cam, global_ref, N)
            masks_h5.flush()
            if mod > 0:
                print(f'    cam{cam}: cleaned {mod} frames (CC filtering)')
            else:
                print(f'    cam{cam}: no cleanup needed')

        # Phase 6: Validate cameras with DINOv2
        # In --cameras mode, only validate the cameras we're processing
        validate_cams = remaining_cams if args.cameras is not None else list(range(n_cams))
        print(f'\n  {_elapsed()} [Phase 6] Validating cameras {validate_cams}...')

        failed_cams = []
        for cam in validate_cams:
            if cam not in seeds:
                failed_cams.append(cam)
                print(f'    cam{cam}: FAIL (no seed)')
                continue
            passed, reason = validate_camera(
                dino, data_h5, masks_ds, cam, global_ref, N,
                min_area=args.min_area, max_area=args.max_area,
                min_sim=args.min_sim)
            status = 'PASS' if passed else 'FAIL'
            print(f'    cam{cam}: {status} -- {reason}')
            if not passed:
                failed_cams.append(cam)

        # Update reference from best passing camera for next iteration
        # Include good copied cameras as candidates for reference
        all_good = [c for c in range(n_cams) if c not in failed_cams and c not in remaining_cams]
        passing_cams = [c for c in validate_cams if c not in failed_cams] + all_good
        if passing_cams and failed_cams:
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

    # ── Phase 7: Cross-view completeness refinement (optional) ───────
    if args.cross_view_refine:
        print(f'\n{_elapsed()} === Phase 7: Cross-view completeness refinement ===')
        # Collect final validation results for median_sim weighting
        vres = {}
        for cam in range(n_cams):
            passed, reason = validate_camera(
                dino, data_h5, masks_ds, cam, global_ref, N,
                min_area=args.min_area, max_area=args.max_area,
                min_sim=args.min_sim)
            # Parse median_sim out of the reason string (simple extraction)
            if 'median_sim=' in reason:
                try:
                    ms = float(reason.split('median_sim=')[1].split(',')[0].split(')')[0])
                except Exception:
                    ms = 1.0
            else:
                ms = 1.0
            vres[cam] = {'passed': passed, 'median_sim': ms}

        ref_cam, ref_score = pick_completeness_ref_cam(masks_ds, N, n_cams, vres)
        print(f'  Reference cam for completeness: cam{ref_cam} (score={ref_score:.1f})')

        print(f'  Building completeness bank from top-{args.cvr_bank_frames} frames...')
        bank = build_completeness_bank(dino, data_h5, masks_ds, ref_cam, N,
                                        top_k=args.cvr_bank_frames)
        if bank is None:
            print('  WARNING: could not build bank; skipping Phase 7')
        else:
            print(f'  Bank: {bank.shape[0]} patches, dim={bank.shape[1]}')
            for cam in range(n_cams):
                if cam == ref_cam:
                    continue
                print(f'  {_elapsed()} cam{cam}: refining...')
                n_ref = cross_view_refine_camera(
                    dino, image_predictor, data_h5, masks_ds, cam, bank, N,
                    sim_thresh=args.cvr_sim_thresh,
                    coverage_thresh=args.cvr_coverage,
                    adjacency_px=args.cvr_adjacency_px,
                    mesh_ref=mesh_ref,
                    mesh_gate_thresh=args.cvr_mesh_gate_thresh,
                    mesh_gate_frac=args.cvr_mesh_gate_frac)
                masks_h5.flush()
                print(f'    cam{cam}: refined {n_ref} frames')

    # ── Cleanup models ───────────────────────────────────────────────
    del dino, image_predictor, video_predictor
    torch.cuda.empty_cache(); gc.collect()

    # ── Save metadata ────────────────────────────────────────────────
    meta = {}
    for cam, s in seeds.items():
        meta[f'cam{cam}'] = {k: v for k, v in s.items() if k != 'mask'}
    with open(output_dir / 'seed_info.json', 'w') as fp:
        json.dump(meta, fp, indent=2, default=str)

    masks_h5.close()

    # In --cameras mode, replace old masks.h5 with new one
    if args.cameras is not None:
        tmp_masks_path = output_dir / 'masks_new.h5'
        if tmp_masks_path.exists():
            import shutil
            shutil.move(str(tmp_masks_path), str(masks_path))
            print(f'  Replaced {masks_path} with updated masks')

    # ── Final validation report ──────────────────────────────────────
    print(f'\n{_elapsed()} === Final Results ===')
    dino_final = load_dino()
    masks_h5 = h5py.File(masks_path, 'r')
    masks_ds2 = masks_h5['masks']
    all_pass = True
    for cam in range(n_cams):
        passed, reason = validate_camera(
            dino_final, data_h5, masks_ds2, cam, global_ref, N,
            min_area=args.min_area, max_area=args.max_area,
            min_sim=args.min_sim)
        status = 'PASS' if passed else 'FAIL'
        print(f'  cam{cam}: {status} -- {reason}')
        if not passed:
            all_pass = False
    masks_h5.close()
    del dino_final; torch.cuda.empty_cache()

    if all_pass:
        print('\n  ALL CAMERAS PASSED!')
    else:
        print(f'\n  Some cameras failed after {MAX_ITERS} iterations.')
        print('  Check visualizations and consider adjusting --min_area/--max_area/--min_sim')

    # ── Visualize ────────────────────────────────────────────────────
    # ── Pipeline-format export (runs BEFORE viz so a viz failure doesn't skip it) ──
    if args.pipeline_tool_masks_dir is not None:
        print(f'\n{_elapsed()} === Exporting to pipeline format ===')
        export_pipeline_masks(
            masks_path=masks_path,
            out_dir=Path(args.pipeline_tool_masks_dir),
            tool_name=args.tool_name,
            cam_serials=args.cam_serials,
            n_cams=n_cams,
            n_frames=N,
            fmt=args.pipeline_mask_format,
        )

    print(f'\n{_elapsed()} === Generating visualizations ===')
    viz_dir = output_dir / 'viz'
    try:
        visualize(data_h5, masks_path, viz_dir, make_video=not args.no_video)
    except Exception as e:
        print(f'  [WARN] visualize failed: {e}')
        print('         masks.h5 and pipeline-format masks are still valid.')

    data_h5.close()
    elapsed = time.time() - t_start
    print(f'\n[DONE] Total time: {elapsed/60:.1f} min ({elapsed:.0f} sec)')


def export_pipeline_masks(masks_path, out_dir, tool_name, cam_serials, n_cams, n_frames,
                           fmt='npz'):
    """Export a binary masks.h5 into the per-frame + objects.yaml layout that
    HO-Cap-Annotation/preprocess/generate_meta.py consumes.

    For each frame and camera, writes <out_dir>/cam{id}_rgb/{frame:04d}.{ext}
    with value 0 (bg) or 1 (tool). Also writes <out_dir>/objects.yaml.

    fmt: 'npz' (default) -> np.savez_compressed, typically 20-100x smaller
              than .npy for binary masks; generate_meta.py supports both.
         'npy' -> legacy uncompressed.

    Single-object only — dino_tool_segment.py produces one-tool binary masks.
    """
    assert fmt in ('npz', 'npy'), f"fmt must be 'npz' or 'npy', got {fmt}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Camera subfolder naming: prefer serial (e.g., '00'), else integer index.
    def _cam_folder_name(i):
        if cam_serials is not None and i < len(cam_serials):
            s = str(cam_serials[i]).zfill(2)
            return f'cam{s}_rgb'
        return f'cam{i}_rgb'

    with h5py.File(masks_path, 'r') as mf:
        masks = mf['masks']     # (N, n_cams, H, W) uint8, 0 or 1
        for cam in range(n_cams):
            cam_dir = out_dir / _cam_folder_name(cam)
            cam_dir.mkdir(parents=True, exist_ok=True)
            # Clean up stale files from the OTHER format so generate_meta
            # doesn't accidentally read a mismatched one.
            other_ext = 'npy' if fmt == 'npz' else 'npz'
            for old in cam_dir.glob(f'*.{other_ext}'):
                old.unlink()
            written = 0
            total_bytes = 0
            for fr in range(n_frames):
                m = masks[fr, cam]
                # Ensure binary with value 1 for the single tool object so that
                # `mask == (object_idx + 1)` lookups work (object_idx 0 -> 1).
                m = (m > 0).astype(np.uint8)
                if fmt == 'npz':
                    out = cam_dir / f'{fr:04d}.npz'
                    np.savez_compressed(out, mask=m)
                else:
                    out = cam_dir / f'{fr:04d}.npy'
                    np.save(out, m)
                total_bytes += out.stat().st_size
                written += 1
            avg_kb = total_bytes / max(written, 1) / 1024.0
            print(f'  cam{cam}: {written} frames ({fmt}, avg {avg_kb:.1f} KB/frame) '
                  f'-> {cam_dir.name}/')

    with open(out_dir / 'objects.yaml', 'w') as fp:
        yaml.safe_dump({'objects': [tool_name]}, fp)
    print(f'  objects.yaml: [{tool_name}]')
    print(f'  pipeline masks ready at {out_dir}')


def repair_masks():
    """Post-processing repair: fix frames where masks are partial/bad.

    Takes existing masks.h5 (e.g. from v2) and repairs frames where the mask
    area drops significantly below the camera's median. Uses SAM2 image predictor
    with DINOv2 click points to regenerate bad masks independently per frame.

    This is safer than modifying propagation because it only touches bad frames.
    """
    parser = argparse.ArgumentParser(description='Repair bad mask frames')
    parser.add_argument('--data_h5', required=True)
    parser.add_argument('--mask_h5', required=True, help='Input masks.h5 to repair')
    parser.add_argument('--tool_mesh', required=True)
    parser.add_argument('--output_dir', required=True)
    _hocap_root = Path(__file__).resolve().parent.parent / 'ho-cap'
    parser.add_argument('--sam2_ckpt',
        default=str(_hocap_root / 'config/checkpoints/sam2/sam2.1_hiera_large.pt'))
    parser.add_argument('--sam2_image_cfg',
        default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--cameras', type=int, nargs='+', default=None,
        help='Only repair these cameras (default: auto-detect bad ones)')
    parser.add_argument('--area_threshold', type=float, default=0.5,
        help='Repair frames with area < threshold * median_area')
    parser.add_argument('--no_video', action='store_true')
    args = parser.parse_args()

    t_start = time.time()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    def _elapsed():
        return f'[{time.time() - t_start:.0f}s]'

    data_h5 = h5py.File(args.data_h5, 'r')
    src_h5 = h5py.File(args.mask_h5, 'r')
    N = data_h5['imgs'].shape[0]
    n_cams = data_h5['imgs'].shape[1]
    print(f'[INFO] {N} frames, {n_cams} cameras')
    print(f'[INFO] Source masks: {args.mask_h5}')

    # Copy source masks to output
    masks_path = Path(args.output_dir) / 'masks.h5'
    print(f'{_elapsed()} Copying source masks to {masks_path}...')
    masks_h5 = h5py.File(masks_path, 'w')
    masks_ds = masks_h5.create_dataset('masks', shape=(N, n_cams, H, W),
                                        dtype=np.uint8, chunks=(1, 1, H, W),
                                        compression='gzip')
    for cam in range(n_cams):
        for fr in range(N):
            masks_ds[fr, cam] = src_h5['masks'][fr, cam]
        print(f'  cam{cam}: copied')
    masks_h5.flush()
    src_h5.close()

    # Compute per-camera statistics
    print(f'\n{_elapsed()} Analyzing mask quality...')
    cam_stats = {}
    for cam in range(n_cams):
        areas = [int(masks_ds[fr, cam].sum()) for fr in range(N)]
        valid_areas = [a for a in areas if a > 50]
        if valid_areas:
            med = float(np.median(valid_areas))
            cam_stats[cam] = {'areas': areas, 'median': med, 'valid_count': len(valid_areas)}
        else:
            cam_stats[cam] = {'areas': areas, 'median': 0, 'valid_count': 0}

    # Compute cross-camera reference area: median of all cameras' medians
    all_medians = [cam_stats[c]['median'] for c in range(n_cams) if cam_stats[c]['median'] > 0]
    cross_ref_area = float(np.median(all_medians)) if all_medians else 0
    print(f'  Cross-camera reference area: {cross_ref_area:.0f}')

    for cam in range(n_cams):
        s = cam_stats[cam]
        # Use cross-camera reference if this camera's median seems too low (drift)
        target = max(s['median'], cross_ref_area * 0.3)
        s['target'] = target
        threshold_val = args.area_threshold * target
        bad_count = sum(1 for a in s['areas'] if a < threshold_val)
        print(f'  cam{cam}: median_area={s["median"]:.0f}, target={target:.0f}, '
              f'bad_frames={bad_count}/{N} (< {threshold_val:.0f})')

    # Determine which cameras need repair
    if args.cameras:
        repair_cams = args.cameras
    else:
        repair_cams = []
        for cam in range(n_cams):
            s = cam_stats[cam]
            if s['target'] == 0:
                continue
            bad = sum(1 for a in s['areas'] if a < args.area_threshold * s['target'])
            if bad > N * 0.05:  # more than 5% bad frames
                repair_cams.append(cam)
    print(f'\n  Cameras to repair: {repair_cams}')

    if not repair_cams:
        print('  No cameras need repair!')
        masks_h5.close(); data_h5.close(); return

    # Load models
    print(f'\n{_elapsed()} Loading DINOv2...')
    dino = load_dino()

    print(f'{_elapsed()} Building mesh DINOv2 reference...')
    mesh_ref = build_mesh_reference(dino, args.tool_mesh)

    # Build rich reference from best camera (not being repaired)
    good_cams = [c for c in range(n_cams) if c not in repair_cams and cam_stats[c]['median'] > 0]
    if good_cams:
        # Pick most stable good camera
        best_ref, best_cv = good_cams[0], 999.0
        for cam in good_cams:
            s = cam_stats[cam]
            valid = [a for a in s['areas'] if a > 50]
            if len(valid) > 1:
                cv_val = float(np.std(valid)) / max(float(np.median(valid)), 1)
                if cv_val < best_cv:
                    best_cv = cv_val; best_ref = cam
        print(f'  Using cam{best_ref} as reference (cv={best_cv:.2f})')
        _, global_ref = build_rich_reference(dino, data_h5, masks_ds, best_ref, sample_every=10)
    else:
        global_ref = mesh_ref
        print(f'  Using mesh reference (no good cameras available)')

    print(f'\n{_elapsed()} Loading SAM2 image predictor...')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_img = build_sam2(args.sam2_image_cfg, args.sam2_ckpt, device='cuda:0')
    image_predictor = SAM2ImagePredictor(sam2_img)

    # Repair bad frames
    total_repaired = 0
    for cam in repair_cams:
        s = cam_stats[cam]
        target = s['target']
        threshold = args.area_threshold * target
        bad_frames = [fr for fr in range(N) if s['areas'][fr] < threshold]
        print(f'\n{_elapsed()} === Repairing cam{cam}: {len(bad_frames)} bad frames '
              f'(threshold={threshold:.0f}, target={target:.0f}) ===')

        repaired = 0
        for fr in bad_frames:
            rgb = data_h5['imgs'][fr, cam]
            fm, ph, pw, fh, fw = get_dino_features(dino, rgb)
            flat = fm.reshape(-1, fm.shape[-1])
            flat_norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
            sims = flat_norm @ global_ref

            # Find top matching patches and use multiple positive points
            top_idx = np.argsort(sims)[::-1][:10]
            good_points = []
            for idx in top_idx:
                if float(sims[idx]) < 0.25:
                    break
                py_p, px_p = idx // pw, idx % pw
                cu = float((px_p * 14 + 7) * (rgb.shape[1] / fw))
                cv_p = float((py_p * 14 + 7) * (rgb.shape[0] / fh))
                good_points.append([cu, cv_p])
                if len(good_points) >= 3:
                    break

            if not good_points:
                continue

            image_predictor.set_image(rgb)

            # Try with multiple positive points for better coverage
            pts = np.array(good_points)
            labels = np.ones(len(pts), dtype=np.int32)
            masks, scores, _ = image_predictor.predict(
                point_coords=pts, point_labels=labels, multimask_output=True)

            # Pick best mask by DINOv2 similarity and area match
            best_mask, best_score = None, -1
            for mi in range(masks.shape[0]):
                m = masks[mi].astype(bool)
                a = int(m.sum())
                if a < 100:
                    continue
                # Area should be reasonable relative to target
                if a > target * 4:
                    continue

                mr = cv2.resize(m.astype(np.uint8), (pw, ph),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                if mr.sum() < 1:
                    continue
                feat = fm[mr].mean(axis=0)
                feat /= (np.linalg.norm(feat) + 1e-8)
                sim = float(feat @ global_ref)

                # Score: DINOv2 similarity * area-closeness-to-target
                area_ratio = min(a / target, target / max(a, 1))
                score = sim * (0.5 + 0.5 * area_ratio)

                if score > best_score:
                    best_score = score
                    best_mask = m.astype(np.uint8)

            if best_mask is not None and best_score > 0.3:
                masks_ds[fr, cam] = best_mask
                repaired += 1

        masks_h5.flush()
        total_repaired += repaired
        print(f'  cam{cam}: repaired {repaired}/{len(bad_frames)} frames')

    print(f'\n{_elapsed()} Total repaired: {total_repaired} frames')

    # Cleanup
    del dino, image_predictor
    torch.cuda.empty_cache(); gc.collect()

    # Visualize
    print(f'\n{_elapsed()} === Generating visualizations ===')
    viz_dir = Path(args.output_dir) / 'viz'
    visualize(data_h5, masks_path, viz_dir, make_video=not args.no_video)

    masks_h5.close()
    data_h5.close()
    elapsed = time.time() - t_start
    print(f'\n[DONE] Total time: {elapsed/60:.1f} min ({elapsed:.0f} sec)')


if __name__ == '__main__':
    import sys
    if '--repair' in sys.argv:
        sys.argv.remove('--repair')
        repair_masks()
    else:
        main()