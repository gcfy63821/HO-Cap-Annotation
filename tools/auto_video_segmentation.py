#!/usr/bin/env python3
"""
Generate tool masks using SAM2 VIDEO predictor with auto-generated seed masks.

Strategy:
1. Extract H5 RGB frames to temporary JPEGs (SAM2 video predictor requires files on disk)
2. Auto-generate seed masks using SAM2 image predictor with spatial+height prior
3. Propagate seed masks through video using SAM2 video predictor (temporal consistency)
4. Aggregate per-camera masks into masks.h5

This mirrors chenrq's successful pipeline (manual seeds → SAM2 video propagation)
but replaces manual annotation with automatic seed generation.
"""

import argparse
import sys
import cv2
import gc
import h5py
import numpy as np
import shutil
import yaml
import torch
from pathlib import Path

# Ensure the ho-cap repo root (parent of tools/) is on sys.path so the
# hocap_annotation package is importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def extract_h5_to_jpegs(hocap_dir, tmp_dir, cam_indices=range(8), frame_range=None):
    """Extract RGB frames from H5 to per-camera JPEG folders.

    Args:
        hocap_dir: Path to hocap directory containing data00000000.h5
        tmp_dir: Where to write JPEGs
        cam_indices: Which cameras to extract
        frame_range: (start, end) tuple or None for all frames

    Returns:
        num_frames, per-camera JPEG directories
    """
    h5_path = Path(hocap_dir) / "data00000000.h5"
    with h5py.File(h5_path, "r") as f:
        num_frames = f["imgs"].shape[0]
        if frame_range:
            start, end = frame_range
            end = min(end, num_frames)
        else:
            start, end = 0, num_frames

        cam_dirs = {}
        for cam_idx in cam_indices:
            cam_dir = Path(tmp_dir) / f"cam{cam_idx:02d}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            cam_dirs[cam_idx] = cam_dir

        for frame_idx in range(start, end):
            for cam_idx in cam_indices:
                rgb = f["imgs"][frame_idx, cam_idx]  # (H, W, 3) uint8 RGB
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                out_path = cam_dirs[cam_idx] / f"color_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), bgr)

            if frame_idx % 100 == 0:
                print(f"  Extracted frame {frame_idx}/{end}")

    print(f"[INFO] Extracted {end - start} frames for {len(cam_indices)} cameras to {tmp_dir}")
    return end - start, cam_dirs


def count_tool_pixels(depth_mm, K, cam_T, H, W, xy_center, xy_radius, z_min, z_max):
    """Count how many pixels fall in the tool spatial region."""
    depth_m = depth_mm.astype(np.float32) / 1000.0
    v, u = np.mgrid[:H, :W]
    pts_cam = np.stack([(u - K[0, 2]) * depth_m / K[0, 0],
                        (v - K[1, 2]) * depth_m / K[1, 1], depth_m],
                       axis=-1).reshape(-1, 3)
    pts_world = pts_cam @ cam_T[:3, :3].T + cam_T[:3, 3]

    valid = depth_m.flatten() > 0.01
    dx = pts_world[:, 0] - xy_center[0]
    dy = pts_world[:, 1] - xy_center[1]
    xy_dist = np.sqrt(dx**2 + dy**2)

    spatial = valid & (xy_dist < xy_radius) & \
              (pts_world[:, 2] > z_min) & (pts_world[:, 2] < z_max)
    return int(spatial.sum())


def get_tool_prompt(depth_mm, K, cam_T, H, W,
                    xy_center, xy_radius, z_min=0.10, z_max=0.30):
    """Get positive (tool) and negative (mortar) point prompts.

    Reused from generate_sam2_masks.py.
    """
    depth_m = depth_mm.astype(np.float32) / 1000.0
    v, u = np.mgrid[:H, :W]
    pts_cam = np.stack([(u - K[0, 2]) * depth_m / K[0, 0],
                        (v - K[1, 2]) * depth_m / K[1, 1], depth_m],
                       axis=-1).reshape(-1, 3)
    pts_world = pts_cam @ cam_T[:3, :3].T + cam_T[:3, 3]

    valid = depth_m.flatten() > 0.01
    dx = pts_world[:, 0] - xy_center[0]
    dy = pts_world[:, 1] - xy_center[1]
    xy_dist = np.sqrt(dx**2 + dy**2)

    spatial = valid & (xy_dist < xy_radius) & \
              (pts_world[:, 2] > z_min) & (pts_world[:, 2] < z_max)

    if spatial.sum() < 10:
        return None, None

    # Top 5% Z → pestle handle
    z_vals = pts_world[:, 2].copy()
    z_vals[~spatial] = -999
    z_thresh = np.percentile(z_vals[spatial], 95)
    top_z = (z_vals.reshape(H, W) > z_thresh)

    if top_z.sum() < 3:
        return None, None

    ys, xs = np.where(top_z)
    pos_pt = (int(xs.mean()), int(ys.mean()))

    # Negative: mortar center (z=0.04-0.09)
    mortar = valid & (xy_dist < xy_radius) & \
             (pts_world[:, 2] > 0.04) & (pts_world[:, 2] < 0.09)
    if mortar.sum() > 50:
        mys, mxs = np.where(mortar.reshape(H, W))
        neg_pt = (int(mxs.mean()), int(mys.mean()))
    else:
        neg_pt = None

    return pos_pt, neg_pt


def find_best_seed_frame(data_h5, cam_idx, K, cam_T, H, W,
                         xy_center, xy_radius, z_min, z_max,
                         sample_every=10):
    """Find the frame with the most tool-region pixels for a given camera."""
    num_frames = data_h5["depths"].shape[0]
    best_frame = 0
    best_count = 0

    for frame_idx in range(0, num_frames, sample_every):
        depth = data_h5["depths"][frame_idx, cam_idx]
        count = count_tool_pixels(depth, K, cam_T, H, W,
                                  xy_center, xy_radius, z_min, z_max)
        if count > best_count:
            best_count = count
            best_frame = frame_idx

    print(f"  cam{cam_idx}: best seed frame={best_frame} ({best_count} tool pixels)")
    return best_frame


def generate_seed_mask(rgb, depth_mm, K, cam_T, H, W,
                       image_predictor, xy_center, xy_radius, z_min, z_max):
    """Generate a seed mask using SAM2 image predictor with spatial prompt.

    Returns:
        mask: (H, W) uint8, 1=pestle, 0=background, or None if failed
    """
    pos_pt, neg_pt = get_tool_prompt(depth_mm, K, cam_T, H, W,
                                     xy_center, xy_radius, z_min, z_max)
    if pos_pt is None:
        return None

    image_predictor.set_image(rgb)

    if neg_pt is not None:
        points = np.array([list(pos_pt), list(neg_pt)])
        labels = np.array([1, 0])
    else:
        points = np.array([list(pos_pt)])
        labels = np.array([1])

    masks, scores, _ = image_predictor.predict(
        point_coords=points, point_labels=labels,
        multimask_output=True)

    # Take smallest valid mask (pestle should be small)
    areas = [m.astype(bool).sum() for m in masks]
    valid_idx = [i for i, a in enumerate(areas) if 100 < a < 8000]
    if valid_idx:
        best = min(valid_idx, key=lambda i: areas[i])
    elif any(100 < a < 15000 for a in areas):
        valid_idx2 = [i for i, a in enumerate(areas) if 100 < a < 15000]
        best = min(valid_idx2, key=lambda i: areas[i])
    else:
        best = min(range(len(areas)), key=lambda i: areas[i])

    mask = masks[best].astype(np.uint8)
    area = mask.sum()
    print(f"    Seed mask area: {area} px ({area/(H*W)*100:.2f}%)")

    if area < 50:
        return None

    return mask


def get_dino_features(model, img_rgb, patch_size=14):
    """Get dense DINOv2 patch features for an image.

    Args:
        model: DINOv2 model
        img_rgb: (H, W, 3) uint8 RGB image
        patch_size: DINOv2 patch size (14)

    Returns:
        features: (n_patches_h, n_patches_w, embed_dim) float32
        patch_h, patch_w: number of patches in each dimension
    """
    from torchvision import transforms

    H, W = img_rgb.shape[:2]
    # Resize to be divisible by patch_size
    new_H = (H // patch_size) * patch_size
    new_W = (W // patch_size) * patch_size

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((new_H, new_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img_rgb).unsqueeze(0).cuda()

    with torch.no_grad():
        features = model.forward_features(img_tensor)
        patch_tokens = features["x_norm_patchtokens"]  # (1, n_patches, embed_dim)

    patch_h = new_H // patch_size
    patch_w = new_W // patch_size
    feat_map = patch_tokens[0].reshape(patch_h, patch_w, -1).cpu().numpy()

    return feat_map, patch_h, patch_w


def get_reference_pestle_features(model, rgb, mask, patch_size=14):
    """Extract average DINOv2 features for the pestle region.

    Args:
        model: DINOv2 model
        rgb: (H, W, 3) uint8
        mask: (H, W) uint8 binary mask of pestle

    Returns:
        ref_feat: (embed_dim,) normalized feature vector
    """
    H, W = rgb.shape[:2]
    feat_map, ph, pw = get_dino_features(model, rgb, patch_size)

    # Resize mask to patch grid
    mask_resized = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)

    # Average features where mask > 0
    pestle_patches = mask_resized > 0
    if pestle_patches.sum() < 1:
        return None

    ref_feat = feat_map[pestle_patches].mean(axis=0)
    ref_feat = ref_feat / (np.linalg.norm(ref_feat) + 1e-8)
    return ref_feat


def find_pestle_by_features(model, rgb, ref_feat, patch_size=14, H=480, W=640):
    """Find the pestle location in a target image using DINOv2 feature matching.

    Args:
        model: DINOv2 model
        rgb: (H, W, 3) uint8 target image
        ref_feat: (embed_dim,) normalized reference feature

    Returns:
        (x, y): pixel coordinates of best matching patch center, or None
        similarity: cosine similarity score
    """
    feat_map, ph, pw = get_dino_features(model, rgb, patch_size)

    # Normalize all patch features
    norms = np.linalg.norm(feat_map, axis=2, keepdims=True) + 1e-8
    feat_norm = feat_map / norms

    # Cosine similarity with reference
    sim_map = (feat_norm * ref_feat[None, None, :]).sum(axis=2)  # (ph, pw)

    # Find best match
    best_idx = np.unravel_index(sim_map.argmax(), sim_map.shape)
    best_sim = sim_map[best_idx]

    # Convert patch coordinates to pixel coordinates in original image
    scale_y = H / (ph * patch_size)
    scale_x = W / (pw * patch_size)
    py = int((best_idx[0] + 0.5) * patch_size * scale_y)
    px = int((best_idx[1] + 0.5) * patch_size * scale_x)

    py = min(max(py, 0), H - 1)
    px = min(max(px, 0), W - 1)

    return (px, py), best_sim


def generate_seed_mask_from_features(rgb, image_predictor, pos_pt, neg_pt=None):
    """Generate seed mask using SAM2 with a feature-matched point prompt.

    Returns:
        mask: (H, W) uint8, or None
    """
    H, W = rgb.shape[:2]
    image_predictor.set_image(rgb)

    if neg_pt is not None:
        points = np.array([list(pos_pt), list(neg_pt)])
        labels = np.array([1, 0])
    else:
        points = np.array([list(pos_pt)])
        labels = np.array([1])

    masks, scores, _ = image_predictor.predict(
        point_coords=points, point_labels=labels,
        multimask_output=True)

    # Take smallest valid mask
    areas = [m.astype(bool).sum() for m in masks]
    valid_idx = [i for i, a in enumerate(areas) if 100 < a < 8000]
    if valid_idx:
        best = min(valid_idx, key=lambda i: areas[i])
    elif any(100 < a < 15000 for a in areas):
        valid_idx2 = [i for i, a in enumerate(areas) if 100 < a < 15000]
        best = min(valid_idx2, key=lambda i: areas[i])
    else:
        best = min(range(len(areas)), key=lambda i: areas[i])

    mask = masks[best].astype(np.uint8)
    area = mask.sum()
    print(f"    Seed mask area: {area} px ({area/(H*W)*100:.2f}%)")

    if area < 50:
        return None
    return mask


def _propagate_chunk(video_predictor, img_paths, seed_mask, base_frame, forward=True):
    """Propagate seed mask through a chunk of frames.

    Returns:
        dict mapping actual_frame_idx → (H, W) uint8 mask
        last_mask: the mask at the last frame (to seed the next chunk)
    """
    if len(img_paths) == 0:
        return {}, seed_mask

    inference_state = video_predictor.init_state(
        img_paths=img_paths,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )
    video_predictor.reset_state(inference_state)

    video_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=seed_mask.astype(np.uint8),
    )

    chunk_masks = {}
    last_mask = seed_mask
    for out_frame_idx, out_obj_ids, out_mask_logits in \
            video_predictor.propagate_in_video(inference_state):
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[0]

        if forward:
            actual_frame = base_frame + out_frame_idx
        else:
            actual_frame = base_frame - out_frame_idx
        chunk_masks[actual_frame] = mask
        last_mask = mask

    del inference_state
    torch.cuda.empty_cache()
    gc.collect()

    return chunk_masks, last_mask


def propagate_masks_for_camera(cam_jpeg_dir, seed_mask, seed_frame_idx,
                               num_frames, video_predictor, chunk_size=100):
    """Propagate seed mask through video in chunks to avoid OOM.

    Processes chunk_size frames at a time, using the last frame's mask
    as the seed for the next chunk.
    """
    all_masks = {}

    # Forward propagation: seed_frame → end, in chunks
    current_seed = seed_mask
    pos = seed_frame_idx
    while pos < num_frames:
        end = min(pos + chunk_size, num_frames)
        img_paths = [
            str(cam_jpeg_dir / f"color_{i:06d}.jpg")
            for i in range(pos, end)
        ]
        print(f"      Forward chunk: frames {pos}-{end-1} ({len(img_paths)} frames)")

        chunk_masks, current_seed = _propagate_chunk(
            video_predictor, img_paths, current_seed, pos, forward=True)
        all_masks.update(chunk_masks)
        pos = end

    # Backward propagation: seed_frame → 0, in chunks
    if seed_frame_idx > 0:
        current_seed = seed_mask
        pos = seed_frame_idx
        while pos > 0:
            start = max(pos - chunk_size, 0)
            # Reversed: from pos down to start
            img_paths = [
                str(cam_jpeg_dir / f"color_{i:06d}.jpg")
                for i in range(pos, start - 1, -1)
            ]
            print(f"      Backward chunk: frames {pos}-{start} ({len(img_paths)} frames)")

            chunk_masks, current_seed = _propagate_chunk(
                video_predictor, img_paths, current_seed, pos, forward=False)
            all_masks.update(chunk_masks)
            pos = start

    return all_masks


def save_masks_h5(all_cam_masks, num_frames, num_cams, H, W, output_path):
    """Save all masks to masks.h5 with shape (N, C, H, W)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        ds = f.create_dataset("masks", shape=(num_frames, num_cams, H, W),
                              dtype=np.uint8, chunks=(1, 1, H, W),
                              compression="gzip")
        for cam_idx in range(num_cams):
            cam_masks = all_cam_masks.get(cam_idx, {})
            for frame_idx in range(num_frames):
                if frame_idx in cam_masks:
                    ds[frame_idx, cam_idx] = cam_masks[frame_idx]

    print(f"[INFO] Saved masks.h5 to {output_path}")


def _render_frame(rgb_all, mask_all, frame_id, H, W):
    """Render an 8-camera grid for one frame."""
    canvas = np.zeros((H * 2, W * 4, 3), dtype=np.uint8)
    for cam_idx in range(8):
        rgb = rgb_all[cam_idx].copy()
        mask = mask_all[cam_idx]

        overlay = rgb.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)

        row = cam_idx // 4
        col = cam_idx % 4
        canvas[row*H:(row+1)*H, col*W:(col+1)*W] = overlay

        cov = mask.sum() / (H*W) * 100
        cv2.putText(canvas, f'cam{cam_idx} {cov:.2f}%',
                    (col*W+5, row*H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.putText(canvas, f'Frame {frame_id} - SAM2 Video Masks',
                (10, H*2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return canvas


def visualize_masks(hocap_dir, masks_h5_path, output_dir, frames=None, make_video=True, fps=15):
    """Generate 8-camera grid visualizations of masks overlaid on RGB.

    Saves PNG snapshots at key frames and an MP4 video of all frames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5f = h5py.File(Path(hocap_dir) / "data00000000.h5", "r")
    mf = h5py.File(masks_h5_path, "r")
    H, W = 480, 640
    num_frames = h5f["imgs"].shape[0]

    if frames is None:
        frames = [0, 50, 100, 200, 350, 500, min(700, num_frames - 1)]

    # PNG snapshots
    for frame_id in frames:
        if frame_id >= num_frames:
            continue
        rgb_all = h5f["imgs"][frame_id]
        mask_all = mf["masks"][frame_id]
        canvas = _render_frame(rgb_all, mask_all, frame_id, H, W)
        out_path = output_dir / f'masks_f{frame_id}.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f'Saved {out_path}')

    # MP4 video of all frames
    if make_video:
        video_path = output_dir / "masks_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W * 4, H * 2))
        print(f"Generating mp4 ({num_frames} frames)...")
        for frame_id in range(num_frames):
            rgb_all = h5f["imgs"][frame_id]
            mask_all = mf["masks"][frame_id]
            canvas = _render_frame(rgb_all, mask_all, frame_id, H, W)
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            if frame_id % 100 == 0:
                print(f"  frame {frame_id}/{num_frames}")
        writer.release()
        print(f'Saved {video_path}')

    h5f.close()
    mf.close()


def main():
    parser = argparse.ArgumentParser(description="SAM2 video mask generation with auto seeds")
    parser.add_argument("--hocap_dir", required=True)
    parser.add_argument("--calib_yaml", required=True)
    parser.add_argument("--output_dir", required=True)
    _repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--sam2_ckpt",
        default=str(_repo_root / "config/checkpoints/sam2/sam2.1_hiera_large.pt"),
        help="Path to SAM2 checkpoint (.pt).")
    parser.add_argument(
        "--sam2_image_cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 image predictor config (resolved inside the sam2 package).")
    parser.add_argument(
        "--sam2_video_cfg",
        default=str(_repo_root / "config/sam2_config/sam2.1_hiera_l.yaml"),
        help="SAM2 video predictor config yaml (absolute or repo-relative).")
    parser.add_argument("--xy_center", type=float, nargs=2, default=[-0.07, -0.01])
    parser.add_argument("--xy_radius", type=float, default=0.12)
    parser.add_argument("--z_min", type=float, default=0.10)
    parser.add_argument("--z_max", type=float, default=0.30)
    parser.add_argument("--cameras", type=int, nargs="+", default=None,
                        help="Only process these camera indices (e.g., --cameras 0 5 7)")
    parser.add_argument("--good_cameras", type=int, nargs="+", default=None,
                        help="Cameras with known-correct spatial seeds (e.g., --good_cameras 0 5 7). "
                             "Other cameras will use DINOv2 feature matching.")
    parser.add_argument("--test_frame_only", action="store_true",
                        help="Only generate seed masks and visualize (no propagation)")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip JPEG extraction if already done")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing masks.h5, skip already-completed cameras")
    args = parser.parse_args()

    hocap_dir = Path(args.hocap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp_jpegs"

    # Load calibration
    with open(args.calib_yaml) as f:
        calib = yaml.safe_load(f)
    cam_Ks = np.array([np.array(c["color_intrinsic_matrix"], dtype=np.float32) for c in calib])
    cam_Ts = np.array([np.array(c["transformation"], dtype=np.float32) for c in calib])

    sx, sy = 640 / 1280, 480 / 720
    scaled_Ks = cam_Ks.copy()
    for i in range(len(scaled_Ks)):
        scaled_Ks[i][0, :] *= sx
        scaled_Ks[i][1, :] *= sy

    H, W = 480, 640

    # Open data H5
    data_h5 = h5py.File(hocap_dir / "data00000000.h5", "r")
    num_frames = data_h5["imgs"].shape[0]
    num_cams = data_h5["imgs"].shape[1]
    active_cams = args.cameras if args.cameras else list(range(num_cams))
    print(f"[INFO] {num_frames} frames, {num_cams} cameras, {W}x{H}")
    if args.cameras:
        print(f"[INFO] Processing only cameras: {active_cams}")

    # ── Step 1: Extract H5 frames to JPEGs ──
    # Skip extraction in test mode (seed generation reads directly from H5)
    if args.test_frame_only:
        print("\n[Step 1] Skipping JPEG extraction (--test_frame_only)")
    elif not args.skip_extraction:
        print("\n[Step 1] Extracting H5 frames to JPEGs...")
        extract_h5_to_jpegs(hocap_dir, tmp_dir, cam_indices=active_cams)
    else:
        print("\n[Step 1] Skipping JPEG extraction (--skip_extraction)")

    # ── Step 2: Find best seed frames and generate seed masks ──
    print("\n[Step 2] Generating seed masks...")

    # Load SAM2 image predictor for seed generation
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image_model = build_sam2(args.sam2_image_cfg, args.sam2_ckpt, device='cuda:0')
    image_predictor = SAM2ImagePredictor(image_model)
    print("[INFO] SAM2 image predictor loaded")

    # Phase 1: Generate seeds for cameras where spatial prior works (known good: 0, 5, 7)
    spatial_cams = []
    seed_masks = {}  # cam_idx → (frame_idx, mask)

    for cam_idx in active_cams:
        print(f"\n  Camera {cam_idx} (spatial prior):")
        K = scaled_Ks[cam_idx]
        cam_T = cam_Ts[cam_idx]

        best_frame = find_best_seed_frame(
            data_h5, cam_idx, K, cam_T, H, W,
            tuple(args.xy_center), args.xy_radius, args.z_min, args.z_max)

        rgb = data_h5["imgs"][best_frame, cam_idx]
        depth = data_h5["depths"][best_frame, cam_idx]

        mask = generate_seed_mask(
            rgb, depth, K, cam_T, H, W,
            image_predictor, tuple(args.xy_center), args.xy_radius,
            args.z_min, args.z_max)

        if mask is not None:
            seed_masks[cam_idx] = (best_frame, mask)
            spatial_cams.append(cam_idx)
            print(f"    ✓ Seed mask generated at frame {best_frame}")
        else:
            print(f"    ✗ Failed with spatial prior")

    # Phase 2: Use DINOv2 feature matching for cameras that need it
    # Only use spatial seeds that the user confirmed as correct
    good_cams = [c for c in args.good_cameras if c in seed_masks] if args.good_cameras else spatial_cams
    need_dino = [c for c in active_cams if c not in good_cams]

    if need_dino and good_cams:
        print(f"\n[Step 2b] DINOv2 feature matching for cameras {need_dino}...")
        print(f"  Using reference from cameras {good_cams}")

        # Load DINOv2
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
        dino_model = dino_model.cuda().eval()
        print("  DINOv2 loaded")

        # Compute reference pestle features from good cameras
        ref_feats = []
        for ref_cam in good_cams:
            ref_frame, ref_mask = seed_masks[ref_cam]
            ref_rgb = data_h5["imgs"][ref_frame, ref_cam]
            feat = get_reference_pestle_features(dino_model, ref_rgb, ref_mask)
            if feat is not None:
                ref_feats.append(feat)
                print(f"    ref cam{ref_cam} frame {ref_frame}: feature extracted")

        if ref_feats:
            # Average reference features
            avg_ref_feat = np.mean(ref_feats, axis=0)
            avg_ref_feat = avg_ref_feat / (np.linalg.norm(avg_ref_feat) + 1e-8)

            for cam_idx in need_dino:
                print(f"\n  Camera {cam_idx} (DINOv2 matching):")

                # Try multiple frames to find the best match
                best_sim = -1
                best_match_frame = 0
                best_match_pt = None

                # Sample frames
                candidate_frames = list(range(0, num_frames, 50))
                for frame_idx in candidate_frames:
                    rgb = data_h5["imgs"][frame_idx, cam_idx]
                    pt, sim = find_pestle_by_features(dino_model, rgb, avg_ref_feat, H=H, W=W)
                    if sim > best_sim:
                        best_sim = sim
                        best_match_frame = frame_idx
                        best_match_pt = pt

                print(f"    Best match: frame {best_match_frame}, sim={best_sim:.3f}, pt={best_match_pt}")

                if best_match_pt is not None and best_sim > 0.3:
                    # Get negative prompt from spatial prior (mortar center)
                    depth = data_h5["depths"][best_match_frame, cam_idx]
                    _, neg_pt = get_tool_prompt(
                        depth, scaled_Ks[cam_idx], cam_Ts[cam_idx], H, W,
                        tuple(args.xy_center), args.xy_radius, args.z_min, args.z_max)

                    rgb = data_h5["imgs"][best_match_frame, cam_idx]
                    mask = generate_seed_mask_from_features(
                        rgb, image_predictor, best_match_pt, neg_pt)

                    if mask is not None:
                        seed_masks[cam_idx] = (best_match_frame, mask)
                        print(f"    ✓ Seed mask generated via DINOv2 at frame {best_match_frame}")
                    else:
                        print(f"    ✗ SAM2 failed on DINOv2 prompt")
                else:
                    print(f"    ✗ No good match found (sim={best_sim:.3f})")

        del dino_model
        torch.cuda.empty_cache()
        gc.collect()

    # Clean up image predictor
    del image_predictor, image_model
    torch.cuda.empty_cache()
    gc.collect()

    # Visualize seed masks
    print("\n[INFO] Visualizing seed masks...")
    seed_viz_dir = output_dir / "viz_seeds"
    seed_viz_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary H5 with just the seed masks for visualization
    seed_canvas = np.zeros((H * 2, W * 4, 3), dtype=np.uint8)
    for cam_idx in range(num_cams):
        if cam_idx in seed_masks:
            frame_idx, mask = seed_masks[cam_idx]
            rgb = data_h5["imgs"][frame_idx, cam_idx].copy()
            overlay = rgb.copy()
            overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)
        else:
            rgb = data_h5["imgs"][0, cam_idx].copy()
            overlay = rgb.copy()
            cv2.putText(overlay, "NO SEED", (W//4, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_idx = -1
            mask = np.zeros((H, W), dtype=np.uint8)

        row = cam_idx // 4
        col = cam_idx % 4
        seed_canvas[row*H:(row+1)*H, col*W:(col+1)*W] = overlay

        cov = mask.sum() / (H*W) * 100
        label = f'cam{cam_idx} f{frame_idx} {cov:.2f}%'
        cv2.putText(seed_canvas, label,
                    (col*W+5, row*H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.putText(seed_canvas, 'Seed Masks (auto-generated)',
                (10, H*2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    seed_viz_path = seed_viz_dir / 'seed_masks.png'
    cv2.imwrite(str(seed_viz_path), cv2.cvtColor(seed_canvas, cv2.COLOR_RGB2BGR))
    print(f'Saved {seed_viz_path}')

    if args.test_frame_only:
        print("\n[INFO] --test_frame_only: stopping after seed mask generation")
        data_h5.close()
        return

    # ── Step 3: SAM2 Video Propagation ──
    print("\n[Step 3] SAM2 video propagation...")

    # Enable performance settings
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from hocap_annotation.wrappers.sam2 import build_sam2_video_predictor
    video_predictor = build_sam2_video_predictor(
        config_file=args.sam2_video_cfg,
        ckpt_path=args.sam2_ckpt,
        device="cuda:0",
    )
    print("[INFO] SAM2 video predictor loaded")

    # Open masks.h5 incrementally to avoid holding all masks in memory
    # Use append mode if file exists (resume)
    masks_h5_path = output_dir / "masks.h5"
    masks_h5_path.parent.mkdir(parents=True, exist_ok=True)
    if masks_h5_path.exists() and args.resume:
        print(f"[INFO] Resuming from existing {masks_h5_path}")
        masks_h5 = h5py.File(masks_h5_path, "a")
        masks_ds = masks_h5["masks"]
    else:
        masks_h5 = h5py.File(masks_h5_path, "w")
        masks_ds = masks_h5.create_dataset(
            "masks", shape=(num_frames, num_cams, H, W),
            dtype=np.uint8, chunks=(1, 1, H, W), compression="gzip")

    for cam_idx in active_cams:
        if cam_idx not in seed_masks:
            print(f"\n  Camera {cam_idx}: skipped (no seed mask)")
            continue

        # Resume: skip if already done (>= 700 non-empty frames)
        if args.resume:
            existing = masks_ds[:, cam_idx]
            non_empty = (existing.sum(axis=(1, 2)) > 0).sum()
            if non_empty >= num_frames * 0.9:
                print(f"\n  Camera {cam_idx}: already complete ({non_empty}/{num_frames} frames), skipping")
                continue

        seed_frame, seed_mask = seed_masks[cam_idx]
        cam_jpeg_dir = tmp_dir / f"cam{cam_idx:02d}"

        print(f"\n  Camera {cam_idx}: propagating from seed frame {seed_frame}...")

        cam_masks = propagate_masks_for_camera(
            cam_jpeg_dir, seed_mask, seed_frame,
            num_frames, video_predictor)

        # Write to H5 immediately and free memory
        for frame_idx, mask in cam_masks.items():
            masks_ds[frame_idx, cam_idx] = mask
        masks_h5.flush()

        valid_count = sum(1 for m in cam_masks.values() if m.sum() > 0)
        print(f"    Got {len(cam_masks)} frames, {valid_count} with non-empty masks")

        del cam_masks
        gc.collect()
        torch.cuda.empty_cache()

    masks_h5.close()
    del video_predictor
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n[Step 4] masks.h5 saved to {masks_h5_path}")

    # ── Step 5: Visualize ──
    print("\n[Step 5] Generating visualizations...")
    viz_dir = output_dir / "viz_masks"
    visualize_masks(hocap_dir, masks_h5_path, viz_dir)

    # Clean up temp JPEGs
    print(f"\n[INFO] Cleaning up temporary JPEGs at {tmp_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    data_h5.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
