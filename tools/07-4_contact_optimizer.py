#!/usr/bin/env python3
"""
Multi-stage contact optimizer for hand-object pose.

Optimizes hand wrist translation + rotation to:
  1. Minimize overall hand-object distance for grasping frames
  2. Minimize each fingertip-to-object distance
  3. Prevent hand-object interpenetration

Multi-stage strategy:
  Stage 1: Analytical global offset — mean fingertip-to-surface vector
  Stage 2: Global refinement — optimize single (delta_rot, delta_trans) for all grasping frames
  Stage 3: Per-frame refinement — per-frame (delta_rot, delta_trans) with smoothness
  Stage 4: Post-processing — temporal smoothing + penetration fix

Outputs:
  - result_hand_optimized.pkl: optimized hand data
  - fingertip_distances.npy: (num_frames, num_hands, 5) distances
  - fingertip_meta.json: metadata

Usage:
    python tools/07-4_contact_optimizer.py \\
        --sequence_folder /path/to/sequence
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torch.nn import Parameter
from torch.optim import Adam
from manopth.manolayer import ManoLayer
from scipy.spatial.transform import Rotation

from hocap_annotation.utils import *
from hocap_annotation.loaders import MySequenceLoader as SequenceLoader

FINGERTIP_INDICES = [20, 4, 8, 16, 12]  # thumb, index, middle, ring, pinky
FINGERTIP_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


# ============================================================
# SDF Grid
# ============================================================

def compute_object_sdf_grid(mesh, resolution=128, padding=0.15):
    """Precompute SDF grid with large padding to cover global offset."""
    import igl
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    half_size = (bounds[1] - bounds[0]).max() / 2 + padding
    grid_min = center - half_size
    grid_max = center + half_size

    lin = np.linspace(0, 1, resolution)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float64)
    pts = pts * (grid_max - grid_min) + grid_min

    V = np.array(mesh.vertices, dtype=np.float64)
    Fa = np.array(mesh.faces, dtype=np.int64)
    signed, _, _, _ = igl.signed_distance(pts, V, Fa)

    sdf = signed.astype(np.float32).reshape(resolution, resolution, resolution)
    return sdf, grid_min.astype(np.float32), grid_max.astype(np.float32)


def query_sdf(points, sdf_grid, grid_min, grid_max):
    """Differentiable SDF query via grid_sample. (N,3) -> (N,)."""
    normalized = (points - grid_min) / (grid_max - grid_min) * 2 - 1
    grid_coords = normalized.view(1, -1, 1, 1, 3)
    vals = F.grid_sample(
        sdf_grid, grid_coords,
        align_corners=True, padding_mode='border', mode='bilinear',
    )
    return vals.view(-1)


# ============================================================
# Differentiable rotation
# ============================================================

def rodrigues(rvec):
    """(3,) -> (3,3) rotation matrix. Differentiable."""
    angle = torch.norm(rvec).clamp(min=1e-8)
    axis = rvec / angle
    s, c = torch.sin(angle), torch.cos(angle)
    K = torch.stack([
        torch.stack([rvec.new_zeros(1).squeeze(), -axis[2], axis[1]]),
        torch.stack([axis[2], rvec.new_zeros(1).squeeze(), -axis[0]]),
        torch.stack([-axis[1], axis[0], rvec.new_zeros(1).squeeze()]),
    ])
    return torch.eye(3, device=rvec.device) + s * K + (1 - c) * K @ K


def apply_wrist_delta(verts_local, delta_rot, delta_trans):
    """Apply rotation + translation delta to precomputed local verts.

    verts_local: (N, 3) — hand verts relative to wrist root
    delta_rot: (3,) — rotation vector
    delta_trans: (3,) — translation

    Returns: (N, 3) world verts
    """
    R = rodrigues(delta_rot)
    return verts_local @ R.T + delta_trans


# ============================================================
# Loss functions (all differentiable)
# ============================================================

def transform_to_obj_frame(points_world, obj_rvec, obj_trans):
    """World -> object canonical frame. Differentiable."""
    R_obj = rodrigues(obj_rvec)
    return (points_world - obj_trans.unsqueeze(0)) @ R_obj


def loss_penetration(hand_verts_obj, sdf_grid, grid_min, grid_max):
    """Penetration loss: penalize vertices with SDF < 0."""
    sdf_vals = query_sdf(hand_verts_obj, sdf_grid, grid_min, grid_max)
    pen = torch.clamp(-sdf_vals, min=0)
    n_pen = (pen > 0).sum().item()
    return pen.sum(), n_pen


def loss_fingertip_attract(tips_obj, sdf_grid, grid_min, grid_max):
    """Attract fingertips to object surface (minimize |SDF|).
    Returns loss and per-tip distances."""
    sdf_vals = query_sdf(tips_obj, sdf_grid, grid_min, grid_max)
    # Attract from outside (positive SDF)
    outside_loss = torch.clamp(sdf_vals, min=0).sum()
    # Also penalize being inside (will be handled by pen loss, but small attract helps)
    return outside_loss, sdf_vals.detach().abs()


def loss_hand_proximity(hand_verts_obj, sdf_grid, grid_min, grid_max, thresh=0.05):
    """Attract hand vertices near the object toward the surface.
    Only affects vertices within `thresh` meters of surface (SDF in [0, thresh])."""
    sdf_vals = query_sdf(hand_verts_obj, sdf_grid, grid_min, grid_max)
    near_mask = (sdf_vals > 0) & (sdf_vals < thresh)
    if near_mask.sum() > 0:
        return sdf_vals[near_mask].mean()
    return torch.zeros(1, device=hand_verts_obj.device).squeeze()


def loss_smoothness(params_seq, w_vel=1.0, w_acc=1.0):
    """Temporal smoothness on (F, D) params."""
    if params_seq.shape[0] < 3:
        return torch.zeros(1, device=params_seq.device).squeeze()
    vel = params_seq[1:] - params_seq[:-1]
    acc = vel[1:] - vel[:-1]
    return torch.norm(vel, dim=-1).mean() * w_vel + torch.norm(acc, dim=-1).mean() * w_acc


# ============================================================
# MANO precomputation
# ============================================================

def precompute_hand_local(mano_layer, pose_all, beta, base_rot_all, side, device):
    """Precompute hand verts/joints in wrist-root-relative frame.
    Returns verts_local (F,778,3), tips_local (F,5,3), original_trans (F,3)."""
    num_frames = pose_all.shape[0]
    beta_t = torch.tensor(beta, device=device, dtype=torch.float32)
    if beta_t.ndim == 1:
        beta_t = beta_t.unsqueeze(0)

    all_verts, all_tips = [], []
    with torch.no_grad():
        for f in range(num_frames):
            pose_t = torch.tensor(pose_all[f:f+1], device=device, dtype=torch.float32)
            verts, joints = mano_layer(pose_t, beta_t)
            v = verts[0].cpu().numpy() / 1000.0
            j = joints[0].cpu().numpy() / 1000.0
            root = j[0]
            v = v - root
            j = j - root
            if side == 'left':
                v[:, 0] *= -1
                j[:, 0] *= -1
                if base_rot_all is not None:
                    rot_idx = min(f, len(base_rot_all) - 1)
                    br = base_rot_all[rot_idx]
                    v = v @ br.T
                    j = j @ br.T
            all_verts.append(v)
            all_tips.append(j[FINGERTIP_INDICES])

    return (
        np.array(all_verts, dtype=np.float32),
        np.array(all_tips, dtype=np.float32),
    )


# ============================================================
# Data loading
# ============================================================

def load_hand_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    hp = data['hand_pose']
    return {k: np.array(hp.get(k, [])) for k in [
        'left_hand_pose', 'left_hand_beta', 'left_hand_translation', 'left_hand_base_rot',
        'right_hand_pose', 'right_hand_beta', 'right_hand_translation', 'right_hand_base_rot',
    ]}


def detect_hands(hand_data):
    hands = []
    if len(hand_data['left_hand_pose']) > 0:
        hands.append('left')
    if len(hand_data['right_hand_pose']) > 0:
        hands.append('right')
    return hands


# ============================================================
# Stage 1: Analytical global offset
# ============================================================

def compute_global_offset(tips_local, trans_all, obj_mesh, obj_Rs, obj_ts, grasp_mask):
    """Compute mean fingertip-to-closest-surface vector across grasping frames.

    Returns delta_trans (3,) and per-frame tip distances for diagnostics.
    """
    offsets = []
    for f in np.where(grasp_mask)[0]:
        tips_world = tips_local[f] + trans_all[f]
        tips_obj = (tips_world - obj_ts[f]) @ obj_Rs[f]  # to canonical
        closest, dists, _ = trimesh.proximity.closest_point(obj_mesh, tips_obj)
        # Vector from fingertip to closest point (in canonical frame)
        vec_obj = closest - tips_obj  # (5, 3)
        # Transform back to world
        vec_world = vec_obj @ obj_Rs[f].T
        offsets.append(vec_world)

    if len(offsets) == 0:
        return np.zeros(3, dtype=np.float32)

    offsets = np.concatenate(offsets, axis=0)  # (N*5, 3)
    # Use median for robustness against outliers
    return np.median(offsets, axis=0).astype(np.float32)


# ============================================================
# Stage 2: Global optimization (gradient-based)
# ============================================================

def optimize_global(
    verts_local_t, tips_local_t, trans_all,
    obj_rvecs_t, obj_trans_t,
    sdf_grid, grid_min, grid_max,
    grasp_frames, device,
    init_delta_trans, init_delta_rot=None,
    steps=100, lr=0.002,
    w_pen=1000.0, w_tip=200.0, w_prox=50.0,
):
    """Optimize single (delta_rot, delta_trans) for all grasping frames."""
    delta_trans = Parameter(torch.tensor(init_delta_trans, device=device, dtype=torch.float32))
    delta_rot = Parameter(torch.zeros(3, device=device, dtype=torch.float32) if init_delta_rot is None
                          else torch.tensor(init_delta_rot, device=device, dtype=torch.float32))

    optimizer = Adam([delta_trans, delta_rot], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        total_pen = torch.zeros(1, device=device).squeeze()
        total_tip = torch.zeros(1, device=device).squeeze()
        total_prox = torch.zeros(1, device=device).squeeze()
        total_n_pen = 0

        R_delta = rodrigues(delta_rot)

        for f in grasp_frames:
            # Apply global delta to this frame
            verts_world = verts_local_t[f] @ R_delta.T + (trans_all[f] + delta_trans)
            tips_world = tips_local_t[f] @ R_delta.T + (trans_all[f] + delta_trans)

            # Transform to object frame
            verts_obj = transform_to_obj_frame(verts_world, obj_rvecs_t[f], obj_trans_t[f])
            tips_obj = transform_to_obj_frame(tips_world, obj_rvecs_t[f], obj_trans_t[f])

            pen, n_pen = loss_penetration(verts_obj, sdf_grid, grid_min, grid_max)
            total_pen = total_pen + pen
            total_n_pen += n_pen

            tip_loss, _ = loss_fingertip_attract(tips_obj, sdf_grid, grid_min, grid_max)
            total_tip = total_tip + tip_loss

            prox = loss_hand_proximity(verts_obj, sdf_grid, grid_min, grid_max, thresh=0.05)
            total_prox = total_prox + prox

        n = max(len(grasp_frames), 1)
        loss = (total_pen / n) * w_pen + (total_tip / n) * w_tip + (total_prox / n) * w_prox

        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0 or step == 0:
            print(f"    global step {step+1:3d}/{steps} | loss: {loss.item():.4f} | "
                  f"pen: {total_pen.item()/n:.4f} ({total_n_pen}) | "
                  f"tip: {total_tip.item()/n:.4f} | prox: {total_prox.item()/n:.4f} | "
                  f"dt: [{delta_trans.data.cpu().numpy()*1000}] mm | "
                  f"dr: [{delta_rot.data.cpu().numpy()*180/np.pi}] deg")

    return delta_trans.detach().cpu().numpy(), delta_rot.detach().cpu().numpy()


# ============================================================
# Stage 3: Per-frame refinement (gradient-based)
# ============================================================

def optimize_per_frame(
    verts_local_t, tips_local_t, trans_all_np,
    obj_rvecs_t, obj_trans_t,
    sdf_grid, grid_min, grid_max,
    grasp_mask, device, num_frames,
    init_delta_trans_global, init_delta_rot_global,
    steps=200, lr=0.001,
    w_pen=1000.0, w_tip=200.0, w_prox=50.0,
    w_reg=5.0, w_smooth=1.0, prox_thresh=0.05,
):
    """Per-frame (delta_rot, delta_trans) optimization with LR scheduling."""
    # Initialize: grasping frames from global, non-grasping from zero
    init_dt = np.zeros((num_frames, 3), dtype=np.float32)
    init_dr = np.zeros((num_frames, 3), dtype=np.float32)
    for f in range(num_frames):
        if grasp_mask[f]:
            init_dt[f] = init_delta_trans_global
            init_dr[f] = init_delta_rot_global

    delta_trans = Parameter(torch.tensor(init_dt, device=device))
    delta_rot = Parameter(torch.tensor(init_dr, device=device))
    target_dt = delta_trans.data.clone()
    target_dr = delta_rot.data.clone()

    optimizer = Adam([delta_trans, delta_rot], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    grasp_indices = np.where(grasp_mask)[0]
    non_grasp_indices = np.where(~grasp_mask)[0]

    # Pre-cache translations on GPU
    trans_all_t = torch.tensor(trans_all_np, device=device, dtype=torch.float32)

    for step in range(steps):
        optimizer.zero_grad()
        total_pen = torch.zeros(1, device=device).squeeze()
        total_tip = torch.zeros(1, device=device).squeeze()
        total_prox = torch.zeros(1, device=device).squeeze()
        total_n_pen = 0

        # Process grasping frames
        for f in grasp_indices:
            R_f = rodrigues(delta_rot[f])
            base_t = trans_all_t[f] + delta_trans[f]
            verts_world = verts_local_t[f] @ R_f.T + base_t
            tips_world = tips_local_t[f] @ R_f.T + base_t

            verts_obj = transform_to_obj_frame(verts_world, obj_rvecs_t[f], obj_trans_t[f])
            tips_obj = transform_to_obj_frame(tips_world, obj_rvecs_t[f], obj_trans_t[f])

            pen, n_pen = loss_penetration(verts_obj, sdf_grid, grid_min, grid_max)
            total_pen = total_pen + pen
            total_n_pen += n_pen

            tip_loss, _ = loss_fingertip_attract(tips_obj, sdf_grid, grid_min, grid_max)
            total_tip = total_tip + tip_loss

            prox = loss_hand_proximity(verts_obj, sdf_grid, grid_min, grid_max, thresh=prox_thresh)
            total_prox = total_prox + prox

        # Process non-grasping frames (penetration only)
        for f in non_grasp_indices:
            R_f = rodrigues(delta_rot[f])
            verts_world = verts_local_t[f] @ R_f.T + trans_all_t[f] + delta_trans[f]
            verts_obj = transform_to_obj_frame(verts_world, obj_rvecs_t[f], obj_trans_t[f])
            pen, n_pen = loss_penetration(verts_obj, sdf_grid, grid_min, grid_max)
            total_pen = total_pen + pen
            total_n_pen += n_pen

        n_grasp = max(len(grasp_indices), 1)
        l_pen = (total_pen / num_frames) * w_pen
        l_tip = (total_tip / n_grasp) * w_tip
        l_prox = (total_prox / n_grasp) * w_prox

        # Regularization: stay close to initialization
        l_reg = (F.mse_loss(delta_trans, target_dt) + F.mse_loss(delta_rot, target_dr)) * w_reg

        # Temporal smoothness (acceleration term weighted higher to reduce jitter)
        l_smooth = (loss_smoothness(delta_trans, w_vel=1.0, w_acc=3.0)
                     + loss_smoothness(delta_rot, w_vel=1.0, w_acc=3.0)) * w_smooth

        loss = l_pen + l_tip + l_prox + l_reg + l_smooth
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0 or step == 0:
            cur_lr = scheduler.get_last_lr()[0]
            print(f"    per-frame step {step+1:3d}/{steps} | loss: {loss.item():.4f} | "
                  f"pen: {l_pen.item():.4f} ({total_n_pen}) | "
                  f"tip: {l_tip.item():.4f} | prox: {l_prox.item():.4f} | "
                  f"reg: {l_reg.item():.4f} | smooth: {l_smooth.item():.4f} | "
                  f"lr: {cur_lr:.6f}")

    return delta_trans.detach().cpu().numpy(), delta_rot.detach().cpu().numpy()


# ============================================================
# Stage 4: Post-processing
# ============================================================

def temporal_smooth(arr, window=3):
    """Smooth (F, D) array temporally."""
    from scipy.ndimage import uniform_filter1d
    out = arr.copy()
    for d in range(arr.shape[1]):
        out[:, d] = uniform_filter1d(arr[:, d], size=window, mode='nearest')
    return out


def fix_penetration_pass(
    verts_local_all, delta_trans, delta_rot, trans_all_np,
    obj_rvecs_t, obj_trans_t, sdf_grid, grid_min, grid_max,
    device, num_frames, steps=30, lr=0.002,
):
    """Fix penetration introduced by smoothing with neighbor-aware smoothness.

    Optimizes all penetrating frames jointly to maintain temporal coherence.
    """
    # First identify which frames need fixing
    pen_frames = []
    for f in range(num_frames):
        vl = torch.tensor(verts_local_all[f], device=device)
        dr = torch.tensor(delta_rot[f], device=device)
        orig_t = torch.tensor(trans_all_np[f], device=device, dtype=torch.float32)
        dt = torch.tensor(delta_trans[f], device=device)
        R_f = rodrigues(dr)
        verts_world = vl @ R_f.T + orig_t + dt
        verts_obj = transform_to_obj_frame(verts_world, obj_rvecs_t[f], obj_trans_t[f])
        sdf_vals = query_sdf(verts_obj, sdf_grid, grid_min, grid_max)
        if (sdf_vals < 0).sum() > 0:
            pen_frames.append(f)

    if len(pen_frames) == 0:
        return delta_trans, 0

    # Optimize all frames' translations jointly to preserve smoothness
    dt_all = Parameter(torch.tensor(delta_trans, device=device, dtype=torch.float32))
    target_dt = dt_all.data.clone()

    opt = Adam([dt_all], lr=lr)
    pen_set = set(pen_frames)

    for step_i in range(steps):
        opt.zero_grad()
        total_loss = torch.zeros(1, device=device).squeeze()
        n_pen_total = 0

        for f in pen_frames:
            vl = torch.tensor(verts_local_all[f], device=device)
            dr = torch.tensor(delta_rot[f], device=device)
            orig_t = torch.tensor(trans_all_np[f], device=device, dtype=torch.float32)
            R_f = rodrigues(dr)
            verts_world = vl @ R_f.T + orig_t + dt_all[f]
            verts_obj = transform_to_obj_frame(verts_world, obj_rvecs_t[f], obj_trans_t[f])
            pen_loss, n = loss_penetration(verts_obj, sdf_grid, grid_min, grid_max)
            total_loss = total_loss + pen_loss * 1000
            n_pen_total += n

        if n_pen_total == 0:
            break

        # Regularization: stay close to target
        reg = F.mse_loss(dt_all, target_dt) * 5.0
        # Smoothness: penalize acceleration to avoid introducing jitter
        smooth = loss_smoothness(dt_all, w_vel=1.0, w_acc=3.0) * 10.0
        (total_loss + reg + smooth).backward()
        opt.step()

    delta_trans = dt_all.detach().cpu().numpy()
    return delta_trans, len(pen_frames)


# ============================================================
# Compute fingertip distances (exact, for reporting)
# ============================================================

def compute_tip_distances_exact(tips_world, obj_mesh, obj_R, obj_t):
    """Exact fingertip-to-surface distances using trimesh."""
    tips_obj = (tips_world - obj_t) @ obj_R
    _, dists, _ = trimesh.proximity.closest_point(obj_mesh, tips_obj)
    return dists.astype(np.float32)


# ============================================================
# Main
# ============================================================

def run_contact_optimizer(
    sequence_folder,
    object_idx=1,
    sdf_resolution=128,
    sdf_padding=0.15,
    grasp_thresh=0.05,
    min_grasp_fingers=2,
    global_steps=100,
    perframe_steps=200,
    global_lr=0.002,
    perframe_lr=0.001,
    w_pen=1000.0,
    w_tip=200.0,
    w_prox=50.0,
    w_reg=5.0,
    w_smooth=10.0,
    smooth_window=5,
    output_suffix="contact_optimizer",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequence_folder = Path(sequence_folder)
    object_idx_0 = object_idx - 1

    # Paths
    data_folder = sequence_folder
    folder_name = data_folder.parent.parent.name
    task_name = data_folder.parent.name
    sequence_name = data_folder.name
    annotated_folder = (
        data_folder.parent.parent.parent
        / f"{folder_name}_annotated" / task_name / sequence_name
    )
    save_folder = annotated_folder / "processed" / output_suffix
    save_folder.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Sequence: {sequence_folder}")
    print(f"[INFO] Output:   {save_folder}")

    # Load data
    data_loader = SequenceLoader(sequence_folder, load_mano=True, load_object=True, device=device)
    num_frames = data_loader.num_frames
    object_ids = data_loader.object_ids
    obj_name = object_ids[object_idx_0]
    print(f"[INFO] Object: {obj_name}, Frames: {num_frames}")

    mesh_path = data_loader.object_cleaned_mesh_files[object_idx_0]
    obj_mesh = trimesh.load(str(mesh_path), process=True, force='mesh')
    print(f"[INFO] Object mesh: {len(obj_mesh.vertices)} verts, {len(obj_mesh.faces)} faces")

    pkl_path = annotated_folder / "result_hand_optimized.pkl"
    if not pkl_path.exists():
        pkl_path = annotated_folder / "result.pkl"
    hand_data = load_hand_data(pkl_path)
    hands = detect_hands(hand_data)
    print(f"[INFO] Hands detected: {hands}")

    # Object poses (frozen)
    pose_o_path = annotated_folder / "processed" / "object_pose_solver" / "poses_o.npy"
    if not pose_o_path.exists():
        pose_o_path = annotated_folder / "processed" / "fd_pose_solver" / "fd_poses_merged_fixed.npy"
    poses_o = np.load(str(pose_o_path)).astype(np.float32)
    if poses_o.ndim == 3:
        poses_o = poses_o[object_idx_0]
    poses_o = poses_o.reshape(num_frames, 7)

    obj_Rs = np.zeros((num_frames, 3, 3), dtype=np.float32)
    obj_ts = np.zeros((num_frames, 3), dtype=np.float32)
    obj_rvecs_np = np.zeros((num_frames, 3), dtype=np.float32)
    for i in range(num_frames):
        q = poses_o[i]
        rot = Rotation.from_quat([q[0], q[1], q[2], q[3]])
        obj_Rs[i] = rot.as_matrix()
        obj_ts[i] = q[4:7]
        obj_rvecs_np[i] = rot.as_rotvec()

    # Coarse SDF grid (large padding for global alignment)
    print(f"[INFO] Computing coarse SDF grid (res={sdf_resolution}, padding={sdf_padding}m)...")
    t0 = time.time()
    sdf_np_coarse, grid_min_coarse, grid_max_coarse = compute_object_sdf_grid(
        obj_mesh, resolution=sdf_resolution, padding=sdf_padding
    )
    print(f"[INFO] Coarse SDF in {time.time()-t0:.1f}s, range: [{sdf_np_coarse.min():.4f}, {sdf_np_coarse.max():.4f}]")
    sdf_grid_coarse = torch.from_numpy(sdf_np_coarse).permute(2, 1, 0).unsqueeze(0).unsqueeze(0).to(device)
    grid_min_coarse_t = torch.from_numpy(grid_min_coarse).to(device)
    grid_max_coarse_t = torch.from_numpy(grid_max_coarse).to(device)

    # Fine SDF grid (small padding, higher effective resolution near object)
    fine_padding = 0.03
    print(f"[INFO] Computing fine SDF grid (res={sdf_resolution}, padding={fine_padding}m)...")
    t0 = time.time()
    sdf_np_fine, grid_min_fine, grid_max_fine = compute_object_sdf_grid(
        obj_mesh, resolution=sdf_resolution, padding=fine_padding
    )
    print(f"[INFO] Fine SDF in {time.time()-t0:.1f}s, range: [{sdf_np_fine.min():.4f}, {sdf_np_fine.max():.4f}]")
    sdf_grid_fine = torch.from_numpy(sdf_np_fine).permute(2, 1, 0).unsqueeze(0).unsqueeze(0).to(device)
    grid_min_fine_t = torch.from_numpy(grid_min_fine).to(device)
    grid_max_fine_t = torch.from_numpy(grid_max_fine).to(device)

    obj_rvecs_t = torch.tensor(obj_rvecs_np, device=device, dtype=torch.float32)
    obj_trans_t = torch.tensor(obj_ts, device=device, dtype=torch.float32)

    # MANO layer
    mano_right = ManoLayer(
        side="right", mano_root=CFG.mano.model_path, use_pca=False, ncomps=45
    ).to(device)

    # ============================================================
    # Process each hand independently
    # ============================================================
    all_tip_dists = {}  # side -> (F, 5)

    for side in hands:
        print(f"\n{'='*70}")
        print(f"Processing {side} hand")
        print(f"{'='*70}")

        pose_np = hand_data[f'{side}_hand_pose'].astype(np.float32)
        beta_np = hand_data[f'{side}_hand_beta'].astype(np.float32)
        trans_np = hand_data[f'{side}_hand_translation'].astype(np.float32).copy()
        base_rot_np = hand_data[f'{side}_hand_base_rot']
        if beta_np.ndim == 1:
            beta_np = beta_np.reshape(1, -1)
        br = base_rot_np if (base_rot_np.ndim == 3 and len(base_rot_np) > 0) else None

        # Precompute local coords
        verts_local, tips_local = precompute_hand_local(
            mano_right, pose_np, beta_np, br, side, device
        )

        # Initial fingertip distances & grasp classification
        print(f"\n[Stage 0] Initial diagnostics (grasp_thresh={grasp_thresh*1000:.0f}mm, min_fingers={min_grasp_fingers})")
        grasp_mask = np.zeros(num_frames, dtype=bool)
        init_tip_dists = np.zeros((num_frames, 5), dtype=np.float32)

        for f in range(num_frames):
            tips_world = tips_local[f] + trans_np[f]
            dists = compute_tip_distances_exact(tips_world, obj_mesh, obj_Rs[f], obj_ts[f])
            init_tip_dists[f] = dists
            close_count = (dists < grasp_thresh).sum()
            grasp_mask[f] = close_count >= min_grasp_fingers

        n_grasp = grasp_mask.sum()
        print(f"  Grasping frames: {n_grasp}/{num_frames}")
        print(f"  Mean tip dist: {init_tip_dists.mean()*1000:.1f} mm")
        for i, name in enumerate(FINGERTIP_NAMES):
            d = init_tip_dists[:, i].mean()
            n_close = (init_tip_dists[:, i] < grasp_thresh).sum()
            print(f"    {name}: {d*1000:.1f} mm ({n_close} close frames)")

        if n_grasp == 0:
            print(f"  No grasping frames detected, skipping optimization for {side} hand")
            all_tip_dists[side] = init_tip_dists
            continue

        grasp_frames = np.where(grasp_mask)[0]

        # Transfer to GPU
        verts_local_t = torch.tensor(verts_local, device=device, dtype=torch.float32)
        tips_local_t = torch.tensor(tips_local, device=device, dtype=torch.float32)
        trans_t = torch.tensor(trans_np, device=device, dtype=torch.float32)

        # ---- Stage 1: Analytical global offset ----
        print(f"\n[Stage 1] Analytical global offset")
        global_offset = compute_global_offset(
            tips_local, trans_np, obj_mesh, obj_Rs, obj_ts, grasp_mask
        )
        print(f"  Estimated offset: [{global_offset*1000}] mm")

        # ---- Stage 2: Global refinement (coarse SDF) ----
        print(f"\n[Stage 2] Global optimization ({global_steps} steps, coarse SDF)")
        global_dt, global_dr = optimize_global(
            verts_local_t, tips_local_t, trans_t,
            obj_rvecs_t, obj_trans_t,
            sdf_grid_coarse, grid_min_coarse_t, grid_max_coarse_t,
            grasp_frames, device,
            init_delta_trans=global_offset,
            steps=global_steps, lr=global_lr,
            w_pen=w_pen, w_tip=w_tip, w_prox=w_prox,
        )
        print(f"  Final global delta_trans: [{global_dt*1000}] mm")
        print(f"  Final global delta_rot: [{global_dr*180/np.pi}] deg")

        # ---- Stage 3: Per-frame refinement (fine SDF) ----
        print(f"\n[Stage 3] Per-frame refinement ({perframe_steps} steps, fine SDF)")
        pf_dt, pf_dr = optimize_per_frame(
            verts_local_t, tips_local_t, trans_np,
            obj_rvecs_t, obj_trans_t,
            sdf_grid_fine, grid_min_fine_t, grid_max_fine_t,
            grasp_mask, device, num_frames,
            init_delta_trans_global=global_dt,
            init_delta_rot_global=global_dr,
            steps=perframe_steps, lr=perframe_lr,
            w_pen=w_pen, w_tip=w_tip, w_prox=w_prox,
            w_reg=w_reg, w_smooth=w_smooth, prox_thresh=0.03,
        )

        # ---- Stage 4: Post-processing ----
        print(f"\n[Stage 4] Post-processing")

        # Temporal smoothing
        pf_dt = temporal_smooth(pf_dt, window=smooth_window)
        pf_dr = temporal_smooth(pf_dr, window=smooth_window)
        print(f"  Applied temporal smoothing (window={smooth_window})")

        # Fix penetration (fine SDF, more steps)
        pf_dt, n_fixed = fix_penetration_pass(
            verts_local, pf_dt, pf_dr, trans_np,
            obj_rvecs_t, obj_trans_t,
            sdf_grid_fine, grid_min_fine_t, grid_max_fine_t,
            device, num_frames, steps=100, lr=0.001,
        )
        print(f"  Fixed penetration in {n_fixed} frames")

        # Apply deltas to original translation
        # new_verts = R_delta @ verts_local + (original_trans + delta_trans)
        # But we store translation only (not rotation) in the pkl.
        # So we need to bake R_delta into verts_local and recompute effective translation.
        # Actually, since pose is frozen and we only change translation in pkl,
        # we need to account for rotation by adjusting translation:
        # verts_world = R_delta @ verts_local + original_trans + delta_trans
        # The MANO forward gives: verts_world_orig = verts_local + original_trans
        # After optimization: verts_world_new = R_delta @ verts_local + original_trans + delta_trans
        # If we store new_trans = original_trans + delta_trans, the viewer will compute:
        #   verts_local + new_trans = verts_local + original_trans + delta_trans
        # This MISSES the R_delta rotation!
        # Solution: bake rotation into the effective translation per-vertex is not possible.
        # Instead, save the delta_rot to base_rot (for left) or as a separate field.
        # Simplest: apply R_delta to each verts_local, then the centroid shift IS the effective trans change.

        # Compute effective new translations that account for both rotation and translation
        new_trans = np.zeros_like(trans_np)
        for f in range(num_frames):
            R_f = Rotation.from_rotvec(pf_dr[f]).as_matrix().astype(np.float32)
            # The root (wrist) is at origin in local frame
            # After rotation: root stays at origin, verts rotate around it
            # new wrist position in world = original_trans + delta_trans
            # But verts are rotated, so we also need to adjust base_rot
            new_trans[f] = trans_np[f] + pf_dt[f]

        # Update base_rot to include the delta rotation
        if side == 'left' and br is not None:
            new_base_rot = np.zeros_like(br)
            for f in range(num_frames):
                R_delta = Rotation.from_rotvec(pf_dr[f]).as_matrix().astype(np.float32)
                rot_idx = min(f, len(br) - 1)
                new_base_rot[rot_idx] = R_delta @ br[rot_idx]
            hand_data[f'{side}_hand_base_rot'] = new_base_rot
        elif side == 'right':
            # For right hand: bake delta_rot into the pose's global orientation (first 3 components)
            # pose[:3] is the global rotation in axis-angle
            new_pose = pose_np.copy()
            for f in range(num_frames):
                R_delta = Rotation.from_rotvec(pf_dr[f])
                R_orig = Rotation.from_rotvec(pose_np[f, :3])
                R_new = R_delta * R_orig
                new_pose[f, :3] = R_new.as_rotvec()
            hand_data[f'{side}_hand_pose'] = new_pose

        hand_data[f'{side}_hand_translation'] = new_trans

        # Final diagnostics
        print(f"\n[Final] {side} hand fingertip distances:")
        final_tip_dists = np.zeros((num_frames, 5), dtype=np.float32)
        total_pen = 0
        for f in range(num_frames):
            R_f = Rotation.from_rotvec(pf_dr[f]).as_matrix().astype(np.float32)
            tips_world = tips_local[f] @ R_f.T + new_trans[f]
            dists = compute_tip_distances_exact(tips_world, obj_mesh, obj_Rs[f], obj_ts[f])
            final_tip_dists[f] = dists

            verts_world = verts_local[f] @ R_f.T + new_trans[f]
            verts_obj = (verts_world - obj_ts[f]) @ obj_Rs[f]
            # Pen check via fine SDF lookup
            norm = (verts_obj - grid_min_fine) / (grid_max_fine - grid_min_fine)
            idx = np.clip((norm * (sdf_resolution - 1)).astype(np.int32), 0, sdf_resolution - 1)
            svals = sdf_np_fine[idx[:, 0], idx[:, 1], idx[:, 2]]
            total_pen += int((svals < 0).sum())

        all_tip_dists[side] = final_tip_dists
        print(f"  Mean tip dist: {init_tip_dists.mean()*1000:.1f} -> {final_tip_dists.mean()*1000:.1f} mm")
        print(f"  Total pen verts: {total_pen}")
        for i, name in enumerate(FINGERTIP_NAMES):
            d_before = init_tip_dists[:, i].mean()
            d_after = final_tip_dists[:, i].mean()
            print(f"    {name}: {d_before*1000:.1f} -> {d_after*1000:.1f} mm")

        # Grasping frames only
        if n_grasp > 0:
            gf_before = init_tip_dists[grasp_mask].mean()
            gf_after = final_tip_dists[grasp_mask].mean()
            print(f"  Grasping frames only: {gf_before*1000:.1f} -> {gf_after*1000:.1f} mm")

    # ============================================================
    # Save results
    # ============================================================
    print(f"\n[INFO] Saving results to {save_folder}")

    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    for side in hands:
        pkl_data['hand_pose'][f'{side}_hand_translation'] = hand_data[f'{side}_hand_translation'].astype(np.float32)
        if f'{side}_hand_pose' in hand_data and side == 'right':
            pkl_data['hand_pose'][f'{side}_hand_pose'] = hand_data[f'{side}_hand_pose'].astype(np.float32)
        if f'{side}_hand_base_rot' in hand_data and side == 'left':
            pkl_data['hand_pose'][f'{side}_hand_base_rot'] = hand_data[f'{side}_hand_base_rot'].astype(np.float32)
    with open(save_folder / "result_hand_optimized.pkl", 'wb') as f:
        pickle.dump(pkl_data, f)

    np.save(save_folder / "poses_o.npy", poses_o)

    tip_dist_array = np.stack([all_tip_dists[s] for s in hands], axis=1)
    np.save(save_folder / "fingertip_distances.npy", tip_dist_array)

    tip_meta = {
        "hands": hands,
        "finger_names": FINGERTIP_NAMES,
        "finger_joint_indices": FINGERTIP_INDICES,
        "shape": "(num_frames, num_hands, 5)",
        "unit": "meters",
        "grasp_thresh_m": grasp_thresh,
        "min_grasp_fingers": min_grasp_fingers,
    }
    with open(save_folder / "fingertip_meta.json", 'w') as f:
        json.dump(tip_meta, f, indent=2)

    print(f"[INFO] Saved: result_hand_optimized.pkl, poses_o.npy, fingertip_distances.npy")
    print(f"[INFO] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stage contact optimizer")
    parser.add_argument("--sequence_folder", type=str, required=True)
    parser.add_argument("--object_idx", type=int, default=1)
    parser.add_argument("--sdf_resolution", type=int, default=128)
    parser.add_argument("--sdf_padding", type=float, default=0.15)
    parser.add_argument("--grasp_thresh", type=float, default=0.05,
                        help="Fingertip distance threshold for grasping (meters)")
    parser.add_argument("--min_grasp_fingers", type=int, default=2)
    parser.add_argument("--global_steps", type=int, default=100)
    parser.add_argument("--perframe_steps", type=int, default=200)
    parser.add_argument("--global_lr", type=float, default=0.002)
    parser.add_argument("--perframe_lr", type=float, default=0.001)
    parser.add_argument("--w_pen", type=float, default=1000.0)
    parser.add_argument("--w_tip", type=float, default=200.0)
    parser.add_argument("--w_prox", type=float, default=50.0)
    parser.add_argument("--w_reg", type=float, default=5.0)
    parser.add_argument("--w_smooth", type=float, default=10.0)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--output_suffix", type=str, default="contact_optimizer")
    args = parser.parse_args()

    run_contact_optimizer(
        sequence_folder=args.sequence_folder,
        object_idx=args.object_idx,
        sdf_resolution=args.sdf_resolution,
        sdf_padding=args.sdf_padding,
        grasp_thresh=args.grasp_thresh,
        min_grasp_fingers=args.min_grasp_fingers,
        global_steps=args.global_steps,
        perframe_steps=args.perframe_steps,
        global_lr=args.global_lr,
        perframe_lr=args.perframe_lr,
        w_pen=args.w_pen,
        w_tip=args.w_tip,
        w_prox=args.w_prox,
        w_reg=args.w_reg,
        w_smooth=args.w_smooth,
        smooth_window=args.smooth_window,
        output_suffix=args.output_suffix,
    )
