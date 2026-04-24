#!/usr/bin/env python
"""
Convert a single HOI4D-style sequence (cam/, rgb/, depth/, mano/ per-frame
files) into the format the HO-Cap-Annotation hand pipeline consumes:

    <out_root>/hoi4d/hoi4d_seqs/<seq_name>/
        data00000000.h5      # imgs (N, 1, H, W, 3) uint8 + depths (N, 1, H, W) uint16 mm
        meta.yaml            # num_frames, serials=['00'], intrinsics, etc.

    <out_root>/hoi4d_calib/
        calibration_hoi4d_<seq_name>.yaml    # 1-cam calibration (identity extrinsic)

    <out_root>/hoi4d_annotated/hoi4d_seqs/<seq_name>/
        gt_result_hand_optimized.pkl         # HOI4D GT converted to our schema

The calibration uses IDENTITY extrinsic, so the pipeline's "world frame" IS
the camera frame of each frame. Comparison to HOI4D GT (which stores per-hand
trans in the camera frame of each frame) is then a direct apples-to-apples
compare of hand translation + MANO theta.

Usage:
    python scripts/hoi4d_adapter/convert_hoi4d.py \
        --hoi4d_seq /abs/path/to/ZY20210800003_H3_C20_N11_S279_s03_T1 \
        --out_root  /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data
"""
import argparse
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml


def load_frame(frame_idx, seq_root, rgb_ext=".png"):
    name = f"{frame_idx:05d}"
    rgb = cv2.imread(str(seq_root / "rgb" / f"{name}{rgb_ext}"))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth_m = np.load(seq_root / "depth" / f"{name}.npy")     # float32, meters
    cam = np.load(seq_root / "cam" / f"{name}.npz", allow_pickle=True)
    intrinsic = np.array(cam["intrinsics"], dtype=np.float64)
    pose = np.array(cam["pose"], dtype=np.float64)            # 4x4 cam-to-world for this frame
    with open(seq_root / "mano" / f"{name}.pkl", "rb") as f:
        mano = pickle.load(f)
    return rgb, depth_m, intrinsic, pose, mano


def write_h5(out_path, imgs, depths_mm):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("imgs", data=imgs, dtype=np.uint8,
                          chunks=(1, 1, imgs.shape[2], imgs.shape[3], 3),
                          compression="lzf")
        f.create_dataset("depths", data=depths_mm, dtype=np.uint16,
                          chunks=(1, 1, depths_mm.shape[2], depths_mm.shape[3]),
                          compression="lzf")
    print(f"[h5]    {out_path}  shape imgs={imgs.shape}, depths={depths_mm.shape}")


def write_calib(out_path, intrinsic, serial="00"):
    cams = [{
        "camera_id": 0,
        "serial_number": serial,
        "transformation": np.eye(4).tolist(),   # identity = world == camera frame
        "color_intrinsic_matrix": intrinsic.tolist(),
        "depth_intrinsic_matrix": intrinsic.tolist(),
    }]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(cams, f, default_flow_style=False, sort_keys=False)
    print(f"[calib] {out_path}")


def write_meta(out_path, num_frames, H, W, calib_path, models_folder,
                serial="00", tool_name="hoi4d"):
    meta = {
        "num_frames": int(num_frames),
        "start_frame": 0,
        "object_ids": [tool_name],
        "mano_sides": ["left", "right"],
        "subject_id": "hoi4d_subject",
        "realsense": {
            "serials": [serial],
            "width": int(W),
            "height": int(H),
        },
        "hololens": {"serial": "none", "pv_height": 720, "pv_width": 1280},
        "have_hololens": False,
        "have_mano": True,
        "task_id": 1,
        "thresholds": [-2.0, 2.0, -2.0, 2.0, -0.5, 3.0],
        "calibration_yaml_path": str(calib_path),
        "models_folder": str(models_folder),
        "betas": [0.0] * 10,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
    print(f"[meta]  {out_path}")


def convert_gt_mano(all_manos, out_path, H, W, intrinsic):
    """HOI4D: per-frame mano.pkl with mano_mask(2), theta(2,48), beta(2,10),
    trans(2,3). Convention here: row 0 = right, row 1 = left (HOI4D standard).
    Build a result_hand_optimized.pkl compatible with dataset.py schema."""
    n = len(all_manos)

    left_pose = np.zeros((n, 48), dtype=np.float32)
    left_trans = np.zeros((n, 3), dtype=np.float32)
    left_mask = np.zeros(n, dtype=bool)
    right_pose = np.zeros((n, 48), dtype=np.float32)
    right_trans = np.zeros((n, 3), dtype=np.float32)
    right_mask = np.zeros(n, dtype=bool)
    beta_left = None
    beta_right = None

    for i, m in enumerate(all_manos):
        mm = m["mano_mask"]           # shape (2,) bool-ish
        theta = np.asarray(m["theta"], dtype=np.float32)
        beta = np.asarray(m["beta"], dtype=np.float32)
        trans = np.asarray(m["trans"], dtype=np.float32)
        # HOI4D convention: [right, left]
        if mm[0]:
            right_pose[i] = theta[0]
            right_trans[i] = trans[0]
            right_mask[i] = True
            if beta_right is None:
                beta_right = beta[0]
        if mm[1]:
            left_pose[i] = theta[1]
            left_trans[i] = trans[1]
            left_mask[i] = True
            if beta_left is None:
                beta_left = beta[1]

    if beta_left is None: beta_left = np.zeros(10, dtype=np.float32)
    if beta_right is None: beta_right = np.zeros(10, dtype=np.float32)

    # Project 3D joints to 2D would require the MANO layer — out of scope here.
    # Leave 2D joints as zeros; the optimizer downstream will regenerate if needed.
    joints_2d_zero = np.zeros((n, 1, 21, 2), dtype=np.float32)

    data = {
        "video_name": "hoi4d_seq",
        "object_names": {"target_object": "target", "tool_object": "tool"},
        "camera_seq": ["00"],
        "camera_intrinsics": np.asarray([intrinsic], dtype=np.float64),
        "camera_extrinsics": np.asarray([np.eye(4)], dtype=np.float64),
        "hand_pose": {
            "left_hand_pose": left_pose,
            "left_hand_beta": beta_left.reshape(1, -1),
            "left_hand_translation": left_trans,
            "left_hand_base_rot": np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
            "right_hand_pose": right_pose,
            "right_hand_beta": beta_right.reshape(1, -1),
            "right_hand_translation": right_trans,
        },
        "hand_joints": {
            "left_hand_joints_2d": joints_2d_zero.copy(),
            "right_hand_joints_2d": joints_2d_zero.copy(),
        },
        "target_object_pose": [],
        "tool_object_pose": [],
        "deformable_object_pointcloud": [],
        "masks": {"deformable": []},
        "num_frames": int(n),
        "start_frame": 0,
        "end_frame": int(n - 1),
        "hoi4d_gt": {
            "left_mask":  left_mask,
            "right_mask": right_mask,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[gt]    {out_path}  (left valid frames: {left_mask.sum()}/{n}, "
          f"right: {right_mask.sum()}/{n})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hoi4d_seq", required=True,
                    help="Path to a single HOI4D sequence folder (contains cam/, rgb/, depth/, mano/)")
    ap.add_argument("--out_root", required=True,
                    help="Root output dir (typically .../HO-Cap-Annotation/data)")
    ap.add_argument("--video_folder_name", default="hoi4d",
                    help="top-level folder name under out_root (default 'hoi4d')")
    ap.add_argument("--task_name", default="hoi4d_seqs",
                    help="task folder name under video_folder_name (default 'hoi4d_seqs')")
    ap.add_argument("--tool_name", default="hoi4d_obj",
                    help="placeholder object name for meta.yaml (hand-only use, unused)")
    ap.add_argument("--skip_gt", action="store_true",
                    help="don't write gt_result_hand_optimized.pkl")
    ap.add_argument("--models_folder", default=None,
                    help="models folder for meta.yaml (default: <out_root>/models)")
    ap.add_argument("--num_frames_limit", type=int, default=None,
                    help="only convert first N frames (for smoke tests)")
    args = ap.parse_args()

    seq_root = Path(args.hoi4d_seq).resolve()
    seq_name = seq_root.name
    out_root = Path(args.out_root).resolve()

    # --- enumerate frames ---
    rgb_dir = seq_root / "rgb"
    assert rgb_dir.is_dir(), f"missing {rgb_dir}"
    frame_ids = sorted(int(p.stem) for p in rgb_dir.glob("*.png"))
    if args.num_frames_limit:
        frame_ids = frame_ids[:args.num_frames_limit]
    n = len(frame_ids)
    print(f"[info] {seq_name}: {n} frames")

    # --- probe first frame for H, W, intrinsics ---
    rgb0, depth0, intr0, _, _ = load_frame(frame_ids[0], seq_root)
    H, W = rgb0.shape[:2]
    print(f"[info] resolution {W}x{H}")

    # --- allocate arrays ---
    imgs = np.zeros((n, 1, H, W, 3), dtype=np.uint8)
    depths_mm = np.zeros((n, 1, H, W), dtype=np.uint16)
    manos = []

    for i, fid in enumerate(frame_ids):
        rgb, depth_m, _, _, mano = load_frame(fid, seq_root)
        imgs[i, 0] = rgb
        depths_mm[i, 0] = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        manos.append(mano)
        if (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{n}")

    # --- output paths ---
    video_folder = out_root / args.video_folder_name
    task_folder = video_folder / args.task_name
    seq_out = task_folder / seq_name
    h5_path = seq_out / "data00000000.h5"
    meta_path = seq_out / "meta.yaml"

    calib_folder = out_root / f"{args.video_folder_name}_calib"
    calib_path = calib_folder / f"calibration_hoi4d_{seq_name}.yaml"

    annotated_folder = out_root / f"{args.video_folder_name}_annotated" / args.task_name / seq_name

    models_folder = Path(args.models_folder) if args.models_folder else (out_root / "models")

    # --- write everything ---
    write_h5(h5_path, imgs, depths_mm)
    write_calib(calib_path, intr0)
    write_meta(meta_path, n, H, W, calib_path, models_folder,
                serial="00", tool_name=args.tool_name)

    if not args.skip_gt:
        convert_gt_mano(manos, annotated_folder / "gt_result_hand_optimized.pkl",
                         H, W, intr0)

    # --- summary ---
    print()
    print("[NEXT STEPS]")
    print(f"  1) Run hand reconstruction on this sequence:")
    print(f"     cd $HAND_ROOT && conda activate reconstruct-hand")
    print(f"     python cluster_reconstruct.py --sequence_folder {seq_out}")
    print(f"     python cluster_optimize_hand.py --file_name {annotated_folder}/result.pkl")
    print(f"  2) Compare against GT:")
    print(f"     {annotated_folder}/gt_result_hand_optimized.pkl        (GT)")
    print(f"     {annotated_folder}/result_hand_optimized.pkl           (your pipeline)")


if __name__ == "__main__":
    main()
