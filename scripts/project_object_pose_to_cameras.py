#!/usr/bin/env python3
"""
Project object mesh back into the RealSense cameras using saved object poses.

This is a 2D verification script for HO-Cap object poses. It renders the tool
mesh into all 8 camera views using:
  - camera intrinsics
  - camera extrinsics
  - object mesh
  - saved object world poses (quat+trans)

It is intentionally separate from the Viser viewer so you can verify whether a
misalignment comes from the 3D viewer or from the underlying tool poses / mesh.

Examples:
    conda activate hocap-annotation

    python scripts/project_object_pose_to_cameras.py \
        --data_folder /viscam/projects/robotool/data/videos_0210/fork_flip_naan/20260210_bigwoodenfork_flip_naan_in_blackpan_10 \
        --pose_source fd_pose_solver

    python scripts/project_object_pose_to_cameras.py \
        --data_folder ... \
        --pose_source joint_pose_solver \
        --frame 120
"""

import argparse
import json
import re
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import trimesh
import yaml
import cv2
import h5py
from tqdm import tqdm

import sys
HOCAP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HOCAP_ROOT))

from hocap_annotation.rendering.offscreen_renderer import OffscreenRenderer
from hocap_annotation.utils.cv_utils import draw_image_grid, draw_image_overlay, create_video_from_rgb_images
from hocap_annotation.utils.io import read_rgb_image, write_rgb_image
from hocap_annotation.utils.transforms import quat_to_mat


def annotated_folder_for(data_folder: Path) -> Path:
    exp_name = data_folder.name
    task_name = data_folder.parent.name
    data_root_name = data_folder.parent.parent.name
    base_dir = data_folder.parent.parent.parent
    return base_dir / f"{data_root_name}_annotated" / task_name / exp_name


def load_pose_array(annotated: Path, pose_source: str) -> np.ndarray:
    source_map = {
        "joint_pose_solver": annotated / "processed" / "joint_pose_solver" / "poses_o.npy",
        "object_pose_solver": annotated / "processed" / "object_pose_solver" / "poses_o.npy",
        "fd_pose_solver": annotated / "processed" / "fd_pose_solver" / "fd_poses_merged_fixed.npy",
    }
    pose_path = source_map[pose_source]
    if not pose_path.exists():
        raise FileNotFoundError(f"pose file not found: {pose_path}")
    poses = np.load(pose_path).astype(np.float32)
    if poses.ndim == 2 and poses.shape[1] == 7:
        poses = poses[None]
    if poses.ndim != 3 or poses.shape[-1] != 7:
        raise ValueError(f"unexpected pose shape {poses.shape} in {pose_path}")
    return poses


def resolve_object_mesh(meta: dict, object_id: str) -> tuple[str, Path]:
    models_folder = Path(meta["models_folder"])
    subject_id = str(meta.get("subject_id", "") or "")
    mapping_path = Path("/viscam/u/chenrq/crq_ws/scripts/mesh_name_mapping.json")

    candidate_ids = [object_id]
    if mapping_path.exists():
        mapping = json.load(open(mapping_path, "r"))
        subject_key = re.sub(r"^\d{8}_", "", subject_id)
        subject_key = re.sub(r"_\d+$", "", subject_key)
        mapped = mapping.get(subject_key)
        if mapped and mapped not in candidate_ids:
            candidate_ids.append(mapped)

    for mesh_id in candidate_ids:
        for cand in ["cleaned_mesh_10000.obj", "textured_mesh.obj", "mesh.obj", "bk.obj"]:
            mesh_path = models_folder / mesh_id / cand
            if mesh_path.exists():
                return mesh_id, mesh_path

    raise FileNotFoundError(
        f"no mesh found for object_id={object_id}; tried {candidate_ids} under {models_folder}"
    )


def load_cameras(meta: dict):
    calib = Path(meta["calibration_yaml_path"])
    cams = yaml.safe_load(open(calib, "r"))
    cam_by_id = {str(c["camera_id"]).zfill(2): c for c in cams}
    serials = meta.get("realsense", {}).get("serials", [])
    Ks = []
    Ts = []
    for serial in serials:
        cam = cam_by_id[serial]
        Ks.append(np.asarray(cam["color_intrinsic_matrix"], dtype=np.float32))
        Ts.append(np.asarray(cam["transformation"], dtype=np.float32))
    return serials, np.stack(Ks, axis=0), Ts


def _read_frame_from_video(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def read_colors(data_folder: Path, serials: list[str], frame_idx: int):
    images = []
    for cam_idx, _serial in enumerate(serials):
        video_path = data_folder / f"cam{cam_idx}_rgb.mp4"
        if video_path.exists():
            images.append(_read_frame_from_video(video_path, frame_idx))
            continue

        # Fallback for older extracted-image layouts.
        serial_path = data_folder / _serial / f"color_{frame_idx:06d}.jpg"
        images.append(read_rgb_image(serial_path))
    return images


def load_mask_dataset(annotated: Path):
    candidates = [
        annotated / "masks" / "tool_masks" / "masks.h5",
        annotated / "masks" / "masks.h5",
    ]
    for p in candidates:
        if p.exists():
            f = h5py.File(p, "r")
            if "masks" not in f:
                f.close()
                continue
            return f, f["masks"], p
    return None, None, None


def colorize_mask(mask: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    out[mask.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description="Project object poses back to all 8 cameras")
    ap.add_argument("--data_folder", type=str, required=True)
    ap.add_argument(
        "--pose_source",
        type=str,
        default="fd_pose_solver",
        choices=["joint_pose_solver", "object_pose_solver", "fd_pose_solver"],
    )
    ap.add_argument("--frame", type=int, default=None, help="If set, render only one frame")
    ap.add_argument("--object_idx", type=int, default=0)
    ap.add_argument("--save_root", type=str, default="")
    ap.add_argument("--video_fps", type=int, default=30)
    ap.add_argument("--with_masks", action="store_true", help="Overlay annotated tool masks too")
    ap.add_argument("--mask_stride", type=int, default=1,
                    help="When --with_masks is set, only save every Nth frame grid")
    ap.add_argument("--no_mask_video", action="store_true",
                    help="When --with_masks is set, skip writing overlay_with_masks.mp4")
    args = ap.parse_args()

    data_folder = Path(args.data_folder).resolve()
    annotated = annotated_folder_for(data_folder)
    meta = yaml.safe_load(open(data_folder / "meta.yaml", "r"))

    serials, Ks, cam_poses = load_cameras(meta)
    object_ids = meta.get("object_ids", [])
    if not object_ids:
        raise ValueError("meta.yaml has no object_ids")
    if not (0 <= args.object_idx < len(object_ids)):
        raise IndexError(f"object_idx {args.object_idx} out of range for {len(object_ids)} objects")

    object_id = object_ids[args.object_idx]
    resolved_id, mesh_path = resolve_object_mesh(meta, object_id)
    poses = load_pose_array(annotated, args.pose_source)
    if args.object_idx >= poses.shape[0]:
        raise IndexError(f"pose array has only {poses.shape[0]} objects; requested {args.object_idx}")

    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    mesh_name = object_id

    renderer = OffscreenRenderer()
    for serial, K in zip(serials, Ks):
        renderer.add_camera(K, serial)
    renderer.add_mesh(mesh, mesh_name)

    if args.save_root:
        save_root = Path(args.save_root).resolve()
    else:
        save_root = annotated / "vis_reproject" / args.pose_source / object_id
    save_root.mkdir(parents=True, exist_ok=True)

    num_frames = poses.shape[1]
    if args.frame is not None:
        frame_indices = [args.frame]
    else:
        stride = max(1, args.mask_stride) if args.with_masks and args.no_mask_video else 1
        frame_indices = list(range(0, num_frames, stride))
    vis_frames = []
    mask_h5, mask_ds, mask_path = load_mask_dataset(annotated) if args.with_masks else (None, None, None)

    print(f"[INFO] data_folder   = {data_folder}")
    print(f"[INFO] annotated     = {annotated}")
    print(f"[INFO] pose_source   = {args.pose_source}")
    print(f"[INFO] object_id     = {object_id}")
    print(f"[INFO] resolved_mesh = {resolved_id}")
    print(f"[INFO] mesh_path     = {mesh_path}")
    print(f"[INFO] save_root     = {save_root}")
    if mask_path is not None:
        print(f"[INFO] mask_path     = {mask_path}")

    try:
        for frame_idx in tqdm(frame_indices, desc="Rendering", ncols=100):
            pose_qt = poses[args.object_idx, frame_idx]
            if np.all(pose_qt == -1):
                print(f"[WARN] frame {frame_idx}: pose is placeholder, skipping")
                continue
            pose_4x4 = quat_to_mat(pose_qt)
            render_images = renderer.get_render_colors(
                width=int(meta["realsense"]["width"]),
                height=int(meta["realsense"]["height"]),
                cam_names=serials,
                cam_poses=cam_poses,
                mesh_names=[mesh_name],
                mesh_poses=[pose_4x4],
            )
            color_images = read_colors(data_folder, serials, frame_idx)
            overlay_images = []
            for cam_idx, (color_image, render_image) in enumerate(zip(color_images, render_images)):
                overlay = draw_image_overlay(color_image, render_image, 0.7)
                if mask_ds is not None:
                    mask = np.asarray(mask_ds[frame_idx, cam_idx]) > 0
                    overlay = draw_image_overlay(overlay, colorize_mask(mask), 0.35)
                overlay_images.append(overlay)
            vis_image = draw_image_grid(
                images=overlay_images,
                names=serials,
                max_cols=4,
                facecolor="black",
                titlecolor="white",
            )
            write_rgb_image(save_root / f"{frame_idx:06d}.jpg", vis_image)
            if not (args.with_masks and args.no_mask_video):
                vis_frames.append(vis_image)
    finally:
        if mask_h5 is not None:
            mask_h5.close()

    if args.frame is None and vis_frames:
        if args.with_masks and args.no_mask_video:
            print(f"[INFO] skipped mask video; wrote sparse frames with stride={max(1, args.mask_stride)}")
        else:
            video_name = "overlay_with_masks.mp4" if args.with_masks else "overlay.mp4"
            video_path = save_root / video_name
            create_video_from_rgb_images(video_path, vis_frames, fps=args.video_fps)
            print(f"[INFO] wrote video: {video_path}")


if __name__ == "__main__":
    main()
