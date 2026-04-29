#!/usr/bin/env python3
"""
Hand + object interactive 3D viewer for HO-Cap-Annotation outputs (Viser).

Loads:
  - hand:   <annotated>/result_hand_optimized.pkl  (preferred)
            <annotated>/result.pkl                 (fallback)
  - object: <annotated>/processed/joint_pose_solver/poses_o.npy   (preferred)
            <annotated>/processed/object_pose_solver/poses_o.npy
            <annotated>/processed/fd_pose_solver/fd_poses_merged_fixed.npy

Modeled after scripts/visualize_hand_viser.py (chunked-aware, frame count taken
from the pkl rather than meta.yaml). Adds object meshes overlaid on the hand
mesh, multi-object support, and a runtime "Object pose source" switch.

Usage:
    conda activate hocap-annotation

    # Single experiment:
    python scripts/visualize_hand_object_viser.py \
        --data_folder data/videos_0101/mallet_crush_nuts/20260104_largeplate_..._1

    # Browse every experiment under a task folder:
    python scripts/visualize_hand_object_viser.py \
        --task_folder data/videos_0101/mallet_crush_nuts

    # Force a particular object pose source:
    python scripts/visualize_hand_object_viser.py \
        --data_folder ... --pose_source fd_pose_solver
"""

import sys
import os
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import viser
import yaml
from scipy.spatial.transform import Rotation as R

# Resolve paths before chdir so relative CLI paths still work.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--data_folder", type=str, default="")
_pre.add_argument("--task_folder", type=str, default="")
_pre.add_argument("--pkl_path", type=str, default="")
_pre_args, _ = _pre.parse_known_args()
_DATA_FOLDER = Path(_pre_args.data_folder).resolve() if _pre_args.data_folder else None
_TASK_FOLDER = Path(_pre_args.task_folder).resolve() if _pre_args.task_folder else None
_PKL_PATH = Path(_pre_args.pkl_path).resolve() if _pre_args.pkl_path else None

HOCAP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HOCAP_ROOT))
os.chdir(str(HOCAP_ROOT))


# ============================================================
# Hand pkl loading (mirrors visualize_hand_viser.py)
# ============================================================

def load_hand_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    hp = data["hand_pose"]

    def _as_np(x):
        if x is None:
            return np.zeros(0)
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, list):
            if len(x) == 0:
                return np.zeros(0)
            out = []
            for item in x:
                if hasattr(item, "detach"):
                    item = item.detach().cpu().numpy()
                out.append(np.asarray(item))
            return np.stack(out, axis=0) if out else np.zeros(0)
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    result = {
        "left_hand_pose": _as_np(hp.get("left_hand_pose", [])),
        "left_hand_beta": np.asarray(hp.get("left_hand_beta", [])).squeeze(),
        "left_hand_translation": _as_np(hp.get("left_hand_translation", [])),
        "left_hand_base_rot": np.asarray(hp.get("left_hand_base_rot", []))
            if hp.get("left_hand_base_rot") is not None else np.eye(3),
        "right_hand_pose": _as_np(hp.get("right_hand_pose", [])),
        "right_hand_beta": np.asarray(hp.get("right_hand_beta", [])).squeeze(),
        "right_hand_translation": _as_np(hp.get("right_hand_translation", [])),
    }
    for k in ["left_hand_pose", "right_hand_pose",
              "left_hand_translation", "right_hand_translation"]:
        if result[k].ndim == 3 and result[k].shape[1] == 1:
            result[k] = result[k][:, 0, :]
    return result


def _infer_num_frames_hand(hand):
    for key in ["right_hand_pose", "left_hand_pose",
                "right_hand_translation", "left_hand_translation"]:
        arr = hand.get(key)
        if arr is not None and getattr(arr, "shape", (0,))[0] > 0:
            return int(arr.shape[0])
    return 0


def reconstruct_left_verts(hand, frame_idx, mano_layer_right):
    pose_arr = hand["left_hand_pose"]
    if pose_arr.ndim < 2 or frame_idx >= pose_arr.shape[0]:
        return None
    try:
        pose = torch.tensor(pose_arr[frame_idx], dtype=torch.float32).cuda().unsqueeze(0)
        trans = torch.tensor(hand["left_hand_translation"][frame_idx],
                              dtype=torch.float32).cuda().unsqueeze(0)
        beta = torch.tensor(hand["left_hand_beta"], dtype=torch.float32).cuda()
        if beta.ndim == 1:
            beta = beta.unsqueeze(0)

        base = hand["left_hand_base_rot"]
        if base.ndim == 3 and base.shape[0] > 0:
            ri = min(frame_idx, base.shape[0] - 1)
            base_rot = torch.tensor(base[ri], dtype=torch.float32).cuda()
        elif base.ndim == 2 and base.shape == (3, 3):
            base_rot = torch.tensor(base, dtype=torch.float32).cuda()
        else:
            base_rot = torch.eye(3, dtype=torch.float32).cuda()

        verts, joints = mano_layer_right(pose, beta)
        verts = verts[0] / 1000.0
        joints = joints[0] / 1000.0
        root = joints[0].clone()
        verts = verts - root
        verts[:, 0] *= -1
        verts = verts @ base_rot.T
        verts = verts + trans
        return verts.detach().cpu().numpy()
    except Exception as e:
        if frame_idx < 3:
            print(f"[WARN] left hand frame {frame_idx}: {e}")
        return None


def reconstruct_right_verts(hand, frame_idx, mano_layer_right):
    pose_arr = hand["right_hand_pose"]
    if pose_arr.ndim < 2 or frame_idx >= pose_arr.shape[0]:
        return None
    try:
        pose = torch.tensor(pose_arr[frame_idx], dtype=torch.float32).cuda().unsqueeze(0)
        trans = torch.tensor(hand["right_hand_translation"][frame_idx],
                              dtype=torch.float32).cuda().unsqueeze(0)
        beta = torch.tensor(hand["right_hand_beta"], dtype=torch.float32).cuda()
        if beta.ndim == 1:
            beta = beta.unsqueeze(0)

        verts, joints = mano_layer_right(pose, beta)
        verts = verts[0] / 1000.0
        joints = joints[0] / 1000.0
        root = joints[0].clone()
        verts = verts - root
        verts = verts + trans
        return verts.detach().cpu().numpy()
    except Exception as e:
        if frame_idx < 3:
            print(f"[WARN] right hand frame {frame_idx}: {e}")
        return None


# ============================================================
# Object loading
# ============================================================

def transform_object_vertices(orig_vertices, pose_7d):
    """pose_7d = [qx, qy, qz, qw, tx, ty, tz] (matches mat_to_quat output)."""
    qx, qy, qz, qw, tx, ty, tz = pose_7d
    R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return (R_mat @ orig_vertices.T).T + np.array([tx, ty, tz])


def discover_object_pose_sources(annotated: Path):
    """Return ordered list of (source_name, npy_path) — best first.

    Each .npy is expected to be (num_obj, num_frames, 7) or (num_frames, 7).
    """
    sources = []
    candidates = [
        ("joint_pose_solver",  annotated / "processed" / "joint_pose_solver"  / "poses_o.npy"),
        ("object_pose_solver", annotated / "processed" / "object_pose_solver" / "poses_o.npy"),
        ("fd_pose_solver",     annotated / "processed" / "fd_pose_solver"     / "fd_poses_merged_fixed.npy"),
    ]
    for name, p in candidates:
        if p.exists():
            sources.append((name, p))
    return sources


def load_object_poses(npy_path: Path):
    """Return (num_obj, num_frames, 7). Promotes (N, 7) to (1, N, 7)."""
    arr = np.load(npy_path)
    if arr.ndim == 2 and arr.shape[1] == 7:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[-1] != 7:
        raise ValueError(f"unexpected pose shape {arr.shape} in {npy_path}; expected (O, N, 7)")
    return arr.astype(np.float32)


def load_object_meshes(meta_path: Path):
    """Return list of (object_id, vertices, faces). Empty if meta missing/unreadable."""
    if not meta_path.exists():
        return []
    try:
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
    except Exception as e:
        print(f"[WARN] failed to read {meta_path}: {e}")
        return []
    object_ids = meta.get("object_ids", []) or []
    models_folder = meta.get("models_folder")
    if not object_ids or not models_folder:
        return []
    models_folder = Path(models_folder)
    out = []
    for obj_id in object_ids:
        for cand in ["cleaned_mesh_10000.obj", "textured_mesh.obj", "mesh.obj"]:
            mp = models_folder / obj_id / cand
            if mp.exists():
                try:
                    m = trimesh.load(str(mp), process=False, force="mesh")
                    out.append((
                        obj_id,
                        np.asarray(m.vertices, dtype=np.float32).copy(),
                        np.asarray(m.faces).copy(),
                    ))
                except Exception as e:
                    print(f"[WARN] failed to load mesh {mp}: {e}")
                break
        else:
            print(f"[WARN] no mesh file for {obj_id} under {models_folder / obj_id}")
    return out


# ============================================================
# Path / experiment helpers
# ============================================================

def annotated_folder_for(data_folder: Path) -> Path:
    """data/<videos_X>/<task>/<exp> -> data/<videos_X>_annotated/<task>/<exp>."""
    exp_name = data_folder.name
    task_name = data_folder.parent.name
    data_root_name = data_folder.parent.parent.name
    base_dir = data_folder.parent.parent.parent
    return base_dir / f"{data_root_name}_annotated" / task_name / exp_name


def data_folder_for(annotated: Path) -> Path:
    """Reverse mapping: annotated -> raw data folder."""
    exp_name = annotated.name
    task_name = annotated.parent.name
    annotated_root = annotated.parent.parent.name  # videos_X_annotated
    if annotated_root.endswith("_annotated"):
        data_root = annotated_root[: -len("_annotated")]
    else:
        data_root = annotated_root
    base_dir = annotated.parent.parent.parent
    return base_dir / data_root / task_name / exp_name


def pick_hand_pkl(annotated: Path, prefer="optimized"):
    order = ["result_hand_optimized.pkl", "result.pkl"]
    if prefer == "reconstruct":
        order = order[::-1]
    for name in order:
        p = annotated / name
        if p.exists():
            return p
    return None


def discover_experiments(task_folder: Path):
    exps = []
    for sub in sorted(task_folder.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "meta.yaml").exists():
            exps.append(sub.name)
            continue
        ann = annotated_folder_for(sub)
        if pick_hand_pkl(ann) is not None:
            exps.append(sub.name)
    return exps


def load_camera_transforms(meta_path: Path):
    if not meta_path.exists():
        return []
    try:
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
    except Exception:
        return []
    calib = meta.get("calibration_yaml_path")
    if not calib or not Path(calib).exists():
        return []
    with open(calib, "r") as f:
        cams = yaml.safe_load(f)
    serial_to_T = {str(c["camera_id"]).zfill(2): np.array(c["transformation"]) for c in cams}
    serials = meta.get("realsense", {}).get("serials", [])
    return [serial_to_T[s] for s in serials if s in serial_to_T]


def flip_z(v):
    """Match visualize_hand_viser / viser_viewer convention so Z points down."""
    v = v.copy()
    v[..., 1] *= -1
    v[..., 2] *= -1
    return v


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Hand + object Viser viewer")
    ap.add_argument("--data_folder", type=str, default="")
    ap.add_argument("--task_folder", type=str, default="")
    ap.add_argument("--pkl_path", type=str, default="",
                    help="Visualize a specific hand pkl directly. The viewer will "
                         "still try to find object poses + meshes via the inferred "
                         "annotated/data folders.")
    ap.add_argument("--prefer", type=str, default="optimized",
                    choices=["optimized", "reconstruct"],
                    help="Prefer result_hand_optimized.pkl (default) or result.pkl.")
    ap.add_argument("--pose_source", type=str, default=None,
                    choices=[None, "joint_pose_solver", "object_pose_solver", "fd_pose_solver"],
                    help="Force a specific object pose source. Default: best available.")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    task_folder = _TASK_FOLDER
    data_folder = _DATA_FOLDER
    pkl_path = _PKL_PATH

    if not (task_folder or data_folder or pkl_path):
        ap.error("Must specify one of --data_folder / --task_folder / --pkl_path")

    # Build experiment list (same logic as visualize_hand_viser.py).
    if pkl_path is not None:
        # Best-effort: derive sequence folder from the pkl's annotated path so
        # we can still pull object meshes + poses.
        annotated_guess = pkl_path.parent
        derived_data_folder = data_folder_for(annotated_guess)
        exp_names = [annotated_guess.name]
        task_folder = None
        data_folder = derived_data_folder if derived_data_folder.exists() else None
    elif task_folder is not None:
        exp_names = discover_experiments(task_folder)
        if not exp_names:
            print(f"[ERROR] no experiments under {task_folder}")
            return
        print(f"[INFO] task_folder = {task_folder}")
        print(f"[INFO] found {len(exp_names)} experiments")
        initial_idx = 0
        if data_folder is not None and data_folder.parent == task_folder:
            if data_folder.name in exp_names:
                initial_idx = exp_names.index(data_folder.name)
        data_folder = task_folder / exp_names[initial_idx]
    else:
        exp_names = [data_folder.name]
        task_folder = data_folder.parent

    multi_exp = len(exp_names) > 1

    # MANO layers
    from manopth.manolayer import ManoLayer
    from hocap_annotation.utils import CFG
    mano_layer_right = ManoLayer(side="right", mano_root=CFG.mano.model_path,
                                  use_pca=False, ncomps=45).cuda()
    mano_layer_left = ManoLayer(side="left", mano_root=CFG.mano.model_path,
                                 use_pca=False, ncomps=45).cuda()
    faces_left = mano_layer_left.th_faces.detach().cpu().numpy()
    faces_right = mano_layer_right.th_faces.detach().cpu().numpy()

    # ---- Per-experiment loader ----
    def load_experiment(exp_name):
        if pkl_path is not None and exp_name == pkl_path.parent.name:
            pkl = pkl_path
            annotated = pkl.parent
            exp_folder = data_folder if data_folder is not None else annotated
        else:
            exp_folder = task_folder / exp_name
            annotated = annotated_folder_for(exp_folder)
            pkl = pick_hand_pkl(annotated, prefer=args.prefer)
            if pkl is None:
                raise FileNotFoundError(f"no result*.pkl in {annotated}")

        meta_path = exp_folder / "meta.yaml"
        cam_transforms = load_camera_transforms(meta_path)

        hand = load_hand_pkl(pkl)
        num_frames_hand = _infer_num_frames_hand(hand)

        # Object pose sources
        sources = discover_object_pose_sources(annotated)
        # Optionally filter to user-forced source
        if args.pose_source is not None:
            sources = [s for s in sources if s[0] == args.pose_source]

        # Object meshes (from meta.yaml). If meta is missing, no objects.
        obj_meshes = load_object_meshes(meta_path)

        return {
            "exp_name": exp_name,
            "exp_folder": exp_folder,
            "annotated": annotated,
            "pkl_path": pkl,
            "label": str(pkl.name),
            "hand": hand,
            "num_frames_hand": num_frames_hand,
            "cam_transforms": cam_transforms,
            "obj_pose_sources": sources,        # [(name, path), ...]
            "obj_meshes": obj_meshes,            # [(obj_id, V, F), ...]
        }

    def load_obj_poses_for(state, source_name):
        """Look up the .npy for the given source_name and return (O, N, 7)."""
        for name, p in state["obj_pose_sources"]:
            if name == source_name:
                try:
                    return load_object_poses(p)
                except Exception as e:
                    print(f"[WARN] failed to load {p}: {e}")
                    return None
        return None

    state = load_experiment(exp_names[0])
    print(f"[INFO] exp={state['exp_name']}  pkl={state['pkl_path']}  hand_frames={state['num_frames_hand']}")
    print(f"[INFO] obj meshes: {[m[0] for m in state['obj_meshes']]}")
    print(f"[INFO] obj pose sources: {[s[0] for s in state['obj_pose_sources']]}")

    # Initial obj source + poses
    initial_src = state["obj_pose_sources"][0][0] if state["obj_pose_sources"] else None
    obj_poses = load_obj_poses_for(state, initial_src) if initial_src else None
    num_frames_obj = obj_poses.shape[1] if obj_poses is not None else 0
    num_frames = max(state["num_frames_hand"], num_frames_obj)
    print(f"[INFO] num_frames (max of hand/obj) = {num_frames}")

    # ---- Viser server ----
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"[INFO] Viser at http://localhost:{args.port}")

    # GUI: experiment nav
    if multi_exp:
        server.gui.add_markdown("### Experiment")
        exp_dropdown = server.gui.add_dropdown("Experiment", options=exp_names,
                                                initial_value=exp_names[0])
        with server.gui.add_folder("Quick Nav", expand_by_default=True):
            btn_prev = server.gui.add_button("< Prev")
            btn_next = server.gui.add_button("Next >")
            exp_status = server.gui.add_markdown(f"**1 / {len(exp_names)}**")
    server.gui.add_markdown("---")

    info_md = server.gui.add_markdown(
        f"`{state['label']}`  \nhand frames: {state['num_frames_hand']}  obj frames: {num_frames_obj}"
    )

    frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, num_frames - 1),
                                          step=1, initial_value=0)
    show_left = server.gui.add_checkbox("Show Left Hand", initial_value=True)
    show_right = server.gui.add_checkbox("Show Right Hand", initial_value=True)
    show_objects = server.gui.add_checkbox("Show Objects", initial_value=True)
    show_cams = server.gui.add_checkbox("Show Cameras", initial_value=False)
    show_axes = server.gui.add_checkbox("Show World Axes", initial_value=True)

    # Object pose source dropdown (only if we have any sources)
    src_options = [s[0] for s in state["obj_pose_sources"]] if state["obj_pose_sources"] else ["<none>"]
    src_dropdown = server.gui.add_dropdown(
        "Obj pose source", options=src_options,
        initial_value=src_options[0],
    )

    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)

    # Colors (matches visualize_hand_viser palette)
    left_color = (100, 149, 237)
    right_color = (255, 160, 122)
    obj_palette = [
        (180, 180, 180), (210, 140, 140), (140, 180, 210),
        (160, 200, 140), (200, 180, 130), (190, 150, 200),
    ]

    axes_handle = server.scene.add_frame("/world_axes", axes_length=0.1, axes_radius=0.002)

    # Cameras
    cam_handles = []

    def setup_cameras(Ts):
        nonlocal cam_handles
        for h in cam_handles:
            h.remove()
        cam_handles = []
        for i, T in enumerate(Ts):
            Rx = np.diag([1.0, -1.0, -1.0])
            T_flip = T.copy()
            T_flip[:3, :3] = Rx @ T[:3, :3]
            T_flip[:3, 3] = Rx @ T[:3, 3]
            wxyz = R.from_matrix(T_flip[:3, :3]).as_quat(scalar_first=True)
            h = server.scene.add_frame(f"/cameras/cam_{i:02d}",
                                         wxyz=wxyz, position=T_flip[:3, 3],
                                         axes_length=0.05, axes_radius=0.001,
                                         visible=show_cams.value)
            cam_handles.append(h)

    setup_cameras(state["cam_transforms"])

    # Mesh handles: hands + per-object
    handles = {"left": None, "right": None}
    obj_handles = {}  # idx -> handle

    def remove_handle(key):
        if handles[key] is not None:
            handles[key].remove()
            handles[key] = None

    def remove_all_obj_handles():
        for k, h in list(obj_handles.items()):
            if h is not None:
                h.remove()
        obj_handles.clear()

    def render_frame(fi):
        # Hands
        if show_left.value:
            lv = reconstruct_left_verts(state["hand"], fi, mano_layer_right)
            remove_handle("left")
            if lv is not None:
                handles["left"] = server.scene.add_mesh_simple(
                    "/left_hand",
                    vertices=flip_z(lv).astype(np.float32),
                    faces=faces_left,
                    color=left_color,
                )
        else:
            remove_handle("left")

        if show_right.value:
            rv = reconstruct_right_verts(state["hand"], fi, mano_layer_right)
            remove_handle("right")
            if rv is not None:
                handles["right"] = server.scene.add_mesh_simple(
                    "/right_hand",
                    vertices=flip_z(rv).astype(np.float32),
                    faces=faces_right,
                    color=right_color,
                )
        else:
            remove_handle("right")

        # Objects (one mesh per object_id, posed by current source's poses_o)
        if show_objects.value and obj_poses is not None and state["obj_meshes"]:
            num_obj = min(obj_poses.shape[0], len(state["obj_meshes"]))
            for oi in range(num_obj):
                obj_id, V, F = state["obj_meshes"][oi]
                if fi >= obj_poses.shape[1]:
                    if oi in obj_handles and obj_handles[oi] is not None:
                        obj_handles[oi].remove()
                        obj_handles[oi] = None
                    continue
                pose = obj_poses[oi, fi]
                # Skip placeholder (-1 fills) frames.
                if np.all(pose == -1):
                    if oi in obj_handles and obj_handles[oi] is not None:
                        obj_handles[oi].remove()
                        obj_handles[oi] = None
                    continue
                world_v = transform_object_vertices(V, pose)
                world_v = flip_z(world_v).astype(np.float32)
                color = obj_palette[oi % len(obj_palette)]
                if oi in obj_handles and obj_handles[oi] is not None:
                    obj_handles[oi].remove()
                obj_handles[oi] = server.scene.add_mesh_simple(
                    f"/objects/{oi:02d}_{obj_id}",
                    vertices=world_v,
                    faces=F,
                    color=color,
                )
        else:
            remove_all_obj_handles()

    def switch_source(src_name):
        nonlocal obj_poses, num_frames_obj, num_frames
        new_poses = load_obj_poses_for(state, src_name) if src_name in [s[0] for s in state["obj_pose_sources"]] else None
        obj_poses = new_poses
        num_frames_obj = obj_poses.shape[1] if obj_poses is not None else 0
        num_frames = max(state["num_frames_hand"], num_frames_obj)
        frame_slider.max = max(0, num_frames - 1)
        info_md.content = (
            f"`{state['label']}`  \nhand frames: {state['num_frames_hand']}  "
            f"obj frames: {num_frames_obj}  src: `{src_name}`"
        )
        render_frame(int(frame_slider.value))

    def switch_experiment(name):
        nonlocal state, obj_poses, num_frames_obj, num_frames
        try:
            state = load_experiment(name)
        except Exception as e:
            print(f"[ERROR] load {name}: {e}")
            return
        print(f"[INFO] switched -> {name}  hand_frames={state['num_frames_hand']}")
        # Clear scene
        for k in list(handles.keys()):
            remove_handle(k)
        remove_all_obj_handles()

        # Refresh obj source dropdown options
        new_src_options = [s[0] for s in state["obj_pose_sources"]] if state["obj_pose_sources"] else ["<none>"]
        src_dropdown.options = new_src_options
        src_dropdown.value = new_src_options[0]
        obj_poses = load_obj_poses_for(state, new_src_options[0]) if state["obj_pose_sources"] else None
        num_frames_obj = obj_poses.shape[1] if obj_poses is not None else 0
        num_frames = max(state["num_frames_hand"], num_frames_obj)

        frame_slider.max = max(0, num_frames - 1)
        frame_slider.value = 0
        info_md.content = (
            f"`{state['label']}`  \nhand frames: {state['num_frames_hand']}  "
            f"obj frames: {num_frames_obj}  src: `{src_dropdown.value}`"
        )
        setup_cameras(state["cam_transforms"])
        if multi_exp:
            idx = exp_names.index(name)
            exp_status.content = f"**{idx + 1} / {len(exp_names)}**"
        render_frame(0)

    render_frame(0)

    # Wire callbacks
    if multi_exp:
        @exp_dropdown.on_update
        def _(event):  # noqa: F841
            switch_experiment(exp_dropdown.value)

        @btn_prev.on_click
        def _(event):  # noqa: F841
            idx = exp_names.index(exp_dropdown.value)
            if idx > 0:
                exp_dropdown.value = exp_names[idx - 1]
                switch_experiment(exp_names[idx - 1])

        @btn_next.on_click
        def _(event):  # noqa: F841
            idx = exp_names.index(exp_dropdown.value)
            if idx < len(exp_names) - 1:
                exp_dropdown.value = exp_names[idx + 1]
                switch_experiment(exp_names[idx + 1])

    @frame_slider.on_update
    def _(event):  # noqa: F841
        render_frame(int(frame_slider.value))

    @show_left.on_update
    def _(event):  # noqa: F841
        render_frame(int(frame_slider.value))

    @show_right.on_update
    def _(event):  # noqa: F841
        render_frame(int(frame_slider.value))

    @show_objects.on_update
    def _(event):  # noqa: F841
        render_frame(int(frame_slider.value))

    @show_cams.on_update
    def _(event):  # noqa: F841
        for h in cam_handles:
            h.visible = show_cams.value

    @show_axes.on_update
    def _(event):  # noqa: F841
        axes_handle.visible = show_axes.value

    @src_dropdown.on_update
    def _(event):  # noqa: F841
        switch_source(src_dropdown.value)

    try:
        while True:
            if playing.value and num_frames > 0:
                nxt = (int(frame_slider.value) + 1) % num_frames
                frame_slider.value = nxt
                render_frame(nxt)
                time.sleep(1.0 / max(1, int(fps_slider.value)))
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] stopped")


if __name__ == "__main__":
    main()
