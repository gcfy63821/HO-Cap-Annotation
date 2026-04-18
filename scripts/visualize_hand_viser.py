#!/usr/bin/env python3
"""
Hand-only interactive 3D viewer for HO-Cap-Annotation outputs (Viser).

Loads merged hand annotations from:
    <data_root>_annotated/<task>/<exp>/result_hand_optimized.pkl   (preferred)
    <data_root>_annotated/<task>/<exp>/result.pkl                  (fallback)

Unlike DataCollection/visualize/viser_viewer.py this viewer is hand-only
(no object mesh, no object pose solver outputs) and is aware of the chunked
batch output: frame count is taken from the pickle, not meta.yaml, so it
shows the full merged range.

Usage:
    # Single experiment:
    conda activate hocap-annotation
    python scripts/visualize_hand_viser.py \
        --data_folder data/videos_0101/mallet_crush_nuts/20260104_largeplate_..._1

    # Browse every experiment under a task folder:
    python scripts/visualize_hand_viser.py \
        --task_folder data/videos_0101/mallet_crush_nuts

    # Point at an already-merged or chunked pkl directly:
    python scripts/visualize_hand_viser.py --pkl_path /abs/path/to/result_hand_optimized.pkl
"""

import sys
import os
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
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


def load_hand_pkl(pkl_path: Path):
    """Return a normalized hand-data dict from result*.pkl."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    hp = data["hand_pose"]

    def _as_np(x):
        if x is None:
            return np.zeros(0)
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, list):
            # list of tensors/arrays: stack into (N, ...)
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
        "num_frames": int(data.get("num_frames", 0)),
        "start_frame": data.get("start_frame", 0),
    }
    # pose can land as (N,1,48) if chunks were list-of-(1,48); squeeze that.
    for k in ["left_hand_pose", "right_hand_pose"]:
        if result[k].ndim == 3 and result[k].shape[1] == 1:
            result[k] = result[k][:, 0, :]
    for k in ["left_hand_translation", "right_hand_translation"]:
        if result[k].ndim == 3 and result[k].shape[1] == 1:
            result[k] = result[k][:, 0, :]
    return result


def _infer_num_frames(hand):
    for key in ["right_hand_pose", "left_hand_pose",
                "right_hand_translation", "left_hand_translation"]:
        arr = hand.get(key)
        if arr is not None and getattr(arr, "shape", (0,))[0] > 0:
            return int(arr.shape[0])
    return int(hand.get("num_frames", 0))


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


def flip_z(v):
    v = v.copy()
    v[..., 1] *= -1
    v[..., 2] *= -1
    return v


def annotated_folder_for(data_folder: Path) -> Path:
    """data/<videos_XXXX>/<task>/<exp> -> data/<videos_XXXX>_annotated/<task>/<exp>."""
    exp_name = data_folder.name
    task_name = data_folder.parent.name
    data_root_name = data_folder.parent.parent.name
    base_dir = data_folder.parent.parent.parent
    return base_dir / f"{data_root_name}_annotated" / task_name / exp_name


def pick_pkl(annotated: Path):
    """Prefer optimized, fall back to reconstruct-only."""
    for name in ["result_hand_optimized.pkl", "result.pkl"]:
        p = annotated / name
        if p.exists():
            return p
    return None


def discover_experiments(task_folder: Path):
    exps = []
    for sub in sorted(task_folder.iterdir()):
        if not sub.is_dir():
            continue
        # Accept experiment if either meta.yaml exists (raw side) or
        # an annotated result pkl exists.
        if (sub / "meta.yaml").exists():
            exps.append(sub.name)
            continue
        ann = annotated_folder_for(sub)
        if pick_pkl(ann) is not None:
            exps.append(sub.name)
    return exps


def load_camera_transforms(meta_path: Path):
    """Return list of 4x4 cam extrinsics, or [] if unavailable."""
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


def main():
    ap = argparse.ArgumentParser(description="Hand-only Viser viewer")
    ap.add_argument("--data_folder", type=str, default="")
    ap.add_argument("--task_folder", type=str, default="")
    ap.add_argument("--pkl_path", type=str, default="",
                    help="Visualize a specific pkl (bypasses data/task_folder).")
    ap.add_argument("--prefer", type=str, default="optimized",
                    choices=["optimized", "reconstruct"],
                    help="Prefer result_hand_optimized.pkl (default) or result.pkl.")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    task_folder = _TASK_FOLDER
    data_folder = _DATA_FOLDER
    pkl_path = _PKL_PATH

    if not (task_folder or data_folder or pkl_path):
        ap.error("Must specify one of --data_folder / --task_folder / --pkl_path")

    # Build experiment list
    if pkl_path is not None:
        exp_names = [pkl_path.parent.name]
        task_folder = None
        data_folder = None
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

    # MANO
    from manopth.manolayer import ManoLayer
    from hocap_annotation.utils import CFG
    mano_layer_right = ManoLayer(side="right", mano_root=CFG.mano.model_path,
                                  use_pca=False, ncomps=45).cuda()
    mano_layer_left = ManoLayer(side="left", mano_root=CFG.mano.model_path,
                                 use_pca=False, ncomps=45).cuda()
    faces_left = mano_layer_left.th_faces.detach().cpu().numpy()
    faces_right = mano_layer_right.th_faces.detach().cpu().numpy()

    def load_experiment(exp_name):
        """Resolve pkl and camera info for this experiment."""
        if pkl_path is not None:
            pkl = pkl_path
            # try to find a sibling meta.yaml in the raw data folder (unknown here)
            meta_path = None
            cam_transforms = []
            label = f"pkl:{pkl.name}"
            exp_folder = pkl.parent
        else:
            exp_folder = task_folder / exp_name
            annotated = annotated_folder_for(exp_folder)
            if args.prefer == "reconstruct":
                pkl = annotated / "result.pkl"
                if not pkl.exists():
                    pkl = annotated / "result_hand_optimized.pkl"
            else:
                pkl = pick_pkl(annotated)
            if pkl is None:
                raise FileNotFoundError(
                    f"no result*.pkl in {annotated}"
                )
            meta_path = exp_folder / "meta.yaml"
            cam_transforms = load_camera_transforms(meta_path)
            label = str(pkl.relative_to(annotated.parent.parent))

        hand = load_hand_pkl(pkl)
        num_frames = _infer_num_frames(hand)
        return {
            "exp_name": exp_name,
            "exp_folder": exp_folder,
            "pkl_path": pkl,
            "label": label,
            "hand": hand,
            "num_frames": num_frames,
            "cam_transforms": cam_transforms,
        }

    state = load_experiment(exp_names[0] if pkl_path is None else exp_names[0])
    print(f"[INFO] exp={state['exp_name']}  pkl={state['pkl_path']}  frames={state['num_frames']}")

    # Viser server
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"[INFO] Viser at http://localhost:{args.port}")

    if multi_exp:
        server.gui.add_markdown("### Experiment")
        exp_dropdown = server.gui.add_dropdown("Experiment", options=exp_names,
                                                initial_value=exp_names[0])
        with server.gui.add_folder("Quick Nav", expand_by_default=True):
            btn_prev = server.gui.add_button("< Prev")
            btn_next = server.gui.add_button("Next >")
            exp_status = server.gui.add_markdown(f"**1 / {len(exp_names)}**")
    server.gui.add_markdown("---")

    info_md = server.gui.add_markdown(f"`{state['label']}`  \nframes: {state['num_frames']}")

    frame_slider = server.gui.add_slider("Frame", min=0,
                                          max=max(0, state["num_frames"] - 1),
                                          step=1, initial_value=0)
    show_left = server.gui.add_checkbox("Show Left Hand", initial_value=True)
    show_right = server.gui.add_checkbox("Show Right Hand", initial_value=True)
    show_cams = server.gui.add_checkbox("Show Cameras", initial_value=False)
    show_axes = server.gui.add_checkbox("Show World Axes", initial_value=True)
    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)

    left_color = (100, 149, 237)
    right_color = (255, 160, 122)

    axes_handle = server.scene.add_frame("/world_axes", axes_length=0.1, axes_radius=0.002)

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

    handles = {"left": None, "right": None}

    def remove(key):
        if handles[key] is not None:
            handles[key].remove()
            handles[key] = None

    def render_frame(fi):
        if show_left.value:
            lv = reconstruct_left_verts(state["hand"], fi, mano_layer_right)
            remove("left")
            if lv is not None:
                handles["left"] = server.scene.add_mesh_simple(
                    "/left_hand",
                    vertices=flip_z(lv).astype(np.float32),
                    faces=faces_left,
                    color=left_color,
                )
        else:
            remove("left")

        if show_right.value:
            rv = reconstruct_right_verts(state["hand"], fi, mano_layer_right)
            remove("right")
            if rv is not None:
                handles["right"] = server.scene.add_mesh_simple(
                    "/right_hand",
                    vertices=flip_z(rv).astype(np.float32),
                    faces=faces_right,
                    color=right_color,
                )
        else:
            remove("right")

    def switch_experiment(name):
        nonlocal state
        try:
            state = load_experiment(name)
        except Exception as e:
            print(f"[ERROR] load {name}: {e}")
            return
        print(f"[INFO] switched -> {name}  frames={state['num_frames']}")
        for k in list(handles.keys()):
            remove(k)
        frame_slider.max = max(0, state["num_frames"] - 1)
        frame_slider.value = 0
        info_md.content = f"`{state['label']}`  \nframes: {state['num_frames']}"
        setup_cameras(state["cam_transforms"])
        if multi_exp:
            idx = exp_names.index(name)
            exp_status.content = f"**{idx + 1} / {len(exp_names)}**"
        render_frame(0)

    render_frame(0)

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

    @show_cams.on_update
    def _(event):  # noqa: F841
        for h in cam_handles:
            h.visible = show_cams.value

    @show_axes.on_update
    def _(event):  # noqa: F841
        axes_handle.visible = show_axes.value

    import time
    try:
        while True:
            if playing.value and state["num_frames"] > 0:
                nxt = (int(frame_slider.value) + 1) % state["num_frames"]
                frame_slider.value = nxt
                render_frame(nxt)
                time.sleep(1.0 / max(1, int(fps_slider.value)))
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] stopped")


if __name__ == "__main__":
    main()
