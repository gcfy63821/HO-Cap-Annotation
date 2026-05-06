"""Render hand (MANO) + tool meshes onto every cam view and write an mp4.

Mirrors diagnose_pose_offset.py's workspace-mode auto-discovery: walk
<videos_root>/<task>/<exp>, find every exp whose annotated side has both
the merged object pose (Stage 4) and ideally hand outputs (Stage 5/7),
and produce one 2x4-grid mp4 per exp at
    <annotated>/<exp>/debug/hand_object_video/<jobname>.mp4

Per frame, per camera tile we:
    1. Read the RGB frame from data00000000.h5 (fallback: cam{i}_rgb.mp4).
    2. Pose the tool mesh by world pose (from joint_pose_solver/poses_o.npy
        if it exists, else fd_pose_solver/fd_poses_merged_fixed.npy),
        transform to cam frame, project + draw faces.
    3. If result_hand_optimized.pkl + processed/joint_pose_solver/poses_m.npy
        exist, rebuild left/right MANO meshes and project them too.
    4. Concatenate the 8 cam tiles into a 2x4 grid frame, append to mp4.

Two modes (mirror diagnose_pose_offset.py):

  (a) Workspace mode (recommended):
      python debug/visualize_hand_object_video.py \\
        --videos_root /viscam/projects/robotool/data/videos_0102 \\
        [--task_filter task_a task_b] \\
        [--frame_step 2]   [--end_frame 800]
      Auto-discovers every exp where the annotated side has the merged
      object pose, and writes one mp4 per exp.

  (b) Single-exp mode (legacy):
      python debug/visualize_hand_object_video.py \\
        --annotated_sequence /viscam/.../videos_0102_annotated/<task>/<exp>
"""

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation as R

HOCAP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HOCAP_ROOT))


# ============================================================
# IO + path helpers (lifted from diagnose_pose_offset.py)
# ============================================================

def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def derive_original_sequence(annotated_sequence: Path) -> Path:
    parts = annotated_sequence.parts
    idx = None
    for i, p in enumerate(parts):
        if p.endswith("_annotated"):
            idx = i; break
    if idx is None:
        raise ValueError(f"Cannot derive original sequence from {annotated_sequence}")
    new_parts = list(parts)
    new_parts[idx] = parts[idx].replace("_annotated", "", 1)
    return Path(*new_parts)


def derive_annotated_for(videos_root: Path, task: str, exp: str) -> Path:
    return videos_root.parent / f"{videos_root.name}_annotated" / task / exp


def find_h5_with_rgbd(sequence_folder: Path) -> Path:
    candidates = sorted(sequence_folder.glob("*.h5")) + sorted(sequence_folder.glob("*.hdf5"))
    for path in candidates:
        try:
            with h5py.File(path, "r") as f:
                names = []
                f.visit(names.append)
            if any(name.endswith("imgs") for name in names):
                return path
        except OSError:
            continue
    raise FileNotFoundError(f"No HDF5 with imgs found in {sequence_folder}")


class VideoRGBSource:
    """Fallback when there's no permanent h5: stream cam*_rgb.mp4 frame by
    frame. Slower (per-frame seek) but works without rebuilding the h5."""

    def __init__(self, sequence_folder: Path, n_cams: int) -> None:
        self.caps = []
        rgb_videos = sorted(sequence_folder.glob("cam*_rgb.mp4"))[:n_cams]
        if not rgb_videos:
            raise FileNotFoundError(f"no cam*_rgb.mp4 under {sequence_folder}")
        self.frame_count = None
        for p in rgb_videos:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise FileNotFoundError(f"failed to open {p}")
            self.caps.append(cap)
            if self.frame_count is None:
                self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def close(self):
        for cap in self.caps: cap.release()

    def get_color(self, frame_idx: int, cam_idx: int) -> np.ndarray:
        cap = self.caps[cam_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            raise ValueError(f"failed to read frame {frame_idx} cam {cam_idx}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def find_dataset(f: h5py.File, target_name: str):
    hit = None
    def visitor(name, obj):
        nonlocal hit
        if hit is not None: return
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == target_name:
            hit = obj
    f.visititems(visitor)
    if hit is None:
        raise KeyError(f"Dataset '{target_name}' not found in {f.filename}")
    return hit


def find_mesh_path(models_folder: Path, object_id: str) -> Path:
    obj_dir = models_folder / object_id
    for cand in ("textured_mesh.obj", "cleaned_mesh_10000.obj", "mesh.obj"):
        path = obj_dir / cand
        if path.exists(): return path
    raise FileNotFoundError(f"No mesh found under {obj_dir}")


TOOL_NAME_MAP_JSON_DEFAULT = Path("/viscam/u/chenrq/crq_ws/scripts/mesh_name_mapping.json")


def resolve_tool_from_name_map(json_path: Path, task: str, exp: str,
                                  models_folder: Path):
    if not json_path or not json_path.is_file():
        return None, None, None
    try:
        mp = json.loads(json_path.read_text())
    except Exception:
        return None, None, None
    hay = f"{task}/{exp}".lower()
    hits = [(k, v) for k, v in mp.items() if k.lower() in hay]
    if not hits: return None, None, None
    hits.sort(key=lambda kv: -len(kv[0]))
    key, name = hits[0]
    for cand in ("textured_mesh.obj", "cleaned_mesh_10000.obj", "mesh.obj"):
        p = models_folder / name / cand
        if p.exists(): return name, p, key
    return name, None, key


# ============================================================
# Object pose loading
# ============================================================

def load_object_world_poses(annotated_sequence: Path, n_frames: int,
                              object_idx: int = 0,
                              prefer: str = "auto") -> tuple:
    """Returns (poses_world (N, 4, 4), source_label).
    prefer ∈ {auto, optimized, fd}. auto: optimized > fd."""
    fd_root = annotated_sequence / "processed" / "fd_pose_solver"
    js_root = annotated_sequence / "processed" / "joint_pose_solver"
    candidates = []
    if prefer in ("auto", "optimized"):
        candidates.append(("optimized", js_root / "poses_o.npy"))
    if prefer in ("auto", "fd"):
        candidates.append(("fd",        fd_root / "fd_poses_merged_fixed.npy"))
    last_err = None
    for label, path in candidates:
        if not path.exists(): continue
        try:
            arr = np.load(path).astype(np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 7:
                arr = arr[object_idx]
            elif arr.ndim == 4 and arr.shape[-2:] == (4, 4):
                arr = arr[object_idx]
            out = np.full((n_frames, 4, 4), np.nan, dtype=np.float32)
            f_use = min(n_frames, arr.shape[0])
            if arr.ndim == 2 and arr.shape[-1] == 7:
                for i in range(f_use):
                    qx, qy, qz, qw, tx, ty, tz = arr[i]
                    T = np.eye(4, dtype=np.float32)
                    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    T[:3, 3] = [tx, ty, tz]
                    out[i] = T
            elif arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                out[:f_use] = arr[:f_use]
            elif arr.ndim == 2 and arr.shape[-1] == 16:
                out[:f_use] = arr[:f_use].reshape(-1, 4, 4)
            else:
                raise ValueError(f"unsupported pose shape {arr.shape} in {path}")
            return out, f"{label}:{path.name}"
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError(f"no usable object pose file for {annotated_sequence}: {last_err}")


# ============================================================
# Hand (MANO) loading + reconstruction
# ============================================================

def _as_np(x):
    """Robust array converter — handles None / list of tensors / tensor / np.
    Mirrors scripts/visualize_hand_viser.py:_as_np so the (N,1,48) chunk-list
    case from result_hand_optimized.pkl is squeezed correctly."""
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


def try_load_hand_data(annotated_sequence: Path):
    """Load hand state from result_hand_optimized.pkl (or result.pkl).
    Returns a normalized dict (same schema as visualize_hand_viser.py's
    load_hand_pkl), or None if no usable pkl is found.

    Note: pose comes from the pkl's left_hand_pose / right_hand_pose fields,
    NOT from processed/joint_pose_solver/poses_m.npy. This matches what
    visualize_hand_viser uses and avoids the 'pose source mismatch' that
    can put hand verts in a different frame than the optimised translation
    + base_rot expect."""
    import pickle
    pkl_path = None
    for cand in ("result_hand_optimized.pkl", "result.pkl"):
        p = annotated_sequence / cand
        if p.exists():
            pkl_path = p
            break
    if pkl_path is None:
        return None
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        hp = data["hand_pose"]
        result = {
            "left_hand_pose":         _as_np(hp.get("left_hand_pose", [])),
            "left_hand_beta":         np.asarray(hp.get("left_hand_beta", [])).squeeze(),
            "left_hand_translation":  _as_np(hp.get("left_hand_translation", [])),
            "left_hand_base_rot":     np.asarray(hp.get("left_hand_base_rot", []))
                if hp.get("left_hand_base_rot") is not None else np.eye(3),
            "right_hand_pose":        _as_np(hp.get("right_hand_pose", [])),
            "right_hand_beta":        np.asarray(hp.get("right_hand_beta", [])).squeeze(),
            "right_hand_translation": _as_np(hp.get("right_hand_translation", [])),
            "num_frames":             int(data.get("num_frames", 0)),
            "start_frame":            data.get("start_frame", 0),
            "_pkl_path":              str(pkl_path),
        }
        # pose can land as (N,1,48) when chunks were list-of-(1,48); squeeze.
        for k in ("left_hand_pose", "right_hand_pose",
                  "left_hand_translation", "right_hand_translation"):
            arr = result[k]
            if arr.ndim == 3 and arr.shape[1] == 1:
                result[k] = arr[:, 0, :]
        return result
    except Exception as e:
        print(f"[WARN] hand load failed ({pkl_path}): {e}")
        return None


def infer_hand_num_frames(hand: dict) -> int:
    for key in ("right_hand_pose", "left_hand_pose",
                "right_hand_translation", "left_hand_translation"):
        arr = hand.get(key)
        if arr is not None and getattr(arr, "shape", (0,))[0] > 0:
            return int(arr.shape[0])
    return int(hand.get("num_frames", 0))


class ManoBundle:
    """Right MANO layer drives BOTH hands — same convention as
    visualize_hand_viser.py. The left hand mesh is produced by:
        verts, joints = mano_layer_right(pose, beta)
        verts = (verts/1000) - root_joint
        verts[:,0] *= -1            # mirror to left
        verts = verts @ base_rot.T  # left-hand-specific orientation
        verts += translation
    Right hand uses the same right layer with no mirror / no base_rot.

    Faces: also use right_layer.th_faces for both — left mesh is just the
    same triangulation with flipped X, so triangle winding is consistent
    enough for visualisation."""

    def __init__(self):
        import torch
        from manopth.manolayer import ManoLayer
        from hocap_annotation.utils import CFG
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_right = ManoLayer(
            side="right", mano_root=CFG.mano.model_path,
            use_pca=False, ncomps=45,
        ).to(self.device)
        self.faces = self.layer_right.th_faces.detach().cpu().numpy()
        # Aliases kept for compatibility with the older two-layer call sites.
        self.faces_left = self.faces
        self.faces_right = self.faces


def reconstruct_left_verts(hand: dict, frame_idx: int, mano_bundle: ManoBundle):
    torch = mano_bundle.torch
    pose_arr = hand["left_hand_pose"]
    if pose_arr.ndim < 2 or frame_idx >= pose_arr.shape[0]:
        return None
    try:
        device = mano_bundle.device
        pose  = torch.tensor(pose_arr[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
        trans = torch.tensor(hand["left_hand_translation"][frame_idx],
                                dtype=torch.float32, device=device).unsqueeze(0)
        beta  = torch.tensor(hand["left_hand_beta"], dtype=torch.float32, device=device)
        if beta.ndim == 1:
            beta = beta.unsqueeze(0)

        base = hand["left_hand_base_rot"]
        if isinstance(base, np.ndarray) and base.ndim == 3 and base.shape[0] > 0:
            ri = min(frame_idx, base.shape[0] - 1)
            base_rot = torch.tensor(base[ri], dtype=torch.float32, device=device)
        elif isinstance(base, np.ndarray) and base.ndim == 2 and base.shape == (3, 3):
            base_rot = torch.tensor(base, dtype=torch.float32, device=device)
        else:
            base_rot = torch.eye(3, dtype=torch.float32, device=device)

        verts, joints = mano_bundle.layer_right(pose, beta)
        verts = verts[0] / 1000.0
        joints = joints[0] / 1000.0
        root = joints[0].clone()
        verts = verts - root
        verts[:, 0] *= -1
        verts = verts @ base_rot.T
        verts = verts + trans
        return verts.detach().cpu().numpy().astype(np.float32)
    except Exception as e:
        if frame_idx < 3:
            print(f"[WARN] left hand frame {frame_idx}: {e}")
        return None


def reconstruct_right_verts(hand: dict, frame_idx: int, mano_bundle: ManoBundle):
    torch = mano_bundle.torch
    pose_arr = hand["right_hand_pose"]
    if pose_arr.ndim < 2 or frame_idx >= pose_arr.shape[0]:
        return None
    try:
        device = mano_bundle.device
        pose  = torch.tensor(pose_arr[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
        trans = torch.tensor(hand["right_hand_translation"][frame_idx],
                                dtype=torch.float32, device=device).unsqueeze(0)
        beta  = torch.tensor(hand["right_hand_beta"], dtype=torch.float32, device=device)
        if beta.ndim == 1:
            beta = beta.unsqueeze(0)

        verts, joints = mano_bundle.layer_right(pose, beta)
        verts = verts[0] / 1000.0
        joints = joints[0] / 1000.0
        root = joints[0].clone()
        verts = verts - root
        verts = verts + trans
        return verts.detach().cpu().numpy().astype(np.float32)
    except Exception as e:
        if frame_idx < 3:
            print(f"[WARN] right hand frame {frame_idx}: {e}")
        return None


# ============================================================
# Drawing helpers
# ============================================================

def project_points(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = points_cam[:, 2:3]
    valid = z[:, 0] > 1e-8
    pts = np.full((points_cam.shape[0], 2), -1.0, dtype=np.float32)
    if np.any(valid):
        proj = points_cam[valid] @ K.T
        pts[valid] = proj[:, :2] / proj[:, 2:3]
    return pts


def project_and_validate(verts_cam: np.ndarray, K: np.ndarray, H: int, W: int):
    """Project verts to 2D, return (pts_int32, valid_bool)."""
    pts = project_points(verts_cam, K)
    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < W)
        & (pts[:, 1] >= 0) & (pts[:, 1] < H)
        & np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
        & (verts_cam[:, 2] > 1e-8)
    )
    return pts.astype(np.int32, copy=False), valid


def paint_fg_mask(mask: np.ndarray, pts_i: np.ndarray, faces: np.ndarray,
                    valid: np.ndarray, every: int = 1):
    """Fill mesh faces (every Nth) onto a single-channel mask (uint8, 0/255)."""
    for face in faces[::every]:
        if valid[face].all():
            cv2.fillConvexPoly(mask, pts_i[face], 255)


def fade_background(img: np.ndarray, fg_mask: np.ndarray,
                       bg_white_alpha: float, dilate_px: int = 2):
    """Blend img toward white where fg_mask == 0. bg_white_alpha=1 → full
    white background; 0 → no change. dilate_px expands the kept-region
    slightly so wireframes drawn ON the boundary aren't washed out."""
    if bg_white_alpha <= 0:
        return img
    if dilate_px > 0:
        k = max(1, int(dilate_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        kept = cv2.dilate(fg_mask, kernel)
    else:
        kept = fg_mask
    bg = np.where(kept[..., None] > 0, 0.0, bg_white_alpha).astype(np.float32)
    white = np.full_like(img, 255)
    blended = (img.astype(np.float32) * (1.0 - bg)
                + white.astype(np.float32) * bg).astype(np.uint8)
    return blended


def draw_mesh(img: np.ndarray, verts_cam: np.ndarray, faces: np.ndarray,
                K: np.ndarray, color: tuple, every: int = 4,
                fill_alpha: float = 0.0,
                fg_mask: np.ndarray = None,
                fg_every: int = 1):
    """Draw a wireframe of every Nth face onto img (RGB). If fill_alpha>0,
    also overlays filled polygons with that alpha. If fg_mask is given,
    accumulate the (more-densely-painted) face fills into it so callers
    can fade the background outside the union of object+hand faces."""
    H, W = img.shape[:2]
    pts_i, valid = project_and_validate(verts_cam, K, H, W)
    if fg_mask is not None:
        paint_fg_mask(fg_mask, pts_i, faces, valid, every=fg_every)
    if fill_alpha > 0:
        layer = img.copy()
        for face in faces[::every]:
            if valid[face].all():
                cv2.fillConvexPoly(layer, pts_i[face], color)
        cv2.addWeighted(layer, fill_alpha, img, 1 - fill_alpha, 0, dst=img)
    for face in faces[::every]:
        if valid[face].all():
            cv2.polylines(img, [pts_i[face]], isClosed=True, color=color,
                            thickness=1, lineType=cv2.LINE_AA)


def parse_color(spec, default):
    """Accept a `#rrggbb` / `rrggbb` hex string or a comma-separated 'r,g,b'.
    Returns an (r, g, b) tuple. Falls back to default on bad input."""
    if not spec:
        return default
    s = spec.strip().lstrip("#")
    try:
        if "," in s:
            r, g, b = (int(x) for x in s.split(",")[:3])
        else:
            if len(s) == 3:
                s = "".join(c * 2 for c in s)
            if len(s) != 6:
                raise ValueError(f"hex must be 6 chars, got {s!r}")
            r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    except Exception as e:
        print(f"[WARN] bad color spec {spec!r} ({e}); using default {default}")
        return default


# Designer-friendly default palette (RGB tuples).
# Object  = amber 400  #fbbf24  (warm gold, pops against most scenes)
# L hand  = pink  200  #fbcfe8  (bright luminous pink — pumped up from
#                                pink-300 so it reads brighter under
#                                pyrender's lit shading)
# R hand  = sky   400  #38bdf8  (cool cyan)
DEFAULT_OBJECT_COLOR     = (251, 191,  36)
DEFAULT_LEFT_HAND_COLOR  = (255, 207, 232)
DEFAULT_RIGHT_HAND_COLOR = ( 56, 189, 248)

# A few named presets; pick with --palette <name>.
PALETTES = {
    "default": (DEFAULT_OBJECT_COLOR, DEFAULT_LEFT_HAND_COLOR, DEFAULT_RIGHT_HAND_COLOR),
    # warm = amber tool, light pink & magenta hands
    "warm":    ((245, 158,  11), (251, 207, 232), (217,  70, 239)),
    # cool = teal tool, cyan & violet hands
    "cool":    (( 20, 184, 166), ( 34, 211, 238), (139,  92, 246)),
    # high-contrast original
    "vivid":   ((  0, 255,   0), (255,  60,  60), ( 60,  60, 255)),
    # pastel
    "pastel":  ((167, 243, 208), (252, 165, 165), (165, 180, 252)),
    # mono (good with bg_white_alpha) — object black, hands red/blue ink
    "ink":     (( 24,  24,  27), (190,  18,  60), ( 30,  64, 175)),
}


def concat_frames_grid(frames, grid=(2, 4)):
    rows = []
    for r in range(grid[0]):
        rows.append(np.concatenate(frames[r * grid[1]:(r + 1) * grid[1]], axis=1))
    return np.concatenate(rows, axis=0)


def open_writer(path: Path, w: int, h: int, fps: int = 20):
    """Try a couple of fourccs in order — H264/avc1 produce broadly
    playable mp4s, mp4v is the safe fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_name in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        w_out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if w_out.isOpened():
            print(f"  [INFO] mp4 writer fourcc={fourcc_name} → {path}")
            return w_out
        w_out.release()
    raise RuntimeError(f"could not open VideoWriter for {path}")


# ============================================================
# DONE detection + discovery
# ============================================================

def is_exp_done(annotated_exp: Path) -> tuple:
    fd_merged = annotated_exp / "processed/fd_pose_solver/fd_poses_merged_fixed.npy"
    js_o = annotated_exp / "processed/joint_pose_solver/poses_o.npy"
    if js_o.exists(): return True, "joint_pose_solver/poses_o.npy"
    if fd_merged.exists(): return True, "fd_poses_merged_fixed.npy"
    return False, "no merged object pose"


def discover_done_exps(videos_root: Path, task_filter=None,
                          only_remasked: bool = False):
    """Walk videos_root and yield exps with merged object pose. When
    `only_remasked` is True, also require <annotated>/<exp>/mask_prompts.json
    so we only render exps that were re-annotated by simple_mask_annotator
    (i.e. ones whose tool_masks/masks.h5 was regenerated)."""
    out, skipped = [], 0
    skipped_no_remask = 0
    for task_dir in sorted(p for p in videos_root.iterdir() if p.is_dir()):
        nm = task_dir.name
        if nm.startswith("realsense_calibrate"): continue
        if nm.startswith("ref_pc"):              continue
        if nm.startswith("posts"):               continue
        if nm == "first_frame":                  continue
        if task_filter and nm not in task_filter: continue
        for exp_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            if not list(exp_dir.glob("cam*_rgb.mp4")): continue
            ann = derive_annotated_for(videos_root, nm, exp_dir.name)
            if only_remasked and not (ann / "mask_prompts.json").exists():
                skipped_no_remask += 1
                continue
            ok, reason = is_exp_done(ann)
            if ok:
                if only_remasked:
                    reason = f"{reason} + mask_prompts.json"
                out.append((nm, exp_dir.name, exp_dir, ann, reason))
            else:
                skipped += 1
    if only_remasked:
        print(f"[discover] only_remasked filter: dropped "
              f"{skipped_no_remask} exps without mask_prompts.json")
    return out, skipped


# ============================================================
# Per-exp render
# ============================================================

def render_one_exp(annotated_sequence: Path,
                     start_frame: int = 0, end_frame: int = -1,
                     frame_step: int = 1,
                     pose_prefer: str = "auto",
                     fps: int = 20,
                     tile_w: int = 640, tile_h: int = 480,
                     tool_name_map_json: Path = None,
                     tool_name_map_disabled: bool = False,
                     object_id: str = None,
                     output_path: Path = None,
                     force: bool = False,
                     no_hand: bool = False,
                     fill_alpha: float = 0.0,
                     mesh_every: int = 4,
                     hand_every: int = 1,
                     bg_white_alpha: float = 0.0,
                     bg_dilate_px: int = 2,
                     object_color: tuple = DEFAULT_OBJECT_COLOR,
                     left_hand_color: tuple = DEFAULT_LEFT_HAND_COLOR,
                     right_hand_color: tuple = DEFAULT_RIGHT_HAND_COLOR,
                     models_folder_override: Path = None,
                     show_deformable: bool = False,
                     deformable_subdir: str = "deformable_masks",
                     deformable_color: tuple = (220, 50, 200),
                     deformable_alpha: float = 0.45) -> dict:
    annotated_sequence = annotated_sequence.resolve()
    original_sequence = derive_original_sequence(annotated_sequence)
    meta_path = original_sequence / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta.yaml: {meta_path}")
    meta = load_yaml(meta_path)

    calib_path = Path(meta["calibration_yaml_path"])
    # --models_folder CLI override takes priority over meta.yaml's
    # 'models_folder' field — useful when meta was generated on another
    # machine / a stale path, or you want to point at a different mesh
    # set (e.g. cleaned vs textured).
    if models_folder_override is not None:
        models_folder = Path(models_folder_override)
        if not models_folder.is_dir():
            print(f"[WARN] --models_folder override not a dir: {models_folder}")
    else:
        models_folder = Path(meta["models_folder"])

    # ---- tool / mesh resolution ----
    if (not tool_name_map_disabled
            and tool_name_map_json is None
            and TOOL_NAME_MAP_JSON_DEFAULT.is_file()):
        tool_name_map_json = TOOL_NAME_MAP_JSON_DEFAULT

    mesh_path = None; mapped_via = None
    if not object_id:
        meta_obj_id = meta["object_ids"][0]
        try:
            mesh_path = find_mesh_path(models_folder, meta_obj_id)
            object_id = meta_obj_id
            mapped_via = "meta.yaml:object_ids[0]"
        except FileNotFoundError:
            mesh_path = None
    if not object_id and tool_name_map_json is not None:
        task = annotated_sequence.parent.name
        exp = annotated_sequence.name
        name, path, key = resolve_tool_from_name_map(
            tool_name_map_json, task, exp, models_folder)
        if name is not None and path is not None:
            object_id = name; mesh_path = path
            mapped_via = f"tool_name_map_json:{key} -> {name}"
    if not object_id:
        object_id = meta["object_ids"][0]
        mapped_via = mapped_via or "meta.yaml:object_ids[0]"
    if mesh_path is None:
        mesh_path = find_mesh_path(models_folder, object_id)

    # ---- output ----
    out_path = output_path if output_path is not None else (
        annotated_sequence / "debug" / "hand_object_video"
        / f"{pose_prefer}_hand_object.mp4"
    )
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists (pass --force)")
        return {"output": str(out_path), "skipped": True}

    # ---- calib + mesh ----
    calib = load_yaml(calib_path)
    serials = [str(cam["camera_id"]).zfill(2) for cam in calib]
    Ks = [np.array(cam["color_intrinsic_matrix"], dtype=np.float32) for cam in calib]
    cam_RTs = [np.array(cam["transformation"], dtype=np.float32) for cam in calib]
    cam_RTs_inv = [np.linalg.inv(T).astype(np.float32) for T in cam_RTs]

    mesh = trimesh.load(mesh_path, process=True, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())
    obj_verts_local = np.asarray(mesh.vertices, dtype=np.float32)
    obj_faces = np.asarray(mesh.faces, dtype=np.int32)

    # ---- RGB source ----
    h5_path = None; video_source = None
    try:
        h5_path = find_h5_with_rgbd(original_sequence)
    except FileNotFoundError:
        video_source = VideoRGBSource(original_sequence, len(serials))

    data_ctx = contextlib.nullcontext(None) if h5_path is None else h5py.File(h5_path, "r")
    with data_ctx as data_f:
        imgs_ds = find_dataset(data_f, "imgs") if data_f else None
        n_frames_total = (int(imgs_ds.shape[0]) if imgs_ds is not None
                            else int(video_source.frame_count))

        if end_frame < 0 or end_frame >= n_frames_total:
            end_frame = n_frames_total - 1
        start_frame = max(0, min(start_frame, end_frame))
        chosen = list(range(start_frame, end_frame + 1, max(1, frame_step)))

        # ---- object world poses ----
        obj_poses_w, pose_src = load_object_world_poses(
            annotated_sequence, n_frames_total, object_idx=0, prefer=pose_prefer)

        # ---- hand setup ----
        # Match scripts/visualize_hand_viser.py: pose comes from
        # result_hand_optimized.pkl (or result.pkl), and the right MANO
        # layer drives both hands.
        hand_data = None; mano = None
        if not no_hand:
            hand_data = try_load_hand_data(annotated_sequence)
            if hand_data is not None:
                try:
                    mano = ManoBundle()
                except Exception as e:
                    print(f"[WARN] could not init MANO ({e}); rendering tool-only")
                    mano = None
                    hand_data = None
            else:
                print(f"[INFO] no result(_hand_optimized).pkl under "
                      f"{annotated_sequence} — rendering tool-only")

        n_cams = len(serials)

        # Optional deformable mask overlay (from batch_deformable_annotator.py)
        deformable_f = None; deformable_ds = None
        if show_deformable:
            p = annotated_sequence / deformable_subdir / "masks.h5"
            if p.exists():
                try:
                    deformable_f = h5py.File(p, "r")
                    deformable_ds = deformable_f.get("masks")
                    if deformable_ds is None or deformable_ds.ndim < 4:
                        deformable_f.close(); deformable_f = None; deformable_ds = None
                    else:
                        print(f"[deformable] using {p}  shape={deformable_ds.shape}")
                except Exception as e:
                    print(f"[deformable] could not open {p}: {e}")
            else:
                print(f"[deformable] no {p} — skipping overlay")

        # tile size: the source frames may be larger; we resize each tile to
        # tile_w/tile_h so the 2x4 grid is exactly (4*tile_w, 2*tile_h).
        grid_w = 4 * tile_w; grid_h = 2 * tile_h
        writer = open_writer(out_path, grid_w, grid_h, fps=fps)

        n_written = 0; t0 = time.time()
        try:
            for fi, frame_idx in enumerate(chosen):
                pose_w = obj_poses_w[frame_idx]
                has_obj = bool(np.all(np.isfinite(pose_w)))
                # Pre-compute hand world verts for this frame, using the
                # pkl-based loaders ported from visualize_hand_viser.py.
                hand_world = {}
                if mano is not None and hand_data is not None:
                    lv = reconstruct_left_verts(hand_data, frame_idx, mano)
                    if lv is not None: hand_world["left"] = lv
                    rv = reconstruct_right_verts(hand_data, frame_idx, mano)
                    if rv is not None: hand_world["right"] = rv

                tiles = []
                for cam_idx in range(n_cams):
                    if imgs_ds is not None:
                        rgb = np.asarray(imgs_ds[frame_idx, cam_idx], dtype=np.uint8)
                    else:
                        rgb = video_source.get_color(frame_idx, cam_idx)

                    # Read deformable mask now; we'll overlay it AFTER
                    # wireframe + bg fade so it sits as a foreground layer.
                    dmask_for_overlay = None
                    if (deformable_ds is not None
                            and frame_idx < deformable_ds.shape[0]
                            and cam_idx < deformable_ds.shape[1]):
                        try:
                            dmask_for_overlay = np.asarray(deformable_ds[frame_idx, cam_idx])
                        except Exception as e:
                            if frame_idx < 3:
                                print(f"[WARN] deformable read frame={frame_idx} "
                                      f"cam={cam_idx}: {e}")

                    img = rgb.copy()
                    H_img, W_img = img.shape[:2]

                    # First pass: paint a binary fg_mask of all object + hand
                    # face fills (denser than the wireframe stride so the
                    # mask covers solid silhouettes, not just edges). Then
                    # fade everything outside that mask toward white. After
                    # that, draw the wireframe on top so contours stay sharp.
                    fg_mask = (np.zeros((H_img, W_img), dtype=np.uint8)
                                if bg_white_alpha > 0 else None)

                    obj_proj = None
                    if has_obj:
                        verts_w = (pose_w[:3, :3] @ obj_verts_local.T).T + pose_w[:3, 3]
                        verts_c = (cam_RTs_inv[cam_idx][:3, :3] @ verts_w.T).T \
                                  + cam_RTs_inv[cam_idx][:3, 3]
                        obj_proj = verts_c
                        if fg_mask is not None:
                            pts_i, valid = project_and_validate(
                                verts_c, Ks[cam_idx], H_img, W_img)
                            paint_fg_mask(fg_mask, pts_i, obj_faces, valid, every=1)

                    hand_proj = {}
                    if hand_world:
                        for side, vw in hand_world.items():
                            vc = (cam_RTs_inv[cam_idx][:3, :3] @ vw.T).T \
                                 + cam_RTs_inv[cam_idx][:3, 3]
                            hand_proj[side] = vc
                            if fg_mask is not None:
                                pts_i, valid = project_and_validate(
                                    vc, Ks[cam_idx], H_img, W_img)
                                paint_fg_mask(fg_mask, pts_i, mano.faces, valid, every=1)

                    if fg_mask is not None:
                        img = fade_background(img, fg_mask, bg_white_alpha,
                                                dilate_px=bg_dilate_px)

                    if obj_proj is not None:
                        draw_mesh(img, obj_proj, obj_faces, Ks[cam_idx],
                                    color=object_color, every=mesh_every,
                                    fill_alpha=fill_alpha)
                    for side, vc in hand_proj.items():
                        color = left_hand_color if side == "left" else right_hand_color
                        draw_mesh(img, vc, mano.faces, Ks[cam_idx],
                                    color=color, every=hand_every,
                                    fill_alpha=fill_alpha)

                    # Foreground deformable overlay — translucent fill +
                    # outline, sits on top of wireframe and bg fade.
                    if dmask_for_overlay is not None:
                        dmask = dmask_for_overlay
                        if dmask.shape != img.shape[:2]:
                            dmask = cv2.resize(dmask.astype(np.uint8),
                                                 (img.shape[1], img.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
                        mb = dmask > 0
                        if mb.any():
                            layer = img.copy()
                            layer[mb] = np.array(deformable_color, dtype=np.uint8)
                            img = cv2.addWeighted(img, 1.0 - deformable_alpha,
                                                    layer, deformable_alpha, 0)
                            cnt, _ = cv2.findContours(
                                mb.astype(np.uint8) * 255,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                            )
                            cv2.drawContours(img, cnt, -1, (255, 255, 0), 2)
                    cv2.putText(img, f"f{frame_idx:06d} cam{serials[cam_idx]}",
                                  (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(img, f"f{frame_idx:06d} cam{serials[cam_idx]}",
                                  (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    if (img.shape[1], img.shape[0]) != (tile_w, tile_h):
                        img = cv2.resize(img, (tile_w, tile_h),
                                          interpolation=cv2.INTER_AREA)
                    tiles.append(img)

                grid = concat_frames_grid(tiles, grid=(2, 4))
                # cv2 wants BGR
                writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                n_written += 1
                if (fi + 1) % 50 == 0:
                    fps_now = (fi + 1) / max(time.time() - t0, 1e-6)
                    print(f"  [{fi+1}/{len(chosen)}] frame {frame_idx}  "
                          f"{fps_now:.1f} fps")
        finally:
            writer.release()
            if video_source is not None: video_source.close()
            if deformable_f is not None:
                try: deformable_f.close()
                except Exception: pass

    summary = {
        "annotated_sequence": str(annotated_sequence),
        "output": str(out_path),
        "tool": object_id, "tool_via": mapped_via,
        "mesh_path": str(mesh_path),
        "pose_source": pose_src,
        "with_hand": hand_data is not None and mano is not None,
        "n_frames_written": n_written,
        "frame_range": [start_frame, end_frame, frame_step],
        "tile_size": [tile_w, tile_h],
        "fps": fps,
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[OK] wrote {out_path}  ({n_written} frames, "
          f"hand={'on' if summary['with_hand'] else 'off'}, "
          f"object_pose={pose_src})")
    return summary


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_root", type=Path, default=None,
                    help="Workspace mode: scan this dir for tasks/exps.")
    ap.add_argument("--annotated_sequence", type=Path, default=None,
                    help="Single-exp mode: <videos>_annotated/<task>/<exp>.")
    ap.add_argument("--task_filter", nargs="+", default=None)
    ap.add_argument("--only_remasked", action="store_true",
                    help="Only render exps that have a mask_prompts.json — "
                         "i.e. were re-annotated by simple_mask_annotator. "
                         "Use after batch_redo_masks + sbatch_run_remask_pipeline "
                         "to QA the regenerated subset.")
    ap.add_argument("--object_id", default=None)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1)
    ap.add_argument("--frame_step", type=int, default=1)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--tile_w", type=int, default=640)
    ap.add_argument("--tile_h", type=int, default=480)
    ap.add_argument("--pose_prefer", choices=("auto", "optimized", "fd"),
                    default="auto")
    ap.add_argument("--no_hand", action="store_true",
                    help="Skip hand rendering even if MANO artifacts exist.")
    ap.add_argument("--mesh_every", type=int, default=4,
                    help="Draw every Nth tool face. Smaller = denser, slower.")
    ap.add_argument("--hand_every", type=int, default=1,
                    help="Draw every Nth hand face.")
    ap.add_argument("--fill_alpha", type=float, default=0.0,
                    help="If >0, overlay filled (translucent) faces with this alpha.")
    ap.add_argument("--bg_white_alpha", type=float, default=0.0,
                    help="0=no change, 1=full white. Fades pixels outside the "
                         "union of projected object+hand mesh fills toward "
                         "white so the foreground pops. Try 0.5–0.7.")
    ap.add_argument("--bg_dilate_px", type=int, default=2,
                    help="Dilate the kept (foreground) region by this many "
                         "pixels before fading bg, so wireframes drawn on "
                         "the silhouette boundary aren't washed out.")
    ap.add_argument("--palette", choices=tuple(PALETTES.keys()), default="default",
                    help="Color preset: default (amber/coral/sky), warm "
                         "(amber/rose/magenta), cool (teal/cyan/violet), "
                         "vivid (the old saturated RGB), pastel (soft), "
                         "ink (dark tool, ink-red/blue hands — best with "
                         "--bg_white_alpha 0.6+).")
    ap.add_argument("--object_color", type=str, default=None,
                    help="Override tool color: hex '#fbbf24' or 'r,g,b'.")
    ap.add_argument("--left_hand_color", type=str, default=None,
                    help="Override left-hand color: hex or 'r,g,b'.")
    ap.add_argument("--right_hand_color", type=str, default=None,
                    help="Override right-hand color: hex or 'r,g,b'.")
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true", default=True)
    ap.add_argument("--tool_name_map_json", type=Path, default=None)
    ap.add_argument("--no_tool_name_map_json", action="store_true")
    ap.add_argument("--models_folder", type=Path, default=None,
                    help="Override the models folder (object meshes root). "
                         "If set, takes priority over meta.yaml's 'models_folder' "
                         "field. Useful when meta was generated on another "
                         "machine / a stale path, or you want to point at a "
                         "different mesh set.")
    ap.add_argument("--show_deformable", action="store_true",
                    help="Overlay deformable mask layer (written by "
                         "scripts/batch_deformable_annotator.py) onto each cam "
                         "tile before wireframe is drawn.")
    ap.add_argument("--deformable_subdir", type=str, default="deformable_masks")
    ap.add_argument("--deformable_color", type=str, default="#dc32c8")
    ap.add_argument("--deformable_alpha", type=float, default=0.45)
    args = ap.parse_args()

    if args.videos_root is None and args.annotated_sequence is None:
        ap.error("provide either --videos_root or --annotated_sequence")
    if args.videos_root is not None and args.annotated_sequence is not None:
        ap.error("--videos_root and --annotated_sequence are mutually exclusive")

    pal_obj, pal_left, pal_right = PALETTES[args.palette]
    object_color     = parse_color(args.object_color, pal_obj)
    left_hand_color  = parse_color(args.left_hand_color, pal_left)
    right_hand_color = parse_color(args.right_hand_color, pal_right)
    print(f"[palette={args.palette}] tool={object_color}  "
          f"left={left_hand_color}  right={right_hand_color}")

    common = dict(
        start_frame=args.start_frame, end_frame=args.end_frame,
        frame_step=args.frame_step, pose_prefer=args.pose_prefer,
        fps=args.fps, tile_w=args.tile_w, tile_h=args.tile_h,
        no_hand=args.no_hand, fill_alpha=args.fill_alpha,
        mesh_every=args.mesh_every, hand_every=args.hand_every,
        bg_white_alpha=args.bg_white_alpha, bg_dilate_px=args.bg_dilate_px,
        object_color=object_color,
        left_hand_color=left_hand_color,
        right_hand_color=right_hand_color,
        tool_name_map_json=args.tool_name_map_json,
        tool_name_map_disabled=args.no_tool_name_map_json,
        models_folder_override=args.models_folder,
        show_deformable=args.show_deformable,
        deformable_subdir=args.deformable_subdir,
        deformable_color=parse_color(args.deformable_color, (220, 50, 200)),
        deformable_alpha=args.deformable_alpha,
        object_id=args.object_id, force=args.force,
    )

    # ---------- single-exp ----------
    if args.annotated_sequence is not None:
        try:
            render_one_exp(annotated_sequence=args.annotated_sequence,
                            output_path=args.output_dir, **common)
        except Exception as e:
            print(f"[ERR] {e}", file=sys.stderr)
            traceback.print_exc(); sys.exit(1)
        return

    # ---------- workspace ----------
    videos_root = args.videos_root.resolve()
    if not videos_root.is_dir():
        ap.error(f"--videos_root not a directory: {videos_root}")

    exps, n_skip = discover_done_exps(
        videos_root,
        task_filter=set(args.task_filter) if args.task_filter else None,
        only_remasked=args.only_remasked)
    if not exps:
        print(f"No DONE exps under {videos_root} (skipped {n_skip} unfinished)")
        sys.exit(0)
    print(f"[INFO] {len(exps)} done exps  (skipped {n_skip} unfinished)\n")

    n_ok = n_fail = n_skip2 = 0
    failures = []
    t0 = time.time()
    for i, (task, exp, _exp_dir, ann, reason) in enumerate(exps, 1):
        prefix = f"[{i:>4d}/{len(exps)}] {task}/{exp}"
        per_exp_out = (args.output_dir / task / exp / f"{args.pose_prefer}_hand_object.mp4"
                        if args.output_dir is not None else None)
        existing = (per_exp_out if per_exp_out is not None else
                     ann / "debug" / "hand_object_video" / f"{args.pose_prefer}_hand_object.mp4")
        if existing.exists() and not args.force:
            print(f"{prefix}  skip  (mp4 exists; pass --force)")
            n_skip2 += 1; continue
        try:
            t1 = time.time()
            summary = render_one_exp(annotated_sequence=ann,
                                          output_path=per_exp_out, **common)
            n_ok += 1
            print(f"{prefix}  OK [{time.time()-t1:.1f}s]  "
                  f"{summary.get('n_frames_written', '?')} frames  "
                  f"({reason})")
        except KeyboardInterrupt:
            print("[INTERRUPTED]"); sys.exit(130)
        except Exception as e:
            n_fail += 1; failures.append((task, exp, str(e)))
            print(f"{prefix}  FAIL  ({type(e).__name__}: {e})")
            if not args.continue_on_error: raise

    print()
    print("=" * 60)
    print(f"summary: ok={n_ok}  skip={n_skip2}  fail={n_fail}  "
          f"elapsed={time.time()-t0:.1f}s")
    if failures:
        print("\nFailures:")
        for task, exp, msg in failures[:10]:
            print(f"  {task}/{exp}: {msg}")
        if len(failures) > 10:
            print(f"  ... and {len(failures)-10} more")
    print("=" * 60)


if __name__ == "__main__":
    main()
