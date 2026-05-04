"""Diagnose fixed object-pose offsets across all DONE exps under a videos_root.

For each exp that has Stage 4/5 outputs (processed/fd_pose_solver/.../ob_in_cam/
populated, or fd_poses_merged_fixed.npy present), this script:
  - Reads the per-frame ob_in_cam pose, the tool mask, and the depth.
  - Projects the mesh into RGB and overlays it onto color + mask.
  - Computes the centroid in camera frame from the saved pose vs. the centroid
    of (depth ∩ mask) — the difference is the depth-vs-pose offset.
  - Writes per-frame overlays + a summary.json to
      <annotated>/<exp>/debug/pose_offset_diag/

Two modes:

  (a) Workspace mode (recommended):
      python debug/diaognose_pose_offset.py \\
        --videos_root /viscam/projects/robotool/data/videos_0102 \\
        [--task_filter task_a task_b] \\
        [--max_frames 12]   [--frames 0 100 250]
      Auto-discovers every <task>/<exp> under videos_root whose annotated side
      has fd_pose_solver outputs ready, and runs the diagnosis on each.

  (b) Single-exp mode (legacy):
      python debug/diaognose_pose_offset.py \\
        --annotated_sequence /viscam/.../videos_0102_annotated/<task>/<exp>
"""

import argparse
import contextlib
import json
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


# ============================================================
# IO + path helpers
# ============================================================

def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def derive_original_sequence(annotated_sequence: Path) -> Path:
    """<base>/<videos>_annotated/<task>/<exp>  →  <base>/<videos>/<task>/<exp>"""
    parts = annotated_sequence.parts
    # Find the videos_<id>_annotated component (any position).
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
            if any(name.endswith("imgs") for name in names) and any(name.endswith("depths") for name in names):
                return path
        except OSError:
            continue
    raise FileNotFoundError(f"No HDF5 with imgs/depths found in {sequence_folder}")


class VideoRGBDSource:
    """Fallback when there's no permanent h5: stream directly from cam*_rgb.mp4
    + cam*_depth.mkv. Slower (per-frame seek) but doesn't need the h5 to be
    persisted on disk."""

    def __init__(self, sequence_folder: Path, serials: list) -> None:
        self.serials = serials
        self.rgb_caps = {}
        self.depth_caps = {}
        self.frame_count = None
        for cam_idx, serial in enumerate(serials):
            rgb_path = sequence_folder / f"cam{cam_idx}_rgb.mp4"
            depth_path = sequence_folder / f"cam{cam_idx}_depth.mkv"
            rgb_cap = cv2.VideoCapture(str(rgb_path))
            depth_cap = cv2.VideoCapture(str(depth_path))
            if not rgb_cap.isOpened() or not depth_cap.isOpened():
                raise FileNotFoundError(f"Failed to open RGB/depth videos for cam {cam_idx} (serial {serial})")
            self.rgb_caps[serial] = rgb_cap
            self.depth_caps[serial] = depth_cap
            if self.frame_count is None:
                self.frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def close(self):
        for cap in list(self.rgb_caps.values()) + list(self.depth_caps.values()):
            cap.release()

    def get_color(self, frame_idx: int, serial: str) -> np.ndarray:
        cap = self.rgb_caps[serial]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Failed to read RGB frame {frame_idx} for cam {serial}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_depth(self, frame_idx: int, serial: str) -> np.ndarray:
        cap = self.depth_caps[serial]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Failed to read depth frame {frame_idx} for cam {serial}")
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        return frame.astype(np.float32) * 0.001


def find_dataset(f: h5py.File, target_name: str):
    hit = None

    def visitor(name, obj):
        nonlocal hit
        if hit is not None:
            return
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == target_name:
            hit = obj

    f.visititems(visitor)
    if hit is None:
        raise KeyError(f"Dataset '{target_name}' not found in {f.filename}")
    return hit


def find_mesh_path(models_folder: Path, object_id: str) -> Path:
    obj_dir = models_folder / object_id
    for cand in ("cleaned_mesh_10000.obj", "textured_mesh.obj", "mesh.obj"):
        path = obj_dir / cand
        if path.exists():
            return path
    raise FileNotFoundError(f"No mesh found under {obj_dir}")


def read_pose_txt(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    arr = np.asarray(arr).reshape(-1)
    if arr.shape[0] == 7:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.from_quat(arr[:4]).as_matrix().astype(np.float32)
        pose[:3, 3] = arr[4:7]
        return pose
    if arr.shape[0] == 16:
        return arr.reshape(4, 4).astype(np.float32)
    raise ValueError(f"Unsupported pose format in {path}: shape {arr.shape}")


def project_points(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = points_cam[:, 2:3]
    valid = z[:, 0] > 1e-8
    pts = np.full((points_cam.shape[0], 2), -1.0, dtype=np.float32)
    if np.any(valid):
        proj = points_cam[valid] @ K.T
        pts[valid] = proj[:, :2] / proj[:, 2:3]
    return pts


def depth_to_xyz(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    z = depth.reshape(-1)
    x = (u.reshape(-1) - K[0, 2]) * z / K[0, 0]
    y = (v.reshape(-1) - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def draw_overlay(color: np.ndarray, mask: np.ndarray, mesh_pts_2d: np.ndarray, mesh_faces: np.ndarray) -> np.ndarray:
    vis = color.copy()
    overlay = color.copy()
    overlay[mask > 0] = (255, 0, 0)
    vis = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)

    valid = (
        (mesh_pts_2d[:, 0] >= 0)
        & (mesh_pts_2d[:, 1] >= 0)
        & np.isfinite(mesh_pts_2d[:, 0])
        & np.isfinite(mesh_pts_2d[:, 1])
    )
    pts_i32 = mesh_pts_2d.astype(np.int32, copy=False)
    for face in mesh_faces[::50]:
        if valid[face].all():
            cv2.polylines(vis, [pts_i32[face]], isClosed=True, color=(0, 255, 0), thickness=1)
    for pt in pts_i32[::300]:
        if pt[0] >= 0 and pt[1] >= 0:
            cv2.circle(vis, tuple(pt), 1, (0, 255, 255), -1)
    return vis


def choose_frames(frame_count: int, max_frames: int) -> list:
    if frame_count <= max_frames:
        return list(range(frame_count))
    return sorted(set(np.linspace(0, frame_count - 1, num=max_frames, dtype=int).tolist()))


# ============================================================
# DONE detection
# ============================================================

def is_exp_done(annotated_exp: Path, object_id: str = None) -> tuple:
    """Returns (is_done, reason). DONE means: at least one object's
    ob_in_cam/<serial>/ has per-frame txts on disk, OR the merged
    fd_poses_merged_fixed.npy exists (Stage 5)."""
    fd_root = annotated_exp / "processed" / "fd_pose_solver"
    if not fd_root.is_dir():
        return False, "no processed/fd_pose_solver"
    merged = fd_root / "fd_poses_merged_fixed.npy"
    if merged.exists():
        return True, "fd_poses_merged_fixed.npy"
    # Otherwise look for any object's per-frame txts.
    candidates = ([fd_root / object_id] if object_id else
                   [p for p in fd_root.iterdir() if p.is_dir()])
    for obj_dir in candidates:
        ob_in_cam = obj_dir / "ob_in_cam"
        if not ob_in_cam.is_dir():
            continue
        for serial_dir in ob_in_cam.iterdir():
            if serial_dir.is_dir() and any(serial_dir.glob("*.txt")):
                return True, f"per-frame txts under {obj_dir.name}/ob_in_cam"
    return False, "no per-frame ob_in_cam txts"


def discover_done_exps(videos_root: Path, task_filter=None) -> list:
    """Walk <videos_root>/<task>/<exp>/ and return the list of exps where
    the annotated side has Stage 4 (or 5) outputs ready. Returns a list of
    tuples: (task, exp, exp_dir, annotated_exp_dir, done_reason)."""
    out = []
    skipped = 0
    for task_dir in sorted(p for p in videos_root.iterdir() if p.is_dir()):
        if task_dir.name.startswith("realsense_calibrate"): continue
        if task_dir.name.startswith("ref_pc"):              continue
        if task_dir.name.startswith("posts"):               continue
        if task_dir.name == "first_frame":                  continue
        if task_filter and task_dir.name not in task_filter: continue
        for exp_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            if not list(exp_dir.glob("cam*_rgb.mp4")):
                continue
            ann = derive_annotated_for(videos_root, task_dir.name, exp_dir.name)
            ok, reason = is_exp_done(ann)
            if ok:
                out.append((task_dir.name, exp_dir.name, exp_dir, ann, reason))
            else:
                skipped += 1
    return out, skipped


# ============================================================
# Per-exp diagnosis
# ============================================================

def diagnose_one_exp(annotated_sequence: Path,
                       frames: list = None,
                       max_frames: int = 12,
                       object_id: str = None,
                       output_dir: Path = None,
                       force: bool = False) -> dict:
    """Run the diagnosis on a single annotated exp. Returns the summary dict.
    Raises on hard errors so the workspace driver can record + skip."""
    annotated_sequence = annotated_sequence.resolve()
    original_sequence = derive_original_sequence(annotated_sequence)
    meta_path = original_sequence / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta.yaml: {meta_path}")
    meta = load_yaml(meta_path)

    if not object_id:
        object_id = meta["object_ids"][0]
    calib_path = Path(meta["calibration_yaml_path"])
    models_folder = Path(meta["models_folder"])
    mesh_path = find_mesh_path(models_folder, object_id)
    tool_masks_h5 = annotated_sequence / "tool_masks" / "masks.h5"
    if not tool_masks_h5.exists():
        # Legacy fallback location.
        legacy = annotated_sequence / "masks" / "masks.h5"
        if legacy.exists():
            tool_masks_h5 = legacy
        else:
            raise FileNotFoundError(f"no masks.h5 under {annotated_sequence}/(tool_masks|masks)/")
    ob_in_cam_root = annotated_sequence / "processed" / "fd_pose_solver" / object_id / "ob_in_cam"
    if not ob_in_cam_root.is_dir():
        raise FileNotFoundError(f"no ob_in_cam dir for object {object_id}: {ob_in_cam_root}")

    out_root = output_dir if output_dir is not None else annotated_sequence / "debug" / "pose_offset_diag"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "summary.json"
    if summary_path.exists() and not force:
        # Quick skip — caller is responsible for setting force when re-running.
        try:
            return json.loads(summary_path.read_text())
        except Exception:
            pass    # fall through to re-run

    calib = load_yaml(calib_path)
    serials = [str(cam["camera_id"]).zfill(2) for cam in calib]
    Ks = {str(cam["camera_id"]).zfill(2): np.array(cam["color_intrinsic_matrix"], dtype=np.float32) for cam in calib}
    cam_RTs = {str(cam["camera_id"]).zfill(2): np.array(cam["transformation"], dtype=np.float32) for cam in calib}

    mesh = trimesh.load(mesh_path, process=True, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    mesh_centroid_local = vertices.mean(axis=0)

    h5_path = None
    video_source = None
    try:
        h5_path = find_h5_with_rgbd(original_sequence)
    except FileNotFoundError:
        video_source = VideoRGBDSource(original_sequence, serials)

    data_ctx = contextlib.nullcontext(None) if h5_path is None else h5py.File(h5_path, "r")
    with data_ctx as data_f, h5py.File(tool_masks_h5, "r") as mask_f:
        imgs_ds = find_dataset(data_f, "imgs") if data_f else None
        depths_ds = find_dataset(data_f, "depths") if data_f else None
        masks_ds = find_dataset(mask_f, "masks")

        frame_count = int(imgs_ds.shape[0]) if imgs_ds is not None else int(masks_ds.shape[0])
        chosen_frames = frames if frames else choose_frames(frame_count, max_frames)
        # Clip to what's actually available.
        chosen_frames = [f for f in chosen_frames if 0 <= f < frame_count]

        summary = {
            "annotated_sequence": str(annotated_sequence),
            "original_sequence": str(original_sequence),
            "mesh_path": str(mesh_path),
            "h5_path": None if h5_path is None else str(h5_path),
            "tool_masks_h5": str(tool_masks_h5),
            "object_id": object_id,
            "frames": chosen_frames,
            "per_frame": [],
        }
        offset_vectors = []
        world_positions = {serial: [] for serial in serials}

        for frame_idx in chosen_frames:
            frame_dir = out_root / f"frame_{frame_idx:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_record = {"frame": frame_idx, "cameras": []}

            for cam_idx, serial in enumerate(serials):
                pose_path = ob_in_cam_root / serial / f"{frame_idx:06d}.txt"
                if not pose_path.exists():
                    continue

                pose_c = read_pose_txt(pose_path)
                if np.all(pose_c == -1):
                    continue

                if imgs_ds is not None:
                    color = np.asarray(imgs_ds[frame_idx, cam_idx], dtype=np.uint8)
                    depth = np.asarray(depths_ds[frame_idx, cam_idx], dtype=np.float32) * 0.001
                else:
                    color = video_source.get_color(frame_idx, serial)
                    depth = video_source.get_depth(frame_idx, serial)
                raw_mask = np.asarray(masks_ds[frame_idx, cam_idx])
                if raw_mask.ndim == 3:
                    raw_mask = raw_mask[0]
                mask = (
                    (raw_mask == 1).astype(np.uint8)
                    if raw_mask.max() <= 1
                    else (raw_mask == (meta["object_ids"].index(object_id) + 1)).astype(np.uint8)
                )

                verts_cam = (pose_c[:3, :3] @ vertices.T).T + pose_c[:3, 3]
                centroid_mesh_cam = pose_c[:3, :3] @ mesh_centroid_local + pose_c[:3, 3]

                valid_depth = (depth > 0) & (mask > 0)
                depth_centroid_cam = None
                offset_cam = None
                if np.count_nonzero(valid_depth) > 50:
                    pts_cam = depth_to_xyz(depth, Ks[serial])
                    depth_centroid_cam = pts_cam[valid_depth.reshape(-1)].mean(axis=0)
                    offset_cam = depth_centroid_cam - centroid_mesh_cam
                    offset_vectors.append(offset_cam)

                pose_w = cam_RTs[serial] @ pose_c
                world_positions[serial].append(pose_w[:3, 3].tolist())

                pts_2d = project_points(verts_cam, Ks[serial])
                overlay = draw_overlay(color, mask, pts_2d, faces)
                cv2.putText(
                    overlay,
                    f"frame {frame_idx:06d} cam {serial}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                if offset_cam is not None:
                    cv2.putText(
                        overlay,
                        f"offset_cam(m): [{offset_cam[0]:+.3f}, {offset_cam[1]:+.3f}, {offset_cam[2]:+.3f}]",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                cv2.imwrite(str(frame_dir / f"cam_{serial}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                frame_record["cameras"].append(
                    {
                        "serial": serial,
                        "pose_path": str(pose_path),
                        "mesh_centroid_cam": centroid_mesh_cam.tolist(),
                        "depth_centroid_cam": None if depth_centroid_cam is None else depth_centroid_cam.tolist(),
                        "offset_cam": None if offset_cam is None else offset_cam.tolist(),
                        "pose_world_translation": pose_w[:3, 3].tolist(),
                    }
                )

            summary["per_frame"].append(frame_record)

    if video_source is not None:
        video_source.close()

    aggregate = {}
    if offset_vectors:
        offsets = np.asarray(offset_vectors, dtype=np.float32)
        aggregate["mean_offset_cam"] = offsets.mean(axis=0).tolist()
        aggregate["median_offset_cam"] = np.median(offsets, axis=0).tolist()
        aggregate["std_offset_cam"] = offsets.std(axis=0).tolist()
        aggregate["offset_norm_mean"] = float(np.linalg.norm(offsets, axis=1).mean())
        aggregate["n_camera_observations"] = int(len(offsets))
    aggregate["world_positions_per_camera"] = world_positions
    summary["aggregate"] = aggregate

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        f"annotated_sequence: {annotated_sequence}",
        f"original_sequence: {original_sequence}",
        f"mesh_path: {mesh_path}",
        f"frames: {chosen_frames}",
    ]
    if "mean_offset_cam" in aggregate:
        lines.append(f"mean_offset_cam: {aggregate['mean_offset_cam']}")
        lines.append(f"median_offset_cam: {aggregate['median_offset_cam']}")
        lines.append(f"std_offset_cam: {aggregate['std_offset_cam']}")
        lines.append(f"offset_norm_mean_m: {aggregate['offset_norm_mean']:.6f}")
        lines.append(f"n_camera_observations: {aggregate['n_camera_observations']}")
    with open(out_root / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--videos_root", type=Path, default=None,
                         help="Workspace mode: scan this dir for tasks/exps and run "
                              "diagnosis on every exp whose annotated side has "
                              "fd_pose_solver outputs ready.")
    parser.add_argument("--annotated_sequence", type=Path, default=None,
                         help="Single-exp mode: path to one "
                              "<videos>_annotated/<task>/<exp>. Mutually exclusive "
                              "with --videos_root.")
    parser.add_argument("--task_filter", nargs="+", default=None,
                         help="Workspace mode: only process these task subfolders.")
    parser.add_argument("--object_id", default=None,
                         help="Override object id; default: first entry in meta.yaml's object_ids.")
    parser.add_argument("--max_frames", type=int, default=12,
                         help="Number of evenly-sampled frames per exp.")
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                         help="Explicit frame ids; overrides --max_frames. Same list "
                              "is applied to every exp in workspace mode.")
    parser.add_argument("--output_dir", type=Path, default=None,
                         help="Where to write diagnostics. In workspace mode, this is "
                              "treated as a ROOT and per-exp subdirs are created. "
                              "Default: per-exp <annotated>/debug/pose_offset_diag/.")
    parser.add_argument("--force", action="store_true",
                         help="Re-run diagnosis even if summary.json already exists.")
    parser.add_argument("--continue_on_error", action="store_true", default=True,
                         help="(workspace mode) keep going if one exp fails. Default ON.")
    args = parser.parse_args()

    if args.videos_root is None and args.annotated_sequence is None:
        parser.error("provide either --videos_root or --annotated_sequence")
    if args.videos_root is not None and args.annotated_sequence is not None:
        parser.error("--videos_root and --annotated_sequence are mutually exclusive")

    # ---------- single-exp mode ----------
    if args.annotated_sequence is not None:
        out_dir = args.output_dir
        try:
            summary = diagnose_one_exp(
                annotated_sequence=args.annotated_sequence,
                frames=args.frames, max_frames=args.max_frames,
                object_id=args.object_id, output_dir=out_dir, force=args.force,
            )
            out_path = (out_dir if out_dir is not None
                          else args.annotated_sequence.resolve() / "debug" / "pose_offset_diag")
            print(f"[INFO] Wrote diagnostics to {out_path}")
            agg = summary.get("aggregate", {})
            if "mean_offset_cam" in agg:
                print(f"[INFO] mean_offset_cam={agg['mean_offset_cam']}  "
                      f"||·||={agg['offset_norm_mean']:.4f} m  "
                      f"(n={agg.get('n_camera_observations', 0)})")
        except Exception as e:
            print(f"[ERR] {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        return

    # ---------- workspace mode ----------
    videos_root = args.videos_root.resolve()
    if not videos_root.is_dir():
        parser.error(f"--videos_root not a directory: {videos_root}")
    annotated_root = videos_root.parent / f"{videos_root.name}_annotated"

    print(f"[INFO] videos_root      = {videos_root}")
    print(f"[INFO] annotated_root   = {annotated_root}")
    if args.task_filter:
        print(f"[INFO] task_filter      = {args.task_filter}")
    print(f"[INFO] frames           = {args.frames if args.frames else f'auto ({args.max_frames} samples)'}")
    print()

    exps, n_skipped = discover_done_exps(
        videos_root, task_filter=set(args.task_filter) if args.task_filter else None
    )
    if not exps:
        print(f"No DONE exps under {videos_root}")
        if n_skipped:
            print(f"   (skipped {n_skipped} exps without fd_pose_solver outputs — "
                  f"run the pipeline first)")
        sys.exit(0)

    print(f"[INFO] DONE exps        = {len(exps)}   (skipped {n_skipped} unfinished)")
    print()

    n_ok = n_fail = n_skip = 0
    failures = []
    aggregates_per_exp = []
    t0 = time.time()
    for i, (task, exp, exp_dir, ann, reason) in enumerate(exps, 1):
        prefix = f"[{i:>4d}/{len(exps)}] {task}/{exp}"
        per_exp_out = (
            args.output_dir / task / exp if args.output_dir is not None else None
        )
        # Quick skip if summary already exists and not forcing.
        existing = (
            (per_exp_out if per_exp_out is not None
             else ann / "debug" / "pose_offset_diag") / "summary.json"
        )
        if existing.exists() and not args.force:
            print(f"{prefix}  skip  (summary.json exists; pass --force to redo)")
            n_skip += 1
            try:
                aggregates_per_exp.append({
                    "task": task, "exp": exp,
                    "summary_path": str(existing),
                    "aggregate": json.loads(existing.read_text()).get("aggregate", {}),
                })
            except Exception:
                pass
            continue
        try:
            t1 = time.time()
            summary = diagnose_one_exp(
                annotated_sequence=ann,
                frames=args.frames, max_frames=args.max_frames,
                object_id=args.object_id, output_dir=per_exp_out, force=args.force,
            )
            n_ok += 1
            agg = summary.get("aggregate", {})
            tag = ""
            if "mean_offset_cam" in agg:
                tag = (f"  ||offset||={agg['offset_norm_mean']:.3f} m  "
                       f"(n={agg.get('n_camera_observations', 0)})")
            print(f"{prefix}  OK    [{time.time()-t1:.1f}s]{tag}  ({reason})")
            aggregates_per_exp.append({
                "task": task, "exp": exp,
                "summary_path": str(existing),
                "aggregate": agg,
            })
        except KeyboardInterrupt:
            print("[INTERRUPTED]")
            sys.exit(130)
        except Exception as e:
            n_fail += 1
            failures.append((task, exp, str(e)))
            print(f"{prefix}  FAIL  ({type(e).__name__}: {e})")
            if not args.continue_on_error:
                raise
    elapsed = time.time() - t0

    # Workspace-level summary.
    ws_root = args.output_dir if args.output_dir is not None else annotated_root
    ws_summary_path = ws_root / "pose_offset_diag_summary.json"
    try:
        ws_root.mkdir(parents=True, exist_ok=True)
        with open(ws_summary_path, "w") as f:
            json.dump({
                "videos_root": str(videos_root),
                "annotated_root": str(annotated_root),
                "n_total": len(exps),
                "n_ok": n_ok, "n_skip": n_skip, "n_fail": n_fail,
                "elapsed_s": round(elapsed, 1),
                "frames_arg": args.frames,
                "max_frames": args.max_frames,
                "object_id_override": args.object_id,
                "exps": aggregates_per_exp,
                "failures": failures,
            }, f, indent=2)
    except Exception as e:
        print(f"[WARN] could not write workspace summary: {e}")

    print()
    print("=" * 60)
    print(f"Summary: ok={n_ok}  skip={n_skip}  fail={n_fail}  "
          f"elapsed={elapsed:.1f}s "
          f"({len(exps)/max(elapsed, 1e-6):.2f} exps/s)")
    if ws_summary_path.exists():
        print(f"Workspace summary written: {ws_summary_path}")
    if failures:
        print("\nFailures:")
        for task, exp, msg in failures[:10]:
            print(f"  {task}/{exp}: {msg}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
    print("=" * 60)


if __name__ == "__main__":
    main()
