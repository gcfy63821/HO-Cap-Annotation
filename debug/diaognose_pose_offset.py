import argparse
import contextlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation as R


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def derive_original_sequence(annotated_sequence: Path) -> Path:
    parts = annotated_sequence.parts
    if "data" not in parts:
        raise ValueError(f"Cannot derive original sequence from {annotated_sequence}")
    data_idx = parts.index("data")
    tail = list(parts[data_idx + 1 :])
    if not tail or not tail[0].endswith("_annotated"):
        raise ValueError(f"Expected annotated path under .../data/<videos>_annotated/...: {annotated_sequence}")
    tail[0] = tail[0].replace("_annotated", "", 1)
    return Path(*parts[: data_idx + 1], *tail)


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
    def __init__(self, sequence_folder: Path, serials: list[str]) -> None:
        self.serials = serials
        self.rgb_caps = {}
        self.depth_caps = {}
        self.frame_count = None
        for serial in serials:
            cam_idx = int(serial)
            rgb_path = sequence_folder / f"cam{cam_idx}_rgb.mp4"
            depth_path = sequence_folder / f"cam{cam_idx}_depth.mkv"
            rgb_cap = cv2.VideoCapture(str(rgb_path))
            depth_cap = cv2.VideoCapture(str(depth_path))
            if not rgb_cap.isOpened() or not depth_cap.isOpened():
                raise FileNotFoundError(f"Failed to open RGB/depth videos for camera {serial}")
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
            raise ValueError(f"Failed to read RGB frame {frame_idx} for camera {serial}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_depth(self, frame_idx: int, serial: str) -> np.ndarray:
        cap = self.depth_caps[serial]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Failed to read depth frame {frame_idx} for camera {serial}")
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


def choose_frames(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= max_frames:
        return list(range(frame_count))
    return sorted(set(np.linspace(0, frame_count - 1, num=max_frames, dtype=int).tolist()))


def main():
    parser = argparse.ArgumentParser(description="Diagnose fixed pose offsets from mesh/camera/world mismatches.")
    parser.add_argument("--annotated_sequence", required=True, type=Path)
    parser.add_argument("--object_id", default=None, help="Override object id; default: infer from meta.yaml or fd_pose_solver folder")
    parser.add_argument("--max_frames", type=int, default=12, help="Number of sampled frames to analyze")
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Explicit frame ids to analyze")
    parser.add_argument("--output_dir", type=Path, default=None, help="Where to write diagnostics")
    args = parser.parse_args()

    annotated_sequence = args.annotated_sequence.resolve()
    original_sequence = derive_original_sequence(annotated_sequence)
    meta_path = original_sequence / "meta.yaml"
    meta = load_yaml(meta_path)

    object_id = args.object_id or meta["object_ids"][0]
    calib_path = Path(meta["calibration_yaml_path"])
    models_folder = Path(meta["models_folder"])
    mesh_path = find_mesh_path(models_folder, object_id)
    tool_masks_h5 = annotated_sequence / "tool_masks" / "masks.h5"
    ob_in_cam_root = annotated_sequence / "processed" / "fd_pose_solver" / object_id / "ob_in_cam"
    out_root = args.output_dir if args.output_dir is not None else annotated_sequence / "debug" / "pose_offset_diag"
    out_root.mkdir(parents=True, exist_ok=True)

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
        frames = args.frames if args.frames else choose_frames(frame_count, args.max_frames)

        summary = {
            "annotated_sequence": str(annotated_sequence),
            "original_sequence": str(original_sequence),
            "mesh_path": str(mesh_path),
            "h5_path": None if h5_path is None else str(h5_path),
            "tool_masks_h5": str(tool_masks_h5),
            "object_id": object_id,
            "frames": frames,
            "per_frame": [],
        }

        offset_vectors = []
        world_positions = {serial: [] for serial in serials}

        for frame_idx in frames:
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
                mask = (raw_mask == 1).astype(np.uint8) if raw_mask.max() <= 1 else (raw_mask == (meta["object_ids"].index(object_id) + 1)).astype(np.uint8)

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
    aggregate["world_positions_per_camera"] = world_positions
    summary["aggregate"] = aggregate

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        f"annotated_sequence: {annotated_sequence}",
        f"original_sequence: {original_sequence}",
        f"mesh_path: {mesh_path}",
        f"frames: {frames}",
    ]
    if "mean_offset_cam" in aggregate:
        lines.append(f"mean_offset_cam: {aggregate['mean_offset_cam']}")
        lines.append(f"median_offset_cam: {aggregate['median_offset_cam']}")
        lines.append(f"std_offset_cam: {aggregate['std_offset_cam']}")
        lines.append(f"offset_norm_mean_m: {aggregate['offset_norm_mean']:.6f}")
    with open(out_root / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[INFO] Wrote diagnostics to {out_root}")


if __name__ == "__main__":
    main()
