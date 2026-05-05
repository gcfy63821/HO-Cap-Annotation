import os
import h5py
import yaml
import numpy as np
from pathlib import Path
import argparse
import re


def _load_mask_file(path):
    """Load a per-frame mask saved as .npy OR .npz. For .npz we expect key 'mask'
    (our writers use np.savez_compressed(..., mask=m)) but fall back to the
    default 'arr_0' key for files saved via np.savez without a keyword."""
    p = Path(path)
    if p.suffix == '.npz':
        with np.load(p) as npz:
            if 'mask' in npz.files:
                return np.array(npz['mask'])
            return np.array(npz['arr_0'])
    return np.load(p)


def _find_mask_file(cam_folder, frame_idx):
    """Return the first existing frame-mask file for (cam_folder, frame_idx),
    trying both compressed (.npz) and plain (.npy) variants with zero-padded
    and un-padded names."""
    for fmt in (f"{frame_idx:04d}.npz", f"{frame_idx}.npz",
                f"{frame_idx:04d}.npy", f"{frame_idx}.npy"):
        p = cam_folder / fmt
        if p.exists():
            return p
    return None


def discover_camera_folders(mask_root_dir):
    """
    Discover actual camera folders from mask directory and return sorted mapping.
    Supports names like:
      - cam8_rgb
      - cam08.mp4
      - cam8.mp4
    Returns:
        list[tuple[int, Path]]: [(camera_id_int, folder_path), ...] sorted by camera_id.
    """
    mask_root_dir = Path(mask_root_dir)
    if not mask_root_dir.exists():
        return []

    camera_folders = []
    for child in mask_root_dir.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"^cam(\d+)(?:_rgb|\.mp4)?$", child.name)
        if m:
            camera_folders.append((int(m.group(1)), child))

    camera_folders.sort(key=lambda x: x[0])
    return camera_folders

def load_masks_from_folder(mask_root_dir, num_frames, num_cams, expected_H=None, expected_W=None, start_frame=0):
    """
    根据mask根目录读取所有摄像头的mask，返回形状 (N, num_cams, H, W)
    mask_root_dir路径格式支持:
    tool_masks/
      cam00.mp4/ 或 cam0_rgb/
        0.npy 或 0000.npy
        1.npy 或 0001.npy
        ...
      cam01.mp4/ 或 cam1_rgb/
      ...
      cam07.mp4/ 或 cam7_rgb/
    
    Args:
        mask_root_dir: mask根目录路径
        num_frames: 帧数
        num_cams: 相机数量
        expected_H: 期望的高度（如果提供，会resize mask）
        expected_W: 期望的宽度（如果提供，会resize mask）
    """
    mask_root_dir = Path(mask_root_dir)
    all_masks = []
    
    # Build camera-folder mapping from actual directory names
    discovered = discover_camera_folders(mask_root_dir)
    if len(discovered) == 0:
        camera_folders = []
    else:
        if len(discovered) < num_cams:
            print(f"[WARNING] Discovered {len(discovered)} camera folder(s), but h5 has {num_cams} camera(s). Missing views will be filled with zeros.")
        elif len(discovered) > num_cams:
            print(f"[WARNING] Discovered {len(discovered)} camera folder(s), but h5 has {num_cams} camera(s). Using first {num_cams} by camera id.")
        camera_folders = [folder for _, folder in discovered[:num_cams]]

    # 从第一帧推断mask的尺寸
    first_mask_shape = None
    for cam_folder in camera_folders:
        # 尝试多种文件名格式
        for fmt in [f"{0:04d}.npy", f"{0}.npy"]:
            npy_path = cam_folder / fmt
            if npy_path.exists():
                mask = np.load(npy_path)
                # 处理mask形状
                if mask.ndim == 3:
                    mask = mask.squeeze(0)  # (1, H, W) -> (H, W)
                if first_mask_shape is None:
                    first_mask_shape = mask.shape
                break
        if first_mask_shape is not None:
            break
    
    # 如果无法推断，使用默认值
    if first_mask_shape is None:
        H, W = expected_H or 480, expected_W or 640
        print(f"[WARNING] Could not infer mask shape, using default: ({H}, {W})")
    else:
        H, W = first_mask_shape
        print(f"[INFO] Inferred mask shape: ({H}, {W})")
    
    # 如果提供了期望尺寸，使用期望尺寸
    if expected_H is not None and expected_W is not None:
        H, W = expected_H, expected_W
        print(f"[INFO] Using expected mask shape: ({H}, {W})")
    
    all_masks = []
    for frame_idx in range(num_frames):
        # mask npy 文件使用原始视频帧号索引
        original_frame_idx = frame_idx + start_frame
        frame_masks = []
        for cam_idx in range(num_cams):
            npy_path = None
            if cam_idx < len(camera_folders):
                cam_folder = camera_folders[cam_idx]
                npy_path = _find_mask_file(cam_folder, original_frame_idx)

            if npy_path is None:
                frame_masks.append(np.zeros((H, W), dtype=np.uint8))
                continue

            mask = _load_mask_file(npy_path)
            
            # 处理mask形状
            if mask.ndim == 3:
                mask = mask.squeeze(0)  # (1, H, W) -> (H, W)
            elif mask.ndim != 2:
                print(f"[WARNING] Unexpected mask shape {mask.shape} from {npy_path}, using zeros")
                frame_masks.append(np.zeros((H, W), dtype=np.uint8))
                continue
            
            # Resize mask if dimensions don't match
            if mask.shape[0] != H or mask.shape[1] != W:
                try:
                    import cv2
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                except ImportError:
                    from scipy.ndimage import zoom
                    scale_h = H / mask.shape[0]
                    scale_w = W / mask.shape[1]
                    mask = zoom(mask, (scale_h, scale_w), order=0).astype(mask.dtype)
            
            frame_masks.append(mask)
        all_masks.append(frame_masks)
    
    all_masks = np.array(all_masks)  # (N, num_cams, H, W)
    print(f"[INFO] Loaded masks with shape: {all_masks.shape}, dtype: {all_masks.dtype}")
    return all_masks
def save_masks_to_h5(masks, h5_path, dataset_name="masks"):
    """
    Save masks numpy array to an h5 file with the given dataset name.
    """
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset(dataset_name, data=masks, compression="gzip")
    print(f"[INFO] Saved {dataset_name} to {h5_path}")


def stream_masks_folder_to_h5(mask_root_dir, h5_path, num_frames, num_cams,
                              expected_H=None, expected_W=None, start_frame=0,
                              dataset_name="masks"):
    """Stream per-frame masks from on-disk .npy/.npz files directly into a
    chunked H5 dataset, frame by frame, without ever holding the full
    (N, n_cams, H, W) array in RAM. Memory use is bounded by one frame
    of masks (~10 MB) plus the H5 chunk cache.

    Mirrors the per-frame logic of load_masks_from_folder but writes each
    frame as soon as it's read.
    """
    mask_root_dir = Path(mask_root_dir)
    discovered = discover_camera_folders(mask_root_dir)
    if len(discovered) == 0:
        camera_folders = []
    else:
        if len(discovered) < num_cams:
            print(f"[WARNING] Discovered {len(discovered)} camera folder(s), but h5 has {num_cams} camera(s). Missing views will be filled with zeros.")
        elif len(discovered) > num_cams:
            print(f"[WARNING] Discovered {len(discovered)} camera folder(s), but h5 has {num_cams} camera(s). Using first {num_cams} by camera id.")
        camera_folders = [folder for _, folder in discovered[:num_cams]]

    # Infer mask shape from the first available file.
    first_mask_shape = None
    for cam_folder in camera_folders:
        for fmt in [f"{0:04d}.npy", f"{0}.npy"]:
            npy_path = cam_folder / fmt
            if npy_path.exists():
                m = np.load(npy_path)
                if m.ndim == 3:
                    m = m.squeeze(0)
                first_mask_shape = m.shape
                break
        if first_mask_shape is not None:
            break

    if first_mask_shape is None:
        H, W = expected_H or 480, expected_W or 640
        print(f"[WARNING] Could not infer mask shape, using default: ({H}, {W})")
    else:
        H, W = first_mask_shape
        print(f"[INFO] Inferred mask shape: ({H}, {W})")

    if expected_H is not None and expected_W is not None:
        H, W = expected_H, expected_W
        print(f"[INFO] Using expected mask shape: ({H}, {W})")

    with h5py.File(h5_path, 'w') as f:
        ds = f.create_dataset(
            dataset_name,
            shape=(num_frames, num_cams, H, W),
            dtype=np.uint8,
            chunks=(1, 1, H, W),
            compression="gzip",
        )
        for frame_idx in range(num_frames):
            original_frame_idx = frame_idx + start_frame
            for cam_idx in range(num_cams):
                npy_path = None
                if cam_idx < len(camera_folders):
                    npy_path = _find_mask_file(camera_folders[cam_idx], original_frame_idx)
                if npy_path is None:
                    ds[frame_idx, cam_idx] = np.zeros((H, W), dtype=np.uint8)
                    continue
                mask = _load_mask_file(npy_path)
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                elif mask.ndim != 2:
                    print(f"[WARNING] Unexpected mask shape {mask.shape} from {npy_path}, using zeros")
                    ds[frame_idx, cam_idx] = np.zeros((H, W), dtype=np.uint8)
                    continue
                if mask.shape[0] != H or mask.shape[1] != W:
                    try:
                        import cv2
                        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    except ImportError:
                        from scipy.ndimage import zoom
                        mask = zoom(mask, (H / mask.shape[0], W / mask.shape[1]), order=0).astype(mask.dtype)
                ds[frame_idx, cam_idx] = mask.astype(np.uint8)
            if (frame_idx + 1) % 200 == 0 or frame_idx == num_frames - 1:
                print(f"  streamed {frame_idx + 1}/{num_frames} frames")
    print(f"[INFO] Saved {dataset_name} to {h5_path} (streamed)")

def detect_num_objects_from_masks(mask_root_dir, num_cams, start_frame=0):
    """
    Auto-detect number of objects by scanning the first few mask frames for
    the max label value. Tries per-frame npy/npz under cam*_rgb/ first; if
    none found, falls back to a quick read of `masks.h5` in the same folder
    (this lets the npz-less, h5-only export path still report a sensible
    object count).
    Returns the number of distinct objects (max label value).
    """
    mask_root_dir = Path(mask_root_dir)
    max_label = 0

    discovered = discover_camera_folders(mask_root_dir)
    camera_folders = [folder for _, folder in discovered[:num_cams]] if discovered else []
    found_any_per_frame = False
    for cam_folder in camera_folders:
        for frame_offset in range(min(5, 9999)):
            frame_idx = start_frame + frame_offset
            p = _find_mask_file(cam_folder, frame_idx)
            if p is not None:
                mask = _load_mask_file(p)
                max_label = max(max_label, int(mask.max()))
                found_any_per_frame = True

    # h5 fallback when no per-frame files exist (h5-only export path).
    if not found_any_per_frame:
        masks_h5 = mask_root_dir / "masks.h5"
        if masks_h5.exists():
            try:
                with h5py.File(masks_h5, "r") as f:
                    ds = f.get("masks")
                    if ds is not None and ds.shape[0] > 0:
                        # Read first 5 frames (one slice; cheap with chunked storage)
                        n = min(5, ds.shape[0])
                        sample = ds[:n]
                        max_label = max(max_label, int(sample.max()))
            except Exception as e:
                print(f"[WARNING] failed to read {masks_h5} for object detection: {e}")
    return max_label


def load_object_names_from_yaml(mask_root_dir):
    """
    Read objects.yaml from the mask directory if it exists.
    Returns list of tool names, or None if not found.
    """
    objects_yaml = Path(mask_root_dir) / "objects.yaml"
    if objects_yaml.exists():
        with open(objects_yaml, 'r') as f:
            data = yaml.safe_load(f)
        if data and "objects" in data:
            print(f"[INFO] Loaded object names from {objects_yaml}: {data['objects']}")
            return data["objects"]
    return None


def slice_masks_h5(source_path, dest_path, start_frame, num_frames, num_cams, expected_H, expected_W, dataset_name="masks"):
    """Slice [start_frame, start_frame+num_frames) out of an existing full
    masks.h5 (or whatever is at `source_path`) and write it to `dest_path`.

    Bypasses the per-frame npz-streaming path of stream_masks_folder_to_h5 —
    use this when the npz files have been deleted (or never existed) but a
    full masks.h5 covering the absolute frame range is on hand.

    The source dataset's last 3 dims must match (num_cams, H, W); the leading
    axis is sliced. If source != dest, a fresh chunked+compressed h5 is
    written. If source == dest and start_frame == 0 + num_frames == full
    length, this is a no-op (we leave the file alone).
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    if not source_path.exists():
        raise FileNotFoundError(f"masks_h5_source does not exist: {source_path}")
    with h5py.File(source_path, "r") as fin:
        ds_in = fin[dataset_name]
        src_N, src_C = ds_in.shape[0], ds_in.shape[1]
        src_H, src_W = ds_in.shape[2], ds_in.shape[3]
        end = start_frame + num_frames
        if end > src_N:
            raise ValueError(
                f"masks_h5_source has {src_N} frames; cannot slice [{start_frame}, {end})"
            )
        if src_C != num_cams:
            print(f"[WARN] masks_h5_source has {src_C} cams but h5 has {num_cams}; "
                  f"taking the first {num_cams}")
        if (src_H, src_W) != (expected_H, expected_W):
            print(f"[WARN] masks_h5_source shape ({src_H}, {src_W}) != expected "
                  f"({expected_H}, {expected_W}); will keep source shape")
        # No-op short-circuit when source IS the destination and slice covers everything.
        if source_path.resolve() == dest_path.resolve() and start_frame == 0 and num_frames == src_N:
            print(f"[INFO] masks_h5_source == dest and slice == full; leaving {dest_path} untouched")
            return
        # Read the slice (chunked dataset reads cheaply per-frame).
        sliced = ds_in[start_frame:end, :num_cams]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dest_path, "w") as fout:
        fout.create_dataset(
            dataset_name,
            data=sliced,
            chunks=(1, 1, sliced.shape[2], sliced.shape[3]),
            compression="gzip",
        )
    print(f"[INFO] Sliced masks {start_frame}..{end} from {source_path} -> {dest_path}  shape={sliced.shape}")


def generate_meta_yaml(h5_path, mask_root_dir, calibration_yaml_path, output_root, subject_id="subject_5", tool_name="blue_scooper", models_folder="models", object_mask_dir=None, start_frame=0, thresholds=None, masks_h5_source=None, no_mask=False):
    """
    Generate meta.yaml for a HO-Cap dataset sequence. Also saves masks as h5 files in their respective directories.
    Auto-detects number of objects from mask labels and reads objects.yaml for tool names.
    Args:
        h5_path (str): Path to the .h5 file containing imgs and depths.
        mask_root_dir (str): Path to the tool_masks folder.
        calibration_yaml_path (str): Path to the original calibration YAML file.
        output_root (str): Output root directory for meta.yaml.
        subject_id (str): Subject ID.
        tool_name (str): Name of the tool/object (used as fallback if objects.yaml not found).
        models_folder (str): Path to the models folder.
        object_mask_dir (str or None): Path to object_masks folder (optional).
        masks_h5_source (str or None): If given, slice that pre-existing masks.h5
            instead of streaming from per-frame npz under mask_root_dir. Skips
            the npz read entirely (lets the pipeline run when the only mask
            artifact on disk is a full masks.h5).
    """
    # Read H5 metadata (shape only — never load imgs into RAM, that's ~40+ GB
    # for long videos at 720p×8 cams).
    with h5py.File(h5_path, 'r') as f:
        imgs_shape = f["imgs"].shape
    num_frames, num_cams = imgs_shape[0], imgs_shape[1]
    img_H, img_W = imgs_shape[2], imgs_shape[3]

    masks_h5_path = Path(mask_root_dir) / "masks.h5"
    if no_mask:
        # Hand-only path: hand reconstruction doesn't read tool masks. Skip
        # the entire mask discovery/streaming step. Downstream object-pose
        # stages would still fail without masks, but the caller has opted in
        # by passing --no_mask, so that's intentional.
        print(f"[INFO] --no_mask: skipping mask discovery & masks.h5 building")
    elif masks_h5_source:
        # Slice from the supplied full masks.h5 — no npz reads.
        slice_masks_h5(
            source_path=masks_h5_source,
            dest_path=masks_h5_path,
            start_frame=start_frame,
            num_frames=num_frames,
            num_cams=num_cams,
            expected_H=img_H,
            expected_W=img_W,
        )
    else:
        # Auto-fallback: if there are NO per-frame npz/npy under cam*_rgb/
        # (which is the case when masks.h5 was produced directly by
        # batch_task_annotator.py / DINO --pipeline_mask_format=h5), but a
        # masks.h5 already sits at the canonical destination, treat THAT as
        # the source instead of streaming a folder of zeros over it. Without
        # this guard, stream_masks_folder_to_h5 silently overwrites a good
        # masks.h5 with all-zero data and downstream fd_pose_solver sees
        # empty masks (frame 0 sum=0 → cam skipped → pose all -1).
        cam_dirs = discover_camera_folders(Path(mask_root_dir))
        any_per_frame = any(
            (folder / f"{start_frame:04d}.npz").exists() or
            (folder / f"{start_frame:04d}.npy").exists() or
            (folder / f"{start_frame}.npz").exists() or
            (folder / f"{start_frame}.npy").exists()
            for _, folder in cam_dirs
        )
        if not any_per_frame and masks_h5_path.exists():
            print(f"[INFO] No per-frame npz/npy under {mask_root_dir}; "
                  f"keeping existing {masks_h5_path} (no overwrite).")
            # Verify shape matches the data h5 — if not, warn loudly.
            try:
                with h5py.File(masks_h5_path, "r") as f:
                    ds = f.get("masks")
                    if ds is None:
                        print(f"[WARN] {masks_h5_path} exists but has no "
                              f"'masks' dataset; downstream will probably fail")
                    elif ds.shape[:2] != (num_frames, num_cams):
                        print(f"[WARN] existing masks.h5 shape {ds.shape} "
                              f"!= h5 imgs ({num_frames}, {num_cams}, ...); "
                              f"frame counts don't match — run with explicit "
                              f"--masks_h5_source to slice/realign.")
            except Exception as e:
                print(f"[WARN] could not verify existing masks.h5: {e}")
        elif not any_per_frame and not masks_h5_path.exists():
            # No mask source at all. Refuse to write a (N, n_cams, H, W)
            # all-zero masks.h5 — that's what bit the user before
            # (Stage 2 silently produced empty masks → Stage 4 fd_pose_solver
            # 'Frame 0: Invalid mask' → all -1 poses).
            raise FileNotFoundError(
                "No mask source found.\n"
                f"  mask_root_dir = {mask_root_dir}\n"
                f"  Looked for per-frame npz/npy under cam*_rgb/ AND for an "
                f"existing masks.h5 — neither present.\n"
                "  Fix one of:\n"
                "    (a) run mesh_reconstruction/sam2/notebooks/batch_task_annotator.py\n"
                "        on this exp first (writes tool_masks/masks.h5);\n"
                "    (b) run DINO+SAM2 via run_auto_annotator.sh which auto-\n"
                "        produces tool_masks/masks.h5;\n"
                "    (c) pass --masks_h5_source /path/to/existing/masks.h5 to\n"
                "        slice from another sequence."
            )
        else:
            # Stream per-frame masks straight into masks.h5 instead of building a
            # giant in-RAM (N, n_cams, H, W) array (~14 GB for 1923 frames @720p×8).
            stream_masks_folder_to_h5(
                mask_root_dir, masks_h5_path, num_frames, num_cams,
                expected_H=img_H, expected_W=img_W, start_frame=start_frame)

    # Save object masks if provided
    if no_mask:
        # Skip every block that touches the (now nonexistent) mask root.
        object_mask_dir = None
    if object_mask_dir is not None:
        object_masks = load_masks_from_folder(object_mask_dir, num_frames, num_cams, expected_H=img_H, expected_W=img_W, start_frame=start_frame)
        object_masks_h5_path = Path(object_mask_dir) / "object_masks.h5"
        save_masks_to_h5(object_masks, object_masks_h5_path, dataset_name="object_masks")

    # Auto-detect number of objects from mask labels
    if no_mask:
        # Without masks we have nothing to count. Pretend single-object.
        num_objects_detected = 1
        print(f"[INFO] --no_mask: forcing num_objects_detected=1 (hand-only)")
    else:
        num_objects_detected = detect_num_objects_from_masks(mask_root_dir, num_cams, start_frame)
        print(f"[INFO] Detected {num_objects_detected} object(s) from mask labels")

    # Determine object_ids list
    # Priority: objects.yaml > auto-detect count with tool_name fallback
    object_names = (None if no_mask
                     else load_object_names_from_yaml(mask_root_dir))
    if object_names is not None:
        # Validate count matches detection
        if num_objects_detected > 0 and len(object_names) != num_objects_detected:
            print(f"[WARNING] objects.yaml has {len(object_names)} names but masks have {num_objects_detected} labels. Using objects.yaml.")
        object_ids = object_names
    elif num_objects_detected <= 1:
        # Single object (or no objects detected): use tool_name as before
        object_ids = [tool_name]
    else:
        # Multiple objects detected but no objects.yaml: use tool_name for first, generic for rest
        print(f"[WARNING] Multiple objects detected ({num_objects_detected}) but no objects.yaml found. "
              f"Using '{tool_name}' for first object and generic names for the rest.")
        object_ids = [tool_name] + [f"object_{i+1}" for i in range(1, num_objects_detected)]

    print(f"[INFO] object_ids: {object_ids}")

    # Get camera serials from calibration YAML
    with open(calibration_yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    calib_serials = [str(cam['camera_id']).zfill(2) for cam in calib_data]

    # Prefer actual camera folder names from mask directory (e.g. cam8_rgb -> '08')
    if no_mask:
        discovered = []
    else:
        discovered = discover_camera_folders(mask_root_dir)
    discovered_serials = [str(cam_id).zfill(2) for cam_id, _ in discovered]
    if len(discovered_serials) >= num_cams:
        cam_serials = discovered_serials[:num_cams]
        print(f"[INFO] Using camera serials from actual mask folders: {cam_serials}")
    else:
        cam_serials = calib_serials[:num_cams]
        if len(discovered_serials) > 0:
            print(f"[WARNING] Discovered {len(discovered_serials)} camera folder(s) but h5 has {num_cams}. Fallback to calibration serials: {cam_serials}")
        else:
            print(f"[WARNING] No camera folders discovered under masks. Using calibration serials: {cam_serials}")
    # Width/height already pulled from imgs_shape above (img_W / img_H).
    width = img_W
    height = img_H

    # Output folder
    output_folder = Path(output_root)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Compose meta.yaml
    meta = {
        "num_frames": int(num_frames),
        "start_frame": int(start_frame),
        "object_ids": object_ids,
        "mano_sides": ['left', 'right'],
        "subject_id": subject_id,
        "realsense": {
            "serials": cam_serials,
            "width": width,
            "height": height
        },
        "hololens": {
            "serial": "hololens_kv5h72",
            "pv_height": 720,
            "pv_width": 1280
        },
        "have_hololens": False,
        "have_mano": True,
        "task_id": 1,
        "thresholds": thresholds if thresholds is not None else [-0.5, 0.35, -0.5, 0.4, -0.3, 0.4],
        "calibration_yaml_path": calibration_yaml_path,
        "models_folder": models_folder,
        "betas": [
            0.051946,
            0.023095,
            0.13714,
            0.039837,
            0.054446,
            0.03033,
            0.041728,
            0.006936,
            0.022853,
            0.010556
        ]
    }

    with open(output_folder / "meta.yaml", "w") as f:
        yaml.dump(meta, f)
    print(f"[INFO] meta.yaml written to {output_folder / 'meta.yaml'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate meta.yaml for HO-Cap dataset.")
    parser.add_argument('--h5_path', type=str, required=True, help='Path to the .h5 file (e.g. /data/folder_name/task_name/sequence_name/data00000000.h5)')
    parser.add_argument('--calibration_yaml_path', type=str, default='/path/to/calibration.yaml', help='Path to calibration YAML (fixed)')
    parser.add_argument('--models_folder', type=str, default='/path/to/models', help='Path to models folder (fixed)')
    parser.add_argument('--subject_id', type=str, default=None, help='Subject ID (default: sequence_name)')
    parser.add_argument('--tool_name', type=str, default=None, help='Tool/object name (default: sequence_name)')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame index in original video (saved to meta.yaml for mask offset)')
    parser.add_argument('--x_min', type=float, default=-0.5, help='Threshold x min (default: -0.5)')
    parser.add_argument('--x_max', type=float, default=0.35, help='Threshold x max (default: 0.35)')
    parser.add_argument('--y_min', type=float, default=-0.5, help='Threshold y min (default: -0.5)')
    parser.add_argument('--y_max', type=float, default=0.4, help='Threshold y max (default: 0.4)')
    parser.add_argument('--z_min', type=float, default=-0.3, help='Threshold z min (default: -0.3)')
    parser.add_argument('--z_max', type=float, default=0.4, help='Threshold z max (default: 0.4)')
    parser.add_argument('--masks_h5_source', type=str, default=None,
        help='If given, slice [start_frame, start_frame+num_frames) out of '
             'this pre-existing masks.h5 instead of streaming from per-frame '
             'npz under tool_masks/cam*_rgb/. Lets the pipeline reuse a full '
             'masks.h5 from a previous DINO run (or external source) without '
             'requiring the npz checkpoint files.')
    parser.add_argument('--no_mask', action='store_true',
        help='Hand-only mode: skip all mask discovery / masks.h5 building. '
             'Use this when the only downstream stage is hand reconstruction '
             '(which does NOT read tool_masks/masks.h5). object_ids will be '
             'set to [tool_name]; downstream object-pose stages will fail '
             'and that is the intended behavior.')
    args = parser.parse_args()

    # Infer folder_name, task_name, and sequence_name from h5_path
    h5_path = Path(args.h5_path)
    # Updated path parsing for new structure: /.../{folder_name}/{task_name}/{sequence_name}/data00000000.h5
    sequence_name = h5_path.parts[-2]  # xxxvideoname
    task_name = h5_path.parts[-3]      # taskname
    folder_name = h5_path.parts[-4]    # videos_0901
    
    print(f"[INFO] Parsed paths - folder_name: {folder_name}, task_name: {task_name}, sequence_name: {sequence_name}")

    # Infer mask_root_dir and object_mask_dir with taskname included
    # Structure: .../videos_0901_annotated/taskname/xxxvideoname/
    mask_root_dir = h5_path.parent.parent.parent.parent / f"{folder_name}_annotated" / task_name / sequence_name / "tool_masks"
    object_mask_dir = h5_path.parent.parent.parent.parent / f"{folder_name}_annotated" / task_name / sequence_name / "object_masks"
    if not mask_root_dir.exists():
        mask_root_dir = h5_path.parent.parent.parent.parent / f"{folder_name}_annotated" / task_name / sequence_name / "masks"
    if not object_mask_dir.exists():
        object_mask_dir = None

    # Infer output_root
    output_root = h5_path.parent

    # Subject and tool name
    subject_id = args.subject_id if args.subject_id is not None else sequence_name
    tool_name = args.tool_name if args.tool_name is not None else sequence_name

    generate_meta_yaml(
        str(h5_path),
        str(mask_root_dir),
        args.calibration_yaml_path,
        str(output_root),
        subject_id=subject_id,
        tool_name=tool_name,
        models_folder=args.models_folder,
        object_mask_dir=str(object_mask_dir) if object_mask_dir is not None else None,
        start_frame=args.start_frame,
        thresholds=[args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max],
        masks_h5_source=args.masks_h5_source,
        no_mask=args.no_mask,
    )
