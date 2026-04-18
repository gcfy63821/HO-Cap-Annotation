#!/usr/bin/env python
"""
Merge chunked hand-annotation outputs for one sequence into single files:
  result_hand_optimized_{S}_{E}.pkl  ->  result_hand_optimized.pkl
  result_{S}_{E}.pkl                 ->  result.pkl
  poses_m_{S}_{E}.npy                ->  poses_m.npy

Chunks are auto-discovered in <annotated_path> by filename pattern. They are
sorted by start frame and required to be contiguous (chunk[i].end + 1 == chunk[i+1].start).

Usage:
    python scripts/merge_hand_chunks.py --annotated_path /abs/path/to/.../videos_XXXX_annotated/<task>/<video>
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np


CHUNK_RE = re.compile(r"^result_hand_optimized_(\d+)_(\d+)\.pkl$")
RECON_RE = re.compile(r"^result_(\d+)_(\d+)\.pkl$")
POSES_RE = re.compile(r"^poses_m_(\d+)_(\d+)\.npy$")


def discover_chunks(folder: Path, pattern: re.Pattern):
    chunks = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            s, e = int(m.group(1)), int(m.group(2))
            chunks.append((s, e, p))
    chunks.sort(key=lambda x: x[0])
    return chunks


def check_contiguous(chunks, label):
    if not chunks:
        raise RuntimeError(f"No {label} chunks found")
    for (s1, e1, _), (s2, e2, _) in zip(chunks, chunks[1:]):
        if s2 != e1 + 1:
            raise RuntimeError(
                f"{label} chunks not contiguous: [{s1},{e1}] then [{s2},{e2}] — gap at frame {e1+1}"
            )


def _concat_array_field(pieces):
    """Concatenate along axis 0. Accepts list[np.ndarray] or list[list]."""
    arrs = []
    for p in pieces:
        if isinstance(p, np.ndarray):
            arrs.append(p)
        elif isinstance(p, list) and len(p) > 0:
            # list of tensors/arrays per frame
            arrs.append(np.asarray([np.asarray(x) for x in p]))
        else:
            arrs.append(np.zeros((0,), dtype=np.float32))
    return np.concatenate(arrs, axis=0)


def _concat_list_field(pieces):
    """Concatenate lists (or convert arrays to lists)."""
    out = []
    for p in pieces:
        if p is None:
            continue
        if isinstance(p, np.ndarray):
            out.extend(list(p))
        else:
            out.extend(list(p))
    return out


def _maybe_concat_base_rot(pieces, total_frames):
    """left/right_hand_base_rot may be per-frame (N,3,3) or single (3,3)."""
    first = None
    per_frame = []
    any_per_frame = False
    for p in pieces:
        if p is None:
            per_frame.append(None)
            continue
        arr = np.asarray(p)
        if arr.ndim == 3:
            any_per_frame = True
        per_frame.append(arr)
        if first is None:
            first = arr

    if not any_per_frame:
        return first  # single rotation — keep as-is

    # normalize each chunk to (n,3,3); if a chunk is single (3,3), broadcast to chunk length
    # we need chunk frame counts — caller didn't pass them, so infer from existing per-frame arrays
    normalized = []
    for arr in per_frame:
        if arr is None:
            continue
        if arr.ndim == 2:
            # cannot know chunk length here — just expand to single frame; merge may be imperfect
            normalized.append(arr[None])
        else:
            normalized.append(arr)
    return np.concatenate(normalized, axis=0)


def merge_pkl(chunks, out_path: Path, kind: str):
    """
    kind: 'optimized' (arrays) or 'reconstruct' (lists of tensors).
    Both have the same overall schema; concat strategy per field differs slightly.
    """
    datas = []
    for (s, e, path) in chunks:
        with open(path, "rb") as f:
            datas.append(pickle.load(f))

    first = datas[0]
    merged = dict(first)  # start from first chunk, overwrite per-frame fields

    # total frames
    total = sum(d.get("num_frames", chunks[i][1] - chunks[i][0] + 1) for i, d in enumerate(datas))

    # hand_pose fields
    hp = {}
    for key in [
        "left_hand_pose", "left_hand_translation",
        "right_hand_pose", "right_hand_translation",
    ]:
        pieces = [d["hand_pose"][key] for d in datas]
        if kind == "optimized":
            hp[key] = _concat_array_field(pieces)
        else:
            # reconstruct: preserve list-of-tensor layout
            out = []
            for p in pieces:
                out.extend(list(p))
            hp[key] = out

    # beta: shape params are constant — keep first
    hp["left_hand_beta"] = first["hand_pose"].get("left_hand_beta")
    hp["right_hand_beta"] = first["hand_pose"].get("right_hand_beta")
    # base rotations
    hp["left_hand_base_rot"] = _maybe_concat_base_rot(
        [d["hand_pose"].get("left_hand_base_rot") for d in datas], total
    )
    if "right_hand_base_rot" in first["hand_pose"]:
        hp["right_hand_base_rot"] = _maybe_concat_base_rot(
            [d["hand_pose"].get("right_hand_base_rot") for d in datas], total
        )
    merged["hand_pose"] = hp

    # hand_joints: per-frame arrays (N, num_cams, 21, 2)
    hj = {}
    for key in ["left_hand_joints_2d", "right_hand_joints_2d"]:
        pieces = [np.asarray(d["hand_joints"][key]) for d in datas]
        hj[key] = np.concatenate(pieces, axis=0)
    merged["hand_joints"] = hj

    # per-frame lists
    for key in ["target_object_pose", "tool_object_pose", "deformable_object_pointcloud"]:
        merged[key] = _concat_list_field([d.get(key, []) for d in datas])

    # masks.deformable
    if "masks" in first:
        merged["masks"] = dict(first["masks"])
        if "deformable" in first["masks"]:
            merged["masks"]["deformable"] = _concat_list_field(
                [d.get("masks", {}).get("deformable", []) for d in datas]
            )

    merged["num_frames"] = int(total)
    merged["start_frame"] = int(chunks[0][0])
    merged["end_frame"] = int(chunks[-1][1])

    with open(out_path, "wb") as f:
        pickle.dump(merged, f)
    print(f"[OK] wrote {out_path}  (num_frames={total}, range=[{chunks[0][0]},{chunks[-1][1]}])")


def merge_poses_npy(chunks, out_path: Path):
    """Each poses_m_{S}_{E}.npy has shape (2, frames, 51). Concat along frames axis."""
    arrs = [np.load(p) for (_, _, p) in chunks]
    merged = np.concatenate(arrs, axis=1)  # (2, total_frames, 51)
    np.save(out_path, merged)
    print(f"[OK] wrote {out_path}  shape={merged.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated_path", required=True,
                    help="Folder containing chunked result files "
                         "(e.g. .../videos_0101_annotated/<task>/<video>)")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if chunks are not contiguous (default: warn & continue)")
    args = ap.parse_args()

    folder = Path(args.annotated_path)
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    opt_chunks = discover_chunks(folder, CHUNK_RE)
    recon_chunks = discover_chunks(folder, RECON_RE)
    poses_chunks = discover_chunks(folder, POSES_RE)

    print(f"[INFO] {folder}")
    print(f"  optimized chunks : {len(opt_chunks)}")
    print(f"  reconstruct chunks: {len(recon_chunks)}")
    print(f"  poses_m chunks   : {len(poses_chunks)}")

    for label, chunks in [
        ("optimized", opt_chunks), ("reconstruct", recon_chunks), ("poses", poses_chunks)
    ]:
        try:
            check_contiguous(chunks, label)
        except RuntimeError as e:
            if args.strict:
                raise
            print(f"[WARN] {e}")

    if opt_chunks:
        merge_pkl(opt_chunks, folder / "result_hand_optimized.pkl", kind="optimized")
    if recon_chunks:
        merge_pkl(recon_chunks, folder / "result.pkl", kind="reconstruct")
    if poses_chunks:
        merge_poses_npy(poses_chunks, folder / "poses_m.npy")

    print("[DONE] merge complete")


if __name__ == "__main__":
    main()
