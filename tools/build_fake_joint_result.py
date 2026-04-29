"""Assemble a fake joint_pose_solver result from existing object + hand outputs.

Skips the expensive Stage 6 (object_pose_solver) and Stage 7 (joint_pose_solver)
optimizations and instead writes the canonical files those stages would produce,
using:
  - processed/fd_pose_solver/fd_poses_merged_fixed.npy   -> poses_o.npy
  - result_hand_optimized.pkl                            -> poses_m.npy + pkl copy

Layout produced (mirrors 07-2_joint_pose_solver_cluster.py):
  <annotated>/processed/joint_pose_solver/
    poses_o.npy                  # (num_obj, num_frames, 7) quat+trans
    poses_m.npy                  # (2, num_frames, 51) [left, right]; missing side filled with -1
    result_hand_optimized.pkl    # copy of <annotated>/result_hand_optimized.pkl
    fake_joint_meta.json         # provenance: source files + is_fake flag

Downstream scripts that read these three files (rendering, holo align, etc.)
will not need any changes; the fake_joint_meta.json is the only marker that
this run skipped the optimizers.
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


def _resolve_paths(sequence_folder: Path):
    base = sequence_folder.parent.parent.parent
    folder_name = sequence_folder.parent.parent.name
    task_name = sequence_folder.parent.name
    seq_name = sequence_folder.name
    annotated = base / f"{folder_name}_annotated" / task_name / seq_name
    fd_merged = annotated / "processed" / "fd_pose_solver" / "fd_poses_merged_fixed.npy"
    hand_pkl = annotated / "result_hand_optimized.pkl"
    out_dir = annotated / "processed" / "joint_pose_solver"
    return annotated, fd_merged, hand_pkl, out_dir


def _side_pose_51(hand_data: dict, side: str) -> np.ndarray | None:
    """Build a (N, 51) hand pose array for one side from the cluster_optimize_hand
    pkl, which stores pose (axis-angle, 48) and translation (3) separately.

    The 51-dim layout matches MANOGroupLayer's expectation
    (mano_group_layer.py: pose[..., :48] = pose, pose[..., 48:51] = translation).
    Returns None if the side is missing/empty.
    """
    pose = np.asarray(hand_data.get(f"{side}_hand_pose", []), dtype=np.float32)
    trans = np.asarray(hand_data.get(f"{side}_hand_translation", []), dtype=np.float32)
    if pose.size == 0 or trans.size == 0:
        return None
    if pose.ndim != 2 or pose.shape[1] != 48:
        raise ValueError(f"{side}_hand_pose has unexpected shape {pose.shape}; expected (N, 48)")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"{side}_hand_translation has unexpected shape {trans.shape}; expected (N, 3)")
    if pose.shape[0] != trans.shape[0]:
        raise ValueError(
            f"{side}_hand_pose / translation frame count mismatch: {pose.shape[0]} vs {trans.shape[0]}"
        )
    return np.concatenate([pose, trans], axis=1).astype(np.float32)  # (N, 51)


def _build_poses_m(hand_data: dict, num_frames_obj: int) -> np.ndarray:
    left = _side_pose_51(hand_data, "left")
    right = _side_pose_51(hand_data, "right")
    if left is None and right is None:
        raise ValueError(
            "result_hand_optimized.pkl has neither left_hand_pose nor right_hand_pose"
        )

    nf = (left if left is not None else right).shape[0]
    if left is not None and right is not None and left.shape[0] != right.shape[0]:
        raise ValueError(
            f"left/right hand pose frame count mismatch: {left.shape[0]} vs {right.shape[0]}"
        )
    filler = np.full((nf, 51), -1.0, dtype=np.float32)
    poses_m = np.stack(
        [left if left is not None else filler, right if right is not None else filler], axis=0
    )

    if num_frames_obj != nf:
        # Truncate to the shorter length so poses_o / poses_m line up frame-by-frame.
        # This mirrors how the real solvers operate on a single num_frames.
        n = min(num_frames_obj, nf)
        poses_m = poses_m[:, :n]
    return poses_m


def main():
    parser = argparse.ArgumentParser(description="Build fake joint_pose_solver result")
    parser.add_argument("--sequence_folder", required=True, help="Path to the sequence folder")
    parser.add_argument("--object_idx", type=int, default=None,
                        help="If set, keep only this object index from fd_poses_merged_fixed (0-based)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing fake outputs")
    args = parser.parse_args()

    sequence_folder = Path(args.sequence_folder).resolve()
    annotated, fd_merged, hand_pkl, out_dir = _resolve_paths(sequence_folder)

    if not fd_merged.exists():
        raise FileNotFoundError(f"missing fd merge: {fd_merged}")
    if not hand_pkl.exists():
        raise FileNotFoundError(f"missing hand pkl: {hand_pkl}")

    out_dir.mkdir(parents=True, exist_ok=True)
    poses_o_out = out_dir / "poses_o.npy"
    poses_m_out = out_dir / "poses_m.npy"
    pkl_out = out_dir / "result_hand_optimized.pkl"
    meta_out = out_dir / "fake_joint_meta.json"

    if not args.overwrite and poses_o_out.exists() and poses_m_out.exists() and pkl_out.exists():
        print(f"[fake-joint] outputs already exist in {out_dir}; pass --overwrite to rebuild")
        return

    poses_o = np.load(fd_merged).astype(np.float32)
    if poses_o.ndim == 2:
        poses_o = poses_o[None]
    if poses_o.ndim != 3 or poses_o.shape[-1] != 7:
        raise ValueError(f"fd_poses_merged_fixed.npy unexpected shape {poses_o.shape}; expected (num_obj, num_frames, 7)")
    if args.object_idx is not None:
        if not (0 <= args.object_idx < poses_o.shape[0]):
            raise IndexError(f"--object_idx {args.object_idx} out of range for {poses_o.shape[0]} objects")
        poses_o = poses_o[args.object_idx : args.object_idx + 1]

    with open(hand_pkl, "rb") as f:
        hand_data_full = pickle.load(f)
    hand_data = hand_data_full.get("hand_pose", {})
    poses_m = _build_poses_m(hand_data, poses_o.shape[1])

    # Frame-count alignment: if hand was shorter, also truncate poses_o.
    if poses_m.shape[1] != poses_o.shape[1]:
        n = min(poses_m.shape[1], poses_o.shape[1])
        poses_o = poses_o[:, :n]
        poses_m = poses_m[:, :n]

    np.save(poses_o_out, poses_o)
    np.save(poses_m_out, poses_m)
    shutil.copy2(hand_pkl, pkl_out)

    meta = {
        "is_fake": True,
        "note": "joint_pose_solver SKIPPED — outputs assembled from fd merge + optimized hand",
        "sources": {
            "poses_o_from": str(fd_merged),
            "hand_pkl_from": str(hand_pkl),
        },
        "shapes": {
            "poses_o": list(poses_o.shape),
            "poses_m": list(poses_m.shape),
        },
        "object_idx_kept": args.object_idx,
    }
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[fake-joint] wrote:")
    print(f"  {poses_o_out}  shape={poses_o.shape}")
    print(f"  {poses_m_out}  shape={poses_m.shape}")
    print(f"  {pkl_out}")
    print(f"  {meta_out}")


if __name__ == "__main__":
    main()
