#!/usr/bin/env bash
# Fetch one or more exp folders from the SCDT server into the local
# videos_dataset / videos_dataset_annotated workspace.
#
# Usage:
#   bash scripts/fetch_exp_from_server.sh \
#       /viscam/projects/robotool/data/videos_0213_annotated/fork_flip_egg/20260214_bigwoodenfork_flip_egg_in_blackpan_105 \
#       [more remote annotated exp paths ...]
#
# For each remote annotated exp path "<root>/videos_<DATE>_annotated/<task>/<exp>"
# this script copies:
#   1. The full original sequence folder
#        <root>/videos_<DATE>/<task>/<exp>
#      → $LOCAL_ORIG_BASE/<task>/<exp>            (default: data/videos_dataset)
#   2. <annotated>/result_hand_optimized.pkl
#      → $LOCAL_ANN_BASE/<task>/<exp>/result_hand_optimized.pkl
#   3. <annotated>/processed/fd_pose_solver/fd_poses_merged_fixed.npy
#      → $LOCAL_ANN_BASE/<task>/<exp>/processed/joint_pose_solver/poses_o.npy
#      (note the rename — locally we treat the FD-merged pose AS poses_o,
#       same trick as tools/build_fake_joint_result.py)
#
# Env overrides:
#   REMOTE_USER (default: chenrq)
#   REMOTE_HOST (default: scdt.stanford.edu)
#   LOCAL_ORIG_BASE (default: /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/videos_dataset)
#   LOCAL_ANN_BASE  (default: /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/videos_dataset_annotated)
#   FORCE=1   (re-copy even if local target already exists)

set -euo pipefail

REMOTE_USER="${REMOTE_USER:-chenrq}"
REMOTE_HOST="${REMOTE_HOST:-scdt.stanford.edu}"
LOCAL_ORIG_BASE="${LOCAL_ORIG_BASE:-/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/videos_dataset}"
LOCAL_ANN_BASE="${LOCAL_ANN_BASE:-/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/videos_dataset_annotated}"
FORCE="${FORCE:-0}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <remote_annotated_exp_path> [more ...]"
    echo "  e.g. /viscam/projects/robotool/data/videos_0213_annotated/fork_flip_egg/20260214_bigwoodenfork_flip_egg_in_blackpan_105"
    exit 1
fi

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

n_ok=0
n_fail=0
declare -a failed=()

for remote_ann in "$@"; do
    echo
    echo "============================================================"
    echo "remote: $remote_ann"
    echo "============================================================"

    # Validate shape: must contain "_annotated/<task>/<exp>" tail.
    if [[ "$remote_ann" != *_annotated/*/* ]]; then
        echo "[ERR] expected path of the form .../videos_<DATE>_annotated/<task>/<exp>"
        n_fail=$((n_fail+1)); failed+=("$remote_ann"); continue
    fi

    # Strip trailing slash for clean parsing.
    remote_ann="${remote_ann%/}"

    exp_name="$(basename "$remote_ann")"
    task_dir="$(dirname "$remote_ann")"           # .../videos_<DATE>_annotated/<task>
    task_name="$(basename "$task_dir")"
    ann_root="$(dirname "$task_dir")"             # .../videos_<DATE>_annotated
    parent_root="$(dirname "$ann_root")"          # .../data
    ann_root_base="$(basename "$ann_root")"       # videos_<DATE>_annotated
    orig_root_base="${ann_root_base%_annotated}"  # videos_<DATE>
    if [[ "$orig_root_base" == "$ann_root_base" ]]; then
        echo "[ERR] could not strip _annotated from $ann_root_base"
        n_fail=$((n_fail+1)); failed+=("$remote_ann"); continue
    fi
    remote_orig_exp="${parent_root}/${orig_root_base}/${task_name}/${exp_name}"

    local_orig_task="${LOCAL_ORIG_BASE}/${task_name}"
    local_orig_exp="${local_orig_task}/${exp_name}"
    local_ann_exp="${LOCAL_ANN_BASE}/${task_name}/${exp_name}"
    local_ann_jps="${local_ann_exp}/processed/joint_pose_solver"

    echo "  task=${task_name}  exp=${exp_name}"
    echo "  remote orig: ${remote_orig_exp}"
    echo "  remote ann : ${remote_ann}"
    echo "  local  orig: ${local_orig_exp}"
    echo "  local  ann : ${local_ann_exp}"

    failed_this=0

    # ---- (1) entire original exp folder ----
    if [[ -d "$local_orig_exp" && "$FORCE" != "1" ]]; then
        echo "  [skip-1/3] original exp dir already exists ($local_orig_exp); pass FORCE=1 to re-copy"
    else
        mkdir -p "$local_orig_task"
        echo "  [1/3] scp -r ${REMOTE}:${remote_orig_exp}  →  ${local_orig_task}/"
        if ! scp -r "${REMOTE}:${remote_orig_exp}" "${local_orig_task}/"; then
            echo "  [ERR] scp original exp failed"
            failed_this=1
        fi
    fi

    # ---- (2) result_hand_optimized.pkl ----
    mkdir -p "$local_ann_exp"
    dst_pkl="${local_ann_exp}/result_hand_optimized.pkl"
    if [[ -f "$dst_pkl" && "$FORCE" != "1" ]]; then
        echo "  [skip-2/3] $dst_pkl exists; pass FORCE=1 to re-copy"
    else
        echo "  [2/3] scp result_hand_optimized.pkl  →  $dst_pkl"
        if ! scp "${REMOTE}:${remote_ann}/result_hand_optimized.pkl" "$dst_pkl"; then
            echo "  [ERR] scp result_hand_optimized.pkl failed"
            failed_this=1
        fi
    fi

    # ---- (3) fd_poses_merged_fixed.npy → joint_pose_solver/poses_o.npy ----
    mkdir -p "$local_ann_jps"
    dst_npy="${local_ann_jps}/poses_o.npy"
    if [[ -f "$dst_npy" && "$FORCE" != "1" ]]; then
        echo "  [skip-3/3] $dst_npy exists; pass FORCE=1 to re-copy"
    else
        echo "  [3/3] scp fd_poses_merged_fixed.npy  →  $dst_npy"
        if ! scp "${REMOTE}:${remote_ann}/processed/fd_pose_solver/fd_poses_merged_fixed.npy" "$dst_npy"; then
            echo "  [ERR] scp fd_poses_merged_fixed.npy failed"
            failed_this=1
        fi
    fi

    if [[ $failed_this -eq 0 ]]; then
        n_ok=$((n_ok+1))
        echo "  [OK] ${task_name}/${exp_name}"
    else
        n_fail=$((n_fail+1)); failed+=("$remote_ann")
        echo "  [FAIL] ${task_name}/${exp_name}"
    fi
done

echo
echo "============================================================"
echo "summary: ok=${n_ok}  fail=${n_fail}"
if [[ $n_fail -gt 0 ]]; then
    echo "failed:"
    for p in "${failed[@]}"; do echo "  $p"; done
    exit 1
fi
echo "============================================================"
