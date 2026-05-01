#!/bin/bash
#
# Reclaim disk + inodes from exps whose final annotation outputs already
# exist. Walks an annotated_root (or a single task / exp), checks each
# experiment's "done" status, and deletes the intermediate artifacts that
# downstream stages no longer read:
#
#   processed/fd_pose_solver/<obj>/ob_in_cam/<serial>/{abs:06d}.txt
#       → per-camera tracking txts. Consumed by the merger; once
#         fd_poses_merged_fixed.npy is on disk they're redundant
#         (~1864 frames × 8 cams ≈ 15k tiny files per object).
#   processed/fd_pose_solver/<obj>/ob_in_world/{abs:06d}.txt
#       → per-frame world-frame txts. Consumed by the merger. Same as above.
#         Pass --keep_ob_in_world to retain (lets you re-run the merger
#         with different smoothing without redoing tracking).
#   tool_masks/cam*_rgb/{abs:04d}.npz
#       → DINO per-frame masks. Once tool_masks/masks.h5 exists every
#         downstream tool reads from masks.h5; npz become dead weight
#         (~14k npz per exp on a 1.8k-frame×8-cam sequence).
#   dino_auto/
#       → DINO debug output dir (internal masks.h5, viz/, seed_info.json).
#         Drop once tool_masks/masks.h5 is in place; pass --keep_dino_auto
#         to retain for debugging.
#   result_{S}_{E}.pkl, result_hand_optimized_{S}_{E}.pkl, poses_m_{S}_{E}.npy
#       → hand chunked-run intermediates. Already auto-cleaned by
#         run_auto_annotator.sh unless --hand_keep_chunk_files; this script
#         picks up any leftovers if a job died after merge.
#   *.full_backup_obj, *.full_backup_hand, *.gen_meta_src
#       → interrupted-chunked-run residuals. Always safe to delete on a
#         "done" exp.
#
# An exp is considered "done" when:
#   processed/fd_pose_solver/fd_poses_merged_fixed.npy  exists.
# (Plus the per-section gates above — masks.h5 must exist before npz/dino_auto
# get cleaned; result_hand_optimized.pkl must exist before chunked hand
# residuals get cleaned.)
#
# Usage:
#   bash scripts/cleanup_done_exps.sh --annotated_root /viscam/projects/robotool/data/videos_0102_annotated
#   bash scripts/cleanup_done_exps.sh --task_folder    /viscam/.../videos_0102_annotated/chopstick_slice_dough
#   bash scripts/cleanup_done_exps.sh --exp_dir        /viscam/.../videos_0102_annotated/<task>/<exp>
#
#   [--dry_run]                only print what would be deleted, don't touch disk
#   [--keep_ob_in_world]       keep per-frame world txts (for re-run merger)
#   [--keep_dino_auto]         keep dino_auto/ debug dirs
#   [--quiet]                  suppress per-exp lines (only print summary)

set -u

DRY_RUN=0
KEEP_OB_IN_WORLD=0
KEEP_DINO_AUTO=0
QUIET=0
ANNROOT=""
TASK_FOLDER=""
EXP_DIRS=()

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --annotated_root)   ANNROOT="$2"; shift 2;;
        --task_folder)      TASK_FOLDER="$2"; shift 2;;
        --exp_dir)          EXP_DIRS+=("$2"); shift 2;;
        --dry_run)          DRY_RUN=1; shift;;
        --keep_ob_in_world) KEEP_OB_IN_WORLD=1; shift;;
        --keep_dino_auto)   KEEP_DINO_AUTO=1; shift;;
        --quiet)            QUIET=1; shift;;
        -h|--help)          sed -n '2,55p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

# ---------- enumerate target exp dirs ----------
TARGETS=()
if [[ -n "$ANNROOT" ]]; then
    [[ -d "$ANNROOT" ]] || { echo "Error: --annotated_root not a dir: $ANNROOT"; exit 1; }
    # videos_X_annotated/<task>/<exp> — exactly 2 levels deep
    while IFS= read -r -d '' d; do
        TARGETS+=("$d")
    done < <(find "$ANNROOT" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z)
elif [[ -n "$TASK_FOLDER" ]]; then
    [[ -d "$TASK_FOLDER" ]] || { echo "Error: --task_folder not a dir: $TASK_FOLDER"; exit 1; }
    while IFS= read -r -d '' d; do
        TARGETS+=("$d")
    done < <(find "$TASK_FOLDER" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi
TARGETS+=("${EXP_DIRS[@]}")

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
    echo "Error: no exp dirs to scan. Pass one of --annotated_root / --task_folder / --exp_dir"
    exit 1
fi

[[ "$DRY_RUN" == "1" ]] && echo "[DRY RUN] no files will be deleted"

# ---------- helpers ----------
# Sum bytes + count files under a path (handles both files and dirs).
_du_count() {
    local p="$1"
    if [[ -e "$p" ]]; then
        local b f
        b=$(du -sb "$p" 2>/dev/null | awk '{print $1}')
        if [[ -d "$p" ]]; then
            f=$(find "$p" -type f 2>/dev/null | wc -l)
        else
            f=1
        fi
        echo "$b $f"
    else
        echo "0 0"
    fi
}

_human_bytes() {
    awk -v b="$1" 'BEGIN {
        units="B KB MB GB TB"; split(units, u, " ");
        i=1; while (b>=1024 && i<5) { b/=1024; i++ }
        printf "%.1f %s\n", b, u[i]
    }'
}

_remove() {
    local target="$1"
    if [[ ! -e "$target" ]]; then return 0; fi
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    rm -rf "$target"
}

# ---------- per-exp cleanup ----------
TOTAL_OK=0; TOTAL_NOT_DONE=0; TOTAL_BYTES=0; TOTAL_FILES=0

for EXP_DIR in "${TARGETS[@]}"; do
    [[ -d "$EXP_DIR" ]] || continue
    EXP_LABEL="$(basename "$(dirname "$EXP_DIR")")/$(basename "$EXP_DIR")"

    MERGED="${EXP_DIR}/processed/fd_pose_solver/fd_poses_merged_fixed.npy"
    if [[ ! -f "$MERGED" ]]; then
        TOTAL_NOT_DONE=$((TOTAL_NOT_DONE + 1))
        [[ "$QUIET" == "1" ]] || echo "[skip not-done] $EXP_LABEL"
        continue
    fi

    EXP_BYTES=0; EXP_FILES=0

    # -- 1) per-cam tracking txts (always safe once merger is done) --
    for OBJ_DIR in "$EXP_DIR"/processed/fd_pose_solver/*/ob_in_cam; do
        [[ -d "$OBJ_DIR" ]] || continue
        read b f < <(_du_count "$OBJ_DIR")
        EXP_BYTES=$((EXP_BYTES + b)); EXP_FILES=$((EXP_FILES + f))
        _remove "$OBJ_DIR"
    done

    # -- 2) per-frame world-frame txts (optional via --keep_ob_in_world) --
    if [[ "$KEEP_OB_IN_WORLD" != "1" ]]; then
        for OBJ_DIR in "$EXP_DIR"/processed/fd_pose_solver/*/ob_in_world; do
            [[ -d "$OBJ_DIR" ]] || continue
            read b f < <(_du_count "$OBJ_DIR")
            EXP_BYTES=$((EXP_BYTES + b)); EXP_FILES=$((EXP_FILES + f))
            _remove "$OBJ_DIR"
        done
    fi

    # -- 3) DINO per-frame npz (only when masks.h5 exists at canonical loc) --
    MASKS_H5="${EXP_DIR}/tool_masks/masks.h5"
    if [[ -f "$MASKS_H5" ]]; then
        for CAM_DIR in "$EXP_DIR"/tool_masks/cam*_rgb; do
            [[ -d "$CAM_DIR" ]] || continue
            read b f < <(_du_count "$CAM_DIR")
            EXP_BYTES=$((EXP_BYTES + b)); EXP_FILES=$((EXP_FILES + f))
            _remove "$CAM_DIR"
        done

        # -- 4) dino_auto/ debug dir (only when canonical masks.h5 is in place) --
        if [[ "$KEEP_DINO_AUTO" != "1" && -d "${EXP_DIR}/dino_auto" ]]; then
            read b f < <(_du_count "${EXP_DIR}/dino_auto")
            EXP_BYTES=$((EXP_BYTES + b)); EXP_FILES=$((EXP_FILES + f))
            _remove "${EXP_DIR}/dino_auto"
        fi
    fi

    # -- 5) hand chunked residuals (only when canonical pkl is in place) --
    if [[ -f "${EXP_DIR}/result_hand_optimized.pkl" ]]; then
        # NB: shell glob — ensure files exist before counting
        shopt -s nullglob
        chunk_files=( "${EXP_DIR}"/result_hand_optimized_[0-9]*_[0-9]*.pkl \
                      "${EXP_DIR}"/result_[0-9]*_[0-9]*.pkl \
                      "${EXP_DIR}"/poses_m_[0-9]*_[0-9]*.npy )
        shopt -u nullglob
        for f in "${chunk_files[@]}"; do
            [[ -f "$f" ]] || continue
            read b cnt < <(_du_count "$f")
            EXP_BYTES=$((EXP_BYTES + b)); EXP_FILES=$((EXP_FILES + cnt))
            _remove "$f"
        done
    fi

    # -- 6) backup-file residuals from interrupted chunked runs --
    SEQ_DIR="$(dirname "$(dirname "$(dirname "$EXP_DIR")")")/$(basename "$(dirname "$(dirname "$EXP_DIR")")")"
    SEQ_DIR_ALT="$(echo "$EXP_DIR" | sed 's|_annotated/|/|')"   # heuristic for raw sequence_folder
    shopt -s nullglob
    backup_files=( "${EXP_DIR}"/*.full_backup_obj "${EXP_DIR}"/*.full_backup_hand \
                   "${EXP_DIR}"/tool_masks/*.full_backup_obj "${EXP_DIR}"/tool_masks/*.full_backup_hand \
                   "${EXP_DIR}"/tool_masks/*.gen_meta_src \
                   "${SEQ_DIR_ALT}"/*.full_backup_obj "${SEQ_DIR_ALT}"/*.full_backup_hand )
    shopt -u nullglob
    for f in "${backup_files[@]}"; do
        [[ -f "$f" ]] || continue
        read b cnt < <(_du_count "$f")
        EXP_BYTES=$((EXP_BYTES + b)); EXP_FILES=$((EXP_FILES + cnt))
        _remove "$f"
    done

    TOTAL_OK=$((TOTAL_OK + 1))
    TOTAL_BYTES=$((TOTAL_BYTES + EXP_BYTES))
    TOTAL_FILES=$((TOTAL_FILES + EXP_FILES))

    if [[ "$QUIET" != "1" ]]; then
        printf "[clean] %-90s freed=%-10s files=%d\n" \
            "$EXP_LABEL" "$(_human_bytes $EXP_BYTES)" "$EXP_FILES"
    fi
done

echo ""
echo "=========================================="
echo "summary  : exps_cleaned=$TOTAL_OK  exps_not_done=$TOTAL_NOT_DONE"
echo "freed    : $(_human_bytes $TOTAL_BYTES)  ($TOTAL_FILES files)"
[[ "$DRY_RUN" == "1" ]] && echo "(dry-run, nothing was actually deleted)"
echo "=========================================="
