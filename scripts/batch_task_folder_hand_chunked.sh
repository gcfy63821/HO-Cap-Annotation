#!/bin/bash
# Chunked hand-annotation batch pipeline.
#
# For every experiment folder under --task_folder, split the video into chunks
# of --chunk_size frames (default 500) and run the full hand pipeline per chunk:
#   1) videos -> h5 (only this chunk's frames)
#   2) generate meta (with correct start_frame so masks align by absolute frame)
#   3) cluster_reconstruct   -> writes result.pkl
#   4) cluster_optimize_hand -> writes result_hand_optimized.pkl + poses_m.npy
#   5) rename all 3 outputs to chunk-suffixed names:
#        result_{S}_{E}.pkl
#        result_hand_optimized_{S}_{E}.pkl
#        poses_m_{S}_{E}.npy
#   6) delete the chunk h5 (unless --keep_h5)
#
# Resume-safe at two levels when --skip_existing is passed:
#   1) experiment-level: if <annotated>/result_hand_optimized.pkl (the MERGED
#      final file) already exists, the whole experiment is skipped.
#   2) chunk-level:      otherwise, individual chunks whose
#      result_hand_optimized_{S}_{E}.pkl already exists are skipped.
# To force re-run of a single experiment: delete its result_hand_optimized.pkl.
# To force re-run of a single chunk:      delete that chunk's *_{S}_{E}.pkl.
#
# After all chunks of an experiment finish, merges them into single
#   result.pkl / result_hand_optimized.pkl / poses_m.npy via merge_hand_chunks.py
# and deletes the per-chunk files (unless --keep_chunk_files).
#
# Usage:
#   bash scripts/batch_task_folder_hand_chunked.sh \
#     --task_folder       /abs/path/to/data/videos_0101/mallet_crush_nuts \
#     --calibration_yaml  /abs/path/to/realsense_calibration_xxxx.yaml \
#     [--tool_name mallet_crush_nuts]   # default: basename of task_folder
#     [--chunk_size 500]                # frames per chunk
#     [--start_frame 0] [--end_frame <total-1>]  # overall window; default = full video
#     [--skip_existing] [--keep_h5]
#     [--no_merge]          # skip merge step, leave per-chunk files
#     [--keep_chunk_files]  # merge but keep per-chunk files for debugging
#     [--h5_scratch_dir DIR]
#         Write the per-chunk h5 into DIR (fast local storage: /dev/shm,
#         /scratch, etc.) and symlink it into the sequence folder, so
#         cluster_reconstruct still reads it from $SEQUENCE_FOLDER/data00000000.h5.
#         Avoids hammering /viscam NFS when many jobs run in parallel.
#         Typical value on slurm: /dev/shm/$USER/$SLURM_JOB_ID

set -u

TASK_FOLDER=""
CALIBRATION_YAML=""
TOOL_NAME=""
CHUNK_SIZE=500
START_FRAME=0
END_FRAME=""          # empty = auto-detect from video
SKIP_EXISTING=0
KEEP_H5=0
NO_MERGE=0
KEEP_CHUNK_FILES=0
ONLY_EXP=""           # if set, only process this single exp under TASK_FOLDER
                      # (used by sbatch_run_hand_array.sh for per-exp scheduling)
# If set, build h5 in this dir and symlink it into the sequence folder so
# cluster_reconstruct still finds it at $SEQUENCE_FOLDER/data00000000.h5.
# Typical values: "/dev/shm/$USER/$SLURM_JOB_ID" or "/scratch/$USER/$SLURM_JOB_ID".
H5_SCRATCH_DIR=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --task_folder)       TASK_FOLDER="$2"; shift 2;;
        --calibration_yaml)  CALIBRATION_YAML="$2"; shift 2;;
        --tool_name)         TOOL_NAME="$2"; shift 2;;
        --chunk_size)        CHUNK_SIZE="$2"; shift 2;;
        --start_frame)       START_FRAME="$2"; shift 2;;
        --end_frame)         END_FRAME="$2"; shift 2;;
        --skip_existing)     SKIP_EXISTING=1; shift;;
        --keep_h5)            KEEP_H5=1; shift;;
        --no_merge)          NO_MERGE=1; shift;;
        --keep_chunk_files)  KEEP_CHUNK_FILES=1; shift;;
        --h5_scratch_dir)    H5_SCRATCH_DIR="$2"; shift 2;;
        --only_exp)          ONLY_EXP="$2"; shift 2;;
        -h|--help)
            sed -n '2,32p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$TASK_FOLDER" || -z "$CALIBRATION_YAML" ]]; then
    echo "Error: --task_folder and --calibration_yaml are required"; exit 1
fi
if [[ ! -d "$TASK_FOLDER" ]]; then
    echo "Error: task folder not found: $TASK_FOLDER"; exit 1
fi
if [[ ! -f "$CALIBRATION_YAML" ]]; then
    echo "Error: calibration yaml not found: $CALIBRATION_YAML"; exit 1
fi
if ! command -v ffprobe >/dev/null 2>&1; then
    echo "Error: ffprobe is required to detect total frame count"; exit 1
fi

TASK_FOLDER_ABS="$(cd "$TASK_FOLDER" && pwd)"
CALIBRATION_YAML="$(readlink -f "$CALIBRATION_YAML")"

# Auto-derive BASE_PATH from task_folder. Structure expected:
#   <BASE_PATH>/<videos_XXXX>/<task>/<exp>/{cam*_rgb.mp4,...}
# So BASE_PATH = parent of the videos_XXXX folder = task_folder/../..
VIDEOS_ROOT_ABS="$(dirname "$TASK_FOLDER_ABS")"
BASE_PATH="$(dirname "$VIDEOS_ROOT_ABS")/"
# Allow these to be overridden via env vars so the same script works on
# different machines without forking. Defaults: local dev machine.
HOCAP_ROOT="${HOCAP_ROOT:-/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation}"
HAND_ROOT="${HAND_ROOT:-/home/ruoqu/crq_ws/robotool/HandReconstruction}"
MODELS_FOLDER="${MODELS_FOLDER:-${HOCAP_ROOT}/data/models}"

SEQUENCE_BASE="${TASK_FOLDER_ABS#$BASE_PATH}"

if [[ -z "$TOOL_NAME" ]]; then
    TOOL_NAME="$(basename "$TASK_FOLDER_ABS")"
fi

echo "=========================================="
echo "Task folder    : $TASK_FOLDER_ABS"
echo "Calibration    : $CALIBRATION_YAML"
echo "Tool name      : $TOOL_NAME"
echo "Chunk size     : $CHUNK_SIZE"
echo "Start frame    : $START_FRAME"
echo "End frame      : ${END_FRAME:-<auto-detect>}"
echo "Skip existing  : $SKIP_EXISTING"
echo "Keep h5        : $KEEP_H5"
echo "=========================================="

# conda.sh path — override via CONDA_SH env var, else try common locations.
_CONDA_SH_CANDIDATES=(
    "${CONDA_SH:-}"
    "/home/ruoqu/miniconda3/etc/profile.d/conda.sh"
    "/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
)
for _p in "${_CONDA_SH_CANDIDATES[@]}"; do
    if [[ -n "$_p" && -f "$_p" ]]; then source "$_p"; break; fi
done

TOTAL_EXP=0; OK_CHUNK=0; FAIL_CHUNK=0; SKIP_CHUNK=0; SKIP_EXP=0
FAILED_CHUNKS=()

for EXP_DIR in "$TASK_FOLDER_ABS"/*/; do
    [[ -d "$EXP_DIR" ]] || continue
    EXP_NAME="$(basename "$EXP_DIR")"
    # Per-exp filter (used by array scheduling — one array element per exp).
    if [[ -n "$ONLY_EXP" && "$EXP_NAME" != "$ONLY_EXP" ]]; then
        continue
    fi
    TOTAL_EXP=$((TOTAL_EXP+1))

    SEQUENCE_NAME="${SEQUENCE_BASE}/${EXP_NAME}"
    SEQUENCE_FOLDER="${BASE_PATH}${SEQUENCE_NAME}"
    H5_PATH="${SEQUENCE_FOLDER}/data00000000.h5"  # what cluster_reconstruct looks for

    # If scratch dir given, put the real h5 there and symlink it at H5_PATH.
    # This keeps the hot I/O off /viscam (NFS) without changing downstream code.
    if [[ -n "$H5_SCRATCH_DIR" ]]; then
        mkdir -p "$H5_SCRATCH_DIR"
        # Unique filename per sequence so multiple experiments don't collide
        # in a shared scratch dir.
        H5_REAL_PATH="${H5_SCRATCH_DIR}/$(echo "$SEQUENCE_NAME" | tr / _)_data00000000.h5"
    else
        H5_REAL_PATH="$H5_PATH"
    fi

    VIDEO_FOLDER="${SEQUENCE_NAME%%/*}"
    REMAINING="${SEQUENCE_NAME#*/}"
    TASK_NAME="${REMAINING%%/*}"
    VIDEO_NAME="${REMAINING#*/}"
    ANNOTATED_PATH="${BASE_PATH}${VIDEO_FOLDER}_annotated/${TASK_NAME}/${VIDEO_NAME}"

    # Experiment-level resume check. After a successful run, all chunk files
    # are merged and deleted, leaving only result_hand_optimized.pkl /
    # result.pkl / poses_m.npy. If we re-run with --skip_existing and the
    # merged optimized pkl is there, we can skip the entire experiment (no
    # need to even ffprobe the video).
    MERGED_FINAL="${ANNOTATED_PATH}/result_hand_optimized.pkl"
    if [[ "$SKIP_EXISTING" == "1" && -f "$MERGED_FINAL" ]]; then
        echo ""
        echo "[$TOTAL_EXP] $EXP_NAME  [SKIP-EXP: merged result_hand_optimized.pkl exists]"
        SKIP_EXP=$((SKIP_EXP+1))
        continue
    fi

    # Detect total frame count from cam0_rgb.mp4 (or any cam*_rgb.mp4)
    REF_VIDEO="$(ls "$SEQUENCE_FOLDER"/cam*_rgb.mp4 2>/dev/null | head -n 1)"
    if [[ -z "$REF_VIDEO" ]]; then
        echo "[WARN] no cam*_rgb.mp4 in $SEQUENCE_FOLDER — skipping"
        continue
    fi
    TOTAL_FRAMES="$(ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 \
        "$REF_VIDEO" 2>/dev/null)"
    if [[ -z "$TOTAL_FRAMES" || "$TOTAL_FRAMES" == "N/A" ]]; then
        # fall back to packet count (faster, may be less accurate for VFR)
        TOTAL_FRAMES="$(ffprobe -v error -select_streams v:0 \
            -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 \
            "$REF_VIDEO" 2>/dev/null)"
    fi
    if [[ -z "$TOTAL_FRAMES" || "$TOTAL_FRAMES" == "N/A" ]]; then
        echo "[WARN] could not determine frame count for $REF_VIDEO — skipping"
        continue
    fi

    # Compute overall window
    LAST_FRAME=$((TOTAL_FRAMES - 1))
    GLOBAL_END=$LAST_FRAME
    if [[ -n "$END_FRAME" ]]; then
        GLOBAL_END=$END_FRAME
        if [[ $GLOBAL_END -gt $LAST_FRAME ]]; then GLOBAL_END=$LAST_FRAME; fi
    fi
    GLOBAL_START=$START_FRAME

    echo ""
    echo "=========================================="
    echo "[$TOTAL_EXP] $EXP_NAME  (total frames: $TOTAL_FRAMES, window: [$GLOBAL_START, $GLOBAL_END])"
    echo "  sequence_folder = $SEQUENCE_FOLDER"
    echo "  annotated_path  = $ANNOTATED_PATH"
    echo "=========================================="

    # Iterate chunks
    CHUNK_START=$GLOBAL_START
    while [[ $CHUNK_START -le $GLOBAL_END ]]; do
        CHUNK_END=$((CHUNK_START + CHUNK_SIZE - 1))
        if [[ $CHUNK_END -gt $GLOBAL_END ]]; then CHUNK_END=$GLOBAL_END; fi

        CHUNK_TAG="${CHUNK_START}_${CHUNK_END}"
        FINAL_PKL="${ANNOTATED_PATH}/result_hand_optimized_${CHUNK_TAG}.pkl"
        RECON_PKL="${ANNOTATED_PATH}/result_${CHUNK_TAG}.pkl"
        POSES_NPY="${ANNOTATED_PATH}/poses_m_${CHUNK_TAG}.npy"

        echo ""
        echo "  --- chunk [${CHUNK_START}, ${CHUNK_END}] ---"

        if [[ "$SKIP_EXISTING" == "1" && -f "$FINAL_PKL" ]]; then
            echo "    [skip] $FINAL_PKL exists"
            SKIP_CHUNK=$((SKIP_CHUNK+1))
            CHUNK_START=$((CHUNK_END + 1))
            continue
        fi

        # --- step 1: videos -> h5 (chunk range) ---
        conda activate hocap-annotation
        cd "$HOCAP_ROOT" || { echo "    [ERR] cd $HOCAP_ROOT"; FAIL_CHUNK=$((FAIL_CHUNK+1)); FAILED_CHUNKS+=("$EXP_NAME:$CHUNK_TAG:cd"); CHUNK_START=$((CHUNK_END + 1)); continue; }

        # always rebuild per chunk — clean both logical (may be symlink) and real
        rm -f "$H5_PATH" "$H5_REAL_PATH"
        if ! python tools/00_convert_videos_to_h5.py \
                --input_dir "$SEQUENCE_FOLDER" \
                --output_file "$H5_REAL_PATH" \
                --start_frame "$CHUNK_START" \
                --end_frame "$CHUNK_END"; then
            echo "    [ERR] h5 conversion"
            FAIL_CHUNK=$((FAIL_CHUNK+1)); FAILED_CHUNKS+=("$EXP_NAME:$CHUNK_TAG:h5")
            CHUNK_START=$((CHUNK_END + 1)); continue
        fi
        # Expose the real file at the path cluster_reconstruct / generate_meta expect.
        if [[ "$H5_REAL_PATH" != "$H5_PATH" ]]; then
            ln -sf "$H5_REAL_PATH" "$H5_PATH"
        fi

        # --- step 2: generate meta (start_frame = CHUNK_START for mask alignment) ---
        # hand-only annotation has no tool/object masks. generate_meta.py still
        # tries to write {annotated_path}/masks/masks.h5 — pre-create the dir so
        # it falls through with zero masks instead of failing on missing folder.
        mkdir -p "${ANNOTATED_PATH}/masks"
        if ! python preprocess/generate_meta.py \
                --h5_path "$H5_PATH" \
                --calibration_yaml_path "$CALIBRATION_YAML" \
                --models_folder "$MODELS_FOLDER" \
                --tool_name "$TOOL_NAME" \
                --start_frame "$CHUNK_START" \
                --x_min -0.6 --x_max 0.6 --y_min -0.5 --y_max 0.6 --z_min -0.5 --z_max 0.4; then
            echo "    [ERR] generate_meta"
            FAIL_CHUNK=$((FAIL_CHUNK+1)); FAILED_CHUNKS+=("$EXP_NAME:$CHUNK_TAG:meta")
            [[ "$KEEP_H5" == "0" ]] && rm -f "$H5_PATH" "$H5_REAL_PATH"
            CHUNK_START=$((CHUNK_END + 1)); continue
        fi

        # --- step 3: hand reconstruction ---
        cd "$HAND_ROOT" || { echo "    [ERR] cd $HAND_ROOT"; FAIL_CHUNK=$((FAIL_CHUNK+1)); FAILED_CHUNKS+=("$EXP_NAME:$CHUNK_TAG:cd_hand"); CHUNK_START=$((CHUNK_END + 1)); continue; }
        conda activate reconstruct-hand

        # remove any stale default-named outputs from a previous chunk
        rm -f "${ANNOTATED_PATH}/result.pkl" \
              "${ANNOTATED_PATH}/result_hand_optimized.pkl" \
              "${ANNOTATED_PATH}/poses_m.npy"

        if ! python cluster_reconstruct.py --sequence_folder "$SEQUENCE_FOLDER"; then
            echo "    [ERR] reconstruct"
            FAIL_CHUNK=$((FAIL_CHUNK+1)); FAILED_CHUNKS+=("$EXP_NAME:$CHUNK_TAG:reconstruct")
            [[ "$KEEP_H5" == "0" ]] && rm -f "$H5_PATH" "$H5_REAL_PATH"
            CHUNK_START=$((CHUNK_END + 1)); continue
        fi

        SAVED_FILE="${ANNOTATED_PATH}/result.pkl"
        if [[ ! -f "$SAVED_FILE" ]]; then
            echo "    [WARN] result.pkl not produced; skipping optimize"
            FAIL_CHUNK=$((FAIL_CHUNK+1)); FAILED_CHUNKS+=("$EXP_NAME:$CHUNK_TAG:no_result")
            [[ "$KEEP_H5" == "0" ]] && rm -f "$H5_PATH" "$H5_REAL_PATH"
            CHUNK_START=$((CHUNK_END + 1)); continue
        fi

        # --- step 4: hand optimization (must be called with file_name=result.pkl
        #     because cluster_optimize_hand uses str.replace("result.pkl", ...)) ---
        OPT_OK=1
        if ! python cluster_optimize_hand.py --file_name "$SAVED_FILE"; then
            echo "    [WARN] optimize failed; keeping reconstruction output only"
            OPT_OK=0
        fi

        # --- step 5: rename outputs to chunk-suffixed names ---
        mv -f "$SAVED_FILE" "$RECON_PKL" 2>/dev/null || true
        if [[ $OPT_OK -eq 1 ]]; then
            mv -f "${ANNOTATED_PATH}/result_hand_optimized.pkl" "$FINAL_PKL" 2>/dev/null || true
            mv -f "${ANNOTATED_PATH}/poses_m.npy" "$POSES_NPY" 2>/dev/null || true
        fi

        # --- step 6: cleanup h5 (symlink and real file) ---
        if [[ "$KEEP_H5" == "0" ]]; then
            rm -f "$H5_PATH" "$H5_REAL_PATH"
        fi

        if [[ $OPT_OK -eq 1 ]]; then
            OK_CHUNK=$((OK_CHUNK+1))
            echo "    [done] chunk $CHUNK_TAG -> $FINAL_PKL"
        else
            FAIL_CHUNK=$((FAIL_CHUNK+1)); FAILED_CHUNKS+=("$EXP_NAME:$CHUNK_TAG:optimize")
        fi

        CHUNK_START=$((CHUNK_END + 1))
    done

    # --- merge chunks for this experiment ---
    if [[ "$NO_MERGE" == "0" ]]; then
        # numeric-only globs so we never match result.pkl / result_hand_optimized.pkl
        CHUNK_COUNT=$(ls "${ANNOTATED_PATH}"/result_hand_optimized_[0-9]*_[0-9]*.pkl 2>/dev/null | wc -l)
        if [[ "$CHUNK_COUNT" -eq 0 ]]; then
            echo "  [merge] no chunks found in $ANNOTATED_PATH — skipping merge"
        else
            echo ""
            echo "  --- merging $CHUNK_COUNT chunks for $EXP_NAME ---"
            conda activate hocap-annotation
            cd "$HOCAP_ROOT" || { echo "  [ERR] cd $HOCAP_ROOT for merge"; continue; }
            if python scripts/merge_hand_chunks.py --annotated_path "$ANNOTATED_PATH"; then
                if [[ "$KEEP_CHUNK_FILES" == "0" ]]; then
                    rm -f "${ANNOTATED_PATH}"/result_hand_optimized_[0-9]*_[0-9]*.pkl \
                          "${ANNOTATED_PATH}"/result_[0-9]*_[0-9]*.pkl \
                          "${ANNOTATED_PATH}"/poses_m_[0-9]*_[0-9]*.npy
                    echo "  [merge] deleted chunk files, kept merged result.pkl / result_hand_optimized.pkl / poses_m.npy"
                fi
            else
                echo "  [ERR] merge failed for $EXP_NAME"
            fi
        fi
    fi
done

echo ""
echo "=========================================="
echo "Summary: experiments=$TOTAL_EXP (skipped=$SKIP_EXP)  chunks_ok=$OK_CHUNK  chunks_skipped=$SKIP_CHUNK  chunks_failed=$FAIL_CHUNK"
if [[ "${#FAILED_CHUNKS[@]}" -gt 0 ]]; then
    echo "Failed chunks:"
    for fc in "${FAILED_CHUNKS[@]}"; do echo "  - $fc"; done
fi
echo "=========================================="
