#!/bin/bash
# Full end-to-end automation for one videos_XXXX/ root:
#   1) auto-discover the initial calibration yaml in realsense_calibrate_*/
#   2) build a 1-frame h5 from a representative experiment
#   3) run tools/00-0_align_cameras.py to cache per-camera point clouds
#   4) run tools/run_global_align_headless.py (no viewer) to produce the
#      globally-aligned yaml + postalign_global.ply
#   5) render a static snapshot PNG of the merged point cloud
#   6) for every task folder under videos_XXXX/, call
#      scripts/batch_task_folder_hand_chunked.sh with the aligned yaml
#
# Folder structure expected:
#   <videos_root>/
#     realsense_calibrate_<date>/realsense_calibration_<date>.yaml
#     <task_folder>/
#       <experiment_folder>/cam*_rgb.mp4  cam*_depth.mkv
#       ...
#     <another_task_folder>/
#
# Usage:
#   bash scripts/run_full_auto.sh \
#     --videos_root /abs/path/to/videos_0101 \
#     [--chunk_size 500] [--skip_existing] [--keep_h5]
#     [--cal_frame_idx 0]      # which frame of the tiny h5 to use for 00-0
#     [--representative_exp NAME] # override auto-picked experiment
#     [--force_recalibrate]    # re-run calibration even if aligned yaml exists
#     [--skip_calibration]     # use the existing original yaml as-is
#     [--skip_hand]            # only do calibration, skip batch_task_folder_hand_chunked

set -u

VIDEOS_ROOT=""
CHUNK_SIZE=500
SKIP_EXISTING_FLAG=""
KEEP_H5_FLAG=""
NO_MERGE_FLAG=""
KEEP_CHUNK_FILES_FLAG=""
CAL_FRAME_IDX=0
REPRESENTATIVE_EXP=""
FORCE_RECAL=0
SKIP_CAL=0
SKIP_HAND=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --videos_root)        VIDEOS_ROOT="$2"; shift 2;;
        --chunk_size)         CHUNK_SIZE="$2"; shift 2;;
        --skip_existing)      SKIP_EXISTING_FLAG="--skip_existing"; shift;;
        --keep_h5)            KEEP_H5_FLAG="--keep_h5"; shift;;
        --no_merge)           NO_MERGE_FLAG="--no_merge"; shift;;
        --keep_chunk_files)   KEEP_CHUNK_FILES_FLAG="--keep_chunk_files"; shift;;
        --cal_frame_idx)      CAL_FRAME_IDX="$2"; shift 2;;
        --representative_exp) REPRESENTATIVE_EXP="$2"; shift 2;;
        --force_recalibrate)  FORCE_RECAL=1; shift;;
        --skip_calibration)   SKIP_CAL=1; shift;;
        --skip_hand)          SKIP_HAND=1; shift;;
        -h|--help) sed -n '2,30p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$VIDEOS_ROOT" ]]; then
    echo "Error: --videos_root is required"; exit 1
fi
VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
if [[ ! -d "$VIDEOS_ROOT" ]]; then
    echo "Error: videos_root not a directory: $VIDEOS_ROOT"; exit 1
fi

HOCAP_ROOT="/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation"
source /home/ruoqu/miniconda3/etc/profile.d/conda.sh
conda activate hocap-annotation

echo "=========================================="
echo "videos_root = $VIDEOS_ROOT"
echo "=========================================="

# --- step A: discover calibration folder and yaml ---
CAL_FOLDER="$(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1)"
if [[ -z "$CAL_FOLDER" ]]; then
    echo "Error: no realsense_calibrate_* folder under $VIDEOS_ROOT"; exit 1
fi
ORIG_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' \
              ! -name '*_global_aligned.yaml' | head -n 1)"
if [[ -z "$ORIG_YAML" ]]; then
    echo "Error: no realsense_calibration_*.yaml in $CAL_FOLDER"; exit 1
fi
YAML_STEM="$(basename "$ORIG_YAML" .yaml)"
ALIGNED_YAML="${CAL_FOLDER}/${YAML_STEM}_global_aligned.yaml"
SNAPSHOT_PNG="${CAL_FOLDER}/calibration_snapshot.png"
POSTALIGN_PLY="${CAL_FOLDER}/postalign_global.ply"

echo "[calibration]"
echo "  folder     : $CAL_FOLDER"
echo "  orig yaml  : $ORIG_YAML"
echo "  aligned    : $ALIGNED_YAML"

# --- step B: discover task folders (everything else under videos_root) ---
TASK_FOLDERS=()
while IFS= read -r -d '' d; do
    name="$(basename "$d")"
    [[ "$name" == realsense_calibrate_* ]] && continue
    TASK_FOLDERS+=("$d")
done < <(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

if [[ "${#TASK_FOLDERS[@]}" -eq 0 ]]; then
    echo "Error: no task folders found under $VIDEOS_ROOT"; exit 1
fi
echo "[tasks]  ${#TASK_FOLDERS[@]} task folder(s):"
for t in "${TASK_FOLDERS[@]}"; do echo "   - $(basename "$t")"; done

# --- step C: run calibration (or skip) ---
if [[ "$SKIP_CAL" == "1" ]]; then
    echo "[cal] --skip_calibration set, using ORIG_YAML as the calibration"
    CAL_YAML_TO_USE="$ORIG_YAML"
elif [[ -f "$ALIGNED_YAML" && "$FORCE_RECAL" == "0" ]]; then
    echo "[cal] aligned yaml already exists, skipping calibration"
    echo "      ($ALIGNED_YAML — pass --force_recalibrate to redo)"
    CAL_YAML_TO_USE="$ALIGNED_YAML"
else
    # C1. pick a representative experiment (first non-tar.gz dir in first task)
    FIRST_TASK="${TASK_FOLDERS[0]}"
    if [[ -n "$REPRESENTATIVE_EXP" ]]; then
        REP_DIR="$FIRST_TASK/$REPRESENTATIVE_EXP"
    else
        REP_DIR=""
        while IFS= read -r -d '' d; do
            name="$(basename "$d")"
            [[ "$name" == *.tar.gz || "$name" == board_reference ]] && continue
            # need at least one cam*_rgb.mp4
            if ls "$d"/cam*_rgb.mp4 >/dev/null 2>&1; then
                REP_DIR="$d"; break
            fi
        done < <(find "$FIRST_TASK" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
    fi
    if [[ -z "$REP_DIR" || ! -d "$REP_DIR" ]]; then
        echo "Error: could not find a representative experiment under $FIRST_TASK"
        exit 1
    fi
    echo "[cal] representative exp: $REP_DIR"

    TINY_H5="${REP_DIR}/data00000000.h5"
    CACHED_PC_DIR="${CAL_FOLDER}/cached_pc"

    # C2. build 1-frame h5
    echo "[cal] building tiny h5 (frame ${CAL_FRAME_IDX})"
    rm -f "$TINY_H5"
    if ! python "$HOCAP_ROOT/tools/00_convert_videos_to_h5.py" \
            --input_dir "$REP_DIR" \
            --output_file "$TINY_H5" \
            --start_frame "$CAL_FRAME_IDX" \
            --end_frame "$CAL_FRAME_IDX"; then
        echo "Error: tiny h5 conversion failed"; exit 1
    fi

    # C3. cache per-camera point clouds
    echo "[cal] caching per-camera point clouds via 00-0_align_cameras.py"
    if ! python "$HOCAP_ROOT/tools/00-0_align_cameras.py" \
            --h5_file "$TINY_H5" \
            --extrinsic_file "$ORIG_YAML" \
            --out_path "$CAL_FOLDER" \
            --frame_idx 0; then
        echo "Error: 00-0 cached-pc step failed"; rm -f "$TINY_H5"; exit 1
    fi

    # C4. global alignment (headless)
    echo "[cal] running headless global alignment"
    if ! python "$HOCAP_ROOT/tools/run_global_align_headless.py" \
            --cached_pc "$CACHED_PC_DIR" \
            --extrinsic_file "$ORIG_YAML" \
            --out_path "$CAL_FOLDER"; then
        echo "Error: global alignment failed"; rm -f "$TINY_H5"; exit 1
    fi

    # C5. cleanup tiny h5
    rm -f "$TINY_H5"
    echo "[cal] deleted tiny h5"

    # C6. snapshot PNG
    if [[ -f "$POSTALIGN_PLY" ]]; then
        if python "$HOCAP_ROOT/scripts/render_calibration_snapshot.py" \
                --ply "$POSTALIGN_PLY" --out "$SNAPSHOT_PNG"; then
            echo "[cal] snapshot -> $SNAPSHOT_PNG"
        else
            echo "[cal] WARNING: snapshot rendering failed (aligned yaml still produced)"
        fi
    else
        echo "[cal] WARNING: $POSTALIGN_PLY not found, cannot render snapshot"
    fi

    if [[ ! -f "$ALIGNED_YAML" ]]; then
        echo "Error: expected $ALIGNED_YAML was not produced"; exit 1
    fi
    CAL_YAML_TO_USE="$ALIGNED_YAML"
fi

echo ""
echo "[cal] using calibration yaml: $CAL_YAML_TO_USE"

# --- step D: run hand-annotation batch on every task folder ---
if [[ "$SKIP_HAND" == "1" ]]; then
    echo "[hand] --skip_hand set, done"
    exit 0
fi

TASK_OK=0
TASK_FAIL=0
FAILED_TASKS=()

for task_dir in "${TASK_FOLDERS[@]}"; do
    task_name="$(basename "$task_dir")"
    echo ""
    echo "##########################################"
    echo "# task: $task_name"
    echo "##########################################"

    if bash "$HOCAP_ROOT/scripts/batch_task_folder_hand_chunked.sh" \
            --task_folder "$task_dir" \
            --calibration_yaml "$CAL_YAML_TO_USE" \
            --chunk_size "$CHUNK_SIZE" \
            $SKIP_EXISTING_FLAG $KEEP_H5_FLAG $NO_MERGE_FLAG $KEEP_CHUNK_FILES_FLAG; then
        TASK_OK=$((TASK_OK+1))
    else
        TASK_FAIL=$((TASK_FAIL+1))
        FAILED_TASKS+=("$task_name")
    fi
done

echo ""
echo "=========================================="
echo "Full-auto summary: tasks_ok=$TASK_OK  tasks_failed=$TASK_FAIL"
if [[ "${#FAILED_TASKS[@]}" -gt 0 ]]; then
    echo "Failed tasks:"
    for t in "${FAILED_TASKS[@]}"; do echo "  - $t"; done
fi
echo "=========================================="
