#!/bin/bash
# Batch pipeline: for every experiment folder inside a task folder, run
#   1) videos -> h5        (tools/00_convert_videos_to_h5.py)
#   2) generate meta       (preprocess/generate_meta.py, with user-supplied calibration)
#   3) hand reconstruct    (HandReconstruction/cluster_reconstruct.py)
#   4) hand optimize       (HandReconstruction/cluster_optimize_hand.py)
#   5) delete the h5       (to save disk)
#
# Resume-safe: pass --skip_existing to skip experiments whose result.pkl
# already exists. Failures in one experiment do NOT stop the loop.
#
# Usage:
#   bash scripts/batch_task_folder_hand.sh \
#     --task_folder       /abs/path/to/data/videos_0101/mallet_crush_nuts \
#     --calibration_yaml  /abs/path/to/realsense_calibration_xxxx.yaml \
#     [--tool_name mallet_crush_nuts]   # default: basename of task_folder
#     [--start_frame 0] [--end_frame 500] \
#     [--skip_existing] [--keep_h5]

TASK_FOLDER=""
CALIBRATION_YAML=""
TOOL_NAME=""
START_FRAME="0"
END_FRAME=""
SKIP_EXISTING=0
KEEP_H5=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --task_folder)       TASK_FOLDER="$2"; shift 2;;
        --calibration_yaml)  CALIBRATION_YAML="$2"; shift 2;;
        --tool_name)         TOOL_NAME="$2"; shift 2;;
        --start_frame)       START_FRAME="$2"; shift 2;;
        --end_frame)         END_FRAME="$2"; shift 2;;
        --skip_existing)     SKIP_EXISTING=1; shift;;
        --keep_h5)           KEEP_H5=1; shift;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$TASK_FOLDER" ]]; then
    echo "Error: --task_folder is required"; exit 1
fi
if [[ -z "$CALIBRATION_YAML" ]]; then
    echo "Error: --calibration_yaml is required"; exit 1
fi
if [[ ! -d "$TASK_FOLDER" ]]; then
    echo "Error: task folder not found: $TASK_FOLDER"; exit 1
fi
if [[ ! -f "$CALIBRATION_YAML" ]]; then
    echo "Error: calibration yaml not found: $CALIBRATION_YAML"; exit 1
fi

# absolute paths
TASK_FOLDER_ABS="$(cd "$TASK_FOLDER" && pwd)"
CALIBRATION_YAML="$(readlink -f "$CALIBRATION_YAML")"

BASE_PATH="/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/"
HOCAP_ROOT="/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation"
HAND_ROOT="/home/ruoqu/crq_ws/robotool/HandReconstruction"
MODELS_FOLDER="${HOCAP_ROOT}/data/models"

# sequence name relative to BASE_PATH (e.g., videos_0101/mallet_crush_nuts)
if [[ "$TASK_FOLDER_ABS" != "$BASE_PATH"* ]]; then
    echo "Error: task folder must be under $BASE_PATH"; exit 1
fi
SEQUENCE_BASE="${TASK_FOLDER_ABS#$BASE_PATH}"

# default tool name = task folder basename
if [[ -z "$TOOL_NAME" ]]; then
    TOOL_NAME="$(basename "$TASK_FOLDER_ABS")"
fi

echo "=========================================="
echo "Task folder    : $TASK_FOLDER_ABS"
echo "Calibration    : $CALIBRATION_YAML"
echo "Tool name      : $TOOL_NAME"
echo "Start frame    : $START_FRAME"
echo "End frame      : ${END_FRAME:-<full>}"
echo "Skip existing  : $SKIP_EXISTING"
echo "Keep h5        : $KEEP_H5"
echo "=========================================="

source /home/ruoqu/miniconda3/etc/profile.d/conda.sh

TOTAL=0; OK=0; FAIL=0; SKIP=0
FAILED_EXPERIMENTS=()

for EXP_DIR in "$TASK_FOLDER_ABS"/*/; do
    [[ -d "$EXP_DIR" ]] || continue
    EXP_NAME="$(basename "$EXP_DIR")"
    TOTAL=$((TOTAL+1))

    SEQUENCE_NAME="${SEQUENCE_BASE}/${EXP_NAME}"
    SEQUENCE_FOLDER="${BASE_PATH}${SEQUENCE_NAME}"
    H5_PATH="${SEQUENCE_FOLDER}/data00000000.h5"

    # annotated path (matches batch_hand_reconstruct.sh convention)
    VIDEO_FOLDER="${SEQUENCE_NAME%%/*}"
    REMAINING="${SEQUENCE_NAME#*/}"
    TASK_NAME="${REMAINING%%/*}"
    VIDEO_NAME="${REMAINING#*/}"
    ANNOTATED_PATH="${BASE_PATH}${VIDEO_FOLDER}_annotated/${TASK_NAME}/${VIDEO_NAME}"

    echo ""
    echo "=========================================="
    echo "[$TOTAL] $EXP_NAME"
    echo "  sequence_folder = $SEQUENCE_FOLDER"
    echo "  annotated_path  = $ANNOTATED_PATH"
    echo "=========================================="

    if [[ "$SKIP_EXISTING" == "1" && -f "${ANNOTATED_PATH}/result.pkl" ]]; then
        echo "  [skip] result.pkl already exists"
        SKIP=$((SKIP+1))
        continue
    fi

    # ------- step 1: videos -> h5 -------
    conda activate hocap-annotation
    cd "$HOCAP_ROOT" || { echo "  [ERROR] cd $HOCAP_ROOT failed"; FAIL=$((FAIL+1)); FAILED_EXPERIMENTS+=("$EXP_NAME:cd"); continue; }

    if [[ -f "$H5_PATH" ]]; then
        echo "  [info] h5 already exists, reusing: $H5_PATH"
    else
        H5_ARGS=(--input_dir "$SEQUENCE_FOLDER" --output_file "$H5_PATH" --start_frame "$START_FRAME")
        [[ -n "$END_FRAME" ]] && H5_ARGS+=(--end_frame "$END_FRAME")
        if ! python tools/00_convert_videos_to_h5.py "${H5_ARGS[@]}"; then
            echo "  [ERROR] h5 conversion failed"
            FAIL=$((FAIL+1)); FAILED_EXPERIMENTS+=("$EXP_NAME:h5")
            continue
        fi
    fi

    # ------- step 2: generate meta -------
    if ! python preprocess/generate_meta.py \
            --h5_path "$H5_PATH" \
            --calibration_yaml_path "$CALIBRATION_YAML" \
            --models_folder "$MODELS_FOLDER" \
            --tool_name "$TOOL_NAME" \
            --start_frame "$START_FRAME" \
            --x_min -0.6 --x_max 0.6 --y_min -0.5 --y_max 0.6 --z_min -0.5 --z_max 0.4; then
        echo "  [ERROR] meta generation failed"
        FAIL=$((FAIL+1)); FAILED_EXPERIMENTS+=("$EXP_NAME:meta")
        [[ "$KEEP_H5" == "0" ]] && rm -f "$H5_PATH"
        continue
    fi

    # ------- step 3: hand reconstruction -------
    cd "$HAND_ROOT" || { echo "  [ERROR] cd $HAND_ROOT failed"; FAIL=$((FAIL+1)); FAILED_EXPERIMENTS+=("$EXP_NAME:cd_hand"); continue; }
    conda activate reconstruct-hand

    if ! python cluster_reconstruct.py --sequence_folder "$SEQUENCE_FOLDER"; then
        echo "  [ERROR] hand reconstruction failed"
        FAIL=$((FAIL+1)); FAILED_EXPERIMENTS+=("$EXP_NAME:reconstruct")
        [[ "$KEEP_H5" == "0" ]] && rm -f "$H5_PATH"
        continue
    fi

    # ------- step 4: hand optimization -------
    SAVED_FILE="${ANNOTATED_PATH}/result.pkl"
    if [[ -f "$SAVED_FILE" ]]; then
        if ! python cluster_optimize_hand.py --file_name "$SAVED_FILE"; then
            echo "  [WARN] hand optimization failed; reconstruction output kept"
            FAIL=$((FAIL+1)); FAILED_EXPERIMENTS+=("$EXP_NAME:optimize")
            [[ "$KEEP_H5" == "0" ]] && rm -f "$H5_PATH"
            continue
        fi
    else
        echo "  [WARN] result.pkl not found at $SAVED_FILE — skipping optimize"
    fi

    # ------- step 5: delete h5 -------
    if [[ "$KEEP_H5" == "0" && -f "$H5_PATH" ]]; then
        rm -f "$H5_PATH"
        echo "  [info] deleted h5: $H5_PATH"
    fi

    OK=$((OK+1))
    echo "  [done] $EXP_NAME"
done

echo ""
echo "=========================================="
echo "Summary: total=$TOTAL  ok=$OK  skipped=$SKIP  failed=$FAIL"
if [[ "${#FAILED_EXPERIMENTS[@]}" -gt 0 ]]; then
    echo "Failed experiments:"
    for fe in "${FAILED_EXPERIMENTS[@]}"; do echo "  - $fe"; done
fi
echo "=========================================="
