#!/bin/bash
# One-liner wrapper for the DINO-based auto-annotation pipeline over a whole
# videos_root/ folder. Auto-discovers the globally-aligned calibration yaml
# under realsense_calibrate_*/ and runs scripts/batch_auto_annotator.sh on
# every task folder.
#
# Assumes calibration has ALREADY been done (i.e. *_global_aligned.yaml
# exists under realsense_calibrate_<date>/). To run calibration itself,
# use scripts/run_full_auto.sh first.
#
# Folder structure expected:
#   <videos_root>/
#     realsense_calibrate_<date>/
#       realsense_calibration_<date>_global_aligned.yaml   <- auto-picked
#       realsense_calibration_<date>.yaml                  <- fallback
#     <task_folder>/<exp>/cam*_rgb.mp4  cam*_depth.mkv
#     <another_task>/...
#
# Usage:
#   bash scripts/run_full_auto_annotator.sh \
#     --videos_root /abs/path/to/videos_XXXX \
#     [--frame0_only]           # DINO-register only on frame 0 (fast cluster mode)
#     [--hand 1] [--optimize 1]
#     [--skip_existing] [--dry_run]
#     [--models_folder /abs/path/models]
#     [--force_tool NAME]       # override auto-match for ALL exps
#     [--mapping_only]          # only run exps listed in tool_keyword_mapping.yaml
#     [--h5_scratch_dir DIR]
#     [--start_frame N] [--end_frame N]
#     [--seed_fast]             # fast-seed mode (if not using --frame0_only)

set -u

VIDEOS_ROOT=""
FRAME0_ONLY_FLAG=""
HAND=""
OPTIMIZE=""
SKIP_EXISTING_FLAG=""
DRY_RUN_FLAG=""
MODELS_FOLDER=""
FORCE_TOOL=""
MAPPING_ONLY_FLAG=""
H5_SCRATCH_DIR=""
START_FRAME=""
END_FRAME=""
SEED_FAST_FLAG=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --videos_root)     VIDEOS_ROOT="$2"; shift 2;;
        --frame0_only)     FRAME0_ONLY_FLAG="--frame0_only"; shift;;
        --hand)            HAND="$2"; shift 2;;
        --optimize)        OPTIMIZE="$2"; shift 2;;
        --skip_existing)   SKIP_EXISTING_FLAG="--skip_existing"; shift;;
        --dry_run)         DRY_RUN_FLAG="--dry_run"; shift;;
        --models_folder)   MODELS_FOLDER="$2"; shift 2;;
        --force_tool)      FORCE_TOOL="$2"; shift 2;;
        --mapping_only)    MAPPING_ONLY_FLAG="--mapping_only"; shift;;
        --h5_scratch_dir)  H5_SCRATCH_DIR="$2"; shift 2;;
        --start_frame)     START_FRAME="$2"; shift 2;;
        --end_frame)       END_FRAME="$2"; shift 2;;
        --seed_fast)       SEED_FAST_FLAG="--seed_fast"; shift;;
        -h|--help) sed -n '2,35p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$VIDEOS_ROOT" ]]; then
    echo "Error: --videos_root is required"; exit 1
fi
VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"

HOCAP_ROOT="${HOCAP_ROOT:-/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation}"

# --- auto-discover calibration yaml (prefer *_global_aligned.yaml) ---
CAL_FOLDER="$(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1)"
if [[ -z "$CAL_FOLDER" ]]; then
    echo "Error: no realsense_calibrate_* folder under $VIDEOS_ROOT"; exit 1
fi

CAL_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name '*_global_aligned.yaml' | head -n 1)"
if [[ -z "$CAL_YAML" ]]; then
    CAL_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' \
                  ! -name '*_global_aligned.yaml' | head -n 1)"
    if [[ -z "$CAL_YAML" ]]; then
        echo "Error: no calibration yaml found in $CAL_FOLDER"; exit 1
    fi
    echo "[WARN] No *_global_aligned.yaml found; falling back to $(basename "$CAL_YAML")."
    echo "       Consider running scripts/run_full_auto.sh first to produce the aligned yaml."
fi

# --- discover task folders ---
TASK_FOLDERS=()
while IFS= read -r -d '' d; do
    name="$(basename "$d")"
    [[ "$name" == realsense_calibrate_* ]] && continue
    TASK_FOLDERS+=("$d")
done < <(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

if [[ "${#TASK_FOLDERS[@]}" -eq 0 ]]; then
    echo "Error: no task folders found under $VIDEOS_ROOT"; exit 1
fi

echo "=========================================="
echo "videos_root    : $VIDEOS_ROOT"
echo "calibration    : $CAL_YAML"
echo "tasks (${#TASK_FOLDERS[@]}):"
for t in "${TASK_FOLDERS[@]}"; do echo "   - $(basename "$t")"; done
[[ -n "$FRAME0_ONLY_FLAG" ]] && echo "mode           : frame0_only"
echo "=========================================="

TASK_OK=0
TASK_FAIL=0
FAILED_TASKS=()

for task_dir in "${TASK_FOLDERS[@]}"; do
    task_name="$(basename "$task_dir")"
    echo ""
    echo "##########################################"
    echo "# task: $task_name"
    echo "##########################################"

    BATCH_ARGS=(
        --task_folder "$task_dir"
        --calibration_yaml "$CAL_YAML"
    )
    [[ -n "$HAND" ]]              && BATCH_ARGS+=(--hand "$HAND")
    [[ -n "$OPTIMIZE" ]]          && BATCH_ARGS+=(--optimize "$OPTIMIZE")
    [[ -n "$MODELS_FOLDER" ]]     && BATCH_ARGS+=(--models_folder "$MODELS_FOLDER")
    [[ -n "$FORCE_TOOL" ]]        && BATCH_ARGS+=(--force_tool "$FORCE_TOOL")
    [[ -n "$H5_SCRATCH_DIR" ]]    && BATCH_ARGS+=(--h5_scratch_dir "$H5_SCRATCH_DIR")
    [[ -n "$START_FRAME" ]]       && BATCH_ARGS+=(--start_frame "$START_FRAME")
    [[ -n "$END_FRAME" ]]         && BATCH_ARGS+=(--end_frame "$END_FRAME")
    [[ -n "$SKIP_EXISTING_FLAG" ]] && BATCH_ARGS+=("$SKIP_EXISTING_FLAG")
    [[ -n "$DRY_RUN_FLAG" ]]       && BATCH_ARGS+=("$DRY_RUN_FLAG")
    [[ -n "$MAPPING_ONLY_FLAG" ]]  && BATCH_ARGS+=("$MAPPING_ONLY_FLAG")
    [[ -n "$SEED_FAST_FLAG" ]]     && BATCH_ARGS+=("$SEED_FAST_FLAG")
    [[ -n "$FRAME0_ONLY_FLAG" ]]   && BATCH_ARGS+=("$FRAME0_ONLY_FLAG")

    if bash "$HOCAP_ROOT/scripts/batch_auto_annotator.sh" "${BATCH_ARGS[@]}"; then
        TASK_OK=$((TASK_OK+1))
    else
        TASK_FAIL=$((TASK_FAIL+1))
        FAILED_TASKS+=("$task_name")
    fi
done

echo ""
echo "=========================================="
echo "Full auto-annotator summary: tasks_ok=$TASK_OK  tasks_failed=$TASK_FAIL"
if [[ "${#FAILED_TASKS[@]}" -gt 0 ]]; then
    echo "Failed tasks:"
    for t in "${FAILED_TASKS[@]}"; do echo "  - $t"; done
fi
echo "=========================================="
