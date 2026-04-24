#!/bin/bash
# Batch counterpart to scripts/run_auto_annotator.sh.
# Iterates every experiment folder under --task_folder, auto-detects the tool
# name by matching against --models_folder (scripts/match_tool_name.py), and
# runs the full DINOv2-based auto annotation pipeline on each experiment.
#
# Tool name rule (see match_tool_name.py for details):
#   - task first token sets the "category" (e.g., mallet, fork, spoon)
#   - exp name adds context (e.g., "greenspoon" -> green_spoon)
#   - longest overlap with a model folder name wins, shorter = more canonical
#
# Resume-safe: with --skip_existing, an experiment whose merged object poses
# already exist (processed/fd_pose_solver/fd_poses_merged_fixed.npy) or whose
# final hand result exists (result_hand_optimized.pkl) is skipped.
#
# Usage:
#   bash scripts/batch_auto_annotator.sh \
#     --task_folder /abs/path/to/data/videos_XXXX/<task> \
#     --calibration_yaml /abs/path/to/.../realsense_calibration_*.yaml \
#     [--hand 1] [--optimize 1]
#     [--skip_existing]
#     [--models_folder /abs/path/models]       # default: <HOCAP_ROOT>/data/models
#     [--start_frame 0] [--end_frame 999]
#     [--rot_thresh 15] [--trans_thresh 0.03]
#     [--force_tool NAME]                      # override auto-match for ALL exps
#     [--dry_run]                              # only print what would run

set -u

TASK_FOLDER=""
CALIBRATION_YAML=""
HAND=""
OPTIMIZE=""
SKIP_EXISTING=0
MODELS_FOLDER=""
START_FRAME=0
END_FRAME=""
ROT_THRESH=15
TRANS_THRESH=0.03
TRACK_REFINE_ITER=10
FORCE_TOOL=""
DRY_RUN=0
H5_SCRATCH_DIR=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --task_folder)        TASK_FOLDER="$2"; shift 2;;
        --calibration_yaml)   CALIBRATION_YAML="$2"; shift 2;;
        --hand)               HAND="$2"; shift 2;;
        --optimize)           OPTIMIZE="$2"; shift 2;;
        --skip_existing)      SKIP_EXISTING=1; shift;;
        --models_folder)      MODELS_FOLDER="$2"; shift 2;;
        --start_frame)        START_FRAME="$2"; shift 2;;
        --end_frame)          END_FRAME="$2"; shift 2;;
        --rot_thresh)         ROT_THRESH="$2"; shift 2;;
        --trans_thresh)       TRANS_THRESH="$2"; shift 2;;
        --track_refine_iter)  TRACK_REFINE_ITER="$2"; shift 2;;
        --force_tool)         FORCE_TOOL="$2"; shift 2;;
        --dry_run)            DRY_RUN=1; shift;;
        --h5_scratch_dir)     H5_SCRATCH_DIR="$2"; shift 2;;
        -h|--help) sed -n '2,30p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$TASK_FOLDER" || -z "$CALIBRATION_YAML" ]]; then
    echo "Error: --task_folder and --calibration_yaml are required"; exit 1
fi
TASK_FOLDER="$(cd "$TASK_FOLDER" && pwd)"
CALIBRATION_YAML="$(readlink -f "$CALIBRATION_YAML")"
TASK_NAME="$(basename "$TASK_FOLDER")"

HOCAP_ROOT="${HOCAP_ROOT:-/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation}"
MODELS_FOLDER="${MODELS_FOLDER:-$HOCAP_ROOT/data/models}"
if [[ ! -d "$MODELS_FOLDER" ]]; then
    echo "Error: models_folder not found: $MODELS_FOLDER"; exit 1
fi

# Derive annotated root path (same derivation as run_auto_annotator.sh)
TASK_PARENT="$(dirname "$TASK_FOLDER")"
VIDEOS_FOLDER_NAME="$(basename "$TASK_PARENT")"
BASE="$(dirname "$TASK_PARENT")"
ANNOTATED_ROOT="${BASE}/${VIDEOS_FOLDER_NAME}_annotated/${TASK_NAME}"

# Manual keyword mapping file (written by scripts/data_inspector_viser.py).
# Lives next to the calibration folder at the videos_root level so every task
# under the same date shares it.
MAPPING_YAML="${TASK_PARENT}/tool_keyword_mapping.yaml"

echo "=========================================="
echo "task folder     : $TASK_FOLDER"
echo "task name       : $TASK_NAME"
echo "models folder   : $MODELS_FOLDER"
echo "calibration     : $CALIBRATION_YAML"
echo "annotated root  : $ANNOTATED_ROOT"
echo "mapping yaml    : $MAPPING_YAML $([[ -f "$MAPPING_YAML" ]] && echo '(found)' || echo '(absent — auto-match only)')"
echo "hand=$HAND  optimize=$OPTIMIZE  skip_existing=$SKIP_EXISTING  dry_run=$DRY_RUN"
[[ -n "$FORCE_TOOL" ]] && echo "force_tool      : $FORCE_TOOL"
echo "=========================================="

TOTAL=0; OK=0; SKIP=0; FAIL=0
FAILED=()
RESOLVED=()   # "exp|tool" lines for dry-run summary

for EXP_DIR in "$TASK_FOLDER"/*/; do
    [[ -d "$EXP_DIR" ]] || continue
    EXP_NAME="$(basename "$EXP_DIR")"
    [[ "$EXP_NAME" == *.tar.gz || "$EXP_NAME" == board_reference ]] && continue
    TOTAL=$((TOTAL+1))

    # --- resume check ---
    ANNO="${ANNOTATED_ROOT}/${EXP_NAME}"
    DONE_MARKER="${ANNO}/processed/fd_pose_solver/fd_poses_merged_fixed.npy"
    if [[ -n "$HAND" && "$HAND" != "0" ]]; then
        DONE_MARKER="${ANNO}/result_hand_optimized.pkl"
    fi
    if [[ "$SKIP_EXISTING" == "1" && -f "$DONE_MARKER" ]]; then
        echo ""
        echo "[$TOTAL] $EXP_NAME  [skip: $DONE_MARKER exists]"
        SKIP=$((SKIP+1)); continue
    fi

    # --- resolve tool name ---
    if [[ -n "$FORCE_TOOL" ]]; then
        TOOL="$FORCE_TOOL"
    else
        MATCH_ARGS=(
            --models_folder "$MODELS_FOLDER"
            --task_name "$TASK_NAME"
            --exp_name "$EXP_NAME"
        )
        [[ -f "$MAPPING_YAML" ]] && MATCH_ARGS+=(--mapping_yaml "$MAPPING_YAML")
        TOOL=$(python "$HOCAP_ROOT/scripts/match_tool_name.py" "${MATCH_ARGS[@]}" 2>/dev/null)
        if [[ -z "$TOOL" ]]; then
            echo ""
            echo "[$TOTAL] $EXP_NAME  [FAIL: no model matched]"
            FAIL=$((FAIL+1)); FAILED+=("$EXP_NAME:no_match")
            continue
        fi
    fi

    # --- resolve mesh path ---
    MESH=""
    for cand in "textured_mesh.obj" "cleaned_mesh_10000.obj" "mesh.obj"; do
        if [[ -f "$MODELS_FOLDER/$TOOL/$cand" ]]; then
            MESH="$MODELS_FOLDER/$TOOL/$cand"; break
        fi
    done
    if [[ -z "$MESH" ]]; then
        echo ""
        echo "[$TOTAL] $EXP_NAME  [FAIL: no mesh in $MODELS_FOLDER/$TOOL/]"
        FAIL=$((FAIL+1)); FAILED+=("$EXP_NAME:no_mesh")
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[$TOTAL] $EXP_NAME"
    echo "  tool : $TOOL"
    echo "  mesh : $MESH"
    echo "=========================================="
    RESOLVED+=("$EXP_NAME|$TOOL|$MESH")

    if [[ "$DRY_RUN" == "1" ]]; then
        continue
    fi

    # --- build run_auto_annotator args ---
    RUN_ARGS=(
        --sequence_folder "$EXP_DIR"
        --calibration_yaml "$CALIBRATION_YAML"
        --tool_name "$TOOL"
        --tool_mesh "$MESH"
        --start_frame "$START_FRAME"
        --rot_thresh "$ROT_THRESH"
        --trans_thresh "$TRANS_THRESH"
        --track_refine_iter "$TRACK_REFINE_ITER"
    )
    [[ -n "$END_FRAME" ]]       && RUN_ARGS+=(--end_frame "$END_FRAME")
    [[ -n "$HAND" ]]            && RUN_ARGS+=(--hand "$HAND")
    [[ -n "$OPTIMIZE" ]]        && RUN_ARGS+=(--optimize "$OPTIMIZE")
    [[ -n "$H5_SCRATCH_DIR" ]]  && RUN_ARGS+=(--h5_scratch_dir "$H5_SCRATCH_DIR")

    if bash "$HOCAP_ROOT/scripts/run_auto_annotator.sh" "${RUN_ARGS[@]}"; then
        OK=$((OK+1))
        echo "[$TOTAL] $EXP_NAME  [done]"
    else
        FAIL=$((FAIL+1))
        FAILED+=("$EXP_NAME:run_failed")
        echo "[$TOTAL] $EXP_NAME  [FAIL: run_auto_annotator exit $?]"
    fi
done

echo ""
echo "=========================================="
echo "Summary: total=$TOTAL  ok=$OK  skipped=$SKIP  failed=$FAIL"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "Failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
if [[ "$DRY_RUN" == "1" && "${#RESOLVED[@]}" -gt 0 ]]; then
    echo ""
    echo "Would run ${#RESOLVED[@]} experiment(s):"
    for r in "${RESOLVED[@]}"; do
        IFS='|' read -r exp tool mesh <<< "$r"
        printf "  %-60s %-25s %s\n" "$exp" "$tool" "$(basename "$mesh")"
    done
fi
echo "=========================================="
