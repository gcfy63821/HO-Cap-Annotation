#!/bin/bash
# Run the full pipeline on every experiment folder inside a task folder.
#
# Usage:
#   ./run_task_folder.sh <task_folder_name> [mode]
#
#   mode:
#     full  (default) -- run full pipeline incl. 06-2 object pose solver and
#                        07-2 joint pose solver.
#     fake            -- skip 06-2 and 07-2; copy fd_pose_merger's
#                        fd_poses_merged_fixed.npy into joint_pose_solver/poses_o.npy
#                        so downstream code that consumes the optimized output
#                        still works.
#
# Example:
#   ./run_task_folder.sh 0315_2          # full
#   ./run_task_folder.sh 0315_2 fake     # skip optimization, fake the output

set -e

TASK_FOLDER="${1:?Usage: $0 <task_folder_name> [full|fake]  (e.g. 0315_2 fake)}"
MODE="${2:-full}"
case "$MODE" in
    full|fake) ;;
    *) echo "Error: mode must be 'full' or 'fake' (got '$MODE')"; exit 1 ;;
esac

DATA_BASE="/home/ruoqu/crq_ws/robotool/DataCollection/data"
ANNOTATED_BASE="/home/ruoqu/crq_ws/robotool/DataCollection/data_annotated"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_PATH="${DATA_BASE}/${TASK_FOLDER}"

if [ ! -d "$TASK_PATH" ]; then
    echo "Error: task folder not found: $TASK_PATH"
    exit 1
fi

echo "=========================================="
echo "Task folder: $TASK_PATH"
echo "Mode:        $MODE"
echo "=========================================="

for EXP_DIR in "$TASK_PATH"/*/; do
    EXP_NAME="$(basename "$EXP_DIR")"

    # Extract tool_name: remove trailing _<digits>
    TOOL_NAME="$(echo "$EXP_NAME" | sed 's/_[0-9]*$//')"

    echo ""
    echo "=========================================="
    echo "Experiment: $EXP_NAME"
    echo "Tool name:  $TOOL_NAME"
    echo "=========================================="

    # Skip if already completed (joint_pose_solver/poses_o.npy is the final
    # output of both 'full' and 'fake' modes). Set FORCE=1 to re-run.
    ANNOTATED_EXP="${ANNOTATED_BASE}/${TASK_FOLDER}/${EXP_NAME}"
    DONE_MARKER="${ANNOTATED_EXP}/processed/joint_pose_solver/poses_o.npy"
    if [ -z "${FORCE:-}" ] && [ -f "$DONE_MARKER" ]; then
        echo "[skip] already completed: $DONE_MARKER (set FORCE=1 to re-run)"
        continue
    fi

    # object_idx is omitted: run_mydata.sh auto-detects from meta.yaml
    # and tracks all objects. Optimization always uses object 1.
    RUN_ARGS=(
        --sequence_name "data/${TASK_FOLDER}/${EXP_NAME}"
        --tool_name "$TOOL_NAME"
        --hand 1
        --uuid "${EXP_NAME}"
    )
    if [ "$MODE" = "full" ]; then
        RUN_ARGS+=(--optimize 1)
    fi

    "$SCRIPT_DIR/run_mydata.sh" "${RUN_ARGS[@]}"

    if [ "$MODE" = "fake" ]; then
        SRC="${ANNOTATED_EXP}/processed/fd_pose_solver/fd_poses_merged_fixed.npy"
        DST_DIR="${ANNOTATED_EXP}/processed/joint_pose_solver"
        DST="${DST_DIR}/poses_o.npy"

        if [ ! -f "$SRC" ]; then
            echo "Error: fd_pose_merger output not found: $SRC"
            exit 1
        fi
        mkdir -p "$DST_DIR"
        cp -f "$SRC" "$DST"
        echo "[fake optimize] copied $SRC -> $DST"
    fi

done

echo ""
echo "=========================================="
echo "All experiments in ${TASK_FOLDER} completed!"
echo "=========================================="
