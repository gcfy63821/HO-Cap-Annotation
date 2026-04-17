#!/bin/bash
# Run the full pipeline on every experiment folder inside a task folder.
#
# Usage:
#   ./run_task_folder.sh <task_folder_name>
#
# Example:
#   ./run_task_folder.sh 0315_2
#
# For each subfolder like "small_brush_1", it extracts tool_name="small_brush"
# and runs run_mydata.sh with the appropriate arguments.

set -e

TASK_FOLDER="${1:?Usage: $0 <task_folder_name>  (e.g. 0315_2)}"
DATA_BASE="/home/ruoqu/crq_ws/robotool/DataCollection/data"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_PATH="${DATA_BASE}/${TASK_FOLDER}"

if [ ! -d "$TASK_PATH" ]; then
    echo "Error: task folder not found: $TASK_PATH"
    exit 1
fi

echo "=========================================="
echo "Task folder: $TASK_PATH"
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

    # object_idx is omitted: run_mydata.sh auto-detects from meta.yaml
    # and tracks all objects. Optimization always uses object 1.
    "$SCRIPT_DIR/run_mydata.sh" \
        --sequence_name "data/${TASK_FOLDER}/${EXP_NAME}" \
        --tool_name "$TOOL_NAME" \
        --hand 1 \
        --optimize 1 \
        --uuid "${EXP_NAME}"

done

echo ""
echo "=========================================="
echo "All experiments in ${TASK_FOLDER} completed!"
echo "=========================================="
