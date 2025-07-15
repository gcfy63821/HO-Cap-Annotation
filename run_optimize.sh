#!/bin/bash

# 默认参数
SEQUENCE_NAME=""
OBJECT_IDX="1"
OUTPUT_IDX="3"
TOOL_NAME=""
BASE_PATH="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/"
OPTIMIZE=""
UUID=""
TRACK_REFINE_ITER="10"
HAND=""
# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sequence_name)
            SEQUENCE_NAME="$2"
            shift 2
            ;;
        --object_idx)
            OBJECT_IDX="$2"
            shift 2
            ;;
        --tool_name)
            TOOL_NAME="$2"
            shift 2
            ;;
        --output_idx)
            OUTPUT_IDX="$2"
            shift 2
            ;;
        --optimize)
            OPTIMIZE="$2"
            shift 2
            ;;
        --track_refine_iter)
            TRACK_REFINE_ITER="$2"
            shift 2
            ;;
        --uuid)
            UUID="$2"
            shift 2
            ;;
        --hand)
            HAND="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# 检查必需的参数是否提供
if [ -z "$SEQUENCE_NAME" ]; then
    echo "Error: --sequence_name is required."
    exit 1
fi

# 自动拼接文件夹路径
SEQUENCE_FOLDER="${BASE_PATH}${SEQUENCE_NAME}"
TOOL_NAME=${TOOL_NAME:-""}  # 默认值为空





if [ -n "$OPTIMIZE" ]; then
    # echo "Running optimize_fd_pose with optimize=$OPTIMIZE..."
    # python tools/05_mano_pose_solver.py --sequence_folder "$SEQUENCE_FOLDER"

    echo "Running optimize_fd_pose with optimize=$OPTIMIZE..."
    python tools/06_object_pose_solver.py --sequence_folder "$SEQUENCE_FOLDER"

    echo "Running joint_pose_optimization with optimize=$OPTIMIZE..."
    python tools/07_joint_pose_solver.py --sequence_folder "$SEQUENCE_FOLDER"
    # echo "Running visualize_ob_in_world with tool_name=$TOOL_NAME..."
    # python debug/visualize_ob_in_world.py --data_path "$SEQUENCE_NAME" --tool_name "$TOOL_NAME" --output_idx "$OUTPUT_IDX" --uuid "$UUID" --pose_file "optimized" --uuid "$UUID" --object_idx "$OBJECT_IDX"

fi

echo "All tasks completed!"

# ./run_local.sh --sequence_name blue_scooper_1/20250704_172530 --tool_name blue_scooper --uuid mask_depth_and_object >> log.txt 2>&1