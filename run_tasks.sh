#!/bin/bash

# 默认参数
SEQUENCE_NAME=""
OBJECT_IDX=()
OUTPUT_IDX=""
TOOL_NAME=""
BASE_PATH="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/"
OPTIMIZE=""
UUID=""
TRACK_REFINE_ITER="20"
HAND=""
TASKS=""
# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sequence_name)
            SEQUENCE_NAME="$2"
            shift 2
            ;;
        --object_idx)
            OBJECT_IDX=($2)
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
        --tasks)
            TASKS="$2"
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

# 根据指定任务运行相关代码
run_hand_detection_and_generation() {
    echo "Running hand detection"
    python tools/02_mp_hand_detection.py --sequence_folder "$SEQUENCE_FOLDER"

    echo "Running hand 3d generation"
    python tools/03_mp_3d_joints_generation.py --sequence_folder "$SEQUENCE_FOLDER"
}

run_fd_pose_solver() {
    for object_idx in "${OBJECT_IDX[@]}"; do
        echo "Running fd_pose_solver with object_idx=$object_idx..."
        python tools/04-1-1_fd_pose_solver_prep.py --sequence_folder "$SEQUENCE_FOLDER" --object_idx "$object_idx" --track_refine_iter "$TRACK_REFINE_ITER"
    done
}

run_fd_merger() {
    echo "Running fd_pose_merger..."
    python tools/04-2_fd_pose_merger.py --sequence_folder "$SEQUENCE_FOLDER"
}

run_optimize() {
    echo "Running optimize_fd_pose..."
    python tools/05_mano_pose_solver.py --sequence_folder "$SEQUENCE_FOLDER"

    echo "Running optimize_fd_pose..."
    python tools/06_object_pose_solver.py --sequence_folder "$SEQUENCE_FOLDER"

    echo "Running joint_pose_optimization..."
    python tools/07_joint_pose_solver.py --sequence_folder "$SEQUENCE_FOLDER"
    
    echo "Running visualize_and_evaluate_result..."
    python debug/visualize_hand_video.py --data_path "$SEQUENCE_NAME" --tool_name "$TOOL_NAME" --object_idx "${OBJECT_IDX[0]}" --uuid "$UUID"
}

run_all_tasks() {
    run_hand_detection_and_generation
    run_fd_pose_solver
    run_fd_merger
    run_optimize
}

# 执行任务
if [ "$TASKS" == "all" ]; then
    run_all_tasks
elif [ "$TASKS" == "optimize" ]; then
    run_optimize
elif [ "$TASKS" == "hand" ]; then
    run_hand_detection_and_generation
elif [ "$TASKS" == "fd_merger" ]; then
    run_fd_merger
else
    IFS=', ' read -r -a TASK_LIST <<< "$TASKS"
    for task in "${TASK_LIST[@]}"; do
        case $task in
            1)
                run_hand_detection_and_generation
                ;;
            2)
                run_fd_pose_solver
                ;;
            3)
                run_fd_merger
                ;;
            4)
                run_optimize
                ;;
            *)
                echo "Unknown task number: $task"
                ;;
        esac
    done
fi

echo "All tasks completed!"

# ./run_tasks.sh --sequence_name my_sequence --tasks 1,2 --object_idx 1 2
