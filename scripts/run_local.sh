#!/bin/bash


# 默认参数
SEQUENCE_NAME=""
OBJECT_IDX=""
OUTPUT_IDX=""
TOOL_NAME=""
BASE_PATH="/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/"
OPTIMIZE=""
UUID=""
TRACK_REFINE_ITER="10"
HAND=""
ROT_THRESH="15"
TRANS_THRESH="0.02"
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
        --rot_thresh)
            ROT_THRESH="$2"
            shift 2
            ;;
        --trans_thresh)
            TRANS_THRESH="$2"
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

echo "START RUNNING"
# 自动拼接文件夹路径
SEQUENCE_FOLDER="${BASE_PATH}${SEQUENCE_NAME}"
TOOL_NAME=${TOOL_NAME:-""}  # 默认值为空
H5_PATH="${SEQUENCE_FOLDER}/data00000000.h5"

# Updated path parsing for new structure: videos_0901/taskname/xxxvideoname
# Extract video folder (e.g., videos_0901), task name, and video name
VIDEO_FOLDER="${SEQUENCE_NAME%%/*}"  # e.g., videos_0901
REMAINING_PATH="${SEQUENCE_NAME#*/}"  # e.g., taskname/xxxvideoname
TASK_NAME="${REMAINING_PATH%%/*}"    # e.g., taskname
VIDEO_NAME="${REMAINING_PATH#*/}"    # e.g., xxxvideoname

# Create annotated path with taskname included: videos_0901_annotated/taskname/xxxvideoname
ANNOTATED_PATH="${BASE_PATH}${VIDEO_FOLDER}_annotated/${TASK_NAME}/${VIDEO_NAME}"



source /home/ruoqu/miniconda3/etc/profile.d/conda.sh

conda activate hocap-annotation

echo "START RUNNING"
cd /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation

# python tools/00_convert_videos_to_h5.py --input_dir "$SEQUENCE_FOLDER" --output_file "$H5_PATH"

# python preprocess/generate_meta.py \
#     --h5_path "$H5_PATH" \
#     --calibration_yaml_path /home/ruoqu/crq_ws/HO-Cap-Annotation/data/videos_1121/realsense_calibration_1121.yaml \
#     --models_folder /home/ruoqu/crq_ws/HO-Cap-Annotation/data/models \
#     --tool_name "$TOOL_NAME"


# python preprocess/generate_meta.py \
#     --h5_path "$H5_PATH" \
#     --calibration_yaml_path /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/videos_0121/realsense_calibration_0121.yaml \
#     --models_folder /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/models \
#     --tool_name "$TOOL_NAME"

# # # # # 运行 04-1-1_fd_pose_solver_prep.py
if [ -n "$OBJECT_IDX" ]; then
    echo "Running fd_pose_solver with object_idx=$OBJECT_IDX..."
    # python tools/04-1-2_fd_pose_solver_cluster.py --sequence_folder "$SEQUENCE_FOLDER" --object_idx "$OBJECT_IDX" --track_refine_iter "$TRACK_REFINE_ITER" --rot_thresh "$ROT_THRESH" --trans_thresh "$TRANS_THRESH"
    # python tools/04-1-4_fd_pose_solver_separate_cluster.py --sequence_folder "$SEQUENCE_FOLDER" --object_idx "$OBJECT_IDX" --track_refine_iter "$TRACK_REFINE_ITER" --rot_thresh "$ROT_THRESH" --trans_thresh "$TRANS_THRESH"
    # python tools/04-1-5_fd_pose_solver_icp_cluster.py --sequence_folder "$SEQUENCE_FOLDER" --object_idx "$OBJECT_IDX" --track_refine_iter "$TRACK_REFINE_ITER" --rot_thresh "$ROT_THRESH" --trans_thresh "$TRANS_THRESH" 
    # python tools/04-1-4_fd_pose_solver_kalman.py --sequence_folder "$SEQUENCE_FOLDER" --activate_2d_tracker --activate_kalman_filter --object_idx "$OBJECT_IDX" --track_refine_iter "$TRACK_REFINE_ITER" --rot_thresh "$ROT_THRESH" --trans_thresh "$TRANS_THRESH" 
    
fi

# python debug/visualize_tracking_result_in_cam.py     --data_path "$SEQUENCE_FOLDER"     --tool_name "$TOOL_NAME"     --object_idx "$OBJECT_IDX"     --extract_images --num_workers 1

# # # # 运行 04-2_fd_pose_merger.py
# echo "Running fd_pose_merger..."
# python tools/04-2-2_fd_pose_merger_cluster.py --sequence_folder "$SEQUENCE_FOLDER"

# # # 运行 04-2-1_adaptive_fd_merger.py
# # echo "Running adaptive fd_pose_merger..."
# # python tools/04-2-1_adaptive_fd_merger.py --sequence_folder "$SEQUENCE_FOLDER"
# python debug/visualize_cluster_video_fast.py --data_path "$SEQUENCE_FOLDER" --tool_name "$TOOL_NAME" --object_idx "$OBJECT_IDX" --uuid "$UUID" --pose_file "fd" --render_hands --num_workers 1
    

# 运行 visualize_ob_in_world.py
# if [ -n "$TOOL_NAME" ]; then
# #     # echo "Running visualize_ob_in_world with tool_name=$TOOL_NAME..."
#     python debug/visualize_ob_in_world.py --data_path "$SEQUENCE_FOLDER" --tool_name "$TOOL_NAME" --output_idx "$OUTPUT_IDX" --uuid "ob_vis"  --object_idx "$OBJECT_IDX" --num_workers 1

# #     # echo "Running visualize_ob_in_world with tool_name=$TOOL_NAME..."
# #     # python debug/visualize_ob_in_world.py --data_path "$SEQUENCE_NAME" --tool_name "$TOOL_NAME" --output_idx "$OUTPUT_IDX" --uuid "$UUID " --object_idx "$OBJECT_IDX" --pose_file "adaptive"
#     python debug/visualize_cluster_video_fast.py --data_path "$SEQUENCE_FOLDER" --tool_name "$TOOL_NAME" --object_idx "$OBJECT_IDX" --uuid "$UUID " --pose_file "fd" --render_hands --num_workers 1
#     # python debug/visualize_cluster_video_fast.py --data_path "$SEQUENCE_FOLDER" --tool_name "$TOOL_NAME" --object_idx "$OBJECT_IDX" --uuid "npy" --pose_file "fd" --render_hands --num_workers 1
    
#     # python debug/visualize_tracking_result_cluster_fast.py  --data_path "$SEQUENCE_FOLDER" 
# fi

# added evaluation

# 运行 visualize 并且 分析结果
# if [ -n "$TOOL_NAME" ]; then
#     echo "Running visualize_and_evaluate_result with tool_name=$TOOL_NAME..."
#     python debug/visualize_and_evaluate_result.py --data_path "$SEQUENCE_NAME" --tool_name "$TOOL_NAME" --output_idx "$OUTPUT_IDX" --uuid "$UUID"  --object_idx "$OBJECT_IDX"

#     echo "Running visualize_and_evaluate_result with tool_name=$TOOL_NAME..."
#     python debug/visualize_and_evaluate_result.py --data_path "$SEQUENCE_NAME" --tool_name "$TOOL_NAME" --output_idx "$OUTPUT_IDX" --uuid "$UUID " --object_idx "$OBJECT_IDX" --pose_file "adaptive"
# fi


# if [ -n "$HAND" ]; then
# ./reconstruct_hand.sh --sequence_name "$SEQUENCE_NAME"

#     echo "Runnning hand detection"
#     python tools/02_mp_hand_detection.py --sequence_folder "$SEQUENCE_FOLDER"

#     echo "Running hand 3d generation"
#     python tools/03_mp_3d_joints_generation.py --sequence_folder "$SEQUENCE_FOLDER"

# fi


if [ -n "$OPTIMIZE" ]; then
    # echo "Running optimize_fd_pose with optimize=$OPTIMIZE..."
    # python tools/05_mano_pose_solver.py --sequence_folder "$SEQUENCE_FOLDER"

    # echo "Running optimize_fd_pose with optimize=$OPTIMIZE..."
    python tools/06-2_object_pose_solver_cluster.py --sequence_folder "$SEQUENCE_FOLDER" --debug

    echo "Running joint_pose_optimization with optimize=$OPTIMIZE..."
    python tools/07-2_joint_pose_solver_cluster.py --sequence_folder "$SEQUENCE_FOLDER" --debug
    
    echo "Running visualize_and_evaluate_result with tool_name=$TOOL_NAME..."
    # python debug/visualize_hand_video.py --data_path "$SEQUENCE_NAME" --tool_name "$TOOL_NAME" --object_idx "$OBJECT_IDX" --uuid "$UUID "
    # python debug/visualize_hand_video.py --data_path "$SEQUENCE_FOLDER" --tool_name "$TOOL_NAME" --object_idx "$OBJECT_IDX" --uuid "$UUID "
    
    # python debug/visualize_cluster_video.py --data_path "$SEQUENCE_NAME" --tool_name "$TOOL_NAME" --object_idx "$OBJECT_IDX" --uuid "$UUID " --pose_file "optimized" --render_hands
    python debug/visualize_cluster_video_fast.py --data_path "$SEQUENCE_FOLDER" --tool_name "$TOOL_NAME" --object_idx "$OBJECT_IDX" --uuid "$UUID " --pose_file "optimized" --render_hands --num_workers 1
    
fi

# echo "All tasks completed!"

# ./run_local.sh --sequence_name blue_scooper_1/20250704_172530 --tool_name blue_scooper --uuid mask_depth_and_object >> log.txt 2>&1
