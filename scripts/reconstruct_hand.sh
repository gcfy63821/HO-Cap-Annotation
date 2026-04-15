#!/bin/bash
#SBATCH --account viscam 
#SBATCH --job-name generate_viz
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1 
#SBATCH --mem=64G
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4
#SBATCH --output=/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation/slurm_outs/%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation/slurm_outs/%j.err
# General purpose script for running FoundationPose estimation on both tool and target objects


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

# hand reconstruction
cd /home/ruoqu/crq_ws/robotool/HandReconstruction


conda activate reconstruct-hand

python cluster_reconstruct.py --sequence_folder "$SEQUENCE_FOLDER"

echo "Done reconstructing hand"

SAVED_FILE="${ANNOTATED_PATH}/result.pkl"

python cluster_optimize_hand.py --file_name "$SAVED_FILE"




echo "All tasks completed!"

# ./run_local.sh --sequence_name blue_scooper_1/20250704_172530 --tool_name blue_scooper --uuid mask_depth_and_object >> log.txt 2>&1
