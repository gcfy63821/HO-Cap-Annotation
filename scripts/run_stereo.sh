#!/bin/bash
#
# run_stereo.sh — 从双目视频生成深度+彩色图，打包为独立的单相机 sequence
#
# 流程:
#   1. [ffs env]  标定双目相机（如果没有标定文件）
#   2. [ffs env]  Fast-FoundationStereo 生成 depth + rectified color
#   3. [ffs env]  将 color/depth 转换为 cam8_rgb.mp4 + cam8_depth.mkv
#   4. 输出到 {sequence_folder}_stereo/ 文件夹
#
# 之后手动流程:
#   - 用 batch_task_annotator 标注 mask
#   - 用 run_local.sh / run_mydata.sh 跑 pipeline
#
# 用法:
#   cd /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation
#   bash scripts/run_stereo.sh \
#     --sequence_folder /path/to/20260122_squeegee_collect_sand_from_table_20
#
#   # 或指定已有标定文件:
#   bash scripts/run_stereo.sh \
#     --sequence_folder /path/to/sequence \
#     --calib_file /path/to/stereo_calib.npz

set -e

# ============================================================================
# 默认参数
# ============================================================================
SEQUENCE_FOLDER=""
FFS_DIR="/home/ruoqu/crq_ws/robotool/Fast-FoundationStereo"
CALIB_FILE=""
TAG_SIZE="0.05"
TAG_GAP="0.01"
APPROX_DISTANCE="1.0"
CAM_ID="8"
START_FRAME="0"
END_FRAME=""
SKIP_FFS="0"

# ============================================================================
# 解析命令行参数
# ============================================================================
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sequence_folder) SEQUENCE_FOLDER="$2"; shift 2 ;;
        --calib_file) CALIB_FILE="$2"; shift 2 ;;
        --tag_size) TAG_SIZE="$2"; shift 2 ;;
        --tag_gap) TAG_GAP="$2"; shift 2 ;;
        --approx_distance) APPROX_DISTANCE="$2"; shift 2 ;;
        --cam_id) CAM_ID="$2"; shift 2 ;;
        --start_frame) START_FRAME="$2"; shift 2 ;;
        --end_frame) END_FRAME="$2"; shift 2 ;;
        --skip_ffs) SKIP_FFS="$2"; shift 2 ;;
        *) echo "Unknown option $1"; exit 1 ;;
    esac
done

if [ -z "$SEQUENCE_FOLDER" ]; then
    echo "Error: --sequence_folder is required."
    echo "Usage: bash scripts/run_stereo.sh --sequence_folder /path/to/sequence_folder"
    exit 1
fi

# ============================================================================
# 路径设置
# ============================================================================
LEFT_VIDEO="${SEQUENCE_FOLDER}/main_cam_left_video.mp4"
RIGHT_VIDEO="${SEQUENCE_FOLDER}/main_cam_right_video.mp4"
STEREO_FOLDER="${SEQUENCE_FOLDER}_stereo"
FFS_OUTPUT="${STEREO_FOLDER}/ffs_output"
FFS_CALIB_DIR="${STEREO_FOLDER}/stereo_calib"

echo "============================================"
echo "Stereo Depth Generation Pipeline"
echo "============================================"
echo "Source:  $SEQUENCE_FOLDER"
echo "Output:  $STEREO_FOLDER"
echo "Cam ID:  $CAM_ID"
echo "============================================"

# 检查视频文件
if [ ! -f "$LEFT_VIDEO" ] || [ ! -f "$RIGHT_VIDEO" ]; then
    echo "Error: main_cam_left_video.mp4 or main_cam_right_video.mp4 not found"
    exit 1
fi

source /home/ruoqu/miniconda3/etc/profile.d/conda.sh
mkdir -p "$STEREO_FOLDER"

# ============================================================================
# Step 1: 标定（如果没有标定文件）
# ============================================================================
if [ "$SKIP_FFS" != "1" ]; then
    conda activate ffs
    cd "$FFS_DIR"

    if [ -z "$CALIB_FILE" ] || [ ! -f "$CALIB_FILE" ]; then
        echo ""
        echo "===== Step 1: Stereo Calibration ====="
        mkdir -p "$FFS_CALIB_DIR"
        python scripts/stereo_calibrate.py \
            --left_video "$LEFT_VIDEO" \
            --right_video "$RIGHT_VIDEO" \
            --tag_size "$TAG_SIZE" \
            --tag_gap "$TAG_GAP" \
            --approx_distance "$APPROX_DISTANCE" \
            --out_dir "$FFS_CALIB_DIR"
        CALIB_FILE="${FFS_CALIB_DIR}/stereo_calib.npz"
    else
        echo "Using existing calibration: $CALIB_FILE"
    fi

    # ============================================================================
    # Step 2: FFS 深度估计
    # ============================================================================
    echo ""
    echo "===== Step 2: Stereo Depth Estimation ====="
    FFS_ARGS="--left_video $LEFT_VIDEO --right_video $RIGHT_VIDEO \
        --calib_file $CALIB_FILE \
        --save_color 1 --save_vis 1 \
        --out_dir $FFS_OUTPUT \
        --start_frame $START_FRAME"
    if [ -n "$END_FRAME" ]; then
        FFS_ARGS="$FFS_ARGS --end_frame $END_FRAME"
    fi
    python scripts/run_stereo_video.py $FFS_ARGS
fi

# ============================================================================
# Step 3: 转换为 pipeline 兼容格式
# ============================================================================
echo ""
echo "===== Step 3: Convert to pipeline format ====="
conda activate ffs
cd "$FFS_DIR"

python scripts/pack_stereo_sequence.py \
    --color_dir "${FFS_OUTPUT}/color" \
    --depth_dir "${FFS_OUTPUT}/depth" \
    --calib_file "${CALIB_FILE:-${FFS_CALIB_DIR}/stereo_calib.npz}" \
    --output_dir "$STEREO_FOLDER" \
    --cam_id "$CAM_ID" \
    --start_frame "$START_FRAME"

echo ""
echo "============================================"
echo "Done! Stereo sequence created at:"
echo "  $STEREO_FOLDER"
echo ""
echo "Contents:"
ls -lh "$STEREO_FOLDER"/cam${CAM_ID}_rgb.mp4 "$STEREO_FOLDER"/cam${CAM_ID}_depth.mkv 2>/dev/null
echo ""
echo "Next steps:"
echo "  1. Annotate masks (SAM2 or batch_task_annotator)"
echo "  2. Run pipeline:"
echo "     cd /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation"
echo "     bash scripts/run_local.sh \\"
echo "       --sequence_name <relative_path_to_stereo_folder> \\"
echo "       --tool_name <tool_name>"
echo "============================================"
