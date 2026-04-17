#!/bin/bash
# Test depth validation improvement on grasp_hammer_2
# Runs tracking twice: without and with depth validation, then compares results.

set -e

SEQUENCE_FOLDER="/home/ruoqu/crq_ws/robotool/DataCollection/data/grasp_hammer/grasp_hammer_2"
TOOL_NAME="yellow_hammer"
OBJECT_IDX=1

source /home/ruoqu/miniconda3/etc/profile.d/conda.sh
conda activate hocap-annotation
cd /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation

echo "============================================"
echo "RUN 1: Baseline (no depth validation)"
echo "============================================"
python tools/04-1-4_fd_pose_solver_kalman.py \
    --no_masked_depth \
    --sequence_folder "$SEQUENCE_FOLDER" \
    --activate_2d_tracker \
    --activate_kalman_filter \
    --object_idx "$OBJECT_IDX" \
    --track_refine_iter 10 \
    --rot_thresh 15 \
    --trans_thresh 0.03 \
    --output_suffix "fd_pose_solver_baseline"

echo ""
echo "============================================"
echo "RUN 2: With depth validation"
echo "============================================"
python tools/04-1-4_fd_pose_solver_kalman.py \
    --no_masked_depth \
    --sequence_folder "$SEQUENCE_FOLDER" \
    --activate_2d_tracker \
    --activate_kalman_filter \
    --object_idx "$OBJECT_IDX" \
    --track_refine_iter 10 \
    --rot_thresh 15 \
    --trans_thresh 0.03 \
    --depth_validation \
    --depth_error_thresh 0.02 \
    --output_suffix "fd_pose_solver_depthval"

echo ""
echo "============================================"
echo "Comparing results"
echo "============================================"
python tools/compare_tracking_results.py \
    --sequence_folder "$SEQUENCE_FOLDER" \
    --baseline_suffix "fd_pose_solver_baseline" \
    --improved_suffix "fd_pose_solver_depthval" \
    --object_idx "$OBJECT_IDX"
