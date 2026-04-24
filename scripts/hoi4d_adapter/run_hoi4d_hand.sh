#!/bin/bash
# Convert a HOI4D-style sequence and run the hand-only annotation pipeline,
# producing both the pipeline's result_hand_optimized.pkl and a
# gt_result_hand_optimized.pkl (converted from HOI4D's own MANO labels)
# side-by-side for comparison.
#
# Usage:
#   bash scripts/hoi4d_adapter/run_hoi4d_hand.sh \
#     --hoi4d_seq /abs/.../ZY20210800003_H3_C20_N11_S279_s03_T1 \
#     --out_root  /abs/.../HO-Cap-Annotation/data \
#     [--skip_convert]      # convert step already done
#     [--skip_reconstruct]  # only convert, don't run the hand pipeline
#     [--num_frames_limit 50]  # smoke-test with first 50 frames

set -u

HOI4D_SEQ=""
OUT_ROOT=""
VIDEO_FOLDER_NAME="hoi4d"
TASK_NAME="hoi4d_seqs"
TOOL_NAME="hoi4d_obj"
NUM_FRAMES_LIMIT=""
SKIP_CONVERT=0
SKIP_RECONSTRUCT=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --hoi4d_seq)         HOI4D_SEQ="$2"; shift 2;;
        --out_root)          OUT_ROOT="$2"; shift 2;;
        --video_folder_name) VIDEO_FOLDER_NAME="$2"; shift 2;;
        --task_name)         TASK_NAME="$2"; shift 2;;
        --tool_name)         TOOL_NAME="$2"; shift 2;;
        --num_frames_limit)  NUM_FRAMES_LIMIT="$2"; shift 2;;
        --skip_convert)      SKIP_CONVERT=1; shift;;
        --skip_reconstruct)  SKIP_RECONSTRUCT=1; shift;;
        -h|--help)           sed -n '2,15p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$HOI4D_SEQ" || -z "$OUT_ROOT" ]]; then
    echo "Error: --hoi4d_seq and --out_root are required"; exit 1
fi
HOI4D_SEQ="$(cd "$HOI4D_SEQ" && pwd)"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
SEQ_NAME="$(basename "$HOI4D_SEQ")"

HOCAP_ROOT="${HOCAP_ROOT:-/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation}"
HAND_ROOT="${HAND_ROOT:-/home/ruoqu/crq_ws/robotool/HandReconstruction}"

# Conda activation (tries common paths)
for _sh in "${CONDA_SH:-}" \
           "/home/ruoqu/miniconda3/etc/profile.d/conda.sh" \
           "/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh" \
           "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    [[ -n "$_sh" && -f "$_sh" ]] && { source "$_sh"; break; }
done

SEQ_OUT="${OUT_ROOT}/${VIDEO_FOLDER_NAME}/${TASK_NAME}/${SEQ_NAME}"
ANNOTATED="${OUT_ROOT}/${VIDEO_FOLDER_NAME}_annotated/${TASK_NAME}/${SEQ_NAME}"
CALIB_YAML="${OUT_ROOT}/${VIDEO_FOLDER_NAME}_calib/calibration_hoi4d_${SEQ_NAME}.yaml"

echo "=========================================="
echo "HOI4D seq   : $HOI4D_SEQ"
echo "out_root    : $OUT_ROOT"
echo "seq_out     : $SEQ_OUT"
echo "annotated   : $ANNOTATED"
echo "calib       : $CALIB_YAML"
echo "=========================================="

# ---- step 1: convert ----
if [[ "$SKIP_CONVERT" == "0" ]]; then
    conda activate hocap-annotation
    CONV_ARGS=(
        --hoi4d_seq "$HOI4D_SEQ"
        --out_root "$OUT_ROOT"
        --video_folder_name "$VIDEO_FOLDER_NAME"
        --task_name "$TASK_NAME"
        --tool_name "$TOOL_NAME"
    )
    [[ -n "$NUM_FRAMES_LIMIT" ]] && CONV_ARGS+=(--num_frames_limit "$NUM_FRAMES_LIMIT")
    cd "$HOCAP_ROOT"
    python scripts/hoi4d_adapter/convert_hoi4d.py "${CONV_ARGS[@]}" || { echo "[ERR] convert"; exit 1; }
fi

if [[ "$SKIP_RECONSTRUCT" == "1" ]]; then
    echo "[done] --skip_reconstruct set"; exit 0
fi

# ---- step 2: cluster_reconstruct (the existing hand pipeline, single-camera) ----
# The pipeline calls MediaPipe/WiLoR per frame and still runs get_hand_root()
# with 1 ray — it produces SOME result, degenerate in depth but usable as a
# comparison baseline (WiLoR itself is per-image monocular).
cd "$HAND_ROOT"
conda activate reconstruct-hand

rm -f "${ANNOTATED}/result.pkl" \
      "${ANNOTATED}/result_hand_optimized.pkl" \
      "${ANNOTATED}/poses_m.npy"

if ! python cluster_reconstruct.py --sequence_folder "$SEQ_OUT"; then
    echo "[ERR] cluster_reconstruct failed"; exit 1
fi

# ---- step 3: optimize ----
RESULT="${ANNOTATED}/result.pkl"
if [[ -f "$RESULT" ]]; then
    if ! python cluster_optimize_hand.py --file_name "$RESULT"; then
        echo "[WARN] optimize failed — reconstruction-only result still saved."
    fi
else
    echo "[WARN] $RESULT not produced; skipping optimize"
fi

# ---- step 4: summary ----
echo ""
echo "=========================================="
echo "Comparison files:"
ls -la "${ANNOTATED}/gt_result_hand_optimized.pkl" 2>/dev/null
ls -la "${ANNOTATED}/result_hand_optimized.pkl"    2>/dev/null
ls -la "${ANNOTATED}/result.pkl"                   2>/dev/null
echo ""
echo "Visualize (side-by-side via visualize_hand_viser):"
echo "  python scripts/visualize_hand_viser.py --pkl_path ${ANNOTATED}/result_hand_optimized.pkl"
echo "  python scripts/visualize_hand_viser.py --pkl_path ${ANNOTATED}/gt_result_hand_optimized.pkl"
echo "=========================================="
