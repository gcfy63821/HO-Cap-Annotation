#!/bin/bash
# Full annotation pipeline with DINOv2-based automatic tool segmentation.
# Replaces the interactive SAM2 mask step with tools/dino_tool_segment.py.
#
# Stages (all single-sequence, single-tool):
#   1. videos -> h5                       (tools/00_convert_videos_to_h5.py)
#   2. DINOv2+SAM2 auto tool mask         (tools/dino_tool_segment.py)
#      + export to pipeline format       (--pipeline_tool_masks_dir)
#   3. generate_meta                      (preprocess/generate_meta.py)
#   4. per-object 6-DoF tracking          (tools/04-1-4_fd_pose_solver_kalman.py)
#   5. multi-view pose merge              (tools/04-2-2_fd_pose_merger_cluster.py)
#   6. (optional) hand reconstruction     (HandReconstruction/cluster_reconstruct.py
#                                          + cluster_optimize_hand.py)
#   7. (optional) object + joint optim    (tools/06-2_*, 07-2_*)
#
# Expected sequence folder layout (before running):
#   <sequence_folder>/
#     cam{i}_rgb.mp4
#     cam{i}_depth.mkv
#
# Results go to:
#   <sequence_folder>/data00000000.h5
#   <sequence_folder>/meta.yaml
#   <videos_root>_annotated/<task>/<sequence>/tool_masks/...
#   <videos_root>_annotated/<task>/<sequence>/{result.pkl,result_hand_optimized.pkl,...}
#   <videos_root>_annotated/<task>/<sequence>/processed/...
#
# Usage:
#   bash scripts/run_auto_annotator.sh \
#     --sequence_folder /abs/path/to/data/videos_XXXX/task/exp \
#     --calibration_yaml /abs/path/to/realsense_calibration_*.yaml \
#     --tool_name mallet                                # mesh auto-resolved
#     [--tool_mesh /abs/path/to/mesh.obj]               # override if needed
#     [--models_folder /abs/path/to/models]             # default $MODELS_FOLDER
#     [--hand 1] [--optimize 1]
#     [--start_frame 0] [--end_frame 999]
#     [--skip_h5] [--skip_masks] [--skip_tracking]
#     [--rot_thresh 15] [--trans_thresh 0.03] [--track_refine_iter 10]
#     [--frame0_only]       # DINO-register only on frame 0, no re-seeding,
#                           # skips Phase 3-7. Fastest on slow/cluster GPUs.
#
# Mesh auto-resolution: when --tool_mesh is omitted, the script looks in
#   $MODELS_FOLDER/$TOOL_NAME/ for one of (in order):
#     textured_mesh.obj, cleaned_mesh_10000.obj, mesh.obj
# MODELS_FOLDER falls back to $HOCAP_ROOT/data/models if the env var isn't set.

set -eu

# ---------- defaults ----------
SEQUENCE_FOLDER=""
CALIBRATION_YAML=""
TOOL_NAME=""
TOOL_MESH=""
HAND=""
OPTIMIZE=""
START_FRAME=0
END_FRAME=""
ROT_THRESH=15
TRANS_THRESH=0.03
TRACK_REFINE_ITER=10
SKIP_H5=0
SKIP_MASKS=0
SKIP_TRACKING=0
# If set, build h5 in this dir and symlink it into the sequence folder so
# downstream tools (cluster_reconstruct, generate_meta, …) still find it at
# $SEQUENCE_FOLDER/data00000000.h5. Typical: /dev/shm/$USER/$SLURM_JOB_ID .
H5_SCRATCH_DIR=""

# DINO seeding tunables (pass-through to tools/dino_tool_segment.py).
# Empty string = use dino_tool_segment's defaults.
DINO_MESH_SCAN_EVERY=""     # stride for anchor-camera seed scan (default 10)
DINO_DENSE_SCAN_EVERY=""    # stride for non-anchor seed scan    (default 5)
DINO_SEED_MIN_AREA=""       # partial views -> lower this (default 100)
DINO_SEED_MAX_AREA=""       # large tools -> raise this    (default 15000)
DINO_SEED_MIN_SIM=""        # partial views -> lower this (default 0.20)
DINO_SEED_FAST=0            # 1 = first-hit mode (try frame 0, then stride, stop on first OK)
DINO_FRAME0_ONLY=0          # 1 = frame-0-only mode (skip Phase 3-7, no re-seeding; fast path for cluster)

HOCAP_ROOT="${HOCAP_ROOT:-/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation}"
HAND_ROOT="${HAND_ROOT:-/home/ruoqu/crq_ws/robotool/HandReconstruction}"
ROBOTOOL_ROOT="$(dirname "$HOCAP_ROOT")"

# SAM2 checkpoint resolution order:
#   1. $SAM2_CKPT env var (explicit)
#   2. $SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt (one-shot root override)
#   3. candidate paths below (first existing file wins):
#        local layout:   <robotool>/mesh_reconstruction/sam2/checkpoints/...
#        cluster layout: <robotool>/sam2/checkpoints/...
if [[ -z "${SAM2_CKPT:-}" ]]; then
    if [[ -n "${SAM2_ROOT:-}" ]]; then
        SAM2_CKPT="${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt"
    else
        _SAM2_CKPT_CANDIDATES=(
            "${ROBOTOOL_ROOT}/mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt"
            "${ROBOTOOL_ROOT}/sam2/checkpoints/sam2.1_hiera_large.pt"
        )
        for _p in "${_SAM2_CKPT_CANDIDATES[@]}"; do
            if [[ -f "$_p" ]]; then SAM2_CKPT="$_p"; break; fi
        done
        SAM2_CKPT="${SAM2_CKPT:-${ROBOTOOL_ROOT}/mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt}"
    fi
fi
SAM2_VIDEO_CFG="${SAM2_VIDEO_CFG:-${HOCAP_ROOT}/config/sam2_config/sam2.1_hiera_l.yaml}"
SAM2_IMAGE_CFG="${SAM2_IMAGE_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"

# ---------- arg parsing ----------
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sequence_folder)     SEQUENCE_FOLDER="$2"; shift 2;;
        --calibration_yaml)    CALIBRATION_YAML="$2"; shift 2;;
        --tool_name)           TOOL_NAME="$2"; shift 2;;
        --tool_mesh)           TOOL_MESH="$2"; shift 2;;
        --hand)                HAND="$2"; shift 2;;
        --optimize)            OPTIMIZE="$2"; shift 2;;
        --start_frame)         START_FRAME="$2"; shift 2;;
        --end_frame)           END_FRAME="$2"; shift 2;;
        --rot_thresh)          ROT_THRESH="$2"; shift 2;;
        --trans_thresh)        TRANS_THRESH="$2"; shift 2;;
        --track_refine_iter)   TRACK_REFINE_ITER="$2"; shift 2;;
        --skip_h5)             SKIP_H5=1; shift;;
        --skip_masks)          SKIP_MASKS=1; shift;;
        --skip_tracking)       SKIP_TRACKING=1; shift;;
        --sam2_ckpt)           SAM2_CKPT="$2"; shift 2;;
        --sam2_video_cfg)      SAM2_VIDEO_CFG="$2"; shift 2;;
        --sam2_image_cfg)      SAM2_IMAGE_CFG="$2"; shift 2;;
        --h5_scratch_dir)      H5_SCRATCH_DIR="$2"; shift 2;;
        --models_folder)       MODELS_FOLDER="$2"; shift 2;;
        --mesh_scan_every)     DINO_MESH_SCAN_EVERY="$2"; shift 2;;
        --dense_scan_every)    DINO_DENSE_SCAN_EVERY="$2"; shift 2;;
        --seed_min_area)       DINO_SEED_MIN_AREA="$2"; shift 2;;
        --seed_max_area)       DINO_SEED_MAX_AREA="$2"; shift 2;;
        --seed_min_sim)        DINO_SEED_MIN_SIM="$2"; shift 2;;
        --seed_fast)           DINO_SEED_FAST=1; shift;;
        --frame0_only)         DINO_FRAME0_ONLY=1; shift;;
        -h|--help)             sed -n '2,45p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

for required in SEQUENCE_FOLDER CALIBRATION_YAML TOOL_NAME; do
    if [[ -z "${!required}" ]]; then
        echo "Error: --${required,,} is required"; exit 1
    fi
done

SEQUENCE_FOLDER="$(cd "$SEQUENCE_FOLDER" && pwd)"
CALIBRATION_YAML="$(readlink -f "$CALIBRATION_YAML")"

# --- resolve tool mesh from tool_name if --tool_mesh not given ---
MODELS_FOLDER="${MODELS_FOLDER:-${HOCAP_ROOT}/data/models}"
if [[ -z "$TOOL_MESH" ]]; then
    if [[ ! -d "$MODELS_FOLDER/$TOOL_NAME" ]]; then
        echo "Error: no model folder $MODELS_FOLDER/$TOOL_NAME."
        echo "       Pass --tool_mesh explicitly or set MODELS_FOLDER / --models_folder."
        exit 1
    fi
    for cand in "textured_mesh.obj" "cleaned_mesh_10000.obj" "mesh.obj"; do
        if [[ -f "$MODELS_FOLDER/$TOOL_NAME/$cand" ]]; then
            TOOL_MESH="$MODELS_FOLDER/$TOOL_NAME/$cand"
            break
        fi
    done
    if [[ -z "$TOOL_MESH" ]]; then
        echo "Error: no mesh file found in $MODELS_FOLDER/$TOOL_NAME/"
        echo "       (looked for: textured_mesh.obj, cleaned_mesh_10000.obj, mesh.obj)"
        exit 1
    fi
    echo "[mesh] auto-resolved from tool_name: $TOOL_MESH"
fi
TOOL_MESH="$(readlink -f "$TOOL_MESH")"

# ---------- derive paths ----------
# Expect .../<base>/<videos_XXXX>/<task>/<exp>
EXP_NAME="$(basename "$SEQUENCE_FOLDER")"
TASK_FOLDER="$(dirname "$SEQUENCE_FOLDER")"
TASK_NAME="$(basename "$TASK_FOLDER")"
VIDEOS_ROOT="$(dirname "$TASK_FOLDER")"
VIDEOS_FOLDER_NAME="$(basename "$VIDEOS_ROOT")"
BASE="$(dirname "$VIDEOS_ROOT")"

H5_PATH="${SEQUENCE_FOLDER}/data00000000.h5"   # what downstream scripts expect
# If a fast scratch dir is given, real h5 lives there and H5_PATH is a symlink.
if [[ -n "$H5_SCRATCH_DIR" ]]; then
    mkdir -p "$H5_SCRATCH_DIR"
    # Name unique to this sequence so concurrent jobs don't collide.
    H5_REAL_PATH="${H5_SCRATCH_DIR}/$(echo "${VIDEOS_FOLDER_NAME}_${TASK_NAME}_${EXP_NAME}" | tr / _)_data00000000.h5"
else
    H5_REAL_PATH="$H5_PATH"
fi
ANNOTATED_PATH="${BASE}/${VIDEOS_FOLDER_NAME}_annotated/${TASK_NAME}/${EXP_NAME}"
TOOL_MASKS_DIR="${ANNOTATED_PATH}/tool_masks"
# MODELS_FOLDER already resolved above (used for mesh auto-resolution)

mkdir -p "$ANNOTATED_PATH"

# conda activation (tries common locations)
_CONDA_SH_CANDIDATES=(
    "${CONDA_SH:-}"
    "/home/ruoqu/miniconda3/etc/profile.d/conda.sh"
    "/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
)
for _p in "${_CONDA_SH_CANDIDATES[@]}"; do
    if [[ -n "$_p" && -f "$_p" ]]; then source "$_p"; break; fi
done

echo "=========================================="
echo "sequence         : $SEQUENCE_FOLDER"
echo "tool             : $TOOL_NAME  (mesh: $TOOL_MESH)"
echo "calibration      : $CALIBRATION_YAML"
echo "annotated out    : $ANNOTATED_PATH"
echo "tool_masks_dir   : $TOOL_MASKS_DIR"
echo "hand=${HAND:-0}  optimize=${OPTIMIZE:-0}"
echo "=========================================="

conda activate hocap-annotation

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg not on PATH"; exit 1
fi

# ---------- Stage 1: videos -> h5 ----------
if [[ "$SKIP_H5" == "1" && -f "$H5_PATH" ]]; then
    echo "[1/7] [skip] h5 exists: $H5_PATH"
else
    echo "[1/7] building h5 ..."
    # Clean any prior state on both the logical path (may be stale symlink) and the real path.
    rm -f "$H5_PATH" "$H5_REAL_PATH"
    H5_ARGS=(--input_dir "$SEQUENCE_FOLDER" --output_file "$H5_REAL_PATH" --start_frame "$START_FRAME")
    [[ -n "$END_FRAME" ]] && H5_ARGS+=(--end_frame "$END_FRAME")
    python "$HOCAP_ROOT/tools/00_convert_videos_to_h5.py" "${H5_ARGS[@]}"
    # Expose the real file at the logical path so downstream reads transparently.
    if [[ "$H5_REAL_PATH" != "$H5_PATH" ]]; then
        ln -sf "$H5_REAL_PATH" "$H5_PATH"
    fi
fi

# ---------- Stage 2: dino auto tool mask + pipeline-format export ----------
# Discover camera serials from calibration to keep folder naming consistent
# with generate_meta.py (cam{serial}_rgb/).
CAM_SERIALS=$(CAL_YAML="$CALIBRATION_YAML" python - <<'PY'
import os, yaml
with open(os.environ["CAL_YAML"]) as f:
    d = yaml.safe_load(f)
if isinstance(d, list):
    cams = d
else:
    cams = [v for k, v in d.get("extrinsics", {}).items() if not k.startswith("tag_")]
cams = sorted(cams, key=lambda c: c.get("camera_id", 0))
print(" ".join(str(c["camera_id"]).zfill(2) for c in cams))
PY
)
echo "  camera serials: $CAM_SERIALS"

if [[ "$SKIP_MASKS" == "1" ]] && ( ls "$TOOL_MASKS_DIR"/cam*_rgb/0000.npz >/dev/null 2>&1 \
                                   || ls "$TOOL_MASKS_DIR"/cam*_rgb/0000.npy >/dev/null 2>&1 ); then
    echo "[2/7] [skip] tool_masks already populated"
else
    echo "[2/7] running DINOv2 + SAM2 auto segmentation ..."
    export PYOPENGL_PLATFORM=egl
    # DINO output dir (contains masks.h5, viz/, seed_info.json for debug)
    DINO_OUT="${ANNOTATED_PATH}/dino_auto"
    mkdir -p "$DINO_OUT"
    DINO_ARGS=(
        --data_h5      "$H5_PATH"
        --calib_yaml   "$CALIBRATION_YAML"
        --tool_mesh    "$TOOL_MESH"
        --output_dir   "$DINO_OUT"
        --sam2_ckpt        "$SAM2_CKPT"
        --sam2_video_cfg   "$SAM2_VIDEO_CFG"
        --sam2_image_cfg   "$SAM2_IMAGE_CFG"
        --pipeline_tool_masks_dir "$TOOL_MASKS_DIR"
        --tool_name "$TOOL_NAME"
        --no_video
    )
    [[ -n "$CAM_SERIALS" ]]            && DINO_ARGS+=(--cam_serials $CAM_SERIALS)
    [[ -n "$DINO_MESH_SCAN_EVERY" ]]   && DINO_ARGS+=(--mesh_scan_every  "$DINO_MESH_SCAN_EVERY")
    [[ -n "$DINO_DENSE_SCAN_EVERY" ]]  && DINO_ARGS+=(--dense_scan_every "$DINO_DENSE_SCAN_EVERY")
    [[ -n "$DINO_SEED_MIN_AREA" ]]     && DINO_ARGS+=(--seed_min_area    "$DINO_SEED_MIN_AREA")
    [[ -n "$DINO_SEED_MAX_AREA" ]]     && DINO_ARGS+=(--seed_max_area    "$DINO_SEED_MAX_AREA")
    [[ -n "$DINO_SEED_MIN_SIM" ]]      && DINO_ARGS+=(--seed_min_sim     "$DINO_SEED_MIN_SIM")
    [[ "$DINO_SEED_FAST" == "1" ]]     && DINO_ARGS+=(--seed_fast)
    [[ "$DINO_FRAME0_ONLY" == "1" ]]   && DINO_ARGS+=(--frame0_only)
    python "$HOCAP_ROOT/tools/dino_tool_segment.py" "${DINO_ARGS[@]}"
fi

# ---------- Stage 3: generate_meta ----------
echo "[3/7] generate_meta ..."
cd "$HOCAP_ROOT"
python preprocess/generate_meta.py \
    --h5_path "$H5_PATH" \
    --calibration_yaml_path "$CALIBRATION_YAML" \
    --models_folder "$MODELS_FOLDER" \
    --tool_name "$TOOL_NAME" \
    --start_frame "$START_FRAME" \
    --x_min -0.6 --x_max 0.6 --y_min -0.5 --y_max 0.6 --z_min -0.5 --z_max 0.4

# Detect object count (auto-picks 1 for single-object DINO output)
NUM_OBJECTS=$(python3 -c "import yaml; m=yaml.safe_load(open('$SEQUENCE_FOLDER/meta.yaml')); print(len(m['object_ids']))")
echo "  detected $NUM_OBJECTS object(s)"

# ---------- Stage 4: per-object 6-DoF tracking ----------
if [[ "$SKIP_TRACKING" != "1" ]]; then
    for OBJ_IDX in $(seq 1 $NUM_OBJECTS); do
        echo "[4/7] fd_pose_solver object $OBJ_IDX / $NUM_OBJECTS ..."
        python tools/04-1-4_fd_pose_solver_kalman.py \
            --no_masked_depth \
            --sequence_folder "$SEQUENCE_FOLDER" \
            --activate_2d_tracker --activate_kalman_filter \
            --object_idx "$OBJ_IDX" \
            --track_refine_iter "$TRACK_REFINE_ITER" \
            --rot_thresh "$ROT_THRESH" --trans_thresh "$TRANS_THRESH"
    done

    # ---------- Stage 5: multi-view pose merge ----------
    echo "[5/7] fd_pose_merger ..."
    python tools/04-2-2_fd_pose_merger_cluster.py --sequence_folder "$SEQUENCE_FOLDER"
else
    echo "[4-5/7] [skip] tracking skipped"
fi

# ---------- Stage 6: hand reconstruction (optional) ----------
if [[ -n "$HAND" && "$HAND" != "0" ]]; then
    echo "[6/7] hand reconstruction ..."
    cd "$HAND_ROOT"
    conda activate reconstruct-hand
    python cluster_reconstruct.py --sequence_folder "$SEQUENCE_FOLDER"
    SAVED_FILE="${ANNOTATED_PATH}/result.pkl"
    if [[ -f "$SAVED_FILE" ]]; then
        python cluster_optimize_hand.py --file_name "$SAVED_FILE"
    fi
    cd "$HOCAP_ROOT"
    conda activate hocap-annotation
fi

# ---------- Stage 7: object + joint optimization (optional) ----------
if [[ -n "$OPTIMIZE" && "$OPTIMIZE" != "0" ]]; then
    echo "[7/7] object + joint optimization (first object only) ..."
    python tools/06-2_object_pose_solver_cluster.py --sequence_folder "$SEQUENCE_FOLDER" --debug
    python tools/07-2_joint_pose_solver_cluster.py  --sequence_folder "$SEQUENCE_FOLDER" --debug
fi

echo ""
echo "=========================================="
echo "Done. Annotated output at:"
echo "  $ANNOTATED_PATH"
echo "=========================================="
