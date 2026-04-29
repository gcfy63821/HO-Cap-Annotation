#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name hauto
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/%j.err
#
# Cluster sbatch wrapper for the FULL auto annotation pipeline
# (object tracking via DINOv2 auto-mask + FoundationPose, plus hand
# reconstruction + joint optimization).
#
# Layout parallel to scripts/sbatch_run_hand_cluster.sh:
#   - Discovers realsense_calibrate_*/realsense_calibration_*.yaml under
#     --videos_root, preferring the *_global_aligned.yaml if present.
#   - Enumerates task folders (everything under videos_root that is not
#     the calibration folder).
#   - For each task, invokes scripts/batch_auto_annotator.sh, which:
#       · auto-matches a tool name per experiment against MODELS_FOLDER
#         (scripts/match_tool_name.py)
#       · calls scripts/run_auto_annotator.sh per experiment: h5 build,
#         DINOv2+SAM2 auto segmentation, generate_meta, fd_pose_solver
#         (per object, Kalman + 2D tracker), fd_pose_merger, then
#         HandReconstruction/cluster_reconstruct.py,
#         HandReconstruction/cluster_optimize_hand.py, then
#         06-2_object_pose_solver_cluster.py, 07-2_joint_pose_solver_cluster.py.
#
# Resume behaviour:
#   --skip_existing is propagated; experiments whose final result
#   (result_hand_optimized.pkl or merged object poses) already exists
#   are skipped.
#
# Performance:
#   - Node-local /dev/shm tmpfs for per-experiment h5 (avoids saturating
#     /viscam NFS when many jobs run in parallel).
#   - H5_COMPRESSION=lzf (5-10x faster than gzip at near-identical size).
#   - OMP/MKL/OpenBLAS threads match --cpus-per-task.
#   - EGL headless rendering for pyrender inside dino_tool_segment.py.
#
# Usage:
#   sbatch scripts/sbatch_run_auto_cluster.sh --videos_root /viscam/projects/robotool/data/videos_0101
#     [--hand 1]               (default 1)
#     [--optimize 1]           (default 1)
#     [--skip_existing]
#     [--force_tool NAME]
#     [--models_folder PATH]
#     [--start_frame 0] [--end_frame N]
#     [--rot_thresh 15] [--trans_thresh 0.03]
#     [--frame0_only]          # DINO-register only on frame 0, no re-seeding,
#                              # skips Phase 3-7. Much faster on cluster.
#     [--dry_run]              # show tool matches per exp, don't actually run

set -u

# ---------- cluster paths (exported so child scripts pick them up) ----------
export HOCAP_ROOT="/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation"
export HAND_ROOT="/viscam/u/chenrq/crq_ws/robotool/HandReconstruction"
export MODELS_FOLDER="/viscam/u/chenrq/models"
export CONDA_SH="/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="hocap-annotation"
HAND_BATCH_SCRIPT="$HOCAP_ROOT/scripts/batch_auto_annotator.sh"

# SAM2 (used by dino_tool_segment.py). On this cluster layout, sam2 lives
# under robotool/, not hocap/, so ROBOTOOL_ROOT cannot be inferred from
# HOCAP_ROOT's parent — SAM2_CKPT must be exported explicitly here.
export SAM2_ROOT="${SAM2_ROOT:-/viscam/u/chenrq/crq_ws/robotool/sam2}"
export SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt}"
export SAM2_VIDEO_CFG="${SAM2_VIDEO_CFG:-$HOCAP_ROOT/config/sam2_config/sam2.1_hiera_l.yaml}"
export SAM2_IMAGE_CFG="${SAM2_IMAGE_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"

# ---------- perf env ----------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export H5_COMPRESSION="lzf"
export PYOPENGL_PLATFORM="egl"

# ---------- fast scratch on /dev/shm ----------
_JOBTAG="${SLURM_JOB_ID:-nonslurm_$$}"
H5_SCRATCH="/dev/shm/${USER}/${_JOBTAG}"
mkdir -p "$H5_SCRATCH"

cleanup_shm() {
    [[ -d "$H5_SCRATCH" ]] && rm -rf "$H5_SCRATCH" 2>/dev/null || true
}
trap cleanup_shm EXIT INT TERM HUP

echo "=========================================="
echo "job       : ${SLURM_JOB_ID:-<non-slurm>}   node: $(hostname)"
echo "cpus/omp  : ${SLURM_CPUS_PER_TASK:-4}"
echo "H5 scratch: $H5_SCRATCH ($(df -h /dev/shm 2>/dev/null | awk 'NR==2 {print $4 " free"}'))"
echo "compress  : $H5_COMPRESSION"
echo "=========================================="

# ---------- arg parsing ----------
VIDEOS_ROOT=""
HAND=1
OPTIMIZE=1
SKIP_EXISTING_FLAG=""
FORCE_TOOL=""
MODELS_FOLDER_OVERRIDE=""
START_FRAME=0
END_FRAME=""
ROT_THRESH=15
TRANS_THRESH=0.03
TRACK_REFINE_ITER=10
DRY_RUN_FLAG=""
MAPPING_ONLY_FLAG=""
DINO_MESH_SCAN_EVERY=""
DINO_DENSE_SCAN_EVERY=""
DINO_SEED_MIN_AREA=""
DINO_SEED_MAX_AREA=""
DINO_SEED_MIN_SIM=""
FRAME0_ONLY_FLAG=""
LONG_VIDEO_THRESHOLD=""    # empty = run_auto_annotator.sh default (1000); 0 = disable
OBJECT_CHUNK_SIZE=""       # empty = run_auto_annotator default (1000); 0 = disable chunking
OBJECT_CHUNK_OVERLAP=""    # empty = run_auto_annotator default (20)

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --videos_root)        VIDEOS_ROOT="$2"; shift 2;;
        --hand)               HAND="$2"; shift 2;;
        --optimize)           OPTIMIZE="$2"; shift 2;;
        --skip_existing)      SKIP_EXISTING_FLAG="--skip_existing"; shift;;
        --force_tool)         FORCE_TOOL="$2"; shift 2;;
        --models_folder)      MODELS_FOLDER_OVERRIDE="$2"; shift 2;;
        --start_frame)        START_FRAME="$2"; shift 2;;
        --end_frame)          END_FRAME="$2"; shift 2;;
        --rot_thresh)         ROT_THRESH="$2"; shift 2;;
        --trans_thresh)       TRANS_THRESH="$2"; shift 2;;
        --track_refine_iter)  TRACK_REFINE_ITER="$2"; shift 2;;
        --dry_run)            DRY_RUN_FLAG="--dry_run"; shift;;
        --mapping_only)       MAPPING_ONLY_FLAG="--mapping_only"; shift;;
        --mesh_scan_every)    DINO_MESH_SCAN_EVERY="$2"; shift 2;;
        --dense_scan_every)   DINO_DENSE_SCAN_EVERY="$2"; shift 2;;
        --seed_min_area)      DINO_SEED_MIN_AREA="$2"; shift 2;;
        --seed_max_area)      DINO_SEED_MAX_AREA="$2"; shift 2;;
        --seed_min_sim)       DINO_SEED_MIN_SIM="$2"; shift 2;;
        --frame0_only)        FRAME0_ONLY_FLAG="--frame0_only"; shift;;
        --long_video_threshold) LONG_VIDEO_THRESHOLD="$2"; shift 2;;
        --object_chunk_size)    OBJECT_CHUNK_SIZE="$2"; shift 2;;
        --object_chunk_overlap) OBJECT_CHUNK_OVERLAP="$2"; shift 2;;
        -h|--help)            sed -n '2,50p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$VIDEOS_ROOT" ]]; then
    echo "Error: --videos_root is required"; exit 1
fi
VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
if [[ ! -d "$VIDEOS_ROOT" ]]; then
    echo "Error: videos_root is not a directory: $VIDEOS_ROOT"; exit 1
fi
[[ -n "$MODELS_FOLDER_OVERRIDE" ]] && export MODELS_FOLDER="$MODELS_FOLDER_OVERRIDE"

# ---------- conda ----------
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg not found on PATH; depth decoding will silently produce zeros."
    echo "       Install: conda install -c conda-forge ffmpeg -y"
    exit 1
fi

echo "videos_root    : $VIDEOS_ROOT"
echo "models folder  : $MODELS_FOLDER"

# ---------- discover calibration ----------
CAL_FOLDER="$(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1)"
if [[ -z "$CAL_FOLDER" ]]; then
    echo "Error: no realsense_calibrate_* folder under $VIDEOS_ROOT"; exit 1
fi
ORIG_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' \
              ! -name '*_global_aligned.yaml' ! -name '*_slider_aligned.yaml' \
              ! -name '*_manual_aligned.yaml' ! -name '*_aligned.yaml' | head -n 1)"
if [[ -z "$ORIG_YAML" ]]; then
    echo "Error: no realsense_calibration_*.yaml in $CAL_FOLDER"; exit 1
fi
ALIGNED_YAML="${CAL_FOLDER}/$(basename "$ORIG_YAML" .yaml)_global_aligned.yaml"
if [[ -f "$ALIGNED_YAML" ]]; then
    CAL_YAML="$ALIGNED_YAML"
    echo "calibration    : $CAL_YAML  (aligned)"
else
    CAL_YAML="$ORIG_YAML"
    echo "calibration    : $CAL_YAML  (original, NO alignment — consider running"
    echo "                  scripts/batch_global_align.sh or manual_align_viser_multi.py first)"
fi

# ---------- discover task folders ----------
TASK_FOLDERS=()
while IFS= read -r -d '' d; do
    name="$(basename "$d")"
    [[ "$name" == realsense_calibrate_* ]] && continue
    TASK_FOLDERS+=("$d")
done < <(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

if [[ "${#TASK_FOLDERS[@]}" -eq 0 ]]; then
    echo "Error: no task folders under $VIDEOS_ROOT"; exit 1
fi
echo "tasks (${#TASK_FOLDERS[@]}):"
for t in "${TASK_FOLDERS[@]}"; do echo "   - $(basename "$t")"; done
echo "=========================================="

# ---------- loop tasks ----------
TASK_OK=0; TASK_FAIL=0
FAILED_TASKS=()

for task_dir in "${TASK_FOLDERS[@]}"; do
    task_name="$(basename "$task_dir")"
    echo ""
    echo "##########################################"
    echo "# task: $task_name"
    echo "##########################################"

    BA_ARGS=(
        --task_folder "$task_dir"
        --calibration_yaml "$CAL_YAML"
        --hand "$HAND"
        --optimize "$OPTIMIZE"
        --models_folder "$MODELS_FOLDER"
        --start_frame "$START_FRAME"
        --rot_thresh "$ROT_THRESH"
        --trans_thresh "$TRANS_THRESH"
        --track_refine_iter "$TRACK_REFINE_ITER"
        --h5_scratch_dir "$H5_SCRATCH"
    )
    [[ -n "$END_FRAME" ]]           && BA_ARGS+=(--end_frame "$END_FRAME")
    [[ -n "$FORCE_TOOL" ]]          && BA_ARGS+=(--force_tool "$FORCE_TOOL")
    [[ -n "$SKIP_EXISTING_FLAG" ]]  && BA_ARGS+=("$SKIP_EXISTING_FLAG")
    [[ -n "$DRY_RUN_FLAG" ]]        && BA_ARGS+=("$DRY_RUN_FLAG")
    [[ -n "$MAPPING_ONLY_FLAG" ]]       && BA_ARGS+=("$MAPPING_ONLY_FLAG")
    [[ -n "$DINO_MESH_SCAN_EVERY" ]]    && BA_ARGS+=(--mesh_scan_every  "$DINO_MESH_SCAN_EVERY")
    [[ -n "$DINO_DENSE_SCAN_EVERY" ]]   && BA_ARGS+=(--dense_scan_every "$DINO_DENSE_SCAN_EVERY")
    [[ -n "$DINO_SEED_MIN_AREA" ]]      && BA_ARGS+=(--seed_min_area    "$DINO_SEED_MIN_AREA")
    [[ -n "$DINO_SEED_MAX_AREA" ]]      && BA_ARGS+=(--seed_max_area    "$DINO_SEED_MAX_AREA")
    [[ -n "$DINO_SEED_MIN_SIM" ]]       && BA_ARGS+=(--seed_min_sim     "$DINO_SEED_MIN_SIM")
    [[ -n "$FRAME0_ONLY_FLAG" ]]        && BA_ARGS+=("$FRAME0_ONLY_FLAG")
    [[ -n "$LONG_VIDEO_THRESHOLD" ]]    && BA_ARGS+=(--long_video_threshold "$LONG_VIDEO_THRESHOLD")
    [[ -n "$OBJECT_CHUNK_SIZE" ]]       && BA_ARGS+=(--object_chunk_size    "$OBJECT_CHUNK_SIZE")
    [[ -n "$OBJECT_CHUNK_OVERLAP" ]]    && BA_ARGS+=(--object_chunk_overlap "$OBJECT_CHUNK_OVERLAP")

    if bash "$HAND_BATCH_SCRIPT" "${BA_ARGS[@]}"; then
        TASK_OK=$((TASK_OK+1))
    else
        TASK_FAIL=$((TASK_FAIL+1))
        FAILED_TASKS+=("$task_name")
    fi
done

echo ""
echo "=========================================="
echo "summary: tasks_ok=$TASK_OK  tasks_failed=$TASK_FAIL"
if [[ "${#FAILED_TASKS[@]}" -gt 0 ]]; then
    echo "failed tasks:"
    for t in "${FAILED_TASKS[@]}"; do echo "  - $t"; done
fi
echo "=========================================="
