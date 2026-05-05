#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name hand_1exp
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/hand_1exp_%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/hand_1exp_%j.err
#
# Hand reconstruction for ONE exp. Uses one GPU, no parallelism.
# Wrapper around scripts/batch_task_folder_hand_chunked.sh --only_exp.
#
# Usage:
#   sbatch scripts/sbatch_run_hand_one_exp.sh \
#       --videos_root /viscam/projects/robotool/data/videos_0102 \
#       --task mallet_crush_nuts \
#       --exp 20260104_..._5 \
#       [--chunk_size 500] [--keep_h5] [--no_skip_existing]
#
# Calibration: auto-discovers *_global_aligned.yaml under
# <videos_root>/realsense_calibrate_*/. Pass --calibration_yaml to override.

set -u

export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export HAND_ROOT="${HAND_ROOT:-/viscam/u/chenrq/crq_ws/robotool/HandReconstruction}"
export MODELS_FOLDER="${MODELS_FOLDER:-/viscam/u/chenrq/models}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
HAND_BATCH_SCRIPT="$HOCAP_ROOT/scripts/batch_task_folder_hand_chunked.sh"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export H5_COMPRESSION=lzf

_JOBTAG="${SLURM_JOB_ID:-nonslurm_$$}"
H5_SCRATCH="/dev/shm/${USER}/${_JOBTAG}"
mkdir -p "$H5_SCRATCH"
trap '[[ -d "$H5_SCRATCH" ]] && rm -rf "$H5_SCRATCH" 2>/dev/null || true' EXIT INT TERM HUP

VIDEOS_ROOT=""
TASK=""
EXP=""
CAL_YAML=""
CHUNK_SIZE=500
SKIP_EXISTING=1
KEEP_H5=0
NO_MERGE=0
KEEP_CHUNK_FILES=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --videos_root)        VIDEOS_ROOT="$2"; shift 2;;
        --task)               TASK="$2"; shift 2;;
        --exp)                EXP="$2"; shift 2;;
        --calibration_yaml)   CAL_YAML="$2"; shift 2;;
        --chunk_size)         CHUNK_SIZE="$2"; shift 2;;
        --skip_existing)      SKIP_EXISTING=1; shift;;
        --no_skip_existing)   SKIP_EXISTING=0; shift;;
        --keep_h5)            KEEP_H5=1; shift;;
        --no_merge)           NO_MERGE=1; shift;;
        --keep_chunk_files)   KEEP_CHUNK_FILES=1; shift;;
        -h|--help)            sed -n '2,30p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$VIDEOS_ROOT" || -z "$TASK" || -z "$EXP" ]]; then
    echo "Error: --videos_root, --task, --exp are all required"; exit 1
fi
VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
TASK_FOLDER="$VIDEOS_ROOT/$TASK"
EXP_FOLDER="$TASK_FOLDER/$EXP"
[[ -d "$TASK_FOLDER" ]] || { echo "Error: task folder not found: $TASK_FOLDER"; exit 1; }
[[ -d "$EXP_FOLDER" ]] || { echo "Error: exp folder not found: $EXP_FOLDER"; exit 1; }
compgen -G "$EXP_FOLDER/cam*_rgb.mp4" >/dev/null 2>&1 \
    || { echo "Error: no cam*_rgb.mp4 in $EXP_FOLDER"; exit 1; }

if [[ -z "$CAL_YAML" ]]; then
    CAL_FOLDER="$(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -1)"
    [[ -z "$CAL_FOLDER" ]] && { echo "Error: no realsense_calibrate_* under $VIDEOS_ROOT"; exit 1; }
    CAL_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name 'realsense_calibration_*_global_aligned.yaml' | head -1)"
fi
[[ -f "$CAL_YAML" ]] || { echo "Error: calibration yaml not found: $CAL_YAML"; exit 1; }

echo "=========================================="
echo "hand_one_exp    job=${SLURM_JOB_ID:-<non-slurm>}    node=$(hostname)"
echo "  videos_root  : $VIDEOS_ROOT"
echo "  task         : $TASK"
echo "  exp          : $EXP"
echo "  exp_folder   : $EXP_FOLDER"
echo "  calibration  : $CAL_YAML"
echo "  chunk_size   : $CHUNK_SIZE"
echo "  scratch      : $H5_SCRATCH"
echo "=========================================="

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

INNER_FLAGS=()
[[ "$SKIP_EXISTING" == "1" ]] && INNER_FLAGS+=(--skip_existing)
[[ "$KEEP_H5" == "1" ]] && INNER_FLAGS+=(--keep_h5)
[[ "$NO_MERGE" == "1" ]] && INNER_FLAGS+=(--no_merge)
[[ "$KEEP_CHUNK_FILES" == "1" ]] && INNER_FLAGS+=(--keep_chunk_files)

bash "$HAND_BATCH_SCRIPT" \
    --task_folder "$TASK_FOLDER" \
    --calibration_yaml "$CAL_YAML" \
    --chunk_size "$CHUNK_SIZE" \
    --h5_scratch_dir "$H5_SCRATCH" \
    --only_exp "$EXP" \
    "${INNER_FLAGS[@]}"
RC=$?
echo "[done] inner rc=$RC"
exit $RC
