#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name redo_msk
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/redo_msk_%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/redo_msk_%j.err
#
# Batch wrapper for scripts/batch_redo_masks.py — re-runs SAM2 image+video
# predictor on every exp queued by scripts/simple_mask_annotator.py and
# overwrites <annotated>/<exp>/tool_masks/masks.h5.
#
# 128 GB / GPU because SAM2 vp.init_state(mp4) preloads the whole decoded
# video into CPU RAM (~6 MB × n_frames per cam). 24h time cap is generous —
# the script persists progress to the CSV after every exp, so you can
# scancel + resubmit and it'll skip 'done' rows.
#
# Usage:
#   sbatch scripts/sbatch_batch_redo_masks.sh \
#       --videos_root /viscam/projects/robotool/data/videos_0102 \
#       [--only_pending]                   # default: pending+failed
#       [--dry_run]                        # build seeds, skip propagate
#       [--scratch_dir $HOME/.cache/redo_masks]
#       [--sam2_checkpoint /abs/path.pt]
#       [--sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml]
#
# Monitor:
#   tail -f /viscam/u/chenrq/crq_ws/slurm_outs/redo_msk_<jobid>.out
#   <annotated_root>/_mask_redo_log.csv      # status column updates per exp

set -u

# ---------- cluster paths ----------
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"

# SAM2 (matches sbatch_run_auto_videos.sh defaults).
export SAM2_ROOT="${SAM2_ROOT:-/viscam/u/chenrq/crq_ws/robotool/sam2}"
SAM2_CKPT_DEFAULT="${SAM2_CKPT_DEFAULT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt}"
SAM2_CFG_DEFAULT="${SAM2_CFG_DEFAULT:-configs/sam2.1/sam2.1_hiera_l.yaml}"

# Scratch on disk by default — /dev/shm is RAM and competes with SAM2's
# decoded video buffers.
SCRATCH_DEFAULT="${SCRATCH_DEFAULT:-$HOME/.cache/batch_redo_masks}"

# ---------- perf env ----------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export H5_COMPRESSION="lzf"
export PYOPENGL_PLATFORM="egl"

# ---------- arg parsing ----------
VIDEOS_ROOT=""
ONLY_PENDING=0
DRY_RUN=0
SCRATCH_DIR=""
SAM2_CKPT=""
SAM2_CFG=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --videos_root)        VIDEOS_ROOT="$2"; shift 2;;
        --only_pending)       ONLY_PENDING=1; shift;;
        --dry_run)            DRY_RUN=1; shift;;
        --scratch_dir)        SCRATCH_DIR="$2"; shift 2;;
        --sam2_checkpoint)    SAM2_CKPT="$2"; shift 2;;
        --sam2_model_cfg)     SAM2_CFG="$2"; shift 2;;
        -h|--help)            sed -n '2,40p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$VIDEOS_ROOT" ]]; then
    echo "Error: --videos_root required"; exit 1
fi
VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
[[ -d "$VIDEOS_ROOT" ]] || { echo "Error: not a dir: $VIDEOS_ROOT"; exit 1; }

[[ -z "$SCRATCH_DIR" ]] && SCRATCH_DIR="$SCRATCH_DEFAULT"
[[ -z "$SAM2_CKPT"   ]] && SAM2_CKPT="$SAM2_CKPT_DEFAULT"
[[ -z "$SAM2_CFG"    ]] && SAM2_CFG="$SAM2_CFG_DEFAULT"

mkdir -p "$SCRATCH_DIR"

ANNOTATED_ROOT="$(dirname "$VIDEOS_ROOT")/$(basename "$VIDEOS_ROOT")_annotated"
REDO_CSV="$ANNOTATED_ROOT/_mask_redo_log.csv"

echo "=========================================================================="
echo "batch_redo_masks    job=${SLURM_JOB_ID:-<non-slurm>}    node=$(hostname -s)"
echo "=========================================================================="
echo "videos_root  : $VIDEOS_ROOT"
echo "annotated    : $ANNOTATED_ROOT"
echo "redo csv     : $REDO_CSV"
echo "scratch_dir  : $SCRATCH_DIR"
echo "only_pending : $ONLY_PENDING"
echo "dry_run      : $DRY_RUN"
echo "sam2 ckpt    : $SAM2_CKPT"
echo "sam2 cfg     : $SAM2_CFG"
echo "mem / cpus   : ${SLURM_MEM_PER_NODE:-?}MB / ${SLURM_CPUS_PER_TASK:-?}"
echo "=========================================================================="

if [[ ! -f "$REDO_CSV" ]]; then
    echo "Warning: $REDO_CSV does not exist."
    echo "         Run scripts/simple_mask_annotator.py first to queue exps."
    # Don't exit — batch_redo_masks.py will print its own no-op message.
fi

# ---------- conda ----------
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

# ---------- assemble args ----------
RUN_ARGS=(
    --videos_root      "$VIDEOS_ROOT"
    --scratch_dir      "$SCRATCH_DIR"
    --sam2_checkpoint  "$SAM2_CKPT"
    --sam2_model_cfg   "$SAM2_CFG"
)
[[ "$ONLY_PENDING" == "1" ]] && RUN_ARGS+=(--only_pending)
[[ "$DRY_RUN"      == "1" ]] && RUN_ARGS+=(--dry_run)

cd "$HOCAP_ROOT"
exec python -u scripts/batch_redo_masks.py "${RUN_ARGS[@]}"
