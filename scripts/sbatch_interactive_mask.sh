#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name imask
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/imask_%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/imask_%j.err
#
# One-shot sbatch wrapper for scripts/interactive_mask_annotator.py.
#
# Allocates a GPU + 128 GB RAM (SAM2 video predictor pre-loads the whole
# decoded mp4 into CPU RAM during Propagate — ~6 MB × n_frames per cam, so
# even one 9000-frame video needs ~50 GB) and starts the Flask server on the
# compute node. Stdout + stderr go to slurm_outs/imask_<jobid>.{out,err},
# which prints the exact SSH tunnel command you need to paste on your laptop.
#
# Usage:
#   sbatch scripts/sbatch_interactive_mask.sh \
#       --videos_root /viscam/projects/robotool/data/videos_0102 \
#       [--source_cam 5] \
#       [--port 8765]                     # change if 8765 is in use
#       [--scratch_dir $HOME/.cache/imask] # default: disk (NOT /dev/shm)
#       [--sam2_checkpoint /abs/path.pt]
#       [--sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml]
#
# After submission:
#   1) tail -f /viscam/u/chenrq/crq_ws/slurm_outs/imask_<jobid>.out
#      to find the SSH tunnel line ("ssh -L 8765:localhost:8765 <node>").
#   2) Paste it on your laptop, then open http://localhost:8765/ in your
#      browser. The page is single-user — don't share the URL.
#   3) When done, scancel <jobid> (the atexit handler wipes the scratch dir).

set -u

# ---------- cluster paths ----------
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"

# SAM2 checkpoint + cfg (same defaults as sbatch_run_auto_videos.sh).
export SAM2_ROOT="${SAM2_ROOT:-/viscam/u/chenrq/crq_ws/robotool/sam2}"
export SAM2_CKPT_DEFAULT="${SAM2_CKPT_DEFAULT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt}"
export SAM2_CFG_DEFAULT="${SAM2_CFG_DEFAULT:-configs/sam2.1/sam2.1_hiera_l.yaml}"

# Scratch on DISK by default — /dev/shm IS RAM and competes with SAM2's
# decoded video buffers. $HOME is on viscam shared FS, fast enough for h5
# chunked writes.
SCRATCH_DEFAULT="${SCRATCH_DEFAULT:-$HOME/.cache/interactive_mask}"

# ---------- perf env ----------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export H5_COMPRESSION="lzf"
export PYOPENGL_PLATFORM="egl"

# ---------- arg parsing ----------
VIDEOS_ROOT=""
EXP_FOLDER=""
CALIBRATION_YAML=""
SOURCE_CAM=5
PORT=8765
SCRATCH_DIR=""
SAM2_CKPT=""
SAM2_CFG=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --videos_root)        VIDEOS_ROOT="$2"; shift 2;;
        --exp_folder)         EXP_FOLDER="$2"; shift 2;;
        --calibration_yaml)   CALIBRATION_YAML="$2"; shift 2;;
        --source_cam)         SOURCE_CAM="$2"; shift 2;;
        --port)               PORT="$2"; shift 2;;
        --scratch_dir)        SCRATCH_DIR="$2"; shift 2;;
        --sam2_checkpoint)    SAM2_CKPT="$2"; shift 2;;
        --sam2_model_cfg)     SAM2_CFG="$2"; shift 2;;
        -h|--help)            sed -n '2,40p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$VIDEOS_ROOT" && -z "$EXP_FOLDER" ]]; then
    echo "Error: must pass --videos_root (workspace mode) or --exp_folder (legacy)."
    exit 1
fi

[[ -z "$SCRATCH_DIR" ]] && SCRATCH_DIR="$SCRATCH_DEFAULT"
[[ -z "$SAM2_CKPT"   ]] && SAM2_CKPT="$SAM2_CKPT_DEFAULT"
[[ -z "$SAM2_CFG"    ]] && SAM2_CFG="$SAM2_CFG_DEFAULT"

mkdir -p "$SCRATCH_DIR"

# ---------- conda ----------
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

NODE="$(hostname -s)"
LOGIN_HOSTS_HINT="(use whichever login host you SSH into, e.g. viscam.stanford.edu)"

echo "=========================================================================="
echo "interactive mask annotator    job=${SLURM_JOB_ID:-<non-slurm>}    node=${NODE}"
echo "=========================================================================="
echo "videos_root  : ${VIDEOS_ROOT:-<unset>}"
echo "exp_folder   : ${EXP_FOLDER:-<unset>}"
echo "calibration  : ${CALIBRATION_YAML:-<unset, picked in UI>}"
echo "source_cam   : ${SOURCE_CAM}"
echo "port         : ${PORT}"
echo "scratch_dir  : ${SCRATCH_DIR}"
echo "sam2 ckpt    : ${SAM2_CKPT}"
echo "sam2 cfg     : ${SAM2_CFG}"
echo "mem / cpus   : ${SLURM_MEM_PER_NODE:-?}MB / ${SLURM_CPUS_PER_TASK:-?}"
echo "--------------------------------------------------------------------------"
echo ""
echo "==> SSH TUNNEL (paste this on your laptop):"
echo "    ssh -N -L ${PORT}:${NODE}:${PORT} <user>@<login_host>"
echo "    ${LOGIN_HOSTS_HINT}"
echo ""
echo "==> Then open in your laptop browser:"
echo "    http://localhost:${PORT}/"
echo ""
echo "==> Stop the server:   scancel ${SLURM_JOB_ID:-<jobid>}"
echo "=========================================================================="

# ---------- assemble args ----------
RUN_ARGS=(
    --port "$PORT"
    --host "0.0.0.0"            # allow tunnel from login node
    --source_cam "$SOURCE_CAM"
    --scratch_dir "$SCRATCH_DIR"
    --sam2_checkpoint "$SAM2_CKPT"
    --sam2_model_cfg  "$SAM2_CFG"
)
[[ -n "$VIDEOS_ROOT"      ]] && RUN_ARGS+=(--videos_root "$VIDEOS_ROOT")
[[ -n "$EXP_FOLDER"       ]] && RUN_ARGS+=(--exp_folder "$EXP_FOLDER")
[[ -n "$CALIBRATION_YAML" ]] && RUN_ARGS+=(--calibration_yaml "$CALIBRATION_YAML")

cd "$HOCAP_ROOT"
exec python -u scripts/interactive_mask_annotator.py "${RUN_ARGS[@]}"
