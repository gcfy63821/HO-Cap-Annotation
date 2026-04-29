#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name hauto1
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/%j.err
#
# Cluster sbatch wrapper that runs `scripts/run_auto_annotator.sh` for ONE
# sequence (the multi-task variant is sbatch_run_auto_cluster.sh).
#
# Same flags as run_auto_annotator.sh — every argument is forwarded
# unchanged. The wrapper just adds:
#   - cluster paths   ($HOCAP_ROOT, $HAND_ROOT, $MODELS_FOLDER)
#   - /dev/shm tmpfs scratch for the h5 (avoids /viscam NFS saturation;
#     auto-cleaned on EXIT/INT/TERM/HUP, including scancel)
#   - conda activation
#   - performance env (OMP/MKL/OpenBLAS, H5_COMPRESSION=lzf, PYOPENGL=egl)
#
# Local-equivalent (what this wrapper translates):
#   bash scripts/run_auto_annotator.sh \
#     --sequence_folder /home/.../videos_0101/.../20260104_..._10 \
#     --calibration_yaml /home/.../realsense_calibration_0101_global_aligned.yaml \
#     --tool_name rubber_mallet \
#     --models_folder /home/.../HO-Cap-Annotation/data/models \
#     --object_chunk_size 600 --fake_optimize
#
# Cluster usage:
#   sbatch scripts/sbatch_run_auto_single.sh \
#     --sequence_folder /viscam/projects/robotool/data/videos_0101/mallet_crush_nuts/20260104_largeplate_..._10 \
#     --calibration_yaml /viscam/projects/robotool/data/videos_0101/realsense_calibrate_0101/realsense_calibration_0101_global_aligned.yaml \
#     --tool_name rubber_mallet \
#     --object_chunk_size 600 \
#     --fake_optimize \
#     [--hand 1] [--resume] [--start_frame 1500] [--end_frame 3364] ...
#
# All other run_auto_annotator.sh flags are accepted as-is. See:
#   scripts/run_auto_annotator.sh --help
#
# Resume tip: on a re-queue, pass --resume to skip stages whose outputs are
# already on disk (h5 / DINO npz / merged fd / hand pkl).

set -u

# ---------- cluster paths (override via env if your layout differs) ----------
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export HAND_ROOT="${HAND_ROOT:-/viscam/u/chenrq/crq_ws/robotool/HandReconstruction}"
export MODELS_FOLDER="${MODELS_FOLDER:-/viscam/u/chenrq/models}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
RUN_SCRIPT="$HOCAP_ROOT/scripts/run_auto_annotator.sh"

# SAM2 (used by tools/dino_tool_segment.py during Stage 2). On viscam,
# sam2 lives under robotool/ rather than hocap/, so SAM2_ROOT can't be
# inferred from $HOCAP_ROOT's parent — set it explicitly.
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

# ---------- /dev/shm scratch for h5 ----------
_JOBTAG="${SLURM_JOB_ID:-nonslurm_$$}"
H5_SCRATCH="/dev/shm/${USER}/${_JOBTAG}"
mkdir -p "$H5_SCRATCH"

cleanup_shm() {
    [[ -d "$H5_SCRATCH" ]] && rm -rf "$H5_SCRATCH" 2>/dev/null || true
}
trap cleanup_shm EXIT INT TERM HUP

echo "=========================================="
echo "job        : ${SLURM_JOB_ID:-<non-slurm>}   node: $(hostname)"
echo "cpus/omp   : ${SLURM_CPUS_PER_TASK:-4}"
echo "HOCAP_ROOT : $HOCAP_ROOT"
echo "MODELS     : $MODELS_FOLDER"
echo "H5 scratch : $H5_SCRATCH ($(df -h /dev/shm 2>/dev/null | awk 'NR==2 {print $4 " free"}'))"
echo "compress   : $H5_COMPRESSION"
echo "=========================================="

# ---------- sanity ----------
if [[ ! -x "$RUN_SCRIPT" && ! -f "$RUN_SCRIPT" ]]; then
    echo "Error: run_auto_annotator.sh not found at $RUN_SCRIPT"; exit 1
fi

# ---------- conda + ffmpeg ----------
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg not on PATH; depth decoding will silently produce zeros."
    echo "       Install: conda install -c conda-forge ffmpeg -y"
    exit 1
fi

# ---------- forward all CLI args verbatim, plus inject --h5_scratch_dir ----------
# If the user already passed --h5_scratch_dir, respect it; otherwise inject ours.
USER_PROVIDED_SCRATCH=0
for tok in "$@"; do
    if [[ "$tok" == "--h5_scratch_dir" ]]; then
        USER_PROVIDED_SCRATCH=1; break
    fi
done

EXTRA_ARGS=()
if [[ "$USER_PROVIDED_SCRATCH" == "0" ]]; then
    EXTRA_ARGS+=(--h5_scratch_dir "$H5_SCRATCH")
fi

# Inject --models_folder default if user didn't pass one (run_auto_annotator
# also has its own default, but we want to override with the cluster path).
USER_PROVIDED_MODELS=0
for tok in "$@"; do
    if [[ "$tok" == "--models_folder" ]]; then
        USER_PROVIDED_MODELS=1; break
    fi
done
if [[ "$USER_PROVIDED_MODELS" == "0" ]]; then
    EXTRA_ARGS+=(--models_folder "$MODELS_FOLDER")
fi

echo "[wrapper] forwarding to: $RUN_SCRIPT"
echo "[wrapper] args: ${EXTRA_ARGS[*]} $*"
echo "=========================================="

bash "$RUN_SCRIPT" "${EXTRA_ARGS[@]}" "$@"
RC=$?

echo ""
echo "=========================================="
echo "[wrapper] run_auto_annotator.sh exit code: $RC"
echo "=========================================="
exit $RC
