#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name va_orch_add
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/va_orch_add_%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/va_orch_add_%j.err
#
# PARALLEL ADD-FRAMES orchestrator: adds the extra annotatable keyframes (default
# the 50%% and the LAST frame, for deformable-object 3-frame annotation) to an
# EXISTING bundle, processing multiple folders AT ONCE (same code orchestrator as
# sbatch_orchestrate.sh, just with --add_frames). Additive + resumable.
#
#   sbatch sbatch_orchestrate_addframes.sh                          # all folders, 4 at a time
#   sbatch sbatch_orchestrate_addframes.sh --videos_root videos_0106 videos_0107
#   FRACS=0.5,1.0 sbatch sbatch_orchestrate_addframes.sh --max_parallel 4 --shards 4
#   watch: tail -f /viscam/u/chenrq/crq_ws/slurm_outs/va_orch_add_<jobid>.out
#
# Needs sbatch/squeue in-job (SLURM_INIT='module load slurm' if module-loaded).
set -u
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
FRACS="${FRACS:-0.5,1.0}"   # the extra annotatable frames (50% + last)
[[ -n "${SLURM_INIT:-}" ]] && eval "$SLURM_INIT"
source "$CONDA_SH"; conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
exec python "$HOCAP_ROOT/volunteer_annotation/internal/orchestrate.py" --add_frames "$FRACS" "$@"
