#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name va_orch
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/va_orch_%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/va_orch_%j.err
#
# CPU-only wrapper that runs orchestrate.py (the code orchestrator). No GPU — it
# just submits + polls. Survives logout; no tmux/nohup needed. All args pass
# through to orchestrate.py.
#
#   sbatch sbatch_orchestrate.sh                                # all folders, 4 at a time
#   sbatch sbatch_orchestrate.sh --max_parallel 4 --shards 4
#   sbatch sbatch_orchestrate.sh --videos_root videos_0106 videos_0107
#   sbatch sbatch_orchestrate.sh --add_frames 0.5,1.0          # 50%+last-frame pass
#   watch: tail -f /viscam/u/chenrq/crq_ws/slurm_outs/va_orch_<jobid>.out
#
# NOTE: needs sbatch/squeue inside the job; if slurm is module-loaded set
#   SLURM_INIT='module load slurm' at submit time (auto-exported).
set -u
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
[[ -n "${SLURM_INIT:-}" ]] && eval "$SLURM_INIT"
source "$CONDA_SH"; conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
exec python "$HOCAP_ROOT/volunteer_annotation/internal/orchestrate.py" "$@"
