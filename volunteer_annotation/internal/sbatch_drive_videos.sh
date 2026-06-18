#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name va_drive
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/va_drive_%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/va_drive_%j.err
#
# CPU-only ORCHESTRATOR job: submits one precompute array per videos_* folder,
# waits for it to finish, then the next. No GPU — it only submits + polls + sleeps.
# Survives logout / no tmux / no nohup needed.
#
#   sbatch volunteer_annotation/internal/sbatch_drive_videos.sh                       # all videos_*
#   sbatch volunteer_annotation/internal/sbatch_drive_videos.sh videos_0106 videos_0107
#   # fresh bundle (env is auto-exported to the job by default):
#   BUNDLE=/viscam/projects/robotool/_va_bundle_v2 \
#     sbatch volunteer_annotation/internal/sbatch_drive_videos.sh videos_0106
#
# Watch:  tail -f /viscam/u/chenrq/crq_ws/slurm_outs/va_drive_<jobid>.out
#
# NOTE: the job needs sbatch/squeue on PATH. If your cluster loads slurm via a
#   module, submit with  SLURM_INIT='module load slurm'  (auto-exported), or add
#   the module line below.
set -u

export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
[[ -n "${SLURM_INIT:-}" ]] && eval "$SLURM_INIT"

exec bash "$HOCAP_ROOT/volunteer_annotation/internal/drive_videos.sh" "$@"
