#!/usr/bin/env bash
# Sequentially precompute each videos_* folder: submit its sbatch array, WAIT
# for that folder's jobs to finish, then move to the next. Auto-resumes (already
# -done exps are skipped by the frontend scan).
#
# RUN THIS ON THE CLUSTER (where squeue/sbatch work), ideally in tmux so it
# survives disconnects:
#     tmux new -s vadrive
#     bash volunteer_annotation/internal/drive_videos.sh 2>&1 | tee drive.log
#     # detach: Ctrl-b d   |  reattach: tmux attach -t vadrive
#
# Process specific folders (in order) instead of all:
#     bash drive_videos.sh videos_0106 videos_0107
#     bash drive_videos.sh /abs/path/videos_0106 ...
#
# Env knobs:
#     DATA_ROOT (default /viscam/projects/robotool/data)
#     BUNDLE    (default /viscam/projects/robotool/_va_bundle_v2)
#     POLL      (default 120s between squeue checks)
#     SHARDS    (forwarded to the frontend; default = frontend's default)
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
FRONTEND="$HERE/sbatch_precompute_array.sh"
DATA_ROOT="${DATA_ROOT:-/viscam/projects/robotool/data}"
BUNDLE="${BUNDLE:-/viscam/projects/robotool/_va_bundle_v2}"
POLL="${POLL:-120}"
EXTRA=()
[[ -n "${SHARDS:-}" ]] && EXTRA+=(--shards "$SHARDS")

command -v squeue >/dev/null 2>&1 || { echo "[drive] ERROR: squeue not on PATH — run this on the cluster (interactive env)"; exit 1; }

# When NOT under SLURM (i.e. run with nohup/bash), self-log to a file so you
# don't need tmux/redirection. Under sbatch, output already goes to the job's
# --output file, so skip self-logging.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    SLURM_OUTS="${SLURM_OUTS:-/viscam/u/chenrq/crq_ws/slurm_outs}"
    DRIVE_LOG="${DRIVE_LOG:-$SLURM_OUTS/drive_$(date +%Y%m%d_%H%M%S).log}"
    mkdir -p "$(dirname "$DRIVE_LOG")"
    echo "[drive] running in background; full log -> $DRIVE_LOG"
    exec >"$DRIVE_LOG" 2>&1
fi

# Resolve folder list: args (names or abs paths) or all videos_* under DATA_ROOT.
ROOTS=()
if [[ $# -gt 0 ]]; then
    for a in "$@"; do [[ "$a" == /* ]] && ROOTS+=("$a") || ROOTS+=("$DATA_ROOT/$a"); done
else
    while IFS= read -r d; do ROOTS+=("${d%/}"); done \
        < <(ls -d "$DATA_ROOT"/videos_*/ 2>/dev/null | grep -v _annotated)
fi

echo "[drive] $(date)  $(hostname)  folders=${#ROOTS[@]}  bundle=$BUNDLE"
for VR in "${ROOTS[@]}"; do
    name=$(basename "$VR")
    echo "==================================================================="
    echo "[drive] $(date)  START  $name"
    [[ -d "$VR" ]] || { echo "[drive] SKIP (not a dir): $VR"; continue; }

    OUT=$(bash "$FRONTEND" --videos_root "$VR" --bundle "$BUNDLE" "${EXTRA[@]}" 2>&1)
    echo "$OUT"
    AJID=$(sed -n 's/.*array job: \([0-9]\+\).*/\1/p' <<< "$OUT")
    MJID=$(sed -n 's/.*merge job: \([0-9]\+\).*/\1/p' <<< "$OUT")

    if [[ -z "$MJID" ]]; then
        echo "[drive] $name: nothing submitted (already complete?) — moving on"
        continue
    fi
    echo "[drive] $name: array=$AJID merge=$MJID — waiting (poll ${POLL}s)..."
    # Wait until both jobs have left the queue (merge has afterany dep on array).
    while squeue -j "${AJID},${MJID}" -h 2>/dev/null | grep -q .; do sleep "$POLL"; done

    DONE=$(find "$BUNDLE/$name" -name _manifest.json 2>/dev/null | wc -l)
    echo "[drive] $(date)  DONE   $name  ($DONE exps in bundle)"
done
echo "[drive] $(date)  ALL FOLDERS DONE"
