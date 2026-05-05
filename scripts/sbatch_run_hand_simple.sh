#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name hand_simp
#SBATCH --partition=viscam
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/hand_simp_%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/hand_simp_%j.err
#
# One sbatch job, one videos_root, 2 GPUs, 4 concurrent workers.
#
# Layout: this script holds 1 slurm allocation (2 GPUs, 128 GB RAM, 16 CPUs).
# Inside that allocation it spawns up to 4 worker subprocesses in parallel,
# each pinned to one of the 2 GPUs via CUDA_VISIBLE_DEVICES (so 2 workers
# share each GPU). Workers consume exps from a queue: for every exp under
# <videos_root>/<task>/<exp>/ that does NOT have
# <annotated>/<task>/<exp>/result_hand_optimized.pkl, one worker runs
# scripts/batch_task_folder_hand_chunked.sh --only_exp <exp>.
#
# Pros over the array variant:
#   - counts as 1 job toward your slurm submission limit, not N
#   - no manifest file, no array bookkeeping
#   - 2 GPUs × 2 workers = 4 streams of progress in one job
#
# Cons:
#   - shares 2 GPUs across 4 workers → each worker has half the GPU memory.
#     Hand reconstruction uses ~4-6 GB per process so this fits on 24 GB+
#     cards (A40 / A100). DO NOT submit this on 11-12 GB cards (Titan Xp,
#     2080Ti) — the exclude list at the top filters most of those out.
#   - if any worker stalls, it pins one GPU; the other 3 continue.
#
# Usage:
#   sbatch scripts/sbatch_run_hand_simple.sh \
#       --videos_root /viscam/projects/robotool/data/videos_0102 \
#       [--parallel 4]              (concurrent worker count; default 4)
#       [--gpus 2]                  (must match #SBATCH --gres=gpu:N; default 2)
#       [--chunk_size 500]
#       [--skip_existing]           (forwarded to inner; default ON)
#       [--no_skip_existing]
#       [--keep_h5] [--no_merge] [--keep_chunk_files]
#       [--dry_run]                 (list pending, don't run)
#
# Resume: --skip_existing is on by default. After scancel + resubmit, every
# exp that already has result_hand_optimized.pkl is auto-skipped at the
# inner level (chunk-build is also skipped → ~zero-cost no-op).

set -u

# ---------- cluster paths ----------
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export HAND_ROOT="${HAND_ROOT:-/viscam/u/chenrq/crq_ws/robotool/HandReconstruction}"
export MODELS_FOLDER="${MODELS_FOLDER:-/viscam/u/chenrq/models}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
HAND_BATCH_SCRIPT="$HOCAP_ROOT/scripts/batch_task_folder_hand_chunked.sh"

# ---------- perf env ----------
# Each worker gets ~cpus_per_task / parallel CPU threads via OMP/MKL.
# 16 cpus / 4 workers ≈ 4 each → matches the inner script's expectation.
TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-16}"

# ---------- /dev/shm scratch ----------
_JOBTAG="${SLURM_JOB_ID:-nonslurm_$$}"
H5_SCRATCH_ROOT="/dev/shm/${USER}/${_JOBTAG}"
mkdir -p "$H5_SCRATCH_ROOT"
cleanup_shm() {
    [[ -d "$H5_SCRATCH_ROOT" ]] && rm -rf "$H5_SCRATCH_ROOT" 2>/dev/null || true
}
trap cleanup_shm EXIT INT TERM HUP

# ---------- args ----------
VIDEOS_ROOT=""
PARALLEL=4
GPUS=2
CHUNK_SIZE=500
SKIP_EXISTING=1
KEEP_H5=0
NO_MERGE=0
KEEP_CHUNK_FILES=0
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --videos_root)        VIDEOS_ROOT="$2"; shift 2;;
        --parallel)           PARALLEL="$2"; shift 2;;
        --gpus)               GPUS="$2"; shift 2;;
        --chunk_size)         CHUNK_SIZE="$2"; shift 2;;
        --skip_existing)      SKIP_EXISTING=1; shift;;
        --no_skip_existing)   SKIP_EXISTING=0; shift;;
        --keep_h5)            KEEP_H5=1; shift;;
        --no_merge)           NO_MERGE=1; shift;;
        --keep_chunk_files)   KEEP_CHUNK_FILES=1; shift;;
        --dry_run)            DRY_RUN=1; shift;;
        -h|--help)            sed -n '2,55p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

[[ -z "$VIDEOS_ROOT" ]] && { echo "Error: --videos_root required"; exit 1; }
VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
[[ -d "$VIDEOS_ROOT" ]] || { echo "Error: not a dir: $VIDEOS_ROOT"; exit 1; }
ANNOTATED_ROOT="$(dirname "$VIDEOS_ROOT")/$(basename "$VIDEOS_ROOT")_annotated"

# Per-worker thread count for OpenMP / MKL (rough split across workers).
PER_WORKER_THREADS=$(( TOTAL_CPUS / PARALLEL ))
[[ "$PER_WORKER_THREADS" -lt 1 ]] && PER_WORKER_THREADS=1

echo "=========================================================================="
echo "hand_simple    job=${SLURM_JOB_ID:-<non-slurm>}    node=$(hostname)"
echo "  videos_root      : $VIDEOS_ROOT"
echo "  annotated_root   : $ANNOTATED_ROOT"
echo "  parallel workers : $PARALLEL"
echo "  gpus allocated   : $GPUS"
echo "  cpus / threads   : $TOTAL_CPUS total, ~$PER_WORKER_THREADS / worker"
echo "  /dev/shm root    : $H5_SCRATCH_ROOT"
echo "=========================================================================="

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg not on PATH"; exit 1
fi

# ---------- discover aligned calibration ----------
CAL_FOLDER="$(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1)"
[[ -z "$CAL_FOLDER" ]] && { echo "Error: no realsense_calibrate_* under $VIDEOS_ROOT"; exit 1; }
CAL_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name 'realsense_calibration_*_global_aligned.yaml' | head -n 1)"
if [[ -z "$CAL_YAML" ]]; then
    echo "Error: no *_global_aligned.yaml in $CAL_FOLDER"
    echo "       Run scripts/batch_global_align.sh or sbatch_run_hand_cluster.sh --run_alignment first."
    exit 1
fi
echo "[cal] using: $CAL_YAML"
echo ""

# ---------- discover pending (task, exp) pairs ----------
# Skip exps that already have a merged result_hand_optimized.pkl. Skip task
# subfolders that aren't real tasks (calibration / posts / first_frame / ...).
PENDING=()    # array of "<task>__<exp>"  (use __ as a separator that's
              # vanishingly rare in real names; revert with parameter expansion)

for TASK_DIR in "$VIDEOS_ROOT"/*/; do
    TASK_NAME="$(basename "$TASK_DIR")"
    [[ "$TASK_NAME" == realsense_calibrate_* ]] && continue
    [[ "$TASK_NAME" == ref_pc* ]] && continue
    [[ "$TASK_NAME" == posts* ]] && continue
    [[ "$TASK_NAME" == "first_frame" ]] && continue
    for EXP_DIR in "$TASK_DIR"*/; do
        [[ -d "$EXP_DIR" ]] || continue
        EXP_NAME="$(basename "$EXP_DIR")"
        # Only real exps (have at least one cam*_rgb.mp4)
        compgen -G "${EXP_DIR}cam*_rgb.mp4" >/dev/null 2>&1 || continue
        # Skip if already done
        DONE_PKL="${ANNOTATED_ROOT}/${TASK_NAME}/${EXP_NAME}/result_hand_optimized.pkl"
        if [[ "$SKIP_EXISTING" == "1" && -f "$DONE_PKL" ]]; then
            continue
        fi
        PENDING+=("${TASK_NAME}__${EXP_NAME}")
    done
done

N_PENDING=${#PENDING[@]}
echo "[scan] pending exps: $N_PENDING"
if [[ "$N_PENDING" -lt 1 ]]; then
    echo "[scan] nothing to do."
    exit 0
fi
# Print first 30 pending lines so the user sees what's queued.
for ((k=0; k<N_PENDING && k<30; k++)); do
    echo "  [$((k+1))] ${PENDING[$k]/__/ / }"
done
[[ "$N_PENDING" -gt 30 ]] && echo "  ... ($((N_PENDING - 30)) more)"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry_run] would launch $PARALLEL workers across $GPUS GPU(s)."
    exit 0
fi

# ---------- worker pool ----------
# Spawn up to $PARALLEL background workers. As each finishes, the main loop
# fires the next pending exp. Each worker's CUDA_VISIBLE_DEVICES is round-
# robined across $GPUS so the load is balanced.
INNER_FLAGS=()
[[ "$SKIP_EXISTING" == "1" ]] && INNER_FLAGS+=(--skip_existing)
[[ "$KEEP_H5" == "1" ]] && INNER_FLAGS+=(--keep_h5)
[[ "$NO_MERGE" == "1" ]] && INNER_FLAGS+=(--no_merge)
[[ "$KEEP_CHUNK_FILES" == "1" ]] && INNER_FLAGS+=(--keep_chunk_files)

run_one() {
    local idx="$1"
    local entry="$2"
    local task="${entry%%__*}"
    local exp="${entry#*__}"
    local gpu=$((idx % GPUS))
    local task_folder="$VIDEOS_ROOT/$task"
    # Per-worker /dev/shm scratch so multiple concurrent h5 builds don't
    # collide on the same filename.
    local scratch="$H5_SCRATCH_ROOT/w${idx}"
    mkdir -p "$scratch"

    local logf="$H5_SCRATCH_ROOT/log_${idx}_${task}_${exp}.log"
    echo "[$(date +%H:%M:%S)] [worker $idx → gpu $gpu] START  $task / $exp  (log: $logf)"

    # Pin this worker to one GPU. Constrain BLAS threads so 4 workers don't
    # collectively oversubscribe the 16 CPUs.
    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$PER_WORKER_THREADS" \
    MKL_NUM_THREADS="$PER_WORKER_THREADS" \
    OPENBLAS_NUM_THREADS="$PER_WORKER_THREADS" \
    NUMEXPR_NUM_THREADS="$PER_WORKER_THREADS" \
    H5_COMPRESSION=lzf \
    bash "$HAND_BATCH_SCRIPT" \
        --task_folder "$task_folder" \
        --calibration_yaml "$CAL_YAML" \
        --chunk_size "$CHUNK_SIZE" \
        --h5_scratch_dir "$scratch" \
        --only_exp "$exp" \
        "${INNER_FLAGS[@]}" >"$logf" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] [worker $idx → gpu $gpu] OK     $task / $exp"
    else
        echo "[$(date +%H:%M:%S)] [worker $idx → gpu $gpu] FAIL   $task / $exp  (rc=$rc, see $logf)"
    fi
    rm -rf "$scratch" 2>/dev/null || true
    return $rc
}

# Track per-worker outcomes via temp files (bash arrays don't survive subshells
# cleanly with `wait -n`).
RESULT_DIR="$H5_SCRATCH_ROOT/_results"
mkdir -p "$RESULT_DIR"

n_started=0
n_ok=0
n_fail=0
running=0

for ((i=0; i<N_PENDING; i++)); do
    entry="${PENDING[$i]}"
    (
        run_one "$i" "$entry"
        echo "$?" > "$RESULT_DIR/$i.rc"
    ) &
    running=$((running+1))
    n_started=$((n_started+1))
    if [[ $running -ge $PARALLEL ]]; then
        wait -n
        running=$((running-1))
    fi
done
wait

# Tally results
for f in "$RESULT_DIR"/*.rc; do
    rc="$(cat "$f")"
    if [[ "$rc" == "0" ]]; then n_ok=$((n_ok+1))
    else n_fail=$((n_fail+1)); fi
done

echo ""
echo "=========================================================================="
echo "summary: started=$n_started  ok=$n_ok  fail=$n_fail"
if [[ "$n_fail" -gt 0 ]]; then
    echo "see per-worker logs under $H5_SCRATCH_ROOT/log_*.log (will be wiped on exit)"
    echo "consider rerunning with --skip_existing to retry only the failed ones"
fi
echo "=========================================================================="

[[ "$n_fail" -gt 0 ]] && exit 1 || exit 0
