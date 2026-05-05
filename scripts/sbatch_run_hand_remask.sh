#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name hand_rmsk
#SBATCH --partition=viscam
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/hand_rmsk_%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/hand_rmsk_%j.err
#
# Re-run hand reconstruction ONLY for exps that were re-mask-prompted via
# scripts/simple_mask_annotator.py (i.e. have mask_prompts.json on the
# annotated side). Existing hand outputs are deleted first, so the redo
# really redoes (no chunk-level skip). Mirrors sbatch_run_hand_simple.sh
# (1 job, 2 GPUs, 4 concurrent workers) but with a different filter +
# pre-cleanup step.
#
# Usage:
#   sbatch scripts/sbatch_run_hand_remask.sh \
#       --videos_root /viscam/projects/robotool/data/videos_0102 \
#       [--parallel 4] [--gpus 2] [--chunk_size 500]
#       [--keep_existing]    (don't pre-delete old hand outputs;
#                              fall back to chunk-level resume —
#                              useful if a previous run partly
#                              completed and you just want to resume)
#       [--dry_run]
#
# Resume: by default this script DELETES result_hand_optimized.pkl + all
# per-chunk artifacts before each exp, so the redo starts clean. The inner
# --skip_existing is OFF accordingly. Pass --keep_existing to flip the
# behavior (then it acts like sbatch_run_hand_simple.sh except still
# filtered to mask_prompts.json exps).

set -u

# ---------- cluster paths ----------
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export HAND_ROOT="${HAND_ROOT:-/viscam/u/chenrq/crq_ws/robotool/HandReconstruction}"
export MODELS_FOLDER="${MODELS_FOLDER:-/viscam/u/chenrq/models}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
HAND_BATCH_SCRIPT="$HOCAP_ROOT/scripts/batch_task_folder_hand_chunked.sh"

TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-16}"

# ---------- /dev/shm scratch ----------
_JOBTAG="${SLURM_JOB_ID:-nonslurm_$$}"
H5_SCRATCH_ROOT="/dev/shm/${USER}/${_JOBTAG}"
mkdir -p "$H5_SCRATCH_ROOT"
trap '[[ -d "$H5_SCRATCH_ROOT" ]] && rm -rf "$H5_SCRATCH_ROOT" 2>/dev/null || true' EXIT INT TERM HUP

# ---------- args ----------
VIDEOS_ROOT=""
PARALLEL=4
GPUS=2
CHUNK_SIZE=500
KEEP_EXISTING=0
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --videos_root)        VIDEOS_ROOT="$2"; shift 2;;
        --parallel)           PARALLEL="$2"; shift 2;;
        --gpus)               GPUS="$2"; shift 2;;
        --chunk_size)         CHUNK_SIZE="$2"; shift 2;;
        --keep_existing)      KEEP_EXISTING=1; shift;;
        --dry_run)            DRY_RUN=1; shift;;
        -h|--help)            sed -n '2,40p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

[[ -z "$VIDEOS_ROOT" ]] && { echo "Error: --videos_root required"; exit 1; }
VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
[[ -d "$VIDEOS_ROOT" ]] || { echo "Error: not a dir: $VIDEOS_ROOT"; exit 1; }
ANNOTATED_ROOT="$(dirname "$VIDEOS_ROOT")/$(basename "$VIDEOS_ROOT")_annotated"
[[ -d "$ANNOTATED_ROOT" ]] || { echo "Error: annotated root missing: $ANNOTATED_ROOT"; exit 1; }

PER_WORKER_THREADS=$(( TOTAL_CPUS / PARALLEL ))
[[ "$PER_WORKER_THREADS" -lt 1 ]] && PER_WORKER_THREADS=1

echo "=========================================================================="
echo "hand_remask    job=${SLURM_JOB_ID:-<non-slurm>}    node=$(hostname)"
echo "  videos_root      : $VIDEOS_ROOT"
echo "  annotated_root   : $ANNOTATED_ROOT"
echo "  parallel workers : $PARALLEL    gpus: $GPUS"
echo "  chunk_size       : $CHUNK_SIZE"
echo "  keep_existing    : $KEEP_EXISTING (0 = wipe old hand outputs first)"
echo "=========================================================================="

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
command -v ffmpeg >/dev/null 2>&1 || { echo "Error: ffmpeg not on PATH"; exit 1; }

CAL_FOLDER="$(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1)"
[[ -z "$CAL_FOLDER" ]] && { echo "Error: no realsense_calibrate_* under $VIDEOS_ROOT"; exit 1; }
CAL_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name 'realsense_calibration_*_global_aligned.yaml' | head -n 1)"
[[ -f "$CAL_YAML" ]] || { echo "Error: no *_global_aligned.yaml in $CAL_FOLDER"; exit 1; }
echo "[cal] using: $CAL_YAML"
echo ""

# ---------- discover (task, exp) pairs that have mask_prompts.json ----------
PENDING=()
NO_PROMPTS_SKIP=0
for TASK_DIR in "$VIDEOS_ROOT"/*/; do
    TASK_NAME="$(basename "$TASK_DIR")"
    [[ "$TASK_NAME" == realsense_calibrate_* ]] && continue
    [[ "$TASK_NAME" == ref_pc* ]] && continue
    [[ "$TASK_NAME" == posts* ]] && continue
    [[ "$TASK_NAME" == "first_frame" ]] && continue
    for EXP_DIR in "$TASK_DIR"*/; do
        [[ -d "$EXP_DIR" ]] || continue
        EXP_NAME="$(basename "$EXP_DIR")"
        compgen -G "${EXP_DIR}cam*_rgb.mp4" >/dev/null 2>&1 || continue
        ANN="${ANNOTATED_ROOT}/${TASK_NAME}/${EXP_NAME}"
        # GATE: must have mask_prompts.json (re-annotated by simple_mask_annotator).
        if [[ ! -f "${ANN}/mask_prompts.json" ]]; then
            NO_PROMPTS_SKIP=$((NO_PROMPTS_SKIP + 1))
            continue
        fi
        PENDING+=("${TASK_NAME}__${EXP_NAME}")
    done
done

N_PENDING=${#PENDING[@]}
echo "[scan] exps WITH mask_prompts.json (queued): $N_PENDING"
echo "[scan] exps without mask_prompts.json (skipped): $NO_PROMPTS_SKIP"
echo ""
if [[ "$N_PENDING" -lt 1 ]]; then
    echo "[scan] nothing to do."
    exit 0
fi

# ---------- pre-cleanup: wipe old hand outputs so the redo really redoes ----------
if [[ "$KEEP_EXISTING" != "1" ]]; then
    echo "[clean] removing old hand outputs for $N_PENDING exps:"
    for entry in "${PENDING[@]}"; do
        TASK_NAME="${entry%%__*}"
        EXP_NAME="${entry#*__}"
        ANN="${ANNOTATED_ROOT}/${TASK_NAME}/${EXP_NAME}"
        # Final merged outputs:
        rm -f "${ANN}/result.pkl" \
              "${ANN}/result_hand_optimized.pkl" \
              "${ANN}/processed/joint_pose_solver/poses_m.npy" 2>/dev/null
        # Per-chunk artifacts (left over from chunked runs):
        rm -f "${ANN}"/result_*_*.pkl \
              "${ANN}"/result_hand_optimized_*_*.pkl \
              "${ANN}"/poses_m_*_*.npy 2>/dev/null
    done
    echo "[clean] done"
    echo ""
fi

# ---------- preview ----------
for ((k=0; k<N_PENDING && k<30; k++)); do
    echo "  [$((k+1))] ${PENDING[$k]/__/ / }"
done
[[ "$N_PENDING" -gt 30 ]] && echo "  ... ($((N_PENDING - 30)) more)"
echo ""
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry_run] would launch $PARALLEL workers across $GPUS GPU(s)."
    exit 0
fi

# ---------- worker pool (same pattern as sbatch_run_hand_simple.sh) ----------
# Inner --skip_existing decision:
#   - default (KEEP_EXISTING=0): outputs were wiped → off (let it redo).
#   - --keep_existing: outputs preserved → on (resume from where we left off).
INNER_FLAGS=()
[[ "$KEEP_EXISTING" == "1" ]] && INNER_FLAGS+=(--skip_existing)

run_one() {
    local idx="$1"
    local entry="$2"
    local task="${entry%%__*}"
    local exp="${entry#*__}"
    local gpu=$((idx % GPUS))
    local task_folder="$VIDEOS_ROOT/$task"
    local scratch="$H5_SCRATCH_ROOT/w${idx}"
    mkdir -p "$scratch"
    local logf="$H5_SCRATCH_ROOT/log_${idx}_${task}_${exp}.log"
    echo "[$(date +%H:%M:%S)] [worker $idx → gpu $gpu] START  $task / $exp  (log: $logf)"

    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS="$PER_WORKER_THREADS"
    export MKL_NUM_THREADS="$PER_WORKER_THREADS"
    export OPENBLAS_NUM_THREADS="$PER_WORKER_THREADS"
    export NUMEXPR_NUM_THREADS="$PER_WORKER_THREADS"
    export H5_COMPRESSION=lzf

    {
        echo "[worker $idx] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv 2>&1 | head -5
    } >>"$logf" 2>&1

    bash "$HAND_BATCH_SCRIPT" \
        --task_folder "$task_folder" \
        --calibration_yaml "$CAL_YAML" \
        --chunk_size "$CHUNK_SIZE" \
        --h5_scratch_dir "$scratch" \
        --only_exp "$exp" \
        "${INNER_FLAGS[@]}" >>"$logf" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] [worker $idx → gpu $gpu] OK     $task / $exp"
    else
        echo "[$(date +%H:%M:%S)] [worker $idx → gpu $gpu] FAIL   $task / $exp  (rc=$rc, see $logf)"
    fi
    rm -rf "$scratch" 2>/dev/null || true
    return $rc
}

RESULT_DIR="$H5_SCRATCH_ROOT/_results"
mkdir -p "$RESULT_DIR"

n_started=0; n_ok=0; n_fail=0; running=0
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

for f in "$RESULT_DIR"/*.rc; do
    rc="$(cat "$f")"
    if [[ "$rc" == "0" ]]; then n_ok=$((n_ok+1)); else n_fail=$((n_fail+1)); fi
done

echo ""
echo "=========================================================================="
echo "summary: started=$n_started  ok=$n_ok  fail=$n_fail"
[[ "$n_fail" -gt 0 ]] && echo "see per-worker logs under $H5_SCRATCH_ROOT/log_*.log (wiped on exit)"
echo "=========================================================================="

[[ "$n_fail" -gt 0 ]] && exit 1 || exit 0
