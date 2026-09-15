#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name va_masks
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/va_masks_%A_%a.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/va_masks_%A_%a.err
#
# SLURM array driver: volunteer point prompts -> SAM2 propagation -> masks.h5
#
# Each task takes a contiguous slice of a worklist and, per experiment:
#   <prompts_dir>/cam*_rgb.json  +  cam*_rgb.mp4
#     -> prompts_to_masks.py (SAM2 video propagation, 8 cams)
#     -> <data_root>/<videos_X>_annotated/<task>/<exp>/tool_masks/{masks.h5, objects.yaml}
#
# TWO MODES:
#
# (A) FRONTEND — run with bash (NOT sbatch): builds worklist then submits array.
#
#   # Single folder — quick test or targeted run:
#   bash sbatch_masks_array.sh --videos_root /viscam/projects/robotool/data/videos_0202
#
#   # All folders under data_root:
#   bash sbatch_masks_array.sh --data_root /viscam/projects/robotool/data
#
#   # With options:
#   bash sbatch_masks_array.sh --videos_root videos_0202 videos_0204 \
#       [--data_root /viscam/projects/robotool/data] \
#       [--prompts_root /viscam/projects/robotool/_va_bundle_v2_prompts] \
#       [--exps_per_task 8] [--max_concurrent 32] [--dry_run]
#
# (B) ARRAY CHILD — invoked automatically by sbatch (via $SLURM_ARRAY_TASK_ID):
#   processes a slice of the worklist, loads SAM2 once per task.
#
# RESUME: exps whose tool_masks/masks.h5 already has the correct shape are
#   skipped automatically. Re-submit the same command to resume.
#
# SIZING (rough estimates):
#   ~500 frames, 8 cams: ~3-4 min/exp; exps_per_task=8 -> ~30 min/task
#   3232 exps / 8 = 404 tasks; 32 concurrent -> ~2.5h wall time
#   Storage: ~12 MB/exp avg -> ~39 GB for 3232 exps

set -u

# ---- cluster paths ----
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
MASKS_SCRIPT="$HOCAP_ROOT/volunteer_annotation/internal/prompts_to_masks.py"
BUILD_WORKLIST="$HOCAP_ROOT/volunteer_annotation/internal/build_masks_worklist.py"
SLURM_OUTS="${SLURM_OUTS:-/viscam/u/chenrq/crq_ws/slurm_outs}"

export SAM2_ROOT="${SAM2_ROOT:-/viscam/u/chenrq/crq_ws/robotool/sam2}"
export SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt}"
export SAM2_VIDEO_CFG="${SAM2_VIDEO_CFG:-$HOCAP_ROOT/config/sam2_config/sam2.1_hiera_l.yaml}"

# ---- perf env ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

# ---- args ----
WORKLIST=""
VIDEOS_ROOTS=()
DATA_ROOT="/viscam/projects/robotool/data"
PROMPTS_ROOT="/viscam/projects/robotool/_va_bundle_v2_prompts"
EXPS_PER_TASK=8
MAX_CONCURRENT=32
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --worklist)       WORKLIST="$2"; shift 2 ;;
        --videos_root)    shift; while [[ $# -gt 0 && "${1:0:2}" != "--" ]]; do VIDEOS_ROOTS+=("$1"); shift; done ;;
        --data_root)      DATA_ROOT="$2"; shift 2 ;;
        --prompts_root)   PROMPTS_ROOT="$2"; shift 2 ;;
        --exps_per_task)  EXPS_PER_TASK="$2"; shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --dry_run)        DRY_RUN=1; shift ;;
        -h|--help)        sed -n '2,50p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =================================================================
#  Mode A: FRONTEND (no $SLURM_ARRAY_TASK_ID) — build worklist + submit
# =================================================================
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" && -z "$WORKLIST" ]]; then
    [[ -f "$BUILD_WORKLIST" ]] || { echo "Error: not found: $BUILD_WORKLIST"; exit 1; }

    # Resolve videos_root list: explicit args, else all videos_* under data_root
    ROOTS=()
    if [[ "${#VIDEOS_ROOTS[@]}" -gt 0 ]]; then
        for r in "${VIDEOS_ROOTS[@]}"; do
            [[ "$r" == /* ]] && ROOTS+=("$r") || ROOTS+=("$DATA_ROOT/$r")
        done
    else
        while IFS= read -r d; do ROOTS+=("${d%/}"); done \
            < <(ls -d "$DATA_ROOT"/videos_*/ 2>/dev/null | grep -v _annotated)
    fi
    [[ "${#ROOTS[@]}" -ge 1 ]] || { echo "Error: no videos_* folders found under $DATA_ROOT"; exit 1; }

    source "$CONDA_SH"; conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
    mkdir -p "$SLURM_OUTS"

    RUN_TS="$(date +%Y%m%d_%H%M%S)"
    for VR in "${ROOTS[@]}"; do
        [[ -d "$VR" ]] || { echo "[skip] not a dir: $VR"; continue; }
        NAME="$(basename "$VR")"
        WL="$SLURM_OUTS/masks_worklist_${NAME}_${RUN_TS}.tsv"

        echo "[frontend] scanning $NAME ..."
        python3 "$BUILD_WORKLIST" \
            --data_root    "$(dirname "$VR")" \
            --prompts_root "$PROMPTS_ROOT" \
            --out          "$WL" 2>&1 | grep -E '^\[worklist\]|skip|pending|->|Error'

        N=$(grep -c . "$WL" 2>/dev/null || echo 0)
        if [[ "$N" -eq 0 ]]; then
            echo "[frontend] $NAME: 0 pending exps — skip"
            continue
        fi

        N_TASKS=$(( (N + EXPS_PER_TASK - 1) / EXPS_PER_TASK ))
        LAST=$(( N_TASKS - 1 ))
        LOG="$SLURM_OUTS/va_masks_${NAME}_${RUN_TS}_%A_%a.out"

        echo "[frontend] $NAME: $N exps -> $N_TASKS tasks (array 0-${LAST}%${MAX_CONCURRENT})"
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[dry_run] sbatch --array=0-${LAST}%${MAX_CONCURRENT} ... --worklist $WL"
            continue
        fi

        JID=$(sbatch --parsable \
            --array="0-${LAST}%${MAX_CONCURRENT}" \
            --output="$LOG" --error="$LOG" \
            "$0" \
            --worklist "$WL" \
            --exps_per_task "$EXPS_PER_TASK")
        echo "[frontend] $NAME -> array job: $JID  (log: $LOG)"
    done
    exit 0
fi

# =================================================================
#  Mode B: ARRAY CHILD
# =================================================================
[[ -n "$WORKLIST" && -f "$WORKLIST" ]] || {
    echo "Error: --worklist <tsv> required in array child mode."; exit 1; }
[[ -f "$MASKS_SCRIPT" ]] || { echo "Error: not found: $MASKS_SCRIPT"; exit 1; }

# ---- /dev/shm scratch for JPEG frames SAM2 needs ----
_JOBTAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-nonslurm_$$}}_${SLURM_ARRAY_TASK_ID:-0}"
export TMP_DIR="/dev/shm/${USER}/${_JOBTAG}"
mkdir -p "$TMP_DIR"
cleanup_shm() { [[ -d "$TMP_DIR" ]] && rm -rf "$TMP_DIR" 2>/dev/null || true; }
trap cleanup_shm EXIT INT TERM HUP

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_LINES=$(wc -l < "$WORKLIST")
START=$(( TASK_ID * EXPS_PER_TASK + 1 ))
END=$(( START + EXPS_PER_TASK - 1 ))
(( END > N_LINES )) && END=$N_LINES

echo "=========================================="
echo "job      : ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-<non-slurm>}} task ${TASK_ID}   node: $(hostname)"
echo "worklist : $WORKLIST ($N_LINES exps)"
echo "slice    : lines ${START}..${END}"
echo "scratch  : $TMP_DIR ($(df -h /dev/shm 2>/dev/null | awk 'NR==2 {print $4 " free"}'))"
echo "=========================================="

if (( START > N_LINES )); then
    echo "[done] array task ${TASK_ID} is past the end of the worklist."
    exit 0
fi

# ---- conda ----
if [[ "$DRY_RUN" == "0" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME"
fi

# Derive the output dir for an experiment:
#   /viscam/.../data/videos_0102/task/exp  ->  /viscam/.../data/videos_0102_annotated/task/exp/tool_masks
# Falls back to <exp>/tool_masks if videos_XXXX ancestor not found.
out_dir_for_exp() {
    local exp="$1" base="$2"
    if [[ -n "$base" ]]; then
        # When --out_base given: out_base/videos_X_annotated/task/exp/tool_masks
        local videos_dir task_dir exp_dir
        exp_dir="$(basename "$exp")"
        task_dir="$(basename "$(dirname "$exp")")"
        videos_dir="$(basename "$(dirname "$(dirname "$exp")")")"
        echo "${base}/${videos_dir}_annotated/${task_dir}/${exp_dir}/tool_masks"
    else
        # Auto-detect: walk up to the videos_XXXX ancestor
        local cur="$exp" prev="" videos_anc=""
        while [[ "$cur" != "/" && "$cur" != "$prev" ]]; do
            if [[ "$(basename "$cur")" == videos_* ]]; then
                videos_anc="$cur"; break
            fi
            prev="$cur"; cur="$(dirname "$cur")"
        done
        if [[ -n "$videos_anc" ]]; then
            local rel="${exp#${videos_anc}/}"
            echo "${videos_anc}_annotated/${rel}/tool_masks"
        else
            echo "${exp}/tool_masks"
        fi
    fi
}

# ---- run the slice ----
n_ok=0; n_fail=0; failed=()

while IFS=$'\t' read -r EXP_DIR PROMPTS_DIR; do
    [[ -z "${EXP_DIR:-}" ]] && continue
    echo ""
    echo "------------------------------------------"
    echo "exp: $EXP_DIR"
    echo "------------------------------------------"

    if [[ ! -d "$EXP_DIR" ]]; then
        echo "[skip] exp dir not found on this node: $EXP_DIR"
        n_fail=$((n_fail+1)); failed+=("$EXP_DIR (no exp dir)"); continue
    fi

    OUT_DIR="$(out_dir_for_exp "$EXP_DIR" "${OUT_BASE:-}")"
    mkdir -p "$OUT_DIR"

    ARGS=(
        --exp         "$EXP_DIR"
        --prompts_dir "$PROMPTS_DIR"
        --out_dir     "$OUT_DIR"
        --tmp_dir     "$TMP_DIR"
        --from_video
        --resume
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry_run] python $MASKS_SCRIPT ${ARGS[*]}"
        echo "          out_dir: $OUT_DIR"
        continue
    fi

    if python "$MASKS_SCRIPT" "${ARGS[@]}"; then
        n_ok=$((n_ok+1))
    else
        rc=$?
        echo "[FAIL rc=$rc] $EXP_DIR"
        n_fail=$((n_fail+1)); failed+=("$EXP_DIR (rc=$rc)")
    fi

    # Free /dev/shm frame dump between experiments.
    rm -rf "$TMP_DIR"/frames_* "$TMP_DIR"/tmp_frames_* 2>/dev/null || true

done < <(sed -n "${START},${END}p" "$WORKLIST")

echo ""
echo "=========================================="
echo "array task ${TASK_ID}: ok=$n_ok fail=$n_fail"
for f in "${failed[@]:-}"; do [[ -n "$f" ]] && echo "  FAILED: $f"; done
echo "=========================================="
exit 0
