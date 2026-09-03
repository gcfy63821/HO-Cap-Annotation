#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name hvol
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/%A_%a.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/%A_%a.err
#
# SLURM array driver for the VOLUNTEER-annotated object pipeline.
#
# One array task takes a contiguous slice of a worklist and, per experiment:
#   volunteer point prompts -> SAM2 propagation -> masks.h5
#     -> generate_meta -> per-object FoundationPose tracking (Kalman + 2D
#        tracker) -> multi-view pose merge
# i.e. scripts/run_volunteer_annotator.sh, which is itself a thin shim in front
# of the existing scripts/run_auto_annotator.sh (Stage 2 DINO replaced).
#
# The worklist is a TSV produced by scripts/volunteer_exp_index.py:
#   <sequence_folder>\t<tool_name>\t<tool_mesh>\t<prompts_dir>
# Build it once on a login node (it only stats files, no GPU):
#
#   python scripts/volunteer_exp_index.py \
#       --prompts_root  /viscam/projects/robotool/_va_bundle_v2_prompts \
#       --data_root     /viscam/projects/robotool/data \
#       --models_folder /viscam/u/chenrq/models \
#       --require_sequence --require_mesh \
#       --out /viscam/u/chenrq/crq_ws/volunteer_worklist.tsv
#
# Then submit (N array tasks, each handling EXPS_PER_TASK experiments):
#
#   sbatch --array=0-99%16 scripts/sbatch_run_volunteer_array.sh \
#       --worklist /viscam/u/chenrq/crq_ws/volunteer_worklist.tsv \
#       --exps_per_task 4 \
#       [--fake_optimize] [--hand 0] [--object_chunk_size 600] [--dry_run]
#
# Sizing: ceil(n_lines / exps_per_task) array tasks covers the whole list.
# Print the number to request with:
#   awk 'END{print int((NR+3)/4)-1}' worklist.tsv    # for exps_per_task=4
#
# Resume: every stage (masks.h5 included) skips when its output already exists,
# so re-submitting the same array after a partial run is cheap. --force redoes
# everything, including the masks.
#
# Any unrecognised flag is forwarded to run_volunteer_annotator.sh, and from
# there to run_auto_annotator.sh.

set -u

# ---------- cluster paths ----------
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export HAND_ROOT="${HAND_ROOT:-/viscam/u/chenrq/crq_ws/robotool/HandReconstruction}"
export MODELS_FOLDER="${MODELS_FOLDER:-/viscam/u/chenrq/models}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
RUN_SCRIPT="$HOCAP_ROOT/scripts/run_volunteer_annotator.sh"

# SAM2 — prompts_to_masks.py resolves the checkpoint from $SAM2_CKPT.
export SAM2_ROOT="${SAM2_ROOT:-/viscam/u/chenrq/crq_ws/robotool/sam2}"
export SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt}"

# ---------- perf env ----------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export H5_COMPRESSION="lzf"
export PYOPENGL_PLATFORM="egl"

# ---------- /dev/shm scratch (h5 + the per-camera JPEG dump SAM2 needs) ----------
_JOBTAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-nonslurm_$$}}_${SLURM_ARRAY_TASK_ID:-0}"
export H5_SCRATCH_DIR="/dev/shm/${USER}/${_JOBTAG}"
mkdir -p "$H5_SCRATCH_DIR"
cleanup_shm() { [[ -d "$H5_SCRATCH_DIR" ]] && rm -rf "$H5_SCRATCH_DIR" 2>/dev/null || true; }
trap cleanup_shm EXIT INT TERM HUP

# ---------- args ----------
WORKLIST=""
EXPS_PER_TASK=4
CALIBRATION_YAML=""
DRY_RUN=0
PASSTHRU=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --worklist)          WORKLIST="$2"; shift 2 ;;
        --exps_per_task)     EXPS_PER_TASK="$2"; shift 2 ;;
        --calibration_yaml)  CALIBRATION_YAML="$2"; shift 2 ;;
        --dry_run)           DRY_RUN=1; shift ;;
        *)                   PASSTHRU+=("$1"); shift ;;
    esac
done

[[ -n "$WORKLIST" && -f "$WORKLIST" ]] || {
    echo "Error: --worklist <tsv> is required (build it with scripts/volunteer_exp_index.py)"; exit 1; }
[[ -f "$RUN_SCRIPT" ]] || { echo "Error: not found: $RUN_SCRIPT"; exit 1; }

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_LINES=$(wc -l < "$WORKLIST")
START=$(( TASK_ID * EXPS_PER_TASK + 1 ))
END=$(( START + EXPS_PER_TASK - 1 ))
(( END > N_LINES )) && END=$N_LINES

echo "=========================================="
echo "job      : ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-<non-slurm>}} task ${TASK_ID}   node: $(hostname)"
echo "worklist : $WORKLIST ($N_LINES exps)"
echo "slice    : lines ${START}..${END}"
echo "scratch  : $H5_SCRATCH_DIR ($(df -h /dev/shm 2>/dev/null | awk 'NR==2 {print $4 " free"}'))"
echo "=========================================="

if (( START > N_LINES )); then
    echo "[done] array task ${TASK_ID} is past the end of the worklist — nothing to do."
    exit 0
fi

# ---------- conda ----------
if [[ "$DRY_RUN" == "0" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME"
    command -v ffmpeg >/dev/null 2>&1 || {
        echo "Error: ffmpeg not on PATH; depth decoding would silently produce zeros."; exit 1; }
fi

# ---------- per-videos_XXXX calibration discovery (cached) ----------
declare -A CAL_CACHE
discover_calibration() {
    local videos_root="$1"
    if [[ -n "${CAL_CACHE[$videos_root]:-}" ]]; then
        echo "${CAL_CACHE[$videos_root]}"; return
    fi
    local cal cal_folder
    cal_folder="$(find "$videos_root" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1)"
    if [[ -n "$cal_folder" ]]; then
        cal="$(find "$cal_folder" -maxdepth 1 -type f -name '*_global_aligned.yaml' | head -n 1)"
        [[ -z "$cal" ]] && cal="$(find "$cal_folder" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' ! -name '*_aligned.yaml' | head -n 1)"
    else
        cal="$(find "$videos_root" -maxdepth 1 -type f -name 'realsense_calibration_*_global_aligned.yaml' | head -n 1)"
        [[ -z "$cal" ]] && cal="$(find "$videos_root" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' ! -name '*_aligned.yaml' | head -n 1)"
    fi
    CAL_CACHE[$videos_root]="$cal"
    echo "$cal"
}

# ---------- run the slice ----------
n_ok=0; n_fail=0; failed=()
while IFS=$'\t' read -r SEQ TOOL_NAME TOOL_MESH PROMPTS_DIR; do
    [[ -z "${SEQ:-}" ]] && continue
    echo ""
    echo "------------------------------------------"
    echo "exp: $SEQ"
    echo "------------------------------------------"

    if [[ ! -d "$SEQ" ]]; then
        echo "[skip] sequence folder missing on this node: $SEQ"
        n_fail=$((n_fail+1)); failed+=("$SEQ (no sequence folder)"); continue
    fi

    CAL="$CALIBRATION_YAML"
    if [[ -z "$CAL" ]]; then
        # Cluster layout is <videos_root>/<task>/<exp>; the local mirror is flat
        # (<videos_root>/<exp>), so try both levels.
        CAL="$(discover_calibration "$(dirname "$(dirname "$SEQ")")")"
        [[ -z "$CAL" ]] && CAL="$(discover_calibration "$(dirname "$SEQ")")"
    fi
    if [[ -z "$CAL" || ! -f "$CAL" ]]; then
        echo "[skip] no calibration yaml for $SEQ"
        n_fail=$((n_fail+1)); failed+=("$SEQ (no calibration)"); continue
    fi

    ARGS=(
        --sequence_folder  "$SEQ"
        --prompts_dir      "$PROMPTS_DIR"
        --calibration_yaml "$CAL"
        --models_folder    "$MODELS_FOLDER"
        --h5_scratch_dir   "$H5_SCRATCH_DIR"
    )
    [[ -n "${TOOL_NAME:-}" ]] && ARGS+=(--tool_name "$TOOL_NAME")
    [[ -n "${TOOL_MESH:-}" && -f "$TOOL_MESH" ]] && ARGS+=(--tool_mesh "$TOOL_MESH")

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry_run] bash $RUN_SCRIPT ${ARGS[*]} ${PASSTHRU[*]:-}"
        continue
    fi

    if bash "$RUN_SCRIPT" "${ARGS[@]}" ${PASSTHRU[@]+"${PASSTHRU[@]}"}; then
        n_ok=$((n_ok+1))
    else
        rc=$?
        echo "[FAIL rc=$rc] $SEQ"
        n_fail=$((n_fail+1)); failed+=("$SEQ (rc=$rc)")
    fi
    # Free the tmpfs between experiments — a full /dev/shm OOM-kills the next one.
    rm -rf "$H5_SCRATCH_DIR"/va_frames 2>/dev/null || true
done < <(sed -n "${START},${END}p" "$WORKLIST")

echo ""
echo "=========================================="
echo "array task ${TASK_ID}: ok=$n_ok fail=$n_fail"
for f in "${failed[@]:-}"; do [[ -n "$f" ]] && echo "  FAILED: $f"; done
echo "=========================================="
exit 0
