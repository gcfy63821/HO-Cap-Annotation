#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name hand_arr
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/hand_arr_%A_%a.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/hand_arr_%A_%a.err
#
# Slurm-array hand reconstruction: one (videos_root, task, EXP) per array
# element, each gets its own GPU. Pending exps are auto-discovered: an exp is
# in the manifest iff <annotated>/<task>/<exp>/result_hand_optimized.pkl is
# missing. Per-exp scheduling means a long task no longer monopolises a GPU
# — each of its missing exps gets its own array slot, in parallel.
#
# Two invocation modes (the script auto-detects which):
#
# (A) Frontend mode  — run with `bash`, NOT `sbatch`:
#     bash scripts/sbatch_run_hand_array.sh \\
#         --data_root /viscam/projects/robotool/data \\
#         [--max_concurrent 16] [--chunk_size 500] [--skip_existing]
#         [--manifest_out /abs/path.tsv]   (default: /tmp/$USER/hand_array_<ts>.tsv)
#
#     What it does:
#       1. Scans every videos_<id>/ under --data_root (or each --videos_root
#          you list explicitly). Skips videos_root with no calibration folder.
#       2. For each task, checks whether at least one exp is missing
#          result_hand_optimized.pkl. Includes only those tasks.
#       3. Writes <videos_root>\\t<task_name> per line to the manifest TSV.
#       4. Runs `exec sbatch --array=0-$((N-1))%MAX scripts/sbatch_run_hand_array.sh
#                  --manifest <tsv>` to schedule N parallel array elements.
#
#     You can also produce the manifest yourself. Format is THREE TAB-
#     separated columns per line:
#         <videos_root_abs>\\t<task_name>\\t<exp_name>
#     and submit directly:
#     sbatch --array=0-$((N-1))%16 scripts/sbatch_run_hand_array.sh \\
#         --manifest /abs/path.tsv
#
# (B) Array child mode  (auto-detected via $SLURM_ARRAY_TASK_ID):
#     Reads the corresponding line from --manifest, calibration is rediscovered
#     from <videos_root>/realsense_calibrate_*/, and
#     scripts/batch_task_folder_hand_chunked.sh is invoked just like
#     sbatch_run_hand_cluster.sh does for that one task.
#
# Notes:
#   - --use_aligned_only is the cluster default. Pre-produce
#     realsense_calibration_*_global_aligned.yaml beforehand (e.g. via
#     scripts/batch_global_align.sh). Tasks without aligned yaml will FAIL
#     in their array element.
#   - --skip_existing forwards to the inner batch and is ON by default —
#     each array element exits ~immediately for tasks that finished between
#     manifest creation and array launch (e.g. on requeue).
#   - h5 scratch goes to /dev/shm so it doesn't slam /viscam NFS when many
#     array children are running concurrently.

set -u

# ---------- cluster paths ----------
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export HAND_ROOT="${HAND_ROOT:-/viscam/u/chenrq/crq_ws/robotool/HandReconstruction}"
export MODELS_FOLDER="${MODELS_FOLDER:-/viscam/u/chenrq/models}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
HAND_BATCH_SCRIPT="$HOCAP_ROOT/scripts/batch_task_folder_hand_chunked.sh"

# ---------- perf env (used in array child mode) ----------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export H5_COMPRESSION="lzf"

# ---------- args ----------
MANIFEST=""
DATA_ROOT=""
VIDEOS_ROOTS=()
MAX_CONCURRENT=16
CHUNK_SIZE=500
SKIP_EXISTING=1            # forwarded to inner batch
KEEP_H5=0
NO_MERGE=0
KEEP_CHUNK_FILES=0
NO_REQUIRE_CALIB=0
MANIFEST_OUT=""
DRY_RUN=0

# Sticky positional accumulator for --videos_root which may take many values.
parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --manifest)       MANIFEST="$2"; shift 2;;
            --manifest_out)   MANIFEST_OUT="$2"; shift 2;;
            --data_root)      DATA_ROOT="$2"; shift 2;;
            --videos_root)
                shift
                while [[ "$#" -gt 0 && "${1:0:2}" != "--" ]]; do
                    VIDEOS_ROOTS+=("$1"); shift
                done
                ;;
            --max_concurrent) MAX_CONCURRENT="$2"; shift 2;;
            --chunk_size)     CHUNK_SIZE="$2"; shift 2;;
            --skip_existing)  SKIP_EXISTING=1; shift;;
            --no_skip_existing) SKIP_EXISTING=0; shift;;
            --keep_h5)        KEEP_H5=1; shift;;
            --no_merge)       NO_MERGE=1; shift;;
            --keep_chunk_files) KEEP_CHUNK_FILES=1; shift;;
            --no_require_calib) NO_REQUIRE_CALIB=1; shift;;
            --dry_run)        DRY_RUN=1; shift;;
            -h|--help)        sed -n '2,80p' "$0"; exit 0;;
            *) echo "Unknown option $1"; exit 1;;
        esac
    done
}
parse_args "$@"


# =================================================================
#  Mode B: ARRAY CHILD
# =================================================================
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
        echo "[child] missing or unreadable --manifest: $MANIFEST"; exit 1
    fi
    LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
    LINE="$(sed -n "${LINE_NO}p" "$MANIFEST")"
    if [[ -z "$LINE" ]]; then
        echo "[child] empty manifest line $LINE_NO"; exit 1
    fi
    VR="$(awk -F'\t' '{print $1}' <<< "$LINE")"
    TASK="$(awk -F'\t' '{print $2}' <<< "$LINE")"
    EXP="$(awk -F'\t' '{print $3}' <<< "$LINE")"
    if [[ -z "$VR" || -z "$TASK" || -z "$EXP" ]]; then
        echo "[child] malformed manifest line (need 3 TAB cols): $LINE"; exit 1
    fi

    # /dev/shm scratch per-array-element.
    _JOBTAG="${SLURM_JOB_ID:-nonslurm_$$}_a${SLURM_ARRAY_TASK_ID}"
    H5_SCRATCH="/dev/shm/${USER}/${_JOBTAG}"
    mkdir -p "$H5_SCRATCH"
    cleanup_shm() { [[ -d "$H5_SCRATCH" ]] && rm -rf "$H5_SCRATCH" 2>/dev/null || true; }
    trap cleanup_shm EXIT INT TERM HUP

    # Conda
    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME"

    # Discover aligned calibration yaml for this videos_root.
    CAL_FOLDER="$(find "$VR" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1)"
    if [[ -z "$CAL_FOLDER" ]]; then
        echo "[child] no realsense_calibrate_* under $VR"; exit 2
    fi
    CAL_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name 'realsense_calibration_*_global_aligned.yaml' | head -n 1)"
    if [[ -z "$CAL_YAML" ]]; then
        echo "[child] no *_global_aligned.yaml under $CAL_FOLDER (run scripts/batch_global_align.sh first)"
        exit 2
    fi

    TASK_FOLDER="$VR/$TASK"
    EXP_DIR="$TASK_FOLDER/$EXP"
    [[ -d "$TASK_FOLDER" ]] || { echo "[child] task folder vanished: $TASK_FOLDER"; exit 2; }
    [[ -d "$EXP_DIR" ]] || { echo "[child] exp folder vanished: $EXP_DIR"; exit 2; }

    echo "=========================================="
    echo "ARRAY CHILD  job=${SLURM_JOB_ID}  array_id=${SLURM_ARRAY_TASK_ID}  node=$(hostname)"
    echo "  videos_root : $VR"
    echo "  task        : $TASK"
    echo "  exp         : $EXP"
    echo "  exp_folder  : $EXP_DIR"
    echo "  calibration : $CAL_YAML"
    echo "  scratch     : $H5_SCRATCH"
    echo "=========================================="

    INNER_FLAGS=()
    [[ "$SKIP_EXISTING" == "1" ]] && INNER_FLAGS+=(--skip_existing)
    [[ "$KEEP_H5" == "1" ]] && INNER_FLAGS+=(--keep_h5)
    [[ "$NO_MERGE" == "1" ]] && INNER_FLAGS+=(--no_merge)
    [[ "$KEEP_CHUNK_FILES" == "1" ]] && INNER_FLAGS+=(--keep_chunk_files)

    bash "$HAND_BATCH_SCRIPT" \
        --task_folder "$TASK_FOLDER" \
        --calibration_yaml "$CAL_YAML" \
        --chunk_size "$CHUNK_SIZE" \
        --h5_scratch_dir "$H5_SCRATCH" \
        --only_exp "$EXP" \
        "${INNER_FLAGS[@]}"
    RC=$?
    echo "[child] inner rc=$RC  ($TASK/$EXP)"
    exit $RC
fi


# =================================================================
#  Mode A: FRONTEND  (build manifest + resubmit as array)
# =================================================================
if [[ -n "$MANIFEST" && -f "$MANIFEST" ]]; then
    # User pre-built a manifest; no scanning, just submit.
    N=$(wc -l < "$MANIFEST")
    [[ "$N" -lt 1 ]] && { echo "manifest empty: $MANIFEST"; exit 0; }
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "would submit: sbatch --array=0-$((N-1))%${MAX_CONCURRENT} $0 --manifest $MANIFEST"
        exit 0
    fi
    echo "submitting array of $N child tasks (concurrency=${MAX_CONCURRENT})"
    exec sbatch --array=0-$((N-1))%${MAX_CONCURRENT} "$0" \
        --manifest "$MANIFEST" \
        --chunk_size "$CHUNK_SIZE" \
        $([[ "$SKIP_EXISTING" == "1" ]] && echo --skip_existing) \
        $([[ "$KEEP_H5" == "1" ]] && echo --keep_h5) \
        $([[ "$NO_MERGE" == "1" ]] && echo --no_merge) \
        $([[ "$KEEP_CHUNK_FILES" == "1" ]] && echo --keep_chunk_files)
fi

# Otherwise: scan to build manifest.
if [[ -z "$DATA_ROOT" && "${#VIDEOS_ROOTS[@]}" -eq 0 ]]; then
    echo "Error: provide --data_root or --videos_root <p1> <p2> ... or --manifest <tsv>"
    exit 1
fi

if [[ -z "$MANIFEST_OUT" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    MANIFEST_OUT="/tmp/${USER}/hand_array_${TS}.tsv"
    mkdir -p "$(dirname "$MANIFEST_OUT")"
fi

DATA_ROOT_ARG="${DATA_ROOT:-}"
echo "[scan] writing manifest -> $MANIFEST_OUT"
echo "[scan] data_root        = ${DATA_ROOT_ARG:-<unset>}"
echo "[scan] videos_roots     = ${VIDEOS_ROOTS[*]:-<unset>}"
echo "[scan] no_require_calib = $NO_REQUIRE_CALIB"

# Inline python: scan, emit one line per pending task.
DATA_ROOT_ARG="$DATA_ROOT_ARG" \
NO_REQUIRE_CALIB="$NO_REQUIRE_CALIB" \
MANIFEST_OUT="$MANIFEST_OUT" \
python3 - "${VIDEOS_ROOTS[@]}" <<'PY'
import os, sys
from pathlib import Path

data_root = (os.environ.get("DATA_ROOT_ARG") or "").strip()
no_require_calib = os.environ.get("NO_REQUIRE_CALIB", "0") == "1"
manifest_out = Path(os.environ["MANIFEST_OUT"])
extra_videos_roots = [Path(p).resolve() for p in sys.argv[1:]]

videos_roots = list(extra_videos_roots)
if data_root:
    dr = Path(data_root).expanduser().resolve()
    if not dr.is_dir():
        print(f"[scan][ERR] data_root not a dir: {dr}"); sys.exit(2)
    for p in sorted(dr.iterdir()):
        if not p.is_dir(): continue
        nm = p.name
        if not nm.startswith("videos_"): continue
        if nm.endswith("_annotated"): continue
        videos_roots.append(p.resolve())

# de-dup, keep order
seen = set(); uniq = []
for v in videos_roots:
    if v in seen: continue
    seen.add(v); uniq.append(v)
videos_roots = uniq

SKIP_PREFIXES = ("realsense_calibrate", "ref_pc", "posts")
SKIP_EXACT = {"first_frame", "tool_meshes", "models"}

def has_calib(vr: Path) -> bool:
    for p in vr.iterdir():
        if p.is_dir() and p.name.startswith("realsense_calibrate"):
            return True
    return False

def is_task_dir(p: Path) -> bool:
    if not p.is_dir(): return False
    if p.name in SKIP_EXACT: return False
    if any(p.name.startswith(s) for s in SKIP_PREFIXES): return False
    return True

lines = []
n_skipped_no_calib = 0
n_tasks_total = 0
n_tasks_with_pending = 0
n_exps_total = 0
n_exps_missing = 0

for vr in videos_roots:
    if not has_calib(vr) and not no_require_calib:
        print(f"[scan]   skip {vr.name}  (no realsense_calibrate_* folder)")
        n_skipped_no_calib += 1
        continue
    annotated_root = vr.parent / f"{vr.name}_annotated"
    for task_dir in sorted(p for p in vr.iterdir() if is_task_dir(p)):
        n_tasks_total += 1
        task_pending = 0
        for exp_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            if not list(exp_dir.glob("cam*_rgb.mp4")): continue
            n_exps_total += 1
            ann = annotated_root / task_dir.name / exp_dir.name
            if (ann / "result_hand_optimized.pkl").exists():
                continue
            n_exps_missing += 1
            task_pending += 1
            # One manifest line per pending exp — gives slurm-array per-exp
            # parallelism (each exp goes to its own GPU).
            lines.append(f"{vr}\t{task_dir.name}\t{exp_dir.name}")
        if task_pending > 0:
            n_tasks_with_pending += 1

manifest_out.parent.mkdir(parents=True, exist_ok=True)
manifest_out.write_text("\n".join(lines) + ("\n" if lines else ""))
print(f"[scan] videos_roots scanned : {len(videos_roots)}")
print(f"[scan]   skipped (no calib) : {n_skipped_no_calib}")
print(f"[scan] tasks total          : {n_tasks_total}")
print(f"[scan] tasks with pending   : {n_tasks_with_pending}")
print(f"[scan] exps total           : {n_exps_total}")
print(f"[scan] exps missing hand    : {n_exps_missing}    <- manifest size = array count")
PY
SCAN_RC=$?
[[ $SCAN_RC -ne 0 ]] && exit $SCAN_RC

N=$(wc -l < "$MANIFEST_OUT")
if [[ "$N" -lt 1 ]]; then
    echo "[scan] no pending tasks — nothing to submit. (manifest: $MANIFEST_OUT)"
    exit 0
fi

echo ""
echo "[scan] manifest:"
head -n 50 "$MANIFEST_OUT" | nl -ba
[[ "$N" -gt 50 ]] && echo "      ... ($((N-50)) more)"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry_run] would submit: sbatch --array=0-$((N-1))%${MAX_CONCURRENT} $0 --manifest $MANIFEST_OUT"
    exit 0
fi

echo "[submit] sbatch --array=0-$((N-1))%${MAX_CONCURRENT} $0 --manifest $MANIFEST_OUT"
exec sbatch --array=0-$((N-1))%${MAX_CONCURRENT} "$0" \
    --manifest "$MANIFEST_OUT" \
    --chunk_size "$CHUNK_SIZE" \
    $([[ "$SKIP_EXISTING" == "1" ]] && echo --skip_existing) \
    $([[ "$KEEP_H5" == "1" ]] && echo --keep_h5) \
    $([[ "$NO_MERGE" == "1" ]] && echo --no_merge) \
    $([[ "$KEEP_CHUNK_FILES" == "1" ]] && echo --keep_chunk_files)
