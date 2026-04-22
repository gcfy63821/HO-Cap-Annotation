#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name hrec
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/%j.err
#
# Cluster-optimized sbatch wrapper for hand annotation pipeline.
#
# Improvements over the previous run_hand_aligned_auto.sh:
#   - h5 files land in /dev/shm (RAM tmpfs, ~10 GB/s), NOT on /viscam NFS
#     -> cuts h5 build time from ~8 min to ~1 min per chunk when many jobs
#        run in parallel and the /viscam link is saturated
#   - H5_COMPRESSION=lzf (5-10x faster than gzip, minimal size cost)
#   - numpy/OpenBLAS/MKL thread count matches --cpus-per-task (4)
#   - /dev/shm scratch gets auto-cleaned even on kill / signal
#   - Defaults to --use_aligned_only so the job never touches calibration;
#     pass --run_alignment to opt back in.
#
# Usage:
#   sbatch scripts/sbatch_run_hand_cluster.sh --videos_root /viscam/projects/robotool/data/videos_0101
#     [--chunk_size 500]
#     [--skip_existing]         # resume-safe; SKIP experiments that already
#                               # have a merged result_hand_optimized.pkl
#     [--keep_h5]
#     [--no_merge] [--keep_chunk_files]
#     [--cal_frame_idx 100]
#     [--representative_exp NAME]
#     [--force_recalibrate]     # only honored with --run_alignment
#     [--skip_calibration]      # use ORIG yaml directly (no aligned)
#     [--use_aligned_only]      # default on this script; uses *_global_aligned.yaml
#     [--run_alignment]         # explicit opt-in to run the alignment step
#     [--skip_hand]             # calibration only
#
# Tip: always pass --skip_existing for resume-safety across kills / requeues.

set -u

# ---------- cluster paths ----------
# Exported so the batch script (which now honors these env vars) picks them up
# instead of its local-dev defaults. No more "_cluster" fork needed.
export HOCAP_ROOT="/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation"
export HAND_ROOT="/viscam/u/chenrq/crq_ws/robotool/HandReconstruction"
export MODELS_FOLDER="/viscam/u/chenrq/models"
export CONDA_SH="/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="hocap-annotation"
HAND_BATCH_SCRIPT="$HOCAP_ROOT/scripts/batch_task_folder_hand_chunked.sh"

# ---------- performance env ----------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export H5_COMPRESSION="lzf"

# ---------- fast scratch on node-local tmpfs ----------
_JOBTAG="${SLURM_JOB_ID:-nonslurm_$$}"
H5_SCRATCH="/dev/shm/${USER}/${_JOBTAG}"
mkdir -p "$H5_SCRATCH"

# Auto-cleanup /dev/shm on ANY exit (success, error, signal, scancel)
cleanup_shm() {
    if [[ -d "$H5_SCRATCH" ]]; then
        rm -rf "$H5_SCRATCH" 2>/dev/null || true
    fi
}
trap cleanup_shm EXIT INT TERM HUP

echo "=========================================="
echo "job           : ${SLURM_JOB_ID:-<non-slurm>}   node: $(hostname)"
echo "cpus/omp      : ${SLURM_CPUS_PER_TASK:-4}"
echo "H5_SCRATCH    : $H5_SCRATCH ($(df -h /dev/shm | awk 'NR==2 {print $4 " free"}'))"
echo "H5_COMPRESSION: $H5_COMPRESSION"
echo "=========================================="

# ---------- arg parsing ----------
VIDEOS_ROOT=""
CHUNK_SIZE=500
SKIP_EXISTING_FLAG=""
KEEP_H5_FLAG=""
NO_MERGE_FLAG=""
KEEP_CHUNK_FILES_FLAG=""
CAL_FRAME_IDX=100   # frame 0 = realsense warm-up, depth = 0
REPRESENTATIVE_EXP=""
FORCE_RECAL=0
SKIP_CAL=0
USE_ALIGNED_ONLY=1   # cluster default: don't re-run alignment
SKIP_HAND=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --videos_root)        VIDEOS_ROOT="$2"; shift 2;;
        --chunk_size)         CHUNK_SIZE="$2"; shift 2;;
        --skip_existing)      SKIP_EXISTING_FLAG="--skip_existing"; shift;;
        --keep_h5)            KEEP_H5_FLAG="--keep_h5"; shift;;
        --no_merge)           NO_MERGE_FLAG="--no_merge"; shift;;
        --keep_chunk_files)   KEEP_CHUNK_FILES_FLAG="--keep_chunk_files"; shift;;
        --cal_frame_idx)      CAL_FRAME_IDX="$2"; shift 2;;
        --representative_exp) REPRESENTATIVE_EXP="$2"; shift 2;;
        --force_recalibrate)  FORCE_RECAL=1; shift;;
        --skip_calibration)   SKIP_CAL=1; USE_ALIGNED_ONLY=0; shift;;
        --use_aligned_only)   USE_ALIGNED_ONLY=1; shift;;
        --run_alignment)      USE_ALIGNED_ONLY=0; shift;;
        --skip_hand)          SKIP_HAND=1; shift;;
        -h|--help) sed -n '2,40p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$VIDEOS_ROOT" ]]; then
    echo "Error: --videos_root is required"; exit 1
fi
VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
if [[ ! -d "$VIDEOS_ROOT" ]]; then
    echo "Error: videos_root not a directory: $VIDEOS_ROOT"; exit 1
fi

# ---------- conda + ffmpeg ----------
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg not found; depth videos will decode as zeros."
    echo "       Install via: conda install -c conda-forge ffmpeg -y"
    exit 1
fi

echo "videos_root = $VIDEOS_ROOT"

# ---------- step A: discover calibration ----------
CAL_FOLDER="$(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1)"
if [[ -z "$CAL_FOLDER" ]]; then
    echo "Error: no realsense_calibrate_* folder under $VIDEOS_ROOT"; exit 1
fi
ORIG_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' \
              ! -name '*_global_aligned.yaml' ! -name '*_slider_aligned.yaml' \
              ! -name '*_manual_aligned.yaml' ! -name '*_aligned.yaml' | head -n 1)"
if [[ -z "$ORIG_YAML" ]]; then
    echo "Error: no realsense_calibration_*.yaml in $CAL_FOLDER"; exit 1
fi
YAML_STEM="$(basename "$ORIG_YAML" .yaml)"
ALIGNED_YAML="${CAL_FOLDER}/${YAML_STEM}_global_aligned.yaml"
SNAPSHOT_PNG="${CAL_FOLDER}/calibration_snapshot.png"
POSTALIGN_PLY="${CAL_FOLDER}/postalign_global.ply"

echo "[calibration]"
echo "  folder     : $CAL_FOLDER"
echo "  orig yaml  : $(basename "$ORIG_YAML")"
echo "  aligned    : $ALIGNED_YAML"

# ---------- step B: discover tasks ----------
TASK_FOLDERS=()
while IFS= read -r -d '' d; do
    name="$(basename "$d")"
    [[ "$name" == realsense_calibrate_* ]] && continue
    TASK_FOLDERS+=("$d")
done < <(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

if [[ "${#TASK_FOLDERS[@]}" -eq 0 ]]; then
    echo "Error: no task folders under $VIDEOS_ROOT"; exit 1
fi
echo "[tasks] ${#TASK_FOLDERS[@]} task folder(s):"
for t in "${TASK_FOLDERS[@]}"; do echo "   - $(basename "$t")"; done

# ---------- step C: calibration ----------
if [[ "$USE_ALIGNED_ONLY" == "1" ]]; then
    if [[ ! -f "$ALIGNED_YAML" ]]; then
        echo "Error: --use_aligned_only but $ALIGNED_YAML does not exist."
        echo "       Produce it first via scripts/batch_global_align.sh or"
        echo "       scripts/manual_align_viser_multi.py, or pass --run_alignment."
        exit 1
    fi
    echo "[cal] using existing aligned yaml (no alignment run)"
    CAL_YAML_TO_USE="$ALIGNED_YAML"
elif [[ "$SKIP_CAL" == "1" ]]; then
    echo "[cal] --skip_calibration, using ORIG yaml"
    CAL_YAML_TO_USE="$ORIG_YAML"
elif [[ -f "$ALIGNED_YAML" && "$FORCE_RECAL" == "0" ]]; then
    echo "[cal] aligned yaml already exists, skipping alignment"
    CAL_YAML_TO_USE="$ALIGNED_YAML"
else
    FIRST_TASK="${TASK_FOLDERS[0]}"
    if [[ -n "$REPRESENTATIVE_EXP" ]]; then
        REP_DIR="$FIRST_TASK/$REPRESENTATIVE_EXP"
    else
        REP_DIR=""
        while IFS= read -r -d '' d; do
            name="$(basename "$d")"
            [[ "$name" == *.tar.gz || "$name" == board_reference ]] && continue
            if ls "$d"/cam*_rgb.mp4 >/dev/null 2>&1; then
                REP_DIR="$d"; break
            fi
        done < <(find "$FIRST_TASK" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
    fi
    if [[ -z "$REP_DIR" || ! -d "$REP_DIR" ]]; then
        echo "Error: no representative exp under $FIRST_TASK"; exit 1
    fi
    echo "[cal] rep_exp: $REP_DIR"

    # Build tiny h5 into /dev/shm to avoid NFS write during calibration too.
    TINY_H5="${H5_SCRATCH}/tiny_cal_data00000000.h5"
    CACHED_PC_DIR="${CAL_FOLDER}/cached_pc"

    echo "[cal] build tiny h5 (video frame $CAL_FRAME_IDX) -> $TINY_H5"
    rm -f "$TINY_H5"
    if ! python "$HOCAP_ROOT/tools/00_convert_videos_to_h5.py" \
            --input_dir "$REP_DIR" --output_file "$TINY_H5" \
            --start_frame "$CAL_FRAME_IDX" --end_frame "$CAL_FRAME_IDX"; then
        echo "Error: tiny h5 conversion failed"; exit 1
    fi

    echo "[cal] cache per-cam PCs (cache_pc_only.py)"
    if ! python "$HOCAP_ROOT/tools/cache_pc_only.py" \
            --h5_file "$TINY_H5" --extrinsic_file "$ORIG_YAML" \
            --out_path "$CAL_FOLDER" --frame_idx 0 \
            --x_threshold -5 5 --y_threshold -5 5 --z_threshold -5 5; then
        echo "Error: cache_pc_only failed"; exit 1
    fi

    echo "[cal] global alignment (headless)"
    if ! python "$HOCAP_ROOT/tools/run_global_align_headless.py" \
            --cached_pc "$CACHED_PC_DIR" --extrinsic_file "$ORIG_YAML" \
            --out_path "$CAL_FOLDER"; then
        echo "Error: global alignment failed"; exit 1
    fi
    rm -f "$TINY_H5"

    if [[ -f "$POSTALIGN_PLY" ]]; then
        python "$HOCAP_ROOT/scripts/render_calibration_snapshot.py" \
            --ply "$POSTALIGN_PLY" --out "$SNAPSHOT_PNG" || echo "[cal] snapshot failed (yaml still produced)"
    fi
    if [[ ! -f "$ALIGNED_YAML" ]]; then
        echo "Error: expected $ALIGNED_YAML was not produced"; exit 1
    fi
    CAL_YAML_TO_USE="$ALIGNED_YAML"
fi

echo ""
echo "[cal] using: $CAL_YAML_TO_USE"

# ---------- step D: hand annotation per task ----------
if [[ "$SKIP_HAND" == "1" ]]; then
    echo "[hand] --skip_hand, done"
    exit 0
fi

TASK_OK=0; TASK_FAIL=0
FAILED_TASKS=()

for task_dir in "${TASK_FOLDERS[@]}"; do
    task_name="$(basename "$task_dir")"
    echo ""
    echo "##########################################"
    echo "# task: $task_name"
    echo "##########################################"

    # The --h5_scratch_dir flag requires the updated batch_task_folder_hand_chunked*.sh
    # (the one that writes h5 to that dir and symlinks into the sequence folder).
    if bash "$HAND_BATCH_SCRIPT" \
            --task_folder "$task_dir" \
            --calibration_yaml "$CAL_YAML_TO_USE" \
            --chunk_size "$CHUNK_SIZE" \
            --h5_scratch_dir "$H5_SCRATCH" \
            $SKIP_EXISTING_FLAG $KEEP_H5_FLAG $NO_MERGE_FLAG $KEEP_CHUNK_FILES_FLAG; then
        TASK_OK=$((TASK_OK+1))
    else
        TASK_FAIL=$((TASK_FAIL+1))
        FAILED_TASKS+=("$task_name")
    fi
done

echo ""
echo "=========================================="
echo "summary: tasks_ok=$TASK_OK  tasks_failed=$TASK_FAIL"
if [[ "${#FAILED_TASKS[@]}" -gt 0 ]]; then
    echo "failed tasks:"
    for t in "${FAILED_TASKS[@]}"; do echo "  - $t"; done
fi
echo "=========================================="
