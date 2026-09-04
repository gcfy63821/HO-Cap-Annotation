#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name depth_extract
#SBATCH --partition=viscam
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/depth_%A_%a.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/depth_%A_%a.err
#
# SLURM-array depth keyframe extractor.  No GPU required.
#
# THREE modes:
#
# (A) FRONTEND — run with `bash` (NOT sbatch):
#     bash sbatch_extract_depth.sh                        # all videos_* folders
#     bash sbatch_extract_depth.sh --videos_root A B C    # specific folders
#     [--shards 8] [--bundle ...] [--data_root ...] [--keyframe_fracs 0,0.1,0.2]
#     [--force] [--dry_run]
#
# (B) ARRAY CHILD (auto via $SLURM_ARRAY_TASK_ID): runs extract_depth_keyframes.py
#     --exp_list <file> --shard <k>/<N> --bundle <bundle>
#
# Example:
#   bash sbatch_extract_depth.sh \
#     --data_root /viscam/projects/robotool/data \
#     --bundle    /viscam/projects/robotool/_va_bundle_v2 \
#     --shards 8

set -u

export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
SLURM_OUTS="${SLURM_OUTS:-/viscam/u/chenrq/crq_ws/slurm_outs}"
INTERNAL="$HOCAP_ROOT/volunteer_annotation/internal"
EXTRACT="$INTERNAL/extract_depth_keyframes.py"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

DATA_ROOT="/viscam/projects/robotool/data"
VIDEOS_ROOTS=()
BUNDLE="/viscam/projects/robotool/_va_bundle_v2"
SHARDS=8
KEYFRAME_FRACS="0,0.1,0.2"
MANIFEST=""
FORCE=0
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --data_root)      DATA_ROOT="$2"; shift 2;;
        --videos_root)    shift; while [[ "$#" -gt 0 && "${1:0:2}" != "--" ]]; do VIDEOS_ROOTS+=("$1"); shift; done;;
        --bundle)         BUNDLE="$2"; shift 2;;
        --shards)         SHARDS="$2"; shift 2;;
        --keyframe_fracs) KEYFRAME_FRACS="$2"; shift 2;;
        --manifest)       MANIFEST="$2"; shift 2;;
        --force)          FORCE=1; shift;;
        --dry_run)        DRY_RUN=1; shift;;
        -h|--help)        sed -n '2,30p' "$0"; exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

FORCE_FLAG=$([[ "$FORCE" == "1" ]] && echo "--force" || echo "")

# =================================================================
#  Mode B: ARRAY CHILD
# =================================================================
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    [[ -n "$MANIFEST" && -f "$MANIFEST" ]] || { echo "[child] bad --manifest: $MANIFEST"; exit 1; }
    source "$CONDA_SH"; conda activate "$CONDA_ENV_NAME"
    echo "=============================="
    echo "ARRAY CHILD shard=${SLURM_ARRAY_TASK_ID}/${SHARDS} node=$(hostname)"
    echo "  manifest=$MANIFEST  bundle=$BUNDLE  fracs=$KEYFRAME_FRACS"
    echo "=============================="
    python "$EXTRACT" \
        --exp_list "$MANIFEST" \
        --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}" \
        --bundle "$BUNDLE" \
        --keyframe_fracs "$KEYFRAME_FRACS" \
        $FORCE_FLAG
    exit $?
fi

# =================================================================
#  Mode A: FRONTEND
# =================================================================
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SLURM_OUTS"

# Build exp list: discover all H5 exps under each videos_root
ROOTS=()
if [[ "${#VIDEOS_ROOTS[@]}" -gt 0 ]]; then
    ROOTS=("${VIDEOS_ROOTS[@]}")
else
    while IFS= read -r d; do ROOTS+=("${d%/}"); done \
        < <(ls -d "$DATA_ROOT"/videos_*/ 2>/dev/null | grep -v _annotated)
fi
[[ "${#ROOTS[@]}" -ge 1 ]] || { echo "No videos_* folders found under $DATA_ROOT"; exit 1; }

# Merge all roots into one exp list
ALL_MANIFEST="$BUNDLE/_depth_exp_list_${RUN_TS}.txt"
mkdir -p "$BUNDLE"
> "$ALL_MANIFEST"
for VR in "${ROOTS[@]}"; do
    [[ -d "$VR" ]] || { echo "[skip] not a dir: $VR"; continue; }
    # Enumerate exp dirs (H5 or mkv); filter out already-done ones unless --force
    python3 - "$VR" "$BUNDLE" $([[ "$FORCE" == "1" ]] && echo "force") << 'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
bundle = Path(sys.argv[2])
force = len(sys.argv) > 3

# Support both H5 and raw mkv/mp4 source formats
exps = {p.parent for p in root.rglob("data00000000.h5")}
exps |= {p.parent for p in root.rglob("cam0_depth.mkv")}
exps |= {p.parent for p in root.rglob("cam0_depth.mp4")}
exps = sorted(exps)

pending = []
for e in exps:
    rel = None
    for anc in e.parents:
        if anc.name.startswith("videos_"):
            rel = e.relative_to(anc.parent); break
    if rel is None:
        rel = Path(e.parent.name) / e.name
    done_marker = bundle / rel / "_depth_manifest.json"
    if force or not done_marker.is_file():
        pending.append(str(e))

print(f"[{root.name}] {len(exps)} total, {len(exps)-len(pending)} done, {len(pending)} pending", file=sys.stderr)
for p in pending:
    print(p)
PY
done >> "$ALL_MANIFEST"

N=$(grep -cve '^[[:space:]]*$' "$ALL_MANIFEST" 2>/dev/null || echo 0)
echo "[frontend] $N experiments pending  (manifest: $ALL_MANIFEST)"
[[ "$N" -ge 1 ]] || { echo "Nothing to do."; exit 0; }

NS=$SHARDS; (( NS > N )) && NS=$N
LOG="$SLURM_OUTS/depth_extract_${RUN_TS}.log"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry_run] would submit array 0-$((NS-1))%${NS} over $N exps"
    head -5 "$ALL_MANIFEST"
    exit 0
fi

JID=$(sbatch --parsable \
    --array=0-$((NS-1))%${NS} \
    --output="$LOG" --error="$LOG" --open-mode=append \
    "$0" \
    --manifest "$ALL_MANIFEST" \
    --bundle "$BUNDLE" \
    --shards "$NS" \
    --keyframe_fracs "$KEYFRAME_FRACS" \
    $FORCE_FLAG)

echo "[frontend] submitted array job $JID  ($N exps / $NS shards)"
echo "  log: $LOG"
echo "  monitor: squeue -j $JID"
