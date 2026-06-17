#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name precompute_arr
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/precompute_%A_%a.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/precompute_%A_%a.err
#
# SLURM-array embedding precompute for the volunteer annotation pipeline.
# Mirrors scripts/sbatch_run_hand_array.sh: one EXP per array element, each on
# its own GPU. Modeled on that script's frontend/child structure.
#
# THREE modes (auto-detected):
#
# (A) FRONTEND  — run with `bash` (NOT sbatch):
#     bash sbatch_precompute_array.sh \
#         --data_root /viscam/projects/robotool/data \
#         --bundle    /viscam/projects/robotool/_va_bundle \
#         [--videos_root /abs/p1 /abs/p2 ...]  [--max_concurrent 16]
#         [--refmask] [--keyframe_fracs 0,0.1,0.2] [--thumb_fracs 0.4,0.6]
#         [--manifest_out /abs/path.txt] [--dry_run]
#     Scans for exps (dirs with cam0_rgb.mp4 or data00000000.h5), writes a
#     one-exp-dir-per-line manifest, then submits
#       sbatch --array=0-(N-1)%MAX <self> --manifest <m> --bundle <b> ...
#     and a dependent merge job that runs after the array.
#
# (B) ARRAY CHILD  (auto via $SLURM_ARRAY_TASK_ID): reads its exp dir from the
#     manifest line, runs precompute_embeddings.py --exp <dir> --bundle <b>
#     --manifest_name manifest_<arrayid>.json (per-shard, no race).
#
# (C) MERGE  (--merge): merges manifest_*.json -> manifest.json (dependent job).
#
# Notes:
#   * --no_refmask is the default (the decode seam is already validated); pass
#     --refmask to also dump native SAM2 reference masks for re-validation.
#   * Resume-safe: re-running re-does exps but overwrites their files+fragments.
#   * Set HOCAP_ROOT / CONDA_SH / CONDA_ENV_NAME if your paths differ.

set -u

export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
INTERNAL="$HOCAP_ROOT/volunteer_annotation/internal"
PRECOMPUTE="$INTERNAL/precompute_embeddings.py"
MERGE="$INTERNAL/merge_manifests.py"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

MANIFEST=""
MANIFEST_OUT=""
DATA_ROOT="/viscam/projects/robotool/data"          # default; override with --data_root
VIDEOS_ROOTS=()
BUNDLE="/viscam/projects/robotool/_va_bundle"        # default; override with --bundle
MAX_CONCURRENT=16
REFMASK=0
KEYFRAME_FRACS="0,0.1,0.2"
THUMB_FRACS="0.4,0.6"
DO_MERGE=0
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --manifest)       MANIFEST="$2"; shift 2;;
        --manifest_out)   MANIFEST_OUT="$2"; shift 2;;
        --data_root)      DATA_ROOT="$2"; shift 2;;
        --videos_root)    shift; while [[ "$#" -gt 0 && "${1:0:2}" != "--" ]]; do VIDEOS_ROOTS+=("$1"); shift; done;;
        --bundle)         BUNDLE="$2"; shift 2;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2;;
        --refmask)        REFMASK=1; shift;;
        --keyframe_fracs) KEYFRAME_FRACS="$2"; shift 2;;
        --thumb_fracs)    THUMB_FRACS="$2"; shift 2;;
        --merge)          DO_MERGE=1; shift;;
        --dry_run)        DRY_RUN=1; shift;;
        -h|--help)        sed -n '2,55p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

REFMASK_FLAG=$([[ "$REFMASK" == "1" ]] && echo "" || echo "--no_refmask")

# =================================================================
#  Mode C: MERGE (dependent job)
# =================================================================
if [[ "$DO_MERGE" == "1" ]]; then
    [[ -n "$BUNDLE" ]] || { echo "[merge] --bundle required"; exit 1; }
    source "$CONDA_SH"; conda activate "$CONDA_ENV_NAME"
    python "$MERGE" --bundle "$BUNDLE"
    exit $?
fi

# =================================================================
#  Mode B: ARRAY CHILD
# =================================================================
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    [[ -n "$MANIFEST" && -f "$MANIFEST" ]] || { echo "[child] bad --manifest: $MANIFEST"; exit 1; }
    [[ -n "$BUNDLE" ]] || { echo "[child] --bundle required"; exit 1; }
    LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
    EXP_DIR="$(sed -n "${LINE_NO}p" "$MANIFEST")"
    [[ -n "$EXP_DIR" && -d "$EXP_DIR" ]] || { echo "[child] bad exp dir at line $LINE_NO: '$EXP_DIR'"; exit 2; }

    source "$CONDA_SH"; conda activate "$CONDA_ENV_NAME"
    echo "=========================================="
    echo "ARRAY CHILD job=${SLURM_JOB_ID} array_id=${SLURM_ARRAY_TASK_ID} node=$(hostname)"
    echo "  exp_dir : $EXP_DIR"
    echo "  bundle  : $BUNDLE   refmask=$REFMASK"
    echo "=========================================="
    python "$PRECOMPUTE" --exp "$EXP_DIR" --bundle "$BUNDLE" \
        --no_merge --skip_existing \
        --keyframe_fracs "$KEYFRAME_FRACS" --thumb_fracs "$THUMB_FRACS" $REFMASK_FLAG
    RC=$?
    echo "[child] rc=$RC ($EXP_DIR)"
    exit $RC
fi

# =================================================================
#  Mode A: FRONTEND  (build manifest + submit array + dependent merge)
# =================================================================
[[ -n "$BUNDLE" ]] || { echo "Error: --bundle required"; exit 1; }
mkdir -p "$BUNDLE"

if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
    if [[ -z "$DATA_ROOT" && "${#VIDEOS_ROOTS[@]}" -eq 0 ]]; then
        echo "Error: provide --data_root or --videos_root ... or --manifest <txt>"; exit 1
    fi
    if [[ -z "$MANIFEST_OUT" ]]; then
        MANIFEST_OUT="/tmp/${USER}/precompute_$(date +%Y%m%d_%H%M%S).txt"
        mkdir -p "$(dirname "$MANIFEST_OUT")"
    fi
    echo "[scan] writing exp manifest -> $MANIFEST_OUT  (skipping done exps for resume)"
    DATA_ROOT_ARG="$DATA_ROOT" MANIFEST_OUT="$MANIFEST_OUT" BUNDLE_ARG="$BUNDLE" \
    python3 - "${VIDEOS_ROOTS[@]}" <<'PY'
import os, sys
from pathlib import Path
data_root = (os.environ.get("DATA_ROOT_ARG") or "").strip()
bundle = (os.environ.get("BUNDLE_ARG") or "").strip()
out = Path(os.environ["MANIFEST_OUT"])
roots = [Path(p).resolve() for p in sys.argv[1:]]
if data_root:
    dr = Path(data_root).expanduser().resolve()
    for p in sorted(dr.iterdir()):
        if p.is_dir() and p.name.startswith("videos_") and not p.name.endswith("_annotated"):
            roots.append(p.resolve())
exps, seen = [], set()
for r in roots:
    for marker in ("cam0_rgb.mp4", "data00000000.h5"):
        for f in r.rglob(marker):
            d = f.parent.resolve()
            if d not in seen:
                seen.add(d); exps.append(d)
exps.sort()
total = len(exps)
if bundle:  # resume: drop exps whose <bundle>/<task>/<exp>/_manifest.json exists
    b = Path(bundle)
    exps = [d for d in exps if not (b / d.parent.name / d.name / "_manifest.json").is_file()]
out.write_text("\n".join(str(e) for e in exps) + ("\n" if exps else ""))
print(f"[scan] {total} exp(s) found, {total - len(exps)} already done, {len(exps)} pending")
PY
    MANIFEST="$MANIFEST_OUT"
fi

N=$(grep -cve '^\s*$' "$MANIFEST")
[[ "$N" -ge 1 ]] || { echo "manifest empty: $MANIFEST"; exit 0; }

SUBMIT_ARGS=(--array=0-$((N-1))%${MAX_CONCURRENT} "$0" --manifest "$MANIFEST" --bundle "$BUNDLE"
             --keyframe_fracs "$KEYFRAME_FRACS" --thumb_fracs "$THUMB_FRACS")
[[ "$REFMASK" == "1" ]] && SUBMIT_ARGS+=(--refmask)

if [[ "$DRY_RUN" == "1" ]]; then
    echo "would submit: sbatch ${SUBMIT_ARGS[*]}"
    echo "then merge:   sbatch --dependency=afterany:<jid> $0 --merge --bundle $BUNDLE"
    exit 0
fi

echo "submitting array of $N exp(s) (concurrency=${MAX_CONCURRENT})"
ARRAY_JID=$(sbatch --parsable "${SUBMIT_ARGS[@]}")
echo "  array job: $ARRAY_JID"
MERGE_JID=$(sbatch --parsable --dependency=afterany:"$ARRAY_JID" \
            --job-name precompute_merge "$0" --merge --bundle "$BUNDLE")
echo "  merge job: $MERGE_JID (runs after array)"
echo "done. manifest.json will be ready at $BUNDLE/manifest.json once merge finishes."
