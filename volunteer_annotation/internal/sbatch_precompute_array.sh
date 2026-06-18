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
# Modeled on scripts/sbatch_run_hand_array.sh, but array elements = SHARDS
# (parallel workers), NOT one-per-exp: precompute is light (~30s/exp) so a
# per-exp job would waste more on GPU alloc + model load than the work itself.
# Each shard loads SAM2 ONCE and processes ~N/SHARDS exps.
#
# THREE modes (auto-detected):
#
# (A) FRONTEND  — run with `bash` (NOT sbatch):
#     bash sbatch_precompute_array.sh \
#         --data_root /viscam/projects/robotool/data \
#         --bundle    /viscam/projects/robotool/_va_bundle \
#         [--videos_root /abs/p1 /abs/p2 ...]  [--shards 4]  [--max_concurrent 16]
#         [--refmask] [--keyframe_fracs 0,0.1,0.2] [--thumb_fracs 0.4,0.6] [--force]
#     --force: recompute everything, ignoring existing _manifest.json (no resume skip).
#         [--manifest_out /abs/path.txt] [--dry_run]
#     Scans for exps (dirs with cam0_rgb.mp4 or data00000000.h5), writes a
#     one-exp-dir-per-line manifest, then submits
#       sbatch --array=0-(N-1)%MAX <self> --manifest <m> --bundle <b> ...
#     and a dependent merge job that runs after the array.
#
# (B) ARRAY CHILD  (auto via $SLURM_ARRAY_TASK_ID): runs precompute_embeddings.py
#     --exp_list <manifest> --shard <id>/<SHARDS> --no_merge --skip_existing,
#     processing every SHARDS-th exp with the model loaded once.
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
SLURM_OUTS="${SLURM_OUTS:-/viscam/u/chenrq/crq_ws/slurm_outs}"   # consolidated run log dir
INTERNAL="$HOCAP_ROOT/volunteer_annotation/internal"
PRECOMPUTE="$INTERNAL/precompute_embeddings.py"
MERGE="$INTERNAL/merge_manifests.py"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1   # stream python progress to the log in real time (no block buffering)

MANIFEST=""
MANIFEST_OUT=""
DATA_ROOT="/viscam/projects/robotool/data"          # default; override with --data_root
VIDEOS_ROOTS=()
BUNDLE="/viscam/projects/robotool/_va_bundle"        # default; override with --bundle
MAX_CONCURRENT=16
SHARDS=4                                             # # array elements (parallel workers); override with --shards
REFMASK=0
KEYFRAME_FRACS="0,0.1,0.2"
THUMB_FRACS="0.4,0.6"
DO_MERGE=0
DRY_RUN=0
FORCE=0
ADD=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --manifest)       MANIFEST="$2"; shift 2;;
        --manifest_out)   MANIFEST_OUT="$2"; shift 2;;
        --data_root)      DATA_ROOT="$2"; shift 2;;
        --videos_root)    shift; while [[ "$#" -gt 0 && "${1:0:2}" != "--" ]]; do VIDEOS_ROOTS+=("$1"); shift; done;;
        --bundle)         BUNDLE="$2"; shift 2;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2;;
        --shards)         SHARDS="$2"; shift 2;;
        --refmask)        REFMASK=1; shift;;
        --keyframe_fracs) KEYFRAME_FRACS="$2"; shift 2;;
        --thumb_fracs)    THUMB_FRACS="$2"; shift 2;;
        --add_frames)     KEYFRAME_FRACS="$2"; THUMB_FRACS=""; ADD=1; shift 2;;
        --merge)          DO_MERGE=1; shift;;
        --force)          FORCE=1; shift;;
        --dry_run)        DRY_RUN=1; shift;;
        -h|--help)        sed -n '2,55p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

REFMASK_FLAG=$([[ "$REFMASK" == "1" ]] && echo "" || echo "--no_refmask")
FORCE_FLAG=$([[ "$FORCE" == "1" ]] && echo "--force" || echo "")

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
    [[ -n "$SHARDS" ]] || { echo "[child] --shards required"; exit 1; }

    source "$CONDA_SH"; conda activate "$CONDA_ENV_NAME"
    echo "=========================================="
    echo "ARRAY CHILD job=${SLURM_JOB_ID} shard=${SLURM_ARRAY_TASK_ID}/${SHARDS} node=$(hostname)"
    echo "  manifest: $MANIFEST   bundle: $BUNDLE   refmask=$REFMASK"
    echo "=========================================="
    # ONE process per shard: loads the SAM2 model ONCE, then does every Nth exp
    # of the manifest (--skip_existing makes it resumable per exp).
    python "$PRECOMPUTE" --exp_list "$MANIFEST" --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}" \
        --bundle "$BUNDLE" --no_merge --skip_existing \
        --keyframe_fracs "$KEYFRAME_FRACS" --thumb_fracs "$THUMB_FRACS" $REFMASK_FLAG $FORCE_FLAG
    RC=$?
    echo "[child] shard ${SLURM_ARRAY_TASK_ID}/${SHARDS} rc=$RC"
    exit $RC
fi

# =================================================================
#  Mode A: FRONTEND  (build manifest + submit array + dependent merge)
# =================================================================
[[ -n "$BUNDLE" ]] || { echo "Error: --bundle required"; exit 1; }
if ! mkdir -p "$BUNDLE" 2>/dev/null; then
    echo "Error: cannot create bundle dir: $BUNDLE"
    echo "  (the built-in defaults are CLUSTER paths under /viscam — on a local box"
    echo "   override with --data_root <dir> --bundle <dir>, or use generate_bundle.sh)"
    exit 1
fi

if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
    if [[ -z "$DATA_ROOT" && "${#VIDEOS_ROOTS[@]}" -eq 0 ]]; then
        echo "Error: provide --data_root or --videos_root ... or --manifest <txt>"; exit 1
    fi
    if [[ -z "$MANIFEST_OUT" ]]; then
        # MUST be on shared storage (NOT /tmp, which is node-local) so array
        # children on compute nodes can read it. Prefer writing it INTO the
        # videos_root folder it describes (co-located with the data); fall back
        # to the bundle dir for multi-root / --data_root scans or if unwritable.
        TS="$(date +%Y%m%d_%H%M%S)"
        if [[ "${#VIDEOS_ROOTS[@]}" -eq 1 && -w "${VIDEOS_ROOTS[0]}" ]]; then
            MANIFEST_OUT="${VIDEOS_ROOTS[0]}/_va_exp_list_${TS}.txt"
        else
            MANIFEST_OUT="$BUNDLE/_exp_list_${TS}.txt"
        fi
    fi
    echo "[scan] writing exp manifest -> $MANIFEST_OUT  (skipping done exps for resume)"
    # --force/--add_frames: don't drop exps in the scan (precompute then skips
    # per-frame, so add-frames revisits already-precomputed exps to add new ones).
    SCAN_BUNDLE=$([[ "$FORCE" == "1" || "$ADD" == "1" ]] && echo "" || echo "$BUNDLE")
    DATA_ROOT_ARG="$DATA_ROOT" MANIFEST_OUT="$MANIFEST_OUT" BUNDLE_ARG="$SCAN_BUNDLE" \
    python3 - "${VIDEOS_ROOTS[@]}" <<'PY'
import os, sys
from pathlib import Path
data_root = (os.environ.get("DATA_ROOT_ARG") or "").strip()
bundle = (os.environ.get("BUNDLE_ARG") or "").strip()
out = Path(os.environ["MANIFEST_OUT"])
roots = [Path(p) for p in sys.argv[1:]]
# Explicit --videos_root scopes the run; only fall back to scanning the whole
# data_root when NO --videos_root was given (avoids the default data_root
# silently pulling in every videos_* dir).
if data_root and not roots:
    dr = Path(data_root).expanduser()
    if not dr.is_dir():
        print(f"[scan][ERR] --data_root not found: {dr}"); sys.exit(2)
    for p in sorted(dr.iterdir()):
        if p.is_dir() and p.name.startswith("videos_") and not p.name.endswith("_annotated"):
            roots.append(p)

# Bounded-depth scan (NOT rglob — that walks the whole NFS subtree incl. depth
# videos / calibration_debug and is very slow). Exps live at videos_root/<exp>
# or videos_root/<task>/<exp>; just stat the marker file in candidate dirs.
def is_exp(d):
    return (d / "cam0_rgb.mp4").exists() or (d / "data00000000.h5").exists()

def subdirs(d):
    try:
        return sorted(x for x in d.iterdir() if x.is_dir())
    except OSError:
        return []

exps, seen = [], set()
for r in roots:
    for c in subdirs(r):                 # videos_root/<exp>
        cands = [c] if is_exp(c) else subdirs(c)   # else try videos_root/<task>/<exp>
        for d in cands:
            rd = d.resolve()
            if rd not in seen and is_exp(d):
                seen.add(rd); exps.append(rd)
exps.sort()
total = len(exps)
def frag_path(b, d):   # mirrors precompute task_exp_from_path nesting
    rel = None
    for anc in d.parents:
        if anc.name.startswith("videos_"):
            rel = d.relative_to(anc.parent); break
    if rel is None:
        rel = Path(d.parent.name) / d.name
    return b / rel / "_manifest.json"
if bundle:  # resume: drop exps already done (per-exp _manifest.json present)
    b = Path(bundle)
    exps = [d for d in exps if not frag_path(b, d).is_file()]
out.write_text("\n".join(str(e) for e in exps) + ("\n" if exps else ""))
print(f"[scan] {total} exp(s) found, {total - len(exps)} already done, {len(exps)} pending")
PY
    MANIFEST="$MANIFEST_OUT"
fi

N=$(grep -cve '^\s*$' "$MANIFEST")
[[ "$N" -ge 1 ]] || { echo "manifest empty: $MANIFEST"; exit 0; }

# Array elements = SHARDS workers (NOT one per exp): each loads SAM2 once and
# processes ~N/SHARDS exps, so the model-load + job overhead is amortized.
NSHARDS=${SHARDS:-4}
(( NSHARDS > N )) && NSHARDS=$N        # no point having more shards than exps

# One consolidated log for the whole run (all array children + the merge job
# append stdout+stderr here) instead of per-exp .out/.err. Each child prints an
# "ARRAY CHILD ... exp_dir: ..." banner so you can still grep a specific exp.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SLURM_OUTS"
LOG="$SLURM_OUTS/precompute_${RUN_TS}.log"
LOG_ARGS=(--output="$LOG" --error="$LOG" --open-mode=append)

SUBMIT_ARGS=(--array=0-$((NSHARDS-1))%${MAX_CONCURRENT} "${LOG_ARGS[@]}"
             "$0" --manifest "$MANIFEST" --bundle "$BUNDLE" --shards "$NSHARDS"
             --keyframe_fracs "$KEYFRAME_FRACS" --thumb_fracs "$THUMB_FRACS")
[[ "$REFMASK" == "1" ]] && SUBMIT_ARGS+=(--refmask)
[[ "$FORCE" == "1" ]] && SUBMIT_ARGS+=(--force)

if [[ "$DRY_RUN" == "1" ]]; then
    echo "would submit: sbatch ${SUBMIT_ARGS[*]}"
    echo "  -> $N exp(s) over $NSHARDS shard(s) (~$(( (N + NSHARDS - 1) / NSHARDS )) exp/shard, model loaded once per shard)"
    echo "consolidated log: $LOG"
    exit 0
fi

echo "submitting $NSHARDS shard(s) for $N exp(s) (~$(( (N + NSHARDS - 1) / NSHARDS )) exp/shard, concurrency=${MAX_CONCURRENT})"
ARRAY_JID=$(sbatch --parsable "${SUBMIT_ARGS[@]}")
echo "  array job: $ARRAY_JID"
MERGE_JID=$(sbatch --parsable --dependency=afterany:"$ARRAY_JID" "${LOG_ARGS[@]}" \
            --job-name precompute_merge "$0" --merge --bundle "$BUNDLE")
echo "  merge job: $MERGE_JID (runs after array)"
echo "log:    $LOG   (single file, all exps + merge; tail -f to watch)"
echo "output: $BUNDLE/manifest.json   (ready once merge finishes)"
