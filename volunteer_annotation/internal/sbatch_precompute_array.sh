#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name precompute_arr
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
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
#     bash sbatch_precompute_array.sh                       # ALL videos_* folders
#     bash sbatch_precompute_array.sh --videos_root A B C   # just these folders
#     [--shards 4] [--bundle ...] [--refmask] [--keyframe_fracs ...] [--force] [--dry_run]
#   Submits ONE sharded array PER videos_* folder (so multiple folders run in
#   PARALLEL). With no --videos_root it auto-discovers every videos_* under
#   --data_root and launches one array each. Each folder uses <=--shards workers
#   (default 4); the cluster's MaxJobs limit caps how many folders run at once
#   (e.g. 16 slots = ~4 folders x 4 shards). RESUMES automatically (re-run same
#   cmd; done exps/folders skipped). One dependent merge job rebuilds
#   <bundle>/manifest.json after all arrays finish.
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
BUNDLE="/viscam/projects/robotool/_va_bundle_v2"        # default; override with --bundle
MAX_CONCURRENT=16                                    # cluster cap on concurrent array tasks
SHARDS=""                                            # # parallel workers; default = MAX_CONCURRENT (use all slots)
REFMASK=0
KEYFRAME_FRACS="0,0.1,0.2"
THUMB_FRACS="0.4,0.6"
DO_MERGE=0
DRY_RUN=0
FORCE=0
ADD=0
NO_MERGE_DEP=0   # skip submitting the dependent merge job (orchestrator merges once at the end)

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
        --no_merge_dep)   NO_MERGE_DEP=1; shift;;
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

RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SLURM_OUTS"
SCAN_BUNDLE=$([[ "$FORCE" == "1" || "$ADD" == "1" ]] && echo "" || echo "$BUNDLE")
ARRAY_JIDS=()

# scan ONE videos_root's exps -> manifest file (drops done exps for resume,
# unless --force/--add_frames). Bounded-depth (no rglob; NFS-fast).
scan_one() {   # $1=videos_root  $2=out_manifest  $3=label
    MANIFEST_OUT="$2" BUNDLE_ARG="$SCAN_BUNDLE" LABEL="$3" python3 - "$1" <<'PY'
import os, sys
from pathlib import Path
out = Path(os.environ["MANIFEST_OUT"]); bundle=(os.environ.get("BUNDLE_ARG") or "").strip(); lbl=os.environ.get("LABEL","")
root = Path(sys.argv[1])
def is_exp(d): return (d/"cam0_rgb.mp4").exists() or (d/"data00000000.h5").exists()
def subdirs(d):
    try: return sorted(x for x in d.iterdir() if x.is_dir())
    except OSError: return []
exps, seen = [], set()
for c in subdirs(root):                       # videos_root/<exp> or videos_root/<task>/<exp>
    for d in ([c] if is_exp(c) else subdirs(c)):
        rd = d.resolve()
        if rd not in seen and is_exp(d): seen.add(rd); exps.append(rd)
exps.sort(); total = len(exps)
def frag(b, d):   # mirrors precompute task_exp_from_path nesting
    rel = None
    for anc in d.parents:
        if anc.name.startswith("videos_"): rel = d.relative_to(anc.parent); break
    if rel is None: rel = Path(d.parent.name)/d.name
    return b/rel/"_manifest.json"
if bundle:
    b = Path(bundle); exps = [d for d in exps if not frag(b, d).is_file()]
out.write_text("\n".join(str(e) for e in exps) + ("\n" if exps else ""))
print(f"[scan {lbl}] {total} found, {total-len(exps)} done, {len(exps)} pending")
PY
}

# submit ONE folder's sharded array (each folder caps at its own shard count;
# the cluster's MaxJobs limit caps how many folders run AT ONCE).
submit_one() {   # $1=manifest  $2=label
    local M="$1" LBL="$2" N NS LOG JID
    N=$(grep -cve '^[[:space:]]*$' "$M" 2>/dev/null || echo 0)
    [[ "$N" -ge 1 ]] || { echo "[$LBL] 0 pending — skip"; return; }
    NS=${SHARDS:-4}; (( NS > N )) && NS=$N
    LOG="$SLURM_OUTS/precompute_${LBL}_${RUN_TS}.log"
    local A=(--array=0-$((NS-1))%${NS} --output="$LOG" --error="$LOG" --open-mode=append
             "$0" --manifest "$M" --bundle "$BUNDLE" --shards "$NS"
             --keyframe_fracs "$KEYFRAME_FRACS" --thumb_fracs "$THUMB_FRACS")
    [[ "$REFMASK" == "1" ]] && A+=(--refmask)
    [[ "$FORCE" == "1" ]] && A+=(--force)
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[$LBL] would submit: $N exps over $NS shards  (log $LOG)"; return
    fi
    JID=$(sbatch --parsable "${A[@]}")
    echo "[$LBL] $N exps over $NS shards -> array job: $JID  (log $LOG)"
    ARRAY_JIDS+=("$JID")
}

if [[ -n "$MANIFEST" && -f "$MANIFEST" ]]; then
    submit_one "$MANIFEST" "manual"                 # advanced: pre-built exp list
else
    # Resolve folder list: explicit --videos_root, else ALL videos_* under data_root.
    ROOTS=()
    if [[ "${#VIDEOS_ROOTS[@]}" -gt 0 ]]; then
        ROOTS=("${VIDEOS_ROOTS[@]}")
    elif [[ -n "$DATA_ROOT" && -d "$DATA_ROOT" ]]; then
        while IFS= read -r d; do ROOTS+=("${d%/}"); done \
            < <(ls -d "$DATA_ROOT"/videos_*/ 2>/dev/null | grep -v _annotated)
    else
        echo "Error: provide --data_root or --videos_root ... or --manifest <txt>"; exit 1
    fi
    [[ "${#ROOTS[@]}" -ge 1 ]] || { echo "no videos_* folders under $DATA_ROOT"; exit 0; }
    echo "[frontend] $(date) ${#ROOTS[@]} folder(s); ONE array per folder (<=${SHARDS:-4} shards each; cluster runs up to its MaxJobs concurrently)"
    for VR in "${ROOTS[@]}"; do
        [[ -d "$VR" ]] || { echo "[$(basename "$VR")] not a dir — skip"; continue; }
        name=$(basename "$VR")
        if [[ -w "$VR" ]]; then M="$VR/_va_exp_list_${RUN_TS}.txt"; else M="$BUNDLE/_exp_list_${name}_${RUN_TS}.txt"; fi
        scan_one "$VR" "$M" "$name"
        submit_one "$M" "$name"
    done
fi

# ONE merge after ALL arrays finish (rebuilds <bundle>/manifest.json from fragments).
if [[ "$DRY_RUN" != "1" && "$NO_MERGE_DEP" != "1" && "${#ARRAY_JIDS[@]}" -gt 0 ]]; then
    DEP="afterany:$(IFS=:; echo "${ARRAY_JIDS[*]}")"
    MLOG="$SLURM_OUTS/precompute_merge_${RUN_TS}.log"
    MJID=$(sbatch --parsable --dependency="$DEP" --output="$MLOG" --error="$MLOG" \
           --job-name precompute_merge "$0" --merge --bundle "$BUNDLE")
    echo "[frontend] merge job: $MJID  (after ${#ARRAY_JIDS[@]} array(s)) -> $BUNDLE/manifest.json"
fi
