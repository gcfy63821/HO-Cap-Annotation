#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name hand_lbl
#SBATCH --partition=viscam
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4,viscam14,viscam15
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/hand_lbl_%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/hand_lbl_%j.err
#
# Re-run hand reconstruction for exps that were manually tagged with a
# given review label in scripts/pose_review_labels.json (default
# label = tool_good_hand_bad). Sister script of sbatch_run_hand_remask.sh
# — same 1 job / N GPU / M concurrent worker layout — but the exp queue
# comes from a JSON of {remote_annotated_path: label} instead of walking
# a single --videos_root.
#
# Usage:
#   sbatch scripts/sbatch_run_hand_by_label.sh
#   sbatch scripts/sbatch_run_hand_by_label.sh --label hand_good_tool_bad
#   sbatch scripts/sbatch_run_hand_by_label.sh --label tool_good_hand_bad --label usable
#   sbatch scripts/sbatch_run_hand_by_label.sh \
#       --labels_json /path/to/pose_review_labels.json \
#       --label tool_good_hand_bad --parallel 4 --gpus 2 --chunk_size 500
#   sbatch scripts/sbatch_run_hand_by_label.sh --label tool_good_hand_bad --dry_run
#
# Options:
#   --labels_json PATH    JSON dict {annotated_exp_path: label} (default:
#                         scripts/pose_review_labels.json next to this file).
#   --label NAME          Review label to include. May be repeated. Default
#                         is a single label "tool_good_hand_bad".
#   --parallel N          Concurrent workers (default 4).
#   --gpus N              GPUs available (default 2). Worker idx → gpu
#                         mapping is round-robin (idx % gpus).
#   --chunk_size N        Inner chunk size (default 500).
#   --keep_existing       Don't pre-delete old hand outputs; pass
#                         --skip_existing to inner so it resumes from
#                         partial chunks. Default: wipe first.
#   --dry_run             Print queue and exit.
#   --remap_root SRC=DST  Rewrite the JSON path prefix SRC → DST before
#                         resolving local paths (handy if the JSON was
#                         written with one mountpoint and the cluster
#                         sees another). May be repeated.
#
# Resume semantics: identical to sbatch_run_hand_remask.sh — by default
# each exp is wiped right before its worker starts (so a scancel'd job
# leaves untouched exps' old outputs intact).

set -u

# ---------- cluster paths ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
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
LABELS_JSON="${SCRIPT_DIR}/pose_review_labels.json"
LABELS=()
PARALLEL=4
GPUS=2
CHUNK_SIZE=500
KEEP_EXISTING=0
DRY_RUN=0
REMAPS=()

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --labels_json)        LABELS_JSON="$2"; shift 2;;
        --label)              LABELS+=("$2"); shift 2;;
        --parallel)           PARALLEL="$2"; shift 2;;
        --gpus)               GPUS="$2"; shift 2;;
        --chunk_size)         CHUNK_SIZE="$2"; shift 2;;
        --keep_existing)      KEEP_EXISTING=1; shift;;
        --dry_run)            DRY_RUN=1; shift;;
        --remap_root)         REMAPS+=("$2"); shift 2;;
        -h|--help)            sed -n '2,55p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

[[ ${#LABELS[@]} -eq 0 ]] && LABELS=("tool_good_hand_bad")
[[ -f "$LABELS_JSON" ]] || { echo "Error: labels json not found: $LABELS_JSON"; exit 1; }

PER_WORKER_THREADS=$(( TOTAL_CPUS / PARALLEL ))
[[ "$PER_WORKER_THREADS" -lt 1 ]] && PER_WORKER_THREADS=1

echo "=========================================================================="
echo "hand_label    job=${SLURM_JOB_ID:-<non-slurm>}    node=$(hostname)"
echo "  labels_json      : $LABELS_JSON"
echo "  labels           : ${LABELS[*]}"
echo "  parallel workers : $PARALLEL    gpus: $GPUS"
echo "  chunk_size       : $CHUNK_SIZE"
echo "  keep_existing    : $KEEP_EXISTING (0 = wipe old hand outputs first)"
echo "  remap_root       : ${REMAPS[*]:-(none)}"
echo "=========================================================================="

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
command -v ffmpeg >/dev/null 2>&1 || { echo "Error: ffmpeg not on PATH"; exit 1; }

# ---------- read JSON, filter by label, expand to (videos_root, task, exp) ----------
# Path shape expected: <root>/videos_<DATE>_annotated/<task>/<exp>
# Output line format: videos_root|task|exp|annotated_exp_path|label
PARSED_TXT="$H5_SCRATCH_ROOT/_pending.txt"
LABELS_CSV="$(IFS=,; echo "${LABELS[*]}")"
REMAPS_CSV="$(IFS=$'\x1f'; echo "${REMAPS[*]:-}")"

python3 - "$LABELS_JSON" "$LABELS_CSV" "$REMAPS_CSV" >"$PARSED_TXT" <<'PYEOF'
import json, os, sys, re

labels_json, labels_csv, remaps_csv = sys.argv[1], sys.argv[2], sys.argv[3]
wanted = set(s for s in labels_csv.split(",") if s)
remaps = []
if remaps_csv:
    for spec in remaps_csv.split("\x1f"):
        if "=" not in spec: continue
        s, d = spec.split("=", 1)
        remaps.append((s.rstrip("/"), d.rstrip("/")))

with open(labels_json) as f:
    data = json.load(f)
if not isinstance(data, dict):
    print("[FATAL] labels json is not a dict", file=sys.stderr); sys.exit(2)

# .../videos_<DATE>_annotated/<task>/<exp>
PAT = re.compile(r'^(?P<root>.*)/(?P<rb>videos_[^/]+_annotated)/(?P<task>[^/]+)/(?P<exp>[^/]+)/?$')

n_total = n_label_match = n_bad_shape = 0
seen = set()
for path, lbl in data.items():
    n_total += 1
    if lbl not in wanted: continue
    n_label_match += 1
    p = path
    for src, dst in remaps:
        if p == src or p.startswith(src + "/"):
            p = dst + p[len(src):]
            break
    p = p.rstrip("/")
    m = PAT.match(p)
    if not m:
        print(f"[WARN-shape] {p}", file=sys.stderr); n_bad_shape += 1; continue
    root, rb, task, exp = m["root"], m["rb"], m["task"], m["exp"]
    orig_rb = rb[:-len("_annotated")]                # videos_<DATE>
    videos_root = f"{root}/{orig_rb}"
    key = (videos_root, task, exp)
    if key in seen: continue
    seen.add(key)
    print(f"{videos_root}|{task}|{exp}|{p}|{lbl}")

print(f"[count] total_in_json={n_total} label_match={n_label_match} "
      f"unique={len(seen)} bad_shape={n_bad_shape}", file=sys.stderr)
PYEOF
PARSE_RC=$?
[[ $PARSE_RC -ne 0 ]] && { echo "Error: parsing labels json failed (rc=$PARSE_RC)"; exit 1; }

# ---------- load PENDING + cache cal_yaml per videos_root ----------
declare -A CAL_FOR_ROOT=()
PENDING=()
N_NO_VR=0; N_NO_CAL=0; N_NO_RGB=0; N_NO_ANN=0

cal_yaml_for_root() {
    local vr="$1"
    if [[ -n "${CAL_FOR_ROOT[$vr]+x}" ]]; then
        echo "${CAL_FOR_ROOT[$vr]}"; return 0
    fi
    if [[ ! -d "$vr" ]]; then
        CAL_FOR_ROOT["$vr"]=""; echo ""; return 0
    fi
    local cal_folder cal_yaml
    cal_folder="$(find "$vr" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' 2>/dev/null | head -n 1)"
    if [[ -z "$cal_folder" ]]; then
        CAL_FOR_ROOT["$vr"]=""; echo ""; return 0
    fi
    cal_yaml="$(find "$cal_folder" -maxdepth 1 -type f -name 'realsense_calibration_*_global_aligned.yaml' 2>/dev/null | head -n 1)"
    [[ -z "$cal_yaml" ]] && cal_yaml=""
    CAL_FOR_ROOT["$vr"]="$cal_yaml"
    echo "$cal_yaml"
}

while IFS='|' read -r videos_root task exp ann_path label; do
    [[ -z "$videos_root" ]] && continue
    if [[ ! -d "$videos_root" ]]; then
        echo "[skip-no-videos_root] $videos_root  ($task/$exp)"
        N_NO_VR=$((N_NO_VR + 1)); continue
    fi
    exp_dir="$videos_root/$task/$exp"
    if ! compgen -G "${exp_dir}/cam*_rgb.mp4" >/dev/null 2>&1; then
        echo "[skip-no-rgb] $exp_dir"
        N_NO_RGB=$((N_NO_RGB + 1)); continue
    fi
    cal_yaml="$(cal_yaml_for_root "$videos_root")"
    if [[ -z "$cal_yaml" ]]; then
        echo "[skip-no-cal] $videos_root  (no realsense_calibrate_*/realsense_calibration_*_global_aligned.yaml)"
        N_NO_CAL=$((N_NO_CAL + 1)); continue
    fi
    # Annotated root must exist (we drop hand outputs into it).
    ann_root="$(dirname "$videos_root")/$(basename "$videos_root")_annotated"
    if [[ ! -d "$ann_root" ]]; then
        echo "[skip-no-annotated] $ann_root"
        N_NO_ANN=$((N_NO_ANN + 1)); continue
    fi
    PENDING+=("${videos_root}|${task}|${exp}|${cal_yaml}")
done <"$PARSED_TXT"

N_PENDING=${#PENDING[@]}
echo ""
echo "[scan] queued: $N_PENDING"
echo "[scan] dropped — no videos_root: $N_NO_VR  no rgb: $N_NO_RGB  "
echo "       no cal: $N_NO_CAL  no annotated_root: $N_NO_ANN"
echo "[scan] unique videos_roots: ${#CAL_FOR_ROOT[@]}"
echo ""
if [[ "$N_PENDING" -lt 1 ]]; then
    echo "[scan] nothing to do."
    exit 0
fi

# ---------- preview ----------
for ((k=0; k<N_PENDING && k<30; k++)); do
    IFS='|' read -r vr t e c <<<"${PENDING[$k]}"
    echo "  [$((k+1))] $(basename "$vr") / $t / $e"
done
[[ "$N_PENDING" -gt 30 ]] && echo "  ... ($((N_PENDING - 30)) more)"
echo ""
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry_run] would launch $PARALLEL workers across $GPUS GPU(s)."
    exit 0
fi

# ---------- worker pool (mirrors sbatch_run_hand_remask.sh) ----------
INNER_FLAGS=()
[[ "$KEEP_EXISTING" == "1" ]] && INNER_FLAGS+=(--skip_existing)

run_one() {
    local idx="$1"
    local entry="$2"
    IFS='|' read -r videos_root task exp cal_yaml <<<"$entry"
    local gpu=$((idx % GPUS))
    local task_folder="$videos_root/$task"
    local ann_root
    ann_root="$(dirname "$videos_root")/$(basename "$videos_root")_annotated"
    local ann="$ann_root/$task/$exp"
    local scratch="$H5_SCRATCH_ROOT/w${idx}"
    mkdir -p "$scratch"
    local logf="$H5_SCRATCH_ROOT/log_${idx}_${task}_${exp}.log"
    echo "[$(date +%H:%M:%S)] [worker $idx → gpu $gpu] START  $task / $exp  (log: $logf)"

    if [[ "$KEEP_EXISTING" != "1" ]]; then
        rm -f "${ann}/result.pkl" \
              "${ann}/result_hand_optimized.pkl" \
              "${ann}/processed/joint_pose_solver/poses_m.npy" 2>/dev/null
        rm -f "${ann}"/result_*_*.pkl \
              "${ann}"/result_hand_optimized_*_*.pkl \
              "${ann}"/poses_m_*_*.npy 2>/dev/null
        echo "[$(date +%H:%M:%S)] [worker $idx] cleaned old hand outputs in $ann" >>"$logf"
    fi

    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS="$PER_WORKER_THREADS"
    export MKL_NUM_THREADS="$PER_WORKER_THREADS"
    export OPENBLAS_NUM_THREADS="$PER_WORKER_THREADS"
    export NUMEXPR_NUM_THREADS="$PER_WORKER_THREADS"
    export H5_COMPRESSION=lzf

    {
        echo "[worker $idx] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "[worker $idx] cal_yaml=$cal_yaml"
        nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv 2>&1 | head -5
    } >>"$logf" 2>&1

    bash "$HAND_BATCH_SCRIPT" \
        --task_folder "$task_folder" \
        --calibration_yaml "$cal_yaml" \
        --chunk_size "$CHUNK_SIZE" \
        --h5_scratch_dir "$scratch" \
        --only_exp "$exp" \
        "${INNER_FLAGS[@]}" >>"$logf" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] [worker $idx → gpu $gpu] OK     $task / $exp"
        local marker="${ann}/.hand_label_done.json"
        cat > "$marker" <<EOF
{
  "kind": "hand_by_label",
  "completed_at": "$(date -Iseconds)",
  "slurm_job_id": "${SLURM_JOB_ID:-non-slurm}",
  "labels_json": "${LABELS_JSON}",
  "labels": "${LABELS_CSV}",
  "calibration_yaml": "${cal_yaml}",
  "chunk_size": ${CHUNK_SIZE},
  "keep_existing": ${KEEP_EXISTING},
  "videos_root": "${videos_root}",
  "task": "${task}",
  "exp": "${exp}",
  "host": "$(hostname -s)"
}
EOF
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
echo "  labels filtered : ${LABELS_CSV}"
echo "  unique vroots   : ${#CAL_FOR_ROOT[@]}"
[[ "$n_fail" -gt 0 ]] && echo "see per-worker logs under $H5_SCRATCH_ROOT/log_*.log (wiped on exit)"
echo "=========================================================================="

[[ "$n_fail" -gt 0 ]] && exit 1 || exit 0
