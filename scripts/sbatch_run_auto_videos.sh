#!/bin/bash
#SBATCH --account viscam
#SBATCH --job-name hauto_v
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4,viscam1,viscam2,viscam3,viscam4
#SBATCH --output=/viscam/u/chenrq/crq_ws/slurm_outs/%j.out
#SBATCH --error=/viscam/u/chenrq/crq_ws/slurm_outs/%j.err
#
# Cluster sbatch wrapper that walks a videos_XXXX/ folder and runs
# scripts/run_auto_annotator.sh on every experiment in every task,
# honoring a manual frame-0 visibility annotation file if present.
#
# Layout assumed (from your data collection / DataCollection convention):
#   <videos_root>/                                  e.g. .../videos_0101
#     realsense_calibrate_*/                        ← calibration (skipped)
#       realsense_calibration_*.yaml
#       realsense_calibration_*_global_aligned.yaml ← preferred
#     <task_name>/                                  e.g. mallet_crush_nuts
#       frame0_visibility.yaml      (OPTIONAL — set by frame0_visibility_inspector.py)
#       <exp_name>/                                 e.g. 20260104_..._1
#         cam{0..7}_rgb.mp4
#         cam{0..7}_depth.mkv
#       <exp_name>/
#       ...
#     <another_task>/
#
# Frame-0 visibility filtering:
#   For each task, if `<task>/frame0_visibility.yaml` exists, ONLY exps with
#   label "visible" are processed by default. Exps labeled "not_visible" /
#   "unsure" / unannotated are skipped.
#   Override via --frame0_mode:
#     visible_only   : only label==visible           (default when yaml present)
#     visible_unsure : visible OR unsure             (rescue mode)
#     all            : ignore the yaml entirely      (default when yaml absent)
#
#   The yaml is produced by:
#     python scripts/frame0_visibility_inspector.py --videos_root <videos_root>
#
# Tool name resolution per exp (priority high → low):
#   1. --force_tool NAME                         (overrides everything)
#   2. tool_keyword_mapping.yaml keyword hit     (manual 词表; see below)
#   3. heuristic auto-match (scripts/match_tool_name.py token overlap)
#
# Mapping yaml (optional, recommended for clusters with curated vocab):
#   Default location: <videos_root>/tool_keyword_mapping.yaml
#   Override with --mapping_yaml /abs/path.yaml
#   Format:
#     mappings:
#       mallet:    rubber_mallet         # if "mallet" appears in exp_name → use rubber_mallet
#       largeplate: largeplate
#       woodenfork: big_wooden_fork
#   Pass --mapping_only to require a keyword hit (exps that don't match any
#   keyword are SKIPPED, never auto-matched). Useful for "only annotate exps
#   whose tool I've manually curated in the vocab".
#
# What runs per exp:
#   scripts/run_auto_annotator.sh
#     --sequence_folder <task>/<exp>
#     --calibration_yaml <discovered>
#     --tool_name <resolved>
#     --models_folder $MODELS_FOLDER
#     --object_chunk_size $OBJECT_CHUNK_SIZE
#     [--hand 1] [--fake_optimize] [--resume] ...
#
# Resume:
#   --resume is forwarded to run_auto_annotator.sh, which itself does
#   per-stage skip when outputs exist (h5, DINO npz, fd_poses_merged_fixed.npy,
#   result_hand_optimized.pkl, ...). For coarser exp-level skipping pass
#   --skip_existing — that bypasses an exp entirely if its final marker is on
#   disk (saves the seconds-per-stage gate scans across many exps).
#
# Usage:
#   sbatch scripts/sbatch_run_auto_videos.sh \
#     --videos_root /viscam/projects/robotool/data/videos_0101 \
#     [--object_chunk_size 600]    (default 600)
#     [--fake_optimize]            (recommended on long videos)
#     [--hand 1]                   (0 disables hand)
#     [--resume]                   (per-stage resume in run_auto_annotator)
#     [--skip_existing]            (exp-level skip: pre-check done marker)
#     [--frame0_mode visible_only|visible_unsure|all]  (auto by default)
#     [--force_tool NAME]          (override per-exp tool matching)
#     [--mapping_yaml PATH]        (词表 yaml; auto-detected at <videos_root>/tool_keyword_mapping.yaml)
#     [--mapping_only]             (only run exps whose name matches a词表 keyword)
#     [--start_frame 0] [--end_frame N]
#     [--dry_run]                  (list what would run, don't actually run)

set -u

# ---------- cluster paths (override via env if your layout differs) ----------
export HOCAP_ROOT="${HOCAP_ROOT:-/viscam/u/chenrq/crq_ws/hocap/HO-Cap-Annotation}"
export HAND_ROOT="${HAND_ROOT:-/viscam/u/chenrq/crq_ws/robotool/HandReconstruction}"
export MODELS_FOLDER="${MODELS_FOLDER:-/viscam/u/chenrq/models}"
export CONDA_SH="${CONDA_SH:-/viscam/u/chenrq/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hocap-annotation}"
RUN_SCRIPT="$HOCAP_ROOT/scripts/run_auto_annotator.sh"
MATCH_SCRIPT="$HOCAP_ROOT/scripts/match_tool_name.py"

# SAM2 (Stage 2 DINO+SAM2). On viscam sam2 lives under robotool/, set explicitly.
export SAM2_ROOT="${SAM2_ROOT:-/viscam/u/chenrq/crq_ws/robotool/sam2}"
export SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt}"
export SAM2_VIDEO_CFG="${SAM2_VIDEO_CFG:-$HOCAP_ROOT/config/sam2_config/sam2.1_hiera_l.yaml}"
export SAM2_IMAGE_CFG="${SAM2_IMAGE_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"

# ---------- perf env ----------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export H5_COMPRESSION="lzf"
export PYOPENGL_PLATFORM="egl"

# ---------- /dev/shm scratch ----------
_JOBTAG="${SLURM_JOB_ID:-nonslurm_$$}"
H5_SCRATCH="/dev/shm/${USER}/${_JOBTAG}"
mkdir -p "$H5_SCRATCH"
cleanup_shm() {
    [[ -d "$H5_SCRATCH" ]] && rm -rf "$H5_SCRATCH" 2>/dev/null || true
}
trap cleanup_shm EXIT INT TERM HUP

# ---------- arg parsing ----------
VIDEOS_ROOT=""
CALIBRATION_YAML=""
HAND=""
FAKE_OPTIMIZE=0
RESUME=0
SKIP_EXISTING=0
FRAME0_MODE="auto"        # auto | visible_only | visible_unsure | all
FORCE_TOOL=""
MAPPING_YAML=""           # path to tool_keyword_mapping.yaml; auto = <videos_root>/tool_keyword_mapping.yaml if present
MAPPING_ONLY=0            # if 1, --require_mapping (skip exps without keyword hit)
START_FRAME=0
END_FRAME=""
ROT_THRESH=15
TRANS_THRESH=0.03
TRACK_REFINE_ITER=10
OBJECT_CHUNK_SIZE=600
HAND_CHUNK_SIZE=""
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --videos_root)        VIDEOS_ROOT="$2"; shift 2;;
        --calibration_yaml)   CALIBRATION_YAML="$2"; shift 2;;
        --hand)               HAND="$2"; shift 2;;
        --fake_optimize)      FAKE_OPTIMIZE=1; shift;;
        --resume)             RESUME=1; shift;;
        --skip_existing)      SKIP_EXISTING=1; shift;;
        --frame0_mode)        FRAME0_MODE="$2"; shift 2;;
        --force_tool)         FORCE_TOOL="$2"; shift 2;;
        --mapping_yaml)       MAPPING_YAML="$2"; shift 2;;
        --mapping_only)       MAPPING_ONLY=1; shift;;
        --models_folder)      export MODELS_FOLDER="$2"; shift 2;;
        --start_frame)        START_FRAME="$2"; shift 2;;
        --end_frame)          END_FRAME="$2"; shift 2;;
        --rot_thresh)         ROT_THRESH="$2"; shift 2;;
        --trans_thresh)       TRANS_THRESH="$2"; shift 2;;
        --track_refine_iter)  TRACK_REFINE_ITER="$2"; shift 2;;
        --object_chunk_size)  OBJECT_CHUNK_SIZE="$2"; shift 2;;
        --hand_chunk_size)    HAND_CHUNK_SIZE="$2"; shift 2;;
        --dry_run)            DRY_RUN=1; shift;;
        -h|--help)            sed -n '2,72p' "$0"; exit 0;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [[ -z "$VIDEOS_ROOT" ]]; then
    echo "Error: --videos_root is required"; exit 1
fi
VIDEOS_ROOT="$(cd "$VIDEOS_ROOT" && pwd)"
[[ -d "$VIDEOS_ROOT" ]] || { echo "Error: not a directory: $VIDEOS_ROOT"; exit 1; }

case "$FRAME0_MODE" in
    auto|visible_only|visible_unsure|all) ;;
    *) echo "Error: --frame0_mode must be auto / visible_only / visible_unsure / all"; exit 1;;
esac

echo "=========================================="
echo "job        : ${SLURM_JOB_ID:-<non-slurm>}   node: $(hostname)"
echo "cpus/omp   : ${SLURM_CPUS_PER_TASK:-4}"
echo "videos_root: $VIDEOS_ROOT"
echo "models     : $MODELS_FOLDER"
echo "scratch    : $H5_SCRATCH"
echo "frame0_mode: $FRAME0_MODE"
echo "chunk_size : $OBJECT_CHUNK_SIZE  fake_optim: $FAKE_OPTIMIZE  hand: ${HAND:-<unset>}"
echo "=========================================="

# ---------- conda + ffmpeg ----------
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg not on PATH"; exit 1
fi

# ---------- discover calibration ----------
if [[ -z "$CALIBRATION_YAML" ]]; then
    CAL_FOLDER="$(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'realsense_calibrate_*' | head -n 1)"
    if [[ -z "$CAL_FOLDER" ]]; then
        # Fallback: a *.yaml directly under videos_root (user's local layout)
        CALIBRATION_YAML="$(find "$VIDEOS_ROOT" -maxdepth 1 -type f -name 'realsense_calibration_*_global_aligned.yaml' | head -n 1)"
        [[ -z "$CALIBRATION_YAML" ]] && CALIBRATION_YAML="$(find "$VIDEOS_ROOT" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' ! -name '*_aligned.yaml' | head -n 1)"
    else
        ALIGNED="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name '*_global_aligned.yaml' | head -n 1)"
        if [[ -n "$ALIGNED" ]]; then
            CALIBRATION_YAML="$ALIGNED"
        else
            CALIBRATION_YAML="$(find "$CAL_FOLDER" -maxdepth 1 -type f -name 'realsense_calibration_*.yaml' ! -name '*_aligned.yaml' | head -n 1)"
        fi
    fi
fi
if [[ -z "$CALIBRATION_YAML" || ! -f "$CALIBRATION_YAML" ]]; then
    echo "Error: could not find a calibration yaml under $VIDEOS_ROOT"
    echo "       Pass one explicitly via --calibration_yaml /abs/path.yaml"
    exit 1
fi
echo "calibration: $CALIBRATION_YAML"

# ---------- discover mapping yaml (optional 词表 for tool name) ----------
# Convention (matches batch_auto_annotator.sh): <videos_root>/tool_keyword_mapping.yaml
# Format:
#   mappings:
#     mallet:    rubber_mallet
#     dough:     mallet_crush_dough
#     plate:     largeplate
# Each kw is matched (case-insensitive substring) against an exp's name; first
# hit wins. The keyword hit takes precedence over the heuristic auto-match.
# With --mapping_only, an exp whose name matches no keyword is SKIPPED (not
# auto-matched) — useful for "only annotate exps whose tool I've curated".
if [[ -z "$MAPPING_YAML" ]]; then
    if [[ -f "$VIDEOS_ROOT/tool_keyword_mapping.yaml" ]]; then
        MAPPING_YAML="$VIDEOS_ROOT/tool_keyword_mapping.yaml"
    fi
fi
if [[ -n "$MAPPING_YAML" && -f "$MAPPING_YAML" ]]; then
    echo "mapping    : $MAPPING_YAML  (mapping_only=$MAPPING_ONLY)"
elif [[ -n "$MAPPING_YAML" ]]; then
    echo "Error: --mapping_yaml file not found: $MAPPING_YAML"; exit 1
else
    echo "mapping    : <none — auto-match heuristic only>"
fi
echo "=========================================="

# ---------- discover tasks ----------
TASKS=()
while IFS= read -r -d '' d; do
    [[ "$(basename "$d")" == realsense_calibrate_* ]] && continue
    TASKS+=("$d")
done < <(find "$VIDEOS_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
[[ "${#TASKS[@]}" -eq 0 ]] && { echo "Error: no task folders under $VIDEOS_ROOT"; exit 1; }
echo "tasks (${#TASKS[@]}):"; for t in "${TASKS[@]}"; do echo "  - $(basename "$t")"; done

# ---------- helper: read frame0_visibility.yaml -> filtered exp list ----------
# Usage: filter_exps_for_task <task_dir>
# Echoes one exp name per line. Honors $FRAME0_MODE.
filter_exps_for_task() {
    local task_dir="$1"
    local mode="$FRAME0_MODE"
    local f0_yaml="${task_dir}/frame0_visibility.yaml"
    if [[ "$mode" == "auto" ]]; then
        if [[ -f "$f0_yaml" ]]; then mode="visible_only"; else mode="all"; fi
    fi

    F0_YAML="$f0_yaml" MODE="$mode" TASK_DIR="$task_dir" python3 - <<'PY'
import os, sys, yaml
from pathlib import Path
task_dir = Path(os.environ["TASK_DIR"])
mode     = os.environ["MODE"]
yaml_path= Path(os.environ["F0_YAML"])

# All candidate exp dirs (must have at least one cam*_rgb.mp4)
candidates = []
for sub in sorted(task_dir.iterdir()):
    if not sub.is_dir(): continue
    name = sub.name
    if name.endswith(".tar.gz") or name == "board_reference": continue
    if not list(sub.glob("cam*_rgb.mp4")): continue
    candidates.append(name)

if mode == "all" or not yaml_path.exists():
    for n in candidates:
        print(n)
    sys.exit(0)

try:
    data = yaml.safe_load(yaml_path.read_text()) or {}
except Exception as e:
    sys.stderr.write(f"[WARN] failed to parse {yaml_path}: {e}; running all exps\n")
    for n in candidates:
        print(n)
    sys.exit(0)

ann = (data.get("annotations") or {})
keep = {"visible"} if mode == "visible_only" else {"visible", "unsure"}
for n in candidates:
    if ann.get(n) in keep:
        print(n)
PY
}

# ---------- per-exp runner ----------
TOTAL=0; OK=0; SKIP=0; FAIL=0
FAILED=()
SKIPPED_NOT_VISIBLE=0

for TASK_DIR in "${TASKS[@]}"; do
    TASK_NAME="$(basename "$TASK_DIR")"
    F0_YAML="${TASK_DIR}/frame0_visibility.yaml"
    EXP_NAMES=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && EXP_NAMES+=("$line")
    done < <(filter_exps_for_task "$TASK_DIR")

    # Count not-visible / unannotated for the summary line
    if [[ -f "$F0_YAML" && ( "$FRAME0_MODE" == "visible_only" || "$FRAME0_MODE" == "auto" ) ]]; then
        ALL_CANDS=$(find "$TASK_DIR" -maxdepth 1 -mindepth 1 -type d \
                    \( ! -name "*.tar.gz" \) ! -name "board_reference" -printf "%f\n" | sort)
        ALL_COUNT=$(echo "$ALL_CANDS" | grep -c . || true)
        SKIPPED_NOT_VISIBLE=$((SKIPPED_NOT_VISIBLE + ALL_COUNT - ${#EXP_NAMES[@]}))
    fi

    echo ""
    echo "##########################################"
    echo "# task: $TASK_NAME"
    if [[ -f "$F0_YAML" ]]; then
        echo "#  frame0_visibility.yaml: present  (mode=$FRAME0_MODE)"
    else
        echo "#  frame0_visibility.yaml: NOT present  (running all exps)"
    fi
    echo "#  exps to run: ${#EXP_NAMES[@]}"
    echo "##########################################"

    [[ "${#EXP_NAMES[@]}" -eq 0 ]] && { echo "  (no exps to run for this task)"; continue; }

    ANNOTATED_ROOT="$(dirname "$VIDEOS_ROOT")/$(basename "$VIDEOS_ROOT")_annotated/${TASK_NAME}"

    for EXP_NAME in "${EXP_NAMES[@]}"; do
        EXP_DIR="${TASK_DIR}/${EXP_NAME}"
        TOTAL=$((TOTAL+1))

        # ---- skip_existing (exp-level coarse skip) ----
        ANNO="${ANNOTATED_ROOT}/${EXP_NAME}"
        if [[ -n "$HAND" && "$HAND" != "0" ]]; then
            DONE_MARKER="${ANNO}/result_hand_optimized.pkl"
        else
            DONE_MARKER="${ANNO}/processed/fd_pose_solver/fd_poses_merged_fixed.npy"
        fi
        if [[ "$SKIP_EXISTING" == "1" && -f "$DONE_MARKER" ]]; then
            echo ""
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [skip: $DONE_MARKER exists]"
            SKIP=$((SKIP+1)); continue
        fi

        # ---- resolve tool name ----
        # Priority: --force_tool > mapping_yaml keyword hit > auto-match heuristic
        if [[ -n "$FORCE_TOOL" ]]; then
            TOOL="$FORCE_TOOL"
        elif [[ -f "$MATCH_SCRIPT" ]]; then
            MATCH_ARGS=(
                --models_folder "$MODELS_FOLDER"
                --task_name "$TASK_NAME"
                --exp_name "$EXP_NAME"
            )
            [[ -n "$MAPPING_YAML" && -f "$MAPPING_YAML" ]] && MATCH_ARGS+=(--mapping_yaml "$MAPPING_YAML")
            [[ "$MAPPING_ONLY" == "1" ]] && MATCH_ARGS+=(--require_mapping)
            TOOL=$(python "$MATCH_SCRIPT" "${MATCH_ARGS[@]}" 2>/tmp/match_err.$$ || true)
            MATCH_RC=$?
            MATCH_ERR="$(cat /tmp/match_err.$$ 2>/dev/null)"; rm -f /tmp/match_err.$$
            if [[ $MATCH_RC -eq 3 ]]; then
                # exit code 3 == --require_mapping but no keyword hit
                echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [skip: no mapping keyword (--mapping_only)]"
                SKIP=$((SKIP+1)); continue
            fi
            if [[ -z "$TOOL" ]]; then
                echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [FAIL: tool matcher returned empty -- $MATCH_ERR]"
                FAIL=$((FAIL+1)); FAILED+=("$TASK_NAME/$EXP_NAME:no_match"); continue
            fi
        else
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [FAIL: no --force_tool and no $MATCH_SCRIPT]"
            FAIL=$((FAIL+1)); FAILED+=("$TASK_NAME/$EXP_NAME:no_matcher"); continue
        fi

        # ---- mesh sanity ----
        MESH=""
        for cand in textured_mesh.obj cleaned_mesh_10000.obj mesh.obj; do
            if [[ -f "$MODELS_FOLDER/$TOOL/$cand" ]]; then
                MESH="$MODELS_FOLDER/$TOOL/$cand"; break
            fi
        done
        if [[ -z "$MESH" ]]; then
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [FAIL: no mesh in $MODELS_FOLDER/$TOOL/]"
            FAIL=$((FAIL+1)); FAILED+=("$TASK_NAME/$EXP_NAME:no_mesh"); continue
        fi

        echo ""
        echo "------------------------------------------"
        echo "[$TOTAL] $TASK_NAME / $EXP_NAME"
        echo "  tool : $TOOL"
        echo "  mesh : $MESH"
        echo "------------------------------------------"

        if [[ "$DRY_RUN" == "1" ]]; then
            continue
        fi

        # ---- assemble run_auto_annotator.sh args ----
        RUN_ARGS=(
            --sequence_folder "$EXP_DIR"
            --calibration_yaml "$CALIBRATION_YAML"
            --tool_name "$TOOL"
            --tool_mesh "$MESH"
            --models_folder "$MODELS_FOLDER"
            --h5_scratch_dir "$H5_SCRATCH"
            --start_frame "$START_FRAME"
            --rot_thresh "$ROT_THRESH"
            --trans_thresh "$TRANS_THRESH"
            --track_refine_iter "$TRACK_REFINE_ITER"
            --object_chunk_size "$OBJECT_CHUNK_SIZE"
        )
        [[ -n "$END_FRAME" ]]        && RUN_ARGS+=(--end_frame "$END_FRAME")
        [[ -n "$HAND" ]]             && RUN_ARGS+=(--hand "$HAND")
        [[ -n "$HAND_CHUNK_SIZE" ]]  && RUN_ARGS+=(--hand_chunk_size "$HAND_CHUNK_SIZE")
        [[ "$RESUME" == "1" ]]       && RUN_ARGS+=(--resume)
        [[ "$FAKE_OPTIMIZE" == "1" ]] && RUN_ARGS+=(--fake_optimize)

        if bash "$RUN_SCRIPT" "${RUN_ARGS[@]}"; then
            OK=$((OK+1))
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [done]"
        else
            RC=$?
            FAIL=$((FAIL+1)); FAILED+=("$TASK_NAME/$EXP_NAME:rc=$RC")
            echo "[$TOTAL] $TASK_NAME / $EXP_NAME  [FAIL: run_auto_annotator rc=$RC]"
        fi
    done
done

echo ""
echo "=========================================="
echo "summary: total=$TOTAL  ok=$OK  skipped(done)=$SKIP  failed=$FAIL"
echo "         skipped by frame0 filter: $SKIPPED_NOT_VISIBLE"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
echo "=========================================="

# Exit non-zero if anything actually failed (for slurm requeue / chained jobs)
[[ "$FAIL" -gt 0 ]] && exit 1 || exit 0
